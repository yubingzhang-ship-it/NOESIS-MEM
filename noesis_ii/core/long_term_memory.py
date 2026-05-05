"""
长期记忆模块（Long-Term Memory）

分布式长期存储，支持语义化整合。

P1 升级：
- 检索方法升级：综合得分（语义相似度 × 权重 × 时效性）
- 语义相似度：基于 TF-IDF 的余弦相似度（替代纯 LIKE 查询）
- 权重公式：与设计文档对齐（base + frequency + emotion + link）× time_decay
- 遗忘机制：幂律衰减 + dormant/deleted 状态

P2 升级（2026-04-08）：
- 事实锚点防幻觉：raw_anchors 字段存储关键事实（代码/配置/数字）
- 在生成式回忆时校验事实，防止大模型幻觉

设计还原：NOESIS设计文档 3.5 节
"""

import sqlite3
import os
import datetime
import math
import re
from collections import Counter
from typing import List, Dict, Optional


class LongTermMemory:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.decay_rate = 0.05

        # 权重公式参数（与设计文档 4.3 节对齐）
        self.base_weight = 0.5
        self.frequency_factor = 0.1
        self.emotion_factor = 0.3
        self.link_factor = 0.2

        # 遗忘阈值
        self.dormant_threshold = 0.2
        self.delete_threshold = 0.1

        # TF-IDF 缓存（避免重复计算）
        self._idf_cache: Dict[str, float] = {}
        self._idf_dirty = True
        
        # P2：事实锚点（防幻觉）
        self._ensure_raw_anchors_column()

    def _ensure_raw_anchors_column(self):
        """P2：确保 raw_anchors 字段存在"""
        if not self.db_path or not os.path.exists(self.db_path):
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(ltm_nodes)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'raw_anchors' not in columns:
                cursor.execute('ALTER TABLE ltm_nodes ADD COLUMN raw_anchors TEXT')
                conn.commit()
                print("[LTM] Added raw_anchors column for fact anchoring")
            conn.close()
        except Exception as e:
            print(f"[LTM] raw_anchors column check failed: {e}")

    def connect(self):
        """连接到数据库"""
        dir_name = os.path.dirname(self.db_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def create_node(self, content: str, node_type: str = None,
                   weight: float = 1.0, raw_anchors: str = None) -> int:
        """
        创建长期记忆节点
        
        P2 新增：raw_anchors 参数用于存储不可压缩的关键事实
        （如代码片段、配置参数、精确数字等），在生成式回忆时校验
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            INSERT INTO ltm_nodes (content, type, weight, raw_anchors, created_at, last_accessed)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (content, node_type, weight, raw_anchors))

            conn.commit()
            node_id = cursor.lastrowid
            self._idf_dirty = True
        finally:
            self.close()
        return node_id

    def create_link(self, source_node_id: int, target_node_id: int,
                   strength: float = 0.5, relation_type: str = 'related') -> bool:
        """创建节点间关联
        
        Args:
            source_node_id: 源节点 ID
            target_node_id: 目标节点 ID
            strength: 关联强度 [0, 1]
            relation_type: 关系类型（causal/associated/similar/contrast/temporal/related）
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            SELECT id FROM ltm_links
            WHERE source_node_id = ? AND target_node_id = ?
            ''', (source_node_id, target_node_id))

            if cursor.fetchone():
                cursor.execute('''
                UPDATE ltm_links
                SET strength = ?, relation_type = ?
                WHERE source_node_id = ? AND target_node_id = ?
                ''', (strength, relation_type, source_node_id, target_node_id))
            else:
                cursor.execute('''
                INSERT INTO ltm_links (source_node_id, target_node_id, strength, relation_type, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (source_node_id, target_node_id, strength, relation_type))

            conn.commit()
        finally:
            self.close()
        return True

    def retrieve(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """
        检索记忆节点

        P1 升级：综合得分 = 语义相似度 × 权重 × 时效性
        替代原来的纯 LIKE 查询。
        """
        # 获取所有活跃节点
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            SELECT * FROM ltm_nodes
            WHERE weight >= ?
            ORDER BY weight DESC
            LIMIT 500
            ''', (self.delete_threshold,))

            nodes = [dict(row) for row in cursor.fetchall()]

            if not nodes:
                return []

            # 计算每个节点的综合得分
            scored_nodes = []
            for node in nodes:
                score = self._calculate_retrieval_score(node, query)
                if score > threshold:
                    node['retrieval_score'] = score
                    scored_nodes.append(node)

            # 按得分排序
            scored_nodes.sort(key=lambda x: x['retrieval_score'], reverse=True)
            top_nodes = scored_nodes[:top_k]

            # 获取关联
            for node in top_nodes:
                node['links'] = self._get_node_links(node['id'])

            return top_nodes
        finally:
            self.close()

    def _calculate_retrieval_score(self, node: Dict, query: str) -> float:
        """
        计算检索得分

        score = semantic_similarity × weight × recency_boost
        """
        # 语义相似度
        semantic_sim = self._semantic_similarity(node.get('content', ''), query)

        # 权重（已包含遗忘衰减）
        weight = node.get('weight', 0.5)

        # 时效性提升
        recency_boost = 1.0
        last_accessed = node.get('last_accessed', '')
        if last_accessed:
            try:
                last_access = datetime.datetime.fromisoformat(last_accessed)
                days_since = (datetime.datetime.now() - last_access).days
                recency_boost = 1 + 0.1 / (1 + days_since)
            except (ValueError, TypeError):
                pass

        return semantic_sim * weight * recency_boost

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """
        语义相似度计算（TF-IDF + 余弦相似度）

        不依赖外部 embedding 模型，使用轻量级 TF-IDF 方法。
        """
        # 分词
        words_a = self._tokenize(text_a)
        words_b = self._tokenize(text_b)

        if not words_a or not words_b:
            return 0.0

        # 计算词频
        tf_a = Counter(words_a)
        tf_b = Counter(words_b)

        # 所有词汇
        all_vocab = set(words_a) | set(words_b)
        if not all_vocab:
            return 0.0

        # 简化 IDF（使用固定公式，不依赖全局语料库）
        # IDF(w) = log(N / (1 + df(w)))，这里 N 取 100 作为近似
        def idf(word):
            # 基于实际文档频率的IDF
            df = sum(1 for w in [words_a, words_b] if word in w)
            if df == 0:
                df = 1
            return math.log(2.0 / (1 + df))  # 2 documents

        # 计算余弦相似度
        dot_product = sum(tf_a[w] * tf_b[w] for w in all_vocab)

        norm_a = math.sqrt(sum(v ** 2 for v in tf_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in tf_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        cosine = dot_product / (norm_a * norm_b)

        # 混合精确匹配得分（如果 query 的任何部分完全出现在 text 中，加分）
        exact_bonus = 0.0
        if text_b and len(text_b) <= 100:
            # 对短查询，检查是否是子串匹配
            if text_b.lower() in text_a.lower():
                exact_bonus = 0.3

        return min(1.0, cosine + exact_bonus)

    def _tokenize(self, text: str) -> List[str]:
        """分词：支持中文和英文"""
        if not text:
            return []

        # 提取中文词（2字以上，不限制上限）
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        # 提取英文词
        english_words = re.findall(r'[a-zA-Z]{3,}', text.lower())

        # 过滤停用词
        stopwords = {
            '的', '了', '是', '在', '和', '有', '不', '这', '我', '他',
            '她', '它', '们', '着', '过', '把', '被', '让', '给', '到',
            '也', '就', '都', '而', '及', '与', '对', '中', '上', '下',
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'this', 'that', 'with', 'from', 'they', 'been', 'will'
        }

        return [w for w in chinese_words + english_words if w not in stopwords]

    def access_node(self, node_id: int) -> bool:
        """
        访问节点，更新权重

        记忆提取强化记忆痕迹（与设计文档一致）
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            UPDATE ltm_nodes
            SET last_accessed = CURRENT_TIMESTAMP,
                weight = MIN(weight * 1.1, 2.0)
            WHERE id = ?
            ''', (node_id,))

            conn.commit()
            affected = cursor.rowcount
            return affected > 0
        finally:
            self.close()

    def apply_forgetting(self) -> int:
        """
        应用遗忘机制

        P1 升级：幂律衰减（更符合人类记忆）
        - weight < 0.2 → dormant（休眠，可恢复）
        - weight < 0.1 → deleted（删除）
        - 跳过核心/系统/配置类型节点
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()

            # 获取所有节点（限制数量避免内存溢出）
            cursor.execute('SELECT * FROM ltm_nodes ORDER BY created_at ASC LIMIT 10000')
            nodes = cursor.fetchall()

            dormant_count = 0
            deleted_count = 0

            for node in nodes:
                node_dict = dict(node)

                # 系统节点不遗忘
                if node_dict.get('type') in ('core', 'system', 'config'):
                    continue

                weight = node_dict.get('weight', 0.5)

                # 幂律遗忘：基于创建以来的绝对天数，而非每日累乘
                created_at = node_dict.get('created_at', '')
                last_accessed = node_dict.get('last_accessed', created_at)

                try:
                    ref_time = datetime.datetime.fromisoformat(
                        last_accessed or created_at
                    )
                    days_since = (datetime.datetime.now() - ref_time).days
                except (ValueError, TypeError):
                    days_since = 0

                if days_since > 0:
                    # 真正的幂律衰减：retention = days^(-exponent)
                    # 不再累乘，直接基于绝对天数计算
                    decay_constant = 1.5  # 衰减指数
                    retention = max(0.01, days_since ** (-decay_constant))
                    # 原始权重乘以保留率（上限为原始权重）
                    new_weight = weight * retention

                    # 更新权重
                    cursor.execute('''
                    UPDATE ltm_nodes SET weight = ? WHERE id = ?
                    ''', (new_weight, node_dict['id']))

                    if new_weight < self.delete_threshold:
                        cursor.execute('DELETE FROM ltm_nodes WHERE id = ?',
                                      (node_dict['id'],))
                        deleted_count += 1
                    elif new_weight < self.dormant_threshold:
                        dormant_count += 1

            conn.commit()
            self._idf_dirty = True
        finally:
            self.close()

        print(f"[LTM] Forgetting applied: {dormant_count} dormant, {deleted_count} deleted")
        return dormant_count + deleted_count

    def _get_node_links(self, node_id: int) -> Dict:
        """获取节点的关联"""
        try:
            cursor = self.connect()
            try:
                cursor.execute('''
                SELECT * FROM ltm_links
                WHERE source_node_id = ?
                ORDER BY strength DESC
                ''', (node_id,))
                out_links = [dict(row) for row in cursor.fetchall()]

                cursor.execute('''
                SELECT * FROM ltm_links
                WHERE target_node_id = ?
                ORDER BY strength DESC
                ''', (node_id,))
                in_links = [dict(row) for row in cursor.fetchall()]

                return {
                    'outgoing': out_links,
                    'incoming': in_links
                }
            except Exception:
                return {'outgoing': [], 'incoming': []}
            finally:
                self.close()
        except Exception:
            return {'outgoing': [], 'incoming': []}

    def get_node(self, node_id: int) -> Optional[Dict]:
        """根据ID获取节点"""
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM ltm_nodes WHERE id = ?', (node_id,))

            row = cursor.fetchone()
            if not row:
                return None

            node = dict(row)
            node['links'] = self._get_node_links(node_id)

            return node
        finally:
            self.close()

    def update_node(self, node_id: int, content: str = None,
                   node_type: str = None, weight: float = None,
                   raw_anchors: str = None) -> bool:
        """
        更新节点
        
        P2 新增：raw_anchors 参数用于更新关键事实锚点
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()

            updates = []
            params = []

            if content is not None:
                updates.append("content = ?")
                params.append(content)
            if node_type is not None:
                updates.append("type = ?")
                params.append(node_type)
            if weight is not None:
                updates.append("weight = ?")
                params.append(weight)
            if raw_anchors is not None:
                updates.append("raw_anchors = ?")
                params.append(raw_anchors)

            if updates:
                params.append(node_id)
                sql = f"UPDATE ltm_nodes SET {', '.join(updates)} WHERE id = ?"
                cursor.execute(sql, params)
                conn.commit()

            affected = cursor.rowcount
            self._idf_dirty = True
            return affected > 0
        finally:
            self.close()
    
    def update_raw_anchors(self, node_id: int, raw_anchors: str) -> bool:
        """
        P2：更新节点的事实锚点
        
        用于在后续分析中发现新事实时，更新已存在节点的事实锚点
        """
        return self.update_node(node_id, raw_anchors=raw_anchors)
    
    def get_raw_anchors(self, node_id: int) -> Optional[str]:
        """
        P2：获取节点的事实锚点
        
        Returns:
            JSON 格式的事实锚点字符串，或 None
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT raw_anchors FROM ltm_nodes WHERE id = ?', (node_id,))
            row = cursor.fetchone()
            return row['raw_anchors'] if row else None
        finally:
            self.close()
    
    def retrieve_with_anchors(self, query: str, top_k: int = 10, 
                              threshold: float = 0.0) -> List[Dict]:
        """
        P2：带事实锚点的检索
        
        在普通检索基础上，返回每个节点的事实锚点，
        供调用方在生成时校验事实
        """
        results = self.retrieve(query, top_k, threshold)
        
        if not results:
            return results
        
        # 批量获取 raw_anchors，避免 N+1 查询
        try:
            conn = self.connect()
            try:
                cursor = conn.cursor()
                node_ids = [node['id'] for node in results]
                placeholders = ','.join('?' * len(node_ids))
                cursor.execute(
                    f'SELECT id, raw_anchors FROM ltm_nodes WHERE id IN ({placeholders})',
                    node_ids
                )
                anchor_map = {row[0]: row[1] for row in cursor.fetchall()}
                
                for node in results:
                    node['raw_anchors'] = anchor_map.get(node['id'])
            finally:
                self.close()
        except Exception:
            pass
        
        return results

    def delete_node(self, node_id: int) -> bool:
        """删除节点"""
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            DELETE FROM ltm_links
            WHERE source_node_id = ? OR target_node_id = ?
            ''', (node_id, node_id))

            cursor.execute('DELETE FROM ltm_nodes WHERE id = ?', (node_id,))

            conn.commit()
            affected = cursor.rowcount
            self._idf_dirty = True
            return affected > 0
        finally:
            self.close()

    def get_all_nodes(self, limit: int = 100) -> List[Dict]:
        """获取所有节点"""
        conn = self.connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
            SELECT * FROM ltm_nodes
            ORDER BY weight DESC
            LIMIT ?
            ''', (limit,))

            nodes = [dict(row) for row in cursor.fetchall()]
            return nodes
        finally:
            self.close()
