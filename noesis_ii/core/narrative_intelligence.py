"""
叙事智能模块（Narrative Intelligence）

对应设计文档 §6.2：将碎片化的记忆元素编织成连贯的自我故事。

核心流程：
1. 线索激活（Cue Activation）：当前情境触发相关记忆碎片
2. 模式完成（Pattern Completion）：从部分线索恢复完整记忆表征
3. 约束满足（Constraint Satisfaction）：寻找最大化多重目标的元素配置
4. 线性化表达（Linearization）：将组装结果组织为时间序列
5. 元认知监控（Metacognitive Monitoring）：评估叙事的可靠性和一致性

叙事扭曲的适应性价值：
- 时间压缩：突出关键转折
- 因果归因：建立可传授的教训
- 情感放大：强化记忆标记
- 一致性平滑：填补记忆间隙
"""

import sqlite3
import os
import json
import math
import time
from typing import Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime


class NarrativeIntelligence:
    """
    叙事智能系统

    将碎片化的记忆元素编织成连贯的自我故事。
    基于设计文档 6.2 节的五步叙事编织过程。
    """

    # 叙事元素类型
    ELEMENT_TYPES = ['event', 'character', 'setting', 'emotion', 'outcome', 'lesson']

    # 叙事扭曲类型（适应性价值）
    DISTORTION_TYPES = ['time_compression', 'causal_attribution',
                        'emotional_amplification', 'consistency_smoothing']

    def __init__(self, db_path: str, config: Dict = None):
        self.db_path = db_path
        self.config = config or {}

        # 配置参数
        self.max_narrative_length = self.config.get('max_narrative_length', 50)
        self.min_cue_similarity = self.config.get('min_cue_similarity', 0.3)
        self.max_elements_per_step = self.config.get('max_elements_per_step', 20)

        # 置信度阈值
        self.min_reliability = self.config.get('min_reliability', 0.4)
        self.dissonance_threshold = self.config.get('dissonance_threshold', 0.7)

    def connect(self) -> sqlite3.Connection:
        """连接数据库"""
        dir_name = os.path.dirname(self.db_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def close(self, conn):
        """关闭数据库连接"""
        if conn:
            conn.close()

    # ═══════════════════════════════════════════════════════
    # 核心叙事编织流程
    # ═══════════════════════════════════════════════════════

    def weave_narrative(self, cue: str, context: Dict = None) -> Dict:
        """
        完整叙事编织流程

        线索激活 → 模式完成 → 约束满足 → 线性化 → 元认知监控

        Args:
            cue: 叙事线索（当前情境/问题/触发词）
            context: 额外上下文信息

        Returns:
            {
                'status': 'success'|'insufficient_elements',
                'narrative': str,         # 叙事文本
                'elements': [...],        # 叙事元素列表
                'reliability': float,     # 可靠性评分
                'distortions': [...],     # 检测到的叙事扭曲
                'consistency': float,     # 一致性评分
                'steps': {...},           # 各步骤详情
            }
        """
        context = context or {}

        # Step 1: 线索激活
        activated = self._cue_activation(cue, context)

        # Step 2: 模式完成
        completed = self._pattern_completion(activated, cue)

        # Step 3: 约束满足
        optimized = self._constraint_satisfaction(completed, context)

        # Step 4: 线性化表达
        linearized = self._linearization(optimized)

        # Step 5: 元认知监控
        monitored = self._metacognitive_monitoring(linearized, cue)

        if not optimized['elements']:
            return {
                'status': 'insufficient_elements',
                'narrative': f'关于「{cue}」，当前记忆中尚无足够素材形成连贯叙事。',
                'elements': [],
                'reliability': 0.0,
                'distortions': [],
                'consistency': 0.0,
                'steps': {
                    'cue_activation': activated,
                    'pattern_completion': completed,
                    'constraint_satisfaction': optimized,
                    'linearization': linearized,
                    'metacognitive_monitoring': monitored,
                },
            }

        return {
            'status': 'success',
            'narrative': monitored['narrative'],
            'elements': optimized['elements'],
            'reliability': monitored['reliability'],
            'distortions': monitored['distortions'],
            'consistency': monitored['consistency'],
            'steps': {
                'cue_activation': activated,
                'pattern_completion': completed,
                'constraint_satisfaction': optimized,
                'linearization': linearized,
                'metacognitive_monitoring': monitored,
            },
        }

    # ═══════════════════════════════════════════════════════
    # Step 1: 线索激活（Cue Activation）
    # ═══════════════════════════════════════════════════════

    def _cue_activation(self, cue: str, context: Dict) -> Dict:
        """
        线索激活：当前情境触发相关记忆碎片

        从 LTM 节点、种子、意识事件中检索与线索相关的元素。
        """
        elements = []

        # 1a. 从长期记忆中激活
        ltm_elements = self._activate_from_ltm(cue)
        elements.extend(ltm_elements)

        # 1b. 从种子中激活
        seed_elements = self._activate_from_seeds(cue)
        elements.extend(seed_elements)

        # 1c. 从意识事件中激活
        event_elements = self._activate_from_conscious_events(cue)
        elements.extend(event_elements)

        # 按激活强度排序
        elements.sort(key=lambda e: e['activation_strength'], reverse=True)

        return {
            'step': 'cue_activation',
            'cue': cue,
            'total_activated': len(elements),
            'elements': elements[:self.max_elements_per_step],
            'sources': {
                'ltm': len(ltm_elements),
                'seeds': len(seed_elements),
                'conscious_events': len(event_elements),
            }
        }

    def _activate_from_ltm(self, cue: str) -> List[Dict]:
        """从长期记忆中激活相关节点"""
        conn = self.connect()
        cursor = conn.cursor()

        # 获取最近活跃的 LTM 节点
        cursor.execute('''
            SELECT id, content, type, weight, created_at, last_accessed
            FROM ltm_nodes
            WHERE weight >= 0.1
            ORDER BY weight DESC, last_accessed DESC
            LIMIT 100
        ''')
        nodes = [dict(row) for row in cursor.fetchall()]
        self.close(conn)

        # 计算每个节点与线索的相似度
        activated = []
        for node in nodes:
            similarity = self._compute_similarity(cue, node.get('content', ''))
            if similarity >= self.min_cue_similarity:
                activated.append({
                    'source': 'ltm',
                    'id': node['id'],
                    'content': node.get('content', ''),
                    'element_type': node.get('type', 'episodic'),
                    'weight': node.get('weight', 0.5),
                    'timestamp': node.get('created_at', ''),
                    'activation_strength': round(similarity * node.get('weight', 0.5), 4),
                })

        return activated

    def _activate_from_seeds(self, cue: str) -> List[Dict]:
        """从种子系统中激活相关种子"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, trace_id, content_summary, trace_type, strength,
                   access_count, condition_pattern
            FROM memory_traces
            WHERE is_active = 1 AND strength > 0.1
            ORDER BY strength DESC
            LIMIT 100
        ''')
        seeds = [dict(row) for row in cursor.fetchall()]
        self.close(conn)

        activated = []
        for seed in seeds:
            # 多维度匹配
            content_sim = self._compute_similarity(cue, seed.get('content_summary', ''))

            # 条件模式匹配
            pattern_sim = 0.0
            pattern = seed.get('condition_pattern', '')
            try:
                cp = json.loads(pattern) if pattern else {}
                keywords = cp.get('primary', {}).get('keywords', [])
                if keywords:
                    matches = sum(1 for kw in keywords if kw in cue)
                    pattern_sim = matches / len(keywords) if keywords else 0.0
            except (json.JSONDecodeError, TypeError):
                pass

            # 综合激活强度 = 内容相似度 × 种子势力 + 条件匹配加成
            strength = content_sim * seed.get('strength', 0.1) + pattern_sim * 0.3
            strength = min(1.0, strength)

            if strength >= self.min_cue_similarity * 0.8:  # 种子阈值稍低
                activated.append({
                    'source': 'seeds',
                    'id': seed['id'],
                    'seed_id': seed.get('trace_id', ''),
                    'content': seed.get('content_summary', ''),
                    'element_type': seed.get('trace_type', 'episodic'),
                    'strength': seed.get('strength', 0.1),
                    'activation_strength': round(strength, 4),
                })

        return activated

    def _activate_from_conscious_events(self, cue: str) -> List[Dict]:
        """从意识事件日志中激活"""
        conn = self.connect()
        try:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    SELECT id, content, relevance_score, timestamp
                    FROM activity_events
                    ORDER BY timestamp DESC
                    LIMIT 50
                ''')
                events = [dict(row) for row in cursor.fetchall()]
            except Exception:
                # Table might not exist yet
                events = []
                self.close(conn)
                return []

            activated = []
            for event in events:
                similarity = self._compute_similarity(cue, event.get('content', ''))
                if similarity >= self.min_cue_similarity:
                    activated.append({
                        'source': 'conscious_events',
                        'id': event['id'],
                        'content': event.get('content', ''),
                        'element_type': 'conscious_event',
                        'relevance_score': event.get('relevance_score', 0),
                        'timestamp': event.get('timestamp', ''),
                        'activation_strength': round(similarity, 4),
                    })

            return activated
        finally:
            self.close(conn)

    # ═══════════════════════════════════════════════════════
    # Step 2: 模式完成（Pattern Completion）
    # ═══════════════════════════════════════════════════════

    def _pattern_completion(self, activated: Dict, cue: str) -> Dict:
        """
        模式完成：从部分线索恢复完整记忆表征

        扩展激活的元素，补充缺失的部分：
        - 时间连续性：填充时间间隙
        - 因果链：补全因果链中缺失的环节
        - 情境完整：补充缺失的情境信息
        """
        elements = activated.get('elements', [])
        completed_elements = list(elements)  # 复制原始元素

        if not elements:
            return {
                'step': 'pattern_completion',
                'original_count': 0,
                'completed_count': 0,
                'elements': [],
                'completions': [],
            }

        completions = []

        # 2a. 时间扩展：查找与已激活元素时间相邻的节点
        time_filled = self._fill_temporal_gaps(elements)
        if time_filled:
            completions.append({'type': 'temporal_fill', 'count': len(time_filled)})
            completed_elements.extend(time_filled)

        # 2b. 关联扩展：查找与已激活元素有关联的节点
        linked = self._follow_associations(elements)
        if linked:
            completions.append({'type': 'association_follow', 'count': len(linked)})
            completed_elements.extend(linked)

        # 2c. 去重和排序
        seen_ids = set()
        unique_elements = []
        for e in completed_elements:
            eid = f"{e['source']}_{e['id']}"
            if eid not in seen_ids:
                seen_ids.add(eid)
                unique_elements.append(e)

        unique_elements.sort(key=lambda e: e.get('activation_strength', 0), reverse=True)

        return {
            'step': 'pattern_completion',
            'original_count': len(elements),
            'completed_count': len(unique_elements),
            'elements': unique_elements[:self.max_narrative_length],
            'completions': completions,
        }

    def _fill_temporal_gaps(self, elements: List[Dict]) -> List[Dict]:
        """填充时间间隙：查找与已激活元素时间相邻的节点"""
        time_elements = [e for e in elements if e.get('timestamp')]
        if len(time_elements) < 2:
            return []

        filled = []
        conn = self.connect()
        try:
            cursor = conn.cursor()

            for elem in time_elements[:5]:
                timestamp = elem.get('timestamp', '')
                if not timestamp or timestamp == '':
                    continue

                try:
                    # 使用参数化查询避免SQL注入
                    elem_ids = [e['id'] for e in elements if isinstance(e['id'], (int, float))]
                    if not elem_ids:
                        continue
                    placeholders = ','.join('?' * len(elem_ids))

                    cursor.execute(f'''
                        SELECT id, content, type, weight, created_at, last_accessed
                        FROM ltm_nodes
                        WHERE created_at LIKE ?
                          AND weight >= 0.1
                          AND id NOT IN ({placeholders})
                        ORDER BY weight DESC
                        LIMIT 3
                    ''', [timestamp[:10] + '%'] + elem_ids)

                    for row in cursor.fetchall():
                        node = dict(row)
                        filled.append({
                            'source': 'ltm',
                            'id': node['id'],
                            'content': node.get('content', ''),
                            'element_type': 'temporal_fill',
                            'weight': node.get('weight', 0.3),
                            'timestamp': node.get('created_at', ''),
                            'activation_strength': 0.25,
                        })
                except Exception:
                    continue
        finally:
            self.close(conn)
        return filled

    def _follow_associations(self, elements: List[Dict]) -> List[Dict]:
        """跟随关联：查找与已激活元素有关联的节点"""
        linked = []
        conn = self.connect()
        try:
            cursor = conn.cursor()

            ltm_ids = [e['id'] for e in elements if e['source'] == 'ltm' and isinstance(e['id'], (int, float))]
            if not ltm_ids:
                return linked

            ltm_ids = ltm_ids[:10]  # 限制数量
            placeholders = ','.join('?' * len(ltm_ids))

            try:
                cursor.execute(f'''
                    SELECT n.id, n.content, n.type, n.weight, n.created_at,
                           l.strength as link_strength
                    FROM ltm_nodes n
                    JOIN ltm_links l ON (
                        (l.source_node_id = n.id AND l.target_node_id IN ({placeholders}))
                        OR (l.target_node_id = n.id AND l.source_node_id IN ({placeholders}))
                    )
                    WHERE n.weight >= 0.1
                      AND n.id NOT IN ({placeholders})
                    ORDER BY l.strength DESC
                    LIMIT 10
                ''', ltm_ids + ltm_ids + ltm_ids)

                for row in cursor.fetchall():
                    node = dict(row)
                    linked.append({
                        'source': 'ltm',
                        'id': node['id'],
                        'content': node.get('content', ''),
                        'element_type': 'associated',
                        'weight': node.get('weight', 0.3),
                        'link_strength': node.get('link_strength', 0.5),
                        'timestamp': node.get('created_at', ''),
                        'activation_strength': 0.35,
                    })
            except Exception:
                pass
        finally:
            self.close(conn)
        return linked

    # ═══════════════════════════════════════════════════════
    # Step 3: 约束满足（Constraint Satisfaction）
    # ═══════════════════════════════════════════════════════

    def _constraint_satisfaction(self, completed: Dict, context: Dict) -> Dict:
        """
        约束满足：寻找最大化多重目标的元素配置

        多重约束：
        - 相关性：与线索的相关程度
        - 一致性：元素间的内在一致性
        - 覆盖度：涵盖多种类型的元素
        - 时间连续性：时间序列的连贯性
        - 情感连贯性：情绪的合理变化
        """
        elements = completed.get('elements', [])

        if not elements:
            return {
                'step': 'constraint_satisfaction',
                'elements': [],
                'constraint_scores': {},
                'selected_count': 0,
            }

        # 对每个元素计算多维度约束分数
        scored_elements = []
        for elem in elements:
            scores = self._evaluate_constraints(elem, elements, context)
            elem['constraint_scores'] = scores
            # 综合分数：加权平均
            elem['overall_constraint_score'] = round(
                scores['relevance'] * 0.4 +
                scores['diversity'] * 0.2 +
                scores['recency'] * 0.15 +
                scores['coherence'] * 0.15 +
                scores['significance'] * 0.1,
                4
            )
            scored_elements.append(elem)

        # 按综合分数排序，选取最优配置
        scored_elements.sort(key=lambda e: e.get('overall_constraint_score', 0), reverse=True)
        selected = scored_elements[:self.max_narrative_length]

        # 计算总体约束满足度
        overall_scores = {
            'relevance': round(sum(e['constraint_scores']['relevance'] for e in selected) / len(selected), 4) if selected else 0,
            'diversity': round(self._compute_diversity(selected), 4),
            'recency': round(sum(e['constraint_scores']['recency'] for e in selected) / len(selected), 4) if selected else 0,
            'coherence': round(sum(e['constraint_scores']['coherence'] for e in selected) / len(selected), 4) if selected else 0,
            'significance': round(sum(e['constraint_scores']['significance'] for e in selected) / len(selected), 4) if selected else 0,
        }

        return {
            'step': 'constraint_satisfaction',
            'elements': selected,
            'constraint_scores': overall_scores,
            'selected_count': len(selected),
        }

    def _evaluate_constraints(self, element: Dict, all_elements: List[Dict],
                               context: Dict) -> Dict:
        """评估单个元素的多维度约束分数"""
        return {
            'relevance': element.get('activation_strength', 0),
            'diversity': 0.5,  # 会在整体层面重新计算
            'recency': self._compute_recency(element),
            'coherence': 0.5,  # 会在整体层面重新计算
            'significance': element.get('weight', element.get('potency', 0.5)),
        }

    def _compute_recency(self, element: Dict) -> float:
        """计算时间新鲜度"""
        timestamp = element.get('timestamp', '')
        if not timestamp:
            return 0.3

        try:
            if isinstance(timestamp, (int, float)):
                # Unix timestamp
                days = (time.time() - timestamp) / 86400
            else:
                # ISO string
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00')) if 'Z' in timestamp else datetime.fromisoformat(timestamp)
                days = (datetime.now() - dt).days
        except (ValueError, TypeError, OSError):
            return 0.3

        # 指数衰减：7天内高，30天后低
        return max(0.1, math.exp(-days / 14))

    def _compute_diversity(self, elements: List[Dict]) -> float:
        """计算元素类型的覆盖度"""
        types = set(e.get('element_type', 'unknown') for e in elements)
        if not types:
            return 0.0
        return len(types) / len(self.ELEMENT_TYPES)

    # ═══════════════════════════════════════════════════════
    # Step 4: 线性化表达（Linearization）
    # ═══════════════════════════════════════════════════════

    def _linearization(self, optimized: Dict) -> Dict:
        """
        线性化表达：将组装结果组织为时间序列

        将选定的元素按时间/逻辑顺序组织成连贯的叙事结构：
        1. 按时间排序
        2. 识别叙事转折点
        3. 生成叙事结构（起承转合）
        """
        elements = optimized.get('elements', [])

        if not elements:
            return {
                'step': 'linearization',
                'narrative_structure': None,
                'elements': [],
                'turning_points': [],
            }

        # 4a. 按时间排序
        sorted_elements = self._sort_by_time(elements)

        # 4b. 识别叙事转折点（权重/情绪突然变化的节点）
        turning_points = self._identify_turning_points(sorted_elements)

        # 4c. 构建叙事结构
        structure = self._build_narrative_structure(sorted_elements, turning_points)

        # 4d. 生成叙事文本
        narrative_text = self._generate_narrative_text(sorted_elements, structure, turning_points)

        return {
            'step': 'linearization',
            'narrative_structure': structure,
            'elements': sorted_elements,
            'turning_points': turning_points,
            'narrative_text': narrative_text,
        }

    def _sort_by_time(self, elements: List[Dict]) -> List[Dict]:
        """按时间排序元素"""
        def get_sort_key(e):
            ts = e.get('timestamp', '')
            if not ts:
                return (0, 0, 0)
            try:
                if isinstance(ts, (int, float)):
                    return (1, ts, 0)
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00')) if 'Z' in ts else datetime.fromisoformat(ts)
                return (1, dt.timestamp(), 0)
            except (ValueError, TypeError):
                return (0, 0, 0)

        return sorted(elements, key=get_sort_key)

    def _identify_turning_points(self, elements: List[Dict]) -> List[Dict]:
        """
        识别叙事转折点

        转折点判断条件：
        - 权重突变（相邻节点权重差 > 阈值）
        - 自我相关性突变
        - 长期影响性质变化
        """
        if len(elements) < 3:
            return []

        turning_points = []
        weights = [e.get('weight', e.get('potency', 0.5)) for e in elements]

        for i in range(1, len(elements) - 1):
            # 权重突变检测
            prev_diff = abs(weights[i] - weights[i - 1])
            next_diff = abs(weights[i + 1] - weights[i])

            if prev_diff > 0.3 or next_diff > 0.3:
                turning_points.append({
                    'index': i,
                    'element': elements[i],
                    'type': 'significance_shift',
                    'weight_change': round(max(prev_diff, next_diff), 3),
                })

            # 长期影响性质变化（记忆痕迹）
            if 'long_term_impact' in elements[i] and 'long_term_impact' in elements[i - 1]:
                if elements[i]['long_term_impact'] != elements[i - 1]['long_term_impact']:
                    turning_points.append({
                        'index': i,
                        'element': elements[i],
                        'type': 'impact_shift',
                        'from': elements[i - 1].get('long_term_impact'),
                        'to': elements[i].get('long_term_impact'),
                    })

        return turning_points

    def _build_narrative_structure(self, elements: List[Dict],
                                    turning_points: List[Dict]) -> Dict:
        """
        构建叙事结构（起承转合）

        将元素分成：开头（起）→ 发展（承）→ 转折（转）→ 结尾（合）
        """
        if not elements:
            return {'phases': {}}

        n = len(elements)
        phase_size = max(1, n // 4)

        return {
            'total_elements': n,
            'phases': {
                'introduction': {  # 起
                    'range': (0, min(phase_size, n)),
                    'element_count': min(phase_size, n),
                    'description': '叙事开端：设定情境和基调',
                },
                'development': {  # 承
                    'range': (phase_size, min(phase_size * 2, n)),
                    'element_count': max(0, min(phase_size, n - phase_size)),
                    'description': '叙事发展：展开主要内容和模式',
                },
                'turning': {  # 转
                    'range': (phase_size * 2, min(phase_size * 3, n)),
                    'element_count': max(0, min(phase_size, n - phase_size * 2)),
                    'turning_points': [tp for tp in turning_points
                                      if phase_size * 2 <= tp['index'] < phase_size * 3],
                    'description': '叙事转折：关键变化和冲突',
                },
                'conclusion': {  # 合
                    'range': (phase_size * 3, n),
                    'element_count': max(0, n - phase_size * 3),
                    'description': '叙事结尾：整合与启示',
                },
            }
        }

    def _generate_narrative_text(self, elements: List[Dict],
                                  structure: Dict,
                                  turning_points: List[Dict]) -> str:
        """
        生成叙事文本

        基于结构化的元素序列，生成连贯的叙事文本。
        包含叙事扭曲（适应性简化）。
        """
        if not elements:
            return ''

        # 提取关键内容
        key_contents = []
        for elem in elements[:15]:  # 最多取 15 个元素
            content = elem.get('content', '')
            if content and len(content) > 5:
                # 截取摘要
                summary = content[:100] + ('...' if len(content) > 100 else '')
                key_contents.append(summary)

        if not key_contents:
            return '记忆碎片尚不足以形成连贯叙事。'

        # 应用叙事扭曲
        narrative_text = self._apply_narrative_distortions(key_contents, turning_points)

        return narrative_text

    def _apply_narrative_distortions(self, contents: List[str],
                                      turning_points: List[Dict]) -> str:
        """
        应用叙事扭曲（适应性价值）

        - 时间压缩：突出关键转折，压缩中间过程
        - 因果归因：建立可传授的教训
        - 情感放大：强化记忆标记
        - 一致性平滑：填补记忆间隙
        """
        if len(contents) <= 2:
            return '；'.join(contents)

        # 时间压缩：如果元素太多，只保留首尾和转折点
        if len(contents) > 5:
            compressed = []
            compressed.append(contents[0])  # 开头
            if len(contents) > 3:
                compressed.append(contents[len(contents) // 3])  # 1/3 处
            compressed.append(contents[len(contents) // 2])  # 中间
            if len(contents) > 4:
                compressed.append(contents[2 * len(contents) // 3])  # 2/3 处
            compressed.append(contents[-1])  # 结尾
            contents = compressed

        # 因果归因
        causal_hint = ''
        if turning_points:
            for tp in turning_points[:2]:
                if tp.get('type') == 'impact_shift':
                    causal_hint = '在这个过程中，经历了重要的转变——'

        # 一致性平滑：用过渡词连接
        connectors = ['最初', '随后', '在此期间', '值得注意的是', '最终']
        parts = []
        for i, content in enumerate(contents):
            if i < len(connectors):
                parts.append(f'{connectors[i]}，{content}')
            else:
                parts.append(content)

        narrative = causal_hint + '；'.join(parts) + '。'
        return narrative

    # ═══════════════════════════════════════════════════════
    # Step 5: 元认知监控（Metacognitive Monitoring）
    # ═══════════════════════════════════════════════════════

    def _metacognitive_monitoring(self, linearized: Dict, cue: str) -> Dict:
        """
        元认知监控：评估叙事的可靠性和一致性

        检查项：
        - 可靠性：基于元素来源和权重的可信度
        - 一致性：叙事元素间的一致性评分
        - 叙事扭曲检测：识别可能的扭曲
        - 覆盖度：叙事对线索的覆盖程度
        """
        elements = linearized.get('elements', [])
        narrative_text = linearized.get('narrative_text', '')
        constraint_scores = linearized.get('constraint_scores', {})

        if not elements:
            return {
                'step': 'metacognitive_monitoring',
                'narrative': '',
                'reliability': 0.0,
                'consistency': 0.0,
                'distortions': [],
                'coverage': 0.0,
                'assessment': 'no_elements',
            }

        # 5a. 可靠性评估
        reliability = self._assess_reliability(elements)

        # 5b. 一致性评估
        consistency = self._assess_consistency(elements)

        # 5c. 叙事扭曲检测
        distortions = self._detect_distortions(elements, linearized)

        # 5d. 覆盖度评估
        coverage = self._assess_coverage(elements, cue)

        # 综合评估
        if reliability > 0.7 and consistency > 0.6:
            assessment = 'reliable'
        elif reliability > 0.5 and consistency > 0.4:
            assessment = 'moderately_reliable'
        elif reliability > self.min_reliability:
            assessment = 'tentative'
        else:
            assessment = 'unreliable'

        return {
            'step': 'metacognitive_monitoring',
            'narrative': narrative_text,
            'reliability': round(reliability, 4),
            'consistency': round(consistency, 4),
            'distortions': distortions,
            'coverage': round(coverage, 4),
            'assessment': assessment,
        }

    def _assess_reliability(self, elements: List[Dict]) -> float:
        """评估叙事可靠性"""
        if not elements:
            return 0.0

        scores = []
        for elem in elements:
            # 基于来源的可靠性
            source_reliability = {
                'ltm': 0.8,
                'seeds': 0.7,
                'conscious_events': 0.9,
            }.get(elem.get('source', ''), 0.5)

            # 基于权重的可靠性
            weight = elem.get('weight', elem.get('potency', 0.5))
            weight_reliability = min(1.0, weight)

            # 基于激活强度的可靠性
            activation = elem.get('activation_strength', 0)
            activation_reliability = min(1.0, activation * 2)

            scores.append(source_reliability * 0.3 + weight_reliability * 0.4 + activation_reliability * 0.3)

        return sum(scores) / len(scores)

    def _assess_consistency(self, elements: List[Dict]) -> float:
        """评估叙事元素间的一致性"""
        if len(elements) < 2:
            return 0.5

        # 计算相邻元素的内容相似度（高相似度 = 高一致性）
        similarities = []
        for i in range(len(elements) - 1):
            text_a = elements[i].get('content', '')
            text_b = elements[i + 1].get('content', '')
            if text_a and text_b:
                sim = self._compute_similarity(text_a, text_b)
                similarities.append(sim)

        if not similarities:
            return 0.5

        # 一致性 = 平均相似度（但也不能太高，否则是重复）
        avg_sim = sum(similarities) / len(similarities)
        # 最佳一致性区间：0.1-0.5（有主题连续性但不重复）
        if 0.1 <= avg_sim <= 0.5:
            return 0.8
        elif avg_sim > 0.5:
            return 0.5  # 太相似，可能重复
        else:
            return max(0.2, avg_sim * 2)  # 太不相似

    def _detect_distortions(self, elements: List[Dict], linearized: Dict) -> List[Dict]:
        """
        检测叙事扭曲

        识别可能的：
        - 时间压缩（元素数量远少于原始激活数）
        - 因果归因（转折点的因果推断可能不准确）
        - 情感放大（高情绪强度元素的权重过高）
        - 一致性平滑（填充元素与真实记忆的差异）
        """
        distortions = []

        # 时间压缩检测
        original_count = linearized.get('narrative_structure', {}).get('total_elements', 0)
        if original_count > 0 and len(elements) < original_count * 0.5:
            distortions.append({
                'type': 'time_compression',
                'severity': 'moderate',
                'description': f'从 {original_count} 个元素压缩到 {len(elements)} 个，可能丢失中间细节',
            })

        # 情感放大检测
        high_emotion = [e for e in elements if e.get('element_type') == 'emotional']
        if len(high_emotion) > len(elements) * 0.3:
            distortions.append({
                'type': 'emotional_amplification',
                'severity': 'low',
                'description': f'情绪元素占比 {len(high_emotion)/len(elements):.0%}，可能放大了情感体验',
            })

        # 一致性平滑检测
        fill_elements = [e for e in elements if e.get('element_type') in ('temporal_fill', 'associated')]
        if fill_elements:
            distortions.append({
                'type': 'consistency_smoothing',
                'severity': 'low',
                'description': f'包含 {len(fill_elements)} 个填充元素，部分细节可能由系统推断',
            })

        if not distortions:
            distortions.append({
                'type': 'minimal_distortion',
                'severity': 'none',
                'description': '叙事扭曲程度低，较为客观',
            })

        return distortions

    def _assess_coverage(self, elements: List[Dict], cue: str) -> float:
        """评估叙事对线索的覆盖度"""
        if not elements or not cue:
            return 0.0

        # 计算线索关键词在叙事元素中的覆盖率
        import re
        cue_keywords = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', cue)
        if not cue_keywords:
            return 0.5

        all_text = ' '.join(e.get('content', '') for e in elements)
        covered = sum(1 for kw in cue_keywords if kw in all_text)
        return covered / len(cue_keywords)

    # ═══════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        计算两段文本的相似度（TF 余弦相似度）

        支持中文和英文
        """
        if not text_a or not text_b:
            return 0.0

        def tokenize(text):
            import re
            tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text.lower())
            return tokens

        tokens_a = tokenize(text_a)
        tokens_b = tokenize(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        # 计算词频
        freq_a = Counter(tokens_a)
        freq_b = Counter(tokens_b)

        # 余弦相似度
        all_vocab = set(freq_a.keys()) | set(freq_b.keys())
        dot_product = sum(freq_a[w] * freq_b[w] for w in all_vocab)
        norm_a = math.sqrt(sum(v ** 2 for v in freq_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in freq_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_narrative_history(self, limit: int = 10) -> List[Dict]:
        """获取最近的叙事历史"""
        if not self.db_path:
            return []

        conn = self.connect()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT * FROM narrative_history
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            history = [dict(row) for row in cursor.fetchall()]
        except Exception:
            history = []
        finally:
            self.close(conn)

        return history

    def save_narrative(self, cue: str, result: Dict) -> int:
        """
        保存叙事到数据库

        Returns:
            记录 ID
        """
        if not self.db_path:
            return -1

        conn = self.connect()
        cursor = conn.cursor()

        # 确保叙事历史表存在
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS narrative_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cue TEXT NOT NULL,
                narrative TEXT NOT NULL,
                reliability REAL,
                consistency REAL,
                assessment TEXT,
                element_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            INSERT INTO narrative_history (cue, narrative, reliability, consistency, assessment, element_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            cue,
            result.get('narrative', ''),
            result.get('reliability', 0),
            result.get('consistency', 0),
            result.get('assessment', ''),
            len(result.get('elements', [])),
        ))

        record_id = cursor.lastrowid
        conn.commit()
        self.close(conn)
        return record_id
