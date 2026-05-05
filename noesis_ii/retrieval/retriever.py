"""
统一检索接口（Retriever）

多源记忆的统一检索：
- 内部记忆（长期记忆）
- 工作记忆
- 人格记忆痕迹
- 外部资源（延展心智）

P1 升级：
- semantic_retrieve 使用 TF-IDF + 余弦相似度
- 整合排序：等级优先 + 权重 + 语义相似度
- remember() 区分构成性/工具性外部资源
"""

import sqlite3
import os
import re
import math
import warnings
from collections import Counter
from typing import List, Dict, Optional

# 旧架构模块（可选导入）
try:
    from noesis_ii.core.long_term_memory import LongTermMemory
    from noesis_ii.core.working_memory import WorkingMemory
except ImportError as e:
    LongTermMemory = None
    WorkingMemory = None

# 新架构模块（可选导入）
try:
    from noesis_ii.core.persona_profile import PersonaProfile
    from noesis_ii.core.multi_criteria_retriever import MultiCriteriaRetriever, RetrievalCriteria
except ImportError:
    PersonaProfile = None
    MultiCriteriaRetriever = None
    RetrievalCriteria = None


class Retriever:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.long_term_memory = LongTermMemory(db_path) if LongTermMemory else None
        self.working_memory = WorkingMemory(db_path) if WorkingMemory else None
        self.persona_profile = PersonaProfile(db_path) if PersonaProfile else None
        self.multi_criteria_retriever = MultiCriteriaRetriever(db_path) if MultiCriteriaRetriever else None

    def retrieve(self, query: str, top_k: int = 10,
                sources: List[str] = None) -> Dict:
        """
        多源记忆的统一检索

        Args:
            query: 查询文本
            top_k: 返回数量
            sources: 检索来源列表 ['internal', 'external', 'seeds', 'working_memory']
        """
        if sources is None:
            sources = ['internal', 'external', 'seeds', 'working_memory']

        results = {}

        # 内部记忆检索
        if 'internal' in sources:
            results['internal'] = self._retrieve_internal(query, top_k)

        # 工作记忆检索
        if 'working_memory' in sources:
            results['working_memory'] = self._retrieve_working_memory(query, top_k)

        # 外部资源检索
        if 'external' in sources:
            results['external'] = self._retrieve_external(query, top_k)

        # 种子检索
        if 'seeds' in sources:
            results['seeds'] = self._retrieve_seeds(query, top_k)

        # 整合结果
        results['integrated'] = self._integrate_results(results, top_k)

        return results

    def _retrieve_internal(self, query: str, top_k: int) -> List[Dict]:
        """检索内部记忆（使用 MultiCriteriaRetriever）"""
        if self.multi_criteria_retriever:
            # 使用新的多条件检索器
            criteria = RetrievalCriteria(
                semantic_query=query,
                access_preference='recent'
            )
            results = self.multi_criteria_retriever.retrieve(criteria, top_k)
            
            internal_results = []
            for result in results:
                internal_results.append({
                    'id': result.memory_id,
                    'content': result.content,
                    'type': 'memory',
                    'weight': result.score,
                    'source': 'internal',
                    'score': result.score,
                    'criteria_matched': result.criteria_matched
                })
            return internal_results
        elif self.long_term_memory:
            # 降级为旧的 LTM 检索
            nodes = self.long_term_memory.retrieve(query, top_k)
            results = []
            for node in nodes:
                results.append({
                    'id': node['id'],
                    'content': node.get('content', ''),
                    'type': node.get('type', 'memory'),
                    'weight': node.get('weight', 0),
                    'source': 'internal',
                    'score': node.get('retrieval_score', 0),
                    'links': node.get('links', {})
                })
            return results
        return []

    def _retrieve_external(self, query: str, top_k: int) -> List[Dict]:
        """检索外部资源（简化版）"""
        # 外部资源模块已移除，返回空列表
        return []

    def _retrieve_seeds(self, query: str, top_k: int) -> List[Dict]:
        """检索人格记忆痕迹（新架构）
        
        改进3（叙事重建）：返回 narrative_hook 字段，
        让调用方区分 [HOOK] 原始钩子 vs [RECONSTRUCTED] 完整内容
        """
        if not self.persona_profile:
            return []
        
        try:
            # 从 PersonaProfile 检索记忆痕迹
            traces = self.persona_profile.retrieve_memories(query, top_k=top_k)
            
            results = []
            for trace in traces:
                # 尝试获取 narrative_hook（从 emotion_data 中提取）
                narrative_hook = ''
                emotion_data = trace.get('emotion_data')
                if emotion_data and isinstance(emotion_data, str):
                    try:
                        ed = json.loads(emotion_data)
                        narrative_hook = ed.get('narrative_hook', '')
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif emotion_data and isinstance(emotion_data, dict):
                    narrative_hook = emotion_data.get('narrative_hook', '')
                
                results.append({
                    'id': trace.get('id', ''),
                    'content': trace.get('content', ''),
                    'type': trace.get('type', 'memory_trace'),
                    'intensity': trace.get('intensity', 0),
                    'source': 'persona_profile',
                    'score': trace.get('score', 0),
                    'narrative_hook': narrative_hook,
                })
            
            return results
        except Exception as e:
            print(f"[RETRIEVER] Error retrieving from PersonaProfile: {e}")
            return []

    def _retrieve_working_memory(self, query: str, top_k: int) -> List[Dict]:
        """检索工作记忆（使用语义匹配而非简单子串匹配）"""
        all_entries = self.working_memory.get_all()

        matching_entries = []
        for entry in all_entries:
            # 使用语义相似度匹配
            similarity = self._text_similarity(
                entry.get('content', ''), query
            )
            if similarity > 0.1:
                matching_entries.append({
                    'id': entry.get('id', ''),
                    'content': entry.get('content', ''),
                    'type': 'working_memory',
                    'weight': 1.0,
                    'source': 'working_memory',
                    'timestamp': entry.get('timestamp', ''),
                    'similarity': similarity
                })

        # 按相似度排序
        matching_entries.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return matching_entries[:top_k]

    def _integrate_results(self, results: Dict, top_k: int) -> List[Dict]:
        """
        整合检索结果

        排序规则：等级优先 + 综合得分
        - A：工作记忆（最高等级）
        - B：构成性外部（视为自己的记忆）
        - C：内部记忆（长期记忆节点）
        - D：人格记忆痕迹
        - E：工具性外部（辅助）
        """
        all_results = []

        if 'working_memory' in results:
            for r in results['working_memory']:
                r['level'] = 'A'
                r['level_name'] = '工作记忆'
                r['sort_key'] = 0
                all_results.append(r)

        if 'external' in results:
            for r in results['external']:
                if r.get('retrieval_type') == 'constitutive':
                    r['level'] = 'B'
                    r['level_name'] = '构成性外部记忆'
                    r['sort_key'] = 1
                else:
                    r['level'] = 'E'
                    r['level_name'] = '工具性外部资源'
                    r['sort_key'] = 4
                all_results.append(r)

        if 'internal' in results:
            for r in results['internal']:
                r['level'] = 'C'
                r['level_name'] = '长期记忆'
                r['sort_key'] = 2
                all_results.append(r)

        if 'seeds' in results:
            for r in results['seeds']:
                r['level'] = 'D'
                r['level_name'] = '人格记忆痕迹'
                r['sort_key'] = 3
                all_results.append(r)

        # 计算综合得分
        for result in all_results:
            weight = (result.get('weight', 0) or
                     result.get('coupling_strength', 0) or
                     result.get('potential', 0) or
                     result.get('similarity', 0) or
                     result.get('score', 0))
            result['final_score'] = (5 - result['sort_key']) * 100 + weight * 10

        # 排序
        all_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        return all_results[:top_k]

    def semantic_retrieve(self, query: str, top_k: int = 10):
        """
        语义检索

        P1 升级：使用 TF-IDF + 余弦相似度的综合检索
        返回格式与 retrieve() 一致（dict），保持向后兼容
        """
        return self.retrieve(query, top_k, ['internal', 'working_memory', 'seeds'])

    def retrieve_by_type(self, content_type: str, top_k: int = 10) -> List[Dict]:
        """按类型检索"""
        if content_type == 'memory':
            if self.multi_criteria_retriever:
                # 使用新的多条件检索器
                criteria = RetrievalCriteria(
                    access_preference='recent'
                )
                results = self.multi_criteria_retriever.retrieve(criteria, top_k)
                return [{
                    'id': result.memory_id,
                    'content': result.content,
                    'type': 'memory',
                    'weight': result.score,
                    'source': 'internal'
                } for result in results]
            elif self.long_term_memory:
                nodes = self.long_term_memory.get_all_nodes(top_k)
                return [{'id': n['id'], 'content': n.get('content', ''),
                         'type': 'memory', 'weight': n.get('weight', 0),
                         'source': 'internal'} for n in nodes]
        elif content_type == 'seed':
            return self._retrieve_seeds('', top_k)
        elif content_type == 'external':
            # 外部资源模块已移除
            return []
        return []

    def get_similar(self, content: str, top_k: int = 5) -> List[Dict]:
        """获取相似内容"""
        return self.semantic_retrieve(content, top_k)

    def get_recent(self, top_k: int = 10) -> List[Dict]:
        """获取最近的记忆"""
        if self.multi_criteria_retriever:
            # 使用新的多条件检索器
            criteria = RetrievalCriteria(
                access_preference='recent'
            )
            results = self.multi_criteria_retriever.retrieve(criteria, top_k)
            return [{
                'id': result.memory_id,
                'content': result.content,
                'type': 'memory',
                'weight': result.score,
                'source': 'internal'
            } for result in results]
        elif self.long_term_memory:
            nodes = self.long_term_memory.get_all_nodes(top_k)
            return [{'id': n['id'], 'content': n.get('content', ''),
                     'type': 'memory', 'weight': n.get('weight', 0),
                     'source': 'internal'} for n in nodes]
        return []

    def get_high_weight(self, top_k: int = 10) -> List[Dict]:
        """获取高权重记忆"""
        if self.multi_criteria_retriever:
            # 使用新的多条件检索器，按访问频率排序（权重的替代）
            criteria = RetrievalCriteria(
                access_preference='weighted'
            )
            results = self.multi_criteria_retriever.retrieve(criteria, top_k)
            return [{
                'id': result.memory_id,
                'content': result.content,
                'type': 'memory',
                'weight': result.score,
                'source': 'internal'
            } for result in results]
        elif self.long_term_memory:
            nodes = self.long_term_memory.get_all_nodes(top_k)
            return [{'id': n['id'], 'content': n.get('content', ''),
                     'type': 'memory', 'weight': n.get('weight', 0),
                     'source': 'internal'} for n in nodes]
        return []

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """计算两段文本的相似度（TF-IDF + 余弦）"""
        words_a = self._tokenize(text_a)
        words_b = self._tokenize(text_b)

        if not words_a or not words_b:
            return 0.0

        tf_a = Counter(words_a)
        tf_b = Counter(words_b)

        all_vocab = set(words_a) | set(words_b)
        if not all_vocab:
            return 0.0

        dot_product = sum(tf_a[w] * tf_b[w] for w in all_vocab)
        norm_a = math.sqrt(sum(v ** 2 for v in tf_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in tf_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if not text:
            return []

        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        english_words = re.findall(r'[a-zA-Z]{3,}', text.lower())

        stopwords = {
            '的', '了', '是', '在', '和', '有', '不', '这', '我', '他',
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all'
        }

        return [w for w in chinese_words + english_words if w not in stopwords]
