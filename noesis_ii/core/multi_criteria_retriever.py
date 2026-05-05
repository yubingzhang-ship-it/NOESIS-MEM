"""
MultiCriteriaRetriever - 多条件记忆检索器

替代原 IABEngine，剥离"因果"术语，明确为多字段加权检索。

修订历史：
  v2.0 (2026-04-10) - 路线A重构
"""

import sqlite3
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalCriteria:
    """检索条件"""
    semantic_query: Optional[str] = None
    temporal_context: Optional[float] = None
    min_association: float = 0.0
    access_preference: str = 'recent'


@dataclass
class RetrievalResult:
    """检索结果"""
    memory_id: int
    content: str
    score: float
    criteria_matched: List[str]


class MultiCriteriaRetriever:
    """多条件记忆检索器 - 非因果推理，是多字段加权检索"""
    
    def __init__(self, db_path: str, embedding_model=None):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.rrf_k = 60
        self.criteria_weights = {
            'semantic': 0.35,
            'temporal': 0.25,
            'access': 0.20,
            'association': 0.20
        }
    
    def retrieve(self, criteria: RetrievalCriteria, top_k: int = 10) -> List[RetrievalResult]:
        """多条件检索记忆"""
        rankings = {}
        
        if criteria.semantic_query:
            rankings['semantic'] = self._semantic_retrieve(criteria.semantic_query, top_k * 2)
        
        if criteria.temporal_context:
            rankings['temporal'] = self._temporal_retrieve(criteria.temporal_context, top_k * 2)
        
        rankings['access'] = self._access_retrieve(criteria.access_preference, top_k * 2)
        
        if criteria.min_association > 0:
            rankings['association'] = self._association_retrieve(criteria.min_association, top_k * 2)
        
        fused_scores = self._rrf_fuse(rankings)
        
        results = []
        for memory_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            content = self._get_memory_content(memory_id)
            matched = [dim for dim, ids in rankings.items() if memory_id in [r[0] for r in ids]]
            results.append(RetrievalResult(memory_id, content, score, matched))
        
        return results
    
    def _semantic_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """语义检索（支持中英文关键词）"""
        import re
        # 支持中文和英文分词
        query_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', query.lower()))
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, content FROM ltm_nodes")
            results = []
            for row in cursor:
                memory_id, content = row
                content_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', content.lower()))
                overlap = len(query_words & content_words)
                if overlap > 0:
                    score = overlap / len(query_words)
                    results.append((memory_id, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def _temporal_retrieve(self, timestamp: float, top_k: int) -> List[Tuple[int, float]]:
        """时间邻近检索"""
        import datetime
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, created_at FROM ltm_nodes ORDER BY ABS(julianday(created_at) - julianday(?)) ASC LIMIT ?",
                (timestamp, top_k)
            )
            results = []
            for row in cursor:
                memory_id, ts = row
                try:
                    # 计算时间差分数
                    node_time = datetime.datetime.fromisoformat(ts) if ts else datetime.datetime.now()
                    query_time = datetime.datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else datetime.datetime.now()
                    days_diff = abs((node_time - query_time).days)
                    # 30天内线性衰减
                    score = max(0.1, 1.0 - days_diff / 30.0)
                except (ValueError, TypeError):
                    score = 0.5
                results.append((memory_id, score))
            return results
    
    def _access_retrieve(self, preference: str, top_k: int) -> List[Tuple[int, float]]:
        """访问频率检索"""
        with sqlite3.connect(self.db_path) as conn:
            # ltm_nodes 使用 last_accessed 字段
            if preference == 'recent':
                cursor = conn.execute(
                    "SELECT id FROM ltm_nodes ORDER BY last_accessed DESC LIMIT ?",
                    (top_k,)
                )
            else:
                # 按权重排序作为访问频率的替代
                cursor = conn.execute(
                    "SELECT id FROM ltm_nodes ORDER BY weight DESC LIMIT ?",
                    (top_k,)
                )
            return [(row[0], 1.0 / (i + 1)) for i, row in enumerate(cursor)]
    
    def _association_retrieve(self, min_strength: float, top_k: int) -> List[Tuple[int, float]]:
        """关联强度检索"""
        with sqlite3.connect(self.db_path) as conn:
            # 使用 ltm_links 表作为关联表
            cursor = conn.execute(
                "SELECT target_node_id, strength FROM ltm_links WHERE strength >= ? ORDER BY strength DESC LIMIT ?",
                (min_strength, top_k)
            )
            return [(row[0], row[1]) for row in cursor]
    
    def _rrf_fuse(self, rankings: Dict[str, List[Tuple[int, float]]]) -> Dict[int, float]:
        """RRF融合多维度排名"""
        scores = {}
        for dim, ranked_list in rankings.items():
            weight = self.criteria_weights.get(dim, 0.25)
            for rank, (memory_id, _) in enumerate(ranked_list):
                rrf_score = weight * (1.0 / (self.rrf_k + rank + 1))
                scores[memory_id] = scores.get(memory_id, 0) + rrf_score
        return scores
    
    def _get_memory_content(self, memory_id: int) -> str:
        """获取记忆内容"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT content FROM ltm_nodes WHERE id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else ""
