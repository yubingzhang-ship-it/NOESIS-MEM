"""
混合检索器（Hybrid Retriever）
结合向量感知层（语义理解）和TF-IDF（精确匹配）
模拟人脑：感知模式识别 + 认知符号推理
"""

import sqlite3
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

from noesis_ii.vector.vector_perception import VectorPerception
from noesis_ii.processes.consolidator import Consolidator


class HybridRetriever:
    """混合检索器：向量感知 + TF-IDF认知"""
    
    def __init__(self, db_path: str = None):
        """
        初始化混合检索器
        
        Args:
            db_path: 数据库路径
        """
        if db_path is None:
            db_path = "d:/Project/NOESIS-II v1.0/noesis_ii/data/noesis.db"
        self.db_path = db_path
        
        # 初始化向量感知层
        self.vector_perception = VectorPerception(db_path)
        
        # 复用Consolidator的TF-IDF实现
        self.consolidator = Consolidator(db_path)
        
        # 检索参数
        self.vector_top_k = 50  # 向量粗筛返回50个
        self.final_top_k = 10   # 最终返回10个
        self.vector_weight = 0.6  # 向量相似度权重
        self.tfidf_weight = 0.4    # TF-IDF权重
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        seed_ids: List[int] = None,
        min_vector_similarity: float = 0.0
    ) -> List[Dict]:
        """
        混合检索：向量粗筛 + TF-IDF精排
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            seed_ids: 候选种子ID列表（可选，如果为None则搜索全部）
            min_vector_similarity: 向量相似度最小阈值
            
        Returns:
            [
                {
                    "seed_id": 123,
                    "content": "...",
                    "similarity": 0.85,
                    "vector_score": 0.92,
                    "tfidf_score": 0.78
                },
                ...
            ]
        """
        # 第一步：向量感知层粗筛（召回）
        if seed_ids is None:
            vector_results = self.vector_perception.search_all(
                query, 
                top_k=self.vector_top_k,
                min_similarity=min_vector_similarity,
                limit=1000  # 限制搜索范围，避免全量计算
            )
        else:
            vector_results = self.vector_perception.search_similar(
                query,
                seed_ids,
                top_k=self.vector_top_k,
                min_similarity=min_vector_similarity
            )
        
        if not vector_results:
            # 向量检索无结果，降级为纯TF-IDF
            return self._tfidf_fallback(query, top_k, seed_ids)
        
        # 提取候选种子ID
        candidate_ids = [seed_id for seed_id, _ in vector_results]
        vector_scores = {seed_id: score for seed_id, score in vector_results}
        
        # 第二步：TF-IDF精排（精确匹配）
        conn = sqlite3.connect(self.db_path)
        results = []
        
        try:
            cursor = conn.cursor()
            
            # 获取候选种子的内容
            placeholders = ','.join('?' * len(candidate_ids))
            cursor.execute(f"""
                SELECT id, content_summary FROM memory_traces
                WHERE id IN ({placeholders}) AND is_active = 1
            """, candidate_ids)
            
            seeds = {row[0]: row[1] for row in cursor.fetchall()}
            
            # 对每个候选种子计算TF-IDF分数
            for seed_id, content in seeds.items():
                tfidf_score = self.consolidator._cosine_similarity(
                    self.consolidator._tf(self.consolidator._tokenize(query)),
                    self.consolidator._tf(self.consolidator._tokenize(content))
                )
                
                # 混合评分
                vector_score = vector_scores[seed_id]
                hybrid_score = (
                    self.vector_weight * vector_score + 
                    self.tfidf_weight * tfidf_score
                )
                
                results.append({
                    "seed_id": seed_id,
                    "content": content,
                    "similarity": hybrid_score,
                    "vector_score": vector_score,
                    "tfidf_score": tfidf_score
                })
            
            # 按混合分数降序排序
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            return results[:top_k]
            
        finally:
            conn.close()
    
    def _tfidf_fallback(
        self, 
        query: str, 
        top_k: int, 
        seed_ids: List[int] = None
    ) -> List[Dict]:
        """
        TF-IDF降级策略（当向量检索无结果时）
        """
        conn = sqlite3.connect(self.db_path)
        results = []
        
        try:
            cursor = conn.cursor()
            
            if seed_ids is None:
                cursor.execute("SELECT id, content_summary FROM memory_traces WHERE is_active = 1")
            else:
                placeholders = ','.join('?' * len(seed_ids))
                cursor.execute(f"""
                    SELECT id, content_summary FROM memory_traces
                    WHERE id IN ({placeholders}) AND is_active = 1
                """, seed_ids)
            
            seeds = [(row[0], row[1]) for row in cursor.fetchall()]
            
            # 计算TF-IDF相似度
            for seed_id, content in seeds:
                tfidf_score = self.consolidator._cosine_similarity(
                    self.consolidator._tf(self.consolidator._tokenize(query)),
                    self.consolidator._tf(self.consolidator._tokenize(content))
                )
                results.append({
                    "seed_id": seed_id,
                    "content": content,
                    "similarity": tfidf_score,
                    "vector_score": 0.0,
                    "tfidf_score": tfidf_score
                })
            
            # 按TF-IDF分数降序排序
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            return results[:top_k]
            
        finally:
            conn.close()
    
    def compare_search(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict:
        """
        对比测试：纯TF-IDF vs 混合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            {
                "query": "...",
                "tfidf_results": [...],
                "hybrid_results": [...],
                "overlap": 3,
                "tfidf_only": 2,
                "hybrid_only": 5
            }
        """
        # 纯TF-IDF检索
        tfidf_results = self._tfidf_fallback(query, top_k)
        tfidf_ids = {r["seed_id"] for r in tfidf_results}
        
        # 混合检索
        hybrid_results = self.hybrid_search(query, top_k)
        hybrid_ids = {r["seed_id"] for r in hybrid_results}
        
        # 计算重叠
        overlap = tfidf_ids & hybrid_ids
        tfidf_only = tfidf_ids - hybrid_ids
        hybrid_only = hybrid_ids - tfidf_ids
        
        return {
            "query": query,
            "tfidf_results": tfidf_results,
            "hybrid_results": hybrid_results,
            "overlap_count": len(overlap),
            "tfidf_only_count": len(tfidf_only),
            "hybrid_only_count": len(hybrid_only)
        }
    
    def get_stats(self) -> Dict:
        """获取检索器统计信息"""
        vector_stats = self.vector_perception.get_stats()
        
        return {
            "vector_perception": vector_stats,
            "search_params": {
                "vector_top_k": self.vector_top_k,
                "final_top_k": self.final_top_k,
                "vector_weight": self.vector_weight,
                "tfidf_weight": self.tfidf_weight
            }
        }


# 测试代码
if __name__ == "__main__":
    print("🔮 混合检索器测试")
    
    retriever = HybridRetriever()
    
    # 测试查询
    queries = [
        "人格模式的形成",
        "如何消除执念？",
        "人工智能会产生意识吗？",
        "打坐时大脑在干什么？"
    ]
    
    print("\n混合检索测试:")
    for query in queries:
        print(f"\n查询: {query}")
        results = retriever.hybrid_search(query, top_k=3)
        
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['seed_id']}] 相似度: {r['similarity']:.4f} "
                  f"(向量: {r['vector_score']:.4f}, TF-IDF: {r['tfidf_score']:.4f})")
            print(f"     内容: {r['content'][:80]}...")
    
    # 对比测试
    print("\n\n对比测试（TF-IDF vs 混合）:")
    query = "如何消除执念？"
    comparison = retriever.compare_search(query, top_k=5)
    
    print(f"\n查询: {query}")
    print(f"重叠: {comparison['overlap_count']}, "
          f"仅TF-IDF: {comparison['tfidf_only_count']}, "
          f"仅混合: {comparison['hybrid_only_count']}")
    
    print("\n混合检索结果:")
    for i, r in enumerate(comparison['hybrid_results'], 1):
        print(f"  {i}. [{r['seed_id']}] {r['similarity']:.4f}: {r['content'][:60]}...")
    
    print("\nTF-IDF结果:")
    for i, r in enumerate(comparison['tfidf_results'], 1):
        print(f"  {i}. [{r['seed_id']}] {r['similarity']:.4f}: {r['content'][:60]}...")
    
    print("\n✓ 混合检索器测试完成")
