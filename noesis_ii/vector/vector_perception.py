"""
向量感知层（Vector Perception Layer）
负责语义向量的生成、存储和检索
作为 PersonaMem 的"感知系统"，模拟人脑的模式识别能力
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import sqlite3
import json
from pathlib import Path

# 配置HuggingFace国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 本地模型路径（优先使用本地）
LOCAL_MODEL_PATH = "d:/Project/NOESIS-II v1.0/models/Liudef/paraphrase-multilingual-MiniLM-L12-v2"


class VectorPerception:
    """向量感知层：语义理解与联想记忆"""

    def __init__(self, db_path: str = None, model_name: str = None):
        """
        初始化向量感知层

        Args:
            db_path: 数据库路径
            model_name: embedding模型名称，默认使用轻量级中文模型
        """
        # 加载embedding模型
        if model_name is None:
            # 优先使用本地模型
            if os.path.exists(LOCAL_MODEL_PATH):
                model_name = LOCAL_MODEL_PATH
                print(f"加载本地embedding模型: {model_name}")
            else:
                # 使用轻量级多语言模型（支持中文+英文）
                model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                print(f"加载远程embedding模型: {model_name}")

        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"OK 向量维度: {self.embedding_dim}")
        
        # 数据库路径
        if db_path is None:
            db_path = "d:/Project/NOESIS-II v1.0/noesis_ii/data/noesis.db"
        self.db_path = db_path
        
        # 向量索引缓存（避免重复计算）
        self.embedding_cache: Dict[int, np.ndarray] = {}
    
    def encode(self, text: str) -> np.ndarray:
        """
        对文本进行向量编码
        
        Args:
            text: 输入文本
            
        Returns:
            embedding向量（numpy数组）
        """
        if not text or not text.strip():
            # 空文本返回零向量
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量编码
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding矩阵（n_samples x embedding_dim）
        """
        if not texts:
            return np.zeros((0, self.embedding_dim))
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数（0-1，越大越相似）
        """
        if vec1.shape != vec2.shape:
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def get_seed_embedding(self, seed_id: int, conn: sqlite3.Connection = None) -> Optional[np.ndarray]:
        """
        获取种子的embedding向量
        
        Args:
            seed_id: 种子ID
            conn: 数据库连接（可选，用于复用连接）
            
        Returns:
            embedding向量，如果不存在返回None
        """
        # 检查缓存
        if seed_id in self.embedding_cache:
            return self.embedding_cache[seed_id]
        
        # 从数据库读取
        close_conn = False
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            close_conn = True
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT content_summary FROM memory_traces WHERE id = ?
            """, (seed_id,))
            row = cursor.fetchone()
            
            if row:
                embedding = self.encode(row[0])
                # 缓存
                self.embedding_cache[seed_id] = embedding
                return embedding
            
            return None
        finally:
            if close_conn:
                conn.close()
    
    def search_similar(
        self, 
        query: str, 
        seed_ids: List[int],
        top_k: int = 50,
        min_similarity: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        在指定种子列表中搜索语义相似的
        
        Args:
            query: 查询文本
            seed_ids: 候选种子ID列表
            top_k: 返回前k个结果
            min_similarity: 最小相似度阈值
            
        Returns:
            [(seed_id, similarity), ...] 按相似度降序排列
        """
        if not seed_ids:
            return []
        
        # 编码查询
        query_embedding = self.encode(query)
        
        # 获取所有候选种子的embedding
        conn = sqlite3.connect(self.db_path)
        results = []
        
        try:
            for seed_id in seed_ids:
                seed_embedding = self.get_seed_embedding(seed_id, conn)
                if seed_embedding is not None:
                    similarity = self.cosine_similarity(query_embedding, seed_embedding)
                    if similarity >= min_similarity:
                        results.append((seed_id, similarity))
        finally:
            conn.close()
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def search_all(
        self,
        query: str,
        top_k: int = 50,
        min_similarity: float = 0.0,
        limit: int = None
    ) -> List[Tuple[int, float]]:
        """
        在所有人格记忆痕迹中搜索
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            min_similarity: 最小相似度阈值
            limit: 限制搜索的种子数量（避免全量计算）
            
        Returns:
            [(seed_id, similarity), ...]
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            cursor = conn.cursor()
            # 获取痕迹ID列表（如果指定limit，随机采样）
            if limit:
                cursor.execute("""
                    SELECT id FROM memory_traces WHERE is_active=1 ORDER BY RANDOM() LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT id FROM memory_traces WHERE is_active=1
                """)
            
            seed_ids = [row[0] for row in cursor.fetchall()]
            
            return self.search_similar(query, seed_ids, top_k, min_similarity)
        finally:
            conn.close()
    
    def clear_cache(self):
        """清空embedding缓存"""
        self.embedding_cache.clear()
    
    def get_stats(self) -> Dict:
        """获取感知层统计信息"""
        return {
            "embedding_dim": self.embedding_dim,
            "cache_size": len(self.embedding_cache),
            "model_name": self.model._first_module().auto_model.config._name_or_path
        }


# 测试代码
if __name__ == "__main__":
    print("向量感知层测试")
    
    vp = VectorPerception()
    
    # 测试编码
    print("\n测试文本编码:")
    texts = [
        "人格模式的形成机制",
        "长期记忆的存储结构",
        "人工智能和意识的区别",
        "认知科学的核心议题"
    ]
    
    embeddings = vp.encode_batch(texts)
    print(f"编码 {len(texts)} 条文本")
    print(f"向量维度: {embeddings.shape}")
    
    # 测试相似度
    print("\n测试语义相似度:")
    query = "如何克服认知偏见？"
    similar_texts = [
        "打破思维定式的方法",
        "心理防御机制的识别",
        "人格模式的内在偏见",
        "今天天气不错"
    ]
    
    query_emb = vp.encode(query)
    for text in similar_texts:
        text_emb = vp.encode(text)
        sim = vp.cosine_similarity(query_emb, text_emb)
        print(f"  '{query}' vs '{text}': {sim:.4f}")
    
    print("\nOK 向量感知层测试完成")
