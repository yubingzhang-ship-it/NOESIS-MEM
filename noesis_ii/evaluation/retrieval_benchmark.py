"""
Retrieval Benchmark (E3 Experiment)

Week 4: 记忆检索性能对比实验

对比不同检索策略的效果：
- BL-1: 纯向量检索 (MiniLM cosine)
- BL-2: BM25 关键词检索
- BL-3: 混合 RRF 融合
- Ours: MultiCriteriaRetriever (多条件检索)
"""

import json
import os
import sys
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.multi_criteria_retriever import MultiCriteriaRetriever, RetrievalCriteria


class SimpleVectorRetriever:
    """纯向量检索基线 (BL-1)"""
    
    def __init__(self):
        self.documents = []
        self.vectors = []
    
    def add_document(self, doc_id: str, content: str, vector: np.ndarray):
        self.documents.append({'id': doc_id, 'content': content})
        self.vectors.append(vector)
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        if not self.vectors:
            return []
        
        # 计算余弦相似度
        vectors = np.array(self.vectors)
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        doc_norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(doc_norms, query_norm)
        
        # 获取 top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'id': self.documents[idx]['id'],
                'content': self.documents[idx]['content'],
                'score': float(similarities[idx])
            })
        
        return results


class BM25Retriever:
    """BM25 关键词检索基线 (BL-2)"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.documents = []
        self.k1 = k1
        self.b = b
        self.avg_doc_len = 0
        self.doc_freqs = {}
        self.N = 0
    
    def add_document(self, doc_id: str, content: str):
        words = content.lower().split()
        self.documents.append({
            'id': doc_id,
            'content': content,
            'words': words,
            'len': len(words)
        })
        
        # 更新文档频率
        unique_words = set(words)
        for word in unique_words:
            self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1
        
        self.N = len(self.documents)
        self.avg_doc_len = sum(d['len'] for d in self.documents) / self.N
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_words = query.lower().split()
        scores = []
        
        for doc in self.documents:
            score = 0.0
            doc_len = doc['len']
            
            for word in query_words:
                if word not in self.doc_freqs:
                    continue
                
                # 词频
                tf = doc['words'].count(word)
                if tf == 0:
                    continue
                
                # IDF
                df = self.doc_freqs[word]
                idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
                
                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * numerator / denominator
            
            scores.append((doc, score))
        
        # 排序返回 top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in scores[:top_k]:
            results.append({
                'id': doc['id'],
                'content': doc['content'],
                'score': float(score)
            })
        
        return results


class HybridRRFRetriever:
    """混合 RRF 融合检索 (BL-3)"""
    
    def __init__(self, k: int = 60):
        self.vector_retriever = SimpleVectorRetriever()
        self.bm25_retriever = BM25Retriever()
        self.k = k  # RRF 常数
    
    def add_document(self, doc_id: str, content: str, vector: np.ndarray):
        self.vector_retriever.add_document(doc_id, content, vector)
        self.bm25_retriever.add_document(doc_id, content)
    
    def search(self, query: str, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        # 获取两种检索的结果
        vector_results = self.vector_retriever.search(query_vector, top_k=top_k * 2)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
        
        # RRF 融合
        scores = {}
        
        # 向量检索排名
        for rank, result in enumerate(vector_results):
            doc_id = result['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.k + rank + 1)
        
        # BM25 检索排名
        for rank, result in enumerate(bm25_results):
            doc_id = result['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.k + rank + 1)
        
        # 排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 构建结果
        results = []
        doc_map = {d['id']: d for d in self.vector_retriever.documents}
        
        for doc_id, score in sorted_scores[:top_k]:
            if doc_id in doc_map:
                results.append({
                    'id': doc_id,
                    'content': doc_map[doc_id]['content'],
                    'score': float(score)
                })
        
        return results


class RetrievalBenchmark:
    """检索性能基准测试"""
    
    def __init__(self):
        self.vector_bl = SimpleVectorRetriever()
        self.bm25_bl = BM25Retriever()
        self.hybrid_bl = HybridRRFRetriever()
        self.multi_criteria = None  # 需要外部初始化
    
    def build_test_collection(self, size: int = 1000) -> Tuple[List[Dict], List[Dict]]:
        """
        构建测试集合
        
        Returns:
            (documents, queries_with_relevance)
        """
        print(f"[Building] Test collection with {size} documents...")
        
        # 生成合成文档
        topics = [
            "technology", "science", "art", "music", "sports",
            "travel", "food", "history", "philosophy", "psychology"
        ]
        
        documents = []
        for i in range(size):
            topic = random.choice(topics)
            doc = {
                'id': f'doc_{i:05d}',
                'content': f"This is a document about {topic}. " * 5 + f"Unique content {i}.",
                'topic': topic,
                'timestamp': datetime.now().isoformat()
            }
            documents.append(doc)
        
        # 生成测试查询（带相关性标注）
        queries = []
        for _ in range(20):
            topic = random.choice(topics)
            query = {
                'query': f"Tell me about {topic}",
                'topic': topic,
                'relevant_docs': [d['id'] for d in documents if d['topic'] == topic]
            }
            queries.append(query)
        
        return documents, queries
    
    def index_documents(self, documents: List[Dict], vectors: List[np.ndarray]):
        """索引文档到所有检索器"""
        print(f"[Indexing] {len(documents)} documents...")
        
        for doc, vec in zip(documents, vectors):
            self.vector_bl.add_document(doc['id'], doc['content'], vec)
            self.bm25_bl.add_document(doc['id'], doc['content'])
            self.hybrid_bl.add_document(doc['id'], doc['content'], vec)
    
    def evaluate_retriever(
        self,
        retriever_name: str,
        retriever: Any,
        queries: List[Dict],
        query_vectors: List[np.ndarray],
        top_k: int = 10
    ) -> Dict:
        """评估单个检索器"""
        
        print(f"[Evaluating] {retriever_name}...")
        
        recalls = []
        ndcgs = []
        latencies = []
        
        for query, query_vec in zip(queries, query_vectors):
            relevant = set(query['relevant_docs'])
            
            # 计时
            start = time.time()
            
            # 根据检索器类型调用不同接口
            if retriever_name == "Vector (BL-1)":
                results = retriever.search(query_vec, top_k=top_k)
            elif retriever_name == "BM25 (BL-2)":
                results = retriever.search(query['query'], top_k=top_k)
            elif retriever_name == "Hybrid RRF (BL-3)":
                results = retriever.search(query['query'], query_vec, top_k=top_k)
            else:
                results = []
            
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
            
            # 计算 Recall@k
            retrieved = set(r['id'] for r in results)
            if relevant:
                recall = len(retrieved & relevant) / len(relevant)
            else:
                recall = 0.0
            recalls.append(recall)
            
            # 计算 NDCG@k
            dcg = 0.0
            for i, result in enumerate(results):
                rel = 1.0 if result['id'] in relevant else 0.0
                dcg += rel / np.log2(i + 2)
            
            # Ideal DCG
            ideal_rels = [1.0] * min(len(relevant), top_k)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        
        return {
            'name': retriever_name,
            'recall_at_k': float(np.mean(recalls)),
            'ndcg_at_k': float(np.mean(ndcgs)),
            'avg_latency_ms': float(np.mean(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95))
        }
    
    def run_comparison(
        self,
        collection_size: int = 1000,
        top_k: int = 10
    ) -> Dict:
        """运行完整对比实验"""
        
        print("="*60)
        print("PersonaMem Week 4 - Retrieval Benchmark (E3)")
        print("="*60)
        print(f"Collection size: {collection_size}")
        print(f"Top-k: {top_k}")
        
        # 构建测试集合
        documents, queries = self.build_test_collection(collection_size)
        
        # 生成合成向量（模拟 MiniLM 384维）
        print("[Generating] Synthetic vectors (384-dim)...")
        doc_vectors = [np.random.randn(384).astype(np.float32) for _ in documents]
        query_vectors = [np.random.randn(384).astype(np.float32) for _ in queries]
        
        # 索引
        self.index_documents(documents, doc_vectors)
        
        # 评估所有基线
        results = {}
        
        results['vector_bl'] = self.evaluate_retriever(
            "Vector (BL-1)", self.vector_bl, queries, query_vectors, top_k
        )
        
        results['bm25_bl'] = self.evaluate_retriever(
            "BM25 (BL-2)", self.bm25_bl, queries, query_vectors, top_k
        )
        
        results['hybrid_bl'] = self.evaluate_retriever(
            "Hybrid RRF (BL-3)", self.hybrid_bl, queries, query_vectors, top_k
        )
        
        return {
            'collection_size': collection_size,
            'top_k': top_k,
            'num_queries': len(queries),
            'results': results
        }


def run_week4_benchmark(output_dir: str = "evaluation_results"):
    """运行 Week 4 完整基准测试"""
    
    benchmark = RetrievalBenchmark()
    
    # 运行不同规模的测试
    all_results = {}
    
    for size in [1000, 10000]:
        print(f"\n{'='*60}")
        print(f"Scale Test: {size} documents")
        print(f"{'='*60}")
        
        results = benchmark.run_comparison(collection_size=size, top_k=10)
        all_results[f'scale_{size}'] = results
        
        # 打印结果
        print(f"\n[Results for {size} documents]")
        print("-" * 60)
        for name, metrics in results['results'].items():
            print(f"\n{metrics['name']}:")
            print(f"  Recall@10:    {metrics['recall_at_k']:.4f}")
            print(f"  NDCG@10:      {metrics['ndcg_at_k']:.4f}")
            print(f"  Avg Latency:  {metrics['avg_latency_ms']:.2f}ms")
            print(f"  P95 Latency:  {metrics['p95_latency_ms']:.2f}ms")
    
    # 汇总报告
    print(f"\n{'='*60}")
    print("Summary Report")
    print(f"{'='*60}")
    
    # 1K 规模对比
    results_1k = all_results['scale_1000']['results']
    print("\n[Scale: 1K documents]")
    print(f"{'Method':<20} {'Recall@10':>10} {'NDCG@10':>10} {'Latency':>12}")
    print("-" * 60)
    for name, metrics in results_1k.items():
        short_name = metrics['name'].split()[0]
        print(f"{short_name:<20} {metrics['recall_at_k']:>10.4f} {metrics['ndcg_at_k']:>10.4f} {metrics['avg_latency_ms']:>10.2f}ms")
    
    # 目标检查
    vector_recall = results_1k['vector_bl']['recall_at_k']
    print(f"\nTarget Recall@10 > 0.80: {'PASS' if vector_recall > 0.80 else 'NEED IMPROVEMENT'}")
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"week4_retrieval_{timestamp}.json")
        
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return bool(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy(all_results), f, ensure_ascii=False, indent=2)
        
        print(f"\n[Saved] Results saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
    run_week4_benchmark()
