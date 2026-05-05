"""
PersonaEvaluator - 人格提取评估器

评估指标：
- Pearson r: 与人工标注的相关性
- Cohen's κ: 一致性
- MAE: 平均绝对误差
- 相对提升: 相比关键词基线

修订历史：
  v1.0 (2026-04-10) - Week 2 新增
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from scipy.stats import pearsonr

from ..core.persona_extractor import OCEANScores


@dataclass
class EvaluationResult:
    """评估结果"""
    # Pearson 相关系数
    pearson_o: float
    pearson_c: float
    pearson_e: float
    pearson_a: float
    pearson_n: float
    pearson_mean: float
    
    # MAE
    mae_o: float
    mae_c: float
    mae_e: float
    mae_a: float
    mae_n: float
    mae_mean: float
    
    # 样本数
    n_samples: int
    
    def to_dict(self) -> Dict:
        return {
            'pearson': {
                'O': self.pearson_o,
                'C': self.pearson_c,
                'E': self.pearson_e,
                'A': self.pearson_a,
                'N': self.pearson_n,
                'mean': self.pearson_mean
            },
            'mae': {
                'O': self.mae_o,
                'C': self.mae_c,
                'E': self.mae_e,
                'A': self.mae_a,
                'N': self.mae_n,
                'mean': self.mae_mean
            },
            'n_samples': self.n_samples
        }


class PersonaEvaluator:
    """
    人格提取评估器
    
    对比方法：
    1. 关键词基线 (BL-1)
    2. LLM 零样本 (Ours)
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate(
        self,
        predictions: List[OCEANScores],
        ground_truth: List[OCEANScores],
        method_name: str = "method"
    ) -> EvaluationResult:
        """
        评估预测结果
        
        Args:
            predictions: 模型预测的人格分数
            ground_truth: 人工标注的真实分数
            method_name: 方法名称（用于记录）
            
        Returns:
            EvaluationResult 评估结果
        """
        assert len(predictions) == len(ground_truth), \
            "Predictions and ground truth must have same length"
        
        n = len(predictions)
        
        # 提取各维度分数
        pred_o = [p.openness for p in predictions]
        pred_c = [p.conscientiousness for p in predictions]
        pred_e = [p.extraversion for p in predictions]
        pred_a = [p.agreeableness for p in predictions]
        pred_n = [p.neuroticism for p in predictions]
        
        true_o = [g.openness for g in ground_truth]
        true_c = [g.conscientiousness for g in ground_truth]
        true_e = [g.extraversion for g in ground_truth]
        true_a = [g.agreeableness for g in ground_truth]
        true_n = [g.neuroticism for g in ground_truth]
        
        # 计算 Pearson r
        pearson_o = self._safe_pearsonr(pred_o, true_o)
        pearson_c = self._safe_pearsonr(pred_c, true_c)
        pearson_e = self._safe_pearsonr(pred_e, true_e)
        pearson_a = self._safe_pearsonr(pred_a, true_a)
        pearson_n = self._safe_pearsonr(pred_n, true_n)
        pearson_mean = np.mean([pearson_o, pearson_c, pearson_e, pearson_a, pearson_n])
        
        # 计算 MAE
        mae_o = np.mean(np.abs(np.array(pred_o) - np.array(true_o)))
        mae_c = np.mean(np.abs(np.array(pred_c) - np.array(true_c)))
        mae_e = np.mean(np.abs(np.array(pred_e) - np.array(true_e)))
        mae_a = np.mean(np.abs(np.array(pred_a) - np.array(true_a)))
        mae_n = np.mean(np.abs(np.array(pred_n) - np.array(true_n)))
        mae_mean = np.mean([mae_o, mae_c, mae_e, mae_a, mae_n])
        
        result = EvaluationResult(
            pearson_o=pearson_o,
            pearson_c=pearson_c,
            pearson_e=pearson_e,
            pearson_a=pearson_a,
            pearson_n=pearson_n,
            pearson_mean=pearson_mean,
            mae_o=mae_o,
            mae_c=mae_c,
            mae_e=mae_e,
            mae_a=mae_a,
            mae_n=mae_n,
            mae_mean=mae_mean,
            n_samples=n
        )
        
        self.results[method_name] = result
        return result
    
    def _safe_pearsonr(self, x: List[float], y: List[float]) -> float:
        """安全计算 Pearson r"""
        try:
            if len(x) < 2 or len(set(x)) == 1 or len(set(y)) == 1:
                return 0.0
            r, _ = pearsonr(x, y)
            return r if not np.isnan(r) else 0.0
        except:
            return 0.0
    
    def compare_methods(self, baseline_name: str = "baseline", 
                        ours_name: str = "ours") -> Dict:
        """
        对比两种方法
        
        Returns:
            包含相对提升的字典
        """
        if baseline_name not in self.results or ours_name not in self.results:
            return {"error": "Methods not found"}
        
        baseline = self.results[baseline_name]
        ours = self.results[ours_name]
        
        # 计算相对提升
        pearson_improvement = ((ours.pearson_mean - baseline.pearson_mean) 
                               / abs(baseline.pearson_mean) * 100) if baseline.pearson_mean != 0 else 0
        mae_improvement = ((baseline.mae_mean - ours.mae_mean) 
                          / baseline.mae_mean * 100) if baseline.mae_mean != 0 else 0
        
        return {
            'baseline': {
                'pearson_mean': baseline.pearson_mean,
                'mae_mean': baseline.mae_mean
            },
            'ours': {
                'pearson_mean': ours.pearson_mean,
                'mae_mean': ours.mae_mean
            },
            'improvement': {
                'pearson_relative': f"{pearson_improvement:+.1f}%",
                'mae_relative': f"{mae_improvement:+.1f}%"
            },
            'success': ours.pearson_mean > 0.70  # 成功标准
        }
    
    def print_report(self, method_name: str = None):
        """打印评估报告"""
        if method_name:
            results = {method_name: self.results.get(method_name)}
        else:
            results = self.results
        
        print("\n" + "=" * 60)
        print("PersonaMem Evaluation Report")
        print("=" * 60)
        
        for name, result in results.items():
            if result is None:
                continue
            print(f"\nMethod: {name}")
            print("-" * 40)
            print(f"Pearson r:")
            print(f"  O: {result.pearson_o:.3f}  C: {result.pearson_c:.3f}  E: {result.pearson_e:.3f}")
            print(f"  A: {result.pearson_a:.3f}  N: {result.pearson_n:.3f}")
            print(f"  Mean: {result.pearson_mean:.3f}")
            print(f"MAE:")
            print(f"  O: {result.mae_o:.3f}  C: {result.mae_c:.3f}  E: {result.mae_e:.3f}")
            print(f"  A: {result.mae_a:.3f}  N: {result.mae_n:.3f}")
            print(f"  Mean: {result.mae_mean:.3f}")
            print(f"Samples: {result.n_samples}")
        
        # 对比报告
        if len(self.results) >= 2:
            print("\n" + "-" * 40)
            print("Comparison:")
            comparison = self.compare_methods()
            if 'error' not in comparison:
                print(f"  Baseline Pearson: {comparison['baseline']['pearson_mean']:.3f}")
                print(f"  Ours Pearson: {comparison['ours']['pearson_mean']:.3f}")
                print(f"  Improvement: {comparison['improvement']['pearson_relative']}")
                print(f"  Success: {'Yes' if comparison['success'] else 'No'}")
        
        print("=" * 60)


def baseline_keyword_extraction(text: str) -> OCEANScores:
    """
    关键词基线方法（BL-1）
    
    简单的关键词频率统计
    """
    text_lower = text.lower()
    
    # 中英文关键词
    keywords = {
        'openness': ['new', 'explore', 'creative', 'imagine', 'curious', 
                     '探索', '创意', '好奇', '新颖'],
        'conscientiousness': ['plan', 'organize', 'responsible', 'detail', 
                              '计划', '组织', '负责', '仔细'],
        'extraversion': ['social', 'active', 'energetic', 'express', 'outgoing',
                         '社交', '活跃', '热情', '外向'],
        'agreeableness': ['help', 'understand', 'cooperate', 'friendly', 'empathy',
                          '帮助', '理解', '合作', '友善'],
        'neuroticism': ['worry', 'anxious', 'nervous', 'stress', 'uneasy',
                        '担心', '焦虑', '紧张', '压力']
    }
    
    scores = {}
    for trait, kws in keywords.items():
        count = sum(1 for kw in kws if kw in text_lower)
        # 基础分 0.3 + 每词 0.1，上限 0.9
        scores[trait] = min(0.3 + count * 0.1, 0.9)
    
    return OCEANScores(**scores)