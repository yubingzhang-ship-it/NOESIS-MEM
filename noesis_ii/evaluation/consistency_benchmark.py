"""
Consistency Benchmark (E2 Experiment)

Week 3: 跨Session人格一致性测试

测试 AI 在多次交互中保持人格一致性的能力。
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.persona_extractor import PersonaExtractor, OCEANScores
from core.persona_constrained_generator import (
    PersonaConstrainedGenerator, GenerationConfig, ValueConflictDetector
)
from llm.longcat_client import LongCatClient


# 20道价值观测试题（覆盖 OCEAN 五维）
VALUE_TEST_QUESTIONS = [
    # Openness (开放性)
    {
        "id": "O1",
        "dimension": "openness",
        "question": "你如何看待尝试全新的工作方法？",
        "question_en": "How do you feel about trying completely new work methods?"
    },
    {
        "id": "O2", 
        "dimension": "openness",
        "question": "抽象艺术和实验音乐对你有吸引力吗？",
        "question_en": "Are you drawn to abstract art and experimental music?"
    },
    {
        "id": "O3",
        "dimension": "openness", 
        "question": "你更喜欢探索未知领域还是坚持熟悉的路径？",
        "question_en": "Do you prefer exploring unknown territories or sticking to familiar paths?"
    },
    {
        "id": "O4",
        "dimension": "openness",
        "question": "面对复杂问题，你会寻找创新的解决方案吗？",
        "question_en": "When facing complex problems, do you seek innovative solutions?"
    },
    # Conscientiousness (尽责性)
    {
        "id": "C1",
        "dimension": "conscientiousness",
        "question": "你通常会提前计划还是随机应变？",
        "question_en": "Do you usually plan ahead or improvise as you go?"
    },
    {
        "id": "C2",
        "dimension": "conscientiousness",
        "question": "完成任务的截止日期对你有多重要？",
        "question_en": "How important are deadlines for completing tasks?"
    },
    {
        "id": "C3",
        "dimension": "conscientiousness",
        "question": "你会如何组织一个复杂的项目？",
        "question_en": "How would you organize a complex project?"
    },
    {
        "id": "C4",
        "dimension": "conscientiousness",
        "question": "细节和完美主义在你的工作中扮演什么角色？",
        "question_en": "What role do details and perfectionism play in your work?"
    },
    # Extraversion (外向性)
    {
        "id": "E1",
        "dimension": "extraversion",
        "question": "大型社交活动让你感到 energized 还是 drained？",
        "question_en": "Do large social events energize or drain you?"
    },
    {
        "id": "E2",
        "dimension": "extraversion",
        "question": "在团队中，你更倾向于领导还是跟随？",
        "question_en": "In a team, do you prefer to lead or follow?"
    },
    {
        "id": "E3",
        "dimension": "extraversion",
        "question": "你如何看待公开演讲和表达观点？",
        "question_en": "How do you feel about public speaking and expressing opinions?"
    },
    {
        "id": "E4",
        "dimension": "extraversion",
        "question": "独处和社交，哪个更让你感到舒适？",
        "question_en": "Which makes you more comfortable: solitude or socializing?"
    },
    # Agreeableness (宜人性)
    {
        "id": "A1",
        "dimension": "agreeableness",
        "question": "当朋友遇到困难时，你会如何回应？",
        "question_en": "How do you respond when a friend is in difficulty?"
    },
    {
        "id": "A2",
        "dimension": "agreeableness",
        "question": "你更倾向于合作竞争还是合作共赢？",
        "question_en": "Do you prefer competitive or cooperative approaches?"
    },
    {
        "id": "A3",
        "dimension": "agreeableness",
        "question": "面对冲突，你会选择妥协还是坚持立场？",
        "question_en": "In conflicts, do you choose compromise or stand your ground?"
    },
    {
        "id": "A4",
        "dimension": "agreeableness",
        "question": "他人的感受在你的决策中有多重要？",
        "question_en": "How important are others' feelings in your decisions?"
    },
    # Neuroticism (神经质)
    {
        "id": "N1",
        "dimension": "neuroticism",
        "question": "面对压力，你通常会感到焦虑还是冷静？",
        "question_en": "Under pressure, do you typically feel anxious or calm?"
    },
    {
        "id": "N2",
        "dimension": "neuroticism",
        "question": "你对未来的不确定性有多担心？",
        "question_en": "How much do you worry about future uncertainty?"
    },
    {
        "id": "N3",
        "dimension": "neuroticism",
        "question": "负面情绪会持续影响你多久？",
        "question_en": "How long do negative emotions typically affect you?"
    },
    {
        "id": "N4",
        "dimension": "neuroticism",
        "question": "你如何看待批评和负面反馈？",
        "question_en": "How do you view criticism and negative feedback?"
    }
]


class ConsistencyBenchmark:
    """人格一致性基准测试"""
    
    def __init__(self, llm_client: LongCatClient, extractor: PersonaExtractor):
        self.llm_client = llm_client
        self.extractor = extractor
        self.generator = PersonaConstrainedGenerator(llm_client, extractor)
        self.conflict_detector = ValueConflictDetector(llm_client)
    
    def run_session_test(
        self,
        target_persona: Dict[str, float],
        questions: List[Dict] = None,
        use_constraint: bool = True
    ) -> Dict:
        """
        运行单 Session 测试
        
        Args:
            target_persona: 目标人格
            questions: 测试问题列表
            use_constraint: 是否使用人格约束
            
        Returns:
            测试结果字典
        """
        questions = questions or VALUE_TEST_QUESTIONS
        
        responses = []
        extracted_personas = []
        kl_divergences = []
        regeneration_counts = []
        
        print(f"\n[Session Test] Running {len(questions)} questions...")
        print(f"  Target persona: O={target_persona['openness']:.2f}, "
              f"C={target_persona['conscientiousness']:.2f}, "
              f"E={target_persona['extraversion']:.2f}, "
              f"A={target_persona['agreeableness']:.2f}, "
              f"N={target_persona['neuroticism']:.2f}")
        print(f"  Constraint: {'ON' if use_constraint else 'OFF'}")
        
        for i, q in enumerate(questions):
            print(f"  Q{i+1}/{len(questions)}: {q['id']} ({q['dimension']})", end=" ")
            
            if use_constraint:
                # 使用人格约束生成
                config = GenerationConfig(
                    temperature=0.7,
                    kl_weight=0.3,
                    kl_threshold=0.15,
                    max_regeneration_attempts=2
                )
                result = self.generator.generate(
                    prompt=q['question'],
                    target_persona=target_persona,
                    config=config
                )
                response = result.text
                kl_div = result.kl_divergence
                regen_count = result.regeneration_count
            else:
                # 无约束生成（基线）
                response = self.llm_client.generate(
                    f"User: {q['question']}\nAssistant:",
                    temperature=0.7
                )
                # 提取人格并计算 KL
                extracted = self.extractor.extract(response)
                kl_div = self._compute_kl(target_persona, extracted.to_dict())
                regen_count = 0
            
            # 提取响应的人格
            extracted = self.extractor.extract(response)
            
            responses.append({
                'question_id': q['id'],
                'dimension': q['dimension'],
                'question': q['question'],
                'response': response[:200],  # 截断存储
                'extracted_persona': extracted.to_dict()
            })
            extracted_personas.append(extracted.to_dict())
            kl_divergences.append(kl_div)
            regeneration_counts.append(regen_count)
            
            status = "✓" if kl_div < 0.15 else "!"
            print(f"[KL={kl_div:.3f}] {status}")
        
        # 计算统计指标
        metrics = self._compute_consistency_metrics(extracted_personas, target_persona)
        metrics['avg_kl_divergence'] = float(np.mean(kl_divergences))
        metrics['max_kl_divergence'] = float(np.max(kl_divergences))
        metrics['total_regenerations'] = sum(regeneration_counts)
        metrics['regeneration_rate'] = sum(regeneration_counts) / len(questions)
        
        return {
            'target_persona': target_persona,
            'responses': responses,
            'metrics': metrics,
            'use_constraint': use_constraint
        }
    
    def run_cross_session_test(
        self,
        target_persona: Dict[str, float],
        num_sessions: int = 3,
        days_between: int = 1
    ) -> Dict:
        """
        运行跨 Session 一致性测试
        
        模拟多天多次对话，检验人格稳定性
        
        Args:
            target_persona: 目标人格
            num_sessions: Session 数量
            days_between: Session 间隔天数
            
        Returns:
            跨 Session 测试结果
        """
        print(f"\n{'='*60}")
        print("Cross-Session Consistency Test")
        print(f"{'='*60}")
        print(f"Sessions: {num_sessions}, Interval: {days_between} days")
        
        session_results = []
        
        for session_idx in range(num_sessions):
            print(f"\n[Session {session_idx + 1}/{num_sessions}]")
            
            # 每个 Session 使用不同的问题子集
            start_idx = (session_idx * 5) % len(VALUE_TEST_QUESTIONS)
            session_questions = VALUE_TEST_QUESTIONS[start_idx:start_idx + 7]
            
            result = self.run_session_test(
                target_persona=target_persona,
                questions=session_questions,
                use_constraint=True
            )
            result['session_id'] = session_idx + 1
            result['simulated_day'] = (session_idx + 1) * days_between
            session_results.append(result)
        
        # 计算跨 Session 稳定性
        stability_metrics = self._compute_cross_session_stability(session_results)
        
        return {
            'target_persona': target_persona,
            'num_sessions': num_sessions,
            'days_between': days_between,
            'session_results': session_results,
            'stability_metrics': stability_metrics
        }
    
    def _compute_consistency_metrics(
        self,
        extracted_personas: List[Dict],
        target_persona: Dict[str, float]
    ) -> Dict:
        """计算一致性指标"""
        
        dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        
        metrics = {}
        
        for dim in dims:
            values = [p.get(dim, 0.5) for p in extracted_personas]
            target = target_persona.get(dim, 0.5)
            
            # 标准差（稳定性）
            metrics[f'{dim}_std'] = float(np.std(values))
            
            # 与目标的平均偏差
            metrics[f'{dim}_bias'] = float(np.mean([abs(v - target) for v in values]))
            
            # 与目标的余弦相似度
            metrics[f'{dim}_cosine'] = self._cosine_similarity_single(values, [target] * len(values))
        
        # 整体指标
        all_values = []
        all_targets = []
        for dim in dims:
            all_values.extend([p.get(dim, 0.5) for p in extracted_personas])
            all_targets.extend([target_persona.get(dim, 0.5)] * len(extracted_personas))
        
        metrics['overall_std'] = float(np.std(all_values))
        metrics['overall_cosine'] = self._cosine_similarity_single(all_values, all_targets)
        metrics['conflict_rate'] = sum(1 for p in extracted_personas 
                                       if self._compute_kl(target_persona, p) > 0.15) / len(extracted_personas)
        
        return metrics
    
    def _compute_cross_session_stability(self, session_results: List[Dict]) -> Dict:
        """计算跨 Session 稳定性"""
        
        dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        
        # 提取每个 Session 的平均人格
        session_personas = []
        for sr in session_results:
            responses = sr['responses']
            avg_persona = {}
            for dim in dims:
                values = [r['extracted_persona'].get(dim, 0.5) for r in responses]
                avg_persona[dim] = float(np.mean(values))
            session_personas.append(avg_persona)
        
        # 计算 Session 间的变异
        stability = {}
        for dim in dims:
            values = [p[dim] for p in session_personas]
            stability[f'{dim}_session_std'] = float(np.std(values))
            stability[f'{dim}_range'] = float(max(values) - min(values))
        
        # 整体稳定性
        all_values = []
        for p in session_personas:
            all_values.extend([p[d] for d in dims])
        stability['overall_session_std'] = float(np.std(all_values))
        
        # 稳定性评分 (越低越稳定，< 0.1 为优秀)
        stability['stability_score'] = 1.0 - min(1.0, stability['overall_session_std'] * 5)
        
        return stability
    
    def _compute_kl(self, target: Dict, actual: Dict) -> float:
        """计算 KL 散度"""
        import math
        kl = 0.0
        epsilon = 1e-10
        
        for dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            p = target.get(dim, 0.5)
            q = actual.get(dim, 0.5)
            p = max(epsilon, min(1 - epsilon, p))
            q = max(epsilon, min(1 - epsilon, q))
            
            kl_pq = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
            kl_qp = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
            kl += (kl_pq + kl_qp) / 2
        
        return kl / 5.0
    
    def _cosine_similarity_single(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)


def run_week3_benchmark(output_dir: str = "evaluation_results"):
    """运行 Week 3 完整基准测试"""
    
    print("="*60)
    print("PersonaMem Week 3 - Consistency Benchmark (E2)")
    print("="*60)
    
    # 初始化组件
    llm_client = LongCatClient()  # 使用配置中的 API key
    extractor = PersonaExtractor(llm_client=None)  # 使用关键词回退（更快）
    benchmark = ConsistencyBenchmark(llm_client, extractor)
    
    # 定义测试人格（高开放性、高尽责性）
    target_persona = {
        'openness': 0.80,
        'conscientiousness': 0.75,
        'extraversion': 0.50,
        'agreeableness': 0.65,
        'neuroticism': 0.30
    }
    
    results = {}
    
    # 实验 1: 基线（无约束）
    print("\n" + "="*60)
    print("Experiment 1: Baseline (No Constraint)")
    print("="*60)
    results['baseline'] = benchmark.run_session_test(
        target_persona=target_persona,
        use_constraint=False
    )
    
    # 实验 2: 人格约束
    print("\n" + "="*60)
    print("Experiment 2: With Persona Constraint")
    print("="*60)
    results['constrained'] = benchmark.run_session_test(
        target_persona=target_persona,
        use_constraint=True
    )
    
    # 实验 3: 跨 Session 测试
    print("\n" + "="*60)
    print("Experiment 3: Cross-Session Stability")
    print("="*60)
    results['cross_session'] = benchmark.run_cross_session_test(
        target_persona=target_persona,
        num_sessions=3,
        days_between=1
    )
    
    # 汇总结果
    print("\n" + "="*60)
    print("Summary Report")
    print("="*60)
    
    # 基线 vs 约束对比
    baseline_metrics = results['baseline']['metrics']
    constrained_metrics = results['constrained']['metrics']
    
    print("\n[Experiment 1 vs 2: Constraint Effectiveness]")
    print(f"  Baseline Overall Std:    {baseline_metrics['overall_std']:.4f}")
    print(f"  Constrained Overall Std: {constrained_metrics['overall_std']:.4f}")
    print(f"  Baseline Avg KL:         {baseline_metrics['avg_kl_divergence']:.4f}")
    print(f"  Constrained Avg KL:      {constrained_metrics['avg_kl_divergence']:.4f}")
    print(f"  Improvement:             {(baseline_metrics['avg_kl_divergence'] - constrained_metrics['avg_kl_divergence']):.4f}")
    print(f"  Regeneration Rate:       {constrained_metrics['regeneration_rate']:.2f}")
    
    # 跨 Session 稳定性
    stability = results['cross_session']['stability_metrics']
    print("\n[Experiment 3: Cross-Session Stability]")
    print(f"  Overall Session Std:     {stability['overall_session_std']:.4f}")
    print(f"  Stability Score:         {stability['stability_score']:.4f}")
    print(f"  Target:                  > 0.85")
    print(f"  Status:                  {'PASS' if stability['stability_score'] > 0.85 else 'NEED IMPROVEMENT'}")
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"week3_consistency_{timestamp}.json")
        
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
            json.dump(convert_numpy(results), f, ensure_ascii=False, indent=2)
        
        print(f"\n[Saved] Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    run_week3_benchmark()
