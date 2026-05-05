"""
PersonaMem Benchmark - 完整评估基准

Week 2: 评估基准测试框架

运行完整实验：
1. 人格提取准确性 (E1)
2. 人格一致性测试 (E2)
3. 记忆检索性能 (E3)
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from noesis_ii.core.persona_extractor import PersonaExtractor, OCEANScores
from noesis_ii.core.persona_updater import PersonaUpdater, UpdateConfig
from noesis_ii.llm.longcat_client import LongCatClient
from noesis_ii.evaluation.persona_caption_dataset import PersonaCaptionDataset
from noesis_ii.evaluation.evaluator import PersonaEvaluator, baseline_keyword_extraction


def run_experiment_e1(dataset: PersonaCaptionDataset, 
                      use_llm: bool = False) -> Dict:
    """
    实验 E1: 人格提取准确性
    
    对比：
    - BL-1: 关键词基线
    - Ours: LLM 零样本（如果 use_llm=True）
    """
    print("\n" + "=" * 60)
    print("Experiment E1: Personality Extraction Accuracy")
    print("=" * 60)
    
    # 划分数据集
    train, val, test = dataset.split(train_ratio=0.7, val_ratio=0.15)
    print(f"Dataset split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    evaluator = PersonaEvaluator()
    
    # BL-1: 关键词基线
    print("\n[BL-1] Running keyword baseline...")
    baseline_preds = []
    for sample in test:
        pred = baseline_keyword_extraction(sample.text)
        baseline_preds.append(pred)
    
    baseline_truth = [OCEANScores(s.openness, s.conscientiousness, s.extraversion,
                                   s.agreeableness, s.neuroticism) for s in test]
    
    baseline_result = evaluator.evaluate(baseline_preds, baseline_truth, "baseline")
    
    # Ours: LLM 方法
    if use_llm:
        print("\n[Ours] Running LLM extraction...")
        # 从配置文件读取 API key
        api_key = None
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 
                                      'default_config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                api_key = config.get('llm', {}).get('api_key')
        except:
            pass
        
        llm_client = LongCatClient(api_key=api_key)
        extractor = PersonaExtractor(llm_client=llm_client)
        
        llm_preds = []
        for i, sample in enumerate(test):
            pred = extractor.extract_ocean(sample.text)
            llm_preds.append(pred)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(test)} samples...")
        
        llm_result = evaluator.evaluate(llm_preds, baseline_truth, "ours")
    else:
        print("\n[Ours] LLM not available, using fallback mode...")
        # 使用相同的 fallback 作为模拟
        llm_preds = baseline_preds
        llm_result = evaluator.evaluate(llm_preds, baseline_truth, "ours")
    
    # 打印报告
    evaluator.print_report()
    
    # 返回结果
    return {
        'baseline': baseline_result.to_dict(),
        'ours': llm_result.to_dict() if use_llm else baseline_result.to_dict(),
        'comparison': evaluator.compare_methods()
    }


def run_experiment_e2(n_rounds: int = 3) -> Dict:
    """
    实验 E2: 人格一致性测试
    
    模拟多轮对话，测试人格稳定性
    """
    print("\n" + "=" * 60)
    print("Experiment E2: Personality Consistency")
    print("=" * 60)
    
    # 创建 extractor 和 updater
    llm_client = LongCatClient()
    extractor = PersonaExtractor(llm_client=llm_client)
    updater = PersonaUpdater(config=UpdateConfig(normal_weight=0.1))
    
    # 模拟对话历史
    conversations = [
        [
            "I love exploring new ideas and creative solutions.",
            "Abstract concepts always fascinate me.",
            "I enjoy thinking outside the box."
        ],
        [
            "Meeting new people energizes me.",
            "I thrive in social situations.",
            "Being around others makes me happy."
        ],
        [
            "I always plan ahead carefully.",
            "Organization is key to success.",
            "I never miss a deadline."
        ]
    ]
    
    all_profiles = []
    
    for round_idx, conv in enumerate(conversations[:n_rounds]):
        print(f"\n[Round {round_idx + 1}] Processing conversation...")
        
        # 提取每句话的人格
        round_profiles = []
        for text in conv:
            profile = extractor.extract_ocean(text)
            round_profiles.append(profile)
        
        # 聚合本轮人格
        avg_profile = OCEANScores(
            openness=sum(p.openness for p in round_profiles) / len(round_profiles),
            conscientiousness=sum(p.conscientiousness for p in round_profiles) / len(round_profiles),
            extraversion=sum(p.extraversion for p in round_profiles) / len(round_profiles),
            agreeableness=sum(p.agreeableness for p in round_profiles) / len(round_profiles),
            neuroticism=sum(p.neuroticism for p in round_profiles) / len(round_profiles)
        )
        
        all_profiles.append(avg_profile)
        print(f"  O:{avg_profile.openness:.2f} C:{avg_profile.conscientiousness:.2f} "
              f"E:{avg_profile.extraversion:.2f} A:{avg_profile.agreeableness:.2f} "
              f"N:{avg_profile.neuroticism:.2f}")
    
    # 计算跨轮稳定性
    if len(all_profiles) >= 2:
        stds = {
            'openness': sum((p.openness - all_profiles[0].openness)**2 
                           for p in all_profiles) / len(all_profiles),
            'conscientiousness': sum((p.conscientiousness - all_profiles[0].conscientiousness)**2 
                                    for p in all_profiles) / len(all_profiles),
            'extraversion': sum((p.extraversion - all_profiles[0].extraversion)**2 
                               for p in all_profiles) / len(all_profiles),
            'agreeableness': sum((p.agreeableness - all_profiles[0].agreeableness)**2 
                                for p in all_profiles) / len(all_profiles),
            'neuroticism': sum((p.neuroticism - all_profiles[0].neuroticism)**2 
                              for p in all_profiles) / len(all_profiles)
        }
        
        print("\n[Stability Analysis]")
        for dim, var in stds.items():
            std = var ** 0.5
            status = "PASS" if std < 0.10 else "FAIL"
            print(f"  {dim}: std={std:.3f} [{status}]")
        
        return {
            'n_rounds': len(all_profiles),
            'stds': {k: v**0.5 for k, v in stds.items()},
            'all_pass': all(v**0.5 < 0.10 for v in stds.values())
        }
    
    return {'n_rounds': len(all_profiles)}


def run_benchmark(output_dir: str = None, use_llm: bool = False) -> Dict:
    """
    运行完整基准测试
    
    Args:
        output_dir: 结果输出目录
        use_llm: 是否使用 LLM（需要 API key）
        
    Returns:
        完整测试结果
    """
    print("\n" + "=" * 70)
    print("PersonaMem Benchmark Suite")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"LLM Mode: {'Enabled' if use_llm else 'Fallback (Keyword)'}")
    
    # 加载数据集
    print("\n[Setup] Loading dataset...")
    dataset = PersonaCaptionDataset()
    print(f"  Loaded {len(dataset)} samples")
    
    # 运行实验
    results = {}
    
    # E1: 人格提取准确性
    results['e1_extraction'] = run_experiment_e1(dataset, use_llm=use_llm)
    
    # E2: 人格一致性
    results['e2_consistency'] = run_experiment_e2(n_rounds=3)
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
        
        # 转换 numpy 类型为 Python 原生类型
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
        
        results_clean = convert_numpy(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, ensure_ascii=False, indent=2)
        
        print(f"\n[Saved] Results saved to {output_path}")
    
    # 总结
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    
    e1 = results['e1_extraction']
    if 'comparison' in e1 and 'success' in e1['comparison']:
        print(f"E1 Extraction: {'PASS' if e1['comparison']['success'] else 'NEED IMPROVEMENT'}")
    
    e2 = results['e2_consistency']
    if 'all_pass' in e2:
        print(f"E2 Consistency: {'PASS' if e2['all_pass'] else 'NEED IMPROVEMENT'}")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # 运行基准测试
    results = run_benchmark(
        output_dir="noesis_ii/evaluation/results",
        use_llm=False  # 默认使用 fallback 模式（无需 API key）
    )