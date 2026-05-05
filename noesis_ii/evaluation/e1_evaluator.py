"""
E1 人格提取准确性评估

计算指标：
- Pearson 相关系数 r
- Cohen's Kappa (一致性)
- 均方根误差 (RMSE)
- 相对提升率
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class EvaluationResult:
    """评估结果"""
    dimension: str
    pearson_r: float       # 仅在信号样本（label != 0.5）上计算
    cohens_kappa: float    # 仅在信号样本上计算
    rmse: float            # 在全部样本上计算
    accuracy: float        # 维度分类准确率（±0.1），全部样本
    signal_mae: float = 0.0  # MAE，仅在信号样本上计算（公平对比）
    signal_count: int = 0   # 参与 Pearson/Kappa 计算的信号样本数
    total_count: int = 0    # 总样本数


def compute_pearson_r(x: List[float], y: List[float]) -> float:
    """
    计算 Pearson 相关系数
    
    Args:
        x: 预测值列表
        y: 真实值列表
        
    Returns:
        相关系数 r (-1 到 1)
    """
    n = len(x)
    if n < 2:
        return 0.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compute_cohens_kappa(
    predictions: List[float], 
    labels: List[float], 
    threshold: float = 0.1
) -> float:
    """
    计算 Cohen's Kappa (简化版)
    
    基于高/中/低分类的一致性
    
    Args:
        predictions: 预测分数
        labels: 真实分数
        threshold: 分类阈值
        
    Returns:
        Kappa 值 (-1 到 1)
    """
    n = len(predictions)
    if n < 2:
        return 0.0
    
    def classify(score: float) -> str:
        if score <= 0.4:
            return 'L'  # Low
        elif score <= 0.6:
            return 'M'  # Medium
        else:
            return 'H'  # High
    
    pred_cats = [classify(p) for p in predictions]
    label_cats = [classify(l) for l in labels]
    
    # 观察一致率
    observed = sum(1 for p, l in zip(pred_cats, label_cats) if p == l) / n
    
    # 期望一致率（随机）
    cat_counts = {'L': 0, 'M': 0, 'H': 0}
    for c in pred_cats:
        cat_counts[c] += 1
    
    label_cat_counts = {'L': 0, 'M': 0, 'H': 0}
    for c in label_cats:
        label_cat_counts[c] += 1
    
    expected = sum(
        (cat_counts[c] / n) * (label_cat_counts[c] / n)
        for c in ['L', 'M', 'H']
    )
    
    if expected == 1.0:
        return 1.0
    
    kappa = (observed - expected) / (1 - expected)
    return max(-1.0, min(1.0, kappa))


def compute_rmse(predictions: List[float], labels: List[float]) -> float:
    """计算均方根误差"""
    n = len(predictions)
    if n == 0:
        return 0.0
    
    squared_errors = [(p - l) ** 2 for p, l in zip(predictions, labels)]
    mse = sum(squared_errors) / n
    return math.sqrt(mse)


def compute_mae(predictions: List[float], labels: List[float]) -> float:
    """计算平均绝对误差"""
    n = len(predictions)
    if n == 0:
        return 0.0
    
    return sum(abs(p - l) for p, l in zip(predictions, labels)) / n


def compute_dimension_accuracy(
    predictions: List[float],
    labels: List[float],
    tolerance: float = 0.1
) -> float:
    """计算分类准确率（预测与真实值在 tolerance 范围内）"""
    n = len(predictions)
    if n == 0:
        return 0.0
    
    correct = sum(
        1 for p, l in zip(predictions, labels)
        if abs(p - l) <= tolerance
    )
    
    return correct / n


def evaluate_extractor(
    predictions: Dict[str, List[float]],
    labels: Dict[str, List[float]]
) -> Dict[str, EvaluationResult]:
    """
    评估提取器在所有维度的表现
    
    评估策略：
    - Pearson r / Kappa: 仅在"信号样本"（label != 0.5）上计算。
      原因：数据集每个样本只标注一个维度，其他维度设为中性0.5。
      用全部样本计算时，大量0.5中性值会"稀释"相关性，掩盖真实能力。
    - RMSE / Accuracy: 在全部样本上计算，反映整体误差水平。
    
    Args:
        predictions: {dim: [预测分数列表]}
        labels: {dim: [真实分数列表]}
        
    Returns:
        {dim: EvaluationResult}
    """
    results = {}
    
    for dim in 'OCEAN':
        if dim not in predictions or dim not in labels:
            continue
        
        pred = predictions[dim]
        label = labels[dim]
        
        if len(pred) < 2:
            continue
        
        # 筛选信号样本（有区分度的样本，排除中性的0.5）
        signal_pairs = [(p, l) for p, l in zip(pred, label) if l != 0.5]
        signal_pred  = [p for p, _ in signal_pairs]
        signal_label = [l for _, l in signal_pairs]
        
        # Pearson / Kappa 只在信号样本上计算
        if len(signal_pred) >= 2:
            pearson_r = compute_pearson_r(signal_pred, signal_label)
            kappa = compute_cohens_kappa(signal_pred, signal_label)
            signal_mae = compute_mae(signal_pred, signal_label)  # 信号样本 MAE（公平对比）
        else:
            pearson_r = 0.0
            kappa = 0.0
            signal_mae = 0.0
        
        # RMSE / Accuracy 在全部样本上计算
        rmse = compute_rmse(pred, label)
        accuracy = compute_dimension_accuracy(pred, label)
        
        results[dim] = EvaluationResult(
            dimension=dim,
            pearson_r=pearson_r,
            cohens_kappa=kappa,
            rmse=rmse,
            accuracy=accuracy,
            signal_mae=signal_mae,
            signal_count=len(signal_pred),
            total_count=len(pred)
        )
    
    return results


def compute_average_metrics(results: Dict[str, EvaluationResult]) -> Dict[str, float]:
    """计算平均指标"""
    n = len(results)
    if n == 0:
        return {}
    
    return {
        'avg_pearson_r': sum(r.pearson_r for r in results.values()) / n,
        'avg_cohens_kappa': sum(r.cohens_kappa for r in results.values()) / n,
        'avg_rmse': sum(r.rmse for r in results.values()) / n,
        'avg_accuracy': sum(r.accuracy for r in results.values()) / n,
        'avg_signal_mae': sum(r.signal_mae for r in results.values()) / n,
    }


def compare_baselines(
    ours_results: Dict[str, EvaluationResult],
    baseline_results: Dict[str, EvaluationResult],
    sample_counts: Dict[str, int] = None
) -> Dict[str, Dict[str, float]]:
    """
    比较 Ours 与 Baseline 的表现
    
    Args:
        ours_results: 我们的方法结果
        baseline_results: 基线方法结果
        sample_counts: 每个维度的有效样本数（可选）
    
    Returns:
        {dim: {'pearson_r_improvement': %, 'kappa_improvement': %}}
    """
    comparison = {}
    
    for dim in 'OCEAN':
        if dim not in ours_results or dim not in baseline_results:
            continue
        
        ours = ours_results[dim]
        baseline = baseline_results[dim]
        
        # 计算绝对提升
        absolute_improvement = ours.pearson_r - baseline.pearson_r
        
        # Pearson r 提升
        if abs(baseline.pearson_r) > 0.1:  # 当 baseline 绝对值大于 0.1 时使用相对提升
            # 相对提升率（使用绝对值作为分母）
            pearson_improvement = absolute_improvement / abs(baseline.pearson_r) * 100
        else:  # 当 baseline 接近 0 时，使用绝对提升
            # 直接使用绝对提升作为提升率
            pearson_improvement = absolute_improvement * 100
        
        # Kappa 提升
        if baseline.cohens_kappa != 0:
            kappa_improvement = (ours.cohens_kappa - baseline.cohens_kappa) / abs(baseline.cohens_kappa) * 100
        else:
            # 如果 baseline 为 0，直接使用 ours 的值作为提升
            kappa_improvement = ours.cohens_kappa * 100
        
        comparison[dim] = {
            'pearson_r_improvement': pearson_improvement,
            'kappa_improvement': kappa_improvement,
            'ours_pearson': ours.pearson_r,
            'baseline_pearson': baseline.pearson_r,
            'ours_kappa': ours.cohens_kappa,
            'baseline_kappa': baseline.cohens_kappa,
            'ours_signal_mae': ours.signal_mae,
            'baseline_signal_mae': baseline.signal_mae,
            'absolute_improvement': ours.pearson_r - baseline.pearson_r
        }
    
    return comparison


def format_report(
    baseline_results: Dict[str, EvaluationResult],
    ours_results: Dict[str, EvaluationResult],
    comparison: Dict[str, Dict[str, float]]
) -> str:
    """格式化评估报告"""
    dim_names = {
        'O': '开放性',
        'C': '尽责性',
        'E': '外向性',
        'A': '宜人性',
        'N': '神经质'
    }
    
    lines = [
        "# E1 人格提取准确性评估报告",
        "",
        "## 评估指标说明",
        "- **Pearson r / Cohen's κ**: 仅在信号样本（该维度为目标维度、label≠0.5）上计算",
        "- **全样本 RMSE**: 在全部样本（含 0.5 中性）上计算",
        "- **信号 MAE**: 仅在信号样本上计算，衡量对有区分度样本的预测误差（越小越好）",
        "- **准确率**: ±0.1 范围内的比例，全部样本",
        "- **信号/总数**: 参与 Pearson/Kappa 计算的信号样本数 / 总样本数",
        "",
        "---",
        "",
        "## Baseline (关键词方法) 结果",
        "",
        "| 维度 | Pearson r | Cohen's κ | 全RMSE | 信号MAE | 信号/总数 |",
        "|------|-----------|-----------|--------|--------|----------|",
    ]
    
    for dim in 'OCEAN':
        if dim in baseline_results:
            r = baseline_results[dim]
            sig = getattr(r, 'signal_count', '?')
            tot = getattr(r, 'total_count', '?')
            lines.append(
                f"| {dim} ({dim_names[dim]}) | {r.pearson_r:.3f} | "
                f"{r.cohens_kappa:.3f} | {r.rmse:.3f} | {r.signal_mae:.3f} | {sig}/{tot} |"
            )
    
    lines.extend([
        "",
        "---",
        "",
        "## Ours (LLM方法) 结果",
        "",
        "| 维度 | Pearson r | Cohen's κ | 全RMSE | 信号MAE | 信号/总数 |",
        "|------|-----------|-----------|--------|--------|----------|",
    ])
    
    for dim in 'OCEAN':
        if dim in ours_results:
            r = ours_results[dim]
            sig = getattr(r, 'signal_count', '?')
            tot = getattr(r, 'total_count', '?')
            lines.append(
                f"| {dim} ({dim_names[dim]}) | {r.pearson_r:.3f} | "
                f"{r.cohens_kappa:.3f} | {r.rmse:.3f} | {r.signal_mae:.3f} | {sig}/{tot} |"
            )
    
    lines.extend([
        "",
        "---",
        "",
        "## Ours vs Baseline 提升",
        "",
        "| 维度 | Pearson r 提升 | Kappa 提升 | 信号MAE 改善 |",
        "|------|----------------|-----------|------------|",
    ])
    
    total_pearson_improvement = 0
    total_absolute_improvement = 0
    valid_dims = 0
    
    for dim in 'OCEAN':
        if dim in comparison:
            c = comparison[dim]
            total_pearson_improvement += c['pearson_r_improvement']
            total_absolute_improvement += c['absolute_improvement']
            valid_dims += 1
            # 信号MAE改善：baseline - ours（越小越好，所以是 baseline - ours）
            mae_delta = c['baseline_signal_mae'] - c['ours_signal_mae']
            lines.append(
                f"| {dim} ({dim_names[dim]}) | {c['pearson_r_improvement']:+.1f}% | "
                f"{c['kappa_improvement']:+.1f}% | {mae_delta:+.3f} |"
            )
    
    # 计算平均相对提升率
    avg_improvement = total_pearson_improvement / valid_dims if valid_dims > 0 else 0
    # 计算平均绝对提升
    avg_absolute_improvement = total_absolute_improvement / valid_dims if valid_dims > 0 else 0
    
    # 计算信号 MAE 改善
    avg_ours_mae = sum(c['ours_signal_mae'] for c in comparison.values()) / valid_dims if valid_dims > 0 else 0
    avg_baseline_mae = sum(c['baseline_signal_mae'] for c in comparison.values()) / valid_dims if valid_dims > 0 else 0
    mae_improvement = avg_baseline_mae - avg_ours_mae  # 越小越好，所以 baseline - ours
    
    lines.extend([
        "",
        "---",
        "",
        "## 信号样本 MAE 对比（仅 label≠0.5 的有区分度样本）",
        "",
        f"- **Baseline 平均信号 MAE**: {avg_baseline_mae:.3f}",
        f"- **Ours 平均信号 MAE**: {avg_ours_mae:.3f}",
        f"- **信号 MAE 改善**: {mae_improvement:+.3f} ({'Ours 更优' if mae_improvement > 0 else 'Baseline 更优'})",
        "",
        "---",
        "",
        "## 总结",
        "",
        f"- **平均 Pearson r 提升**: {avg_improvement:+.1f}%",
        f"- **平均绝对提升**: {avg_absolute_improvement:+.3f}",
        f"- **信号 MAE 改善**: {mae_improvement:+.3f}",
        f"- **研究计划目标**: ≥ +20%",
        f"- **评估结论**: {'✅ 达标' if avg_improvement >= 20 else '❌ 未达标'}",
    ])
    
    return "\n".join(lines)
