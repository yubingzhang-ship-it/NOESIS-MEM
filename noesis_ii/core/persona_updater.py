"""
PersonaUpdater - 人格动态更新器

Week 2 新增模块

核心功能：
- 从经验中学习，更新人格表示
- 加权滑动平均（稳定性）
- 显著事件触发大幅更新（适应性）
- 防漂移保护（不超过历史均值 ±0.2）

修订历史：
  v1.0 (2026-04-10) - Week 2 新增
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .persona_extractor import OCEANScores


@dataclass
class UpdateConfig:
    """更新配置"""
    normal_weight: float = 0.1      # 普通事件权重
    significant_weight: float = 0.3  # 显著事件权重
    drift_threshold: float = 0.2     # 防漂移阈值
    min_history: int = 5             # 最小历史样本数


class PersonaUpdater:
    """
    人格动态更新器
    
    更新策略：
    1. 加权滑动平均：新信号按权重融入现有分布
    2. 显著事件检测：情绪强度 > 0.8 或明确价值观表达
    3. 防漂移保护：限制单次更新幅度
    """
    
    def __init__(self, config: UpdateConfig = None):
        self.config = config or UpdateConfig()
        self.history: List[OCEANScores] = []
    
    def update(
        self,
        current: OCEANScores,
        new_signal: OCEANScores,
        intensity: float = 0.5,
        is_significant: bool = False
    ) -> OCEANScores:
        """
        更新人格分布
        
        Args:
            current: 当前人格分布
            new_signal: 新的人格信号
            intensity: 事件强度 (0-1)
            is_significant: 是否为显著事件
            
        Returns:
            更新后的人格分布
        """
        # 确定更新权重
        if is_significant:
            weight = self.config.significant_weight
        else:
            # 根据强度调整权重
            weight = self.config.normal_weight * (0.5 + intensity * 0.5)
        
        # 计算历史均值（用于防漂移保护）
        history_mean = self._compute_history_mean()
        
        # 加权更新
        updated = {}
        for dim in ['openness', 'conscientiousness', 'extraversion', 
                    'agreeableness', 'neuroticism']:
            curr_val = getattr(current, dim)
            new_val = getattr(new_signal, dim)
            
            # 滑动平均
            blended = (1 - weight) * curr_val + weight * new_val
            
            # 防漂移保护
            if history_mean:
                mean_val = history_mean.get(dim, blended)
                lower_bound = mean_val - self.config.drift_threshold
                upper_bound = mean_val + self.config.drift_threshold
                blended = np.clip(blended, lower_bound, upper_bound)
            
            updated[dim] = blended
        
        # 记录历史
        self.history.append(new_signal)
        if len(self.history) > 100:  # 保留最近100个
            self.history = self.history[-100:]
        
        return OCEANScores(**updated)
    
    def _compute_history_mean(self) -> Optional[Dict[str, float]]:
        """计算历史均值"""
        if len(self.history) < self.config.min_history:
            return None
        
        mean = {}
        for dim in ['openness', 'conscientiousness', 'extraversion',
                    'agreeableness', 'neuroticism']:
            values = [getattr(h, dim) for h in self.history]
            mean[dim] = np.mean(values)
        return mean
    
    def detect_significant_event(
        self,
        text: str,
        emotion: str = None,
        intensity: float = 0.5
    ) -> bool:
        """
        检测是否为显著事件
        
        标准：
        1. 情绪强度 > 0.8
        2. 包含明确价值观表达的关键词
        3. 涉及重大决策或人生事件
        """
        # 强度阈值
        if intensity > 0.8:
            return True
        
        # 价值观关键词
        value_keywords = [
            'believe', 'value', 'principle', 'never', 'always',
            'important', 'matter', 'care about', 'stand for',
            '相信', '价值观', '原则', '永远', '总是',
            '重要', '在乎', '坚持'
        ]
        
        text_lower = text.lower()
        has_value_expression = any(kw in text_lower for kw in value_keywords)
        
        # 重大事件关键词
        life_events = [
            'decided', 'choice', 'change', 'transform',
            '决定', '选择', '改变', '转变', '人生'
        ]
        has_life_event = any(kw in text_lower for kw in life_events)
        
        return has_value_expression and has_life_event and intensity > 0.5
    
    def compute_stability(self, window_size: int = 10) -> float:
        """
        计算人格稳定性（变异系数的倒数）
        
        Returns:
            稳定性分数 (0-1)，越高越稳定
        """
        if len(self.history) < window_size:
            return 1.0  # 默认稳定
        
        recent = self.history[-window_size:]
        
        # 计算各维度的标准差
        stds = []
        for dim in ['openness', 'conscientiousness', 'extraversion',
                    'agreeableness', 'neuroticism']:
            values = [getattr(h, dim) for h in recent]
            stds.append(np.std(values))
        
        # 平均变异系数
        mean_std = np.mean(stds)
        
        # 转换为稳定性分数（std越小越稳定）
        stability = max(0, 1 - mean_std * 2)
        return stability
    
    def suggest_update_strategy(self, current: OCEANScores, 
                                 new_signal: OCEANScores) -> Dict:
        """
        建议更新策略
        
        Returns:
            包含建议权重和是否为显著事件的字典
        """
        # 计算差异幅度
        diffs = []
        for dim in ['openness', 'conscientiousness', 'extraversion',
                    'agreeableness', 'neuroticism']:
            diff = abs(getattr(new_signal, dim) - getattr(current, dim))
            diffs.append(diff)
        
        max_diff = max(diffs)
        avg_diff = np.mean(diffs)
        
        # 根据差异决定策略
        if max_diff > 0.3:
            return {
                'weight': self.config.significant_weight,
                'is_significant': True,
                'reason': f'Large deviation detected (max_diff={max_diff:.2f})'
            }
        elif avg_diff > 0.15:
            return {
                'weight': self.config.normal_weight * 1.5,
                'is_significant': False,
                'reason': f'Moderate shift (avg_diff={avg_diff:.2f})'
            }
        else:
            return {
                'weight': self.config.normal_weight,
                'is_significant': False,
                'reason': f'Minor adjustment (avg_diff={avg_diff:.2f})'
            }