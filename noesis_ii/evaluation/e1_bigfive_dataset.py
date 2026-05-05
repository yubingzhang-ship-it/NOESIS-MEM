"""
Big Five Personality Facets Dataset Adapter

适配 Big Five Personality Facets Dataset 用于 E1 实验评估

数据集来源: https://github.com/lunat5078/BigFive-Personality-Facets-Dataset
"""

import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class BigFiveSample:
    """大五人格样本"""
    id: str
    dimension: str
    subdimension: str
    polarity: str
    label: int
    lang: str
    text: str
    ocean_label: Dict[str, float]  # 转换后的 OCEAN 分数
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'dimension': self.dimension,
            'subdimension': self.subdimension,
            'polarity': self.polarity,
            'label': self.label,
            'lang': self.lang,
            'text': self.text,
            'ocean_label': self.ocean_label
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BigFiveSample':
        return cls(
            id=data['id'],
            dimension=data['dimension'],
            subdimension=data['subdimension'],
            polarity=data['polarity'],
            label=data['label'],
            lang=data['lang'],
            text=data['text'],
            ocean_label=data['ocean_label']
        )


class BigFiveDataset:
    """
    Big Five Personality Facets Dataset 适配器
    
    将原始的二元标签数据集转换为 OCEAN 五维分数格式
    
    数据集标签含义：
    - polarity=positive, label=1: 表现出积极（高分）行为 → 该维度得高分(0.8)
    - polarity=positive, label=0: 未表现出积极行为    → 该维度得低分(0.2)
    - polarity=negative, label=0: 表现出消极（低分）行为 → 该维度得低分(0.2)
    - polarity=negative, label=1: 未表现出消极行为    → 该维度得高分(0.8)
    """
    
    def __init__(self, data_dir: str = None):
        self.samples: List[BigFiveSample] = []
        
        # 如果未指定目录，自动定位到项目根目录下的 data/bigfive/data
        if data_dir is None:
            # 从本文件位置向上两级找到项目根目录
            _this_dir = os.path.dirname(os.path.abspath(__file__))
            _project_root = os.path.dirname(os.path.dirname(_this_dir))
            data_dir = os.path.join(_project_root, "data", "bigfive", "data")
        
        self.data_dir = data_dir
        
        if os.path.exists(data_dir):
            self._load_from_files()
        else:
            raise FileNotFoundError(f"Big Five dataset directory not found: {data_dir}")
    
    def _load_from_files(self):
        """从文件加载数据集"""
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        # 转换为 OCEAN 分数
                        ocean_label = self._convert_to_ocean(item)
                        sample = BigFiveSample(
                            id=item['id'],
                            dimension=item['dimension'],
                            subdimension=item['subdimension'],
                            polarity=item['polarity'],
                            label=item['label'],
                            lang=item['lang'],
                            text=item['text'],
                            ocean_label=ocean_label
                        )
                        self.samples.append(sample)
    
    def _convert_to_ocean(self, item: Dict) -> Dict[str, float]:
        """
        将二元标签转换为 OCEAN 五维分数
        
        规则：
        - 对于目标维度：根据极性和标签计算分数
        - 对于其他维度：设置为中等值 0.5
        
        通用逻辑（O/C/E/A）：
        - polarity=positive, label=1 → 展现了积极行为 → 高分(0.8)
        - polarity=positive, label=0 → 未展现积极行为 → 低分(0.2)
        - polarity=negative, label=0 → 展现了消极行为 → 低分(0.2)
        - polarity=negative, label=1 → 未展现消极行为 → 高分(0.8)
        
        N（Neuroticism）特殊逻辑：
        - 数据集中 positive polarity = "展现了情绪稳定行为"（如：冷静应对被打断）
          → 情绪稳定 = 低神经质 → N 低分(0.2)
        - 数据集中 negative polarity = "展现了情绪化行为"（如：发怒、怒斥）
          → 情绪化 = 高神经质 → N 高分(0.8)
        - 即 N 维度的"高分"与 polarity 方向相反（negative → 高 N）
        """
        # 维度映射（包含拼写错误的处理）
        dimension_map = {
            "Agreeableness": "A",
            "Agreeabableness": "A",  # 处理拼写错误
            "Conscientiousness": "C",
            "Extraversion": "E",
            "Neuroticism": "N",
            "Neurooticism": "N",  # 处理拼写错误
            "Openness": "O"
        }
        
        # 初始化 OCEAN 分数
        ocean = {
            "O": 0.5,  # 默认中等值
            "C": 0.5,
            "E": 0.5,
            "A": 0.5,
            "N": 0.5
        }
        
        # 获取当前维度的缩写
        current_dim = item['dimension']
        if current_dim in dimension_map:
            dim_key = dimension_map[current_dim]
            
            if dim_key == "N":
                # Neuroticism 特殊处理：
                # polarity=negative（情绪化行为）→ 高神经质(0.8)
                # polarity=positive（情绪稳定行为）→ 低神经质(0.2)
                # label 在此数据集中只有 0(negative) 和 1(positive) 两种，
                # 且 polarity 与 label 一一对应（negative→0, positive→1）
                if item['polarity'] == "negative":
                    score = 0.8 if item['label'] == 0 else 0.2
                else:  # positive
                    score = 0.2 if item['label'] == 1 else 0.8
            else:
                # 通用逻辑（O/C/E/A）
                if item['polarity'] == "positive":
                    score = 0.8 if item['label'] == 1 else 0.2
                else:
                    # negative polarity: label=0 means acted negatively (low),
                    # label=1 means did NOT act negatively (high)
                    score = 0.2 if item['label'] == 0 else 0.8
            
            ocean[dim_key] = score
        
        return ocean
    
    def get_samples(self, max_samples: int = None) -> List[BigFiveSample]:
        """获取样本列表，可选限制数量
        
        当指定max_samples时，从每个维度均匀采样
        """
        if not max_samples:
            return self.samples
        
        # 按维度分组
        samples_by_dim = {}
        for sample in self.samples:
            dim = sample.dimension
            if dim not in samples_by_dim:
                samples_by_dim[dim] = []
            samples_by_dim[dim].append(sample)
        
        # 计算每个维度应取的样本数
        num_dims = len(samples_by_dim)
        samples_per_dim = max_samples // num_dims
        remaining = max_samples % num_dims
        
        # 从每个维度采样
        result = []
        for dim, dim_samples in samples_by_dim.items():
            # 确保不超过维度的实际样本数
            take = min(samples_per_dim, len(dim_samples))
            if remaining > 0:
                take += 1
                remaining -= 1
            result.extend(dim_samples[:take])
        
        return result
    
    def get_by_dimension(self, dimension: str) -> List[BigFiveSample]:
        """按维度获取样本"""
        return [s for s in self.samples if s.dimension == dimension]
    
    def get_by_subdimension(self, subdimension: str) -> List[BigFiveSample]:
        """按子维度获取样本"""
        return [s for s in self.samples if s.subdimension == subdimension]
    
    def split(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[List, List, List]:
        """划分训练/验证/测试集"""
        import random
        random.shuffle(self.samples)
        n = len(self.samples)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = self.samples[:train_end]
        val = self.samples[train_end:val_end]
        test = self.samples[val_end:]
        
        return train, val, test
    
    def save(self, output_path: str):
        """保存处理后的数据集"""
        data = [s.to_dict() for s in self.samples]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx) -> BigFiveSample:
        return self.samples[idx]


# 测试代码
if __name__ == "__main__":
    # 初始化数据集
    dataset = BigFiveDataset()
    print(f"Loaded {len(dataset)} samples")
    
    # 查看前5个样本
    for i, sample in enumerate(dataset.samples[:5]):
        print(f"Sample {i+1}:")
        print(f"  Text: {sample.text}")
        print(f"  Dimension: {sample.dimension}")
        print(f"  Subdimension: {sample.subdimension}")
        print(f"  Polarity: {sample.polarity}")
        print(f"  Label: {sample.label}")
        print(f"  OCEAN: {sample.ocean_label}")
        print()
    
    # 按维度统计
    dimensions = set([s.dimension for s in dataset.samples])
    for dim in dimensions:
        count = len(dataset.get_by_dimension(dim))
        print(f"{dim}: {count} samples")
