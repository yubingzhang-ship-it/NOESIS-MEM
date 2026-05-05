"""
中文 Big Five 人格数据集适配器

适配 OpenReview 上的中文 BigFive 数据集用于 E1 实验评估

数据集来源: https://openreview.net/pdf?id=6q-mLPJICaa
"""

import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ChineseBigFiveSample:
    """中文大五人格样本"""
    id: str
    text: str
    dimension: str
    score: float  # 0-1 分数
    lang: str = "zh"
    
    @property
    def ocean_label(self) -> Dict[str, float]:
        """转换为 OCEAN 五维分数格式"""
        ocean = {
            "O": 0.5,  # 默认中等值
            "C": 0.5,
            "E": 0.5,
            "A": 0.5,
            "N": 0.5
        }
        
        # 维度映射
        dimension_map = {
            "开放性": "O",
            "尽责性": "C",
            "外向性": "E",
            "宜人性": "A",
            "神经质": "N"
        }
        
        if self.dimension in dimension_map:
            dim_key = dimension_map[self.dimension]
            ocean[dim_key] = self.score
        
        return ocean
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'dimension': self.dimension,
            'score': self.score,
            'lang': self.lang
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChineseBigFiveSample':
        return cls(
            id=data['id'],
            text=data['text'],
            dimension=data['dimension'],
            score=data['score'],
            lang=data.get('lang', 'zh')
        )


class ChineseBigFiveDataset:
    """
    中文 Big Five 人格数据集适配器
    
    加载和处理中文 BigFive 数据集
    """
    
    def __init__(self, data_path: str = "data/chinese_bigfive/data.json"):
        self.samples: List[ChineseBigFiveSample] = []
        self.data_path = data_path
        
        if os.path.exists(data_path):
            self._load_from_file()
        else:
            # 如果文件不存在，尝试创建示例数据
            self._create_sample_data()
            print(f"警告: 中文BigFive数据集文件不存在，已创建示例数据")
    
    def _load_from_file(self):
        """从文件加载数据集"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                sample = ChineseBigFiveSample.from_dict(item)
                self.samples.append(sample)
    
    def _create_sample_data(self):
        """创建示例数据"""
        # 创建示例数据
        sample_data = [
            {
                "id": "1",
                "text": "我喜欢尝试新事物，对未知的世界充满好奇",
                "dimension": "开放性",
                "score": 0.8
            },
            {
                "id": "2",
                "text": "我做事总是井井有条，喜欢制定计划",
                "dimension": "尽责性",
                "score": 0.9
            },
            {
                "id": "3",
                "text": "我性格开朗，喜欢和朋友聚会",
                "dimension": "外向性",
                "score": 0.85
            },
            {
                "id": "4",
                "text": "我总是为他人着想，乐于助人",
                "dimension": "宜人性",
                "score": 0.9
            },
            {
                "id": "5",
                "text": "我容易感到焦虑，对未来有些担忧",
                "dimension": "神经质",
                "score": 0.7
            },
            {
                "id": "6",
                "text": "我喜欢传统的生活方式，不太愿意改变",
                "dimension": "开放性",
                "score": 0.3
            },
            {
                "id": "7",
                "text": "我做事比较随性，不喜欢被规则束缚",
                "dimension": "尽责性",
                "score": 0.4
            },
            {
                "id": "8",
                "text": "我喜欢安静的环境，更愿意独处",
                "dimension": "外向性",
                "score": 0.35
            },
            {
                "id": "9",
                "text": "我比较直接，说话不会拐弯抹角",
                "dimension": "宜人性",
                "score": 0.4
            },
            {
                "id": "10",
                "text": "我心态平和，很少感到紧张或焦虑",
                "dimension": "神经质",
                "score": 0.25
            }
        ]
        
        # 保存示例数据
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        # 加载示例数据
        self._load_from_file()
    
    def get_samples(self, max_samples: int = None) -> List[ChineseBigFiveSample]:
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
    
    def get_by_dimension(self, dimension: str) -> List[ChineseBigFiveSample]:
        """按维度获取样本"""
        return [s for s in self.samples if s.dimension == dimension]
    
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
    
    def __getitem__(self, idx) -> ChineseBigFiveSample:
        return self.samples[idx]


# 测试代码
if __name__ == "__main__":
    # 初始化数据集
    dataset = ChineseBigFiveDataset()
    print(f"Loaded {len(dataset)} samples")
    
    # 查看前5个样本
    for i, sample in enumerate(dataset.samples[:5]):
        print(f"Sample {i+1}:")
        print(f"  Text: {sample.text}")
        print(f"  Dimension: {sample.dimension}")
        print(f"  Score: {sample.score}")
        print(f"  OCEAN: {sample.ocean_label}")
        print()
    
    # 按维度统计
    dimensions = set([s.dimension for s in dataset.samples])
    for dim in dimensions:
        count = len(dataset.get_by_dimension(dim))
        print(f"{dim}: {count} samples")
