"""
Personality-CAPTION Dataset Adapter

适配 Personality-CAPTION 数据集用于 PersonaMem 评估

数据集格式:
- text: 对话文本或描述
- openness: OCEAN 开放性分数 (0-1)
- conscientiousness: OCEAN 尽责性分数 (0-1)
- extraversion: OCEAN 外向性分数 (0-1)
- agreeableness: OCEAN 宜人性分数 (0-1)
- neuroticism: OCEAN 神经质分数 (0-1)

数据来源: Personality-CAPTION (基于 BigFive 的人格标注数据集)
"""

import json
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class PersonaSample:
    """人格样本"""
    text: str
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    source: str = "synthetic"
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism,
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonaSample':
        return cls(
            text=data['text'],
            openness=data['openness'],
            conscientiousness=data['conscientiousness'],
            extraversion=data['extraversion'],
            agreeableness=data['agreeableness'],
            neuroticism=data['neuroticism'],
            source=data.get('source', 'synthetic')
        )


class PersonaCaptionDataset:
    """
    Personality-CAPTION 数据集适配器
    
    由于 Personality-CAPTION 是英文数据集，我们同时提供：
    1. 英文原版样本（用于验证方法）
    2. 中文合成样本（用于实际应用）
    """
    
    def __init__(self, data_path: str = None):
        self.samples: List[PersonaSample] = []
        self.data_path = data_path
        
        if data_path:
            self._load_from_file()
        else:
            # 使用内置合成数据
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples: int = 100):
        """生成合成数据集（用于快速验证）"""
        
        # 英文样本模板（基于 OCEAN 特征）
        english_templates = [
            # High Openness
            ("I love exploring new ideas and trying creative approaches to problems.", 
             0.85, 0.50, 0.60, 0.55, 0.40),
            ("Abstract concepts and philosophical discussions always fascinate me.",
             0.80, 0.45, 0.50, 0.50, 0.45),
            
            # High Conscientiousness
            ("I always plan ahead and organize my work carefully before starting.",
             0.50, 0.85, 0.55, 0.60, 0.35),
            ("Being punctual and meeting deadlines is very important to me.",
             0.45, 0.80, 0.50, 0.55, 0.40),
            
            # High Extraversion
            ("I enjoy meeting new people and thrive in social situations.",
             0.60, 0.50, 0.85, 0.65, 0.35),
            ("Being the center of attention energizes me rather than drains me.",
             0.55, 0.45, 0.80, 0.60, 0.40),
            
            # High Agreeableness
            ("I always try to help others and consider their feelings first.",
             0.55, 0.60, 0.50, 0.85, 0.40),
            ("Cooperation and harmony in a team matter more than being right.",
             0.50, 0.55, 0.45, 0.80, 0.35),
            
            # High Neuroticism
            ("I often worry about things going wrong and feel anxious easily.",
             0.50, 0.45, 0.40, 0.50, 0.85),
            ("Stressful situations tend to overwhelm me more than others.",
             0.45, 0.40, 0.45, 0.45, 0.80),
            
            # Low Openness
            ("I prefer sticking to what I know rather than trying new things.",
             0.20, 0.55, 0.50, 0.55, 0.50),
            
            # Low Conscientiousness
            ("I tend to go with the flow rather than planning everything out.",
             0.55, 0.20, 0.60, 0.50, 0.55),
            
            # Low Extraversion
            ("I prefer spending time alone or with a small group of close friends.",
             0.50, 0.55, 0.20, 0.60, 0.50),
            
            # Low Agreeableness
            ("I focus on achieving my goals even if it means being competitive.",
             0.55, 0.60, 0.55, 0.20, 0.50),
            
            # Low Neuroticism
            ("I stay calm under pressure and don't let emotions affect my decisions.",
             0.55, 0.60, 0.55, 0.55, 0.15),
        ]
        
        # 中文样本模板
        chinese_templates = [
            # 高开放性
            ("我喜欢探索新想法，尝试创造性的解决问题方法。",
             0.85, 0.50, 0.60, 0.55, 0.40),
            ("抽象概念和哲学讨论总是让我着迷。",
             0.80, 0.45, 0.50, 0.50, 0.45),
            
            # 高尽责性
            ("我总是提前计划，仔细组织工作后再开始。",
             0.50, 0.85, 0.55, 0.60, 0.35),
            ("准时和遵守截止日期对我来说非常重要。",
             0.45, 0.80, 0.50, 0.55, 0.40),
            
            # 高外向性
            ("我喜欢结识新朋友，在社交场合中如鱼得水。",
             0.60, 0.50, 0.85, 0.65, 0.35),
            ("成为关注的焦点让我感到兴奋而不是疲惫。",
             0.55, 0.45, 0.80, 0.60, 0.40),
            
            # 高宜人性
            ("我总是尽力帮助他人，优先考虑他们的感受。",
             0.55, 0.60, 0.50, 0.85, 0.40),
            ("团队合作和和谐比证明自己正确更重要。",
             0.50, 0.55, 0.45, 0.80, 0.35),
            
            # 高神经质
            ("我经常担心事情出错，容易感到焦虑。",
             0.50, 0.45, 0.40, 0.50, 0.85),
            ("压力情境往往比其他更容易让我不知所措。",
             0.45, 0.40, 0.45, 0.45, 0.80),
            
            # 低开放性
            ("我喜欢坚持自己熟悉的事物，而不是尝试新东西。",
             0.20, 0.55, 0.50, 0.55, 0.50),
            
            # 低尽责性
            ("我倾向于随波逐流，而不是把一切都计划好。",
             0.55, 0.20, 0.60, 0.50, 0.55),
            
            # 低外向性
            ("我喜欢独处或与少数亲密朋友在一起。",
             0.50, 0.55, 0.20, 0.60, 0.50),
            
            # 低宜人性
            ("即使意味着竞争，我也会专注于实现自己的目标。",
             0.55, 0.60, 0.55, 0.20, 0.50),
            
            # 低神经质
            ("我在压力下保持冷静，不让情绪影响决策。",
             0.55, 0.60, 0.55, 0.55, 0.15),
        ]
        
        # 合并并扩展样本
        all_templates = english_templates + chinese_templates
        
        # 添加随机扰动生成更多样本
        for i in range(n_samples):
            template = random.choice(all_templates)
            text, o, c, e, a, n = template
            
            # 添加小幅度随机扰动（±0.05）
            sample = PersonaSample(
                text=text,
                openness=max(0, min(1, o + random.uniform(-0.05, 0.05))),
                conscientiousness=max(0, min(1, c + random.uniform(-0.05, 0.05))),
                extraversion=max(0, min(1, e + random.uniform(-0.05, 0.05))),
                agreeableness=max(0, min(1, a + random.uniform(-0.05, 0.05))),
                neuroticism=max(0, min(1, n + random.uniform(-0.05, 0.05))),
                source='synthetic'
            )
            self.samples.append(sample)
    
    def _load_from_file(self):
        """从文件加载数据集"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    self.samples.append(PersonaSample.from_dict(item))
        except Exception as e:
            print(f"[WARN] Failed to load dataset from {self.data_path}: {e}")
            print("[INFO] Falling back to synthetic data")
            self._generate_synthetic_data()
    
    def save(self, output_path: str):
        """保存数据集到文件"""
        data = [s.to_dict() for s in self.samples]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def split(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[List, List, List]:
        """划分训练/验证/测试集"""
        random.shuffle(self.samples)
        n = len(self.samples)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = self.samples[:train_end]
        val = self.samples[train_end:val_end]
        test = self.samples[val_end:]
        
        return train, val, test
    
    def get_by_trait(self, trait: str, min_val: float = 0.7) -> List[PersonaSample]:
        """按特质筛选样本"""
        trait_map = {
            'O': 'openness',
            'C': 'conscientiousness',
            'E': 'extraversion',
            'A': 'agreeableness',
            'N': 'neuroticism'
        }
        trait_key = trait_map.get(trait.upper(), trait)
        
        return [s for s in self.samples if getattr(s, trait_key) >= min_val]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx) -> PersonaSample:
        return self.samples[idx]