"""
PersonaExtractor - 人格提取器

替代原 deep_personality，用 LLM 零样本推理替代关键词 Bag-of-Words 方法。

技术改进：
- v1.x: 关键词频率统计（BoW，2000年代方法）
- v2.0: LLM 零样本推理（现代 NLP 方法）

修订历史：
  v2.0 (2026-04-10) - 路线A重构：LLM替代关键词
"""

import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class OCEANScores:
    """OCEAN 五大人格维度分数"""
    openness: float          # 开放性
    conscientiousness: float # 尽责性
    extraversion: float      # 外向性
    agreeableness: float     # 宜人性
    neuroticism: float       # 神经质
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism
        }


class PersonaExtractor:
    """人格提取器 - 使用 LLM 零样本推理"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self._fallback_mode = False
        
    def extract(self, text: str, context: Optional[List[str]] = None) -> OCEANScores:
        """从文本中提取 OCEAN 五大人格维度（别名方法）"""
        return self.extract_ocean(text, context)
    
    def extract_ocean(self, text: str, context: Optional[List[str]] = None) -> OCEANScores:
        """从文本中提取 OCEAN 五大人格维度"""
        if self.llm is None or self._fallback_mode:
            return self._extract_ocean_fallback(text)
        
        prompt = self._build_ocean_prompt(text, context)
        
        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=200)
            return self._parse_ocean_response(response)
        except Exception as e:
            print(f"[PersonaExtractor] LLM提取失败: {e}")
            return self._extract_ocean_fallback(text)
    
    def _build_ocean_prompt(self, text: str, context: Optional[List[str]]) -> str:
        """构建 OCEAN 提取提示"""
        ctx_str = ""
        if context:
            ctx_str = "\n对话上下文:\n" + "\n".join([f"- {c}" for c in context[-3:]])
        
        return f"""分析以下文本中说话者的五大人格特质（OCEAN模型），为每个维度给出0-1之间的分数。

待分析文本: "{text}"{ctx_str}

请按以下JSON格式输出（仅输出JSON，不要其他内容）:
{{
    "openness": 0.7,
    "conscientiousness": 0.5,
    "extraversion": 0.8,
    "agreeableness": 0.6,
    "neuroticism": 0.3
}}

评分标准:
- openness: 开放性（好奇心、创造力）
- conscientiousness: 尽责性（组织性、自律）
- extraversion: 外向性（社交性、活力）
- agreeableness: 宜人性（合作性、同理心）
- neuroticism: 神经质（情绪稳定性，高分=不稳定）
"""
    
    def _parse_ocean_response(self, response: str) -> OCEANScores:
        """解析 LLM 返回的 OCEAN 分数"""
        try:
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return OCEANScores(
                    openness=float(data.get('openness', 0.5)),
                    conscientiousness=float(data.get('conscientiousness', 0.5)),
                    extraversion=float(data.get('extraversion', 0.5)),
                    agreeableness=float(data.get('agreeableness', 0.5)),
                    neuroticism=float(data.get('neuroticism', 0.5))
                )
        except Exception as e:
            print(f"[PersonaExtractor] 解析失败: {e}")
        
        return OCEANScores(0.5, 0.5, 0.5, 0.5, 0.5)
    
    def _extract_ocean_fallback(self, text: str) -> OCEANScores:
        """备用方案：简化启发式（仅降级使用）"""
        text_lower = text.lower()
        
        # 中英文关键词
        openness_kw = ['new', 'explore', 'creative', 'imagine', 'curious', '探索', '创意', '好奇']
        conscientiousness_kw = ['plan', 'organize', 'responsible', 'detail', 'punctual', '计划', '组织', '负责']
        extraversion_kw = ['social', 'active', 'energetic', 'express', 'outgoing', '社交', '活跃', '热情']
        agreeableness_kw = ['help', 'understand', 'cooperate', 'friendly', 'empathy', '帮助', '理解', '合作']
        neuroticism_kw = ['worry', 'anxious', 'nervous', 'stress', 'uneasy', '担心', '焦虑', '紧张']
        
        def score(keywords):
            count = sum(1 for kw in keywords if kw in text_lower)
            return min(0.3 + count * 0.12, 0.9)
        
        return OCEANScores(
            openness=score(openness_kw),
            conscientiousness=score(conscientiousness_kw),
            extraversion=score(extraversion_kw),
            agreeableness=score(agreeableness_kw),
            neuroticism=score(neuroticism_kw)
        )
