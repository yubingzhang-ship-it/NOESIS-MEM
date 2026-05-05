"""
E2 实验答题提取器

从 LLM 回答中提取 OCEAN 分数和选择
用于计算跨Session一致性
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import re
import json


@dataclass
class ExtractedResponse:
    """提取的回答结果"""
    choice: str  # 'A' or 'B'
    confidence: float  # 置信度 0-1
    ocean_scores: Dict[str, float]  # 估计的 OCEAN 分数
    reasoning: str  # 选择理由
    raw_text: str  # 原始回答


class ResponseExtractor:
    """
    从 LLM 回答中提取结构和语义信息
    
    方法：
    1. 规则提取：解析 A/B 选择
    2. LLM 推断：基于回答风格推断 OCEAN 分数
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    def extract(self, response_text: str, question) -> ExtractedResponse:
        """
        从回答中提取信息
        
        Args:
            response_text: LLM 的原始回答
            question: 对应的题目对象
        """
        # 1. 提取选择
        choice = self._extract_choice(response_text)
        
        # 2. 提取理由
        reasoning = self._extract_reasoning(response_text)
        
        # 3. 基于选择和理由推断 OCEAN 分数
        ocean_scores = self._infer_ocean_from_response(
            choice, reasoning, question.dimension
        )
        
        # 4. 计算置信度
        confidence = self._compute_confidence(response_text, choice)
        
        return ExtractedResponse(
            choice=choice,
            confidence=confidence,
            ocean_scores=ocean_scores,
            reasoning=reasoning,
            raw_text=response_text
        )
    
    def _extract_choice(self, text: str) -> str:
        """提取 A 或 B 选择"""
        text = text.strip().upper()
        
        # 优先匹配明确的选择
        patterns = [
            r'^选择\s*([AB])',
            r'^我选择\s*([AB])',
            r'^答案是?\s*([AB])',
            r'^选\s*([AB])',
            r'\b([AB])\s*[.。、]',
            r'^([AB])[.。、]',
            r'\b([AB])\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # 如果没有明确匹配，统计 A/B 出现次数
        a_count = len(re.findall(r'\bA\b', text))
        b_count = len(re.findall(r'\bB\b', text))
        
        if a_count > b_count:
            return 'A'
        elif b_count > a_count:
            return 'B'
        
        return 'A'  # 默认
    
    def _extract_reasoning(self, text: str) -> str:
        """提取选择理由"""
        # 去掉开头的选择部分
        text = re.sub(r'^(选择|我选择|答案是?|选)\s*[AB][.。、]?\s*', '', text)
        return text.strip()
    
    def _infer_ocean_from_response(
        self, 
        choice: str, 
        reasoning: str, 
        target_dimension: str
    ) -> Dict[str, float]:
        """
        基于回答推断 OCEAN 分数
        
        使用题目设计的维度倾向作为参考
        结合回答风格进行微调
        """
        # 基础分数：基于选择
        base_scores = {
            'O': 0.5,
            'C': 0.5,
            'E': 0.5,
            'A': 0.5,
            'N': 0.5
        }
        
        # 根据回答长度和确定性微调
        reasoning_length = len(reasoning)
        
        # 长回答通常意味着更高的 Openness
        if reasoning_length > 100:
            base_scores['O'] += 0.15
        elif reasoning_length < 30:
            base_scores['O'] -= 0.1
        
        # 根据目标维度和选择调整
        # 这里简化处理，实际应该用 LLM 进行更细致的推断
        
        # 确保分数在 [0, 1] 范围内
        for dim in base_scores:
            base_scores[dim] = max(0.0, min(1.0, base_scores[dim]))
        
        return base_scores
    
    def _compute_confidence(self, text: str, choice: str) -> float:
        """计算回答置信度"""
        confidence = 0.5
        
        # 选择明确
        if choice in text[:50]:
            confidence += 0.2
        
        # 有理由说明
        if len(text) > 50:
            confidence += 0.15
        
        # 包含决策词
        decision_words = ['选择', '认为', '相信', '倾向', '决定']
        if any(word in text for word in decision_words):
            confidence += 0.15
        
        return min(1.0, confidence)
    
    def extract_batch(
        self, 
        responses: List[str], 
        questions: List
    ) -> List[ExtractedResponse]:
        """批量提取"""
        results = []
        for response, question in zip(responses, questions):
            results.append(self.extract(response, question))
        return results
    
    def extract_ocean_with_llm(
        self, 
        response_text: str, 
        question_text: str,
        llm=None
    ) -> Dict[str, float]:
        """
        使用 LLM 推断 OCEAN 分数（更准确但需要 LLM 调用）
        
        Args:
            response_text: LLM 的原始回答
            question_text: 题目文本
            llm: LLM 客户端
        """
        if llm is None:
            llm = self.llm
        
        if llm is None:
            # fallback: 使用简化推断
            return self._infer_ocean_from_response('', '', 'O')
        
        prompt = f"""分析以下回答者的人格特征。

题目：{question_text}

回答：{response_text}

请估计回答者在 OCEAN 五因素模型上的得分（0.0-1.0）：
- O (Openness): 开放性
- C (Conscientiousness): 尽责性
- E (Extraversion): 外向性
- A (Agreeableness): 宜人性
- N (Neuroticism): 神经质（注意：神经质分数低表示情绪更稳定）

请以 JSON 格式返回：
{{"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5}}
"""
        
        response = llm.chat(prompt)
        
        try:
            # 尝试解析 JSON
            scores = json.loads(response)
            return scores
        except:
            # fallback
            return {'O': 0.5, 'C': 0.5, 'E': 0.5, 'A': 0.5, 'N': 0.5}


def compute_ocean_stability(
    session_a: List[Dict[str, float]], 
    session_b: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    计算 OCEAN 稳定性（标准差）
    
    Args:
        session_a: Session A 的 OCEAN 分数列表
        session_b: Session B 的 OCEAN 分数列表
        
    Returns:
        各维度的标准差
    """
    import statistics
    
    dimensions = ['O', 'C', 'E', 'A', 'N']
    stabilities = {}
    
    for dim in dimensions:
        scores_a = [s[dim] for s in session_a]
        scores_b = [s[dim] for s in session_b]
        
        # 计算配对标准差
        paired_scores = list(zip(scores_a, scores_b))
        diffs = [abs(a - b) for a, b in paired_scores]
        
        # 平均差异作为稳定性指标
        stabilities[dim] = statistics.mean(diffs) if diffs else 0.0
    
    return stabilities


def compute_ocean_cosine(
    session_a: List[Dict[str, float]], 
    session_b: List[Dict[str, float]]
) -> float:
    """
    计算 OCEAN 向量余弦相似度
    
    Args:
        session_a: Session A 的 OCEAN 分数列表
        session_b: Session B 的 OCEAN 分数列表
        
    Returns:
        平均余弦相似度
    """
    import math
    
    dimensions = ['O', 'C', 'E', 'A', 'N']
    cosines = []
    
    for scores_a, scores_b in zip(session_a, session_b):
        vec_a = [scores_a[d] for d in dimensions]
        vec_b = [scores_b[d] for d in dimensions]
        
        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
        norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
        
        if norm_a > 0 and norm_b > 0:
            cosine = dot_product / (norm_a * norm_b)
            cosines.append(cosine)
    
    return sum(cosines) / len(cosines) if cosines else 0.0
