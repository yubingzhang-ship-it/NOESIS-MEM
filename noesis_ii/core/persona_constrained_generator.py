"""
Persona Constrained Generator

Week 3: 生成时人格约束（KL惩罚）

实现生成时的人格一致性约束，确保 AI 输出与积累的人格保持一致。
"""

import json
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """生成配置"""
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 0.9
    # KL 惩罚参数
    kl_weight: float = 0.3  # KL 惩罚权重
    kl_threshold: float = 0.15  # 触发重新生成的阈值
    max_regeneration_attempts: int = 3  # 最大重新生成次数


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    final_persona: Dict[str, float]  # 生成文本提取的人格
    target_persona: Dict[str, float]  # 目标人格
    kl_divergence: float
    is_consistent: bool
    regeneration_count: int  # 重新生成次数
    constraint_applied: bool  # 是否应用了约束


class PersonaConstrainedGenerator:
    """
    人格约束生成器
    
    在生成响应时应用人格一致性约束，使用 KL 散度作为偏差度量。
    """
    
    def __init__(self, llm_client, persona_extractor):
        """
        初始化
        
        Args:
            llm_client: LLM 客户端（如 LongCatClient）
            persona_extractor: 人格提取器（PersonaExtractor）
        """
        self.llm_client = llm_client
        self.extractor = persona_extractor
    
    def generate(
        self,
        prompt: str,
        target_persona: Dict[str, float],
        config: Optional[GenerationConfig] = None,
        context: Optional[str] = None
    ) -> GenerationResult:
        """
        生成带人格约束的响应
        
        Args:
            prompt: 用户输入提示
            target_persona: 目标人格（OCEAN五维）
            config: 生成配置
            context: 可选的上下文信息
            
        Returns:
            GenerationResult 包含生成结果和一致性信息
        """
        config = config or GenerationConfig()
        
        # 构建系统提示，注入人格约束
        system_prompt = self._build_system_prompt(target_persona, context)
        
        # 第一次生成
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        generated_text = self.llm_client.generate(
            full_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # 提取生成文本的人格
        extracted_persona = self._extract_persona(generated_text)
        
        # 计算 KL 散度
        kl_div = self._compute_kl_divergence(target_persona, extracted_persona)
        
        regeneration_count = 0
        
        # 如果偏差过大，尝试重新生成
        while kl_div > config.kl_threshold and regeneration_count < config.max_regeneration_attempts:
            regeneration_count += 1
            
            # 增强约束提示
            enhanced_prompt = self._build_regeneration_prompt(
                prompt, target_persona, extracted_persona, kl_div
            )
            
            generated_text = self.llm_client.generate(
                enhanced_prompt,
                temperature=max(0.3, config.temperature - 0.1 * regeneration_count),  # 降低温度
                max_tokens=config.max_tokens
            )
            
            extracted_persona = self._extract_persona(generated_text)
            kl_div = self._compute_kl_divergence(target_persona, extracted_persona)
        
        is_consistent = kl_div <= config.kl_threshold
        
        return GenerationResult(
            text=generated_text,
            final_persona=extracted_persona,
            target_persona=target_persona,
            kl_divergence=kl_div,
            is_consistent=is_consistent,
            regeneration_count=regeneration_count,
            constraint_applied=regeneration_count > 0 or not is_consistent
        )
    
    def _build_system_prompt(self, target_persona: Dict[str, float], context: Optional[str]) -> str:
        """构建带人格约束的系统提示"""
        
        # 将 OCEAN 分数转换为性格描述
        traits = []
        
        if target_persona.get('openness', 0.5) > 0.6:
            traits.append("富有创意和好奇心")
        elif target_persona.get('openness', 0.5) < 0.4:
            traits.append("务实和传统")
            
        if target_persona.get('conscientiousness', 0.5) > 0.6:
            traits.append("有条理和自律")
        elif target_persona.get('conscientiousness', 0.5) < 0.4:
            traits.append("灵活和随性")
            
        if target_persona.get('extraversion', 0.5) > 0.6:
            traits.append("外向和热情")
        elif target_persona.get('extraversion', 0.5) < 0.4:
            traits.append("内向和安静")
            
        if target_persona.get('agreeableness', 0.5) > 0.6:
            traits.append("友善和合作")
        elif target_persona.get('agreeableness', 0.5) < 0.4:
            traits.append("直率和独立")
            
        if target_persona.get('neuroticism', 0.5) > 0.6:
            traits.append("情绪敏感")
        elif target_persona.get('neuroticism', 0.5) < 0.4:
            traits.append("情绪稳定")
        
        trait_desc = "、".join(traits) if traits else "平衡和中性"
        
        prompt = f"""You are an AI assistant with the following personality traits: {trait_desc}.

When responding, embody these traits naturally in your tone, word choice, and perspective.

Guidelines:
- Let your personality shine through authentically
- Be consistent in your approach across all responses
- Adapt your communication style to match your traits
"""
        
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        return prompt
    
    def _build_regeneration_prompt(
        self,
        original_prompt: str,
        target_persona: Dict[str, float],
        current_persona: Dict[str, float],
        kl_div: float
    ) -> str:
        """构建重新生成的增强提示"""
        
        # 找出偏差最大的维度
        max_diff_dim = max(
            target_persona.keys(),
            key=lambda k: abs(target_persona.get(k, 0.5) - current_persona.get(k, 0.5))
        )
        
        dim_names = {
            'openness': '开放性 (Openness)',
            'conscientiousness': '尽责性 (Conscientiousness)',
            'extraversion': '外向性 (Extraversion)',
            'agreeableness': '宜人性 (Agreeableness)',
            'neuroticism': '神经质 (Neuroticism)'
        }
        
        correction_note = f"""
IMPORTANT: Your previous response showed a personality deviation in {dim_names.get(max_diff_dim, max_diff_dim)}.
Target: {target_persona.get(max_diff_dim, 0.5):.2f}, Current: {current_persona.get(max_diff_dim, 0.5):.2f}

Please regenerate your response while paying special attention to embodying the target personality more consistently.
"""
        
        system_prompt = self._build_system_prompt(target_persona, None)
        return f"{system_prompt}\n{correction_note}\n\nUser: {original_prompt}\nAssistant:"
    
    def _extract_persona(self, text: str) -> Dict[str, float]:
        """从文本中提取人格"""
        try:
            ocean = self.extractor.extract(text)
            return ocean.to_dict()
        except Exception:
            # 如果提取失败，返回中性分数
            return {
                'openness': 0.5,
                'conscientiousness': 0.5,
                'extraversion': 0.5,
                'agreeableness': 0.5,
                'neuroticism': 0.5
            }
    
    def _compute_kl_divergence(self, target: Dict[str, float], actual: Dict[str, float]) -> float:
        """
        计算人格偏差距离（欧氏距离）
        
        OCEAN分数不是概率分布，KL散度在此场景下不适用。
        使用欧氏距离作为替代度量，范围 [0, ~2.24]
        """
        total_sq_diff = 0.0
        
        for dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            p = target.get(dim, 0.5)
            q = actual.get(dim, 0.5)
            total_sq_diff += (p - q) ** 2
        
        # 欧氏距离 / sqrt(5) 归一化到 [0, 1] 范围
        euclidean = math.sqrt(total_sq_diff)
        return euclidean / math.sqrt(5)  # 归一化


class ValueConflictDetector:
    """
    价值观冲突检测器
    
    检测生成内容是否与积累的人格/价值观存在矛盾。
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def detect_conflict(
        self,
        generated_text: str,
        persona_history: List[Dict],
        threshold: float = 0.7
    ) -> Tuple[bool, Optional[str]]:
        """
        检测价值观冲突
        
        Args:
            generated_text: 生成的文本
            persona_history: 人格历史记录
            threshold: 冲突判定阈值
            
        Returns:
            (是否存在冲突, 冲突原因)
        """
        if not persona_history:
            return False, None
        
        # 构建冲突检测提示
        prompt = self._build_conflict_prompt(generated_text, persona_history)
        
        try:
            response = self.llm_client.generate(prompt, temperature=0.1, max_tokens=200)
            
            # 解析响应
            if "CONFLICT" in response.upper():
                # 提取冲突原因
                lines = response.split('\n')
                reason = None
                for line in lines:
                    if 'reason:' in line.lower() or '原因：' in line:
                        reason = line.split(':', 1)[-1].strip()
                        break
                return True, reason or "Detected value conflict"
            
            return False, None
            
        except Exception as e:
            print(f"[WARN] Conflict detection failed: {e}")
            return False, None
    
    def _build_conflict_prompt(self, text: str, history: List[Dict]) -> str:
        """构建冲突检测提示"""
        
        # 提取历史中的价值观声明
        value_statements = []
        for record in history[-5:]:  # 最近5条
            if 'values' in record:
                value_statements.append(record['values'])
        
        values_text = '\n'.join([f"- {v}" for v in value_statements]) if value_statements else "No explicit values recorded."
        
        prompt = f"""Analyze if the following response conflicts with the established personality/values.

Established values/statements:
{values_text}

Response to analyze:
"{text[:500]}"

Instructions:
1. Check for contradictions in stance, values, or personality
2. Consider if the tone is inconsistent with previous interactions
3. Look for sudden changes in opinion without justification

Output format:
- If NO conflict: "NO CONFLICT"
- If CONFLICT: "CONFLICT" followed by "Reason: [brief explanation]"

Analysis:"""
        
        return prompt
