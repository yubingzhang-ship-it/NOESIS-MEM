"""
E1 LLM 人格提取器

Ours: LLM 零样本推理提取 OCEAN 分数
"""

from typing import Dict, Optional, List
import json


LLM_EXTRACTION_PROMPT = """Analyze the following text and evaluate the speaker's personality traits according to the Big Five personality model.

Please carefully read the text and infer the speaker's tendencies on each dimension and subdimension.

Big Five Factors and Subdimensions:
- O (Openness):
  * Values: appreciation for diverse values and beliefs
  * Ideas: intellectual curiosity and creativity
  * Feelings: sensitivity to feelings and emotions
  * Fantasy: imagination and daydreaming
  * Aesthetics: appreciation for art and beauty
  * Actions: willingness to try new activities
- C (Conscientiousness):
  * Self-Efficacy: belief in one's own abilities
  * Self-Discipline: ability to control impulses
  * Orderliness: preference for order and organization
  * Dutifulness: sense of responsibility and duty
  * Deliberation: careful decision-making
  * Achievement Striving: ambition and goal-setting
- E (Extraversion):
  * Warmth: friendliness and affection
  * Positive Emotions: cheerfulness and enthusiasm
  * Gregariousness: sociability and preference for company
  * Excitement Seeking: desire for excitement and stimulation
  * Energy: activity level and vitality
  * Assertiveness: confidence and dominance
- A (Agreeableness):
  * Trust: belief in others' trustworthiness
  * Tender-Mindedness: compassion and empathy
  * Straightforwardness: honesty and directness
  * Modesty: humility and lack of vanity
  * Cooperation: willingness to work with others
  * Compliance: tendency to avoid conflict
  * Altruism: willingness to help others
- N (Neuroticism):
  * Vulnerability: susceptibility to stress (high score means easily stressed)
  * Self-Consciousness: concern about others' opinions (high score means very concerned)
  * Impulsiveness: tendency to act without thinking (high score means impulsive)
  * Depression: tendency to feel sad or hopeless (high score means prone to depression)
  * Anxiety: tendency to feel worried or nervous (high score means anxious)
  * Anger: tendency to feel angry or irritable (high score means easily angered)

Important Note for Neuroticism:
- High Neuroticism (score 0.7-1.0): Emotional instability, easily stressed, anxious, or irritable
- Low Neuroticism (score 0.0-0.3): Emotional stability, calm, resilient, able to handle stress well

Each dimension should be scored on a scale of 0.0-1.0:
- 0.0-0.3: Low
- 0.4-0.6: Medium
- 0.7-1.0: High

Text:
{text}

Please return the analysis results in JSON format, generating specific scores based on the text content. Include both the main OCEAN dimensions and the relevant subdimension:
{{
  "O": 0.7,
  "C": 0.8,
  "E": 0.6,
  "A": 0.9,
  "N": 0.3,
  "subdimension": "Altruism",
  "subdimension_score": 0.95
}}
"""


# 批量分析 Prompt 模板
LLM_BATCH_PROMPT_HEADER = """You are a personality analyst. Analyze the following {n} texts and evaluate each speaker's Big Five personality traits.

Big Five scoring rules (0.0-1.0):
- O (Openness): curiosity, creativity, openness to experience
- C (Conscientiousness): organization, discipline, goal-orientation
- E (Extraversion): sociability, assertiveness, positive emotions
- A (Agreeableness): cooperation, trust, empathy, altruism
- N (Neuroticism): HIGH=emotional instability/anxiety/anger; LOW=calm/resilient

Score range: 0.0-0.3=Low, 0.4-0.6=Medium, 0.7-1.0=High
For N: describe emotional INSTABILITY (high score = anxious/angry/unstable)

Also identify the most relevant subdimension for each text.

Texts:
"""

LLM_BATCH_PROMPT_FOOTER = """
Return a JSON array with exactly {n} objects, one per text, in order:
[
  {{"id": 0, "O": 0.7, "C": 0.8, "E": 0.6, "A": 0.9, "N": 0.3, "subdimension": "Altruism", "subdimension_score": 0.95}},
  {{"id": 1, "O": 0.4, "C": 0.5, "E": 0.7, "A": 0.6, "N": 0.4, "subdimension": "Warmth", "subdimension_score": 0.75}},
  ...
]

IMPORTANT: Return ONLY the JSON array, no explanation. Exactly {n} items.
"""


class LLMPersonaExtractor:
    """
    基于 LLM 零样本推理的人格提取器（Ours 方法）
    
    方法：使用精心设计的 prompt 让 LLM 从文本中推断 OCEAN 分数
    """
    
    # OCEAN 维度映射（支持多种命名格式）
    DIM_MAPPING = {
        'O': ['openness', 'open', 'o'],
        'C': ['conscientiousness', 'conscientious', 'c'],
        'E': ['extraversion', 'extravert', 'extraverted', 'e'],
        'A': ['agreeableness', 'agreeable', 'a'],
        'N': ['neuroticism', 'neurotic', 'n', 'emotional_stability']
    }
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    def _extract_dim_score(self, scores: dict, dim: str) -> float:
        """从字典中提取指定维度的分数"""
        # 直接键
        if dim in scores:
            return float(scores[dim])
        
        # 映射键
        for alias in self.DIM_MAPPING.get(dim, []):
            if alias in scores:
                return float(scores[alias])
        
        # 默认值
        return 0.5
    
    def _parse_single(self, scores: dict) -> dict:
        """从解析好的 dict 中提取标准化结果"""
        result = {}
        for dim in 'OCEAN':
            result[dim] = max(0.0, min(1.0, self._extract_dim_score(scores, dim)))
        result['subdimension'] = scores.get('subdimension', 'Unknown')
        result['subdimension_score'] = max(0.0, min(1.0, float(scores.get('subdimension_score', 0.5))))
        return result

    def _neutral(self) -> dict:
        return {'O': 0.5, 'C': 0.5, 'E': 0.5, 'A': 0.5, 'N': 0.5,
                'subdimension': 'Unknown', 'subdimension_score': 0.5}

    def extract(self, text: str) -> Dict:
        """使用 LLM 提取单条 OCEAN 分数"""
        if self.llm is None:
            return self._neutral()
        
        prompt = LLM_EXTRACTION_PROMPT.format(text=text)
        
        try:
            if hasattr(self.llm, 'chat'):
                response = self.llm.chat(prompt)
            elif hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
            else:
                raise AttributeError("LLM client has no .chat() or .generate() method")
            
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                scores = json.loads(response)
            
            return self._parse_single(scores)
            
        except Exception as e:
            print(f"LLM 提取失败: {e}")
            return self._neutral()

    def extract_batch(self, texts: List[str]) -> List[Dict]:
        """
        批量提取：将多条文本合并为一次 LLM 调用，返回等长列表。
        
        Args:
            texts: 文本列表（建议 10-100 条）
            
        Returns:
            与 texts 等长的 OCEAN 结果列表，失败项填 neutral
        """
        n = len(texts)
        if n == 0:
            return []
        if self.llm is None:
            return [self._neutral() for _ in texts]

        # 构造批量 prompt
        items_str = ""
        for i, text in enumerate(texts):
            # 截断过长文本，避免超 token
            truncated = text[:500] if len(text) > 500 else text
            items_str += f"\n[{i}] {truncated}\n"

        prompt = (
            LLM_BATCH_PROMPT_HEADER.format(n=n)
            + items_str
            + LLM_BATCH_PROMPT_FOOTER.format(n=n)
        )

        # 批量输出需要更多 token：每条约 80 tokens，留 20% 余量
        batch_max_tokens = max(1000, n * 100)

        if hasattr(self.llm, 'chat'):
            response = self.llm.chat(prompt, max_tokens=batch_max_tokens)
        elif hasattr(self.llm, 'generate'):
            response = self.llm.generate(prompt, max_tokens=batch_max_tokens)
        else:
            raise AttributeError("No chat/generate method")

        # 解析 JSON 数组（含自动修复）
        import re
        raw_list = self._parse_json_array(response, n)

        if not isinstance(raw_list, list):
            raise ValueError(f"Expected list, got {type(raw_list)}")

        # 按 id 对齐，缺失填 neutral
        results = [self._neutral()] * n
        for item in raw_list:
            idx = item.get('id', None)
            if idx is None:
                continue
            if 0 <= int(idx) < n:
                results[int(idx)] = self._parse_single(item)

        return results

    def _parse_json_array(self, response: str, expected_n: int) -> list:
        """
        从 LLM 响应中提取 JSON 数组，带多级修复。
        失败时直接抛出异常（不 fallback）。
        """
        import re

        # Step 1: 找 [ ... ] 块（完整或截断）
        arr_match = re.search(r'\[.*\]', response, re.DOTALL)
        if arr_match:
            raw = arr_match.group()
        else:
            # 响应被截断：找最后一个 [ 开头的内容
            bracket_pos = response.rfind('[')
            if bracket_pos == -1:
                print(f"[DEBUG] LLM 原始响应（找不到 [ 符号）:\n{response[:500]}")
                raise ValueError("Response contains no JSON array marker")

            raw = response[bracket_pos:]
            # 响应截断：打印调试信息
            print(f"[DEBUG] 响应被截断（无结束 ]），尝试修复。原始长度={len(response)}，截断前 200 字符:\n{response[:200]}")
            print(f"[DEBUG] 截断末尾 200 字符:\n{response[-200:]}")

            # 修复截断的响应：找到最后一个完整对象，截掉不完整部分
            # 找最后一个 "}, {" 或 "}" 结尾
            last_complete = raw.rfind('},')
            if last_complete == -1:
                last_complete = raw.rfind('}')
            if last_complete == -1:
                raise ValueError("Cannot find any complete object in truncated response")

            raw = raw[:last_complete + 1] + ']'
            print(f"[DEBUG] 截断修复后长度={len(raw)}，末尾:\n{raw[-100:]}")

        # Step 2: 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON 直接解析失败: {e}")
            print(f"[DEBUG] 原始 JSON 块（前 500 字符）:\n{raw[:500]}")
            print(f"[DEBUG] 原始 JSON 块（末 200 字符）:\n{raw[-200:]}")

        # Step 3: 常见 LLM JSON 问题修复
        fixed = raw

        # 3a: 删除行尾注释 // ...
        fixed = re.sub(r'//[^\n"]*', '', fixed)

        # 3b: 删除 trailing comma before ] or }
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)

        # 3c: 单引号 → 双引号
        fixed = re.sub(r"'([^']*)'", lambda m: '"' + m.group(1).replace('"', '\\"') + '"', fixed)

        # 3d: 去掉 ... 省略号行
        fixed = re.sub(r'\.\.\.\s*,?\s*\n', '', fixed)

        # 3e: 如果末尾没有 ] 补上
        stripped = fixed.rstrip()
        if not stripped.endswith(']'):
            last_obj = stripped.rfind('}')
            if last_obj != -1:
                fixed = stripped[:last_obj + 1] + ']'

        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e2:
            print(f"[DEBUG] JSON 修复后仍失败: {e2}")
            print(f"[DEBUG] 修复后 JSON（前 500 字符）:\n{fixed[:500]}")
            print(f"[DEBUG] 修复后 JSON（末 200 字符）:\n{fixed[-200:]}")
            raise ValueError(f"JSON parse failed after repair: {e2}") from e2

    async def async_extract(self, text: str) -> Dict:
        """异步版本"""
        if self.llm is None:
            return self._neutral()
        
        prompt = LLM_EXTRACTION_PROMPT.format(text=text)
        
        try:
            if hasattr(self.llm, 'achat'):
                response = await self.llm.achat([{"role": "user", "content": prompt}])
            elif hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
            else:
                raise AttributeError("LLM client has no async method")
            
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                scores = json.loads(response)
            
            return self._parse_single(scores)
            
        except Exception as e:
            print(f"LLM 异步提取失败: {e}")
            return self._neutral()


# 全局实例（需要外部设置 llm）
LLM_EXTRACTOR = LLMPersonaExtractor()

