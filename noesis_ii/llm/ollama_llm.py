"""
Ollama 本地 LLM 推理接口

支持本地部署的小模型，用于：
1. 备选推理（远程 API 不可用时）
2. 轻量级任务（人格分析、摘要等）
3. 隐私敏感场景

修订历史：
  v1.1 (2026-04-08) - Ollama 稳定性补丁
    - deepseek-r1:1.5b 会将 token 分配给 <think>，导致 response 为空
    - 新增 _force_json_extract() 从 thinking 提取 JSON
    - 新增 _fallback_extract() 规则提取后备方案
    - generate() 增加空响应检测和自动修复
"""

import requests
import json
import re
from typing import Optional


class OllamaLLM:
    """
    Ollama 本地模型推理
    
    v1.1 稳定性补丁：
    - 支持 deepseek-r1:1.5b 等将 thinking 放入 response 的模型
    - 自动检测空响应并从 thinking 提取内容
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "deepseek-r1:1.5b"
    TIMEOUT = 180  # 3 分钟超时
    MIN_RESPONSE_LEN = 10  # 最小有效响应长度

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model = model or self.DEFAULT_MODEL
        self.api_base = f"{self.base_url}/api"
        
        # 检测模型是否使用 thinking 模式
        self._uses_thinking = 'deepseek' in self.model.lower() or 'r1' in self.model.lower()

    def is_available(self) -> bool:
        """检查 ollama 服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list:
        """列出可用模型"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except Exception as e:
            print(f"[OLLAMA] List models failed: {e}")
            return []

    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str:
        """
        生成文本

        Args:
            prompt: 用户输入
            system: 系统提示
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式输出

        Returns:
            生成的文本
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            # Ollama 使用 /api/chat 端点
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()

            result = response.json()

            if stream:
                # 流式输出：收集所有内容
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk:
                                full_content += chunk["message"].get("content", "")
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                return full_content
            else:
                # 非流式输出
                raw_content = result.get("message", {}).get("content", "")
                
                # v1.1 稳定性补丁：检测并修复空响应
                if not raw_content or len(raw_content.strip()) < self.MIN_RESPONSE_LEN:
                    print(f"[OLLAMA] Response too short ({len(raw_content)} chars), checking for thinking content...")
                    fixed_content = self._fix_empty_response(raw_content, result)
                    if fixed_content:
                        print(f"[OLLAMA] Successfully extracted {len(fixed_content)} chars from thinking")
                        return fixed_content
                    else:
                        print("[OLLAMA] Warning: Could not extract valid content, returning raw response")
                        return raw_content
                
                return raw_content

        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama request timeout ({self.TIMEOUT}s)")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    def generate_with_options(
        self,
        prompt: str,
        system: str = None,
        **options,
    ) -> str:
        """
        带额外参数的生成（传递给 Ollama options）

        常用选项：
        - temperature: 0-2，温度越高越随机
        - num_predict: 最大生成 token 数
        - top_p: 核采样概率
        - top_k: top-k 采样
        - repeat_penalty: 重复惩罚
        - seed: 随机种子（用于可复现）
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")


# 便捷函数
_ollama_instance = None


def get_ollama(model: str = None) -> OllamaLLM:
    """获取全局 Ollama 实例"""
    global _ollama_instance
    if _ollama_instance is None:
        _ollama_instance = OllamaLLM(model=model)
    return _ollama_instance


def is_ollama_available() -> bool:
    """检查 ollama 是否可用"""
    try:
        ollama = get_ollama()
        return ollama.is_available()
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
# v1.1 稳定性补丁：处理 deepseek-r1 thinking 问题
# ═══════════════════════════════════════════════════════════════════

def _extract_thinking(raw_content: str) -> str:
    """
    从原始响应中提取 <think> 标签内容
    
    deepseek-r1 等模型会将思考过程放在 <think> 标签中
    """
    if not raw_content:
        return ""
    
    # 匹配 <think>...</think> 标签内容
    thinking_match = re.search(r'<think>\s*(.*?)\s*</think>', raw_content, re.DOTALL)
    if thinking_match:
        return thinking_match.group(1).strip()
    
    # 备选：匹配 <think> 开头到 </think> 结尾
    if '<think>' in raw_content and '</think>' not in raw_content:
        # thinking 未闭合，尝试提取到末尾
        thinking_part = raw_content.split('<think>')[1].strip()
        return thinking_part
    
    return ""


def _force_json_extract(thinking_text: str) -> str:
    """
    从 thinking 文本强制提取 JSON
    
    尝试多种提取策略：
    1. 正则提取完整的 JSON 对象
    2. 提取 JSON 数组
    3. 提取键值对片段
    """
    if not thinking_text:
        return ""
    
    # 策略1：提取完整的 JSON 对象 {...}
    json_match = re.search(r'\{[^{}]*"[^"]+"\s*:\s*[^{}]+\}', thinking_text, re.DOTALL)
    if json_match:
        return json_match.group()
    
    # 策略2：提取可能不完整的 JSON
    # 查找第一个 { 和最后一个 }
    first_brace = thinking_text.find('{')
    last_brace = thinking_text.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        potential_json = thinking_text[first_brace:last_brace + 1]
        # 简单验证：检查是否是有效的 JSON 开头
        if potential_json.startswith('{') and '"' in potential_json:
            return potential_json
    
    # 策略3：查找 JSON 数组 [...]
    array_match = re.search(r'\[[\s\S]*"[^"]+"\s*:', thinking_text)
    if array_match:
        return thinking_text[array_match.start():array_match.start() + 500]
    
    return ""


def _fallback_extract(thinking_text: str) -> str:
    """
    规则提取后备方案
    
    当 JSON 提取失败时，使用启发式规则提取结构化信息：
    1. 提取列表项（- 或 * 开头的行）
    2. 提取键值对
    3. 提取关键词
    """
    if not thinking_text:
        return ""
    
    lines = thinking_text.split('\n')
    extracted_parts = []
    
    for line in lines:
        line = line.strip()
        
        # 跳过太短的行
        if len(line) < 3:
            continue
        
        # 跳过思考过程描述
        if any(kw in line.lower() for kw in ['思考', '想', '分析', '判断', '因为', '所以']):
            if len(line) < 10:
                continue
        
        # 提取列表项
        if line.startswith('- ') or line.startswith('* '):
            extracted_parts.append(line[2:])
            continue
        
        # 提取键值对（格式：键：值 或 键 - 值）
        kv_match = re.match(r'[""\u201c\u201d]?([\u4e00-\u9fff\w]+)[""\u201c\u201d]?\s*[:：\-]\s*(.+)', line)
        if kv_match:
            key = kv_match.group(1).strip()
            value = kv_match.group(2).strip().rstrip(',;。')
            extracted_parts.append(f'"{key}": "{value}"')
    
    if extracted_parts:
        return "{" + ", ".join(extracted_parts[:10]) + "}"
    
    # 最后手段：返回 thinking 的前 200 字符
    return thinking_text[:200] if len(thinking_text) > 200 else thinking_text


# ═══════════════════════════════════════════════════════════════════
# OllamaLLM 补丁方法
# ═══════════════════════════════════════════════════════════════════

def _patch_ollama_fix_response(self, raw_content: str, full_result: dict) -> str:
    """
    修复空响应：从 thinking 中提取内容
    
    Args:
        raw_content: 原始响应内容
        full_result: 完整的 API 响应
        
    Returns:
        修复后的有效内容，或原始内容（如果无法修复）
    """
    # 1. 尝试从 raw_content 提取 thinking
    thinking_content = _extract_thinking(raw_content)
    
    if not thinking_content:
        # 2. 检查 API 响应中是否有其他字段包含 thinking
        message = full_result.get("message", {})
        
        # 有些 API 可能把 thinking 放在其他字段
        for field in ["thinking", "reasoning", "content"]:
            if field in message and field != "content":
                alt_content = message.get(field, "")
                if "<think>" in alt_content:
                    thinking_content = _extract_thinking(alt_content)
                    break
    
    if not thinking_content:
        return raw_content
    
    # 3. 从 thinking 中提取 JSON
    json_content = _force_json_extract(thinking_content)
    
    if json_content:
        return json_content
    
    # 4. JSON 提取失败，使用规则提取
    return _fallback_extract(thinking_content)


# 为 OllamaLLM 类添加补丁方法
OllamaLLM._fix_empty_response = _patch_ollama_fix_response
