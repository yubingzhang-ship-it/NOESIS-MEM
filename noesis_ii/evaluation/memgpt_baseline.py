"""
MemGPT-style Baseline Implementation

Week 4: MemGPT 风格的记忆管理基线

MemGPT 核心思想：
- 区分工作记忆（Working Memory）和外部记忆（External Memory）
- 使用函数调用（Function Calling）管理记忆
- FIFO 淘汰策略
"""

import json
import time
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MemGPTMessage:
    """MemGPT 风格的消息"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp
        }


@dataclass
class WorkingMemory:
    """
    MemGPT 工作记忆
    
    类似操作系统的 RAM，容量有限，访问快速
    """
    system_instructions: str = ""
    conversation_history: List[MemGPTMessage] = field(default_factory=list)
    user_persona: Dict = field(default_factory=dict)
    agent_persona: Dict = field(default_factory=dict)
    
    # 容量限制
    max_history: int = 10
    
    def add_message(self, role: str, content: str):
        """添加消息，FIFO 淘汰"""
        msg = MemGPTMessage(role=role, content=content)
        self.conversation_history.append(msg)
        
        # FIFO 淘汰
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_context(self) -> str:
        """获取工作记忆上下文"""
        context = f"System: {self.system_instructions}\n\n"
        
        if self.user_persona:
            context += f"User Persona: {json.dumps(self.user_persona)}\n\n"
        
        for msg in self.conversation_history:
            context += f"{msg.role.capitalize()}: {msg.content}\n"
        
        return context


class MemGPTBaseline:
    """
    MemGPT 风格基线实现
    
    特点：
    - 工作记忆 + 外部记忆架构
    - 函数调用风格记忆管理
    - 基于关键词的记忆检索
    """
    
    def __init__(self, llm_client=None, working_memory_size: int = 10):
        self.llm_client = llm_client
        self.working_memory = WorkingMemory(max_history=working_memory_size)
        
        # 外部记忆（类似硬盘）
        self.external_memory: List[Dict] = []
        self.memory_counter = 0
    
    def store_to_external_memory(self, content: str, category: str = "general") -> str:
        """
        存储到外部记忆
        
        模拟 MemGPT 的 core_memory_replace 函数
        """
        self.memory_counter += 1
        memory_id = f"mem_{self.memory_counter:06d}"
        
        memory = {
            'id': memory_id,
            'content': content,
            'category': category,
            'timestamp': time.time(),
            'access_count': 0
        }
        
        self.external_memory.append(memory)
        return memory_id
    
    def search_external_memory(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        搜索外部记忆
        
        简单关键词匹配（MemGPT 使用向量检索）
        """
        query_words = set(query.lower().split())
        
        scored_memories = []
        for mem in self.external_memory:
            mem_words = set(mem['content'].lower().split())
            overlap = len(query_words & mem_words)
            score = overlap / max(len(query_words), 1)
            
            scored_memories.append((mem, score))
        
        # 按分数排序
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for mem, score in scored_memories[:top_k]:
            if score > 0:
                mem['access_count'] += 1
                results.append({
                    'id': mem['id'],
                    'content': mem['content'],
                    'score': score
                })
        
        return results
    
    def chat(self, user_input: str) -> str:
        """
        对话接口
        
        模拟 MemGPT 的完整流程：
        1. 接收用户输入
        2. 决定是否需要检索外部记忆
        3. 生成响应
        4. 更新工作记忆
        """
        # 添加到工作记忆
        self.working_memory.add_message('user', user_input)
        
        # 检查是否需要检索外部记忆
        # 简单启发式：如果工作记忆中无相关信息，则检索
        relevant_memories = self._retrieve_if_needed(user_input)
        
        # 构建提示
        prompt = self._build_prompt(user_input, relevant_memories)
        
        # 生成响应
        if self.llm_client:
            response = self.llm_client.generate(prompt, temperature=0.7)
        else:
            response = self._mock_generate(user_input, relevant_memories)
        
        # 添加到工作记忆
        self.working_memory.add_message('assistant', response)
        
        # 存储重要信息到外部记忆
        self._store_important_info(user_input, response)
        
        return response
    
    def _retrieve_if_needed(self, query: str) -> List[Dict]:
        """判断是否需要检索外部记忆"""
        # 简单策略：总是检索（MemGPT 使用更复杂的决策）
        return self.search_external_memory(query, top_k=3)
    
    def _build_prompt(self, user_input: str, memories: List[Dict]) -> str:
        """构建完整提示"""
        prompt = self.working_memory.get_context()
        
        if memories:
            prompt += "\n[Relevant Memories]\n"
            for mem in memories:
                prompt += f"- {mem['content']}\n"
        
        prompt += f"\nUser: {user_input}\nAssistant:"
        
        return prompt
    
    def _mock_generate(self, user_input: str, memories: List[Dict]) -> str:
        """模拟生成"""
        # 简单的模拟响应
        if memories:
            return f"Based on my memory, I recall: {memories[0]['content'][:50]}..."
        return f"I understand: {user_input[:30]}..."
    
    def _store_important_info(self, user_input: str, response: str):
        """存储重要信息到外部记忆"""
        # 简单启发式：存储用户输入（实际 MemGPT 使用 LLM 判断）
        if len(user_input) > 20:
            self.store_to_external_memory(user_input, category="user_input")


class Mem0Baseline:
    """
    Mem0 风格基线实现
    
    Mem0 特点：
    - 分层记忆：工作记忆、短期记忆、长期记忆
    - 自动分类存储（facts, preferences, etc.）
    - 基于重要性的记忆保留
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
        # 分层记忆
        self.working_memory: List[Dict] = []  # 当前对话
        self.short_term_memory: List[Dict] = []  # 最近对话
        self.long_term_memory: List[Dict] = []  # 重要事实
        
        self.memory_counter = 0
    
    def add_to_working_memory(self, role: str, content: str):
        """添加到工作记忆"""
        self.working_memory.append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })
        
        # 限制工作记忆大小
        if len(self.working_memory) > 10:
            # 转移到短期记忆
            old_msg = self.working_memory.pop(0)
            self._promote_to_short_term(old_msg)
    
    def _promote_to_short_term(self, message: Dict):
        """提升到短期记忆"""
        self.short_term_memory.append(message)
        
        # 短期记忆也有限制
        if len(self.short_term_memory) > 50:
            old_msg = self.short_term_memory.pop(0)
            # 简单启发式：长消息可能是重要事实
            if len(old_msg['content']) > 50:
                self._promote_to_long_term(old_msg)
    
    def _promote_to_long_term(self, message: Dict):
        """提升到长期记忆"""
        self.memory_counter += 1
        
        memory = {
            'id': f"ltm_{self.memory_counter:06d}",
            'content': message['content'],
            'category': self._classify_content(message['content']),
            'timestamp': message['timestamp'],
            'importance': len(message['content']) / 100  # 简单重要性评分
        }
        
        self.long_term_memory.append(memory)
    
    def _classify_content(self, content: str) -> str:
        """内容分类"""
        content_lower = content.lower()
        
        if any(w in content_lower for w in ['like', 'love', 'prefer', 'hate', '讨厌', '喜欢']):
            return 'preference'
        elif any(w in content_lower for w in ['is', 'are', 'was', 'were', '是', '有']):
            return 'fact'
        elif any(w in content_lower for w in ['plan', 'will', 'going to', '计划', '将要']):
            return 'plan'
        else:
            return 'general'
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关记忆"""
        query_words = set(query.lower().split())
        
        all_memories = (
            [{'source': 'working', **m} for m in self.working_memory] +
            [{'source': 'short_term', **m} for m in self.short_term_memory] +
            [{'source': 'long_term', **m} for m in self.long_term_memory]
        )
        
        scored = []
        for mem in all_memories:
            mem_words = set(mem['content'].lower().split())
            overlap = len(query_words & mem_words)
            
            # 加权：长期记忆更重要
            weight = 1.0
            if mem.get('source') == 'long_term':
                weight = 1.5
            
            score = overlap * weight
            scored.append((mem, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [m for m, s in scored[:top_k] if s > 0]
    
    def chat(self, user_input: str) -> str:
        """对话接口"""
        # 添加到工作记忆
        self.add_to_working_memory('user', user_input)
        
        # 检索相关记忆
        memories = self.retrieve(user_input, top_k=5)
        
        # 构建上下文
        context = self._build_context(memories)
        
        # 生成响应
        if self.llm_client:
            response = self.llm_client.generate(
                f"{context}\nUser: {user_input}\nAssistant:",
                temperature=0.7
            )
        else:
            response = self._mock_generate(user_input, memories)
        
        # 添加到工作记忆
        self.add_to_working_memory('assistant', response)
        
        return response
    
    def _build_context(self, memories: List[Dict]) -> str:
        """构建上下文"""
        context_parts = []
        
        for mem in memories:
            source = mem.get('source', 'unknown')
            content = mem['content']
            context_parts.append(f"[{source.upper()}] {content}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _mock_generate(self, user_input: str, memories: List[Dict]) -> str:
        """模拟生成"""
        if memories:
            return f"I remember you mentioned: {memories[0]['content'][:40]}..."
        return f"I see. Tell me more about that."


def compare_baselines():
    """对比 MemGPT 和 Mem0 基线"""
    print("="*60)
    print("MemGPT vs Mem0 Baseline Comparison")
    print("="*60)
    
    # 创建实例
    memgpt = MemGPTBaseline()
    mem0 = Mem0Baseline()
    
    # 模拟对话
    conversations = [
        "Hi, I'm Alex. I love hiking and photography.",
        "What's your favorite outdoor activity?",
        "I went hiking last weekend. It was amazing!",
        "Do you remember what I like to do?",
        "I also enjoy cooking, especially Italian food.",
    ]
    
    print("\n[MemGPT Style]")
    print("-" * 60)
    for msg in conversations:
        response = memgpt.chat(msg)
        print(f"User: {msg}")
        print(f"Assistant: {response}")
        print()
    
    print(f"External memory size: {len(memgpt.external_memory)}")
    
    print("\n[Mem0 Style]")
    print("-" * 60)
    for msg in conversations:
        response = mem0.chat(msg)
        print(f"User: {msg}")
        print(f"Assistant: {response}")
        print()
    
    print(f"Long-term memory size: {len(mem0.long_term_memory)}")
    print(f"Short-term memory size: {len(mem0.short_term_memory)}")


if __name__ == "__main__":
    compare_baselines()
