"""
E2 实验组管理器

三组对比：
- BL-1: 无人格约束的纯 LLM（GPT 基线）
- BL-2: 静态 System Prompt 人格（Persona-Chat 方式）
- Ours: PersonaMem 动态人格积累
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json


@dataclass
class SessionResult:
    """Session 结果"""
    session_id: str
    group: str
    timestamp: str
    responses: List[Dict[str, Any]]  # 每道题的回答
    ocean_profile: Dict[str, float]  # 估计的 OCEAN 画像
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseGroupManager(ABC):
    """实验组管理器基类"""
    
    def __init__(self, group_name: str, llm_client=None):
        self.group_name = group_name
        self.llm = llm_client
        self.session_history: List[SessionResult] = []
    
    @abstractmethod
    def generate_system_prompt(self) -> str:
        """生成该组的 system prompt"""
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict]) -> str:
        """对话接口"""
        pass
    
    @abstractmethod
    def get_persona_state(self) -> Dict[str, Any]:
        """获取人格状态（用于 PersonaMem 组）"""
        pass
    
    def record_session(self, session_result: SessionResult):
        """记录 session 结果"""
        self.session_history.append(session_result)
    
    def get_session_count(self) -> int:
        """获取已记录的 session 数"""
        return len(self.session_history)
    
    def save_history(self, path: str):
        """保存历史记录"""
        data = {
            'group': self.group_name,
            'sessions': [
                {
                    'session_id': s.session_id,
                    'timestamp': s.timestamp,
                    'responses': s.responses,
                    'ocean_profile': s.ocean_profile,
                    'metadata': s.metadata
                }
                for s in self.session_history
            ]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class BL1GroupManager(BaseGroupManager):
    """
    BL-1: 无人格约束的纯 LLM
    
    特点：
    - 只有通用任务指令
    - 无任何人格相关上下文
    - 模拟"健忘"的 AI
    """
    
    def __init__(self, llm_client=None):
        super().__init__('BL-1', llm_client)
        self._system_prompt = None
    
    def generate_system_prompt(self) -> str:
        """生成无人格约束的 system prompt"""
        return """你是一个乐于助人的 AI 助手。请回答用户的问题。
回答应该清晰、有条理。如果不确定，请如实说明。"""
    
    async def chat(self, messages: List[Dict]) -> str:
        """纯 LLM 对话"""
        if self.llm is None:
            raise ValueError("LLM client is required for BL-1")
        
        # 构建消息
        system_msg = {"role": "system", "content": self.generate_system_prompt()}
        all_messages = [system_msg] + messages
        
        response = await self.llm.achat(all_messages)
        return response
    
    def get_persona_state(self) -> Dict[str, Any]:
        """无人格状态"""
        return {
            'has_persona': False,
            'ocean_profile': None,
            'memory_traces': []
        }


class BL2GroupManager(BaseGroupManager):
    """
    BL-2: 静态 System Prompt 人格
    
    特点：
    - 固定的 personas prompt
    - 模拟 Persona-Chat 方式
    - 人格不会随对话演化
    """
    
    def __init__(
        self, 
        llm_client=None,
        persona_description: str = None
    ):
        super().__init__('BL-2', llm_client)
        
        # 默认人格描述
        self.persona_description = persona_description or """你是一个理性、务实的人。
你喜欢分析和逻辑推理，做决定时倾向于深思熟虑。
你重视诚实和效率，不喜欢废话和虚情假意。
在表达观点时，你倾向于直接、有话直说。"""
        
        self._system_prompt = None
    
    def generate_system_prompt(self) -> str:
        """生成静态人格 system prompt"""
        return f"""你是一个具有特定人格的 AI 助手。请在回答中体现以下人格特征：

{self.persona_description}

请根据这个人格特征来回答问题。如果问题与你的判断冲突，选择你人格中更倾向的选项并说明理由。"""
    
    async def chat(self, messages: List[Dict]) -> str:
        """静态人格对话"""
        if self.llm is None:
            raise ValueError("LLM client is required for BL-2")
        
        system_msg = {"role": "system", "content": self.generate_system_prompt()}
        all_messages = [system_msg] + messages
        
        response = await self.llm.achat(all_messages)
        return response
    
    def get_persona_state(self) -> Dict[str, Any]:
        """静态人格状态"""
        return {
            'has_persona': True,
            'is_static': True,
            'ocean_profile': None,  # 静态人格无法量化
            'memory_traces': []
        }


class OursGroupManager(BaseGroupManager):
    """
    Ours: PersonaMem 动态人格积累
    
    特点：
    - 使用 PersonaMem 系统
    - 人格随对话动态演化
    - 具备记忆检索能力
    """
    
    def __init__(
        self, 
        llm_client=None,
        persona_profile_path: str = None
    ):
        super().__init__('Ours', llm_client)
        
        # PersonaMem 组件
        self.persona_profile_path = persona_profile_path
        self.current_persona: Dict[str, float] = {
            'O': 0.5, 'C': 0.5, 'E': 0.5, 'A': 0.5, 'N': 0.5
        }
        self.memory_traces: List[Dict] = []
        self.conversation_history: List[Dict] = []
        
        # 加载已有的人格状态（如果存在）
        if persona_profile_path:
            self._load_state()
    
    def generate_system_prompt(self) -> str:
        """生成 PersonaMem 人格 prompt"""
        # 根据当前 persona 动态生成 prompt
        ocean_desc = self._get_ocean_description()
        
        return f"""你是一个具有特定人格的 AI 助手。你的核心人格特征如下：

{ocean_desc}

请在回答中体现这些人格特征。记住你的特点，保持一致的说话风格和价值观。"""
    
    def _get_ocean_description(self) -> str:
        """根据 OCEAN 分数生成描述"""
        descriptions = []
        
        # Openness
        if self.current_persona['O'] > 0.7:
            descriptions.append("你思维开放，喜欢尝试新事物，对未知充满好奇。")
        elif self.current_persona['O'] < 0.3:
            descriptions.append("你做事稳健，偏好已验证的方法和经验。")
        
        # Conscientiousness
        if self.current_persona['C'] > 0.7:
            descriptions.append("你做事认真负责，有很强的自律性。")
        elif self.current_persona['C'] < 0.3:
            descriptions.append("你比较随性，不喜欢被规则束缚。")
        
        # Extraversion
        if self.current_persona['E'] > 0.7:
            descriptions.append("你性格外向，喜欢与人交流。")
        elif self.current_persona['E'] < 0.3:
            descriptions.append("你比较内敛，偏好独处或小范围交流。")
        
        # Agreeableness
        if self.current_persona['A'] > 0.7:
            descriptions.append("你待人和善，重视和谐的人际关系。")
        elif self.current_persona['A'] < 0.3:
            descriptions.append("你比较直接，有时会直言不讳。")
        
        # Neuroticism (注意：分数高表示情绪不稳定)
        if self.current_persona['N'] < 0.3:
            descriptions.append("你情绪稳定，面对压力能保持冷静。")
        elif self.current_persona['N'] > 0.7:
            descriptions.append("你比较敏感，情绪波动较大。")
        
        return " ".join(descriptions) if descriptions else "你是一个理性、务实的人。"
    
    async def chat(self, messages: List[Dict]) -> str:
        """PersonaMem 增强对话"""
        if self.llm is None:
            raise ValueError("LLM client is required for Ours")
        
        # 记录对话历史
        for msg in messages:
            self.conversation_history.append(msg)
        
        system_msg = {"role": "system", "content": self.generate_system_prompt()}
        all_messages = [system_msg] + messages
        
        response = await self.llm.achat(all_messages)
        
        # 更新人格状态
        self._update_persona(response)
        
        return response
    
    def _update_persona(self, response: str):
        """从响应中更新人格（简化版）"""
        # 实际应该用 LLM 进行人格提取
        # 这里简化处理：记录响应
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # 更新 memory trace
        self.memory_traces.append({
            'timestamp': datetime.now().isoformat(),
            'response_length': len(response),
            'content': response[:200]  # 截断存储
        })
        
        # 保持 memory traces 在合理范围内
        if len(self.memory_traces) > 100:
            self.memory_traces = self.memory_traces[-100:]
    
    def get_persona_state(self) -> Dict[str, Any]:
        """获取当前人格状态"""
        return {
            'has_persona': True,
            'is_static': False,
            'ocean_profile': self.current_persona.copy(),
            'memory_traces': self.memory_traces.copy(),
            'conversation_count': len(self.conversation_history) // 2
        }
    
    def _load_state(self):
        """加载已有状态"""
        try:
            import os
            if os.path.exists(self.persona_profile_path):
                with open(self.persona_profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.current_persona = data.get('ocean_profile', self.current_persona)
                    self.memory_traces = data.get('memory_traces', [])
        except Exception:
            pass  # 忽略加载错误
    
    def save_state(self):
        """保存当前状态"""
        if self.persona_profile_path:
            data = {
                'ocean_profile': self.current_persona,
                'memory_traces': self.memory_traces,
                'saved_at': datetime.now().isoformat()
            }
            import os
            os.makedirs(os.path.dirname(self.persona_profile_path), exist_ok=True)
            with open(self.persona_profile_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


class GroupManagerFactory:
    """实验组工厂"""
    
    @staticmethod
    def create_bl1(llm_client=None) -> BL1GroupManager:
        return BL1GroupManager(llm_client)
    
    @staticmethod
    def create_bl2(
        llm_client=None,
        persona_description: str = None
    ) -> BL2GroupManager:
        return BL2GroupManager(llm_client, persona_description)
    
    @staticmethod
    def create_ours(
        llm_client=None,
        persona_profile_path: str = None
    ) -> OursGroupManager:
        return OursGroupManager(llm_client, persona_profile_path)
    
    @staticmethod
    def create_all(
        llm_client=None,
        bl2_persona: str = None,
        ours_profile_path: str = None
    ) -> Dict[str, BaseGroupManager]:
        """创建所有实验组"""
        return {
            'BL-1': GroupManagerFactory.create_bl1(llm_client),
            'BL-2': GroupManagerFactory.create_bl2(llm_client, bl2_persona),
            'Ours': GroupManagerFactory.create_ours(llm_client, ours_profile_path)
        }
