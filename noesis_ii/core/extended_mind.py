"""
延展-耦合系统（Extended Mind）

实现延展心智和结构耦合理论（Clark & Chalmers 1998 + 自创生理论）。

核心概念：
- CoupledResource：可能成为系统一部分的外部资源
- 耦合强度 C = (可及性 × 可靠性 × 历史深度 × 功能整合)^0.25
- C > 0.8：构成性耦合（资源成为系统一部分，"回忆"而非"查询"）
- 0.3 < C < 0.8：工具性耦合（辅助工具）
- C < 0.3：无关资源

设计还原：NOESIS设计文档 3.8 节
"""

import sqlite3
import os
import json
import uuid
import time
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Callable, Optional, Any


@dataclass
class CoupledResource:
    """
    耦合资源：可能成为系统一部分的外部资源

    满足 Clark & Chalmers 的条件：
    1. 恒常可及（constant access）
    2. 自动认可（automatic endorsement）
    3. 功能整合（functional integration）
    """

    resource_id: str
    resource_type: str  # 'file', 'database', 'api', 'agent', 'url'
    name: str = ""
    description: str = ""

    # 耦合属性
    constant_access: bool = False
    automatic_endorsement: bool = False
    functional_integration: float = 0.0

    # 耦合历史（交互记录）
    coupling_history: List[Dict] = field(default_factory=list)
    trust_level: float = 0.5

    # 资源接口（可调用接口函数）
    interface: Optional[Callable] = None

    # 持久化字段
    added_at: str = ""
    last_accessed: str = ""

    def check_coupling_strength(self) -> float:
        """
        计算当前耦合强度

        C = f(可及性, 可靠性, 历史深度, 功能整合)
        使用几何平均（短板效应）：任一维度过低会严重拉低整体
        """
        # 可及性（最近使用时间衰减）
        recency_score = self._calculate_recency()

        # 可靠性（信任度）
        reliability_score = self.trust_level

        # 历史深度（交互次数）
        history_score = min(1.0, len(self.coupling_history) / 50.0)

        # 功能整合
        integration_score = self.functional_integration

        # 几何平均（短板效应）
        product = recency_score * reliability_score * history_score * integration_score

        if product <= 0:
            return 0.0

        coupling = product ** 0.25
        return coupling

    def _calculate_recency(self) -> float:
        """计算可及性得分（基于最近使用时间）"""
        if not self.coupling_history:
            return 0.1

        last_access = self.coupling_history[-1].get('timestamp', 0)
        if not last_access:
            return 0.1

        # 支持 float 时间戳和 ISO 字符串
        if isinstance(last_access, str):
            try:
                last_access = datetime.fromisoformat(last_access).timestamp()
            except (ValueError, TypeError):
                return 0.1

        hours_since = (time.time() - last_access) / 3600.0

        # 指数衰减：24小时内高可及性
        if hours_since < 1:
            return 1.0
        elif hours_since < 24:
            return 0.9 * math.exp(-hours_since / 24.0) + 0.1
        elif hours_since < 168:  # 1 week
            return 0.5 * math.exp(-(hours_since - 24) / 168.0) + 0.05
        else:
            return 0.05

    def record_interaction(self, success: bool, query_type: str = ""):
        """记录交互事件"""
        self.coupling_history.append({
            'timestamp': time.time(),
            'query_type': query_type,
            'success': success
        })

        # 更新信任度（贝叶斯更新）
        if success:
            # 成功：信任度缓慢上升
            self.trust_level = min(1.0, self.trust_level * 1.02 + 0.01)
        else:
            # 失败：信任度下降
            self.trust_level = max(0.05, self.trust_level * 0.9)

        self.last_accessed = datetime.now().isoformat()

        # 保持历史记录不超过 200 条
        if len(self.coupling_history) > 200:
            self.coupling_history = self.coupling_history[-100:]

    def get_coupling_type(self) -> str:
        """判断耦合类型"""
        strength = self.check_coupling_strength()
        if strength > 0.8:
            return 'constitutive'  # 构成性
        elif strength > 0.3:
            return 'instrumental'  # 工具性
        else:
            return 'irrelevant'  # 无关

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            'resource_id': self.resource_id,
            'resource_type': self.resource_type,
            'name': self.name,
            'description': self.description,
            'constant_access': self.constant_access,
            'automatic_endorsement': self.automatic_endorsement,
            'functional_integration': self.functional_integration,
            'trust_level': self.trust_level,
            'coupling_strength': self.check_coupling_strength(),
            'coupling_type': self.get_coupling_type(),
            'interaction_count': len(self.coupling_history),
            'added_at': self.added_at,
            'last_accessed': self.last_accessed
        }


class SocialCoupling:
    """社会耦合：交互记忆系统（Transactive Memory）"""

    def __init__(self, partner_id: str, partner_name: str = ""):
        self.partner_id = partner_id
        self.partner_name = partner_name or partner_id
        self.strength = 0.3  # 初始强度
        self.interactions: List[Dict] = []
        self.shared_domains: List[str] = []
        self.trust_level = 0.5

    def record_interaction(self, topic: str, quality: str = "neutral"):
        """记录社会交互"""
        self.interactions.append({
            'timestamp': time.time(),
            'topic': topic,
            'quality': quality
        })
        
        # 限制交互记录长度
        if len(self.interactions) > 200:
            self.interactions = self.interactions[-100:]

        # 更新强度
        if quality == 'positive':
            self.strength = min(1.0, self.strength + 0.05)
            self.trust_level = min(1.0, self.trust_level * 1.02 + 0.01)
        elif quality == 'negative':
            self.strength = max(0.05, self.strength - 0.03)
            self.trust_level = max(0.05, self.trust_level * 0.95)

        # 更新共享领域
        if topic and topic not in self.shared_domains:
            self.shared_domains.append(topic)

        # 衰减
        self.strength *= 0.995

    def query_transactive_memory(self, query: str) -> Optional[Dict]:
        """查询伙伴的知识领域"""
        relevant = [d for d in self.shared_domains if d.lower() in query.lower()]
        if relevant and self.strength > 0.3:
            return {
                'partner': self.partner_name,
                'relevant_domains': relevant,
                'coupling_strength': self.strength
            }
        return None


class ExtendedMind:
    """
    延展心智系统

    记忆分布在大脑-身体-环境-社会网络中。
    系统边界是动态的，由耦合强度决定。
    """

    # 耦合构成判定阈值
    CONSTITUTION_THRESHOLD = 0.8
    INSTRUMENTAL_THRESHOLD = 0.3

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

        # 内存中的资源状态
        self.coupled_resources: Dict[str, CoupledResource] = {}
        self.social_couplings: Dict[str, SocialCoupling] = {}

        # 从数据库加载
        self._load_resources()

    def connect(self):
        """连接到数据库"""
        dir_name = os.path.dirname(self.db_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def _load_resources(self):
        """从数据库加载已保存的资源"""
        try:
            conn = self.connect()
            cursor = conn.cursor()

            # 加载外部资源
            try:
                cursor.execute('SELECT * FROM extended_resources ORDER BY added_at')
                for row in cursor.fetchall():
                    data = dict(row)
                    resource = CoupledResource(
                        resource_id=data['resource_id'],
                        resource_type=data.get('resource_type', 'unknown'),
                        name=data.get('name', ''),
                        description=data.get('description', ''),
                        constant_access=bool(data.get('constant_access', 0)),
                        automatic_endorsement=bool(data.get('automatic_endorsement', 0)),
                        functional_integration=data.get('functional_integration', 0.0),
                        trust_level=data.get('trust_level', 0.5),
                        added_at=data.get('added_at', ''),
                        last_accessed=data.get('last_accessed', '')
                    )

                    # 加载交互历史
                    history_json = data.get('coupling_history', '')
                    if history_json:
                        try:
                            resource.coupling_history = json.loads(history_json)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    self.coupled_resources[resource.resource_id] = resource
            except sqlite3.OperationalError:
                pass  # 表不存在

            self.close()
        except Exception:
            pass

    def establish_coupling(self, resource: Dict) -> str:
        """
        建立与新资源的耦合

        Args:
            resource: 资源描述字典，至少包含 resource_type 或 type
                      可选：name, description, constant_access, interface

        Returns:
            resource_id 字符串
        """
        resource_id = resource.get('resource_id') or str(uuid.uuid4())

        # 兼容旧接口：type → resource_type, content → description
        r_type = resource.get('resource_type') or resource.get('type', 'unknown')
        r_name = resource.get('name', '')
        r_desc = resource.get('description') or resource.get('content', '')

        coupled = CoupledResource(
            resource_id=resource_id,
            resource_type=r_type,
            name=r_name,
            description=r_desc,
            constant_access=resource.get('constant_access', False),
            automatic_endorsement=resource.get('automatic_endorsement', False),
            interface=resource.get('interface'),
            added_at=datetime.now().isoformat()
        )

        # 初始交互
        coupled.record_interaction(success=True, query_type='establish')

        self.coupled_resources[resource_id] = coupled

        # 持久化
        self._save_resource(coupled)

        return resource_id

    def check_coupling_strength(self, resource_id: str) -> float:
        """计算指定资源的耦合强度"""
        resource = self.coupled_resources.get(resource_id)
        if not resource:
            return 0.0
        return resource.check_coupling_strength()

    def remember(self, query: str, context: Dict = None) -> Dict:
        """
        回忆：整合内部和外部资源

        根据耦合强度决定资源的角色：
        - 构成性外部：视为"自己的记忆"
        - 工具性外部：视为"辅助工具"
        """
        results = {
            'internal': [],
            'constitutive_external': [],
            'instrumental_external': [],
            'social': [],
            'confidence': 0.0
        }

        # 1. 外部资源查询
        for rid, resource in self.coupled_resources.items():
            coupling = resource.check_coupling_strength()

            if coupling > self.CONSTITUTION_THRESHOLD:
                # 构成性耦合 → 视为"回忆"
                result = self._retrieve_from_resource(resource, query)
                if result:
                    result['retrieval_type'] = 'constitutive'
                    result['coupling_strength'] = coupling
                    results['constitutive_external'].append(result)
            elif coupling > self.INSTRUMENTAL_THRESHOLD:
                # 工具性耦合 → 视为"查询"
                result = self._retrieve_from_resource(resource, query)
                if result:
                    result['retrieval_type'] = 'instrumental'
                    result['coupling_strength'] = coupling
                    results['instrumental_external'].append(result)

        # 2. 社会耦合查询
        for partner_id, coupling in self.social_couplings.items():
            if coupling.strength > 0.5:
                social_memory = coupling.query_transactive_memory(query)
                if social_memory:
                    results['social'].append(social_memory)

        # 3. 计算整体置信度
        all_results = (results['constitutive_external'] +
                      results['instrumental_external'] +
                      results['social'])
        if all_results:
            results['confidence'] = max(
                r.get('coupling_strength', 0.3) for r in all_results
            )

        # 兼容旧接口：external = constitutive + instrumental 合并
        results['external'] = results['constitutive_external'] + results['instrumental_external']
        results['internal'] = []  # ExtendedMind 不直接管理内部记忆

        return results

    def _retrieve_from_resource(self, resource: CoupledResource, query: str) -> Optional[Dict]:
        """从资源检索信息"""
        # 如果资源有接口函数，调用它
        if resource.interface:
            try:
                import signal
                # 设置超时保护（10秒）
                def timeout_handler(signum, frame):
                    raise TimeoutError("Resource retrieval timed out")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                try:
                    result = resource.interface(query)
                    resource.record_interaction(success=result is not None, query_type='retrieve')
                    if result:
                        return {
                            'resource_id': resource.resource_id,
                            'name': resource.name,
                            'content': result,
                            'source': 'external'
                        }
                finally:
                    signal.alarm(0)  # 取消超时
            except (TimeoutError, Exception):
                resource.record_interaction(success=False, query_type='retrieve')

        # 否则用名称/描述做简单匹配
        if query.lower() in resource.name.lower() or query.lower() in resource.description.lower():
            resource.record_interaction(success=True, query_type='match')
            return {
                'resource_id': resource.resource_id,
                'name': resource.name,
                'content': resource.description,
                'source': 'external'
            }

        resource.record_interaction(success=False, query_type='no_match')
        return None

    def record_social_coupling(self, partner_id: str, interaction: str,
                       quality: str = "neutral", topic: str = "") -> Dict:
        """
        社会耦合：记录与其他智能体的交互

        Args:
            partner_id: 伙伴标识
            interaction: 交互内容
            quality: 交互质量（positive/negative/neutral）
            topic: 交互主题
        """
        if partner_id not in self.social_couplings:
            self.social_couplings[partner_id] = SocialCoupling(partner_id)

        coupling = self.social_couplings[partner_id]
        coupling.record_interaction(topic or interaction, quality)

        return {
            'partner': coupling.partner_name,
            'coupling_strength': coupling.strength,
            'trust_level': coupling.trust_level,
            'shared_domains': coupling.shared_domains
        }

    def update_coupling_strength(self, resource_id: str, interaction_success: bool,
                                query_type: str = "") -> float:
        """更新资源的耦合强度（通过记录交互）"""
        resource = self.coupled_resources.get(resource_id)
        if not resource:
            return 0.0

        resource.record_interaction(interaction_success, query_type)
        self._save_resource(resource)
        return resource.check_coupling_strength()

    def remove_external_resource(self, resource_id: str) -> bool:
        """移除外部资源"""
        existed = resource_id in self.coupled_resources
        if existed:
            del self.coupled_resources[resource_id]

        deleted = False
        try:
            conn = self.connect()
            try:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM extended_resources WHERE resource_id = ?', (resource_id,))
                conn.commit()
                deleted = cursor.rowcount > 0
            finally:
                self.close()
        except Exception:
            pass

        return existed or deleted

    def get_external_resources(self) -> List[Dict]:
        """获取所有外部资源（含耦合强度）"""
        return [r.to_dict() for r in self.coupled_resources.values()]

    def get_coupling_strengths(self) -> Dict[str, float]:
        """获取所有资源的耦合强度"""
        return {
            rid: r.check_coupling_strength()
            for rid, r in self.coupled_resources.items()
        }

    def get_constitutive_resources(self) -> List[Dict]:
        """获取构成性耦合资源（C > 0.8）"""
        return [
            r.to_dict() for r in self.coupled_resources.values()
            if r.check_coupling_strength() > self.CONSTITUTION_THRESHOLD
        ]

    def get_instrumental_resources(self) -> List[Dict]:
        """获取工具性耦合资源（0.3 < C < 0.8）"""
        return [
            r.to_dict() for r in self.coupled_resources.values()
            if self.INSTRUMENTAL_THRESHOLD < r.check_coupling_strength() <= self.CONSTITUTION_THRESHOLD
        ]

    def _save_resource(self, resource: CoupledResource):
        """持久化资源到数据库"""
        try:
            conn = self.connect()
            try:
                cursor = conn.cursor()

                history_json = json.dumps([
                    {**h, 'timestamp': h.get('timestamp', 0)}
                    for h in resource.coupling_history[-50:]  # 只保存最近50条
                ])

                cursor.execute('''
                INSERT OR REPLACE INTO extended_resources
                (resource_id, resource_type, name, description,
                 constant_access, automatic_endorsement, functional_integration,
                 trust_level, coupling_history, added_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    resource.resource_id,
                    resource.resource_type,
                    resource.name,
                    resource.description,
                    int(resource.constant_access),
                    int(resource.automatic_endorsement),
                    resource.functional_integration,
                    resource.trust_level,
                    history_json,
                    resource.added_at,
                    resource.last_accessed
                ))

                conn.commit()
            finally:
                self.close()
        except Exception as e:
            print(f"[EXTENDED_MIND] Save resource failed: {e}")

    def get_social_couplings(self) -> List[Dict]:
        """获取所有社会耦合"""
        return [
            {
                'partner_id': c.partner_id,
                'partner_name': c.partner_name,
                'strength': c.strength,
                'trust_level': c.trust_level,
                'shared_domains': c.shared_domains,
                'interaction_count': len(c.interactions)
            }
            for c in self.social_couplings.values()
        ]
