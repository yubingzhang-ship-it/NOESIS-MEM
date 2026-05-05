"""
重放引擎（Replay Engine）

设计文档定义的睡眠/静止期多阶段回放机制：
- NREM 阶段：正向重放，巩固记忆（权重强化 + 关联加强）
- REM 阶段：反向重放支持规划
- 微觉醒整合：创造性联想 + 系统状态更新
- 情绪安全重放：对高情绪记忆进行安全处理

设计还原：NOESIS设计文档 4.1 节 "睡眠/静止期"
"""

import sqlite3
import os
import datetime
import random
import math
import warnings
from typing import List, Dict, Optional, Tuple

# 旧架构模块（可选导入）
try:
    from noesis_ii.core.long_term_memory import LongTermMemory
except ImportError as e:
    warnings.warn(f"Legacy modules not available: {e}")
    LongTermMemory = None

# 新架构模块（可选导入）
try:
    from noesis_ii.core.persona_profile import PersonaProfile
except ImportError:
    PersonaProfile = None


class ReplayEngine:
    """
    重放引擎：记忆的自我优化机制

    类比：睡眠中的记忆巩固 + REM 创造性联想

    回放模式：
    - forward（正向）：巩固记忆痕迹，强化关联
    - reverse（反向）：从目标反向追溯，支持规划
    - emotional（情绪）：安全重放高情绪记忆，配合对治种子
    - creative（创造性）：跨记忆联想，生成新关联
    - full（完整）：NREM → REM → 微觉醒 完整周期
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.long_term_memory = LongTermMemory(db_path) if LongTermMemory else None
        self.persona_profile = PersonaProfile(db_path) if PersonaProfile else None

    def run(self, mode: str = 'full', limit: int = 10) -> Dict:
        """
        运行重放引擎

        Args:
            mode: 回放模式 (forward/reverse/emotional/creative/full)
            limit: 每阶段处理记忆数

        Returns:
            回放结果摘要
        """
        results = {
            'mode': mode,
            'stages': {},
            'total_processed': 0,
            'timestamp': datetime.datetime.now().isoformat()
        }

        if mode == 'full':
            # 完整睡眠周期
            results['stages']['nrem'] = self._stage_nrem(limit)
            results['stages']['rem'] = self._stage_rem(limit)
            results['stages']['micro_awakening'] = self._stage_micro_awakening(limit)
        elif mode == 'forward':
            results['stages']['nrem'] = self._stage_nrem(limit)
        elif mode == 'reverse':
            results['stages']['rem'] = self._stage_rem(limit)
        elif mode == 'emotional':
            results['stages']['emotional'] = self._stage_emotional_regulation(limit)
        elif mode == 'creative':
            results['stages']['creative'] = self._stage_creative_association(limit)
        else:
            # 默认正向
            results['stages']['nrem'] = self._stage_nrem(limit)

        results['total_processed'] = sum(
            s.get('processed', 0) for s in results['stages'].values()
        )

        print(f"[REPLAY] Replay done: mode={mode}, processed={results['total_processed']}")
        return results

    def _stage_nrem(self, limit: int) -> Dict:
        """
        NREM 阶段：正向重放巩固

        按时间顺序重放记忆，强化记忆痕迹和关联。
        类比：海马体向新皮层的信息传输。
        """
        # 选择记忆：优先中高权重、较久未访问的
        all_nodes = self.long_term_memory.get_all_nodes(100)
        if not all_nodes:
            return {'processed': 0, 'details': []}

        # 按权重排序，选择中等权重的进行巩固（高权重已足够强）
        selected = sorted(all_nodes, key=lambda x: x['weight'])[:limit]

        details = []
        for node in selected:
            try:
                # 正向重放：访问节点强化权重
                self.long_term_memory.access_node(node['id'])

                # 强化关联
                links = node.get('links', {})
                out_links = links.get('outgoing', [])
                for link in out_links:
                    self.long_term_memory.access_node(link.get('target_node_id', 0))

                details.append({
                    'memory_id': node['id'],
                    'content': node['content'][:50],
                    'weight_before': node['weight'],
                    'stage': 'NREM'
                })
            except Exception as e:
                details.append({
                    'memory_id': node.get('id'),
                    'error': str(e)
                })

        print(f"[REPLAY-NREM] Forward replay: {len(details)} memories consolidated")
        return {'processed': len(details), 'details': details}

    def _stage_rem(self, limit: int) -> Dict:
        """
        REM 阶段：反向重放 + 对治种子强化

        反向重放：从最近记忆向早期追溯，支持规划和创造性整合。
        同时强化对治种子，处理负面记忆。
        """
        all_nodes = self.long_term_memory.get_all_nodes(100)
        if not all_nodes:
            return {'processed': 0, 'details': []}

        # 反向选择：按创建时间倒序（从新到旧）
        selected = sorted(
            all_nodes,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )[:limit]

        details = []
        counteract_results = []

        for node in selected:
            try:
                # 反向重放：仍然访问节点但记录为反向
                self.long_term_memory.access_node(node['id'])

                # 检查是否有需要对治的种子
                counteract = self._check_and_counteract(node)
                if counteract:
                    counteract_results.append(counteract)

                details.append({
                    'memory_id': node['id'],
                    'content': node['content'][:50],
                    'stage': 'REM',
                    'counteracted': counteract is not None
                })
            except Exception as e:
                details.append({
                    'memory_id': node.get('id'),
                    'error': str(e)
                })

        print(f"[REPLAY-REM] Reverse replay: {len(details)} memories, "
              f"{len(counteract_results)} counteracted")

        return {
            'processed': len(details),
            'counteracted': len(counteract_results),
            'details': details
        }

    def _stage_emotional_regulation(self, limit: int) -> Dict:
        """
        情绪调节阶段：安全重放高情绪记忆

        对高权重（可能是高情绪强度）记忆进行安全处理，
        配合对治种子，降低负面记忆的过度影响。
        """
        all_nodes = self.long_term_memory.get_all_nodes(100)

        # 选择高权重记忆（可能是高情绪记忆）
        emotional_nodes = [n for n in all_nodes if n.get('weight', 0) > 0.7]
        if not emotional_nodes:
            emotional_nodes = all_nodes[:limit]

        selected = emotional_nodes[:limit]
        details = []

        for node in selected:
            try:
                # 安全重放：轻微访问（不大幅强化）
                self.long_term_memory.access_node(node['id'])

                # 情绪安全处理
                safe_result = self._safe_emotional_replay(node)

                # 对治强化
                counteract = self._check_and_counteract(node)

                details.append({
                    'memory_id': node['id'],
                    'content': node['content'][:50],
                    'weight': node['weight'],
                    'safe_replay': safe_result,
                    'counteracted': counteract is not None
                })
            except Exception as e:
                details.append({
                    'memory_id': node.get('id'),
                    'error': str(e)
                })

        safe_count = sum(1 for d in details if d.get('safe_replay'))
        print(f"[REPLAY-EMOTIONAL] Emotional regulation: {len(details)} memories, "
              f"{safe_count} safely replayed")

        return {
            'processed': len(details),
            'safe_replays': safe_count,
            'details': details
        }

    def _stage_creative_association(self, limit: int) -> Dict:
        """
        创造性联想阶段：跨记忆关联

        发现记忆间的非显性关联，生成新的概念连接。
        """
        all_nodes = self.long_term_memory.get_all_nodes(50)
        if len(all_nodes) < 2:
            return {'processed': 0, 'associations': []}

        # 随机选取记忆对，检测潜在关联
        associations = []
        attempts = min(limit * 3, len(all_nodes) * 2)

        for _ in range(attempts):
            pair = random.sample(all_nodes, min(2, len(all_nodes)))
            if len(pair) < 2:
                continue

            node_a, node_b = pair
            if node_a['id'] == node_b['id']:
                continue

            # 计算简单关联度（共同关键词）
            similarity = self._simple_similarity(
                node_a.get('content', ''),
                node_b.get('content', '')
            )

            if similarity > 0.2:
                # 检查是否已有关联
                links_a = node_a.get('links', {}).get('outgoing', [])
                existing_targets = [l.get('target_node_id') for l in links_a]

                if node_b['id'] not in existing_targets:
                    # 创建新关联
                    self.long_term_memory.create_link(
                        node_a['id'], node_b['id'],
                        strength=min(1.0, similarity)
                    )
                    associations.append({
                        'from_id': node_a['id'],
                        'from_content': node_a['content'][:30],
                        'to_id': node_b['id'],
                        'to_content': node_b['content'][:30],
                        'similarity': similarity
                    })

                if len(associations) >= limit:
                    break

        print(f"[REPLAY-CREATIVE] Creative associations: {len(associations)} new links")
        return {'processed': len(associations), 'associations': associations}

    def _stage_micro_awakening(self, limit: int) -> Dict:
        """
        微觉醒阶段：整合 + 系统状态更新

        在 NREM 和 REM 之后，进行整合性的检查和更新。
        """
        all_nodes = self.long_term_memory.get_all_nodes(100)

        # 统计信息
        total = len(all_nodes)
        high_weight = sum(1 for n in all_nodes if n.get('weight', 0) > 0.7)
        avg_weight = sum(n.get('weight', 0) for n in all_nodes) / max(total, 1)

        # 系统状态快照
        summary = {
            'total_memories': total,
            'high_weight_count': high_weight,
            'high_weight_ratio': high_weight / max(total, 1),
            'average_weight': round(avg_weight, 3),
            'stage': 'micro_awakening'
        }

        print(f"[REPLAY-MICRO] System snapshot: {total} memories, "
              f"avg_weight={avg_weight:.3f}")

        return {
            'processed': total,
            'system_snapshot': summary
        }

    def _safe_emotional_replay(self, memory: Dict) -> bool:
        """
        情绪记忆安全重放

        对高情绪记忆进行"淡化"处理：
        不删除也不大幅衰减，而是确保对治种子存在。
        """
        content = memory.get('content', '')

        # 检测负面情绪关键词
        negative_keywords = ['痛苦', '失败', '恐惧', '焦虑', '悲伤', '愤怒',
                            'pain', 'fail', 'fear', 'anxiety', 'sad', 'angry']
        negative_count = sum(content.count(kw) for kw in negative_keywords)

        if negative_count > 0:
            # 高情绪记忆：记录对治需求
            print(f"[REPLAY] Safe replay for emotional memory: "
                  f"{content[:30]}... (neg_score={negative_count})")
            return True

        return False

    def _check_and_counteract(self, memory: Dict) -> Optional[Dict]:
        """
        检查并处理负面记忆

        对于负面记忆，记录处理信息。
        """
        content = memory.get('content', '')

        # 简化检测：如果内容包含负面关键词
        negative_keywords = ['痛苦', '失败', '恐惧', '焦虑', '悲伤', '愤怒', '错误']
        negative_hits = sum(content.count(kw) for kw in negative_keywords)

        if negative_hits > 0:
            # 新架构：存储正向经验到 PersonaProfile
            if self.persona_profile:
                try:
                    trace_id = self.persona_profile.store_experience(
                        experience="面对挑战并成长的经历 - 将困难转化为学习机会",
                        trace_type='counteract',
                        intensity=0.6,
                        context={'source': 'replay_engine', 'target_memory': memory.get('id')}
                    )
                    return {
                        'counteract_trace_id': trace_id,
                        'target_memory_id': memory['id'],
                        'negative_score': negative_hits
                    }
                except Exception:
                    pass

        return None

    def _simple_similarity(self, text_a: str, text_b: str) -> float:
        """简单的文本相似度（基于公共词汇）"""
        import re

        words_a = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text_a.lower()))
        words_b = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text_b.lower()))

        if not words_a or not words_b:
            return 0.0

        common = words_a & words_b
        # Jaccard 相似度
        return len(common) / len(words_a | words_b)

    def replay_specific_memory(self, memory_id: int) -> Dict:
        """重放特定记忆"""
        memory = self.long_term_memory.get_node(memory_id)
        if not memory:
            return {'error': 'Memory not found'}

        self.long_term_memory.access_node(memory_id)
        return {
            'memory_id': memory_id,
            'content': memory.get('content', '')[:50],
            'weight': memory.get('weight', 0),
            'status': 'replayed'
        }

    def replay_by_topic(self, topic: str, limit: int = 5) -> List[Dict]:
        """按主题重放记忆"""
        memories = self.long_term_memory.retrieve(topic, limit)
        results = []

        for memory in memories:
            self.long_term_memory.access_node(memory['id'])
            results.append({
                'memory_id': memory['id'],
                'content': memory.get('content', '')[:50],
                'weight': memory.get('weight', 0)
            })

        return results

    def get_replay_statistics(self) -> Dict:
        """获取重放统计信息"""
        all_nodes = self.long_term_memory.get_all_nodes()
        total = len(all_nodes)

        high_weight = [n for n in all_nodes if n.get('weight', 0) > 0.7]
        low_weight = [n for n in all_nodes if n.get('weight', 0) < 0.3]

        return {
            'total_memories': total,
            'high_weight_memories': len(high_weight),
            'low_weight_memories': len(low_weight),
            'high_weight_ratio': len(high_weight) / max(total, 1),
            'low_weight_ratio': len(low_weight) / max(total, 1)
        }

    def schedule_replay(self, mode: str = 'full', limit: int = 10,
                       interval: int = 3600) -> Dict:
        """调度重放（返回调度信息供 Scheduler 使用）"""
        return {
            'mode': mode,
            'count': limit,
            'interval': interval,
            'status': 'scheduled'
        }
