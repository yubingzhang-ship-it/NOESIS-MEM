"""
深化进程（Deepener）

设计还原：NOESIS设计文档 4.1 节 "深化期"
"""

import sqlite3
import os
import datetime
import json
import requests
import warnings
from typing import Dict, List, Optional

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

# Ollama 本地 LLM
try:
    from noesis_ii.llm import OllamaLLM, is_ollama_available
except ImportError:
    OllamaLLM = None
    is_ollama_available = lambda: False


class Deepener:
    """
    深化进程

    从高权重记忆中提取深层模式，更新人格系统。
    类比：睡眠中的记忆深化和模式识别。
    """

    def __init__(self, db_path: str, llm_config: Dict = None):
        self.db_path = db_path
        self.llm_config = llm_config or {}

        # 旧架构组件（可选）
        self.long_term_memory = LongTermMemory(db_path) if LongTermMemory else None
        
        # 新架构组件（可选）
        self.persona_profile = PersonaProfile(db_path) if PersonaProfile else None

        # 初始化 Ollama 本地 LLM（用于轻量级分析，节省成本）
        self._ollama = None
        self._init_ollama()

    def _init_ollama(self):
        """初始化 Ollama 本地 LLM"""
        if OllamaLLM and is_ollama_available():
            try:
                self._ollama = OllamaLLM(model='deepseek-r1:1.5b')
                print("[DEEPENER] Ollama available, using deepseek-r1:1.5b for lightweight analysis")
            except Exception as e:
                print(f"[DEEPENER] Ollama init failed: {e}")
                self._ollama = None
        else:
            print("[DEEPENER] Ollama not available, will use remote LLM only")

    def run(self) -> Dict:
        """运行深化进程"""
        results = {}

        # 1. 分析高权重节点
        print("[DEEPENER] Phase 1: Analyzing high-weight nodes...")
        high_weight_nodes = self._analyze_high_weight_nodes()

        # 2. 提取统计模式
        print("[DEEPENER] Phase 2: Statistical pattern extraction...")
        patterns = self._extract_statistical_patterns(high_weight_nodes)

        # 3. LLM 深度分析（如果可用）
        llm_analysis = None
        if high_weight_nodes and self.llm_config.get('api_key'):
            print("[DEEPENER] Phase 3: LLM deep analysis...")
            analysis_content = self._prepare_analysis_content(high_weight_nodes)
            llm_analysis = self._analyze_with_llm(analysis_content)

        # 4. 生成/更新人格（新架构）
        print("[DEEPENER] Phase 4: Personality emergence...")
        personality = self._generate_personality()

        # 5. 汇总
        results = {
            'high_weight_nodes_count': len(high_weight_nodes),
            'patterns': patterns,
            'personality': personality,
            'llm_analysis': llm_analysis,
            'timestamp': datetime.datetime.now().isoformat()
        }

        print(f"[DEEPENER] Deepening complete: "
              f"{len(high_weight_nodes)} nodes, "
              f"{len(patterns)} patterns")

        return results
    
    def _generate_personality(self) -> Dict:
        """生成人格（新架构）"""
        if self.persona_profile:
            try:
                persona = self.persona_profile.get_current_persona()
                return {
                    'status': 'ok',
                    'openness': persona.openness,
                    'conscientiousness': persona.conscientiousness,
                    'extraversion': persona.extraversion,
                    'agreeableness': persona.agreeableness,
                    'neuroticism': persona.neuroticism,
                    'source': 'persona_profile'
                }
            except Exception as e:
                print(f"[DEEPENER] PersonaProfile error: {e}")
        
        # 降级处理
        return {
            'status': 'fallback',
            'message': 'PersonaProfile not available',
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }

    def _analyze_high_weight_nodes(self, limit: int = 100) -> List[Dict]:
        """分析高权重节点"""
        high_weight_nodes = self.long_term_memory.get_all_nodes(limit)

        analysis = []
        for node in high_weight_nodes:
            node_id = node['id']
            node_details = self.long_term_memory.get_node(node_id)

            link_strength = 0
            if node_details and 'links' in node_details:
                for link in node_details['links']['outgoing']:
                    link_strength += link.get('strength', 0)
                for link in node_details['links']['incoming']:
                    link_strength += link.get('strength', 0)

            analysis.append({
                'id': node['id'],
                'content': node['content'],
                'weight': node['weight'],
                'link_strength': link_strength,
                'last_accessed': node.get('last_accessed', ''),
                'created_at': node.get('created_at', '')
            })

        analysis.sort(key=lambda x: (x['weight'], x['link_strength']), reverse=True)
        return analysis

    def _extract_statistical_patterns(self, high_weight_nodes: List[Dict]) -> List[Dict]:
        """提取统计模式（增强版）"""
        patterns = []

        if not high_weight_nodes:
            return patterns

        # 1. 权重分布模式
        weights = [n['weight'] for n in high_weight_nodes]
        avg_weight = sum(weights) / len(weights)
        max_weight = max(weights) if weights else 0
        min_weight = min(weights) if weights else 0

        patterns.append({
            'type': 'weight_distribution',
            'average': round(avg_weight, 3),
            'max': round(max_weight, 3),
            'min': round(min_weight, 3),
            'std': round(self._std_dev(weights), 3),
            'description': f'权重分布：均值={avg_weight:.2f}, 范围=[{min_weight:.2f}, {max_weight:.2f}]'
        })

        # 2. 关联网络密度
        avg_link = sum(n['link_strength'] for n in high_weight_nodes) / len(high_weight_nodes)
        strongly_linked = sum(1 for n in high_weight_nodes if n['link_strength'] > 1.0)
        patterns.append({
            'type': 'network_density',
            'average_link_strength': round(avg_link, 3),
            'strongly_linked_count': strongly_linked,
            'description': f'关联密度：平均关联强度={avg_link:.2f}, 强关联节点={strongly_linked}'
        })

        # 3. 内容长度分布
        content_lengths = [len(n['content']) for n in high_weight_nodes]
        avg_length = sum(content_lengths) / len(content_lengths)
        patterns.append({
            'type': 'content_length',
            'average': round(avg_length, 1),
            'description': f'平均内容长度：{avg_length:.0f} 字符'
        })

        # 4. 时间分布
        time_patterns = self._analyze_time_patterns(high_weight_nodes)
        patterns.extend(time_patterns)

        # 5. 主题聚类（简化版：基于关键词共现）
        topic_patterns = self._analyze_topic_clusters(high_weight_nodes)
        patterns.extend(topic_patterns)

        return patterns

    def _analyze_time_patterns(self, nodes: List[Dict]) -> List[Dict]:
        """分析时间模式"""
        patterns = []
        time_data = []

        for node in nodes:
            created = node.get('created_at', '')
            if created:
                try:
                    t = datetime.datetime.fromisoformat(created)
                    time_data.append(t)
                except (ValueError, TypeError):
                    continue

        if len(time_data) >= 2:
            latest = max(time_data)
            oldest = min(time_data)
            span_days = (latest - oldest).days

            # 按小时分布
            hour_counts = [0] * 24
            for t in time_data:
                hour_counts[t.hour] += 1

            peak_hour = hour_counts.index(max(hour_counts))
            patterns.append({
                'type': 'temporal_distribution',
                'span_days': span_days,
                'peak_hour': peak_hour,
                'description': f'时间跨度：{span_days} 天，活跃高峰：{peak_hour}:00'
            })

        return patterns

    def _analyze_topic_clusters(self, nodes: List[Dict]) -> List[Dict]:
        """分析主题聚类（简化版）"""
        import re
        from collections import Counter

        # 提取所有关键词
        all_keywords = []
        for node in nodes:
            text = node.get('content', '')
            words = re.findall(r'[\u4e00-\u9fff]{2,4}|[a-zA-Z]{3,}', text)
            all_keywords.extend([w.lower() for w in words])

        if not all_keywords:
            return []

        # 统计高频关键词
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(8)

        return [{
            'type': 'topic_clusters',
            'top_keywords': [{'word': w, 'count': c} for w, c in top_keywords],
            'description': f'高频主题：{", ".join(f"{w}({c})" for w, c in top_keywords[:5])}'
        }]

    def _prepare_analysis_content(self, nodes: List[Dict]) -> str:
        """准备 LLM 分析内容"""
        sections = []

        for i, node in enumerate(nodes[:15], 1):
            content = node.get('content', '')[:200]
            weight = node.get('weight', 0)
            sections.append(f"{i}. [权重:{weight:.2f}] {content}")

        return "\n".join(sections)

    def _analyze_with_llm(self, content: str) -> Optional[str]:
        """
        使用 LLM 分析内容（优先 Ollama，节省成本）

        分工原则：
        - Ollama（本地）：日常深化分析，零成本
        - LongCat（云端）：高权重节点深度分析，需要强推理能力
        """
        # 优先使用 Ollama 本地模型
        if self._ollama is not None:
            return self._analyze_with_ollama(content)

        # 回退到远程 LLM
        api_key = self.llm_config.get('api_key')
        api_base = self.llm_config.get('api_base', 'https://api.longcat.chat/openai/v1')
        model = self.llm_config.get('model', 'LongCat-Flash-Lite')

        if not api_key:
            return None

        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }

            prompt = f"""请分析以下高权重记忆节点，完成以下任务：
1. 识别主要主题和模式
2. 分析知识结构和关联
3. 评估思维特征
4. 提取核心价值观倾向
5. 生成人格特征摘要

记忆节点：
{content}"""

            data = {
                'model': model,
                'messages': [
                    {
                        'role': 'system',
                        'content': '你是一个专业的记忆分析专家，能够从记忆内容中提取深层模式和人格特征。'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.3,
                'max_tokens': 1500
            }

            print("[DEEPENER] Sending LLM request for deep analysis...")
            response = requests.post(f'{api_base}/chat/completions',
                                   headers=headers, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()
            analysis = result['choices'][0]['message']['content']
            print("[DEEPENER] LLM deep analysis done")
            return analysis

        except Exception as e:
            print(f"[DEEPENER] LLM analysis failed: {e}")
            return None

    def _analyze_with_ollama(self, content: str) -> Optional[str]:
        """使用 Ollama 本地模型进行轻量级深度分析"""
        import re

        # 清理内容
        content = re.sub(r'[\x00-\x1F\x7F]', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        max_content_length = 3000  # Ollama 模型上下文较短
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        system_prompt = "你是一个专业的记忆分析专家，能够从记忆内容中提取深层模式和人格特征。"

        prompt = f"""请分析以下高权重记忆节点，完成以下任务：
1. 识别主要主题和模式
2. 分析知识结构和关联
3. 评估思维特征
4. 提取核心价值观倾向
5. 生成人格特征摘要

记忆节点：
{content}"""

        try:
            print("[DEEPENER] Using Ollama for deep analysis...")
            result = self._ollama.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens=1200,
            )
            print("[DEEPENER] Ollama analysis done")
            return result.strip() if result else None

        except Exception as e:
            print(f"[DEEPENER] Ollama analysis failed: {e}, falling back to remote LLM")
            # 回退到远程 LLM
            return self._analyze_with_remote_llm(content)

    def _analyze_with_remote_llm(self, content: str) -> Optional[str]:
        """回退到远程 LLM（LongCat）"""
        api_key = self.llm_config.get('api_key')
        api_base = self.llm_config.get('api_base', 'https://api.longcat.chat/openai/v1')
        model = self.llm_config.get('model', 'LongCat-Flash-Lite')

        if not api_key:
            return None

        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }

            prompt = f"""请分析以下高权重记忆节点，完成以下任务：
1. 识别主要主题和模式
2. 分析知识结构和关联
3. 评估思维特征
4. 提取核心价值观倾向
5. 生成人格特征摘要

记忆节点：
{content}"""

            data = {
                'model': model,
                'messages': [
                    {
                        'role': 'system',
                        'content': '你是一个专业的记忆分析专家，能够从记忆内容中提取深层模式和人格特征。'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.3,
                'max_tokens': 1500
            }

            print("[DEEPENER] Falling back to remote LLM...")
            response = requests.post(f'{api_base}/chat/completions',
                                   headers=headers, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()
            analysis = result['choices'][0]['message']['content']
            print("[DEEPENER] Remote LLM analysis done")
            return analysis

        except Exception as e:
            print(f"[DEEPENER] Remote LLM analysis failed: {e}")
            return None
            return None

    def _std_dev(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        return variance ** 0.5

    def update_personality(self) -> Dict:
        """更新人格"""
        return self._generate_personality()

    def get_personality_summary(self) -> str:
        """获取人格摘要"""
        persona = self._generate_personality()
        if persona.get('status') == 'ok':
            return f"OCEAN: O={persona['openness']:.2f}, C={persona['conscientiousness']:.2f}, " \
                   f"E={persona['extraversion']:.2f}, A={persona['agreeableness']:.2f}, " \
                   f"N={persona['neuroticism']:.2f}"
        return "Personality not available"

    def get_high_weight_nodes(self, limit: int = 20) -> List[Dict]:
        """获取高权重节点"""
        nodes = self._analyze_high_weight_nodes(limit)
        return nodes

    def analyze_memory_patterns(self) -> List[Dict]:
        """分析记忆模式"""
        high_weight_nodes = self._analyze_high_weight_nodes()
        return self._extract_statistical_patterns(high_weight_nodes)
