import sqlite3
import os
import datetime
import requests
import json
import re
import warnings

# 旧架构模块（可选导入）
try:
    from noesis_ii.core.working_memory import WorkingMemory
    from noesis_ii.core.long_term_memory import LongTermMemory
except ImportError as e:
    warnings.warn(f"Legacy modules not available: {e}")
    WorkingMemory = None
    LongTermMemory = None

# 新架构模块（可选导入）
try:
    from noesis_ii.core.persona_profile import PersonaProfile
except ImportError:
    PersonaProfile = None

from noesis_ii.config_loader import ConfigLoader


class Consolidator:
    def __init__(self, db_path, llm_config=None):
        self.db_path = db_path
        
        # 旧架构组件（可选）
        self.working_memory = WorkingMemory(db_path) if WorkingMemory else None
        self.long_term_memory = LongTermMemory(db_path) if LongTermMemory else None
        
        # 新架构组件（可选）
        self.persona_profile = PersonaProfile(db_path) if PersonaProfile else None

        # 初始化 Ollama 本地 LLM（用于轻量级任务）
        self._ollama = None
        self._init_ollama()

        # 加载LLM配置
        if llm_config:
            self.llm_config = llm_config
        else:
            # 加载默认配置
            try:
                config_loader = ConfigLoader('config/default_config.yaml')
                config = config_loader.load()
                self.llm_config = config.get('llm', {})
            except Exception:
                self.llm_config = {}

    def _init_ollama(self):
        """初始化 Ollama 本地 LLM"""
        try:
            from noesis_ii.llm import OllamaLLM, is_ollama_available
            if is_ollama_available():
                self._ollama = OllamaLLM(model='deepseek-r1:1.5b')
                print("[CONSOLIDATOR] Ollama available, using deepseek-r1:1.5b for lightweight tasks")
            else:
                print("[CONSOLIDATOR] Ollama not available, will use remote LLM only")
        except Exception as e:
            print(f"[CONSOLIDATOR] Ollama init failed: {e}")
            self._ollama = None
    
    def run(self, limit=3, batch_size=3, max_workers=3, async_mode=False):
        """运行整理进程（支持并发 + 批量分析，避免超时）
        
        Args:
            limit:       本次运行最多处理条目数
            batch_size:  每批处理的条目数（控制单次LLM调用负载）
            max_workers: 并发工作线程数（默认3）
            async_mode:  True=后台线程异步执行（立即返回 None），False=阻塞等待
        
        Returns:
            async_mode=False: 成功处理的条目数 (int)
            async_mode=True:  None（后台执行，不阻塞）
        """
        if async_mode:
            import threading
            t = threading.Thread(
                target=self._run_sync,
                args=(limit, batch_size, max_workers),
                daemon=True,
                name="consolidator-async"
            )
            t.start()
            print(f"[CONSOLIDATOR] Async started: limit={limit}, max_workers={max_workers}")
            return None
        return self._run_sync(limit, batch_size, max_workers)

    def _run_sync(self, limit=3, batch_size=3, max_workers=3):
        """同步执行整理（供 run() 和异步线程共用）"""
        import concurrent.futures
        import time

        # 检查旧架构组件是否可用
        if not self.working_memory:
            print("[CONSOLIDATOR] WorkingMemory not available, skipping consolidation")
            return 0
        
        # 运行软删除遗忘调度（每次 consolidate 自动检查）
        self._run_soft_forget()
        
        # 获取待整理的工作记忆条目
        pending_entries = self.working_memory.get_pending(limit)
        
        if not pending_entries:
            print("[CONSOLIDATOR] No pending working memories")
            return 0

        total = len(pending_entries)
        print(f"[CONSOLIDATOR] Processing {total} entries (max_workers={max_workers})")
        t_start = time.time()

        consolidated_count = 0
        failed_entries = []

        # 并发处理（每个 entry 独立线程，最多 max_workers 个并发）
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers,
                                                    thread_name_prefix="consolidator") as pool:
            future_to_entry = {
                pool.submit(self._consolidate_entry, entry): entry
                for entry in pending_entries
            }
            for future in concurrent.futures.as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    success = future.result(timeout=120)  # 单条最多 120 秒
                    if success:
                        consolidated_count += 1
                        print(f"[CONSOLIDATOR] Done id={entry['id']} "
                              f"({consolidated_count}/{total})")
                    else:
                        failed_entries.append(entry['id'])
                except concurrent.futures.TimeoutError:
                    print(f"[CONSOLIDATOR] Timeout on entry id={entry['id']}")
                    failed_entries.append(entry['id'])
                except Exception as e:
                    print(f"[CONSOLIDATOR] Error on entry id={entry['id']}: {e}")
                    failed_entries.append(entry['id'])

        elapsed = time.time() - t_start
        if failed_entries:
            print(f"[CONSOLIDATOR] Failed entries: {failed_entries}")
        print(f"[CONSOLIDATOR] Done: {consolidated_count}/{total} in {elapsed:.1f}s "
              f"(avg {elapsed/max(total,1):.1f}s/entry)")
        return consolidated_count
    
    def _consolidate_entry(self, entry):
        """整理单条工作记忆（简化版 - 兼容新旧架构）"""
        content = entry['content']
        emotion = entry.get('emotion', 'neutral') or 'neutral'

        # 统一编码处理：确保UTF-8，处理emoji等特殊字符
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        content = content.encode('utf-8', errors='ignore').decode('utf-8')

        # 使用LLM分析内容，生成总结 + 结构化情绪
        analysis_result = self._analyze_with_llm(content)

        if not analysis_result:
            print("[CONSOLIDATOR] LLM analysis failed, using basic consolidation")
            summary = content[:200] if len(content) > 200 else content
            emotion_data = self._build_default_emotion(emotion)
        else:
            summary = analysis_result.get('summary', content[:200])
            # 提取结构化情绪数据
            raw_emotion = analysis_result.get('emotion', {})
            emotion_data = self._normalize_emotion(raw_emotion, emotion)
            print(f"[CONSOLIDATOR] Emotion: valence={emotion_data.get('valence', 0):.2f} "
                  f"arousal={emotion_data.get('arousal', 0):.2f} "
                  f"dominant={emotion_data.get('dominant', 'neutral')}")

        # OCEAN 人格分析（与内容分析并行，互补）
        ocean_data = self._analyze_persona(content, emotion)
        if ocean_data:
            print(f"[CONSOLIDATOR] OCEAN: O={ocean_data['openness']:.2f} C={ocean_data['conscientiousness']:.2f} "
                  f"E={ocean_data['extraversion']:.2f} A={ocean_data['agreeableness']:.2f} N={ocean_data['neuroticism']:.2f}")

        # 新架构：存储到 PersonaProfile
        if self.persona_profile:
            try:
                trace_id = self.persona_profile.store_experience(
                    experience=content,
                    trace_type='consolidated',
                    intensity=0.7 if emotion != 'neutral' else 0.5,
                    context={
                        'emotion': emotion,
                        'summary': summary,
                        'ocean': ocean_data,
                        'emotion_data': emotion_data,
                    }
                )
                print(f"[CONSOLIDATOR] Stored to PersonaProfile: trace_id={trace_id}")
            except Exception as e:
                print(f"[CONSOLIDATOR] PersonaProfile storage warning: {e}")

        # 旧架构：存储到 LTM（如果可用）
        if self.long_term_memory:
            try:
                # 创建节点
                node_id = self.long_term_memory.create_node(
                    summary,
                    node_type='consolidated',
                    weight=1.0
                )
                print(f"[CONSOLIDATOR] Stored to LTM: node_id={node_id}")
            except Exception as e:
                print(f"[CONSOLIDATOR] LTM storage warning: {e}")

        # 标记为已整理
        if self.working_memory:
            try:
                self.working_memory.mark_consolidated(entry['id'])
            except Exception as e:
                print(f"[CONSOLIDATOR] Mark consolidated warning: {e}")

        return True

    @staticmethod
    def _build_default_emotion(emotion: str) -> dict:
        """从简单情绪标签构建默认的 emotion_data 结构"""
        positive = {'joy', 'gratitude', 'love', 'happy', '快乐', '感恩', '喜悦', '兴奋', '满足'}
        negative = {'anger', 'fear', 'sad', '愤怒', '恐惧', '悲伤', '焦虑', '沮丧', '紧张'}

        emotion_lower = (emotion or 'neutral').lower()
        if emotion_lower == 'neutral':
            return {
                'valence': 0.0,
                'arousal': 0.2,
                'dominant': 'neutral',
                'tags': ['中性'],
                'narrative_hook': ''
            }
        elif any(e in emotion_lower for e in positive):
            return {
                'valence': 0.5,
                'arousal': 0.6,
                'dominant': emotion,
                'tags': ['积极', emotion],
                'narrative_hook': ''
            }
        elif any(e in emotion_lower for e in negative):
            return {
                'valence': -0.5,
                'arousal': 0.7,
                'dominant': emotion,
                'tags': ['消极', emotion],
                'narrative_hook': ''
            }
        else:
            return {
                'valence': 0.0,
                'arousal': 0.3,
                'dominant': emotion or 'neutral',
                'tags': [emotion] if emotion else ['中性'],
                'narrative_hook': ''
            }

    @staticmethod
    def _normalize_emotion(raw: dict, fallback_label: str) -> dict:
        """规范 LLM 返回的情绪数据，确保字段完整且值合法"""
        if not isinstance(raw, dict):
            return Consolidator._build_default_emotion(fallback_label)

        def clamp(val, lo, hi, default):
            try:
                return max(lo, min(hi, float(val)))
            except (TypeError, ValueError):
                return default

        return {
            'valence': clamp(raw.get('valence'), -1.0, 1.0, 0.0),
            'arousal': clamp(raw.get('arousal'), 0.0, 1.0, 0.3),
            'dominant': str(raw.get('dominant', fallback_label or 'neutral'))[:20],
            'tags': [str(t)[:10] for t in (raw.get('tags') or [])][:4] or ['中性'],
            'narrative_hook': str(raw.get('narrative_hook', ''))[:100],
        }
    
    def _analyze_with_llm(self, content, max_retries=2):
        """使用LLM分析内容，生成总结和记忆碎片（带重试机制）
        
        Args:
            content: 要分析的内容
            max_retries: 最大重试次数
        """
        api_key = self.llm_config.get('api_key')
        api_base = self.llm_config.get('api_base', 'https://api.longcat.chat/openai/v1')
        model = self.llm_config.get('model', 'LongCat-Flash-Lite')
        
        if not api_key:
            print("[CONSOLIDATOR] No LLM API key configured, using fallback")
            return self._analyze_fallback(content)
        
        try:
            # 清理内容，去除控制字符（保留换行和制表符）
            import re
            # 去除控制字符（保留换行\n、制表符\t、回车\r）
            content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', content)
            # 去除多余的空白
            content = re.sub(r'\s+', ' ', content)
            # 处理emoji：保留但确保编码正确
            content = content.encode('utf-8', errors='ignore').decode('utf-8')
            
            # 限制内容长度，避免超出LLM上下文限制
            max_content_length = 5000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            # 构建请求参数
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            prompt = f"""请分析以下内容，返回JSON格式：

【核心理念：稀疏记忆】
大模型能补全的内容 = 噪音。只存储大模型无法自行推断的信息。
- 不存储：常识、定义、概念解释、通用知识（LLM已知道）
- 存储钩子：只有当事人知道的细节、独特的情绪体验、个人化的关联

【分析要求】
1. summary：记忆钩子摘要（30-80字）
   - 不是概括总结，而是"最能唤醒这段记忆的线索"
   - 保留具体细节（时间、地点、特殊场景），去掉通用描述
   - 如果没有独特细节，只保留最核心的1-2个关键词

2. fragments：0-3个不可压缩的记忆碎片（不是必须填满）
   - 只有当内容包含"LLM无法自行推断"的信息时才提取
   - 常识性内容不提取（如"Python是一种编程语言"这种）
   - 每个碎片必须是独立的、具体的、有检索价值的

3. emotion：情绪分析
   - valence：效价，-1.0（极消极）到 1.0（极积极），0 为中性
   - arousal：唤醒度，0.0（平静/无聊）到 1.0（高度激动/紧张）
   - dominant：主导情绪（一个词，如 焦虑/好奇/满足/沮丧/兴奋/平静）
   - tags：情绪标签列表（2-4个，如 ["紧迫","压力","专注"]）
   - narrative_hook：叙事钩子（最重要！），一个标志性意象或短语，能唤醒这段记忆的线索
     示例："凌晨三点屏幕上的红色报错"、"咖啡洒在键盘上"、"他说'算了'时的表情"

【判断标准】
问自己：如果一年后再看到这个摘要，能不能立刻想起原始内容？
- 能 → 提取钩子
- 不能 → 提取更多细节，直到能

【过滤规则】
- 过滤掉网页导航路径、网站标题、作者信息等无用内容
- 过滤掉大模型已知的通用知识
- 不要包含与主题无关的噪音信息

JSON格式：
{{
    "summary": "记忆钩子摘要（具体、独特）",
    "fragments": [
        {{"content": "不可压缩的碎片1"}}
    ],
    "emotion": {{
        "valence": 0.0,
        "arousal": 0.0,
        "dominant": "情绪词",
        "tags": ["标签1", "标签2"],
        "narrative_hook": "标志性意象（必须填写）"
    }}
}}

只返回JSON，不要其他文字。

内容：
{content}"""

            data = {
                'model': model,
                'messages': [
                    {
                        'role': 'system',
                        'content': '你是一个专业的记忆分析助手，遵循"稀疏记忆"原则：大模型能补全的内容是噪音，只提取不可压缩的记忆钩子。请只返回JSON格式，不要包含其他文字。'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.3,
                'max_tokens': 1000
            }
            
            import sys
            import time
            
            # 进度提示（实时显示）
            # 重试循环
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    print(f"[CONSOLIDATOR] LLM analyzing... (attempt {attempt + 1}/{max_retries + 1})", flush=True)
                    
                    # 发送请求（减少超时时间，避免长时间阻塞）
                    response = requests.post(f'{api_base}/chat/completions', headers=headers, json=data, timeout=45)
                    response.raise_for_status()
                    break  # 成功，跳出重试循环
                    
                except requests.exceptions.Timeout as e:
                    last_error = e
                    print(f"[CONSOLIDATOR] Timeout on attempt {attempt + 1}")
                    if attempt < max_retries:
                        import time
                        time.sleep(1)  # 等待1秒后重试
                    continue
                except requests.exceptions.RequestException as e:
                    last_error = e
                    print(f"[CONSOLIDATOR] Request error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries:
                        import time
                        time.sleep(1)
                    continue
            else:
                # 所有重试都失败
                print(f"[CONSOLIDATOR] All {max_retries + 1} attempts failed: {last_error}")
                return self._analyze_fallback(content)
            
            print("[CONSOLIDATOR] Processing response...", flush=True)

            result = response.json()
            llm_response = result['choices'][0]['message']['content']

            print(f"[CONSOLIDATOR] LLM response: {llm_response[:200]}...")
            
            # 简化的JSON解析：只尝试提取大括号内容
            analysis_result = None
            try:
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    analysis_result = json.loads(json_str)
            except Exception as e:
                print(f"[CONSOLIDATOR] JSON parsing failed: {e}")
                return None
            
            # 验证必要字段
            if not analysis_result or 'summary' not in analysis_result or 'fragments' not in analysis_result:
                print("[CONSOLIDATOR] LLM response format incorrect")
                return None

            print(f"[CONSOLIDATOR] LLM analysis success: {len(analysis_result['fragments'])} fragments")
            return analysis_result

        except Exception as e:
            print(f"[CONSOLIDATOR] LLM analysis failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # 分层记忆提炼（新的架构）
    # ------------------------------------------------------------------ #

    def _analyze_ltm(self, content: str) -> dict:
        """
        深层记忆提炼：去除知识性内容，只存储线索、思路、连接、逻辑关系
        
        目标：建立记忆网络，通过一个点能提炼出一张记忆网
        
        优先使用 Ollama（如果可用），否则使用远程 LLM
        """
        # 优先使用 Ollama 本地模型
        if self._ollama is not None:
            return self._analyze_ltm_ollama(content)

        # 回退到远程 LLM
        api_key = self.llm_config.get('api_key')
        api_base = self.llm_config.get('api_base', 'https://api.longcat.chat/openai/v1')
        model = self.llm_config.get('model', 'LongCat-Flash-Lite')

        if not api_key:
            print("[CONSOLIDATOR-LTM] No LLM API key, using fallback")
            return self._analyze_ltm_fallback(content)
        
        try:
            import re
            content = re.sub(r'[\x00-\x1F\x7F]', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            max_content_length = 5000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            # 深层记忆提炼 prompt：稀疏记忆原则——只存 LLM 无法自行推断的部分
            prompt = f"""分析以下内容，提取"深层记忆"部分。

【核心理念：稀疏记忆】
大模型能补全的内容 = 噪音。只存储大模型无法自行推断的信息。

【提炼原则】
去除：大模型已知的知识性内容、事实、数据、定义、概念解释、通用逻辑
保留：个人化的推理思路、独特的连接方式、非典型的思维模式

【提取内容】
1. memory_clues: 2-4个检索线索（足够检索到即可，不需要穷尽）
2. logic_chains: 推理链条（只有当推理过程非常规/独特时才提取）
3. connections: 与其他概念的个人化关联（不是通用关联）
4. thinking_patterns: 思维模式（只有当思维方式独特时才提取）

【示例】
输入："量子力学是描述微观世界运动规律的理论"
提炼：通用知识 → memory_clues: ["量子力学"], 其余全为空（LLM 已知）

输入："我觉得量子力学的概率解释和佛教的无常观有深层联系，都是在说'确定性是幻觉'"
提炼：个人化关联 → memory_clues: ["量子力学", "无常观"],
logic_chains: ["概率解释→确定性幻觉→无常"],
connections: ["量子概率与佛教无常的跨域类比"],
thinking_patterns: ["跨学科类比思维"]

JSON格式：
{{
    "memory_clues": ["线索1", "线索2"],
    "logic_chains": ["逻辑链"],
    "connections": ["关联"],
    "thinking_patterns": ["模式"]
}}

注意：如果没有独特的个人化内容，memory_clues 保留1-2个关键词即可，其他字段可以为空数组。
只返回JSON，不要其他文字。

内容：
{content}"""

            data = {
                'model': model,
                'messages': [
                    {
                        'role': 'system',
                        'content': '你是一个专业的记忆分析专家，擅长提取深层的逻辑关系和思维模式。'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.3,
                'max_tokens': 800
            }
            
            print("[CONSOLIDATOR-LTM] Analyzing deep memory...", flush=True)
            response = requests.post(f'{api_base}/chat/completions', headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            llm_response = result['choices'][0]['message']['content']
            
            # 解析 JSON
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                ltm_result = json.loads(json_str)
                print(f"[CONSOLIDATOR-LTM] Success: {len(ltm_result.get('memory_clues', []))} clues, "
                      f"{len(ltm_result.get('logic_chains', []))} logic chains")
                return ltm_result
            else:
                print("[CONSOLIDATOR-LTM] JSON parse failed, using fallback")
                return self._analyze_ltm_fallback(content)
                
        except Exception as e:
            print(f"[CONSOLIDATOR-LTM] Error: {e}, using fallback")
            return self._analyze_ltm_fallback(content)

    def _analyze_ltm_fallback(self, content: str) -> dict:
        """LTM 提炼的后备方案：基于内容关键词提取"""
        import re
        # 提取中文词组作为线索
        words = re.findall(r'[\u4e00-\u9fff]{2,}', content)
        word_freq = {}
        for word in words:
            if len(word) >= 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        # 取最高频的词作为线索
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        memory_clues = [w[0] for w in sorted_words[:5]]
        
        return {
            'memory_clues': memory_clues,
            'logic_chains': ['从现象到结论的推理'],
            'connections': ['与其他记忆的关联待建立'],
            'thinking_patterns': ['分析性思维']
        }

    def _analyze_ltm_ollama(self, content: str) -> dict:
        """
        使用 Ollama 本地模型进行 LTM 提炼
        """
        import re
        content = re.sub(r'[\x00-\x1F\x7F]', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        max_content_length = 3000  # Ollama 模型上下文较短
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        system_prompt = "你是记忆分析专家，遵循稀疏记忆原则：只提取LLM无法自行推断的信息。只返回JSON。"
        
        prompt = f"""分析以下内容，提取"深层记忆"部分。

稀疏记忆原则：大模型能补全的=噪音，只存不可压缩的。
- 去除：通用知识、定义、概念解释
- 保留：个人化的推理、独特关联、非典型思维

JSON格式：
{{"memory_clues": ["线索"], "logic_chains": ["逻辑"], "connections": ["关联"], "thinking_patterns": ["模式"]}}
如果没有独特内容，只保留1-2个关键词线索，其他为空数组。

内容：
{content}"""

        try:
            print("[CONSOLIDATOR-LTM] Using Ollama...", flush=True)
            result = self._ollama.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens=600,
            )
            
            # 解析 JSON
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                ltm_result = json.loads(json_str)
                print(f"[CONSOLIDATOR-LTM] Ollama success: {len(ltm_result.get('memory_clues', []))} clues")
                return ltm_result
            else:
                print("[CONSOLIDATOR-LTM] Ollama JSON parse failed, using fallback")
                return self._analyze_ltm_fallback(content)
                
        except Exception as e:
            print(f"[CONSOLIDATOR-LTM] Ollama failed: {e}, using fallback")
            return self._analyze_ltm_fallback(content)

    def _analyze_persona(self, content: str, emotion: str = 'neutral') -> dict:
        """
        人格提炼：从经验内容中提取 OCEAN 五维人格分数

        返回格式（直接兼容 PersonaProfile）：
        {
            "openness": 0.0-1.0,        # 开放性：好奇心、创造力
            "conscientiousness": 0.0-1.0, # 尽责性：自律、条理
            "extraversion": 0.0-1.0,     # 外向性：社交、活力
            "agreeableness": 0.0-1.0,   # 宜人性：合作、信任
            "neuroticism": 0.0-1.0      # 神经质：情绪不稳定程度（高分=不稳定）
        }

        优先使用 Ollama（如果可用），否则使用远程 LLM
        """
        # 优先使用 Ollama 本地模型
        if self._ollama is not None:
            return self._analyze_persona_ollama(content)

        # 回退到远程 LLM
        api_key = self.llm_config.get('api_key')
        api_base = self.llm_config.get('api_base', 'https://api.longcat.chat/openai/v1')
        model = self.llm_config.get('model', 'LongCat-Flash-Lite')

        if not api_key:
            print("[CONSOLIDATOR-PERSONA] No LLM API key, skipping")
            return None

        try:
            import re
            content = re.sub(r'[\x00-\x1F\x7F]', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            max_content_length = 5000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }

            # OCEAN 五维分析 prompt
            prompt = f"""分析这段内容，评估说话者/作者在 OCEAN 五大人格维度上的倾向。

【OCEAN 五维度说明】
- openness（开放性）：好奇心、创造力、对新事物的接受程度。思考深度、艺术敏感度、哲学兴趣也归此维度。
- conscientiousness（尽责性）：自律、条理、责任感、目标导向、守时、对细节的关注。
- extraversion（外向性）：社交活跃度、能量水平、表达欲、对外部刺激的需求。
- agreeableness（宜人性）：合作意愿、信任他人、乐于助人、同理心、宽容度。
- neuroticism（神经质）：情绪波动、焦虑倾向、压力反应、内耗程度。（注：高分=情绪不稳定）

【评分要求】
- 只根据这段内容推断，不要编造或假设
- 如果内容不足以推断某个维度，给 0.5（中性）
- 0.0-0.3=低，0.4-0.6=中，0.7-1.0=高
- 重点关注：情绪词、行为描述、态度表达、价值观流露

【评分示例】
输入："项目deadline快到了，还有很多功能没完成，感到焦虑"
分析：焦虑/压力 -> neuroticism高；担心进度 -> conscientiousness中高
输出：{{"openness": 0.5, "conscientiousness": 0.7, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.8}}

输入："坚持每天6点起床跑步5公里，配速5分30秒，已经一个月了"
分析：自律习惯 -> conscientiousness高；坚持运动 -> 对健康有追求
输出：{{"openness": 0.5, "conscientiousness": 0.9, "extraversion": 0.4, "agreeableness": 0.5, "neuroticism": 0.3}}

只返回 JSON，不要其他文字。

内容：
{content}"""

            data = {
                'model': model,
                'messages': [
                    {
                        'role': 'system',
                        'content': '你是心理分析专家，擅长评估五大人格维度(OCEAN)。严格按照评分标准，给出 0.0-1.0 的数值。只返回 JSON。'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.3,
                'max_tokens': 300
            }

            print("[CONSOLIDATOR-PERSONA] Analyzing OCEAN...", flush=True)
            response = requests.post(f'{api_base}/chat/completions', headers=headers, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()
            llm_response = result['choices'][0]['message']['content']

            # 解析 JSON
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                ocean_result = json.loads(json_str)
                # 验证并规范化字段
                dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
                validated = {}
                for dim in dims:
                    raw = ocean_result.get(dim, 0.5)
                    try:
                        validated[dim] = max(0.0, min(1.0, float(raw)))
                    except (TypeError, ValueError):
                        validated[dim] = 0.5
                print(f"[CONSOLIDATOR-PERSONA] OCEAN: O={validated['openness']:.2f} C={validated['conscientiousness']:.2f} "
                      f"E={validated['extraversion']:.2f} A={validated['agreeableness']:.2f} N={validated['neuroticism']:.2f}")
                return validated
            else:
                print("[CONSOLIDATOR-PERSONA] JSON parse failed")
                return None

        except Exception as e:
            print(f"[CONSOLIDATOR-PERSONA] Error: {e}")
            return None

    def _analyze_persona_ollama(self, content: str) -> dict:
        """
        使用 Ollama 本地模型提取 OCEAN 五维人格分数
        """
        import re
        content = re.sub(r'[\x00-\x1F\x7F]', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        max_content_length = 3000  # Ollama 模型上下文较短
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        system_prompt = "你是心理分析专家，评估五大人格维度(OCEAN)。只返回JSON：{\"openness\":0.0-1.0,\"conscientiousness\":0.0-1.0,\"extraversion\":0.0-1.0,\"agreeableness\":0.0-1.0,\"neuroticism\":0.0-1.0}"

        prompt = f"""分析这段内容，评估说话者/作者在 OCEAN 五大人格维度上的倾向。

- openness：好奇心、创造力、对新事物的接受程度
- conscientiousness：自律、条理、责任感、目标导向
- extraversion：社交活跃度、能量水平、表达欲
- agreeableness：合作意愿、信任他人、乐于助人
- neuroticism：情绪波动、焦虑倾向、压力反应（高分=不稳定）

如果内容不足以推断某个维度，给 0.5（中性）。只返回 JSON。

内容：
{content}"""

        try:
            print("[CONSOLIDATOR-PERSONA] Using Ollama...", flush=True)
            result = self._ollama.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens= 200,
            )

            # 解析 JSON
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                ocean_result = json.loads(json_str)
                dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
                validated = {}
                for dim in dims:
                    raw = ocean_result.get(dim, 0.5)
                    try:
                        validated[dim] = max(0.0, min(1.0, float(raw)))
                    except (TypeError, ValueError):
                        validated[dim] = 0.5
                print(f"[CONSOLIDATOR-PERSONA] Ollama OCEAN: O={validated['openness']:.2f} C={validated['conscientiousness']:.2f} "
                      f"E={validated['extraversion']:.2f} A={validated['agreeableness']:.2f} N={validated['neuroticism']:.2f}")
                return validated
            else:
                print("[CONSOLIDATOR-PERSONA] Ollama JSON parse failed")
                return None

        except Exception as e:
            print(f"[CONSOLIDATOR-PERSONA] Ollama failed: {e}")
            return None

    def build_persona_prompt(self) -> str:
        """
        构建人格注入 prompt
        
        返回一个提示语，让大模型知道说话的是个什么样的人
        """
        # 新架构：从 PersonaProfile 获取人格
        if self.persona_profile:
            try:
                persona = self.persona_profile.get_current_persona()
                parts = []
                if persona.openness > 0.5:
                    parts.append(f"开放性：高")
                if persona.conscientiousness > 0.5:
                    parts.append(f"尽责性：高")
                if persona.extraversion > 0.5:
                    parts.append(f"外向性：高")
                if persona.agreeableness > 0.5:
                    parts.append(f"宜人性：高")
                if persona.neuroticism > 0.5:
                    parts.append(f"神经质：高")
                
                if parts:
                    return "【关于说话者】" + " | ".join(parts)
            except Exception as e:
                print(f"[CONSOLIDATOR] Build persona prompt warning: {e}")
        
        return ""

    # ------------------------------------------------------------------ #
    # P2：事实锚点提取（防幻觉）
    # ------------------------------------------------------------------ #
    
    @staticmethod
    def _extract_anchors(raw_content: str) -> str:
        """
        P2：提取不可压缩的关键事实
        
        用于防止生成式回忆时的幻觉，特别是对于：
        - 代码片段（函数名、变量名、语法）
        - 配置参数（API key 名称、端口号、路径）
        - 精确数字（日期、版本号、数量）
        - 专有名词（产品名、人名、地名）
        
        返回 JSON 格式的事实锚点数组
        """
        anchors = []
        
        # 1. 提取代码片段（函数定义、变量赋值、import 等）
        code_patterns = [
            r'(?:def|class|import|from|const|let|var|function)\s+[\w.]+',  # 代码声明
            r'[\w.]+\s*[=:]\s*[\w."\'\[\]{}]+',  # 变量赋值
            r'[\w.]+\([\w\s,.*]*\)',  # 函数调用
            r'https?://\S+',  # URL
        ]
        for pattern in code_patterns:
            matches = re.findall(pattern, raw_content)
            for m in matches[:3]:  # 每个模式最多取3个
                if len(m) > 5 and len(m) < 200:  # 过滤太短或太长的
                    anchors.append({'type': 'code', 'value': m.strip()})
        
        # 2. 提取配置参数（key=value, --flag 等）
        config_patterns = [
            r'--?[\w]+[\s=]+["\']?[\w./:-]+["\']?',  # 命令行参数
            r'[\w_]+[\s]*[:=][\s]*["\']?[\w./:-]+["\']?',  # 配置项
        ]
        for pattern in config_patterns:
            matches = re.findall(pattern, raw_content)
            for m in matches[:2]:
                if len(m) > 5 and len(m) < 100:
                    anchors.append({'type': 'config', 'value': m.strip()})
        
        # 3. 提取精确数字（版本号、日期、数量）
        number_patterns = [
            r'\bv?\d+\.\d+(?:\.\d+)?\b',  # 版本号（v1.0.1, 1.0.1）
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',  # 日期
            r'\b\d+\s*(?:MB|GB|KB|px|em|%)?\b',  # 带单位的数字
            r'\b(?:第)?\d+(?:个|次|条|位|人|年|月|日)?\b',  # 数量词
        ]
        for pattern in number_patterns:
            matches = re.findall(pattern, raw_content)
            for m in matches[:3]:
                if len(m) > 1:
                    anchors.append({'type': 'number', 'value': m.strip()})
        
        # 4. 提取带引号的专有名词
        quoted_patterns = [
            r'["\'][^"\']{3,30}["\']',  # 引号内的词组
        ]
        for pattern in quoted_patterns:
            matches = re.findall(pattern, raw_content)
            for m in matches[:2]:
                anchors.append({'type': 'quoted', 'value': m.strip()})
        
        # 去重并限制总数
        seen = set()
        unique_anchors = []
        for a in anchors:
            key = (a['type'], a['value'])
            if key not in seen and a['value'] not in str(seen):
                seen.add(key)
                unique_anchors.append(a)
        
        # 最多保留 10 个锚点
        unique_anchors = unique_anchors[:10]
        
        return json.dumps(unique_anchors, ensure_ascii=False)
    
    @staticmethod
    def _is_factual_content(raw_content: str) -> bool:
        """
        P2：判断内容是否包含需要锚定的事实
        
        以下类型内容应该被视为"事实性"：
        - 包含代码片段
        - 包含配置/参数
        - 包含精确数字
        - 包含技术术语
        """
        # 技术术语关键词
        technical_keywords = [
            'import', 'export', 'function', 'class', 'def ', 'const', 'var', 'let',
            'api', 'http', 'url', 'json', 'xml', 'sql', 'db', 'config',
            'port', 'host', 'path', 'file', 'dir', 'folder',
            'version', 'v1', 'v2', 'beta', 'alpha',
            'bug', 'fix', 'error', 'warning', 'exception',
            'test', 'unit', 'integration', 'deploy',
        ]
        
        content_lower = raw_content.lower()
        
        # 检查是否包含技术关键词
        tech_count = sum(1 for kw in technical_keywords if kw in content_lower)
        if tech_count >= 2:
            return True
        
        # 检查是否包含多个精确数字
        numbers = re.findall(r'\d+', raw_content)
        if len(numbers) >= 3:
            return True
        
        # 检查是否有代码特征（括号、箭头函数等）
        code_chars = sum(1 for c in '()=><{}[]' if c in raw_content)
        if code_chars >= 4:
            return True
        
        return False

    # ------------------------------------------------------------------ #
    # 文本相似度工具（TF-IDF 加权余弦相似度，无需额外依赖）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str):
        """简单分词：中文按字拆分，英文按空格，过滤标点"""
        import re
        # 保留中文字符和英文单词
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())
        return tokens

    @staticmethod
    def _tf(tokens: list) -> dict:
        """计算词频（TF）"""
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = len(tokens) or 1
        return {t: c / total for t, c in tf.items()}

    @staticmethod
    def _cosine_similarity(vec1: dict, vec2: dict) -> float:
        """计算两个稀疏向量的余弦相似度"""
        import math
        common = set(vec1) & set(vec2)
        if not common:
            return 0.0
        dot = sum(vec1[t] * vec2[t] for t in common)
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _is_similar(self, content1, content2):
        """判断两段内容是否相似（TF 加权余弦相似度 > 0.45）

        相比原 Jaccard 词集方法，此实现：
        - 考虑词频权重，高频词贡献更大
        - 对中文按字拆分，覆盖无空格文本
        - 阈值 0.45 在短文本上精度更高
        """
        tf1 = self._tf(self._tokenize(str(content1)))
        tf2 = self._tf(self._tokenize(str(content2)))
        sim = self._cosine_similarity(tf1, tf2)
        return sim > 0.45

    def _is_related(self, content1, content2):
        """判断两段内容是否相关（TF 余弦相似度 > 0.15）

        宽松阈值：只要有一定词汇重叠就认为相关，用于建立关联边。
        """
        tf1 = self._tf(self._tokenize(str(content1)))
        tf2 = self._tf(self._tokenize(str(content2)))
        sim = self._cosine_similarity(tf1, tf2)
        return sim > 0.15
    
    def _get_node_content(self, node_id):
        """获取节点内容"""
        all_nodes = self.long_term_memory.get_all_nodes()
        for node in all_nodes:
            if node['id'] == node_id:
                return node['content']
        return ""
    
    def consolidate_all(self):
        """整理所有待整理的工作记忆"""
        while True:
            count = self.run(10)
            if count == 0:
                break
    
    def get_statistics(self):
        """获取整理统计信息"""
        # 获取待整理条目数量
        pending_count = len(self.working_memory.get_pending())
        
        # 获取已整理条目数量
        all_entries = self.working_memory.get_all()
        consolidated_count = sum(1 for entry in all_entries if entry['is_consolidated'])
        
        return {
            'pending': pending_count,
            'consolidated': consolidated_count,
            'total': len(all_entries)
        }
    
    def cleanup(self):
        """清理过期的工作记忆"""
        deleted = self.working_memory.expire_old_entries()
        print(f"清理了 {deleted} 条过期的工作记忆")
        return deleted

    def _run_soft_forget(self):
        """运行软删除遗忘调度（在 consolidate 前自动调用）

        同时检查工作记忆和记忆痕迹的遗忘状态。
        阈值设计（借鉴 Kimi Claw 的遗忘曲线）：
        - working_memory: TTL 过期 → dormant, 7天 → forgotten
        - memory_traces: strength < 0.05 且 30 天未访问 → dormant, 90天 → forgotten
        """
        import time as _time
        t = _time.time()

        # 工作记忆遗忘
        if self.working_memory:
            try:
                wm_expired = self.working_memory.expire_old_entries()
                if wm_expired > 0:
                    print(f"[CONSOLIDATOR] WM soft-forget: {wm_expired} entries transitioned")
            except Exception as e:
                print(f"[CONSOLIDATOR] WM forget warning: {e}")

        # 记忆痕迹遗忘
        if self.persona_profile:
            try:
                result = self.persona_profile.soft_forget()
                total = result['active_to_dormant'] + result['dormant_to_forgotten']
                if total > 0:
                    print(f"[CONSOLIDATOR] Trace soft-forget: {result}")
            except Exception as e:
                print(f"[CONSOLIDATOR] Trace forget warning: {e}")
