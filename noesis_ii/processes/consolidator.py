"""
Consolidator Module - Memory Consolidation Process

Handles the consolidation of working memory to long-term memory.

Core Features:
- Concurrent batch processing with thread pool
- LLM-based content analysis for memory hooks
- OCEAN personality extraction
- Deep memory extraction (logic chains, connections, thinking patterns)
- Soft delete forget mechanism
- Fact anchor extraction for hallucination prevention
- TF-IDF based similarity comparison

Architecture:
- Supports both legacy (WorkingMemory/LongTermMemory) and new (PersonaProfile) architectures
- Falls back to local Ollama for lightweight tasks when available
- Remote LLM for heavy analysis tasks
"""

import sqlite3
import os
import datetime
import requests
import json
import re
import warnings

# Legacy modules (optional import)
try:
    from noesis_ii.core.working_memory import WorkingMemory
    from noesis_ii.core.long_term_memory import LongTermMemory
except ImportError as e:
    warnings.warn(f"Legacy modules not available: {e}")
    WorkingMemory = None
    LongTermMemory = None

# New architecture modules (optional import)
try:
    from noesis_ii.core.persona_profile import PersonaProfile
except ImportError:
    PersonaProfile = None

from noesis_ii.config_loader import ConfigLoader


class Consolidator:
    """
    Memory Consolidator Process
    
    Transforms working memory entries into consolidated long-term memory traces.
    Supports both synchronous and asynchronous execution modes.
    """

    def __init__(self, db_path, llm_config=None):
        self.db_path = db_path
        
        # Legacy architecture components (optional)
        self.working_memory = WorkingMemory(db_path) if WorkingMemory else None
        self.long_term_memory = LongTermMemory(db_path) if LongTermMemory else None
        
        # New architecture components (optional)
        self.persona_profile = PersonaProfile(db_path) if PersonaProfile else None

        # Initialize Ollama local LLM (for lightweight tasks)
        self._ollama = None
        self._init_ollama()

        # Load LLM configuration
        if llm_config:
            self.llm_config = llm_config
        else:
            try:
                config_loader = ConfigLoader('config/default_config.yaml')
                config = config_loader.load()
                self.llm_config = config.get('llm', {})
            except Exception:
                self.llm_config = {}

    def _init_ollama(self):
        """Initialize Ollama local LLM for lightweight tasks"""
        try:
            from noesis_ii.llm import OllamaLLM, is_ollama_available
            if is_ollama_available():
                self._ollama = OllamaLLM(model='deepseek-r1:1.5b')
                print("[CONSOLIDATOR] Ollama available, using deepseek-r1:1.5b")
            else:
                print("[CONSOLIDATOR] Ollama not available, using remote LLM only")
        except Exception as e:
            print(f"[CONSOLIDATOR] Ollama init failed: {e}")
            self._ollama = None
    
    def run(self, limit=3, batch_size=3, max_workers=3, async_mode=False):
        """
        Run consolidation process (supports concurrency + batch analysis)
        
        Args:
            limit:       Maximum entries to process this run
            batch_size:  Entries per batch (controls LLM load per call)
            max_workers: Maximum concurrent threads (default: 3)
            async_mode:  True=background thread (returns None immediately), False=blocking
        
        Returns:
            async_mode=False: Number of successfully processed entries (int)
            async_mode=True:  None (background execution)
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
        """Synchronous consolidation execution (shared by run() and async thread)"""
        import concurrent.futures
        import time

        if not self.working_memory:
            print("[CONSOLIDATOR] WorkingMemory not available, skipping")
            return 0
        
        # Run soft-delete forgetting schedule (automatic check on each consolidate)
        self._run_soft_forget()
        
        # Get pending working memory entries
        pending_entries = self.working_memory.get_pending(limit)
        
        if not pending_entries:
            print("[CONSOLIDATOR] No pending working memories")
            return 0

        total = len(pending_entries)
        print(f"[CONSOLIDATOR] Processing {total} entries (max_workers={max_workers})")
        t_start = time.time()

        consolidated_count = 0
        failed_entries = []

        # Concurrent processing with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers,
                                                    thread_name_prefix="consolidator") as pool:
            future_to_entry = {
                pool.submit(self._consolidate_entry, entry): entry
                for entry in pending_entries
            }
            for future in concurrent.futures.as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    success = future.result(timeout=120)  # Max 120s per entry
                    if success:
                        consolidated_count += 1
                        print(f"[CONSOLIDATOR] Done id={entry['id']} ({consolidated_count}/{total})")
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
        """Consolidate single working memory entry (compatible with both architectures)"""
        content = entry['content']
        emotion = entry.get('emotion', 'neutral') or 'neutral'

        # Handle encoding: ensure UTF-8, handle special characters
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        content = content.encode('utf-8', errors='ignore').decode('utf-8')

        # Analyze content with LLM to generate summary + structured emotion
        analysis_result = self._analyze_with_llm(content)

        if not analysis_result:
            print("[CONSOLIDATOR] LLM analysis failed, using basic consolidation")
            summary = content[:200] if len(content) > 200 else content
            emotion_data = self._build_default_emotion(emotion)
        else:
            summary = analysis_result.get('summary', content[:200])
            raw_emotion = analysis_result.get('emotion', {})
            emotion_data = self._normalize_emotion(raw_emotion, emotion)
            print(f"[CONSOLIDATOR] Emotion: valence={emotion_data.get('valence', 0):.2f} "
                  f"arousal={emotion_data.get('arousal', 0):.2f} "
                  f"dominant={emotion_data.get('dominant', 'neutral')}")

        # OCEAN personality analysis (parallel to content analysis)
        ocean_data = self._analyze_persona(content, emotion)
        if ocean_data:
            print(f"[CONSOLIDATOR] OCEAN: O={ocean_data['openness']:.2f} C={ocean_data['conscientiousness']:.2f} "
                  f"E={ocean_data['extraversion']:.2f} A={ocean_data['agreeableness']:.2f} N={ocean_data['neuroticism']:.2f}")

        # New architecture: Store to PersonaProfile
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

        # Legacy architecture: Store to LTM (if available)
        if self.long_term_memory:
            try:
                node_id = self.long_term_memory.create_node(
                    summary,
                    node_type='consolidated',
                    weight=1.0
                )
                print(f"[CONSOLIDATOR] Stored to LTM: node_id={node_id}")
            except Exception as e:
                print(f"[CONSOLIDATOR] LTM storage warning: {e}")

        # Mark as consolidated
        if self.working_memory:
            try:
                self.working_memory.mark_consolidated(entry['id'])
            except Exception as e:
                print(f"[CONSOLIDATOR] Mark consolidated warning: {e}")

        return True

    @staticmethod
    def _build_default_emotion(emotion: str) -> dict:
        """Build default emotion_data structure from simple emotion label"""
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
        """Normalize LLM emotion data, ensure complete fields and valid values"""
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
        """Analyze content with LLM, generate summary and memory fragments (with retry)"""
        api_key = self.llm_config.get('api_key')
        api_base = self.llm_config.get('api_base', 'https://api.longcat.chat/openai/v1')
        model = self.llm_config.get('model', 'LongCat-Flash-Lite')
        
        if not api_key:
            print("[CONSOLIDATOR] No LLM API key configured, using fallback")
            return self._analyze_fallback(content)
        
        try:
            # Clean content: remove control characters
            content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            content = content.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Limit content length to avoid context overflow
            max_content_length = 5000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            prompt = f"""Analyze the following content and return JSON:

[Core Principle: Sparse Memory]
Content that LLM can infer = noise. Only store information LLM cannot infer.
- Don't store: common knowledge, definitions, concept explanations
- Store hooks: personal details, unique emotional experiences, personalized associations

[Analysis Requirements]
1. summary: Memory hook summary (30-80 chars)
   - Not a general summary, but "the clue that best awakens this memory"
   - Keep specific details (time, place, special scenarios)
   - If no unique details, keep only 1-2 core keywords

2. fragments: 0-3 non-compressible memory fragments (optional)
   - Only extract when content contains "LLM-inferable" information
   - Don't extract common knowledge
   - Each fragment must be independent, specific, retrievable

3. emotion: Emotional analysis
   - valence: -1.0 (negative) to 1.0 (positive), 0 = neutral
   - arousal: 0.0 (calm) to 1.0 (excited)
   - dominant: primary emotion (e.g., anxious/curious/satisfied/frustrated)
   - tags: emotion tags (2-4, e.g., ["urgent", "stress", "focus"])
   - narrative_hook: iconic image or phrase that triggers memory recall

[JSON Format]:
{{
    "summary": "Memory hook summary",
    "fragments": [{{"content": "fragment"}}],
    "emotion": {{
        "valence": 0.0,
        "arousal": 0.0,
        "dominant": "emotion",
        "tags": ["tag1", "tag2"],
        "narrative_hook": "iconic image"
    }}
}}

Return only JSON, no other text.

Content:
{content}"""

            data = {
                'model': model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a professional memory analysis assistant. Follow "sparse memory" principle: only extract non-compressible memory hooks. Return only JSON.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.3,
                'max_tokens': 1000
            }
            
            # Retry loop
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    print(f"[CONSOLIDATOR] LLM analyzing... (attempt {attempt + 1}/{max_retries + 1})", flush=True)
                    response = requests.post(f'{api_base}/chat/completions', headers=headers, json=data, timeout=45)
                    response.raise_for_status()
                    break
                except requests.exceptions.Timeout as e:
                    last_error = e
                    print(f"[CONSOLIDATOR] Timeout on attempt {attempt + 1}")
                    if attempt < max_retries:
                        import time
                        time.sleep(1)
                    continue
                except requests.exceptions.RequestException as e:
                    last_error = e
                    print(f"[CONSOLIDATOR] Request error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries:
                        import time
                        time.sleep(1)
                    continue
            else:
                print(f"[CONSOLIDATOR] All {max_retries + 1} attempts failed: {last_error}")
                return self._analyze_fallback(content)
            
            print("[CONSOLIDATOR] Processing response...", flush=True)

            result = response.json()
            llm_response = result['choices'][0]['message']['content']

            print(f"[CONSOLIDATOR] LLM response: {llm_response[:200]}...")
            
            # Simple JSON parsing: extract content between braces
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
            
            if not analysis_result or 'summary' not in analysis_result:
                print("[CONSOLIDATOR] LLM response format incorrect")
                return None

            print(f"[CONSOLIDATOR] LLM analysis success: {len(analysis_result.get('fragments', []))} fragments")
            return analysis_result

        except Exception as e:
            print(f"[CONSOLIDATOR] LLM analysis failed: {e}")
            return None

    def _analyze_fallback(self, content: str) -> dict:
        """Fallback analysis when LLM is unavailable"""
        summary = content[:100] if len(content) > 100 else content
        return {
            'summary': summary,
            'fragments': [],
            'emotion': {'valence': 0.0, 'arousal': 0.2, 'dominant': 'neutral', 'tags': ['中性'], 'narrative_hook': ''}
        }

    # ------------------------------------------------------------------ #
    # Deep Memory Extraction (new architecture)
    # ------------------------------------------------------------------ #

    def _analyze_ltm(self, content: str) -> dict:
        """
        Deep memory extraction: Remove knowledge content, store only clues, connections, logic relations.
        
        Goal: Build memory network where one point can retrieve a network of memories.
        
        Priority: Use Ollama if available, otherwise use remote LLM.
        """
        if self._ollama is not None:
            return self._analyze_ltm_ollama(content)

        api_key = self.llm_config.get('api_key')
        api_base = self.llm_config.get('api_base', 'https://api.longcat.chat/openai/v1')
        model = self.llm_config.get('model', 'LongCat-Flash-Lite')

        if not api_key:
            print("[CONSOLIDATOR-LTM] No LLM API key, using fallback")
            return self._analyze_ltm_fallback(content)
        
        try:
            content = re.sub(r'[\x00-\x1F\x7F]', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            max_content_length = 5000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            prompt = f"""Analyze content and extract "deep memory" components.

[Core Principle: Sparse Memory]
Content LLM can infer = noise. Only store information LLM cannot infer.

[Extraction Rules]
Remove: factual knowledge, data, definitions, concept explanations, general logic
Keep: personal reasoning, unique connections, non-typical thinking patterns

[Extract Components]
1. memory_clues: 2-4 retrieval clues
2. logic_chains: inference chains (only if reasoning is unconventional)
3. connections: personalized associations (not generic)
4. thinking_patterns: thinking patterns (only if unique)

[JSON Format]:
{{
    "memory_clues": ["clue1", "clue2"],
    "logic_chains": ["logic"],
    "connections": ["association"],
    "thinking_patterns": ["pattern"]
}}

Return only JSON.

Content:
{content}"""

            data = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': 'You are a memory analysis expert.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.3,
                'max_tokens': 800
            }
            
            print("[CONSOLIDATOR-LTM] Analyzing deep memory...", flush=True)
            response = requests.post(f'{api_base}/chat/completions', headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            llm_response = result['choices'][0]['message']['content']
            
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                ltm_result = json.loads(json_str)
                print(f"[CONSOLIDATOR-LTM] Success: {len(ltm_result.get('memory_clues', []))} clues")
                return ltm_result
            else:
                print("[CONSOLIDATOR-LTM] JSON parse failed, using fallback")
                return self._analyze_ltm_fallback(content)
                
        except Exception as e:
            print(f"[CONSOLIDATOR-LTM] Error: {e}, using fallback")
            return self._analyze_ltm_fallback(content)

    def _analyze_ltm_fallback(self, content: str) -> dict:
        """Fallback for LTM extraction: keyword-based"""
        words = re.findall(r'[\u4e00-\u9fff]{2,}', content)
        word_freq = {}
        for word in words:
            if len(word) >= 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        memory_clues = [w[0] for w in sorted_words[:5]]
        
        return {
            'memory_clues': memory_clues,
            'logic_chains': ['inference from observation'],
            'connections': ['associations pending'],
            'thinking_patterns': ['analytical thinking']
        }

    def _analyze_ltm_ollama(self, content: str) -> dict:
        """Deep memory extraction using Ollama local model"""
        content = re.sub(r'[\x00-\x1F\x7F]', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        max_content_length = 3000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        system_prompt = "You are memory analysis expert. Follow sparse memory principle. Return only JSON."
        
        prompt = f"""Analyze content, extract deep memory.
Sparse memory: store only non-inferable info.
- Remove: general knowledge, definitions
- Keep: personal reasoning, unique connections

JSON: {{"memory_clues": [], "logic_chains": [], "connections": [], "thinking_patterns": []}}

Content:
{content}"""

        try:
            print("[CONSOLIDATOR-LTM] Using Ollama...", flush=True)
            result = self._ollama.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens=600,
            )
            
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                ltm_result = json.loads(json_str)
                print(f"[CONSOLIDATOR-LTM] Ollama success: {len(ltm_result.get('memory_clues', []))} clues")
                return ltm_result
            else:
                print("[CONSOLIDATOR-LTM] Ollama JSON parse failed")
                return self._analyze_ltm_fallback(content)
                
        except Exception as e:
            print(f"[CONSOLIDATOR-LTM] Ollama failed: {e}")
            return self._analyze_ltm_fallback(content)

    def _analyze_persona(self, content: str, emotion: str = 'neutral') -> dict:
        """
        Persona extraction: Extract OCEAN five-factor personality scores from experience content.
        
        Returns:
        {
            "openness": 0.0-1.0,         # Curiosity, creativity
            "conscientiousness": 0.0-1.0, # Self-discipline, organization
            "extraversion": 0.0-1.0,      # Sociability, energy
            "agreeableness": 0.0-1.0,     # Cooperation, trust
            "neuroticism": 0.0-1.0       # Emotional stability (high = unstable)
        }
        
        Priority: Use Ollama if available, otherwise use remote LLM.
        """
        if self._ollama is not None:
            return self._analyze_persona_ollama(content)

        api_key = self.llm_config.get('api_key')
        api_base = self.llm_config.get('api_base', 'https://api.longcat.chat/openai/v1')
        model = self.llm_config.get('model', 'LongCat-Flash-Lite')

        if not api_key:
            print("[CONSOLIDATOR-PERSONA] No LLM API key, skipping")
            return None

        try:
            content = re.sub(r'[\x00-\x1F\x7F]', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            max_content_length = 5000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }

            prompt = f"""Analyze this content and evaluate the speaker's OCEAN personality traits.

[OCEAN Dimensions]
- openness: curiosity, creativity, openness to new experiences
- conscientiousness: self-discipline, organization, responsibility, goal-oriented
- extraversion: sociability, energy level, expressiveness
- agreeableness: cooperation, trust, helpfulness, empathy
- neuroticism: emotional volatility, anxiety, stress response (high = unstable)

[Scoring]
- Score only based on provided content
- If insufficient info, score 0.5 (neutral)
- 0.0-0.3=low, 0.4-0.6=medium, 0.7-1.0=high

Return only JSON with OCEAN scores.

Content:
{content}"""

            data = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': 'You are a psychology expert. Evaluate OCEAN traits. Return only JSON.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.3,
                'max_tokens': 300
            }

            print("[CONSOLIDATOR-PERSONA] Analyzing OCEAN...", flush=True)
            response = requests.post(f'{api_base}/chat/completions', headers=headers, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()
            llm_response = result['choices'][0]['message']['content']

            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                ocean_result = json.loads(json_str)
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
        """OCEAN personality extraction using Ollama local model"""
        content = re.sub(r'[\x00-\x1F\x7F]', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        max_content_length = 3000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        system_prompt = "You are a psychology expert. Evaluate OCEAN traits. Return only JSON."

        prompt = f"""Analyze content for OCEAN personality traits.
- openness: curiosity, creativity, openness
- conscientiousness: discipline, organization, responsibility
- extraversion: sociability, energy, expressiveness
- agreeableness: cooperation, trust, empathy
- neuroticism: emotional volatility (high = unstable)

If insufficient info, use 0.5. Return only JSON.

Content:
{content}"""

        try:
            print("[CONSOLIDATOR-PERSONA] Using Ollama...", flush=True)
            result = self._ollama.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens=200,
            )

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
                return validated
            else:
                print("[CONSOLIDATOR-PERSONA] Ollama JSON parse failed")
                return None

        except Exception as e:
            print(f"[CONSOLIDATOR-PERSONA] Ollama failed: {e}")
            return None

    def build_persona_prompt(self) -> str:
        """
        Build persona injection prompt.
        
        Returns a prompt that tells LLM about the speaker's personality traits.
        """
        if self.persona_profile:
            try:
                persona = self.persona_profile.get_current_persona()
                parts = []
                if persona.openness > 0.5:
                    parts.append(f"Openness: High")
                if persona.conscientiousness > 0.5:
                    parts.append(f"Conscientiousness: High")
                if persona.extraversion > 0.5:
                    parts.append(f"Extraversion: High")
                if persona.agreeableness > 0.5:
                    parts.append(f"Agreeableness: High")
                if persona.neuroticism > 0.5:
                    parts.append(f"Neuroticism: High")
                
                if parts:
                    return "[Speaker Profile] " + " | ".join(parts)
            except Exception as e:
                print(f"[CONSOLIDATOR] Build persona prompt warning: {e}")
        
        return ""

    # ------------------------------------------------------------------ #
    # P2: Fact Anchors (Anti-hallucination)
    # ------------------------------------------------------------------ #
    
    @staticmethod
    def _extract_anchors(raw_content: str) -> str:
        """
        P2: Extract non-compressible key facts for hallucination prevention.
        
        Used to prevent hallucinations during generative recall, especially for:
        - Code snippets (function names, variables, syntax)
        - Config parameters (API keys, ports, paths)
        - Precise numbers (dates, versions, quantities)
        - Proper nouns (product names, people, places)
        
        Returns JSON array of fact anchors.
        """
        anchors = []
        
        # Extract code patterns
        code_patterns = [
            r'(?:def|class|import|from|const|let|var|function)\s+[\w.]+',
            r'[\w.]+\s*[=:]\s*[\w."\'\[\]{}]+',
            r'[\w.]+\([\w\s,.*]*\)',
            r'https?://\S+',
        ]
        for pattern in code_patterns:
            matches = re.findall(pattern, raw_content)
            for m in matches[:3]:
                if len(m) > 5 and len(m) < 200:
                    anchors.append({'type': 'code', 'value': m.strip()})
        
        # Extract config parameters
        config_patterns = [
            r'--?[\w]+[\s=]+["\']?[\w./:-]+["\']?',
            r'[\w_]+[\s]*[:=][\s]*["\']?[\w./:-]+["\']?',
        ]
        for pattern in config_patterns:
            matches = re.findall(pattern, raw_content)
            for m in matches[:2]:
                if len(m) > 5 and len(m) < 100:
                    anchors.append({'type': 'config', 'value': m.strip()})
        
        # Extract precise numbers
        number_patterns = [
            r'\bv?\d+\.\d+(?:\.\d+)?\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b\d+\s*(?:MB|GB|KB|px|em|%)?\b',
            r'\b(?:第)?\d+(?:个|次|条|位|人|年|月|日)?\b',
        ]
        for pattern in number_patterns:
            matches = re.findall(pattern, raw_content)
            for m in matches[:3]:
                if len(m) > 1:
                    anchors.append({'type': 'number', 'value': m.strip()})
        
        # Extract quoted proper nouns
        quoted_patterns = [r'["\'][^"\']{3,30}["\']']
        for pattern in quoted_patterns:
            matches = re.findall(pattern, raw_content)
            for m in matches[:2]:
                anchors.append({'type': 'quoted', 'value': m.strip()})
        
        # Deduplicate
        seen = set()
        unique_anchors = []
        for a in anchors:
            key = (a['type'], a['value'])
            if key not in seen:
                seen.add(key)
                unique_anchors.append(a)
        
        return json.dumps(unique_anchors[:10], ensure_ascii=False)
    
    @staticmethod
    def _is_factual_content(raw_content: str) -> bool:
        """
        P2: Determine if content contains factual information requiring anchoring.
        
        Content types considered "factual":
        - Code snippets
        - Config/parameters
        - Precise numbers
        - Technical terminology
        """
        technical_keywords = [
            'import', 'export', 'function', 'class', 'def ', 'const', 'var', 'let',
            'api', 'http', 'url', 'json', 'xml', 'sql', 'db', 'config',
            'port', 'host', 'path', 'file', 'dir', 'folder',
            'version', 'v1', 'v2', 'beta', 'alpha',
            'bug', 'fix', 'error', 'warning', 'exception',
            'test', 'unit', 'integration', 'deploy',
        ]
        
        content_lower = raw_content.lower()
        tech_count = sum(1 for kw in technical_keywords if kw in content_lower)
        if tech_count >= 2:
            return True
        
        numbers = re.findall(r'\d+', raw_content)
        if len(numbers) >= 3:
            return True
        
        code_chars = sum(1 for c in '()=><{}[]' if c in raw_content)
        if code_chars >= 4:
            return True
        
        return False

    # ------------------------------------------------------------------ #
    # Text Similarity Utilities (TF-IDF weighted cosine similarity)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str):
        """Simple tokenization: Chinese by character, English by word"""
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())
        return tokens

    @staticmethod
    def _tf(tokens: list) -> dict:
        """Calculate term frequency (TF)"""
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = len(tokens) or 1
        return {t: c / total for t, c in tf.items()}

    @staticmethod
    def _cosine_similarity(vec1: dict, vec2: dict) -> float:
        """Calculate cosine similarity between two sparse vectors"""
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
        """Check if two contents are similar (TF-weighted cosine similarity > 0.45)"""
        tf1 = self._tf(self._tokenize(str(content1)))
        tf2 = self._tf(self._tokenize(str(content2)))
        sim = self._cosine_similarity(tf1, tf2)
        return sim > 0.45

    def _is_related(self, content1, content2):
        """Check if two contents are related (TF cosine similarity > 0.15)"""
        tf1 = self._tf(self._tokenize(str(content1)))
        tf2 = self._tf(self._tokenize(str(content2)))
        sim = self._cosine_similarity(tf1, tf2)
        return sim > 0.15
    
    def _get_node_content(self, node_id):
        """Get node content by ID"""
        all_nodes = self.long_term_memory.get_all_nodes()
        for node in all_nodes:
            if node['id'] == node_id:
                return node['content']
        return ""
    
    def consolidate_all(self):
        """Consolidate all pending working memory entries"""
        while True:
            count = self.run(10)
            if count == 0:
                break
    
    def get_statistics(self):
        """Get consolidation statistics"""
        pending_count = len(self.working_memory.get_pending())
        all_entries = self.working_memory.get_all()
        consolidated_count = sum(1 for entry in all_entries if entry['is_consolidated'])
        
        return {
            'pending': pending_count,
            'consolidated': consolidated_count,
            'total': len(all_entries)
        }
    
    def cleanup(self):
        """Clean up expired working memory entries"""
        deleted = self.working_memory.expire_old_entries()
        print(f"Cleaned {deleted} expired working memory entries")
        return deleted

    def _run_soft_forget(self):
        """
        Run soft-delete forget schedule (automatically called before consolidation)
        
        Threshold design (inspired by Kimi Claw):
        - working_memory: TTL expired → dormant, 7 days → forgotten
        - memory_traces: strength < 0.05 and 30 days inactive → dormant, 90 days → forgotten
        """
        import time as _time

        # Working memory forgetting
        if self.working_memory:
            try:
                wm_expired = self.working_memory.expire_old_entries()
                if wm_expired > 0:
                    print(f"[CONSOLIDATOR] WM soft-forget: {wm_expired} entries")
            except Exception as e:
                print(f"[CONSOLIDATOR] WM forget warning: {e}")

        # Memory trace forgetting
        if self.persona_profile:
            try:
                result = self.persona_profile.soft_forget()
                total = result['active_to_dormant'] + result['dormant_to_forgotten']
                if total > 0:
                    print(f"[CONSOLIDATOR] Trace soft-forget: {result}")
            except Exception as e:
                print(f"[CONSOLIDATOR] Trace forget warning: {e}")
