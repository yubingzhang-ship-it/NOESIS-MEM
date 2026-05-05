"""
PersonaProfile - Core Storage for Personality-Consistent Memory System

Replaces the original AlayaSeeds, removes Buddhist terminology, focuses on verifiable personality consistency storage.

Core Concepts:
- MemoryTrace: Basic unit of memory, stores "how to regenerate" rules for experiences
- ConditionPattern: Trigger conditions, not the content itself
- LongTermImpact: Long-term impact assessment (replaces KarmaVector)
- RetrievalConditions: Multi-condition retrieval (replaces Four Conditions)

Key Features:
- Lightweight: Only stores summaries and condition patterns, original content in LTM
- Retrievable: Weighted retrieval based on multiple conditions
- Evolvable: Supports memory trace enhancement and decay

Revision History:
  v3.0 (2026-04-10) - Route A Refactor: Remove terminology, focus on personality consistency
"""

import sqlite3
import os
import json
import time
import math
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import datetime


# =========================================================================
# Data Classes
# =========================================================================

@dataclass
class LongTermImpact:
    """
    Long-term Impact Assessment (replaces KarmaVector)

    Multi-dimensional assessment of experience's long-term impact on system personality:
    - social_valence: Social valence [-1, 1], positive = positive social influence
    - authenticity: Authenticity [-1, 1], positive = authentic expression
    - engagement: Engagement [-1, 1], positive = deep engagement
    - coherence: Coherence contribution [-1, 1], contribution to personality consistency

    Usage Examples:
    - Sincere help: LongTermImpact(0.8, 0.7, 0.6, 0.5)
    - False response: LongTermImpact(-0.3, -0.8, 0.2, -0.4)
    """
    social_valence: float = 0.0
    authenticity: float = 0.0
    engagement: float = 0.0
    coherence: float = 0.0

    def net_impact(self) -> float:
        """Calculate net impact (weighted average)"""
        return (
            self.social_valence * 0.30 +
            self.authenticity * 0.25 +
            self.engagement * 0.25 +
            self.coherence * 0.20
        )

    def is_positive(self) -> bool:
        """Check if positive impact"""
        return self.net_impact() > 0.2

    def is_negative(self) -> bool:
        """Check if negative impact"""
        return self.net_impact() < -0.2

    def to_dict(self) -> dict:
        return {
            'social_valence': round(self.social_valence, 4),
            'authenticity': round(self.authenticity, 4),
            'engagement': round(self.engagement, 4),
            'coherence': round(self.coherence, 4),
            'net_impact': round(self.net_impact(), 4),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LongTermImpact':
        return cls(
            social_valence=data.get('social_valence', 0.0),
            authenticity=data.get('authenticity', 0.0),
            engagement=data.get('engagement', 0.0),
            coherence=data.get('coherence', 0.0),
        )


class TraceType(Enum):
    """Memory trace types"""
    PERCEPTUAL = "perceptual"
    CONCEPTUAL = "conceptual"
    EMOTIONAL = "emotional"
    ACTION = "action"
    EPISODIC = "episodic"


class RelationType(Enum):
    """Memory trace relationship types (inspired by Kimi Claw's 5 relationship types)

    causal: Causal relationship (A causes B)
    associated: General association (A and B related, but not in other types)
    similar: Similar relationship (A and B similar in some aspect)
    contrast: Contrast relationship (A and B opposite/contrasting)
    temporal: Temporal relationship (A happened before/after B)
    """
    CAUSAL = "causal"
    ASSOCIATED = "associated"
    SIMILAR = "similar"
    CONTRAST = "contrast"
    TEMPORAL = "temporal"


@dataclass
class MemoryTrace:
    """
    Memory Trace: Basic storage unit for personality consistency

    Key Features:
    - Latency: No specific content before retrieval
    - Conditionality: Requires condition pattern to be met for retrieval
    - Evolvability: Enhanced after retrieval, decays without long-term retrieval
    """
    trace_id: str
    trace_type: str

    # Core attributes
    strength: float = 0.1  # Trace strength [0,1]

    # Long-term impact assessment
    long_term_impact: LongTermImpact = field(default_factory=LongTermImpact)

    # Condition pattern (trigger conditions for retrieval)
    condition_pattern: Dict = field(default_factory=dict)

    # Access history
    access_history: List[Dict] = field(default_factory=list)

    # Self-relevance
    self_relevance: float = 0.0

    # Status
    last_accessed: float = 0.0
    access_count: int = 0
    is_active: bool = True

    # Database ID
    db_id: Optional[int] = None

    # Content summary (for display)
    content_summary: str = ""


@dataclass
class Persona:
    """
    Personality Representation: OCEAN Five-Factor Model

    Based on Big Five Personality Model (OCEAN):
    - Openness: Openness to experience (curiosity, creativity)
    - Conscientiousness: Conscientiousness (self-discipline, organization)
    - Extraversion: Extraversion (sociability, energy)
    - Agreeableness: Agreeableness (cooperation, trust)
    - Neuroticism: Neuroticism (emotional stability, low score = stable)

    All dimensions range [0, 1], 0.5 is neutral baseline
    """
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'openness': round(self.openness, 4),
            'conscientiousness': round(self.conscientiousness, 4),
            'extraversion': round(self.extraversion, 4),
            'agreeableness': round(self.agreeableness, 4),
            'neuroticism': round(self.neuroticism, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Persona':
        """Create from dictionary"""
        return cls(
            openness=data.get('openness', 0.5),
            conscientiousness=data.get('conscientiousness', 0.5),
            extraversion=data.get('extraversion', 0.5),
            agreeableness=data.get('agreeableness', 0.5),
            neuroticism=data.get('neuroticism', 0.5),
        )


@dataclass
class RetrievedMemory:
    """Retrieval result"""
    trace: MemoryTrace
    content: Dict
    relevance_score: float
    retrieval_conditions: Dict
    timestamp: float


# =========================================================================
# Main System
# =========================================================================

class PersonaProfile:
    """
    Persona Profile: Storage and management of persistent personality representation

    Core Methods:
    - store_experience() — Store experience as memory trace
    - retrieve_by_conditions() — Multi-condition retrieval
    - update_trace_strength() — Update trace strength
    """

    def __init__(self, db_path: str = None, config: Dict = None):
        self.db_path = db_path
        self.config = config or {}
        self._trace_cache: Dict[int, MemoryTrace] = {}

        if db_path:
            self._init_db()
            self._migrate_db()

    def _init_db(self):
        """Initialize database tables"""
        if not self.db_path:
            return

        dir_name = os.path.dirname(self.db_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Memory traces table (replaces alaya_seeds)
        # memory_state: active → dormant → forgotten
        # emotion_data: structured emotion vector JSON {valence, arousal, dominant, tags, narrative_hook}
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT UNIQUE,
                content_summary TEXT,
                trace_type TEXT,
                strength REAL DEFAULT 0.1,
                long_term_impact TEXT,
                condition_pattern TEXT,
                access_history TEXT,
                self_relevance REAL DEFAULT 0.0,
                last_accessed REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                memory_state TEXT DEFAULT 'active',
                dormant_since REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                emotion_data TEXT DEFAULT '{}'
            )
        ''')

        # Trace associations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trace_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_trace_id INTEGER,
                target_trace_id INTEGER,
                relation_type TEXT,
                strength REAL DEFAULT 0.5,
                context TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_trace_id) REFERENCES memory_traces(id),
                FOREIGN KEY (target_trace_id) REFERENCES memory_traces(id)
            )
        ''')

        conn.commit()
        conn.close()

    def _migrate_db(self):
        """Migrate from old table structure"""
        if not self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if old table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alaya_seeds'")
        if cursor.fetchone():
            print("[PersonaProfile] Migrating from alaya_seeds to memory_traces...")

            # Migrate alaya_seeds to memory_traces
            cursor.execute('''
                SELECT seed_id, content, content_summary, seed_type, potency,
                       karma_quality, created_at, updated_at, manifestation_count,
                       condition_pattern, vasana_imprints
                FROM alaya_seeds
            ''')

            seeds = cursor.fetchall()
            migrated = 0

            for seed in seeds:
                (seed_id, content, content_summary, seed_type, potency,
                 karma_quality, created_at, updated_at, manifestation_count,
                 condition_pattern, vasana_imprints) = seed

                # Map seed_type to trace_type
                trace_type = seed_type if seed_type else 'episodic'

                # Map potency to strength
                strength = potency if potency else 0.5

                # Map karma_quality to long_term_impact
                long_term_impact = 0.5
                if karma_quality:
                    # Simple mapping: good=0.8, neutral=0.5, bad=0.2
                    karma_map = {'善': 0.8, '无记': 0.5, '恶': 0.2}
                    long_term_impact = karma_map.get(karma_quality, 0.5)

                # Calculate self_relevance (based on manifestation_count)
                self_relevance = min(1.0, 0.3 + (manifestation_count or 0) * 0.05)

                # Insert into memory_traces
                cursor.execute('''
                    INSERT OR IGNORE INTO memory_traces
                    (trace_id, content_summary, trace_type, strength, long_term_impact,
                     self_relevance, created_at, updated_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    seed_id,
                    content_summary or content[:200] if content else '',
                    trace_type,
                    strength,
                    long_term_impact,
                    self_relevance,
                    created_at or datetime.datetime.now().isoformat(),
                    updated_at or datetime.datetime.now().isoformat(),
                    1
                ))

                if cursor.rowcount > 0:
                    migrated += 1

            conn.commit()
            print(f"[PersonaProfile] Migrated {migrated} seeds to memory_traces")

        conn.close()

    def _connect(self) -> sqlite3.Connection:
        """Get database connection"""
        dir_name = os.path.dirname(self.db_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def store_experience(self, experience: str,
                         trace_type: str = None,
                         emotion: str = None,
                         intensity: float = 0.5,
                         context: Dict = None,
                         summary: str = None) -> int:
        """
        Store experience as memory trace

        Args:
            experience: Raw experience content
            summary: LLM-extracted summary (preferred)
            trace_type: Trace type
            emotion: Emotion label
            intensity: Intensity
            context: Context

        Returns: Database ID of the trace
        """
        context = context or {}
        trace_type = trace_type or 'episodic'

        # Assess long-term impact
        long_term_impact = self._assess_impact(emotion)

        # Decide storage content
        if summary:
            trace_content = summary
        elif experience:
            trace_content = experience[:500] if len(experience) > 500 else experience
        else:
            trace_content = ""

        # Extract condition pattern
        condition_pattern = self._extract_condition_pattern(trace_content, context)

        # Inject OCEAN personality data into condition_pattern
        ocean_data = context.get('ocean') if isinstance(context, dict) else None
        if ocean_data and isinstance(ocean_data, dict):
            condition_pattern['ocean'] = ocean_data

        # Structured emotion data
        emotion_data = context.get('emotion_data') if isinstance(context, dict) else None
        if not emotion_data or not isinstance(emotion_data, dict):
            emotion_data = {'valence': 0.0, 'arousal': 0.3, 'dominant': emotion or 'neutral',
                           'tags': [emotion] if emotion else [], 'narrative_hook': ''}

        # Generate trace_id
        pattern_str = json.dumps(condition_pattern, sort_keys=True)
        trace_id = hashlib.md5(pattern_str.encode()).hexdigest()[:16]

        # Check for similar traces
        existing = self._find_similar_trace(condition_pattern)

        conn = self._connect()
        try:
            cursor = conn.cursor()

            if existing:
                # Merge: enhance existing trace
                new_strength = min(1.0, (existing['strength'] or 0.1) * 1.1 + intensity * 0.05)
                new_count = (existing['access_count'] or 0) + 1

                # Update access history
                access_history = []
                if existing.get('access_history'):
                    try:
                        access_history = json.loads(existing['access_history'])
                    except (json.JSONDecodeError, TypeError):
                        pass

                    access_history.append({
                        'timestamp': time.time(),
                        'type': 'reinforce',
                        'context': context,
                    })
                    access_history = access_history[-50:]

                    emotion_json = json.dumps(emotion_data, ensure_ascii=False)
                    cursor.execute('''
                        UPDATE memory_traces
                        SET strength = ?, access_count = ?,
                            access_history = ?, updated_at = CURRENT_TIMESTAMP,
                            emotion_data = ?
                        WHERE id = ?
                    ''', (new_strength, new_count, json.dumps(access_history, ensure_ascii=False),
                          emotion_json, existing['id']))

                    db_id = existing['id']
            else:
                # Create new trace
                impact_json = json.dumps(long_term_impact.to_dict(), ensure_ascii=False)
                emotion_json = json.dumps(emotion_data, ensure_ascii=False)

                cursor.execute('''
                    INSERT INTO memory_traces
                    (trace_id, content_summary, trace_type, strength, long_term_impact,
                     condition_pattern, access_history, self_relevance,
                     last_accessed, access_count, is_active, created_at, updated_at,
                     emotion_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                ''', (
                    trace_id,
                    trace_content[:200] if trace_content else "",
                    trace_type,
                    intensity * 0.1 + 0.1,
                    impact_json,
                    json.dumps(condition_pattern, ensure_ascii=False),
                    json.dumps([{'timestamp': time.time(), 'type': 'creation'}], ensure_ascii=False),
                    0.0,
                    0.0,
                    1,
                    1,
                    emotion_json,
                ))
                db_id = cursor.lastrowid

            conn.commit()
        finally:
            conn.close()
        return db_id

    def retrieve_memories(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve memory traces (compatible with old retriever interface)"""
        import re
        tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', query)
        if not tokens:
            tokens = [query] if query else []
        traces = self.retrieve_by_conditions(tokens, top_k=top_k)
        # Improvement 3 (Narrative Reconstruction): Return emotion_data to support narrative_hook extraction
        return [{'id': t.get('db_id', ''), 'content': t.get('content', ''), 'type': t.get('trace_type', 'memory_trace'), 'intensity': t.get('strength', 0), 'score': t.get('relevance', 0), 'emotion_data': t.get('emotion_data', '{}')} for t in traces]

    def retrieve_by_conditions(self, conditions: List[str],
                                top_k: int = 5) -> List[Dict]:
        """
        Multi-condition retrieval of memory traces

        Based on four retrieval conditions:
        1. semantic_match: Semantic matching score
        2. recency: Temporal proximity
        3. strength: Trace strength
        4. context_match: Context matching score
        """
        if not conditions:
            return []

        conn = self._connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT *, emotion_data FROM memory_traces
                WHERE memory_state = 'active' AND strength > 0.05
                ORDER BY strength DESC
            ''')
            all_traces = [self._row_to_trace(row) for row in cursor.fetchall()]
        finally:
            conn.close()

        results = []
        now = time.time()

        for trace in all_traces:
            # Calculate retrieval condition matching
            retrieval_scores = self._calculate_retrieval_scores(trace, conditions, now)

            if retrieval_scores['overall'] < 0.4:
                continue

            # Calculate relevance score
            effective_strength = trace.strength * retrieval_scores['decay_factor']
            relevance = (
                effective_strength *
                retrieval_scores['overall'] *
                (1.0 + trace.self_relevance * 0.5)
            )

            if relevance < 0.01:
                continue

            results.append({
                'trace_id': trace.trace_id,
                'db_id': trace.db_id,
                'content': trace.content_summary,
                'trace_type': trace.trace_type,
                'relevance': round(relevance, 4),
                'strength': round(trace.strength, 4),
                'retrieval_scores': {k: round(v, 4) for k, v in retrieval_scores.items()},
                'self_relevance': round(trace.self_relevance, 4),
                'access_count': trace.access_count,
                'emotion_data': row['emotion_data'] if 'emotion_data' in row.keys() else '{}',
            })

        results.sort(key=lambda r: r['relevance'], reverse=True)
        top_results = results[:top_k]

        # Update retrieved traces
        if top_results:
            self._update_after_retrieval([r['db_id'] for r in top_results], conditions)

        return top_results

    def _calculate_retrieval_scores(self, trace: MemoryTrace,
                                     conditions: List[str], now: float) -> Dict:
        """Calculate multi-condition retrieval scores"""
        import re

        scores = {}
        condition_text = ' '.join(conditions)
        condition_tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', condition_text)

        # 1. Semantic matching
        semantic = self._check_semantic_match(trace.condition_pattern, condition_text, condition_tokens)
        scores['semantic'] = semantic

        # 2. Temporal proximity
        recency = self._check_recency(trace.last_accessed, now)
        scores['recency'] = recency

        # 3. Context matching
        context = self._check_context_match(trace.condition_pattern, condition_text, condition_tokens)
        scores['context'] = context

        # 4. Enhancement conditions
        enhancement = self._check_enhancement(trace.condition_pattern, conditions)
        scores['enhancement'] = enhancement

        # Decay factor
        time_since = now - trace.last_accessed if trace.last_accessed > 0 else float('inf')
        decay = self._strength_decay(time_since)
        scores['decay_factor'] = round(decay, 4)

        # Weighted average × square root of minimum
        valid_scores = [scores['semantic'], scores['recency'], scores['context'], scores['enhancement']]
        avg = sum(valid_scores) / len(valid_scores)
        minimum = min(valid_scores)
        scores['overall'] = round(avg * (minimum ** 0.5), 4)

        return scores

    def _check_semantic_match(self, condition_pattern: Dict, condition_text: str,
                               condition_tokens: List[str] = None) -> float:
        """Semantic matching score"""
        primary = condition_pattern.get('primary', {})
        keywords = primary.get('keywords', [])

        if not keywords:
            return 0.5

        if condition_tokens is None:
            import re
            condition_tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', condition_text)

        def _match(kw: str) -> bool:
            if kw in condition_text:
                return True
            return any(t in kw for t in condition_tokens)

        matches = sum(1 for kw in keywords if _match(kw))
        return min(1.0, matches / max(1, len(keywords)))

    def _check_recency(self, last_accessed: float, now: float) -> float:
        """Temporal proximity"""
        if last_accessed <= 0:
            return 0.7  # New trace

        gap = now - last_accessed
        if gap < 3600:
            return 0.9
        elif gap < 86400:
            return 0.6
        elif gap < 604800:
            return 0.3
        else:
            return 0.1

    def _check_context_match(self, condition_pattern: Dict, condition_text: str,
                              condition_tokens: List[str] = None) -> float:
        """Context matching score"""
        obj = condition_pattern.get('object', {})
        keywords = obj.get('keywords', [])

        if not keywords:
            return 0.5

        if condition_tokens is None:
            import re
            condition_tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', condition_text)

        def _match(kw: str) -> bool:
            if kw in condition_text:
                return True
            return any(t in kw for t in condition_tokens)

        matches = sum(1 for kw in keywords if _match(kw))
        return min(1.0, matches / max(1, len(keywords)))

    def _check_enhancement(self, condition_pattern: Dict, conditions: List[str]) -> float:
        """Enhancement condition matching"""
        enh = condition_pattern.get('enhancement', {})
        emotions = enh.get('emotions', [])
        contexts = enh.get('contexts', [])

        if not emotions and not contexts:
            return 0.5

        condition_text = ' '.join(conditions).lower()
        matches = sum(1 for kw in emotions + contexts if kw.lower() in condition_text)
        total = len(emotions) + len(contexts)

        return min(1.0, matches / max(1, total)) if total > 0 else 0.5

    def _assess_impact(self, emotion: str = None) -> LongTermImpact:
        """Assess long-term impact"""
        if not emotion:
            return LongTermImpact()

        # Simplified emotion mapping
        positive = {'joy', 'gratitude', 'love', 'happy', '快乐', '感恩', '喜悦'}
        negative = {'anger', 'fear', 'sad', '愤怒', '恐惧', '悲伤'}
        authentic = {'honest', 'sincere', '真诚', '诚实', '坦率'}
        engaged = {'focused', 'engaged', '专注', '投入', '认真'}

        emotion_lower = emotion.lower()

        return LongTermImpact(
            social_valence=0.5 if any(e in emotion_lower for e in positive) else (-0.5 if any(e in emotion_lower for e in negative) else 0.0),
            authenticity=0.3 if any(e in emotion_lower for e in authentic) else 0.0,
            engagement=0.4 if any(e in emotion_lower for e in engaged) else 0.0,
            coherence=0.0,
        )

    def _extract_condition_pattern(self, content: str, context: Dict) -> Dict:
        """Extract condition pattern"""
        import re
        tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{2,}', content)

        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        keywords = sorted(freq.items(), key=lambda x: -x[1])[:10]
        primary_keywords = [k for k, v in keywords]
        object_keywords = [k for k, v in keywords if v >= 1][:5]

        emotions = context.get('emotions', [])
        contexts = context.get('contexts', [])

        return {
            'primary': {'keywords': primary_keywords},
            'object': {'keywords': object_keywords},
            'enhancement': {'emotions': emotions, 'contexts': contexts},
        }

    def _find_similar_trace(self, condition_pattern: Dict) -> Optional[Dict]:
        """Find similar trace"""
        if not self.db_path:
            return None

        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM memory_traces
                WHERE condition_pattern IS NOT NULL AND condition_pattern != '{}'
                ORDER BY strength DESC LIMIT 20
            ''')
            rows = cursor.fetchall()

            for row in rows:
                trace = dict(row)
                try:
                    existing_pattern = json.loads(trace['condition_pattern'])
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

                new_keywords = set(condition_pattern.get('primary', {}).get('keywords', []))
                old_keywords = set(existing_pattern.get('primary', {}).get('keywords', []))
                overlap = new_keywords & old_keywords

                if len(overlap) >= min(3, len(new_keywords)):
                    return trace

            return None
        finally:
            conn.close()

    def _row_to_trace(self, row) -> MemoryTrace:
        """Convert database row to MemoryTrace"""
        access_history = []
        if row['access_history']:
            try:
                access_history = json.loads(row['access_history'])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        condition_pattern = {}
        if row['condition_pattern']:
            try:
                condition_pattern = json.loads(row['condition_pattern'])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        long_term_impact = LongTermImpact()
        if row['long_term_impact']:
            try:
                long_term_impact = LongTermImpact.from_dict(json.loads(row['long_term_impact']))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        return MemoryTrace(
            trace_id=row['trace_id'] or hashlib.md5(f"trace_{row['id']}".encode()).hexdigest()[:16],
            trace_type=row['trace_type'] or 'episodic',
            strength=row['strength'] if row['strength'] is not None else 0.1,
            long_term_impact=long_term_impact,
            condition_pattern=condition_pattern,
            access_history=access_history,
            self_relevance=row['self_relevance'] if row['self_relevance'] is not None else 0.0,
            last_accessed=row['last_accessed'] if row['last_accessed'] is not None else 0.0,
            access_count=row['access_count'] if row['access_count'] is not None else 0,
            is_active=bool(row['is_active']),
            db_id=row['id'],
            content_summary=row['content_summary'] or "",
        )

    def _update_after_retrieval(self, db_ids: List[int], conditions: List[str]):
        """Update traces after retrieval"""
        if not db_ids or not self.db_path:
            return

        conn = self._connect()
        try:
            cursor = conn.cursor()

            for db_id in db_ids:
                cursor.execute('''
                    UPDATE memory_traces
                    SET strength = MAX(0.05, strength * 0.95),
                        last_accessed = ?,
                        access_count = access_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (time.time(), db_id))

            conn.commit()
        finally:
            conn.close()

    def _strength_decay(self, time_since: float) -> float:
        """Strength decay"""
        days = time_since / 86400
        return (1 + days) ** -0.3

    # ------------------------------------------------------------------ #
    # Soft Delete Forgetting Mechanism (inspired by Kimi Claw's memory decay model)
    # ------------------------------------------------------------------ #

    def soft_forget(self, strength_threshold=0.05, dormant_days=30, forget_days=90):
        """
        Soft delete decaying memory traces: active → dormant → forgotten

        Three-stage forgetting:
        1. active → dormant: strength < threshold and not accessed for dormant_days
        2. dormant → forgotten: dormant for more than forget_days
        3. forgotten: not auto-deleted, can be manually purged with purge_forgotten()

        Difference from hard delete: dormant/forgotten traces can be recovered under specific conditions,
        closer to human memory's "forgotten memories can be triggered back" feature.

        Args:
            strength_threshold: Strength threshold for entering dormant (default: 0.05)
            dormant_days: Minimum unaccessed days to enter dormant (default: 30 days)
            forget_days: Minimum dormant days to enter forgotten (default: 90 days)

        Returns:
            dict: {'active_to_dormant': int, 'dormant_to_forgotten': int}
        """
        if not self.db_path:
            return {'active_to_dormant': 0, 'dormant_to_forgotten': 0}

        now = time.time()
        conn = self._connect()
        try:
            cursor = conn.cursor()

            # Stage 1: active → dormant
            # Condition: strength < threshold and not accessed for dormant_days
            cursor.execute('''
                UPDATE memory_traces
                SET memory_state = 'dormant',
                    dormant_since = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE memory_state = 'active'
                  AND strength < ?
                  AND (last_accessed = 0 OR (? - last_accessed) > ?)
            ''', (now, strength_threshold, now, dormant_days * 86400))
            active_to_dormant = cursor.rowcount

            # Stage 2: dormant → forgotten
            cursor.execute('''
                UPDATE memory_traces
                SET memory_state = 'forgotten',
                    updated_at = CURRENT_TIMESTAMP
                WHERE memory_state = 'dormant'
                  AND (dormant_since = 0 OR (? - dormant_since) > ?)
            ''', (now, forget_days * 86400))
            dormant_to_forgotten = cursor.rowcount

            conn.commit()
            result = {'active_to_dormant': active_to_dormant, 'dormant_to_forgotten': dormant_to_forgotten}
            if active_to_dormant > 0 or dormant_to_forgotten > 0:
                print(f"[PersonaProfile] Soft-forget: {active_to_dormant} active→dormant, "
                      f"{dormant_to_forgotten} dormant→forgotten")
            return result
        finally:
            conn.close()

    def recover_trace(self, db_id: int) -> bool:
        """
        Recover from dormant/forgotten state to active (memory recall)

        Simulates human memory feature: a supposedly forgotten memory can be reactivated by cues.
        After recovery, strength increases slightly (+0.1), simulating reconsolidation effect.
        """
        if not self.db_path:
            return False

        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE memory_traces
                SET memory_state = 'active',
                    dormant_since = 0,
                    strength = MIN(1.0, strength + 0.1),
                    last_accessed = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND memory_state IN ('dormant', 'forgotten')
            ''', (time.time(), db_id))
            conn.commit()
            affected = cursor.rowcount
            if affected > 0:
                print(f"[PersonaProfile] Recovered trace id={db_id} (strength +0.1)")
            return affected > 0
        finally:
            conn.close()

    def purge_forgotten(self, days=180):
        """Hard delete forgotten traces older than specified days (manual call required, irreversible)"""
        if not self.db_path:
            return 0

        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM memory_traces
                WHERE memory_state = 'forgotten'
                  AND (? - last_accessed) > ?
            ''', (time.time(), days * 86400))
            conn.commit()
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"[PersonaProfile] Purged {deleted} forgotten traces older than {days} days")
            return deleted
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Graph Association Enhancement (Improvement 2: inspired by Kimi Claw's relation types + dynamic strength management)
    # ------------------------------------------------------------------ #

    RELATION_TYPE_WEIGHTS = {
        'causal': 0.8,      # Causal relationship has highest weight
        'contrast': 0.7,    # Contrast relationship second
        'temporal': 0.6,    # Temporal relationship
        'similar': 0.5,     # Similar relationship
        'associated': 0.4,  # General association has lowest weight
        'related': 0.3,     # Compatible with old default type
    }

    def create_trace_link(self, source_trace_id: int, target_trace_id: int,
                          relation_type: str = 'associated',
                          strength: float = None, context: str = '') -> bool:
        """
        Create association between memory traces

        Args:
            source_trace_id: Source trace database ID
            target_trace_id: Target trace database ID
            relation_type: Relationship type (causal/associated/similar/contrast/temporal)
            strength: Association strength, defaults to auto-set based on relation type
            context: Association context description

        Returns:
            bool: Whether creation was successful
        """
        if not self.db_path or source_trace_id == target_trace_id:
            return False

        # Auto-set strength (based on relation type default weight)
        if strength is None:
            strength = self.RELATION_TYPE_WEIGHTS.get(relation_type, 0.4)

        conn = self._connect()
        try:
            cursor = conn.cursor()
            # Check if same-type association already exists
            cursor.execute('''
                SELECT id, strength FROM trace_links
                WHERE source_trace_id = ? AND target_trace_id = ? AND relation_type = ?
            ''', (source_trace_id, target_trace_id, relation_type))

            existing = cursor.fetchone()
            if existing:
                # Already exists: enhance strength (take larger value, simulate dynamic reinforcement)
                old_strength = existing['strength'] if hasattr(existing, '__getitem__') else existing[1]
                new_strength = max(old_strength, strength)
                cursor.execute('''
                    UPDATE trace_links
                    SET strength = ?, context = ?, created_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (new_strength, context, existing['id'] if hasattr(existing, '__getitem__') else existing[0]))
            else:
                cursor.execute('''
                    INSERT INTO trace_links (source_trace_id, target_trace_id, relation_type, strength, context)
                    VALUES (?, ?, ?, ?, ?)
                ''', (source_trace_id, target_trace_id, relation_type, strength, context))

            conn.commit()
            return True
        except Exception as e:
            print(f"[PersonaProfile] create_trace_link error: {e}")
            return False
        finally:
            conn.close()

    def get_linked_traces(self, trace_id: int, relation_type: str = None,
                          min_strength: float = 0.2, limit: int = 10) -> List[Dict]:
        """
        Get other traces associated with the specified trace

        Args:
            trace_id: Trace database ID
            relation_type: Filter by relation type (None=all types)
            min_strength: Minimum strength threshold
            limit: Maximum number of results

        Returns:
            List[Dict]: Associated trace list, including relation_type and strength
        """
        if not self.db_path:
            return []

        conn = self._connect()
        try:
            cursor = conn.cursor()
            if relation_type:
                cursor.execute('''
                    SELECT tl.relation_type, tl.strength, tl.context,
                           mt.id, mt.trace_id, mt.content_summary, mt.trace_type, mt.strength as trace_strength
                    FROM trace_links tl
                    JOIN memory_traces mt ON (
                        (tl.target_trace_id = mt.id AND tl.source_trace_id = ?)
                        OR (tl.source_trace_id = mt.id AND tl.target_trace_id = ?)
                    )
                    WHERE tl.relation_type = ? AND tl.strength >= ? AND mt.memory_state = 'active'
                    ORDER BY tl.strength DESC LIMIT ?
                ''', (trace_id, trace_id, relation_type, min_strength, limit))
            else:
                cursor.execute('''
                    SELECT tl.relation_type, tl.strength, tl.context,
                           mt.id, mt.trace_id, mt.content_summary, mt.trace_type, mt.strength as trace_strength
                    FROM trace_links tl
                    JOIN memory_traces mt ON (
                        (tl.target_trace_id = mt.id AND tl.source_trace_id = ?)
                        OR (tl.source_trace_id = mt.id AND tl.target_trace_id = ?)
                    )
                    WHERE tl.strength >= ? AND mt.memory_state = 'active'
                    ORDER BY tl.strength DESC LIMIT ?
                ''', (trace_id, trace_id, min_strength, limit))

            results = []
            for row in cursor.fetchall():
                r = dict(row)
                results.append({
                    'db_id': r.get('id'),
                    'trace_id': r.get('trace_id'),
                    'content': r.get('content_summary', ''),
                    'trace_type': r.get('trace_type'),
                    'trace_strength': round(r.get('trace_strength', 0), 4),
                    'relation_type': r.get('relation_type'),
                    'link_strength': round(r.get('strength', 0), 4),
                    'context': r.get('context', ''),
                })
            return results
        finally:
            conn.close()

    def strengthen_links(self, trace_id: int, boost: float = 0.05) -> int:
        """
        Dynamically strengthen all association edges related to the specified trace

        Simulates the association enhancement effect during memory retrieval:
        Every time a trace is retrieved, its associated edges are slightly enhanced,
        associations not retrieved for a long time naturally decay.

        Args:
            trace_id: Trace database ID
            boost: Enhancement magnitude (default: 0.05)

        Returns:
            int: Number of affected association edges
        """
        if not self.db_path:
            return 0

        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE trace_links
                SET strength = MIN(1.0, strength + ?)
                WHERE source_trace_id = ? OR target_trace_id = ?
            ''', (boost, trace_id, trace_id))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def decay_links(self, days: float = 30, decay_rate: float = 0.95) -> int:
        """
        Decay association edge strength for edges not retrieved for a long time

        Args:
            days: How many days without update before edges decay (default: 30 days)
            decay_rate: Decay coefficient (default: 0.95, i.e., 5% decay each time)

        Returns:
            int: Number of affected association edges
        """
        if not self.db_path:
            return 0

        conn = self._connect()
        try:
            cursor = conn.cursor()
            threshold = time.time() - days * 86400
            cursor.execute('''
                UPDATE trace_links
                SET strength = strength * ?
                WHERE julianday('now') - julianday(created_at) > ?
                  AND strength > 0.1
            ''', (decay_rate, days))
            conn.commit()
            affected = cursor.rowcount
            if affected > 0:
                # Clean up edges with too low strength
                cursor.execute('DELETE FROM trace_links WHERE strength < 0.05')
                deleted = cursor.rowcount
                if deleted > 0:
                    print(f"[PersonaProfile] Decayed {affected} links, purged {deleted} weak links")
            return affected
        finally:
            conn.close()

    def infer_relation_type(self, content_a: str, content_b: str) -> str:
        """
        Infer relationship type between two memory contents

        Based on simple keyword matching heuristics:
        - Causal words (because/lead to/therefore/due to) → causal
        - Contrast words (but/however/opposite/different) → contrast
        - Similar words (similar/same/alike/identical) → similar
        - Temporal words (before/after/then/next) → temporal
        - Others → associated
        """
        combined = content_a + ' ' + content_b

        causal_words = ['因为', '导致', '所以', '由于', '引起', '造成', '原因', '结果',
                       'because', 'cause', 'result', 'therefore', 'lead to']
        contrast_words = ['但是', '然而', '相反', '对比', '不同', '差异', '矛盾',
                         'but', 'however', 'contrast', 'opposite', 'different']
        similar_words = ['类似', '相同', '相似', '一致', '一样', '同样',
                        'similar', 'same', 'alike', 'equivalent']
        temporal_words = ['之前', '之后', '接着', '随后', '然后', '同时', '后来',
                         'before', 'after', 'then', 'next', 'followed', 'later']

        scores = {
            'causal': sum(1 for w in causal_words if w in combined),
            'contrast': sum(1 for w in contrast_words if w in combined),
            'similar': sum(1 for w in similar_words if w in combined),
            'temporal': sum(1 for w in temporal_words if w in combined),
        }

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else 'associated'

    def get_all_traces(self, limit: int = 100) -> List[Dict]:
        """Get all traces"""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memory_traces ORDER BY strength DESC LIMIT ?', (limit,))
            traces = [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
        return traces

    def get_traces_by_type(self, trace_type: str, limit: int = 100) -> List[Dict]:
        """Get traces by type"""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM memory_traces
                WHERE trace_type = ?
                ORDER BY strength DESC
                LIMIT ?
            ''', (trace_type, limit))
            traces = [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
        return traces

    def get_trace(self, trace_id) -> Optional[Dict]:
        """Get single trace"""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memory_traces WHERE trace_id = ? OR id = ?', (str(trace_id), trace_id))
            row = cursor.fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def update_trace(self, trace_id, **kwargs) -> bool:
        """Update trace"""
        conn = self._connect()
        try:
            cursor = conn.cursor()

            updates = []
            params = []

            for key, value in kwargs.items():
                if key in ['strength', 'self_relevance', 'is_active']:
                    updates.append(f"{key} = ?")
                    params.append(value)

            if updates:
                trace_val = int(trace_id) if isinstance(trace_id, int) else trace_id
                sql = f"UPDATE memory_traces SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE id = ? OR trace_id = ?"
                cursor.execute(sql, params + [trace_val, trace_val])
                conn.commit()
                affected = cursor.rowcount
                return affected > 0

            return False
        finally:
            conn.close()

    def delete_trace(self, trace_id) -> bool:
        """Delete trace"""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM memory_traces WHERE id = ? OR trace_id = ?', (trace_id, trace_id))
            conn.commit()
            affected = cursor.rowcount
            return affected > 0
        finally:
            conn.close()

    def get_current_persona(self) -> Persona:
        """
        Get current personality (based on memory trace analysis)

        Priority:
        1. LLM-inferred OCEAN scores stored in memory_traces.context (most accurate)
        2. Keyword fallback (conservative estimate when original traces have no OCEAN data)
        """
        if not self.db_path:
            return Persona()

        conn = self._connect()
        try:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM memory_traces
                WHERE memory_state = 'active'
                ORDER BY strength DESC
            ''')
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return Persona()

        # ── Priority 1: LLM-inferred OCEAN scores (from condition_pattern.ocean) ──
        dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        ocean_scores = {d: [] for d in dims}

        import json as _json
        for row in rows:
            raw_pattern = row['condition_pattern']
            if raw_pattern:
                try:
                    pattern = _json.loads(raw_pattern) if isinstance(raw_pattern, str) else (raw_pattern or {})
                except Exception:
                    pattern = {}
            else:
                pattern = {}

            ocean_data = pattern.get('ocean') if isinstance(pattern, dict) else None
            if ocean_data and isinstance(ocean_data, dict):
                for dim in dims:
                    raw = ocean_data.get(dim)
                    if raw is not None:
                        try:
                            score = max(0.0, min(1.0, float(raw)))
                            strength = row['strength'] or 0.5
                            ocean_scores[dim].append(score * strength)
                        except (TypeError, ValueError):
                            pass

        # ── Priority 2: Keyword fallback ──
        # Only for keywords explicitly appearing in trace content (not LLM-rewritten)
        ocean_keywords = {
            'openness': ['好奇', '创造', '想象', '艺术', '创新', '探索', '开放', '新颖',
                        'curious', 'creative', 'imagine', 'art', 'innovation', 'explore', 'open'],
            'conscientiousness': ['认真', '负责', '自律', '条理', '计划', '准时', '可靠', '勤奋',
                                 'careful', 'responsible', 'discipline', 'organized', 'plan', 'reliable', 'diligent'],
            'extraversion': ['社交', '活跃', '热情', '外向', '健谈', '自信', '乐观', '积极',
                            'social', 'active', 'enthusiastic', 'outgoing', 'talkative', 'confident', 'optimistic'],
            'agreeableness': ['友善', '合作', '信任', '体贴', '帮助', '同情', '宽容', '和谐',
                             'friendly', 'cooperative', 'trust', 'considerate', 'help', 'empathy', 'harmony'],
            'neuroticism': ['焦虑', '紧张', '担忧', '敏感', '情绪化', '压力', '不安', '消极',
                           'anxiety', 'nervous', 'worry', 'sensitive', 'emotional', 'stress', 'negative']
        }

        keyword_scores = {d: [] for d in dims}
        for row in rows:
            content = (row['content_summary'] or '').lower()
            strength = row['strength'] or 0.1
            for dim, keywords in ocean_keywords.items():
                for kw in keywords:
                    if kw in content:
                        keyword_scores[dim].append(min(1.0, content.count(kw) * strength * 0.5))

        # ── Calculate final scores ──
        def calc_dimension(ocean_list, kw_list):
            # Use weighted average when LLM OCEAN data exists, otherwise use keywords
            if ocean_list:
                return max(0.1, min(0.9, sum(ocean_list) / len(ocean_list)))
            if kw_list:
                return max(0.1, min(0.9, sum(kw_list) / len(kw_list)))
            return 0.5

        return Persona(
            openness=calc_dimension(ocean_scores['openness'], keyword_scores['openness']),
            conscientiousness=calc_dimension(ocean_scores['conscientiousness'], keyword_scores['conscientiousness']),
            extraversion=calc_dimension(ocean_scores['extraversion'], keyword_scores['extraversion']),
            agreeableness=calc_dimension(ocean_scores['agreeableness'], keyword_scores['agreeableness']),
            neuroticism=calc_dimension(ocean_scores['neuroticism'], keyword_scores['neuroticism'])
        )

    # ------------------------------------------------------------------ #
    # OCEAN Online Dynamic Update (allows personality to evolve naturally with dialogue)
    # ------------------------------------------------------------------ #

    def update_ocean_from_interaction(
        self,
        ocean_delta: Dict[str, float],
        interaction_text: str = "",
        learning_rate: float = 0.05,
        decay_existing: float = 0.98,
    ) -> bool:
        """
        Online fine-tune OCEAN scores (real-time update at dialogue level)

        Principles:
        - Do not directly modify historical traces (avoid data pollution)
        - Create a new trace with type='online_update', low strength (0.15)
        - get_current_persona() automatically includes it in weighted average
        - Effect: After multiple interactions, OCEAN scores slowly drift in that direction

        Args:
            ocean_delta:      Change amount for each dimension, e.g., {"openness": 0.1, "neuroticism": -0.05}
                              Only pass dimensions to update, unpassed dimensions stay neutral 0.5
            interaction_text: Interaction text that triggered this update (for generating content_summary)
            learning_rate:    Learning rate (0-1), controls single update magnitude, default 0.05
            decay_existing:   Slight decay for existing online_update traces (prevent accumulation), default 0.98

        Returns:
            bool: Whether update trace was successfully created

        Usage Example:
            # Detected user showing curiosity in dialogue, fine-tune openness +0.1
            persona.update_ocean_from_interaction(
                ocean_delta={"openness": 0.1},
                interaction_text="User asked many deep questions, showing strong curiosity",
            )
        """
        if not self.db_path:
            return False

        # Current personality as baseline
        current = self.get_current_persona().to_dict()

        # Calculate updated OCEAN scores (current value + delta with learning rate)
        dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        updated_ocean = {}
        for dim in dims:
            base = current.get(dim, 0.5)
            delta = ocean_delta.get(dim, 0.0)
            # Update with learning rate, clip to [0.1, 0.9]
            updated_ocean[dim] = max(0.1, min(0.9, base + delta * learning_rate))

        # Slight decay for existing online_update traces (optional, prevent old updates from accumulating too much)
        if decay_existing < 1.0:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE memory_traces
                    SET strength = MAX(0.05, strength * ?)
                    WHERE trace_type = 'online_update' AND is_active = 1
                ''', (decay_existing,))
                conn.commit()
            finally:
                conn.close()

        # Generate summary for this update
        changed = [f"{dim}→{updated_ocean[dim]:.2f}" for dim in dims
                   if abs(ocean_delta.get(dim, 0.0)) > 0.001]
        summary = f"[Online Persona Update] {', '.join(changed)}"
        if interaction_text:
            summary += f" | Trigger: {interaction_text[:80]}"

        # Store as online_update trace (low strength, prevent single update from being too significant)
        condition_pattern = {
            'primary': {'keywords': ['在线更新', 'online_update', 'persona']},
            'object': {'keywords': changed[:5]},
            'enhancement': {},
            'ocean': updated_ocean,   # Key: OCEAN injection, get_current_persona() will read it
        }

        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO memory_traces
                (trace_id, content_summary, trace_type, strength, long_term_impact,
                 condition_pattern, access_history, self_relevance,
                 last_accessed, access_count, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (
                hashlib.md5(f"online_update_{time.time()}".encode()).hexdigest()[:16],
                summary[:200],
                'online_update',
                0.15,   # Low strength, prevent single update weight from being too large
                json.dumps({'net_impact': 0.1}, ensure_ascii=False),
                json.dumps(condition_pattern, ensure_ascii=False),
                json.dumps([{'timestamp': time.time(), 'type': 'online_update'}], ensure_ascii=False),
                0.3,    # Medium self_relevance
                time.time(),
                1,
                1,
            ))
            conn.commit()
            new_id = cursor.lastrowid
            print(f"[PersonaProfile] OCEAN online update: id={new_id}, changes={changed}")
            return True
        except Exception as e:
            print(f"[PersonaProfile] OCEAN update failed: {e}")
            return False
        finally:
            conn.close()
