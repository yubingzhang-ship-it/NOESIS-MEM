"""
PersonaProfile - 人格一致性记忆系统的核心存储

替代原 AlayaSeeds，剥离佛教术语，专注于可验证的人格一致性存储。

核心概念：
- MemoryTrace: 记忆的基本单位，存储经验的"如何再次生成"规则
- ConditionPattern: 触发条件，不是存储内容本身
- LongTermImpact: 长期影响评估（替代 KarmaVector）
- RetrievalConditions: 多条件检索（替代四缘）

关键特性：
- 轻量级：只存储摘要和条件模式，原始内容存 LTM
- 可检索：基于多条件加权检索
- 可演化：支持记忆痕迹的增强和衰减

修订历史：
  v3.0 (2026-04-10) - 路线A重构：剥离术语，专注人格一致性
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


# ═══════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class LongTermImpact:
    """
    长期影响评估（替代 KarmaVector）
    
    多维度评估经验对系统人格的长期影响：
    - social_valence: 社交效价 [-1, 1]，正值=积极社交影响
    - authenticity: 真实性 [-1, 1]，正值=真实表达
    - engagement: 参与度 [-1, 1]，正值=深度参与
    - coherence: 一致性贡献 [-1, 1]，对人格一致性的贡献
    
    使用示例：
    - 真诚助人：LongTermImpact(0.8, 0.7, 0.6, 0.5)
    - 虚假回应：LongTermImpact(-0.3, -0.8, 0.2, -0.4)
    """
    social_valence: float = 0.0
    authenticity: float = 0.0
    engagement: float = 0.0
    coherence: float = 0.0
    
    def net_impact(self) -> float:
        """计算净影响（加权平均）"""
        return (
            self.social_valence * 0.30 +
            self.authenticity * 0.25 +
            self.engagement * 0.25 +
            self.coherence * 0.20
        )
    
    def is_positive(self) -> bool:
        """判断是否为积极影响"""
        return self.net_impact() > 0.2
    
    def is_negative(self) -> bool:
        """判断是否为消极影响"""
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
    """记忆痕迹类型"""
    PERCEPTUAL = "perceptual"
    CONCEPTUAL = "conceptual"
    EMOTIONAL = "emotional"
    ACTION = "action"
    EPISODIC = "episodic"


class RelationType(Enum):
    """记忆痕迹间关系类型（借鉴 Kimi Claw 的 5 种关系分类）
    
    causal: 因果关系（A 导致 B）
    associated: 一般关联（A 和 B 相关，但不属于以下类型）
    similar: 相似关系（A 和 B 在某方面类似）
    contrast: 对比关系（A 和 B 相对立/相反）
    temporal: 时间关系（A 发生在 B 之前/之后）
    """
    CAUSAL = "causal"
    ASSOCIATED = "associated"
    SIMILAR = "similar"
    CONTRAST = "contrast"
    TEMPORAL = "temporal"


@dataclass
class MemoryTrace:
    """
    记忆痕迹：人格一致性的基本存储单位
    
    关键特性：
    - 潜在性：未检索前无具体内容
    - 条件性：需满足条件模式才能检索
    - 可演化：检索后增强，长期未检索衰减
    """
    trace_id: str
    trace_type: str
    
    # 核心属性
    strength: float = 0.1  # 痕迹强度 [0,1]
    
    # 长期影响评估
    long_term_impact: LongTermImpact = field(default_factory=LongTermImpact)
    
    # 条件模式（触发检索的条件）
    condition_pattern: Dict = field(default_factory=dict)
    
    # 访问历史
    access_history: List[Dict] = field(default_factory=list)
    
    # 自我相关性
    self_relevance: float = 0.0
    
    # 状态
    last_accessed: float = 0.0
    access_count: int = 0
    is_active: bool = True
    
    # 数据库 ID
    db_id: Optional[int] = None
    
    # 内容摘要（用于显示）
    content_summary: str = ""


@dataclass
class Persona:
    """
    人格表示：OCEAN 五大人格维度
    
    基于大五人格模型（Big Five / OCEAN）：
    - Openness: 开放性（好奇心、创造力）
    - Conscientiousness: 尽责性（自律、条理）
    - Extraversion: 外向性（社交、活力）
    - Agreeableness: 宜人性（合作、信任）
    - Neuroticism: 神经质（情绪稳定性，低分=稳定）
    
    所有维度范围 [0, 1]，0.5 为中性基准
    """
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'openness': round(self.openness, 4),
            'conscientiousness': round(self.conscientiousness, 4),
            'extraversion': round(self.extraversion, 4),
            'agreeableness': round(self.agreeableness, 4),
            'neuroticism': round(self.neuroticism, 4),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Persona':
        """从字典创建"""
        return cls(
            openness=data.get('openness', 0.5),
            conscientiousness=data.get('conscientiousness', 0.5),
            extraversion=data.get('extraversion', 0.5),
            agreeableness=data.get('agreeableness', 0.5),
            neuroticism=data.get('neuroticism', 0.5),
        )


@dataclass
class RetrievedMemory:
    """检索结果"""
    trace: MemoryTrace
    content: Dict
    relevance_score: float
    retrieval_conditions: Dict
    timestamp: float


# ═══════════════════════════════════════════════════════════════
# 主系统
# ═══════════════════════════════════════════════════════════════

class PersonaProfile:
    """
    人格档案：持久人格表示的存储和管理
    
    核心方法：
    - store_experience() — 存储经验为记忆痕迹
    - retrieve_by_conditions() — 多条件检索
    - update_trace_strength() — 更新痕迹强度
    """
    
    def __init__(self, db_path: str = None, config: Dict = None):
        self.db_path = db_path
        self.config = config or {}
        self._trace_cache: Dict[int, MemoryTrace] = {}
        
        if db_path:
            self._init_db()
            self._migrate_db()
    
    def _init_db(self):
        """初始化数据库表"""
        if not self.db_path:
            return
        
        dir_name = os.path.dirname(self.db_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 记忆痕迹表（替代 alaya_seeds）
        # memory_state: active(正常) → dormant(休眠) → forgotten(遗忘)
        # emotion_data: 结构化情绪向量 JSON {valence, arousal, dominant, tags, narrative_hook}
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
        
        # 痕迹关联表
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
        """从旧表结构迁移"""
        if not self.db_path:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查是否存在旧表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alaya_seeds'")
        if cursor.fetchone():
            print("[PersonaProfile] Migrating from alaya_seeds to memory_traces...")
            
            # 迁移 alaya_seeds 到 memory_traces
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
                
                # 映射 seed_type 到 trace_type
                trace_type = seed_type if seed_type else 'episodic'
                
                # 映射 potency 到 strength
                strength = potency if potency else 0.5
                
                # 映射 karma_quality 到 long_term_impact
                long_term_impact = 0.5
                if karma_quality:
                    # 简单映射：善=0.8, 无记=0.5, 恶=0.2
                    karma_map = {'善': 0.8, '无记': 0.5, '恶': 0.2}
                    long_term_impact = karma_map.get(karma_quality, 0.5)
                
                # 计算 self_relevance（基于 manifestation_count）
                self_relevance = min(1.0, 0.3 + (manifestation_count or 0) * 0.05)
                
                # 插入到 memory_traces
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
        """获取数据库连接"""
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
        存储经验为记忆痕迹
        
        Args:
            experience: 原始经验内容
            summary: LLM 提取的摘要（优先使用）
            trace_type: 痕迹类型
            emotion: 情绪标签
            intensity: 强度
            context: 上下文
        
        Returns: 痕迹的数据库 ID
        """
        context = context or {}
        trace_type = trace_type or 'episodic'
        
        # 评估长期影响
        long_term_impact = self._assess_impact(emotion)
        
        # 决定存储内容
        if summary:
            trace_content = summary
        elif experience:
            trace_content = experience[:500] if len(experience) > 500 else experience
        else:
            trace_content = ""
        
        # 提取条件模式
        condition_pattern = self._extract_condition_pattern(trace_content, context)

        # OCEAN 人格数据注入 condition_pattern（存储在 condition_pattern 列，不改 schema）
        ocean_data = context.get('ocean') if isinstance(context, dict) else None
        if ocean_data and isinstance(ocean_data, dict):
            condition_pattern['ocean'] = ocean_data

        # 结构化情绪数据
        emotion_data = context.get('emotion_data') if isinstance(context, dict) else None
        if not emotion_data or not isinstance(emotion_data, dict):
            emotion_data = {'valence': 0.0, 'arousal': 0.3, 'dominant': emotion or 'neutral',
                           'tags': [emotion] if emotion else [], 'narrative_hook': ''}

        # 生成 trace_id
        pattern_str = json.dumps(condition_pattern, sort_keys=True)
        trace_id = hashlib.md5(pattern_str.encode()).hexdigest()[:16]
        
        # 检查相似痕迹
        existing = self._find_similar_trace(condition_pattern)
        
        conn = self._connect()
        try:
            cursor = conn.cursor()

            if existing:
                # 融合：增强现有痕迹
                new_strength = min(1.0, (existing['strength'] or 0.1) * 1.1 + intensity * 0.05)
                new_count = (existing['access_count'] or 0) + 1

                # 更新访问历史
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
                # 创建新痕迹
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
        """检索记忆痕迹（兼容旧retriever接口）"""
        import re
        tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', query)
        if not tokens:
            tokens = [query] if query else []
        traces = self.retrieve_by_conditions(tokens, top_k=top_k)
        # 改进3（叙事重建）：返回 emotion_data 以支持 narrative_hook 提取
        return [{'id': t.get('db_id', ''), 'content': t.get('content', ''), 'type': t.get('trace_type', 'memory_trace'), 'intensity': t.get('strength', 0), 'score': t.get('relevance', 0), 'emotion_data': t.get('emotion_data', '{}')} for t in traces]

    def retrieve_by_conditions(self, conditions: List[str],
                                top_k: int = 5) -> List[Dict]:
        """
        多条件检索记忆痕迹
        
        基于四个检索条件加权：
        1. semantic_match: 语义匹配度
        2. recency: 时间邻近度
        3. strength: 痕迹强度
        4. context_match: 上下文匹配度
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
            # 计算检索条件匹配
            retrieval_scores = self._calculate_retrieval_scores(trace, conditions, now)
            
            if retrieval_scores['overall'] < 0.4:
                continue
            
            # 计算相关性得分
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
        
        # 更新被检索的痕迹
        if top_results:
            self._update_after_retrieval([r['db_id'] for r in top_results], conditions)
        
        return top_results
    
    def _calculate_retrieval_scores(self, trace: MemoryTrace, 
                                     conditions: List[str], now: float) -> Dict:
        """计算多条件检索得分"""
        import re
        
        scores = {}
        condition_text = ' '.join(conditions)
        condition_tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', condition_text)
        
        # 1. 语义匹配
        semantic = self._check_semantic_match(trace.condition_pattern, condition_text, condition_tokens)
        scores['semantic'] = semantic
        
        # 2. 时间邻近度
        recency = self._check_recency(trace.last_accessed, now)
        scores['recency'] = recency
        
        # 3. 上下文匹配
        context = self._check_context_match(trace.condition_pattern, condition_text, condition_tokens)
        scores['context'] = context
        
        # 4. 增强条件
        enhancement = self._check_enhancement(trace.condition_pattern, conditions)
        scores['enhancement'] = enhancement
        
        # 衰减因子
        time_since = now - trace.last_accessed if trace.last_accessed > 0 else float('inf')
        decay = self._strength_decay(time_since)
        scores['decay_factor'] = round(decay, 4)
        
        # 加权平均 × 最小值平方根
        valid_scores = [scores['semantic'], scores['recency'], scores['context'], scores['enhancement']]
        avg = sum(valid_scores) / len(valid_scores)
        minimum = min(valid_scores)
        scores['overall'] = round(avg * (minimum ** 0.5), 4)
        
        return scores
    
    def _check_semantic_match(self, condition_pattern: Dict, condition_text: str,
                               condition_tokens: List[str] = None) -> float:
        """语义匹配度"""
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
        """时间邻近度"""
        if last_accessed <= 0:
            return 0.7  # 新痕迹
        
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
        """上下文匹配度"""
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
        """增强条件匹配"""
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
        """评估长期影响"""
        if not emotion:
            return LongTermImpact()
        
        # 简化的情感映射
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
        """提取条件模式"""
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
        """查找相似痕迹"""
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
        """数据库行转 MemoryTrace"""
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
        """检索后更新痕迹"""
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
        """强度衰减"""
        days = time_since / 86400
        return (1 + days) ** -0.3

    # ------------------------------------------------------------------ #
    # 软删除遗忘机制（借鉴 Kimi Claw 记忆衰减模型）
    # ------------------------------------------------------------------ #

    def soft_forget(self, strength_threshold=0.05, dormant_days=30, forget_days=90):
        """软删除衰减记忆痕迹：active → dormant → forgotten

        遗忘三阶段：
        1. active → dormant: strength < threshold 且超过 dormant_days 未被访问
        2. dormant → forgotten: 休眠超过 forget_days
        3. forgotten: 不自动删除，可手动 purge_forgotten() 清理

        与硬删除的区别：dormant/forgotten 痕迹在特定条件下可被唤醒恢复，
        更贴近人类记忆的「遗忘后可被提示唤醒」特性。

        Args:
            strength_threshold: 进入休眠的 strength 阈值（默认 0.05）
            dormant_days: 进入休眠所需的最小未访问天数（默认 30 天）
            forget_days: 进入遗忘状态所需的最小休眠天数（默认 90 天）

        Returns:
            dict: {'active_to_dormant': int, 'dormant_to_forgotten': int}
        """
        if not self.db_path:
            return {'active_to_dormant': 0, 'dormant_to_forgotten': 0}

        now = time.time()
        conn = self._connect()
        try:
            cursor = conn.cursor()

            # 阶段1：active → dormant
            # 条件：strength < threshold 且超过 dormant_days 未被访问
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

            # 阶段2：dormant → forgotten
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
        """从 dormant/forgotten 状态恢复为 active（记忆唤醒）

        模拟人类记忆的特性：一个被认为遗忘的记忆可以被线索重新激活。
        恢复后 strength 会小幅提升（+0.1），模拟再巩固效应。
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
        """硬删除 forgotten 超过指定天数的痕迹（需手动调用，不可逆）"""
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
    # 图关联增强（改进2：借鉴 Kimi Claw 的关系类型 + 动态强度管理）
    # ------------------------------------------------------------------ #

    RELATION_TYPE_WEIGHTS = {
        'causal': 0.8,      # 因果关系权重最高
        'contrast': 0.7,    # 对比关系次之
        'temporal': 0.6,    # 时间关系
        'similar': 0.5,     # 相似关系
        'associated': 0.4,  # 一般关联权重最低
        'related': 0.3,     # 兼容旧的默认类型
    }

    def create_trace_link(self, source_trace_id: int, target_trace_id: int,
                          relation_type: str = 'associated',
                          strength: float = None, context: str = '') -> bool:
        """创建记忆痕迹间的关联

        Args:
            source_trace_id: 源痕迹数据库 ID
            target_trace_id: 目标痕迹数据库 ID
            relation_type: 关系类型（causal/associated/similar/contrast/temporal）
            strength: 关联强度，默认根据关系类型自动设置
            context: 关联上下文说明

        Returns:
            bool: 是否创建成功
        """
        if not self.db_path or source_trace_id == target_trace_id:
            return False

        # 自动设置强度（根据关系类型默认权重）
        if strength is None:
            strength = self.RELATION_TYPE_WEIGHTS.get(relation_type, 0.4)

        conn = self._connect()
        try:
            cursor = conn.cursor()
            # 检查是否已存在同类型关联
            cursor.execute('''
                SELECT id, strength FROM trace_links
                WHERE source_trace_id = ? AND target_trace_id = ? AND relation_type = ?
            ''', (source_trace_id, target_trace_id, relation_type))

            existing = cursor.fetchone()
            if existing:
                # 已存在：增强强度（取较大值，模拟动态强化）
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
        """获取与指定痕迹关联的其他痕迹

        Args:
            trace_id: 痕迹数据库 ID
            relation_type: 过滤关系类型（None=所有类型）
            min_strength: 最低强度阈值
            limit: 最大返回数量

        Returns:
            List[Dict]: 关联痕迹列表，含 relation_type 和 strength
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
        """动态强化与指定痕迹相关的所有关联边

        模拟记忆检索时的关联增强效应：
        每次检索到一个痕迹，其关联边会轻微增强，长期不被检索的关联自然衰减。

        Args:
            trace_id: 痕迹数据库 ID
            boost: 增强幅度（默认 0.05）

        Returns:
            int: 受影响的关联边数量
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
        """衰减长期未被检索的关联边强度

        Args:
            days: 超过多少天未更新的边会被衰减（默认 30 天）
            decay_rate: 衰减系数（默认 0.95，即每次衰减 5%）

        Returns:
            int: 受影响的关联边数量
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
                # 清理强度过低的边
                cursor.execute('DELETE FROM trace_links WHERE strength < 0.05')
                deleted = cursor.rowcount
                if deleted > 0:
                    print(f"[PersonaProfile] Decayed {affected} links, purged {deleted} weak links")
            return affected
        finally:
            conn.close()

    def infer_relation_type(self, content_a: str, content_b: str) -> str:
        """推断两段记忆内容间的关系类型

        基于简单的关键词匹配启发式：
        - 因果词（因为/导致/所以/由于）→ causal
        - 对比词（但是/然而/相反/对比）→ contrast
        - 相似词（类似/相同/相似/一致）→ similar
        - 时间词（之前/之后/接着/随后）→ temporal
        - 其他 → associated
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
        """获取所有痕迹"""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memory_traces ORDER BY strength DESC LIMIT ?', (limit,))
            traces = [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
        return traces
    
    def get_traces_by_type(self, trace_type: str, limit: int = 100) -> List[Dict]:
        """根据类型获取痕迹"""
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
        """获取单个痕迹"""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memory_traces WHERE trace_id = ? OR id = ?', (str(trace_id), trace_id))
            row = cursor.fetchone()
        finally:
            conn.close()
        return dict(row) if row else None
    
    def update_trace(self, trace_id, **kwargs) -> bool:
        """更新痕迹"""
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
        """删除痕迹"""
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
        获取当前人格（基于记忆痕迹分析）

        优先级：
        1. memory_traces.context 中存储的 LLM 推断 OCEAN 分数（最准确）
        2. 关键词 fallback（原始痕迹没有 OCEAN 数据时的保守估算）
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

        # ── 优先级1：LLM 推断的 OCEAN 分数（来自 condition_pattern.ocean）──
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

        # ── 优先级2：关键词 fallback ──
        # 仅用于痕迹内容中明确出现的关键词（未经 LLM 改写）
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

        # ── 计算最终分数 ──
        def calc_dimension(ocean_list, kw_list):
            # 有 LLM OCEAN 数据时用加权平均，无则用关键词
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
    # OCEAN 在线动态更新（让人格随对话自然演化）
    # ------------------------------------------------------------------ #

    def update_ocean_from_interaction(
        self,
        ocean_delta: Dict[str, float],
        interaction_text: str = "",
        learning_rate: float = 0.05,
        decay_existing: float = 0.98,
    ) -> bool:
        """
        在线微调 OCEAN 分数（对话级实时更新）
        
        原理：
        - 不直接修改历史痕迹（避免数据污染）
        - 创建一条 type='online_update' 的新痕迹，strength 较低（0.15）
        - get_current_persona() 会自动将其纳入加权平均
        - 效果：多次交互后 OCEAN 分数会朝该方向缓慢偏移
        
        Args:
            ocean_delta:      各维度的变化量，例如 {"openness": 0.1, "neuroticism": -0.05}
                              只需传入要更新的维度，未传入的维度保持中性 0.5
            interaction_text: 触发此次更新的交互文本（用于生成 content_summary）
            learning_rate:    学习率（0-1），控制单次更新的幅度，默认 0.05
            decay_existing:   对已有 online_update 痕迹做轻微衰减（防止积累过多），默认 0.98
        
        Returns:
            bool: 是否成功创建更新痕迹
        
        使用示例：
            # 对话中检测到用户展示好奇心，微调 openness +0.1
            persona.update_ocean_from_interaction(
                ocean_delta={"openness": 0.1},
                interaction_text="用户问了很多深入问题，展现了强烈好奇心",
            )
        """
        if not self.db_path:
            return False

        # 当前人格作为基准
        current = self.get_current_persona().to_dict()

        # 计算更新后的 OCEAN 分数（当前值 + 带学习率的 delta）
        dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        updated_ocean = {}
        for dim in dims:
            base = current.get(dim, 0.5)
            delta = ocean_delta.get(dim, 0.0)
            # 带学习率的更新，并 clip 到 [0.1, 0.9]
            updated_ocean[dim] = max(0.1, min(0.9, base + delta * learning_rate))

        # 对已有 online_update 痕迹做轻微衰减（可选，防止旧更新积累过多影响）
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

        # 生成此次更新的摘要
        changed = [f"{dim}→{updated_ocean[dim]:.2f}" for dim in dims
                   if abs(ocean_delta.get(dim, 0.0)) > 0.001]
        summary = f"[在线人格更新] {', '.join(changed)}"
        if interaction_text:
            summary += f" | 触发: {interaction_text[:80]}"

        # 存储为 online_update 痕迹（strength 较低，避免单次更新过于显著）
        condition_pattern = {
            'primary': {'keywords': ['在线更新', 'online_update', 'persona']},
            'object': {'keywords': changed[:5]},
            'enhancement': {},
            'ocean': updated_ocean,   # 关键：OCEAN 注入，get_current_persona() 会读取
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
                0.15,   # 较低 strength，避免单次更新权重过大
                json.dumps({'net_impact': 0.1}, ensure_ascii=False),
                json.dumps(condition_pattern, ensure_ascii=False),
                json.dumps([{'timestamp': time.time(), 'type': 'online_update'}], ensure_ascii=False),
                0.3,    # self_relevance 中等
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
