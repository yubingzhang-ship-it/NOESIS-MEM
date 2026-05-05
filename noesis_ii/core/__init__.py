"""
NOESIS-II Core Modules - 路线A重构版

核心模块（已去术语化）：
- PersonaProfile: 人格一致性存储（替代 AlayaSeeds）
- PersonaExtractor: LLM人格提取（替代 deep_personality）
- MultiCriteriaRetriever: 多条件检索（替代 IABEngine）
- ConsistencyChecker: 一致性检查（新增）

修订历史：
  v3.0 (2026-04-10) - 路线A重构：剥离术语，专注人格一致性
"""

from .persona_profile import PersonaProfile, MemoryTrace, LongTermImpact
from .persona_extractor import PersonaExtractor, OCEANScores
from .multi_criteria_retriever import MultiCriteriaRetriever, RetrievalCriteria, RetrievalResult
from .consistency_checker import ConsistencyChecker, ConsistencyReport
from .persona_updater import PersonaUpdater, UpdateConfig
from .persona_constrained_generator import (
    PersonaConstrainedGenerator, GenerationConfig, GenerationResult,
    ValueConflictDetector
)

# 废弃模块兼容导入（带警告）
from ._deprecated import AlayaSeeds, PersonalitySeeds, IABEngine

__all__ = [
    # PersonaProfile
    'PersonaProfile',
    'MemoryTrace',
    'LongTermImpact',
    
    # PersonaExtractor
    'PersonaExtractor',
    'OCEANScores',
    
    # MultiCriteriaRetriever
    'MultiCriteriaRetriever',
    'RetrievalCriteria',
    'RetrievalResult',
    
    # ConsistencyChecker
    'ConsistencyChecker',
    'ConsistencyReport',
    
    # PersonaUpdater
    'PersonaUpdater',
    'UpdateConfig',
    
    # PersonaConstrainedGenerator
    'PersonaConstrainedGenerator',
    'GenerationConfig',
    'GenerationResult',
    'ValueConflictDetector',
    
    # 废弃兼容（将在v3.1移除）
    'AlayaSeeds',
    'PersonalitySeeds',
    'IABEngine',
]
