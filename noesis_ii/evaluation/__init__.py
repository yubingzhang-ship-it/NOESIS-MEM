"""
PersonaMem Evaluation Module

评估基准和实验框架
"""

from .persona_caption_dataset import PersonaCaptionDataset
from .evaluator import PersonaEvaluator
from .benchmark import run_benchmark

# E2 一致性实验
from .e2_question_bank import QuestionBank, Question
from .e2_response_extractor import ResponseExtractor, ExtractedResponse
from .e2_group_manager import (
    BaseGroupManager,
    BL1GroupManager,
    BL2GroupManager,
    OursGroupManager,
    GroupManagerFactory,
    SessionResult
)
from .e2_experiment_runner import (
    E2ExperimentRunner,
    ExperimentConfig,
    ExperimentResult
)

__all__ = [
    # 基础评估
    'PersonaCaptionDataset',
    'PersonaEvaluator', 
    'run_benchmark',
    
    # E2 一致性实验
    'QuestionBank',
    'Question',
    'ResponseExtractor',
    'ExtractedResponse',
    'BaseGroupManager',
    'BL1GroupManager',
    'BL2GroupManager',
    'OursGroupManager',
    'GroupManagerFactory',
    'SessionResult',
    'E2ExperimentRunner',
    'ExperimentConfig',
    'ExperimentResult',
]
