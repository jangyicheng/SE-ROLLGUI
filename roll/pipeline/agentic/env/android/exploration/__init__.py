"""
AndroidWorld/MobileWorld Environment Exploration Module.

Provides:
- Environment Exploration: Free exploration of mobile GUI environments
- Task Initialization: Params persistence and deterministic task initialization
- Trajectory Formatting: Utilities for formatting exploration data

Architecture follows the self-evolving loop design in docs/self-evolve_zh.md:
    Exploration -> Curriculum Task Generation -> Training -> Evaluation -> Feedback

Key components:
- Explorer: Free exploration of mobile environments
- ParamsManager: Deterministic params generation and persistence
- TaskInitializer: Task initialization with OpenMobile-style snapshot recovery
- TrajectoryFormatter: Formatting exploration data for curriculum generator
"""

from .explorer import AndroidWorldExplorer, MobileWorldExplorer
from .model_client import ExplorerModelWrapper, VLMModelFactory
from .params_manager import AndroidWorldParamsManager, MobileWorldParamsManager
from .task_initializer import AndroidWorldTaskInitializer, MobileWorldTaskInitializer
from .trajectory_formatter import TrajectoryFormatter, format_trajectory_for_curriculum

__all__ = [
    "AndroidWorldExplorer",
    "MobileWorldExplorer",
    "ExplorerModelWrapper",
    "VLMModelFactory",
    "AndroidWorldParamsManager",
    "MobileWorldParamsManager",
    "AndroidWorldTaskInitializer",
    "MobileWorldTaskInitializer",
    "TrajectoryFormatter",
    "format_trajectory_for_curriculum",
]
