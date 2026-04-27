"""
Type definitions for the Self-Evolving Mode.

Defines all TypedDict structures used across self_evolve_types, self_evolve_io,
self_evolve_feedback, and self_evolve_coordinator modules.
"""

from typing import TypedDict, NotRequired, List, Dict, Any, Optional


class JudgeEpisodeInput(TypedDict, total=False):
    """Input structure for evaluating a single episode via MobileJudge."""

    task_id: str
    instruction: str
    snapshot: str
    current_screenshot_paths: List[str]
    final_screenshot_path: str
    reference_screenshot_paths: List[str]
    reference_instruction: str
    action_history: List[str]
    action_thoughts: NotRequired[List[str]]
    env_signals: NotRequired[Dict[str, Any]]


class JudgeEpisodeResult(TypedDict, total=False):
    """Output structure from MobileJudge evaluation."""

    task_id: str
    reward: float
    success: bool
    feedback_summary: str
    feedback_strings: List[str]
    failure_reasons: List[str]
    dimension_scores: NotRequired[Dict[str, float]]
    raw_judge_text: str
    usage: NotRequired[Dict[str, int]]


class TaskPerformance(TypedDict, total=False):
    """Single task performance aggregated over one round."""

    task_id: str
    snapshot: str
    instruction: str
    success_rate: float
    avg_reward: float
    total_episodes: int
    success_episodes: int
    feedback_strings: List[str]
    failure_reasons: List[str]


class RoundFeedback(TypedDict, total=False):
    """Aggregated feedback for a complete round, consumed by CurriculumTaskGenerator."""

    round_id: int
    total_tasks: int
    task_performances: List[TaskPerformance]
    overall_success_rate: float
    overall_avg_reward: float
    exploration_data_paths: List[str]


class ExplorationTaskContext(TypedDict, total=False):
    """Standardized exploration data consumed by CurriculumTaskGenerator."""

    exploration_id: str
    environment: str
    action_sequence: str
    screenshots: List[str]
    discovered_apps: List[str]
    init_result: NotRequired[Dict[str, Any]]


class GeneratorOutput(TypedDict, total=False):
    """Result of one curriculum generation round."""

    generated_task_files: List[str]
    round_id: int
    feedback_ref: str
    total_tasks: int


class RoundUpdateResult(TypedDict, total=False):
    """Result of a complete round update operation."""

    success: bool
    round_id: int
    feedback_file: str
    generated_task_files: List[str]
    parquet_file: str
    error: Optional[str]
