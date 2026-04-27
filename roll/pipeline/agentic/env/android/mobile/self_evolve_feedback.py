"""
Round feedback aggregation for Self-Evolving Mode.

Aggregates episode-level judge results into a round-level feedback file
consumed by CurriculumTaskGenerator.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from roll.utils.logging import get_logger

from .self_evolve_io import read_judge_result, scan_judge_results, write_round_feedback
from .self_evolve_types import JudgeEpisodeResult, RoundFeedback, TaskPerformance

logger = get_logger()


def build_round_feedback(
    round_dir: Path,
    output_path: Optional[Path] = None,
    task_instructions: Optional[Dict[str, str]] = None,
    task_snapshots: Optional[Dict[str, str]] = None,
) -> RoundFeedback:
    """Aggregate all judge_result.json files in round_dir into round_feedback.json.

    Args:
        round_dir: Directory containing episode subdirectories with judge_result.json
        output_path: Optional explicit output path. Defaults to round_dir / "round_feedback.json"
        task_instructions: Optional dict mapping task_id -> instruction text
        task_snapshots: Optional dict mapping task_id -> snapshot string

    Returns:
        RoundFeedback dict written to disk
    """
    judge_results = scan_judge_results(round_dir)
    if not judge_results:
        logger.warning(f"No judge results found in {round_dir}")
        feedback: RoundFeedback = {
            "round_id": 0,
            "total_tasks": 0,
            "task_performances": [],
            "overall_success_rate": 0.0,
            "overall_avg_reward": 0.0,
            "exploration_data_paths": [],
        }
    else:
        feedback = _aggregate_judge_results(judge_results, task_instructions, task_snapshots)

    if output_path is None:
        output_path = round_dir / "round_feedback.json"

    write_round_feedback(round_dir, feedback)
    return feedback


def _aggregate_judge_results(
    results: List[JudgeEpisodeResult],
    task_instructions: Optional[Dict[str, str]] = None,
    task_snapshots: Optional[Dict[str, str]] = None,
) -> RoundFeedback:
    """Aggregate a list of episode-level judge results into task-level performances."""
    task_instructions = task_instructions or {}
    task_snapshots = task_snapshots or {}

    # Group results by task_id
    task_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "rewards": [],
            "successes": [],
            "feedback_strings": [],
            "failure_reasons": [],
        }
    )

    for result in results:
        task_id = result.get("task_id", "unknown")
        reward = result.get("reward", 0.0)
        success = result.get("success", False)
        feedback_strings = result.get("feedback_strings", [])
        failure_reasons = result.get("failure_reasons", [])

        task_data[task_id]["rewards"].append(reward)
        task_data[task_id]["successes"].append(success)
        task_data[task_id]["feedback_strings"].extend(feedback_strings)
        task_data[task_id]["failure_reasons"].extend(failure_reasons)

    task_performances: List[TaskPerformance] = []
    total_rewards: List[float] = []
    total_successes: List[bool] = []

    for task_id, data in task_data.items():
        rewards = data["rewards"]
        successes = data["successes"]
        total_episodes = len(rewards)
        success_episodes = sum(successes)
        success_rate = success_episodes / total_episodes if total_episodes > 0 else 0.0
        avg_reward = sum(rewards) / total_episodes if total_episodes > 0 else 0.0

        total_rewards.extend(rewards)
        total_successes.extend(successes)

        performance: TaskPerformance = {
            "task_id": task_id,
            "snapshot": task_snapshots.get(task_id, ""),
            "instruction": task_instructions.get(task_id, ""),
            "success_rate": round(success_rate, 4),
            "avg_reward": round(avg_reward, 4),
            "total_episodes": total_episodes,
            "success_episodes": success_episodes,
            "feedback_strings": _deduplicate_strings(data["feedback_strings"])[:10],
            "failure_reasons": _deduplicate_strings(data["failure_reasons"]),
        }
        task_performances.append(performance)

    overall_success_rate = (
        sum(total_successes) / len(total_successes) if total_successes else 0.0
    )
    overall_avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0

    feedback: RoundFeedback = {
        "round_id": 0,
        "total_tasks": len(task_performances),
        "task_performances": task_performances,
        "overall_success_rate": round(overall_success_rate, 4),
        "overall_avg_reward": round(overall_avg_reward, 4),
        "exploration_data_paths": [],
    }
    return feedback


def _deduplicate_strings(strings: List[str]) -> List[str]:
    """Remove duplicate and empty strings, preserving order."""
    seen: set = set()
    result: List[str] = []
    for s in strings:
        stripped = s.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            result.append(stripped)
    return result


def get_task_stats_from_round_feedback(feedback: RoundFeedback) -> Dict[str, float]:
    """Extract commonly-used statistics from a RoundFeedback dict."""
    return {
        "overall_success_rate": feedback.get("overall_success_rate", 0.0),
        "overall_avg_reward": feedback.get("overall_avg_reward", 0.0),
        "total_tasks": feedback.get("total_tasks", 0),
    }
