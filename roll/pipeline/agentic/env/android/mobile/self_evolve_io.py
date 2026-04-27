"""
I/O utilities for Self-Evolving Mode.

Handles:
- Atomic JSON file read/write (`.tmp` + `os.rename`)
- Directory organization for self-evolve round data
- Exploration data loading from TrajectoryFormatter output
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from roll.utils.logging import get_logger

from .self_evolve_types import (
    ExplorationTaskContext,
    JudgeEpisodeInput,
    JudgeEpisodeResult,
    RoundFeedback,
)

logger = get_logger()


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_round_dir(feedback_root: Path, round_id: int) -> Path:
    return feedback_root / f"round_{round_id:04d}"


def get_episode_dir(round_dir: Path, task_id: str, episode_id: str) -> Path:
    return ensure_dir(round_dir / "episodes" / task_id / episode_id)


# ---------------------------------------------------------------------------
# Atomic JSON I/O
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON with atomic rename (write to .tmp then rename)."""
    tmp_path = path.with_suffix(".tmp")
    ensure_dir(path.parent)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Judge I/O
# ---------------------------------------------------------------------------


def write_judge_input(episode_dir: Path, data: JudgeEpisodeInput) -> Path:
    path = episode_dir / "judge_input.json"
    atomic_write_json(path, data)
    logger.debug(f"Wrote judge input: {path}")
    return path


def read_judge_input(episode_dir: Path) -> JudgeEpisodeInput:
    return read_json(episode_dir / "judge_input.json")  # type: ignore[return-value]


def write_judge_result(episode_dir: Path, result: JudgeEpisodeResult) -> Path:
    path = episode_dir / "judge_result.json"
    atomic_write_json(path, result)
    logger.debug(f"Wrote judge result: {path}")
    return path


def read_judge_result(episode_dir: Path) -> JudgeEpisodeResult:
    return read_json(episode_dir / "judge_result.json")  # type: ignore[return-value]


def scan_judge_results(round_dir: Path) -> List[JudgeEpisodeResult]:
    """Scan all judge_result.json files under round_dir."""
    results: List[JudgeEpisodeResult] = []
    for path in round_dir.rglob("judge_result.json"):
        try:
            results.append(read_json(path))  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Failed to read judge result {path}: {e}")
    return results


# ---------------------------------------------------------------------------
# Round feedback I/O
# ---------------------------------------------------------------------------


def write_round_feedback(round_dir: Path, feedback: RoundFeedback) -> Path:
    path = round_dir / "round_feedback.json"
    atomic_write_json(path, feedback)
    logger.info(f"Wrote round feedback: {path}")
    return path


def read_round_feedback(round_dir: Path) -> RoundFeedback:
    return read_json(round_dir / "round_feedback.json")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Exploration data loading (bridge to TrajectoryFormatter)
# ---------------------------------------------------------------------------


def load_exploration_result(exploration_dir: Path, exploration_id: str) -> Dict[str, Any]:
    """Load exploration result JSON from TrajectoryFormatter output."""
    result_path = exploration_dir / f"{exploration_id}_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Exploration result not found: {result_path}")
    return read_json(result_path)  # type: ignore[return-value]


def load_trajectory(exploration_dir: Path, exploration_id: str) -> List[Dict[str, Any]]:
    """Load trajectory.jsonl from TrajectoryFormatter output."""
    trajectory_path = exploration_dir / exploration_id / "trajectory" / "trajectory.jsonl"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")
    trajectory: List[Dict[str, Any]] = []
    with open(trajectory_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectory.append(json.loads(line))
    return trajectory


def extract_action_sequence(trajectory: List[Dict[str, Any]]) -> str:
    """Convert raw trajectory steps to human-readable action sequence text."""
    lines: List[str] = []
    for step in trajectory:
        step_idx = step.get("step", 0)
        action = step.get("action", "")
        action_type = step.get("action_type", "unknown")
        if isinstance(action, str):
            lines.append(f"Step {step_idx}: {action_type} - {action[:120]}")
        elif isinstance(action, dict):
            lines.append(f"Step {step_idx}: {action_type} - {json.dumps(action, ensure_ascii=False)[:120]}")
        else:
            lines.append(f"Step {step_idx}: {action_type}")
    return "\n".join(lines)


def extract_screenshot_paths(trajectory: List[Dict[str, Any]]) -> List[str]:
    """Extract screenshot file paths from raw trajectory steps."""
    return [step.get("screenshot_path", "") for step in trajectory if step.get("screenshot_path")]


def load_exploration_init_result(init_output_dir: Path, task_name: str, instance_id: int = 0) -> Dict[str, Any]:
    """Load task initialization result from TaskInitializer output."""
    result_path = init_output_dir / task_name / f"instance_{instance_id}" / "task_init_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Init result not found: {result_path}")
    return read_json(result_path)  # type: ignore[return-value]


def build_exploration_task_context(
    exploration_dir: Path,
    exploration_id: str,
    init_output_dir: Optional[Path] = None,
) -> ExplorationTaskContext:
    """Build standardized ExplorationTaskContext from TrajectoryFormatter output.

    This bridges the exploration module output to the curriculum task generator.
    """
    exp_result = load_exploration_result(exploration_dir, exploration_id)
    trajectory = load_trajectory(exploration_dir, exploration_id)

    context: ExplorationTaskContext = {
        "exploration_id": exploration_id,
        "environment": exp_result.get("environment", "unknown"),
        "action_sequence": extract_action_sequence(trajectory),
        "screenshots": extract_screenshot_paths(trajectory),
        "discovered_apps": exp_result.get("discovered_apps", []),
    }

    if init_output_dir:
        context["init_result"] = {}
        for task_result_path in init_output_dir.rglob("task_init_result.json"):
            try:
                init_data = read_json(task_result_path)  # type: ignore[arg-type]
                context["init_result"] = {
                    "task_name": init_data.get("task_name", ""),
                    "params_path": init_data.get("params_path", ""),
                    "init_screenshot": init_data.get("init_screenshot", ""),
                }
                break
            except Exception as e:
                logger.warning(f"Failed to read init result {task_result_path}: {e}")

    return context


def build_exploration_task_context_from_formatted(
    formatted_data: Dict[str, Any],
) -> ExplorationTaskContext:
    """Build ExplorationTaskContext from already-formatted TrajectoryFormatter output.

    Use this when the exploration data has already been processed by TrajectoryFormatter.
    """
    trajectory = formatted_data.get("trajectory", [])
    if isinstance(trajectory, list) and trajectory and isinstance(trajectory[0], dict):
        action_sequence = extract_action_sequence(trajectory)
        screenshots = extract_screenshot_paths(trajectory)
    else:
        action_sequence = formatted_data.get("action_sequence", "")
        screenshots = formatted_data.get("screenshots", [])

    context: ExplorationTaskContext = {
        "exploration_id": formatted_data.get("exploration_id", "unknown"),
        "environment": formatted_data.get("environment", "unknown"),
        "action_sequence": action_sequence,
        "screenshots": screenshots,
        "discovered_apps": formatted_data.get("discovered_apps", []),
        "init_result": formatted_data.get("init_result"),
    }
    return context


# ---------------------------------------------------------------------------
# Task JSON helpers
# ---------------------------------------------------------------------------


def scan_generated_task_jsons(generated_task_dir: Path) -> List[Path]:
    """Scan directory for all .json task files."""
    return sorted(generated_task_dir.rglob("*.json"))


def ensure_snapshot_in_task(task_json_path: Path) -> None:
    """Validate that a task JSON has a snapshot field. Raises ValueError if missing."""
    data = read_json(task_json_path)
    if not data.get("snapshot"):
        raise ValueError(f"Task JSON {task_json_path} missing required 'snapshot' field")


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


def get_parquet_dir(round_dir: Path) -> Path:
    return ensure_dir(round_dir / "parquet")


def get_generated_tasks_dir(round_dir: Path) -> Path:
    return ensure_dir(round_dir / "generated_tasks")
