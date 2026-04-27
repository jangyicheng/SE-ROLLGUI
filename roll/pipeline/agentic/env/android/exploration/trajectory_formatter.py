"""
Trajectory Formatting Utilities for Curriculum Generation.

Formats exploration and task initialization data into a structure
suitable for the curriculum task generator.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from roll.utils.logging import get_logger

logger = get_logger()


class TrajectoryFormatter:
    """Formats exploration data for curriculum generation."""

    def __init__(self, exploration_dir: Optional[str] = None, init_output_dir: Optional[str] = None):
        self.exploration_dir = Path(exploration_dir) if exploration_dir else None
        self.init_output_dir = Path(init_output_dir) if init_output_dir else None

    def load_exploration_result(self, exploration_id: str) -> Dict[str, Any]:
        """Load exploration result from directory."""
        if not self.exploration_dir:
            raise ValueError("exploration_dir not set")
        result_path = self.exploration_dir / f"{exploration_id}_result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Exploration result not found: {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_trajectory(self, exploration_id: str) -> List[Dict[str, Any]]:
        """Load exploration trajectory from directory."""
        if not self.exploration_dir:
            raise ValueError("exploration_dir not set")
        trajectory_path = self.exploration_dir / exploration_id / "trajectory" / "trajectory.jsonl"
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")
        trajectory = []
        with open(trajectory_path, "r", encoding="utf-8") as f:
            for line in f:
                trajectory.append(json.loads(line.strip()))
        return trajectory

    def load_init_result(self, task_name: str, instance_id: int = 0) -> Dict[str, Any]:
        """Load task initialization result."""
        if not self.init_output_dir:
            raise ValueError("init_output_dir not set")
        result_path = self.init_output_dir / task_name / f"instance_{instance_id}" / "task_init_result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Init result not found: {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_action_sequence(self, trajectory: List[Dict[str, Any]]) -> str:
        """Extract human-readable action sequence from trajectory."""
        actions = []
        for step in trajectory:
            step_idx = step.get("step", 0)
            action = step.get("action", "")
            action_type = step.get("action_type", "unknown")
            current_app = step.get("current_app", "")

            if isinstance(action, str):
                actions.append(f"Step {step_idx}: {action_type} - {action[:100]}")
            elif isinstance(action, dict):
                action_str = json.dumps(action, ensure_ascii=False)[:100]
                actions.append(f"Step {step_idx}: {action_type} - {action_str}")
            else:
                actions.append(f"Step {step_idx}: {action_type}")
        return "\n".join(actions)

    def extract_screenshots(self, trajectory: List[Dict[str, Any]]) -> List[str]:
        """Extract screenshot paths from trajectory."""
        screenshots = []
        for step in trajectory:
            screenshot_path = step.get("screenshot_path")
            if screenshot_path:
                screenshots.append(screenshot_path)
        return screenshots

    def extract_discovered_apps(self, exploration_result: Dict[str, Any]) -> List[str]:
        """Extract discovered apps from exploration result."""
        return exploration_result.get("discovered_apps", [])

    def extract_discovered_actions(self, exploration_result: Dict[str, Any]) -> List[str]:
        """Extract discovered action types from exploration result."""
        return exploration_result.get("discovered_action_types", [])

    def format_for_curriculum(
        self,
        exploration_result: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        init_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Format exploration data for curriculum generation.

        Args:
            exploration_result: Result from AndroidWorldExplorer
            trajectory: List of trajectory steps
            init_results: Optional list of task initialization results

        Returns:
            Formatted dictionary for curriculum generator
        """
        action_sequence = self.extract_action_sequence(trajectory)
        screenshots = self.extract_screenshots(trajectory)
        discovered_apps = self.extract_discovered_apps(exploration_result)
        discovered_actions = self.extract_discovered_actions(exploration_result)

        formatted = {
            "exploration_id": exploration_result.get("exploration_id", "unknown"),
            "timestamp": exploration_result.get("timestamp", ""),
            "environment": exploration_result.get("environment", "unknown"),
            "model": exploration_result.get("model", "unknown"),
            "stats": {
                "total_steps": exploration_result.get("actual_steps", 0),
                "max_steps": exploration_result.get("max_steps", 0),
                "num_screenshots": len(screenshots),
                "num_discovered_apps": len(discovered_apps),
                "num_discovered_actions": len(discovered_actions),
            },
            "discovered_apps": discovered_apps,
            "discovered_action_types": discovered_actions,
            "action_sequence": action_sequence,
            "screenshots": screenshots,
            "init_results": init_results or [],
        }

        return formatted

    def format_context_review_for_prompt(
        self,
        init_results: List[Dict[str, Any]],
    ) -> str:
        """Format initialization results as a context review prompt section.

        This creates a text section describing the task initialization
        results, suitable for including in a curriculum generation prompt.
        """
        lines = ["## Context Review Results\n"]
        lines.append(f"Total tasks reviewed: {len(init_results)}\n")

        success_count = sum(1 for r in init_results if r.get("initialization", {}).get("success", False))
        lines.append(f"Successfully initialized: {success_count}\n")
        lines.append(f"Failed: {len(init_results) - success_count}\n\n")

        for result in init_results[:10]:
            task_name = result.get("task_name", "unknown")
            success = result.get("initialization", {}).get("success", False)
            apps = result.get("app_snapshot_restored", [])
            blockers = result.get("blockers", [])

            status = "SUCCESS" if success else "FAILED"
            lines.append(f"### {task_name}: {status}")
            if apps:
                lines.append(f"  Apps restored: {', '.join(apps)}")
            if blockers:
                lines.append(f"  Blockers: {', '.join(blockers)}")
            lines.append("")

        return "\n".join(lines)

    def generate_synthetic_task(
        self,
        exploration_data: Dict[str, Any],
        task_template: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a synthetic task based on exploration data.

        Args:
            exploration_data: Formatted exploration data
            task_template: Task instruction template
            params: Optional task parameters from initialization

        Returns:
            Synthetic task dictionary
        """
        task_id = f"{exploration_data.get('exploration_id', 'unknown')}_{hash(task_template) % 10000:04d}"
        return {
            "id": task_id,
            "instruction": task_template,
            "source": "self_evolving",
            "exploration_id": exploration_data.get("exploration_id"),
            "environment": exploration_data.get("environment"),
            "discovered_apps": exploration_data.get("discovered_apps", []),
            "action_sequence": exploration_data.get("action_sequence", ""),
            "params": params or {},
            "params_path": params.get("params_path") if params else None,
            "seed": params.get("seed") if params else None,
        }


def format_trajectory_for_curriculum(
    exploration_dir: str,
    init_output_dir: Optional[str] = None,
    exploration_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to format exploration data for curriculum generation.

    Args:
        exploration_dir: Directory containing exploration outputs
        init_output_dir: Optional directory containing task initialization outputs
        exploration_id: Specific exploration ID to format (if None, uses most recent)

    Returns:
        Formatted data dictionary for curriculum generator
    """
    formatter = TrajectoryFormatter(exploration_dir=exploration_dir, init_output_dir=init_output_dir)

    if not exploration_id:
        exploration_dir_path = Path(exploration_dir)
        exploration_ids = [d.name for d in exploration_dir_path.iterdir() if d.is_dir()]
        if not exploration_ids:
            raise ValueError(f"No exploration results found in {exploration_dir}")
        exploration_id = sorted(exploration_ids)[-1]

    exploration_result = formatter.load_exploration_result(exploration_id)
    trajectory = formatter.load_trajectory(exploration_id)

    init_results = []
    if init_output_dir:
        formatter.init_output_dir = Path(init_output_dir)
        for task_result_path in Path(init_output_dir).rglob("task_init_result.json"):
            with open(task_result_path, "r", encoding="utf-8") as f:
                init_results.append(json.load(f))

    return formatter.format_for_curriculum(exploration_result, trajectory, init_results)


def format_multiple_explorations(exploration_dirs: List[str]) -> Dict[str, Any]:
    """Format multiple exploration results together.

    Args:
        exploration_dirs: List of exploration output directories

    Returns:
        Combined formatted data from all explorations
    """
    combined = {
        "explorations": [],
        "all_discovered_apps": set(),
        "all_discovered_actions": set(),
        "total_steps": 0,
    }

    for exp_dir in exploration_dirs:
        try:
            formatter = TrajectoryFormatter(exploration_dir=exp_dir)
            exploration_dir_path = Path(exp_dir)
            exploration_ids = [d.name for d in exploration_dir_path.iterdir() if d.is_dir()]

            for exp_id in exploration_ids:
                try:
                    result = formatter.load_exploration_result(exp_id)
                    trajectory = formatter.load_trajectory(exp_id)
                    formatted = formatter.format_for_curriculum(result, trajectory)

                    combined["explorations"].append(formatted)
                    combined["all_discovered_apps"].update(formatted["discovered_apps"])
                    combined["all_discovered_actions"].update(formatted["discovered_action_types"])
                    combined["total_steps"] += formatted["stats"]["total_steps"]
                except Exception as e:
                    logger.warning(f"Failed to process exploration {exp_id}: {e}")
        except Exception as e:
            logger.warning(f"Failed to process exploration directory {exp_dir}: {e}")

    combined["all_discovered_apps"] = list(combined["all_discovered_apps"])
    combined["all_discovered_actions"] = list(combined["all_discovered_actions"])

    return combined
