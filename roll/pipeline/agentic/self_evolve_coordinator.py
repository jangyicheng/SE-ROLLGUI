"""
Self-Evolve Coordinator for the Agentic Pipeline.

Provides two hooks:
- on_episode_end(): called when an episode finishes, invokes judge and returns result
- on_round_end(): called when a round finishes, aggregates feedback and triggers task generation
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from roll.utils.logging import get_logger

from .self_evolve_io import (
    build_exploration_task_context,
    ensure_dir,
    get_episode_dir,
    get_round_dir,
    read_judge_result,
    write_judge_input,
    write_judge_result,
)
from .self_evolve_types import (
    ExplorationTaskContext,
    JudgeEpisodeInput,
    JudgeEpisodeResult,
    RoundUpdateResult,
)
from .self_evolve_feedback import build_round_feedback

logger = get_logger()


class SelfEvolveCoordinator:
    """Coordinates the self-evolving loop within the agentic pipeline.

    Two-stage design:
    - Episode level: collects episode data, invokes judge, writes JSON artifacts
    - Round level: aggregates feedback, generates tasks, refreshes parquet
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        self.enabled: bool = config.get("enabled", False)
        self.feedback_root: Path = Path(config.get("feedback_root", "./trajectories/self_evolve"))
        self.generated_task_root: Path = Path(
            config.get("generated_task_root", "./data/tasks/generated/mobile_self_evolve")
        )
        self.parquet_root: Path = Path(config.get("parquet_root", "./data/self_evolve_parquet"))
        self.round_id: int = 0
        self._judge_evaluator: Optional[Any] = None
        self._generator: Optional[Any] = None
        self._pending_episode_count: int = 0

    # -------------------------------------------------------------------------
    # Judge integration (calls existing mobilejudge.py methods, no modification)
    # -------------------------------------------------------------------------

    @property
    def judge_evaluator(self) -> Any:
        """Lazy-load the judge evaluator from mobilejudge.py."""
        if self._judge_evaluator is None:
            from .mobilejudge import MobileJudgeEvaluator

            self._judge_evaluator = MobileJudgeEvaluator(
                key_identification_screenshot_model="o4-mini",
                key_points_outcome_model="o4-mini",
                max_image=50,
            )
            logger.info("MobileJudgeEvaluator initialized (self-evolve mode)")
        return self._judge_evaluator

    def _build_judge_input(
        self,
        episode_artifacts: Dict[str, Any],
        task_id: str,
        episode_id: str,
    ) -> JudgeEpisodeInput:
        """Build JudgeEpisodeInput from episode artifacts collected in the env manager."""
        screenshot_paths = episode_artifacts.get("screenshot_paths", [])
        action_history = episode_artifacts.get("action_history", [])
        env_signals = episode_artifacts.get("env_signals", {})

        final_screenshot_path = screenshot_paths[-1] if screenshot_paths else ""

        judge_input: JudgeEpisodeInput = {
            "task_id": task_id,
            "instruction": episode_artifacts.get("instruction", ""),
            "snapshot": episode_artifacts.get("snapshot", ""),
            "current_screenshot_paths": screenshot_paths,
            "final_screenshot_path": final_screenshot_path,
            "reference_screenshot_paths": [],
            "reference_instruction": "",
            "action_history": action_history,
            "env_signals": env_signals,
        }
        return judge_input

    def _call_judge(self, judge_input: JudgeEpisodeInput) -> JudgeEpisodeResult:
        """Invoke MobileJudge using its existing evaluate_trajectory method."""
        evaluator = self.judge_evaluator
        task = judge_input.get("instruction", "")
        input_images = judge_input.get("current_screenshot_paths", [])
        last_actions = judge_input.get("action_history", [])

        response_text, reward, details = evaluator.evaluate_trajectory(
            task=task,
            input_image_paths=input_images,
            last_actions=last_actions,
            images_path=input_images,
        )

        # Map to JudgeEpisodeResult structure
        success = reward >= 0.5
        feedback_summary = response_text[:200] if response_text else ""

        result: JudgeEpisodeResult = {
            "task_id": judge_input.get("task_id", "unknown"),
            "reward": float(reward),
            "success": success,
            "feedback_summary": feedback_summary,
            "feedback_strings": [response_text] if response_text else [],
            "failure_reasons": [],
            "raw_judge_text": response_text,
            "usage": details.get("usage", {}) if details else {},
        }

        if details:
            result["dimension_scores"] = {
                "goal_completion": float(reward),
                "ui_state_match": float(reward),
                "constraint_satisfaction": float(reward),
            }
            if "key_points" in details:
                result["feedback_strings"] = [details["key_points"], response_text]

        return result

    # -------------------------------------------------------------------------
    # Episode-level hook
    # -------------------------------------------------------------------------

    def on_episode_end(
        self,
        episode_artifacts: Dict[str, Any],
        task_id: str,
        episode_id: str,
    ) -> JudgeEpisodeResult:
        """Called at the end of each episode.

        Writes judge_input.json, invokes MobileJudge, writes judge_result.json,
        and returns the result for the caller to use (e.g. to overwrite reward).

        Args:
            episode_artifacts: Dict containing screenshot_paths, action_history, etc.
            task_id: Unique task identifier
            episode_id: Unique episode identifier within the task

        Returns:
            JudgeEpisodeResult with reward, success, and feedback
        """
        if not self.enabled:
            return self._dummy_judge_result(task_id)

        round_dir = get_round_dir(self.feedback_root, self.round_id)
        episode_dir = get_episode_dir(round_dir, task_id, episode_id)
        ensure_dir(episode_dir)

        judge_input = self._build_judge_input(episode_artifacts, task_id, episode_id)
        write_judge_input(episode_dir, judge_input)

        try:
            judge_result = self._call_judge(judge_input)
            write_judge_result(episode_dir, judge_result)
            self._pending_episode_count += 1
            logger.debug(
                f"[SelfEvolve] Round {self.round_id} | Task {task_id} | "
                f"Reward={judge_result['reward']:.3f} | Success={judge_result['success']}"
            )
            return judge_result
        except Exception as e:
            logger.error(f"[SelfEvolve] Judge call failed: {e}")
            # Fallback: return dummy result so training continues
            return self._dummy_judge_result(task_id, error=str(e))

    def _dummy_judge_result(self, task_id: str, error: str = "") -> JudgeEpisodeResult:
        """Return a neutral judge result when judge is disabled or fails."""
        return {
            "task_id": task_id,
            "reward": 0.0,
            "success": False,
            "feedback_summary": f"Judge unavailable: {error}" if error else "Self-evolve disabled",
            "feedback_strings": [],
            "failure_reasons": [],
            "raw_judge_text": "",
        }

    # -------------------------------------------------------------------------
    # Round-level hook
    # -------------------------------------------------------------------------

    def on_round_end(
        self,
        exploration_dirs: Optional[List[str]] = None,
        context_template_dir: Optional[str] = None,
        task_instructions: Optional[Dict[str, str]] = None,
        task_snapshots: Optional[Dict[str, str]] = None,
    ) -> RoundUpdateResult:
        """Called when a round completes.

        1. Aggregates all judge_result.json into round_feedback.json
        2. Generates new tasks via CurriculumTaskGenerator
        3. Runs prepare_data.py to generate new parquet
        4. Returns the update result

        Args:
            exploration_dirs: List of exploration output directories for task generation
            context_template_dir: Directory containing context template JSONs
            task_instructions: Optional dict of task_id -> instruction
            task_snapshots: Optional dict of task_id -> snapshot

        Returns:
            RoundUpdateResult with paths to generated files
        """
        if not self.enabled:
            logger.info("[SelfEvolve] Disabled, skipping round-end processing")
            return {"success": False, "round_id": self.round_id, "error": "Disabled"}

        self.round_id += 1
        round_dir = get_round_dir(self.feedback_root, self.round_id)
        ensure_dir(round_dir)

        result: RoundUpdateResult = {
            "success": False,
            "round_id": self.round_id,
            "feedback_file": "",
            "generated_task_files": [],
            "parquet_file": "",
            "error": None,
        }

        try:
            # Step 1: Build round feedback
            feedback = build_round_feedback(
                round_dir=round_dir,
                task_instructions=task_instructions,
                task_snapshots=task_snapshots,
            )
            feedback_path = round_dir / "round_feedback.json"
            result["feedback_file"] = str(feedback_path)
            logger.info(
                f"[SelfEvolve] Round {self.round_id}: {feedback.get('total_tasks', 0)} tasks, "
                f"SR={feedback.get('overall_success_rate', 0):.3f}, "
                f"episodes={self._pending_episode_count}"
            )
            self._pending_episode_count = 0

            # Step 2: Generate new tasks
            generated_files = self._generate_tasks(
                feedback_file=str(feedback_path),
                exploration_dirs=exploration_dirs,
                context_template_dir=context_template_dir,
                round_dir=round_dir,
            )
            result["generated_task_files"] = generated_files

            # Step 3: Prepare parquet
            if generated_files:
                parquet_file = self._run_prepare_data(round_dir)
                result["parquet_file"] = parquet_file
                logger.info(f"[SelfEvolve] Round {self.round_id} parquet: {parquet_file}")

            result["success"] = True
            logger.info(f"[SelfEvolve] Round {self.round_id} update complete")

        except Exception as e:
            logger.error(f"[SelfEvolve] Round {self.round_id} update failed: {e}")
            result["error"] = str(e)

        return result

    def _generate_tasks(
        self,
        feedback_file: str,
        exploration_dirs: Optional[List[str]],
        context_template_dir: Optional[str],
        round_dir: Path,
    ) -> List[str]:
        """Generate tasks using CurriculumTaskGenerator."""
        if self._generator is None:
            from .curriculum_task_generator import MobileSpecificTaskGenerator

            self._generator = MobileSpecificTaskGenerator(
                openai_api_key="",  # Will be read from env if empty
                model="gpt-4o",
            )

        output_dir = str(get_generated_tasks_dir(round_dir))
        index_dir = str(self.generated_task_root / "index")
        ensure_dir(Path(output_dir))
        ensure_dir(Path(index_dir))

        generated_files: List[str] = []

        # Build exploration task contexts from exploration_dirs
        exploration_contexts: List[ExplorationTaskContext] = []
        if exploration_dirs:
            for exp_dir in exploration_dirs:
                exp_path = Path(exp_dir)
                if not exp_path.exists():
                    logger.warning(f"Exploration dir not found: {exp_dir}")
                    continue
                # Find all exploration IDs in the directory
                for exp_id_dir in exp_path.iterdir():
                    if exp_id_dir.is_dir() and (exp_id_dir / "trajectory").exists():
                        try:
                            ctx = build_exploration_task_context(exp_path, exp_id_dir.name)
                            exploration_contexts.append(ctx)
                        except Exception as e:
                            logger.warning(f"Failed to load exploration {exp_id_dir.name}: {e}")

        # Use the generator's new exploration-based method
        try:
            if exploration_contexts:
                files = self._generator.generate_tasks_from_exploration(
                    exploration_contexts=exploration_contexts,
                    feedback_file=feedback_file,
                    output_dir=output_dir,
                    index_dir=index_dir,
                    iteration=self.round_id,
                )
                generated_files.extend(files)
        except AttributeError:
            # Fallback: if generate_tasks_from_exploration doesn't exist yet, use context-based
            logger.warning("[SelfEvolve] generate_tasks_from_exploration not available, skipping")
        except Exception as e:
            logger.error(f"[SelfEvolve] Task generation failed: {e}")

        return generated_files

    def _run_prepare_data(self, round_dir: Path) -> str:
        """Run prepare_data.py to generate parquet from generated tasks."""
        generated_dir = get_generated_tasks_dir(round_dir)
        parquet_dir = ensure_dir(round_dir / "parquet")
        train_parquet = parquet_dir / "train.parquet"

        if not any(generated_dir.rglob("*.json")):
            logger.warning(f"[SelfEvolve] No task JSONs found in {generated_dir}, skipping parquet")
            return ""

        script_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "pipeline"
            / "agentic"
            / "env"
            / "android"
            / "mobile"
            / "prepare_data.py"
        )

        cmd = [
            sys.executable,
            str(script_path),
            "--mode", "visual",
            "--data_dir", str(generated_dir),
            "--local_dir", str(parquet_dir),
            "--train_category", "mobile",
            "--train_category_dir", str(generated_dir),
            "--self_evolve_mode",
            "--train_data_size", "1000",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.error(f"[SelfEvolve] prepare_data.py failed: {result.stderr[:500]}")
            else:
                logger.info(f"[SelfEvolve] prepare_data.py succeeded")
        except Exception as e:
            logger.error(f"[SelfEvolve] prepare_data.py exception: {e}")

        return str(train_parquet) if train_parquet.exists() else ""

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_current_round_dir(self) -> Path:
        return get_round_dir(self.feedback_root, self.round_id)

    def get_latest_parquet(self) -> Optional[Path]:
        """Return the latest train.parquet path if it exists."""
        for round_num in range(self.round_id, 0, -1):
            parquet_dir = self.feedback_root / f"round_{round_num:04d}" / "parquet"
            train_parquet = parquet_dir / "train.parquet"
            if train_parquet.exists():
                return train_parquet
        return None
