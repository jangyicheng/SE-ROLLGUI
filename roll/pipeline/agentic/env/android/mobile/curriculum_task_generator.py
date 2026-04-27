import argparse
import base64
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import openai
from PIL import Image


class MobileSpecificTaskGenerator:
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o",
        enable_diversity: bool = False,
        software: str = "mobile_app",
        easy_tasks: int = 1,
        medium_tasks: int = 1,
        hard_tasks: int = 1,
        image_download_timeout: int = 10,
    ):
        """Initialize mobile specific task generator."""
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.enable_diversity = enable_diversity
        self.software = software
        self.easy_tasks = easy_tasks
        self.medium_tasks = medium_tasks
        self.hard_tasks = hard_tasks
        self.image_download_timeout = image_download_timeout

    def load_trajectory_data(self, trajectory_dir: str) -> Dict[str, Any]:
        """Load trajectory data JSON file from trajectory directory."""
        traj_dir = Path(trajectory_dir)
        json_files = list(traj_dir.glob("**/*.json"))
        if not json_files:
            raise FileNotFoundError(f"No trajectory JSON file found in {trajectory_dir}")

        trajectory_file = json_files[0]
        print(f"Loading exploration trajectory from: {trajectory_file}")
        with open(trajectory_file, "r", encoding="utf-8") as f:
            trajectory_data = json.load(f)
        return trajectory_data

    def _is_url(self, source: str) -> bool:
        source_lower = source.lower()
        return source_lower.startswith("http://") or source_lower.startswith("https://")

    def _load_image_bytes_from_source(self, source: str) -> bytes:
        if self._is_url(source):
            req = urllib.request.Request(
                source,
                headers={"User-Agent": "ACuRL-Mobile-TaskGenerator/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.image_download_timeout) as resp:
                return resp.read()

        with open(source, "rb") as f:
            return f.read()

    def _source_to_display_name(self, source: str, fallback_idx: int) -> str:
        if self._is_url(source):
            parsed = urllib.parse.urlparse(source)
            name = os.path.basename(parsed.path)
            return name or f"remote_{fallback_idx}.png"
        return Path(source).name

    def get_screenshots_from_sources(self, screenshot_sources: Sequence[str]) -> List[Dict[str, Any]]:
        """Load screenshots from mixed local path and URL sources."""
        screenshots: List[Dict[str, Any]] = []

        for idx, source in enumerate(screenshot_sources):
            try:
                image_data = self._load_image_bytes_from_source(str(source))
                image = Image.open(io.BytesIO(image_data))
                image.load()
                screenshots.append(
                    {
                        "filename": self._source_to_display_name(str(source), idx),
                        "path": str(source),
                        "base64": base64.b64encode(image_data).decode("utf-8"),
                        "size": image.size,
                        "source_type": "url" if self._is_url(str(source)) else "local",
                    }
                )
            except (
                FileNotFoundError,
                OSError,
                urllib.error.URLError,
                ValueError,
            ) as err:
                print(f"Warning: failed to load screenshot source {source}: {err}")

        return screenshots

    def get_screenshots_from_folder(
        self,
        screenshots_folder: str,
        file_pattern: str = "*.png",
        extra_sources: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get screenshots from folder and optional extra URL/local sources."""
        folder = Path(screenshots_folder)

        def natural_sort_key(path: Path):
            parts = re.split(r"(\d+)", path.name)
            return [int(part) if part.isdigit() else part.lower() for part in parts]

        local_png_files = sorted(folder.glob(file_pattern), key=natural_sort_key)
        all_sources = [str(p) for p in local_png_files]

        if extra_sources:
            all_sources.extend([str(s) for s in extra_sources])

        return self.get_screenshots_from_sources(all_sources)

    def get_environment_exploration_screenshots(
        self,
        environment_exploration_dir: str,
        extra_sources: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get mobile exploration screenshots (step*.png and optional URL/local sources)."""
        screenshots = self.get_screenshots_from_folder(
            environment_exploration_dir,
            "**/step*.png",
            extra_sources=extra_sources,
        )
        print(f"Found {len(screenshots)} environment exploration screenshots")
        return screenshots

    def get_context_review_screenshots(
        self, context_review_dir: str, extra_sources: Optional[Sequence[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get context review screenshots (local + optional URL/local sources)."""
        screenshots = self.get_screenshots_from_folder(
            context_review_dir,
            "*.png",
            extra_sources=extra_sources,
        )
        print(f"Found {len(screenshots)} context review screenshots")
        return screenshots

    def format_environment_exploration_actions(self, trajectory_data: Dict[str, Any]) -> str:
        """Format exploration actions for prompt."""
        raw_actions = trajectory_data.get("raw_actions", [])
        actions_text = "Base Mobile Exploration Actions:\n"
        for i, action in enumerate(raw_actions):
            actions_text += f"Step {i+1}: {action}\n"
        return actions_text

    def _build_content_with_screenshots(
        self,
        software_exploration_text: str,
        environment_exploration_screenshots: List[Dict[str, Any]],
        context_review_text: str,
        context_review_screenshots: List[Dict[str, Any]],
        task_requirements_text: str,
    ) -> List[Dict[str, Any]]:
        """Build multimodal content array in deterministic order."""
        content: List[Dict[str, Any]] = []

        # Part 1: Software Exploration
        content.append({"type": "text", "text": software_exploration_text})

        # Add environment exploration screenshots
        for screenshot in environment_exploration_screenshots:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot['base64']}",
                        "detail": "high",
                    },
                }
            )

        # Part 2: Context Review
        content.append({"type": "text", "text": context_review_text})

        # Add context review screenshots
        for screenshot in context_review_screenshots:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot['base64']}",
                        "detail": "high",
                    },
                }
            )

        # Part 3: Task Generation Requirements
        content.append({"type": "text", "text": task_requirements_text})
        return content

    def _call_openai_api(self, content: List[Dict[str, Any]]) -> Optional[str]:
        """Make OpenAI API call with the given content."""
        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": 16384,
        }

        # 候选模型中，只有部分模型支持高质量 multimodal reasoning at temperature 1.0
        reasoning_models = ["o4-mini", "gpt-5"]
        if self.model not in reasoning_models:
            request_params["temperature"] = 1.0

        print(f"Making API call to {self.model} with {len(content)} content items...")
        response = self.client.chat.completions.create(**request_params)
        print(response.choices[0].message.content)
        return response.choices[0].message.content

    def load_existing_tasks_for_diversity(
        self,
        existing_task_files: List[str],
        context_filename: Optional[str] = None,
        iteration: int = 0,
    ) -> List[str]:
        """Load existing tasks for diversity enhancement."""
        existing_tasks: List[str] = []

        for task_file in existing_task_files:
            # Filter by context filename if provided and not iteration 0
            if context_filename and iteration > 0 and context_filename not in os.path.basename(task_file):
                continue

            if os.path.exists(task_file):
                with open(task_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "instruction" in data:
                    existing_tasks.append(data["instruction"])

        if context_filename and iteration > 0:
            print(
                f"Loaded {len(existing_tasks)} existing tasks from {len(existing_task_files)} files "
                f"(filtered by '{context_filename}')"
            )
        else:
            print(f"Loaded {len(existing_tasks)} existing tasks from {len(existing_task_files)} files")
        return existing_tasks

    def load_feedback_data(self, feedback_file: str) -> Optional[Dict[str, Any]]:
        """Load feedback data from previous iterations.

        Supports both old format (task_performances[].accuracy) and
        self-evolve format (task_performances[].success_rate + feedback_strings).
        """
        if not os.path.exists(feedback_file):
            print(f"Warning: Feedback file not found: {feedback_file}")
            return None

        with open(feedback_file, "r", encoding="utf-8") as f:
            feedback_data = json.load(f)

        # Normalize old format (accuracy) to new format (success_rate)
        task_performances = feedback_data.get("task_performances", [])
        for perf in task_performances:
            if "accuracy" in perf and "success_rate" not in perf:
                perf["success_rate"] = perf["accuracy"]

        print(f"Loaded feedback data from: {feedback_file}")
        return feedback_data

    def load_round_feedback(self, feedback_file: str) -> Optional[Dict[str, Any]]:
        """Load round_feedback.json produced by self_evolve_feedback.build_round_feedback."""
        return self.load_feedback_data(feedback_file)

    def format_feedback_summary(self, feedback_data: Dict[str, Any], context_filename: Optional[str] = None) -> str:
        """Format feedback data into summary for prompt.

        Supports both old format (accuracy) and self-evolve format (success_rate).
        """
        if not feedback_data:
            return ""

        task_performances = feedback_data.get("task_performances", [])
        if not task_performances:
            return ""

        # Normalize accuracy -> success_rate
        for perf in task_performances:
            if "accuracy" in perf and "success_rate" not in perf:
                perf["success_rate"] = perf["accuracy"]

        # Build task performance summary with numbered list
        task_performance_lines: List[str] = []
        for task_perf in task_performances:
            task_id = task_perf.get("task_id", "")
            success_rate = task_perf.get("success_rate", task_perf.get("accuracy", 0))
            original_task = task_perf.get("original_task", {})
            instruction = original_task.get("instruction", "")
            if context_filename and context_filename in task_id:
                task_performance_lines.append(f"{instruction} (Average SR: {success_rate:.0%})")

        task_performance_text = "\n".join(f"{i+1}. {line}" for i, line in enumerate(task_performance_lines))

        return f"""## Feedback on Previous Tasks
Here is a structured summary of learner performance from earlier iterations:

{task_performance_text}"""

    def format_feedback_summary_v2(self, feedback_data: Dict[str, Any]) -> str:
        """Format self-evolve round feedback with success rate tiers and failure reasons.

        Produces a richer prompt section that includes:
        - Success rate tiers: high (>=80%), mid (30-80%), low (<30%)
        - Failure reasons extracted from judge output
        - Detailed per-task feedback strings
        """
        if not feedback_data:
            return ""

        task_performances = feedback_data.get("task_performances", [])
        if not task_performances:
            return ""

        # Normalize accuracy -> success_rate
        for perf in task_performances:
            if "accuracy" in perf and "success_rate" not in perf:
                perf["success_rate"] = perf["accuracy"]

        lines = ["## Self-Evolve Feedback Summary\n"]

        # Categorize by success rate tier
        high_sr = [p for p in task_performances if p.get("success_rate", 0) >= 0.8]
        mid_sr = [p for p in task_performances if 0.3 <= p.get("success_rate", 0) < 0.8]
        low_sr = [p for p in task_performances if 0 <= p.get("success_rate", 0) < 0.3]

        if high_sr:
            lines.append(f"### High Success Rate Tasks (SR >= 80%): {len(high_sr)} tasks")
            for p in high_sr[:5]:
                lines.append(f"  - {p.get('task_id', 'unknown')}: SR={p.get('success_rate', 0):.0%}")
            if len(high_sr) > 5:
                lines.append(f"  ... and {len(high_sr) - 5} more")
            lines.append("  Strategy: Generate substantially harder tasks with new constraints.")

        if mid_sr:
            lines.append(f"\n### Medium Success Rate Tasks (30% <= SR < 80%): {len(mid_sr)} tasks")
            for p in mid_sr[:5]:
                lines.append(f"  - {p.get('task_id', 'unknown')}: SR={p.get('success_rate', 0):.0%}")
            if len(mid_sr) > 5:
                lines.append(f"  ... and {len(mid_sr) - 5} more")
            lines.append("  Strategy: Generate same-difficulty but different scenario tasks.")

        if low_sr:
            lines.append(f"\n### Low Success Rate Tasks (SR < 30%): {len(low_sr)} tasks")
            for p in low_sr[:5]:
                reasons = p.get("failure_reasons", [])[:3]
                fb = p.get("feedback_strings", [])[:2]
                reason_str = "; ".join(reasons) if reasons else ""
                fb_str = " | ".join(fb) if fb else ""
                lines.append(f"  - {p.get('task_id', 'unknown')}: SR={p.get('success_rate', 0):.0%}")
                if reason_str:
                    lines.append(f"    Failure reasons: {reason_str}")
                if fb_str:
                    lines.append(f"    Feedback: {fb_str[:200]}")
            if len(low_sr) > 5:
                lines.append(f"  ... and {len(low_sr) - 5} more")
            lines.append("  Strategy: Generate scaffolded prerequisite tasks targeting the specific failure patterns above.")

        overall_sr = feedback_data.get("overall_success_rate", 0)
        overall_reward = feedback_data.get("overall_avg_reward", 0)
        lines.append(f"\n### Overall Round Statistics")
        lines.append(f"  Overall Success Rate: {overall_sr:.1%}")
        lines.append(f"  Overall Average Reward: {overall_reward:.3f}")
        lines.append(f"  Total Tasks Evaluated: {feedback_data.get('total_tasks', len(task_performances))}")

        return "\n".join(lines)

    def build_iteration_0_prompt(
        self,
        environment_exploration_actions: str,
        environment_exploration_screenshots: List[Dict[str, Any]],
        context_review_screenshots: List[Dict[str, Any]],
        existing_tasks: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Build prompt for iteration 0 for mobile tasks."""
        total_tasks = self.easy_tasks + self.medium_tasks + self.hard_tasks

        software_exploration_text = f"""You are an expert mobile application task designer.
Your goal is to generate exactly {total_tasks} realistic, high-value, and frequently used atomic tasks** for the given mobile application.Each task should represent a **single, minimal yet meaningful operation** within the application — a fundamental action that users frequently perform as part of larger goals.

You will be provided with:
1. A record of random application exploration, including several actions and corresponding screenshots. 
   - This shows the possible functionality of the application.
2. An initial state of the application (with its own screenshots). 
   - This represents the exact starting point from which learners will begin.

## Mobile Application Exploration
Actions:
{environment_exploration_actions}

The corresponding {len(environment_exploration_screenshots)} screenshots across the actions:"""

        context_review_text = f"""## Initial State
The initial mobile state is shown in {len(context_review_screenshots)} screenshots.

Initial state screenshots:"""
        # Add diversity section if enabled
        diversity_section = ""
        if self.enable_diversity and existing_tasks:
            diversity_section = f"""## Diversity Requirements
Avoid repeating these existing tasks:
{chr(10).join(f"{i+1}. {task}" for i, task in enumerate(existing_tasks))}

- Avoid creating tasks similar to the existing examples above
- Generate creative variations and novel approaches
- Ensure variety in complexity, action types, and user scenarios
- Think of different user personas and use cases"""

        # 这里压缩了原始prompt
        task_requirements_text = f"""## Task Generation Requirements
Generate exactly {total_tasks} atomic mobile tasks with these rules:

1. Each task is a single clear user goal.
2. Must be achievable from provided initial state.
3. Must align with observed mobile capabilities from exploration evidence.
4. Must be unambiguous with explicit completion criteria.
5. Prefer high-frequency, practical mobile operations.
6. Use intent-oriented wording, not low-level click instructions.
7. Ensure diversity across intents and UI surfaces.
8. Keep tasks compatible with mobile action semantics such as open_app, press_home, press_back, long_press, scroll, drag, and text input.

{diversity_section}

Return the tasks in the following JSON structure:

```json
{{
  "tasks": [
    {{"task_id": "task_1", "task_description": "An atomic application operation achievable from the given initial state"}},
    {{"task_id": "task_2", "task_description": "..."}}
    ......
  ]
}}
```
"""
        print(task_requirements_text)
        content = self._build_content_with_screenshots(
            software_exploration_text=software_exploration_text,
            environment_exploration_screenshots=environment_exploration_screenshots,
            context_review_text=context_review_text,
            context_review_screenshots=context_review_screenshots,
            task_requirements_text=task_requirements_text,
        )
        return self._call_openai_api(content)

    def build_iteration_1plus_prompt(
        self,
        environment_exploration_actions: str,
        environment_exploration_screenshots: List[Dict[str, Any]],
        context_review_screenshots: List[Dict[str, Any]],
        existing_tasks: Optional[List[str]] = None,
        feedback_data: Optional[Dict[str, Any]] = None,
        context_filename: Optional[str] = None,
    ) -> Optional[str]:
        """Build prompt for iteration 1+ for mobile tasks."""
        total_tasks = self.easy_tasks + self.medium_tasks + self.hard_tasks

        software_exploration_text = f"""You are a professional instructional designer specializing in mobile application learning task design. 
Your objective is to generate exactly new {total_tasks} high-quality, realistic learning tasks based on the previous feedback.


You will be provided with:
1. A record of random application exploration, including several actions and corresponding screenshots. 
    - This shows the possible functionality of the application.
2. An initial state of the application (with its own screenshots). 
    - This represents the exact starting point from which learners will begin.
3. A summary of learner performance on earlier tasks.  
    - This indicates which tasks were difficult, what the learner can already solve, and what gaps or weaknesses remain.

## Mobile Application Exploration
Actions:
{environment_exploration_actions}

The corresponding {len(environment_exploration_screenshots)} screenshots across the actions:"""

        context_review_text = f"""## Initial State
The current initial state is shown in {len(context_review_screenshots)} screenshots.

Initial state screenshots:"""

        # Add feedback section (always present for iteration 1+)
        feedback_section = self.format_feedback_summary(feedback_data or {}, context_filename)
        # Add diversity section if enabled
        diversity_section = ""
        if self.enable_diversity and existing_tasks:
            diversity_section = f"""## Diversity Requirements
Avoid repeating these existing tasks:
{chr(10).join(f"{i+1}. {task}" for i, task in enumerate(existing_tasks))}

- Avoid creating tasks similar to the existing examples above
- Generate creative variations and novel approaches
- Ensure variety in complexity, action types, and user scenarios
- Think of different user personas and use cases"""

        # 这里压缩了原始prompt
        task_requirements_text = f"""## Task Generation Requirements
Based on the random exploration actions and screenshots, combined with the initial state screenshots, generate exactly {total_tasks} specific tasks that:

1. High-level user goals only, no procedural UI click-by-click instructions.
2. Anchor each task in provided initial mobile state.
3. Use exploration evidence to keep tasks realistic.
4. Keep tasks specific and measurable.
5. Cover distinct mobile intents and UI regions.
6. Integrate feedback:
   - High SR: generate substantially harder and clearly different goals.
   - Mid SR: generate comparable-difficulty but different scenarios.
   - Low SR: generate scaffolded prerequisite goals.
7. Maintain compatibility with mobile interaction patterns (app launch, navigation back/home, touch gestures, text input, dialog handling).

{feedback_section}

Return the tasks in the following JSON structure:

```json
{{
  "tasks": [
    {{
      "task_id": "task_1",
      "original_task_description": "The original task description from the previous feedback",
      "reasoning": "Explain why and how this task was modified given the agent's SR and feedback context.",
      "new_task_description": "High-level yet specific user goal derived from the initial state and refined through feedback."
    }},
    {{
      "task_id": "task_2",
      "original_task_description": "...",
      "reasoning": "...",
      "new_task_description": "..."
    }},......
  ]
}}```"""
        print(task_requirements_text)
        content = self._build_content_with_screenshots(
            software_exploration_text,
            environment_exploration_screenshots,
            context_review_text,
            context_review_screenshots,
            task_requirements_text,
        )
        return self._call_openai_api(content)

    def generate_specific_tasks(
        self,
        environment_exploration_actions: str,
        environment_exploration_screenshots: List[Dict[str, Any]],
        context_review_screenshots: List[Dict[str, Any]],
        existing_tasks: Optional[List[str]] = None,
        feedback_data: Optional[Dict[str, Any]] = None,
        iteration: int = 0,
        context_filename: Optional[str] = None,
    ) -> Optional[str]:
        """Generate tasks by dispatching to iteration-specific prompt builders."""

        # If no feedback data provided, use iteration 0 prompt
        if not feedback_data or iteration == 0:
            return self.build_iteration_0_prompt(
                environment_exploration_actions=environment_exploration_actions,
                environment_exploration_screenshots=environment_exploration_screenshots,
                context_review_screenshots=context_review_screenshots,
                existing_tasks=existing_tasks,
            )

        # Otherwise use iteration 1+ prompt with feedback
        return self.build_iteration_1plus_prompt(
            environment_exploration_actions=environment_exploration_actions,
            environment_exploration_screenshots=environment_exploration_screenshots,
            context_review_screenshots=context_review_screenshots,
            existing_tasks=existing_tasks,
            feedback_data=feedback_data,
            context_filename=context_filename,
        )

    def create_task_config_file(
        self,
        task_data: Dict[str, Any],
        context_filename: str,
        output_dir: str,
        context_template: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create mobile-compatible task configuration file.

        Validates that the 'snapshot' field is present, as it is required
        for environment initialization in self-evolve mode.
        """
        task_id = task_data["task_id"]
        task_description = task_data.get("task_description", task_data.get("new_task_description", ""))

        if not task_description:
            raise ValueError(f"Task {task_id} is missing task_description")

        original_task_description_raw = task_data.get("original_task_description")
        if original_task_description_raw:
            original_task_description = original_task_description_raw.split(" (Average SR")[0]
        else:
            original_task_description = None

        reasoning = task_data.get("reasoning", "")

        filename = f"{context_filename}_{task_id}.json"
        full_task_id = f"{context_filename}_{task_id}"

        config = json.loads(json.dumps(context_template)) if context_template else {}

        # snapshot is required for environment initialization
        snapshot_value = config.get("snapshot")
        if not snapshot_value:
            # Try to extract from context_template if it's a list (legacy format)
            if isinstance(context_template, dict):
                raw = context_template.get("snapshot")
                if isinstance(raw, list) and raw:
                    snapshot_value = raw[0]
                elif isinstance(raw, str):
                    snapshot_value = raw

        if not snapshot_value:
            raise ValueError(
                f"Task {task_id} is missing required 'snapshot' field. "
                f"Ensure the context_template includes a 'snapshot' key for environment initialization."
            )

        config["id"] = full_task_id
        config["snapshot"] = snapshot_value
        config["instruction"] = task_description
        config["source"] = config.get("source", "self_evolve")

        if original_task_description:
            config["original_task"] = original_task_description
        if reasoning:
            config["reasoning"] = reasoning

        output_path = Path(output_dir) / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"Created task config: {output_path}")
        return str(output_path)

    def update_task_index_json(
        self,
        generated_task_files: List[str],
        context_filename: str,
        output_dir: str,
        model: Optional[str] = None,
        index_dir: Optional[str] = None,
    ) -> str:
        """Create or update index JSON listing generated task IDs."""
        _ = context_filename
        _ = output_dir

        new_task_ids: List[str] = []
        for task_file in generated_task_files:
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                task_id = task_data.get("id")
                if task_id:
                    new_task_ids.append(f"{self.software}.{task_id}")
            except Exception as err:
                print(f"Warning: Could not read task file {task_file}: {err}")

        task_index_filename = f"{self.software}.generated_task.json"
        if model:
            task_index_filename = f"{self.software}.generated_task_{model}.json"

        if not index_dir:
            raise ValueError("index_dir must be provided")

        os.makedirs(index_dir, exist_ok=True)
        task_index_path = Path(index_dir) / task_index_filename

        existing_task_ids: List[str] = []
        if task_index_path.exists():
            try:
                with open(task_index_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                existing_task_ids = existing_data.get("tasks", [])
                print(f"Loaded {len(existing_task_ids)} existing task IDs from task index file")
            except Exception as err:
                print(f"Warning: Could not read existing task index file: {err}")

        all_task_ids = existing_task_ids.copy()
        for task_id in new_task_ids:
            if task_id not in all_task_ids:
                all_task_ids.append(task_id)

        task_index_data = {"tasks": sorted(all_task_ids)}

        with open(task_index_path, "w", encoding="utf-8") as f:
            json.dump(task_index_data, f, ensure_ascii=False, indent=2)

        print(f"Updated task index JSON: {task_index_path}")
        print(f"Added {len(new_task_ids)} new task IDs, total: {len(all_task_ids)} task IDs")
        return str(task_index_path)

    def generate_tasks_from_exploration(
        self,
        exploration_contexts: List[Dict[str, Any]],
        feedback_file: Optional[str] = None,
        output_dir: str = "./generated_tasks",
        index_dir: str = "./task_index",
        iteration: int = 0,
    ) -> List[str]:
        """Generate tasks from standardized exploration contexts (self-evolve entry point).

        This is the primary self-evolve entry point. It receives exploration data
        produced by the exploration module (TrajectoryFormatter + TaskInitializer) and
        generates new curriculum tasks based on the round feedback.

        Args:
            exploration_contexts: List of ExplorationTaskContext dicts from self_evolve_io
            feedback_file: Optional path to round_feedback.json from self_evolve_feedback
            output_dir: Output directory for generated task JSON files
            index_dir: Directory for task index JSON
            iteration: Current self-evolve round number

        Returns:
            List of generated task file paths
        """
        print(f"[SelfEvolve] Generating tasks from {len(exploration_contexts)} exploration contexts")

        # Load feedback if available
        feedback_data = None
        if feedback_file:
            feedback_data = self.load_feedback_data(feedback_file)
            if feedback_data:
                print("[SelfEvolve] Loaded round feedback for adaptive generation")
                print(f"  Tasks: {feedback_data.get('total_tasks', 0)}, "
                      f"Overall SR: {feedback_data.get('overall_success_rate', 0):.1%}")

        desired_total_tasks = self.easy_tasks + self.medium_tasks + self.hard_tasks
        os.makedirs(output_dir, exist_ok=True)

        generated_files: List[str] = []

        for ctx_idx, exploration_data in enumerate(exploration_contexts):
            exp_id = exploration_data.get("exploration_id", f"exp_{ctx_idx}")
            screenshots = exploration_data.get("screenshots", [])
            action_sequence = exploration_data.get("action_sequence", "")
            init_result = exploration_data.get("init_result", {})

            # Load screenshots from paths
            screenshot_sources: List[str] = []
            for sc_path in screenshots[:20]:
                if sc_path and os.path.exists(sc_path):
                    screenshot_sources.append(sc_path)

            exploration_screenshots = self.get_screenshots_from_sources(screenshot_sources)

            # Build exploration actions text
            if action_sequence:
                actions_text = f"Exploration Actions:\n{action_sequence}"
            else:
                actions_text = "Exploration Actions: No explicit action sequence recorded."

            # Load existing tasks for diversity
            existing_tasks: Optional[List[str]] = None
            if self.enable_diversity:
                existing_tasks = []

            # Determine which format_feedback_summary to use
            if feedback_data and iteration > 0:
                feedback_text = self.format_feedback_summary_v2(feedback_data)
            else:
                feedback_text = ""

            # Build the task requirements section
            total_tasks_str = desired_total_tasks
            task_requirements_text = f"""## Task Generation Requirements
Based on the exploration evidence below, generate exactly {total_tasks_str} tasks.

1. Each task is a single clear user goal achievable from the initial state.
2. Use intent-oriented wording, not low-level click instructions.
3. Keep tasks specific and measurable with explicit completion criteria.
4. Maintain compatibility with mobile interaction patterns (open_app, tap, swipe, long_press, type, press_back, press_home).

## Feedback Strategy (Self-Evolve Mode)
{feedback_text}

Return the tasks in this JSON structure:
```json
{{
  "tasks": [
    {{
      "task_id": "task_1",
      "task_description": "A clear, high-level user goal"
    }}
  ]
}}
```"""

            # Build multimodal content
            content: List[Dict[str, Any]] = []
            content.append({
                "type": "text",
                "text": f"You are a mobile GUI task designer.\n\n## Exploration Context\nExploration ID: {exp_id}\nDiscovered Apps: {', '.join(exploration_data.get('discovered_apps', []))}\n\n## Exploration Actions\n{actions_text}\n\n{task_requirements_text}",
            })

            # Add screenshots
            for screenshot in exploration_screenshots[:15]:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot['base64']}",
                        "detail": "high",
                    },
                })

            # Call LLM
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self._call_openai_api(content)
                    if not response:
                        continue

                    cleaned = response.strip()
                    for marker in ["```json", "```"]:
                        if cleaned.startswith(marker):
                            cleaned = cleaned[len(marker):]
                    for marker in ["```", "`"]:
                        if cleaned.endswith(marker):
                            cleaned = cleaned[:-len(marker)]
                    cleaned = cleaned.strip()

                    tasks_data = json.loads(cleaned)
                    tasks = tasks_data.get("tasks", [])
                    break
                except json.JSONDecodeError as e:
                    print(f"  Attempt {attempt+1}: JSON parse error: {e}")
                    if attempt == max_retries - 1:
                        print(f"  Failed to parse response after {max_retries} attempts")
                        continue

            if not tasks:
                continue

            # Generate task files
            context_filename = f"{self.software}_{exp_id}"
            for task in tasks:
                task_id = task.get("task_id")
                if isinstance(task_id, str) and task_id.startswith("task_"):
                    task["task_id"] = f"task_{str(uuid.uuid4())[:16]}"

                context_template: Dict[str, Any] = {
                    "source": "self_evolve",
                    "exploration_id": exp_id,
                    "environment": exploration_data.get("environment", ""),
                    "round_id": iteration,
                }

                if init_result:
                    context_template["params_path"] = init_result.get("params_path", "")
                    context_template["init_screenshot"] = init_result.get("init_screenshot", "")

                # Inherit snapshot from init_result if available
                if init_result.get("task_name"):
                    context_template["snapshot"] = init_result.get("task_name")

                try:
                    config_file = self.create_task_config_file(
                        task_data=task,
                        context_filename=context_filename,
                        output_dir=output_dir,
                        context_template=context_template,
                    )
                    generated_files.append(config_file)
                except ValueError as e:
                    print(f"  Skipping task {task_id}: {e}")

            print(f"  Context {ctx_idx+1}/{len(exploration_contexts)}: "
                  f"generated {len(tasks)} tasks from {exp_id}")

        # Update task index
        if generated_files and index_dir:
            self.update_task_index_json(
                generated_task_files=generated_files,
                context_filename=self.software,
                output_dir=output_dir,
                model=self.model,
                index_dir=index_dir,
            )

        print(f"[SelfEvolve] Total: {len(generated_files)} task files generated")
        return generated_files

    def generate_tasks_from_context(
        self,
        environment_exploration_dir: str,
        context_review_dir: str,
        context_filename: str,
        output_dir: str,
        index_dir: str,
        existing_task_files: Optional[List[str]] = None,
        feedback_file: Optional[str] = None,
        iteration: int = 0,
        context_template: Optional[Dict[str, Any]] = None,
        environment_extra_sources: Optional[Sequence[str]] = None,
        context_extra_sources: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Generate tasks from a single mobile context review sample."""
        print(f"Generating tasks from context: {context_filename}")

        # Load trajectory data from trajectory directory
        trajectory_data = self.load_trajectory_data(environment_exploration_dir)
        environment_exploration_screenshots = self.get_environment_exploration_screenshots(
            environment_exploration_dir,
            extra_sources=environment_extra_sources,
        )

        # Get context review screenshots
        context_review_screenshots = self.get_context_review_screenshots(
            context_review_dir,
            extra_sources=context_extra_sources,
        )

        if not context_review_screenshots:
            print("Warning: No context review screenshots found")
            return []

        # Format environment exploration actions
        environment_exploration_actions = self.format_environment_exploration_actions(trajectory_data)

        existing_tasks: Optional[List[str]] = None
        if self.enable_diversity and existing_task_files:
            existing_tasks = self.load_existing_tasks_for_diversity(
                existing_task_files,
                context_filename,
                iteration,
            )
            print(f"Loaded {len(existing_tasks)} existing tasks for diversity enhancement")

        feedback_data: Optional[Dict[str, Any]] = None
        if feedback_file:
            feedback_data = self.load_feedback_data(feedback_file)
            if feedback_data:
                print("Loaded feedback data for adaptive task generation")
            else:
                iteration = 0
        else:
            iteration = 0

        desired_total_tasks = self.easy_tasks + self.medium_tasks + self.hard_tasks
        max_retries = 3
        attempt_index = 0
        tasks: List[Dict[str, Any]] = []

        while attempt_index < max_retries:
            print("Generating specific tasks based on initial state...")
            tasks_response = self.generate_specific_tasks(
                environment_exploration_actions=environment_exploration_actions,
                environment_exploration_screenshots=environment_exploration_screenshots,
                context_review_screenshots=context_review_screenshots,
                existing_tasks=existing_tasks,
                feedback_data=feedback_data,
                iteration=iteration,
                context_filename=context_filename,
            )

            if not tasks_response:
                if attempt_index < max_retries - 1:
                    print("Failed to generate tasks response, retrying...")
                else:
                    print("Failed to generate tasks")
                attempt_index += 1
                continue

            cleaned_response = tasks_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            try:
                tasks_data = json.loads(cleaned_response)
                tasks = tasks_data.get("tasks", [])

                if len(tasks) != desired_total_tasks:
                    if attempt_index < max_retries - 1:
                        print(f"Warning: Expected {desired_total_tasks} tasks, got {len(tasks)}. Retrying...")
                        attempt_index += 1
                        continue
                    print(f"Warning: Expected {desired_total_tasks} tasks, got {len(tasks)}.")

                for task in tasks:
                    task_id = task.get("task_id")
                    if isinstance(task_id, str) and task_id.startswith("task_"):
                        task["task_id"] = f"task_{str(uuid.uuid4())[:16]}"
                break
            except json.JSONDecodeError as err:
                print(f"Error parsing tasks JSON: {err}")
                print(f"Cleaned response: {cleaned_response}")
                attempt_index += 1
                if attempt_index < max_retries:
                    print("Retrying...")
                    continue
                return []

        if not tasks:
            return []

        os.makedirs(output_dir, exist_ok=True)

        generated_files: List[str] = []
        for task in tasks:
            config_file = self.create_task_config_file(
                task_data=task,
                context_filename=context_filename,
                output_dir=output_dir,
                context_template=context_template,
            )
            generated_files.append(config_file)

        print(f"Generated {len(generated_files)} specific task configuration files")

        task_index_file = self.update_task_index_json(
            generated_task_files=generated_files,
            context_filename=context_filename,
            output_dir=output_dir,
            model=self.model,
            index_dir=index_dir,
        )
        if task_index_file:
            print(f"Generated task index JSON file: {task_index_file}")

        return generated_files


def process_all_tasks(generator: MobileSpecificTaskGenerator, args: argparse.Namespace, software: str):
    """Process all tasks in context review folder for mobile generation."""
    context_review_dir = Path(args.context_review_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context_pattern = f"{software}.*"
    context_review_dirs = list(context_review_dir.glob(context_pattern))

    if not context_review_dirs:
        print(f"No context review directories found matching pattern: {context_pattern}")
        return

    print(f"Found {len(context_review_dirs)} context review directories")
    print(f"Context review directory: {context_review_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Diversity enabled: {args.enable_diversity}")
    print(f"Software: {software}")
    print(f"Mobile templates root: {args.mobile_templates_dir}")

    context_count = 0
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for context_review_item_dir in sorted(context_review_dirs):
        if not context_review_item_dir.is_dir():
            continue

        context_count += 1
        context_review_name = context_review_item_dir.name
        print(f"Processing context review {context_count}: {context_review_name}")

        training_dir = context_review_item_dir / "training_step_0"
        if not training_dir.exists():
            print(f"Warning: No training_step_0 directory found in {context_review_name}")
            failed_count += 1
            continue

        uid_dirs = [d for d in training_dir.iterdir() if d.is_dir() and len(d.name.split("-")) == 5]
        if not uid_dirs:
            print(f"Warning: No UID directory found in {context_review_name}/training_step_0")
            failed_count += 1
            continue

        context_review_path = str(uid_dirs[0])
        context_filename = context_review_name.replace(f"{software}.", "")

        tasks_root = Path(args.mobile_templates_dir) / software
        context_task_json = tasks_root / "context_review" / f"{context_filename}.json"
        context_template = None
        if context_task_json.exists():
            with open(context_task_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                context_template = data
        else:
            print(f"Warning: mobile context template not found: {context_task_json}")

        print(f"Context review path: {context_review_path}")
        print(f"Initial state file: {context_filename}")

        existing_generated_files = list(output_dir.glob(f"{context_filename}_task_*.json"))
        if existing_generated_files:
            print(
                f"Skipping {context_review_name} - already processed ({len(existing_generated_files)} task files found)"
            )
            skipped_count += 1
            continue

        existing_task_files: List[str] = []
        if args.enable_diversity:
            all_existing_dirs: List[Path] = []
            if args.existing_tasks_dir:
                existing_tasks_dir = Path(args.existing_tasks_dir)
                if existing_tasks_dir.exists():
                    all_existing_dirs.append(existing_tasks_dir)

            for existing_dir in all_existing_dirs:
                task_files = list(existing_dir.glob("**/*_task_*.json"))
                existing_task_files.extend([str(f) for f in task_files])

            if existing_task_files:
                print(f"Using diversity enhancement with {len(existing_task_files)} existing task files")
                print(f"Scanned directories: {[str(d) for d in all_existing_dirs]}")

        print("Running mobile generator...")
        try:
            generated_files = generator.generate_tasks_from_context(
                environment_exploration_dir=str(args.environment_exploration_dir),
                context_review_dir=context_review_path,
                context_filename=context_filename,
                output_dir=str(output_dir),
                index_dir=args.index_dir,
                existing_task_files=existing_task_files,
                feedback_file=getattr(args, "feedback_file", None),
                iteration=getattr(args, "iteration", 0),
                context_template=context_template,
            )

            if generated_files:
                print(f"Context review {context_review_name} completed successfully")
                success_count += 1
            else:
                print(f"Context review {context_review_name} failed - no files generated")
                failed_count += 1

        except Exception as err:
            print(f"Context review {context_review_name} failed with error: {err}")
            failed_count += 1

    print("Processing Summary:")
    print(f"Total context reviews found: {context_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {output_dir}")
    print(f"Diversity enabled: {args.enable_diversity}")
    print(f"Software: {software}")


def parse_bool(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return False
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw_value}")


def main():
    parser = argparse.ArgumentParser(description="Generate mobile learning tasks from context review")
    parser.add_argument(
        "--environment-exploration-dir",
        required=True,
        help="Directory containing trajectory data",
    )
    parser.add_argument(
        "--context-review-dir",
        required=True,
        help="Directory containing context review task folders",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Output directory for generated task files",
    )
    parser.add_argument(
        "--index-dir",
        required=True,
        help="Directory for task index file",
    )
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--enable-diversity",
        required=True,
        help="Enable diversity enhancement (true/false)",
    )
    parser.add_argument(
        "--existing-tasks-dir",
        help="Directory containing existing task files for diversity enhancement",
    )
    parser.add_argument("--software", required=True, help="Software name")
    parser.add_argument("--easy-tasks", type=int, default=16, help="Number of easy tasks to generate")
    parser.add_argument(
        "--medium-tasks",
        type=int,
        default=10,
        help="Number of medium tasks to generate",
    )
    parser.add_argument("--hard-tasks", type=int, default=6, help="Number of hard tasks to generate")
    parser.add_argument(
        "--feedback-file",
        help="Path to feedback file from previous iterations",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Current iteration number",
    )
    parser.add_argument(
        "--mobile-templates-dir",
        default="./data/tasks/examples/mobile",
        help="Root directory for mobile context templates",
    )
    parser.add_argument(
        "--image-download-timeout",
        type=int,
        default=10,
        help="Timeout in seconds for remote image downloads",
    )

    args = parser.parse_args()

    if not os.path.exists(args.environment_exploration_dir):
        print(f"Error: Environment exploration directory {args.environment_exploration_dir} does not exist")
        return

    if not os.path.exists(args.context_review_dir):
        print(f"Error: Context review directory {args.context_review_dir} does not exist")
        return

    enable_diversity = parse_bool(args.enable_diversity)
    args.enable_diversity = enable_diversity

    generator = MobileSpecificTaskGenerator(
        openai_api_key=args.api_key,
        model=args.model,
        enable_diversity=enable_diversity,
        software=args.software,
        easy_tasks=args.easy_tasks,
        medium_tasks=args.medium_tasks,
        hard_tasks=args.hard_tasks,
        image_download_timeout=args.image_download_timeout,
    )

    process_all_tasks(generator, args, args.software)


    def load_exploration_data(self, exploration_dir: str) -> Dict[str, Any]:
        """Load AndroidWorld/MobileWorld exploration data.

        Args:
            exploration_dir: Directory containing exploration output

        Returns:
            Dictionary with exploration result and trajectory
        """
        exp_path = Path(exploration_dir)
        result_files = list(exp_path.glob("*_result.json"))
        if not result_files:
            raise FileNotFoundError(f"No exploration result found in {exploration_dir}")

        with open(result_files[0], "r", encoding="utf-8") as f:
            result = json.load(f)

        return result

    def load_init_data(self, init_output_dir: str) -> Dict[str, Dict[str, Any]]:
        """Load task initialization results.

        Args:
            init_output_dir: Directory containing task initialization outputs

        Returns:
            Dictionary mapping task_name_instance to init result
        """
        init_path = Path(init_output_dir)
        init_results = {}

        for task_dir in init_path.iterdir():
            if not task_dir.is_dir():
                continue
            for instance_dir in task_dir.iterdir():
                if not instance_dir.is_dir():
                    continue
                result_file = instance_dir / "task_init_result.json"
                if result_file.exists():
                    with open(result_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    key = f"{data['task_name']}_{data['instance_id']}"
                    init_results[key] = data

        return init_results

    def format_context_review_for_prompt(
        self,
        init_results: Dict[str, Dict[str, Any]],
        max_tasks: int = 10,
    ) -> str:
        """Format initialization results as context review text for prompt.

        Args:
            init_results: Dictionary of task initialization results
            max_tasks: Maximum number of tasks to include in prompt

        Returns:
            Formatted text describing context review results
        """
        lines = ["## Task Initialization Results\n"]
        lines.append(f"Total tasks reviewed: {len(init_results)}\n")

        success_count = sum(
            1 for r in init_results.values() if r.get("initialization", {}).get("success", False)
        )
        lines.append(f"Successfully initialized: {success_count}\n")
        lines.append(f"Failed: {len(init_results) - success_count}\n\n")

        for i, (key, result) in enumerate(init_results.items()):
            if i >= max_tasks:
                break
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


# Compatibility alias for drop-in usage.
SpecificTaskGenerator = MobileSpecificTaskGenerator


if __name__ == "__main__":
    main()
