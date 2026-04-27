import asyncio
import base64
import io
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .utils import OpenaiEngine, vllm_OpenaiEngine

identify_key_points_system_msg = """You are an expert evaluator for mobile GUI tasks.

Objective:
Extract only the explicit key points that must be satisfied for successful task completion.

Instructions:
1. Read the task carefully.
2. Extract key points that are directly stated.
3. Do not infer unstated requirements.
4. If terms imply sorting or ranking (best, highest, cheapest, latest, nearest, etc.), convert them into explicit key points (for example: "Sort by highest").

Respond with:
Key Points: a numbered list of explicit key points, one per line.
"""

identify_key_screenshot_system_msg = """You are an expert evaluator for mobile GUI trajectories.

Objective:
Determine whether the provided screenshot contains evidence that is important for judging task success.

Instructions:
1. Describe what is visible on the mobile screen.
2. Decide whether the screenshot includes useful evidence for key task requirements.
3. Consider mobile signals such as app state, page transitions, selected options, permission dialogs, keyboard input, navigation outcomes, and confirmation results.

Output format:
### Reasoning: <your reasoning>
### Score: <1-5>

Scoring:
1 = no useful evidence
2 = weak or ambiguous evidence
3 = partially useful evidence
4 = strong evidence but incomplete
5 = clear critical evidence
"""

outcome_system_msg = """You are an expert evaluator of mobile computer-use agents.

You must determine whether the task is successfully completed by analyzing:
- initial screenshot(s)
- final screenshot
- action history
- selected intermediate screenshots

Evaluation process:
1) Understand the task and its key points.
2) Compare initial and final screen states.
3) Verify each key point with explicit evidence.
4) Apply strict criteria:
   - No unrelated operations outside task scope.
   - Correct mobile UI semantics (correct app/screen/component).
   - Required filters/ranking/constraints must be correctly applied.
   - Repeated non-progress actions indicate failure.
   - If task requires commit actions (submit/save/confirm), they must be completed.
5) Make final decision.

Output exactly two lines:
Thoughts: <concise but explicit verification of each key point>
Status: "success" or "failure"
"""


class MobileJudgeEvaluator:
    def __init__(
        self,
        key_identification_screenshot_model: str = "o4-mini",
        key_points_outcome_model: str = "o4-mini",
        max_image: int = 50,
        score_threshold: int = 3,
        use_vllm_for_key_screenshot: bool = False,
        vllm_base_url: Optional[str] = None,
    ):
        self.max_image = max_image
        self.score_threshold = score_threshold

        if use_vllm_for_key_screenshot:
            assert vllm_base_url is not None, "vllm_base_url must be provided if use_vllm_for_key_screenshot is True"

        if not use_vllm_for_key_screenshot:
            self.key_identification_screenshot_model = OpenaiEngine(model=key_identification_screenshot_model)
        else:
            self.key_identification_screenshot_model = vllm_OpenaiEngine(
                model=key_identification_screenshot_model,
                base_url=vllm_base_url,
            )

        self.key_points_outcome_model = OpenaiEngine(model=key_points_outcome_model)

    def encode_image(self, image: Image.Image) -> str:
        """Convert a PIL image to base64 string."""
        if image.mode == "RGBA":
            image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _encode_image_from_path(self, image_path: str) -> str:
        with Image.open(image_path) as image:
            return self.encode_image(image)

    def _generate_with_usage(self, engine, messages: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, int]]:
        """
        统一处理模型调用返回，兼容 list 与 metadata 两种格式。
        """
        usage: Dict[str, int] = {}
        try:
            result = engine.generate(messages, return_metadata=True)
            if isinstance(result, dict):
                contents = result.get("contents", [])
                usage = result.get("usage", {}) or {}
                return contents, usage
            if isinstance(result, list):
                usage = getattr(engine, "last_usage", {}) or {}
                return result, usage
        except TypeError:
            # 兼容未实现 return_metadata 的历史 engine。
            result = engine.generate(messages)
            usage = getattr(engine, "last_usage", {}) or {}
            return result, usage

        fallback = engine.generate(messages)
        usage = getattr(engine, "last_usage", {}) or {}
        return fallback, usage

    @staticmethod
    def _normalize_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
        if not usage:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
            }
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "cached_tokens": int(usage.get("cached_tokens", 0) or 0),
        }

    @classmethod
    def _sum_usages(cls, usages: List[Dict[str, Any]]) -> Dict[str, int]:
        total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
        for usage in usages:
            normalized = cls._normalize_usage(usage)
            for key in total:
                total[key] += normalized.get(key, 0)
        return total

    async def get_score(self, response: str) -> float:
        reward = 0.0
        try:
            lowered = response.lower()
            if "status:" in lowered:
                status_text = lowered.split("status:", 1)[1]
                reward = 1.0 if "success" in status_text else 0.0
            else:
                reward = 1.0 if "success" in lowered else 0.0
        except Exception:
            reward = 0.0
        return reward

    async def identify_key_points(
        self,
        task: str,
        input_image_paths: Optional[List[str]],
        return_usage: bool = False,
    ):
        text = f"Task: {task}"

        input_images_msg = []
        if input_image_paths is not None:
            for input_image_path in input_image_paths:
                encoded = self._encode_image_from_path(input_image_path)
                input_images_msg.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
                            "detail": "high",
                        },
                    }
                )

        messages = [
            {"role": "system", "content": identify_key_points_system_msg},
            {
                "role": "user",
                "content": [{"type": "text", "text": text}] + input_images_msg,
            },
        ]
        responses, usage = await asyncio.to_thread(
            self._generate_with_usage,
            self.key_points_outcome_model,
            messages,
        )
        response_text = responses[0] if responses else ""
        if return_usage:
            return response_text, self._normalize_usage(usage)
        return response_text

    async def judge_image(
        self,
        task: str,
        input_image_paths: Optional[List[str]],
        image_path: str,
        key_points: str,
        return_usage: bool = False,
    ):
        text = (
            f"**Task**: {task}\n\n"
            f"**Key Points for Task Completion**: {key_points}\n\n"
            "The snapshot of the mobile interface is shown in the image."
        )

        input_images_msg = []
        if input_image_paths is not None:
            for input_image_path in input_image_paths:
                encoded = self._encode_image_from_path(input_image_path)
                input_images_msg.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
                            "detail": "high",
                        },
                    }
                )

        messages = [{"role": "system", "content": identify_key_screenshot_system_msg}]
        if input_images_msg:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "The input images are:"}] + input_images_msg,
                }
            )

        jpg_base64_str = self._encode_image_from_path(image_path)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{jpg_base64_str}",
                            "detail": "high",
                        },
                    },
                ],
            }
        )

        responses, usage = await asyncio.to_thread(
            self._generate_with_usage,
            self.key_identification_screenshot_model,
            messages,
        )
        response_text = responses[0] if responses else ""
        if return_usage:
            return response_text, self._normalize_usage(usage)
        return response_text

    async def cuajudge_general_eval(
        self,
        task: str,
        input_image_paths: Optional[List[str]],
        action_thoughts: Optional[List[str]],
        last_actions: List[str],
        images_path: List[str],
    ):
        prompt = """## Task to Evaluate
{task}

## Key Points for Successful Completion
{key_points}

## Agent's Action History
{last_actions}

## Important Intermediate Screenshots and Analysis
The following screenshots were identified as potentially important in the agent's trajectory:
{thoughts}

---
Now, evaluate whether the agent successfully completed the task following the systematic evaluation process outlined in your instructions."""

        key_points_raw, key_points_usage = await self.identify_key_points(
            task=task,
            input_image_paths=input_image_paths,
            return_usage=True,
        )
        key_points_raw = key_points_raw.replace("\n\n", "\n")

        try:
            key_points = key_points_raw.split("**Key Points**:", 1)[1]
            key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
        except Exception:
            key_points = key_points_raw.split("Key Points:")[-1]
            key_points = "\n".join(line.lstrip() for line in key_points.splitlines())

        tasks = [
            self.judge_image(
                task=task,
                input_image_paths=input_image_paths,
                image_path=image_path,
                key_points=key_points,
                return_usage=True,
            )
            for image_path in images_path
        ]
        image_results = await asyncio.gather(*tasks) if tasks else []

        whole_content_img = []
        whole_thoughts = []
        record = []
        image_usages = []
        pattern = r"[1-5]"

        for (response, usage), image_path in zip(image_results, images_path):
            image_usages.append(usage)
            score = 0
            thought = ""
            try:
                score_text = response.split("### Score", 1)[1]
                thought = (
                    response.split("### Reasoning:")[-1].strip().lstrip("\n").split("### Score")[0].replace("\n", " ")
                )
                score = int(re.findall(pattern, score_text)[0])
            except Exception as err:
                print(f"Error processing response: {err}")
                score = 0

            record.append({"Response": response, "Score": int(score)})
            if int(score) >= self.score_threshold:
                jpg_base64_str = self._encode_image_from_path(image_path)
                whole_content_img.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{jpg_base64_str}",
                            "detail": "high",
                        },
                    }
                )
                if thought:
                    whole_thoughts.append(thought)

        whole_content_img = whole_content_img[: self.max_image]
        whole_thoughts = whole_thoughts[: self.max_image]

        if len(whole_content_img) == 0:
            prompt = """## Task to Evaluate
{task}

## Key Points for Successful Completion
{key_points}

## Agent's Action History
{last_actions}

## Important Intermediate Screenshots and Analysis
No intermediate screenshots were flagged as particularly important. Please rely on the initial and final screenshots, along with the action history, to evaluate task completion.

---
Now, evaluate whether the agent successfully completed the task following the systematic evaluation process outlined in your instructions."""

        if action_thoughts is not None:
            last_actions_text = "\n".join(
                f"{i+1}. {action}. Reasoning: {action_thought}"
                for i, (action, action_thought) in enumerate(zip(last_actions, action_thoughts))
            )
        else:
            last_actions_text = "\n".join(f"{i+1}. {action}" for i, action in enumerate(last_actions))

        text = prompt.format(
            task=task,
            last_actions=last_actions_text,
            key_points=key_points,
            thoughts="\n".join(f"{i+1}. {thought}" for i, thought in enumerate(whole_thoughts)),
        )

        input_images_msg = []
        if input_image_paths is not None:
            for path in input_image_paths:
                encoded = self._encode_image_from_path(path)
                input_images_msg.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
                            "detail": "high",
                        },
                    }
                )

        final_screenshot_msg = []
        if images_path and len(images_path) > 0:
            final_image_path = images_path[-1]
            final_encoded = self._encode_image_from_path(final_image_path)
            final_screenshot_msg.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{final_encoded}",
                        "detail": "high",
                    },
                }
            )

        messages = [{"role": "system", "content": outcome_system_msg}]

        if input_images_msg:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "The initial screenshot(s) are:"}] + input_images_msg,
                }
            )

        if final_screenshot_msg:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "The final screenshot is:"}] + final_screenshot_msg,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": text}] + whole_content_img,
            }
        )

        outcome_responses, outcome_usage = await asyncio.to_thread(
            self._generate_with_usage,
            self.key_points_outcome_model,
            messages,
        )
        response_text = outcome_responses[0] if outcome_responses else ""
        reward = await self.get_score(response_text)

        usage_total = self._sum_usages(
            [
                key_points_usage,
                self._sum_usages(image_usages),
                self._normalize_usage(outcome_usage),
            ]
        )

        # 关键逻辑说明：usage 分层记录，便于离线测试报告统计各阶段 token 开销。
        details = {
            "key_points": key_points,
            "thoughts": whole_thoughts,
            "image_judge_record": record,
            "usage": {
                "key_points": self._normalize_usage(key_points_usage),
                "image_judge_total": self._sum_usages(image_usages),
                "outcome": self._normalize_usage(outcome_usage),
                "total": usage_total,
            },
        }

        return response_text, reward, details

    async def evaluate_trajectory(
        self,
        task: str,
        actions: List[str],
        screenshot_paths: List[str],
        input_image_paths: Optional[List[str]] = None,
    ):
        """标准化轨迹入口，便于离线评测工具直接调用。"""
        return await self.cuajudge_general_eval(
            task=task,
            input_image_paths=input_image_paths,
            action_thoughts=None,
            last_actions=actions,
            images_path=screenshot_paths,
        )


# Compatibility alias for drop-in usage.
CUAJudgeEvaluator = MobileJudgeEvaluator
