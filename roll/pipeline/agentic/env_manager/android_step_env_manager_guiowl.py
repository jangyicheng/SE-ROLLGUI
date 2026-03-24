import base64
from contextlib import nullcontext
from threading import Lock
from typing import Dict, List, Optional, Tuple

import gem
import numpy as np
import PIL
import ray
import torch
from codetiming import Timer
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin

from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.env_manager.android_utils import get_skill, read_memory
from roll.pipeline.agentic.env_manager.base_env_manager import (
    BaseEnvManager,
    RolloutCache,
)
# from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.pipeline.agentic.env_manager.gui_traj_env_manager import GuiTrajEnvManager 
from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.utils.constants import GenerateStopReason
from roll.utils.env_action_limiter import get_global_limiter
from roll.utils.functionals import aggregate_metrics, pad_to_length
from roll.utils.hash_utils import compute_object_hash
from roll.utils.logging import get_logger


logger = get_logger()

import copy
from io import BytesIO

def pil_to_base64(image: PIL.Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_summary(response: str):
    try:
        return response.split("Action:")[-1].split("<tool_call>")[0].strip()
    except Exception:
        return "Invalid action format, executed failed."



# for qwen2.5-VL
# SYSTEM_PROMPT = 'You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen\'s resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n# Response format\n\nResponse format for every step:\n1) Thought: one concise sentence explaining the next move (no multi-step reasoning).\n2) Action: a short imperative describing what to do in the UI.\n3) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.\n\nRules:\n- Output exactly in the order: Thought, Action, <tool_call>.\n- Be brief: one sentence for Thought, one for Action.\n- Do not output anything else outside those three parts.\n- If finishing, use action=terminate in the tool call.'

# for Gui-Owl
SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `answer`: Terminate the current task and output the answer.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "answer", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=open`, `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one for Action.
- Do not output anything else outside those two parts.
- If finishing, use action=terminate in the tool call.'''

custom_system_prompt = """
# Skill Memory
You can get information from skill memory every turn. It will provide general steps and useful tips to complete the task.
"""



class AndroidStepEnvManager(GuiTrajEnvManager):
    def __init__(
        self,
        worker_config: EnvManagerConfig,
        pipeline_config: AgenticConfig,
        env_config: Dict,
        tokenizer: PreTrainedTokenizer,
        processor: ProcessorMixin,
        generate_scheduler,
        output_queue: GroupQueueManager,
        thread_lock: Lock,
        mode="train",
        extra_data_provider=None,
        *args,
        **kwargs,
    ):
        """
        rollout_data_type:
            - step: the trajectory is stored step by step, each step contains observation and llm_response
                Data1: System Prompt + Observation1 -> LLM Response1
                Data2: System Prompt + His2 + Observation2 -> LLM Response2
                Data3: System Prompt + His3 + Observation3 -> LLM Response3
                ...
            - trajectory: the trajectory is stored as a whole episode, containing all observations and llm_responses
                Data1: System Prompt + Observation1 + LLM Response1 + Observation2 + LLM Response2 + ... + ObservationN + LLM ResponseN
        """
        BaseEnvManager.__init__(self)
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: Dict = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor: ProcessorMixin = processor
        self.extra_data_provider = extra_data_provider
        self.collator = DataCollatorWithPaddingForMM(
            tokenizer=self.tokenizer,
            processor=self.processor,
            answer_key=None,
            extra_data_provider=self.extra_data_provider,
        )
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RequestScheduler = generate_scheduler

        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = 0
        self.current_step = -1
        self.running = False
        self.use_thread_lock = self.env_config.get(
            "use_thread_lock", False
        )  # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = nullcontext()
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)
        self.env_config["config"].update({
            "android_env_id": self.env_config["env_id"],
            "android_group_id": self.env_config["group_id"],
            "max_steps": self.env_config["max_steps"],
        })
        self.rollout_data_type = self.env_config.get("rollout_data_type", "trajectory")
        with self.thread_lock, self.env_step_limiter:
            self.env = gem.make(env_id=self.env_config["env_type"], **self.env_config["config"])

        self.agent_system_template = SYSTEM_PROMPT

        if "memory" in self.env_config:
            self.memory_path = self.env_config["memory"]
            self.memory = read_memory(self.memory_path)
            self.agent_system_template += custom_system_prompt
        else:
            self.memory = None

        # TODO: add rewards_scheduler for local ray reward workers
        # TODO: make it to be a Agent System
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env,
        )

        self.trajectory_cache: Optional[RolloutCache] = None
        self.cur_user_messages = []
        self.last_image = self.env_config.get("last_image", 3)
        
    @property
    def task(self):
        return self.env.task


    def make_decision(self, rollout_cache: RolloutCache):
        lm_input, messages = self.format_messages(rollout_cache)

        input_ids = lm_input.batch["input_ids"]
        if input_ids.shape[1] >= self.pipeline_config.sequence_length:
            self.logger.warning(
                f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {input_ids.shape[1]},"
                f"maybe you should increase the response_length"
            )
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        max_new_tokens = min(
            self.env_config["max_tokens_per_step"],
            self.worker_config.generating_args.max_new_tokens,
            self.pipeline_config.sequence_length - input_ids.shape[1],
        )
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]
        # todo: make retry if the format is incorrect
        lm_output: DataProto = self.llm_proxy.generate(
            messages=messages, lm_input=lm_input, generation_config=generation_config
        )

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})
        response_ids = lm_output.batch["responses"][0]
        response_ids = response_ids.tolist()  # to one-dim list
        content = self.rollout_cache.history[-1]

        if "infer_logprobs" in lm_output.batch:
            infer_logprobs = lm_output.batch["infer_logprobs"][0][-len(response_ids) :]
            content["infer_logprobs"] = infer_logprobs.tolist()

        content["response_ids"] = response_ids
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output


    def _build_instruction_text(self, goal: str, previous_actions: list[str]) -> str:
        if not previous_actions:
            history = "No previous action."
        else:
            history = "\n".join([f"Step{i+1}: {a}" for i, a in enumerate(previous_actions)])
        return (
            "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {goal}\n\n"
            f"Previous actions:\n{history}"
        )

    # 放到 AndroidStepEnvManager 类内
    def cut_current_messages(self, messages, last_image=2):
        non_empty_user_indices = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and msg.get("content") and len(msg["content"]) > 0:
                non_empty_user_indices.append(i)

        if len(non_empty_user_indices) > last_image:
            indices_to_clear = non_empty_user_indices[:-last_image]
        else:
            indices_to_clear = []

        for index in indices_to_clear:
            if index == 1:
                messages[index]["content"] = [messages[index]["content"][0]]
            else:
                messages[index]["content"] = []
        return messages


    def convert_format(self, goal, messages):
        new_messages = copy.deepcopy(messages[:1])
        history = []

        for i, msg in enumerate(messages):
            if (
                msg.get("role") == "user"
                and (msg["content"] == [] or (len(msg["content"]) == 1 and msg["content"][0]["type"] == "text"))
                and i + 1 < len(messages)
                and messages[i + 1].get("role") == "assistant"
                and messages[i + 1].get("content")
            ):
                history.append(
                    messages[i + 1]["content"][0]["text"]
                    .split("Action:")[-1]
                    .split("<tool_call>")[0]
                    .strip()
                )

            if i != 1 and msg.get("role") == "user" and msg["content"] != []:
                if len(history) == 0:
                    new_messages = copy.deepcopy(messages)
                    new_messages[1]["content"][0]["text"] = self._build_instruction_text(goal, [])
                    return new_messages

                new_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._build_instruction_text(goal, history)},
                            {"type": "image_url", "image_url": {"url": msg["content"][0]["image_url"]["url"]}},
                        ],
                    }
                )
                new_messages += copy.deepcopy(messages[i + 1 :])
                return new_messages

        return copy.deepcopy(messages)
        

    # 3) 整体替换 format_messages
    def format_messages(self, rollout_cache: RolloutCache) -> Tuple[DataProto, List[Dict]]:
        content = rollout_cache.history[-1]
        observation = content["observation"]  # np.ndarray(H,W,3)
        image_pil = PIL.Image.fromarray(observation, mode="RGB")
        screenshot_url = pil_to_base64(image_pil)
        goal = self.env.task["goal"] if isinstance(self.env.task, dict) else self.env.task.goal
        
        last_message = rollout_cache.history[-1].get("messages", [])
        if last_message and len(last_message) > 0:
            self.cur_user_messages.append(last_message[-1])

        # 1) 累积 GUI-Owl 风格消息
        if len(self.cur_user_messages) == 0:
            first_text = self._build_instruction_text(goal, [])
            self.cur_user_messages = [
                {"role": "system", "content": [{"type": "text", "text": self.agent_system_template}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": first_text},
                        {"type": "image_url", "image_url": {"url": screenshot_url}},
                    ],
                },
            ]
        else:
            self.cur_user_messages.append(
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": screenshot_url}}]}
            )

        # 2) 控制上下文长度 + 压缩历史
        self.cur_user_messages = self.cut_current_messages(self.cur_user_messages, self.last_image)
        messages = self.convert_format(goal, self.cur_user_messages)

        # 3) 转成 processor 兼容格式（image_url -> image）
        processor_messages = copy.deepcopy(messages)
        for msg in processor_messages:
            if msg.get("role") != "user" or not msg.get("content"):
                continue
            for c in msg["content"]:
                if c.get("type") == "image_url":
                    c["type"] = "image"
                    c["image"] = f"data:image/png;base64,{c['image_url']['url']}"
                    c.pop("image_url", None)

        lm_input_texts = self.processor.apply_chat_template(
            processor_messages, add_generation_prompt=True, tokenize=False
        )

        # 本步只有当前截图一张，和 gui_owl 实际调用一致
        images = [[image_pil]]
        features = [
            {
                self.collator.prompt_key: lm_input_texts,
                self.collator.image_key: images,
                self.collator.image_flag_key: True,
            }
        ]
        inputs = self.collator(features)
        lm_input: DataProto = DataProto.from_single_dict(inputs)

        input_ids = lm_input.batch["input_ids"]
        attention_mask = lm_input.batch["attention_mask"]
        lm_input.batch.update({"position_ids": attention_mask.cumsum(dim=-1)})

        current_cache = rollout_cache.history[-1]
        current_cache["prompt_ids"] = input_ids.tolist()[0]
        current_cache["state_hash"] = compute_object_hash(observation)
        current_cache["messages"] = messages
        current_cache["non_tensor_batch"] = lm_input.non_tensor_batch

        return lm_input, messages

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        """
        Construct step-wise training samples from the collected trajectory.
        """
        if "observation" in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)
        
        samples: List[DataProto] = []
        episode_score = sum([i["reward"] for i in self.rollout_cache.history])
        for step, history in enumerate(rollout_cache.history):
            token_ids = history["prompt_ids"] + history["response_ids"]
            response_masks = [0] * len(history["prompt_ids"]) + [1] * len(history["response_ids"])
            input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
            response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)
            infer_logprobs = []
            if "infer_logprobs" in history:
                infer_logprobs = [0] * len(history["prompt_ids"]) + history["infer_logprobs"]

            first_response_idx = response_masks.index(1)
            prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
            prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
            score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
            # score_tensor[0][-1] = max(history["reward"], episode_score)
            score_tensor[0][-1] = history["reward"]
            position_ids = attention_mask.cumsum(dim=-1)

            input_ids = pad_to_length(
                input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id
            )
            attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
            response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)
            lm_input = DataProto(
                batch=TensorDict(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "response_mask": response_mask,
                        "prompt_mask": prompt_mask,
                        "scores": score_tensor,
                    },
                    batch_size=input_ids.shape[0],
                ),
                non_tensor_batch={
                    "episode_scores": np.array([episode_score], dtype=object),
                    "step_scores": np.array([history["reward"]], dtype=object),  # step-level reward, return by env
                    "tags": np.array([self.rollout_cache.tag], dtype=object),
                    "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
                    "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
                    "state_hash": np.array([history["state_hash"]], dtype=object),
                    "step": np.array([step], dtype=object),
                },
            )
            lm_input.non_tensor_batch.update(history["non_tensor_batch"])
            lm_input.meta_info.update({"task": self.task["name"]})
            
            
            if len(infer_logprobs):
                infer_logprobs = torch.tensor(infer_logprobs, dtype=torch.float).unsqueeze(0)
                infer_logprobs = pad_to_length(
                    infer_logprobs, length=self.pipeline_config.sequence_length, pad_value=0
                )
                lm_input.batch["infer_logprobs"] = infer_logprobs[:, 1:]

            samples.append(lm_input)
        batch: DataProto = DataProto.concat(samples)

        response_length = batch.batch["response_mask"].float().sum(-1).mean().item()
        metrics_agg_mode = self.rollout_cache.history[-1].get("metrics_agg_mode", {})
        history_metrics = [item.get("metrics", {}) for item in self.rollout_cache.history]
        env_metric = aggregate_metrics(history_metrics=history_metrics, metrics_agg_mode=metrics_agg_mode)
        metric_names = {
            "num_actions": rollout_cache.step,
            "response_length": response_length,
        }
        for metric_name, val in metric_names.items():
            for agg in ["mean", "max", "min"]:
                env_metric[f"{metric_name}_{agg}"] = val

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        batch.meta_info = {"metrics": env_metric}
        
        # 保存 DataProto 到全局轨迹缓存
        try:
            self.save_trajectory(rollout=batch)
        except Exception as e:
            self.logger.debug(f"save_trajectory failed: {e}")        



        return batch


class AndroidEnvGroupFilter:
    def __init__(self, config: AgenticConfig, env_manager_config: EnvManagerConfig, mode: str):
        self.config = config
        self.env_manager_config = env_manager_config
        self.mode = mode

    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        """
        return True to filter out this group
        """
        if self.mode == "val":
            return False
        group_episode_reward = np.array([rollout.non_tensor_batch["episode_scores"][0] for rollout in group])
        reward_mean = group_episode_reward.mean()
        reward_std = group_episode_reward.std()
        if reward_mean == 0:
            logger.info(
                f"Filter group {group_id} episode {episode_id} with reward mean {reward_mean:.4f} and std {reward_std:.4f}"
            )
            return True
        return False