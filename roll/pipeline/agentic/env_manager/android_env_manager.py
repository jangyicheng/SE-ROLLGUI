import base64
from contextlib import nullcontext
from threading import Lock
from typing import Dict, List, Optional, Tuple

import gem
import numpy as np
import PIL
import torch
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
from roll.pipeline.agentic.env_manager.token_mask_utils import (
    split_by_token,
    token_ids_to_assistant_mask,
    compute_conversation_end_token_id,
    custom_apply_chat_template,
)
from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.utils.constants import GenerateStopReason
from roll.utils.env_action_limiter import get_global_limiter
from roll.utils.functionals import aggregate_metrics, pad_to_length
from roll.utils.logging import get_logger
from tensordict import TensorDict

logger = get_logger()


def build_user_content(text: str, image: str, info: str | None = None) -> dict:
    if info:
        text = info + text
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text,
            },
            {
                "type": "image",
                "image": f"data:image/jpeg;base64,{image}",
            },
        ],
    }


def extract_summary(response: str):
    return response.split("Action:")[1].strip().split("\n")[0]


SYSTEM_PROMPT = 'You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen\'s resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n# Response format\n\nResponse format for every step:\n1) Thought: one concise sentence explaining the next move (no multi-step reasoning).\n2) Action: a short imperative describing what to do in the UI.\n3) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.\n\nRules:\n- Output exactly in the order: Thought, Action, <tool_call>.\n- Be brief: one sentence for Thought, one for Action.\n- Do not output anything else outside those three parts.\n- If finishing, use action=terminate in the tool call.'
custom_system_prompt = """
# Skill Memory
You can get information from skill memory every turn. It will provide general steps and useful tips to complete the task.
"""


class AndroidTrajEnvManager(TrajEnvManager):
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

    def save_trajectory(self, rollout: RolloutCache):
        """
        保存当前 rollout 到对应 task 的最佳轨迹缓存中。
        只在训练模式下执行，且只保留 step 数最少但 score > 0.5 的轨迹（原逻辑）。
        """
        task = self.task
        
        # 验证模式不保存轨迹
        if self.mode == "val":
            return

        try:
            if rollout.step > 0:
                episode_score = sum(i["reward"] for i in rollout.history)
            else:
                episode_score = 0.0

            # 只关心分数 > 0.5 的轨迹(success)
            if episode_score > 0.5:
                # 初始化该 task 的缓存（如果还没有）
                if self.trajectory_cache is None:
                    self.trajectory_cache = {}

                if task not in self.trajectory_cache:
                    self.trajectory_cache[task] = rollout
                else:
                    # 与该 task 已有最佳轨迹比较
                    current_best = self.trajectory_cache[task]
                    # 原逻辑：保留 step 更少的轨迹（假设步数少代表更高效）
                    if rollout.step < current_best.step:
                        self.trajectory_cache[task] = rollout

        except Exception as e:
            logger.info(f"save_trajectory error for task {task}: {e}")

    def get_best_trajectory(self, rollout: RolloutCache) -> RolloutCache:
        """
        返回当前 task 对应的最佳轨迹（如果存在），否则返回当前传入的 rollout。
        """
        task = self.task

        # 先尝试保存当前 rollout（可能会更新缓存）
        self.save_trajectory(rollout)

        # 如果该 task 有缓存的最佳轨迹，则返回它
        if self.trajectory_cache is not None and task in self.trajectory_cache:
            return self.trajectory_cache[task]
        
        # 否则返回当前 rollout 本身
        return rollout

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
        lm_output: DataProto = self.llm_proxy.generate(
            messages=messages, lm_input=lm_input, generation_config=generation_config
        )

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    @property
    def task(self):
        return self.env.task

    def format_messages(self, history: RolloutCache, return_episode: bool = False) -> Tuple[DataProto, List[Dict]]:
        messages = [
            {"role": "system", "content": self.agent_system_template},
        ]
        images = []
        if not return_episode:
            # the message format in rollout
            previous_action = []
            for idx, content in enumerate(history.history):
                if "llm_response" in content:
                    previous_action.append(
                        f"Step {idx + 1}: " + extract_summary(content["llm_response"].replace("<|im_end|>", ""))
                    )
            images.append([PIL.Image.fromarray(content["observation"], mode="RGB")])
            previous_action = "\n".join(previous_action)
            messages.append(
                build_user_content(
                    f"The user query: {self.env.task["goal"]}\nTask progress (You have done the following operation on the current device): \n{'None' if previous_action == '' else previous_action}.",
                    base64.b64encode(content["observation"]).decode("utf-8"),
                    info=get_skill(self.memory, self.task.skill_key, self.task._get_params()) if self.memory else None,
                )
            )
        else:
            # return the full episode messages following the format: (obs, response, obs, response, ...)
            previous_action = ""
            for idx, content in enumerate(history.history):
                assert "observation" in content, (
                    "The current EnvManager is specifically tailored for standard RL interaction "
                    "sequences, following the format of (s, a, r, s, a, r...)."
                )
                messages.append(
                    build_user_content(
                        f"The user query: {self.env.task["goal"]}.\nTask progress (You have done the following operation on the current device): {'None' if previous_action == '' else previous_action}.",
                        base64.b64encode(content["observation"]).decode("utf-8"),
                        info=get_skill(self.memory, self.task.skill_key, self.task._get_params())
                        if self.memory
                        else None,
                    )
                )
                images.append(PIL.Image.fromarray(content["observation"], mode="RGB"))

                if "llm_response" in content:
                    messages.append({
                        "role": "assistant",
                        "content": content["llm_response"].replace("<|im_end|>", ""),
                    })
                    previous_action += (
                        f"Step {idx + 1}: " + extract_summary(content["llm_response"].replace("<|im_end|>", "")) + "\n"
                    )
        lm_input_texts = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        features = [
            {
                self.collator.prompt_key: lm_input_texts,
                self.collator.image_key: images,
                self.collator.image_flag_key: True,
            }
        ]
        inputs = self.collator(features)
        lm_input: DataProto = DataProto.from_single_dict(inputs)
        # 取出必要的 tensor
        input_ids = inputs["input_ids"]           # shape: (1, seq_len)
        attention_mask = inputs.get("attention_mask")  # 有些 processor 会自动生成，有些不会
        attention_mask = attention_mask.to(dtype=torch.long)

        # 构建 position_ids
        # 最常见做法：直接用 attention_mask 的累积和（从 0 开始）
        position_ids = attention_mask.cumsum(dim=-1)   # 让它从 0 开始      
        needed_keys = {"attention_mask", "position_ids"}

        for key in needed_keys:
            if key not in lm_input.batch:
                if key == "attention_mask":
                    lm_input.batch[key] = attention_mask
                elif key == "position_ids":
                    lm_input.batch[key] = position_ids

        
        return lm_input , messages

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        # TODO: check inconsistent tokenization between successive encode-decode operations
        #  can potentially lead to a training crash. check token in token out
        #  the same as TrajEnvManager.

        if "observation" in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)
        rollout_cache = self.get_best_trajectory(rollout=rollout_cache)

        scores = [i["reward"] for i in self.rollout_cache.history]
        episode_score = sum(scores)

        lm_input, messages = self.format_messages(rollout_cache, return_episode=True)

        input_ids = lm_input.batch["input_ids"]
        trajectory_token_lens = input_ids.shape[-1]
        attention_mask = lm_input.batch["attention_mask"]
        start_idx = (input_ids[0] == 151644).to(torch.int8).argmax().item() #找到第一个特殊 token（常见为 <|im_start|>，用于跳过前面可能的系统提示或非对话部分

        token_ids = input_ids[0, start_idx:].tolist()
        token_ids_split = split_by_token(token_ids, token_ids[0])
        response_masks_list = token_ids_to_assistant_mask(
            messages=messages, input_ids_list=token_ids_split, tokenizer=self.tokenizer
        )
        response_masks = [item for items in response_masks_list for item in items]

        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)

        # first_response_idx = response_masks.index(1)
        last_response_idx = len(response_masks) - 1 - response_masks[::-1].index(1)# 找到最后一个 assistant token 的位置（通常是最后一句回复的结束位置）
        # prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        # prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
        prompt_mask = ~response_mask
        score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
        score_tensor[0][last_response_idx] = episode_score

        input_ids = input_ids[:, start_idx : last_response_idx + 1]
        attention_mask = attention_mask[:, start_idx : last_response_idx + 1]
        position_ids = lm_input.batch["position_ids"][:, start_idx : last_response_idx + 1]
        
        logger.info(f"Rollout token lens: {input_ids.shape}")
        response_length = response_mask.sum(dim=-1).float().mean().item()
        input_ids = pad_to_length(
            input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id
        )
        logger.info(f"Rollout token lens after padding: {input_ids.shape}")
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        
        lm_input.batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
            "position_ids":position_ids
        })
        lm_input.non_tensor_batch.update({
            "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
            "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
            "messages_list": np.array([messages], dtype=object),
            "tags": np.array([self.rollout_cache.tag], dtype=object),
            "step_scores": np.array([scores], dtype=object),
            "episode_scores": np.array([episode_score], dtype=object),
        })

        metrics_agg_mode = self.rollout_cache.history[-1].get("metrics_agg_mode", {})
        history_metrics = [item.get("metrics", {}) for item in self.rollout_cache.history]
        env_metric = aggregate_metrics(history_metrics=history_metrics, metrics_agg_mode=metrics_agg_mode)

        metric_names = {
            "num_actions": rollout_cache.step,
            "trajectory_token_lens": trajectory_token_lens,
            "response_length": response_length,
        }
        for metric_name, val in metric_names.items():
            for agg in ["mean", "max", "min"]:
                env_metric[f"{metric_name}_{agg}"] = val

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        lm_input.meta_info = {"metrics": env_metric}
        return lm_input


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
