import base64
from contextlib import nullcontext
import re
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
from roll.distributed.scheduler.router import RouterManager
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.env_manager.android_utils import read_memory
from roll.pipeline.agentic.env_manager.base_env_manager import (
    BaseEnvManager,
    RolloutCache,
)
# from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.pipeline.agentic.env_manager.gui_traj_env_manager import GuiTrajEnvManager 
from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.utils.constants import EpisodeStopReason, GenerateStopReason, RAY_NAMESPACE
from roll.utils.env_action_limiter import get_global_limiter
from roll.utils.functionals import aggregate_metrics, pad_to_length
from roll.utils.hash_utils import compute_object_hash
from roll.utils.logging import get_logger


logger = get_logger()


def extract_summary(response: str): # 该函数可能会报错
    # return response
    try:
        summary = response.split("Action:")[1].strip().split("\n")[0]
    except:
        summary = "Action: Invalid action format, executed failed."
    return summary

SYSTEM_PROMPT = """
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Thought: one concise sentence explaining the next move (no multi-step reasoning).
2) Action: a short imperative describing what to do in the UI.
3) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Thought, Action, <tool_call>.
- Be brief: one sentence for Thought, one for Action.
- Do not output anything else outside those three parts.
- If finishing, use action=terminate in the tool call.
- If the user query is to find an answer to a specific question and you have determined, provided or confirmed the answer, use action=terminate and status=success in the tool call to end the task.
"""

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
        self.generate_scheduler: RouterManager = generate_scheduler

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
            self.env_step_limiter = get_global_limiter(tag=f"{env_tag}_{self.mode}", max_concurrent_calls=self.max_env_step_concurrent)

        self.env_config["config"].update({
            "android_env_id": self.env_config["env_id"],
            "android_group_id": self.env_config["group_id"],
            "max_steps": self.env_config["max_steps"],
        })

        # Initialize reward scheduler and reward proxy BEFORE creating the environment
        # This allows passing reward components through env_config to the environment constructor
        self.reward_scheduler: Optional[RouterManager] = None
        self.reward_proxy: Optional[BaseLLMProxy] = None
        self.reward_tokenizer: Optional[PreTrainedTokenizer] = None

        # Create environment kwargs from config (convert OmegaConf to dict to avoid type errors)
        env_kwargs = dict(self.env_config['config'])

        # Try to get reward scheduler from Ray named actor
        if self.pipeline_config.reward:
            self.reward_scheduler = ray.get_actor(
                name=f"RewardScheduler-{pipeline_config.reward.name}",
                namespace=RAY_NAMESPACE
            )
            # Get reward tokenizer
            self.reward_tokenizer = default_tokenizer_provider(
                model_args=pipeline_config.reward.model_args
            )
            # Create reward proxy (without env reference since env doesn't exist yet)
            self.reward_proxy = create_llm_proxy(
                generate_scheduler=self.reward_scheduler,
                llm_proxy_config=pipeline_config.reward.llm_proxy,
                tokenizer=self.reward_tokenizer,
                env=None,
            )
            self.logger.info(f"Initialized reward proxy with scheduler: RewardScheduler-{pipeline_config.reward.name}")

            # Inject reward components into env_kwargs (not OmegaConf config)
            env_kwargs['current_env_id'] = self.env_config["env_id"]
            env_kwargs['reward_tokenizer'] = self.reward_tokenizer
            env_kwargs['reward_proxy'] = self.reward_proxy
            if self.pipeline_config.reward.generating_args:
                env_kwargs['reward_generating_args'] = self.pipeline_config.reward.generating_args.to_dict()
        with self.thread_lock, self.env_step_limiter:
            self.env = gem.make(env_id=self.env_config["env_type"], **env_kwargs)

            
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
        self.keep_last_k = self.env_config.get("keep_last_k", 30)  # 仅保留最近的k轮对话历史

        
    @property
    def task(self):
        return self.env.task


    def make_decision(self, rollout_cache: RolloutCache):
        lm_input, messages = self.format_messages(rollout_cache)
        # cache length of newly appended prompt to help to compute response_mask
        rollout_cache.history[-1]["input_ids_length"] = lm_input.batch["input_ids"].shape[1]
        rollout_cache.history[-1]["prompt_ids_length"] = rollout_cache.history[-1]["input_ids_length"] - (
            (rollout_cache.history[-2]["input_ids_length"] + rollout_cache.history[-2]["response_ids_length"])
            if len(rollout_cache.history) >= 2
            else 0
        )


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
        # cache length of response_ids to help to compute response_mask
        # eos_token should be taken into account
        rollout_cache.history[-1]["response_ids_length"] = len(lm_output.batch["responses"][0])
        self.logger.debug(
            f"env_id={self.env_config['env_id']}, global_step={self.current_step}, episode_id={self.episode_id}, turn_idx={rollout_cache.step}, "
            f"input_ids_length={rollout_cache.history[-1]['input_ids_length']}, prompt_ids_length={rollout_cache.history[-1]['prompt_ids_length']}, "
            f"response_ids_length={rollout_cache.history[-1]['response_ids_length']}"
        )

        if "infer_logprobs" in lm_output.batch:
            infer_logprobs = lm_output.batch["infer_logprobs"][0][-len(response_ids) :]
            content["infer_logprobs"] = infer_logprobs.tolist()

        content["response_ids"] = response_ids
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def format_messages(self, rollout_cache: RolloutCache) -> Tuple[DataProto, List[Dict]]:
        def build_user_content(text: str, image: str) -> dict:
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

        messages = [
            {"role": "system", "content": self.agent_system_template},
        ]
        images = []

        current_content = rollout_cache.history[-1]
        images.append([PIL.Image.fromarray(current_content["observation"], mode="RGB")])

        all_actions: List[Tuple[int, str]] = []
        for idx, content in enumerate(rollout_cache.history):            
            if "llm_response" in content:
                action_text = extract_summary(content["llm_response"].replace("<|im_end|>", ""))
                all_actions.append((idx + 1, action_text))

        recent_actions = all_actions[-self.keep_last_k :] if self.keep_last_k else all_actions

        prompt_parts: List[str] = []
        prompt_parts.append(f"The user query: {self.env.task['goal']}")
        prompt_parts.append("")

        if recent_actions:
            prompt_parts.append("Task progress (You have done the following operations on the current device):")
            start_step = len(rollout_cache.history) - len(recent_actions) + 1
            for step_idx, action_text in recent_actions:
                cleaned_text = self._clean_action_text(action_text)
                prompt_parts.append(f"Step{step_idx}: {cleaned_text}")

        prompt_parts.append("")
        prompt_parts.append("Current Screenshot: <image>")
        prompt_parts.append("")
        prompt_parts.append("Please analyze the current screenshot and history to generate the next step.")
        user_text = "\n".join(prompt_parts)

        messages.append(
            build_user_content(
                user_text,
                base64.b64encode(current_content["observation"]).decode("utf-8"),
            )
        )
        
        # if self.env_config['env_id'] == 0:  
        #     if len(recent_actions) > 3 and self.episode_id == 0:
        #         self.logger.debug("=== Full messages (WITHOUT image base64) ===")
                
        #         # 打印 system message
        #         self.logger.info(f"System: {self.agent_system_template}")
        #         self.logger.info("-" * 80)
                
        #         # 打印 user message（文本部分）
        #         self.logger.info("User Message:")
        #         self.logger.info(user_text)
        #         self.logger.info("-" * 80)          
                      
        # lm_input_texts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
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
        input_ids = lm_input.batch["input_ids"]
        attention_mask = lm_input.batch["attention_mask"]
        # Huggingface Transformers prefer position_ids to be 0-based.
        # Attn Mask: [1, 1, 1, ..., 1, 0, 0, ..., 0]
        # cumsum: [1, 2, 3, ..., n, n+1, n+1, ..., n+1]
        # cumsum - 1: [0, 1, 2, ..., n-1, n, n, ..., n]
        position_ids = attention_mask.cumsum(dim=-1) - 1
        lm_input.batch.update({
            "position_ids": position_ids,
        })
        current_cache = rollout_cache.history[-1]

        current_cache["prompt_ids"] = input_ids.tolist()[0]
        current_cache["state_hash"] = compute_object_hash(current_content["observation"])
        current_cache["messages"] = messages
        current_cache["non_tensor_batch"] = lm_input.non_tensor_batch

        return lm_input, messages

    def _clean_action_text(self, text: str) -> str:
        if not text:
            return text

        text = text.replace("\\n", " ")
        text = text.replace("\\r", " ")
        text = text.replace("\\t", " ")
        text = text.replace("\\", "")
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = text.replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()

        return text

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
            # Huggingface Transformers prefer position_ids to be 0-based.
            # Attn Mask: [1, 1, 1, ..., 1, 0, 0, ..., 0]
            # cumsum: [1, 2, 3, ..., n, n+1, n+1, ..., n+1]
            # cumsum - 1: [0, 1, 2, ..., n-1, n, n, ..., n]
            position_ids = attention_mask.cumsum(dim=-1) - 1

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
            self.logger.info(
                f"Filter group {group_id} episode {episode_id} with reward mean {reward_mean:.4f} and std {reward_std:.4f}"
            )
            return True
        return False