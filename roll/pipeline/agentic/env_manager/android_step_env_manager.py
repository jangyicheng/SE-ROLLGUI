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
from roll.pipeline.agentic.env_manager.traj_env_manager import TrajEnvManager
from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.utils.constants import GenerateStopReason
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
        summary = "Action format can't be parsed!"
    return summary


SYSTEM_PROMPT = 'You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen\'s resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n# Response format\n\nResponse format for every step:\n1) Thought: one concise sentence explaining the next move (no multi-step reasoning).\n2) Action: a short imperative describing what to do in the UI.\n3) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.\n\nRules:\n- Output exactly in the order: Thought, Action, <tool_call>.\n- Be brief: one sentence for Thought, one for Action.\n- Do not output anything else outside those three parts.\n- If finishing, use action=terminate in the tool call.'

custom_system_prompt = """
# Skill Memory
You can get information from skill memory every turn. It will provide general steps and useful tips to complete the task.
"""


class AndroidStepEnvManager(TrajEnvManager):
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
        
        
        # self.trajectory_cache_manager = ray.get_actor("global_traj_manager")
        # self.trajectory_cache_actor   = ray.get_actor("global_traj_cache")
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

    def format_messages(self, rollout_cache: RolloutCache) -> Tuple[DataProto, List[Dict]]:
        def build_user_content(text: str, image: str, info: str | None = None) -> dict:
            if info is not None:
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

        messages = [
            {"role": "system", "content": self.agent_system_template},
        ]
        images = []
        # the message format in rollout
        previous_action = []
        for idx, content in enumerate(rollout_cache.history):
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
                info=get_skill(self.memory, self.task.skill_key, self.task._get_params())
                if self.memory is not None
                else None,
            )
        )
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
        position_ids = attention_mask.cumsum(dim=-1)
        lm_input.batch.update({
            "position_ids": position_ids,
        })
        current_cache = rollout_cache.history[-1]

        current_cache["prompt_ids"] = input_ids.tolist()[0]
        current_cache["state_hash"] = compute_object_hash(content["observation"])
        current_cache["messages"] = messages
        current_cache["non_tensor_batch"] = lm_input.non_tensor_batch
        return lm_input, messages

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        """
        Construct step-wise training samples from the collected trajectory.
        """
        if "observation" in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)
        # rollout_cache = self.get_best_trajectory(rollout=rollout_cache)
        # self.save_trajectory(rollout=rollout_cache)
        
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
        # try:
        #     self.save_trajectory(rollout=batch)
        # except Exception as e:
        #     self.logger.debug(f"save_trajectory failed: {e}")        

        return batch

    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
        it will continuously play episodes until it receives a command that data collection is complete.
        The seed needs to be reset to ensure consistency across all groups.

        Seed update logic:
        group_seed = base_seed + group_id
        episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info["seed"] + self.env_config["group_seed"]
        rollout_cache: RolloutCache = self.reset()
        start_step = self.current_step

        log_stats = {"generate_time": [], "step_time": [], "current_step": []}

        while self.running and rollout_cache is not None:
            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.make_decision(rollout_cache)
                stop_reason = lm_output.meta_info.pop("stop_reason")
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            with Timer(name="step", logger=None) as step_timer:
                if stop_reason == GenerateStopReason.FINISH:
                    rollout_cache: RolloutCache = self.step(lm_output)
            log_stats["step_time"].append(step_timer.last)

            if self.running and (rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH):
                self.logger.debug(
                    f"group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id} start_step {start_step} gen_stats: {log_stats}"
                )
                log_stats = {"generate_time": [], "step_time": [], "current_step": []}

                rollout: DataProto = self.formulate_rollouts(rollout_cache)
                traj_group_id = (
                    f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                )
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array(
                    [traj_group_id] * rollout.batch.batch_size[0], dtype=object
                )
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                ray.get(
                    self.output_queue.put.remote(self.env_config["group_id"], self.episode_id, start_step, rollout)
                )

                rollout_cache = self.reset()
                start_step = self.current_step

        ray.get(self.output_queue.put.remote(self.env_config["group_id"], self.episode_id, start_step, None))

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