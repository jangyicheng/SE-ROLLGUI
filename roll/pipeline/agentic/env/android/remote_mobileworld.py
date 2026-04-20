import base64
import requests
import numpy as np
import os
from gem import Env
import time
from roll.utils.logging import get_logger
from pathlib import Path
import json
from io import BytesIO
from PIL import Image
from .tasks import MOBILEWORLD_TASK_LIST
from .task_manager_utils import TaskManagerUtilsMixin
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import functools

logger = get_logger()
TIMEOUT = 240


def log_time_on_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"[{func.__name__}] failed after {elapsed:.2f}s: {e}")
            raise

    return wrapper



class RemoteMobileEnv(TaskManagerUtilsMixin, Env):
    def __init__(
        self,
        adb_path: str = "/root/android-sdk/platform-tools/adb",
        console_ports: list[int] | str = [],
        grpc_ports: list[int] | str = [],
        task: str | None = None,
        task_family: str = "mobile_world",
        max_steps: int = 60,
        group_seed: int = 0,
        max_image_tokens: int = 600,
        envs_num: int | None = None,
        save_dir: str = "trajectories",
        group_size: int = 1,
        mode: str = "train",
        n_task: int = 3,
        **kwargs,
    ):
        # 基础配置解析
        self.service_url = kwargs.get(
            "service_url",
            os.environ.get("MOBILEWORLD_ENV_SERVICE", os.environ.get("ANDROID_ENV_SERVICE", "http://localhost:18000")),
        ).rstrip("/")
        self.env_id = kwargs.get("android_env_id", 0)
        # self.task_manager_url = kwargs.get("task_manager_url", "http://localhost:5001")
        self.success_threshold = float(kwargs.get("success_threshold", 0.99))
        self.mode = mode
        
        # 端口解析 (保持原逻辑并增加轻微健壮性)
        self.console_ports = eval(console_ports) if isinstance(console_ports, str) else console_ports
        self.grpc_ports = eval(grpc_ports) if isinstance(grpc_ports, str) else grpc_ports
        if not self.console_ports or not self.grpc_ports:
            raise ValueError("console_ports and grpc_ports must be provided")

        self.console_port = self.console_ports[self.env_id % len(self.console_ports)]
        self.grpc_port = self.grpc_ports[self.env_id % len(self.grpc_ports)]

        self.max_steps = max_steps
        self.max_steps_forall = max_steps
        self.task_family = task_family
        self.adb_path = adb_path
        self.max_image_tokens = max_image_tokens
        self.assigned_task = None
        self.scheduler_mode = self.mode

        legacy_task_manager_url = kwargs.get("task_manager_url", None)
        self.task_manager_train_url = kwargs.get(
            "task_manager_train_url",
            legacy_task_manager_url or "http://localhost:5001",
        ).rstrip("/")
        self.task_manager_eval_url = kwargs.get(
            "task_manager_eval_url",
            legacy_task_manager_url or "http://localhost:5002",
        ).rstrip("/")

        self.task_manager_url = (
            self.task_manager_train_url if self.scheduler_mode == "train" else self.task_manager_eval_url
        )

        # 任务初始化
        if task == "all_task":
            task_list = MOBILEWORLD_TASK_LIST
        else:
            task_list = task.split(",") if task else []

        if mode == "train":
            n_task = int(1e9)  # 训练模式下不限制任务数量
        if self.env_id == 0:
            logger.info(
                f"[task-manager-init] mode={self.scheduler_mode}, tasks={len(task_list)}, "
                f"n_task={n_task}"
            )
            
        shared_timestamp = kwargs.get("shared_task_timestamp", None)
        seed = int(kwargs.get("task_seed", 42))
        self.timestamp = self._initialize_active_task_manager(
            task_list=task_list,
            group_size=group_size,
            n_task=n_task,
            seed=seed,
            shared_timestamp=shared_timestamp,
        )
        
        self.save_dir = Path(save_dir) / f"{self.timestamp}"
        self.current_task_dir = None
        # 远程 Server 初始化
        self._init_server(adb_path, max_image_tokens)

        self.current_obs = None
        self.current_steps = 0
        self.start_time = None
        self.task = None
        self._task_returned_for_current_episode = False
        self._need_recover = False

    def _is_failed_payload(self, data):
        if not isinstance(data, dict):
            return True
        if data.get("status") not in (None, "ok"):
            return True
        if data.get("detail") is not None:
            return True
        if data.get("observation") is None:
            return True
        return False
    def _normalize_target_task(self, target_task):
        if not isinstance(target_task, dict):
            return target_task

        for key in ("task", "task_id", "id", "name"):
            value = target_task.get(key)
            if value is not None:
                return value
        return target_task

    def _try_recover(self):
        try:
            requests.post(f"{self.service_url}/close", json={"console_port": self.console_port}, timeout=10)
        except Exception:
            pass
        try:
            self._init_server(self.adb_path, self.max_image_tokens)
            self._need_recover = False
        except Exception as e:
            logger.warning(f"recover failed: {e}")
            self._need_recover = True

    @retry(
        stop=stop_after_attempt(3),  # 最多尝试3次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避等待
        retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError)),  # 捕获网络异常和JSON解析异常
    )
    @log_time_on_error
    def call_init(self, payload):
        resp = requests.post(f"{self.service_url}/init", json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp

    @retry(
        stop=stop_after_attempt(3),  # 最多尝试3次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避等待
        retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError)),  # 捕获网络异常和JSON解析异常
    )
    @log_time_on_error
    def call_reset(self, payload):
        resp = requests.post(f"{self.service_url}/reset", json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    @retry(
        stop=stop_after_attempt(3),  # 最多尝试3次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避等待
        retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError)),  # 捕获网络异常和JSON解析异常
    )
    @log_time_on_error
    def call_step(self, payload):
        resp = requests.post(f"{self.service_url}/step", json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def _init_server(self, adb_path, max_image_tokens):
        payload = {
            "console_port": self.console_port,
            "grpc_port": self.grpc_port,
            "max_steps": self.max_steps_forall,
            "adb_path": adb_path,
            "max_image_tokens": max_image_tokens,
            "step_wait_time": 5,
        }
        resp = self.call_init(payload)
        # print(f"payload: {payload}")
        if resp.status_code != 200:
            raise RuntimeError(f"Server init failed: {resp.text}")

    def _decode_obs(self, observation_data):
        """解析 MobileWorld 观测：screenshot_b64(PNG) -> numpy 数组。"""
        if not isinstance(observation_data, dict):
            raise ValueError("invalid observation payload")

        screenshot_b64 = observation_data.get("screenshot_b64")
        if not screenshot_b64:
            raise ValueError("missing screenshot_b64 in observation")

        img_bytes = base64.b64decode(screenshot_b64)
        with Image.open(BytesIO(img_bytes)) as img:
            obs_np = np.array(img.convert("RGB"))
        return obs_np

    def _save_step_data(self, step_idx, obs_np, action):
        """存储单步轨迹：图片 + JSON 元数据"""
        if not self.current_task_dir:
            return

        # 1. 存储图片
        img_path = self.current_task_dir / f"step_{step_idx:03d}.png"
        Image.fromarray(obs_np).save(img_path)

        # 2. 存储该步的详细信息
        step_info = {
            "step": step_idx,
            "action": action,
            "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        }

        with open(self.current_task_dir / "steps.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(step_info, ensure_ascii=False) + "\n")

    def reset(self, go_home: bool = True, seed: int | None = None, target_task: str | None = None):
        super().reset(seed=seed)

        if target_task == "finish":
            while True:
                time.sleep(60)

        self.current_steps = 0
        self.start_time = time.time()
        self.assigned_task = target_task
        self._task_returned_for_current_episode = False
        normalized_task = self._normalize_target_task(target_task)

        payload = {
            "console_port": self.console_port,
            "go_home": go_home,
            "task": normalized_task,
            "task_family": self.task_family,
        }

        try:
            data = self.call_reset(payload)
        except Exception as e:
            self._return_task_once(reason=f"reset_exception:{e}")
            self._need_recover = True
            self._try_recover()
            return None, {
                "env_failed": True,
                "failed_stage": "reset",
                "failure_reason": str(e),
                "recoverable": True,
            }

        if self._is_failed_payload(data):
            self._return_task_once(reason=f"reset_failed:{data.get('error', 'unknown')}")
            self._need_recover = True
            self._try_recover()
            return None, {
                "env_failed": True,
                "failed_stage": "reset",
                "failure_reason": data.get("error") or data.get("detail", "service returned status=failed"),
                "recoverable": True,
                "raw": data,
            }

        observation = data.get("observation")
        self.current_obs = self._decode_obs(observation)
        self.task = data.get("task", {})
        # self.max_steps = self.task.get("max_steps", self.max_steps)

        task_name = self.task.get("name", "unknown_task")
        timestamp = time.strftime("%Y%m%d_%H%M")
        self.current_task_dir = self.save_dir / f"{task_name}" / f"{timestamp}"
        self.current_task_dir.mkdir(parents=True, exist_ok=True)

        with open(self.current_task_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.task, f, ensure_ascii=False, indent=4)

        self._save_step_data(0, self.current_obs, action="RESET")

        info = {
            "task_name": self.task.get("name"),
            "goal": self.task.get("goal"),
            "task_params": self.task.get("params"),
            "initialized": self.task.get("initialized", True),
        }
        if isinstance(observation, dict):
            if observation.get("ask_user_response") is not None:
                info["ask_user_response"] = observation.get("ask_user_response")
            if observation.get("tool_call") is not None:
                info["tool_call"] = observation.get("tool_call")

        return self.current_obs, info

    def step(self, action: str | dict):
        payload = {
            "console_port": self.console_port,
            "action": action,
        }

        try:
            data = self.call_step(payload)
        except Exception as e:
            self._return_task_once(reason=f"step_exception:{e}")
            self._need_recover = True
            self._try_recover()
            return self.current_obs, 0.0, True, None, {
                "env_failed": True,
                "failed_stage": "step",
                "failure_reason": str(e),
                "recoverable": True,
            }

        if self._is_failed_payload(data):
            self._return_task_once(reason=f"step_failed:{data.get('error', 'unknown')}")
            self._need_recover = True
            self._try_recover()
            return self.current_obs, 0.0, True, None, {
                "env_failed": True,
                "failed_stage": "step",
                "failure_reason": data.get("error") or data.get("detail", "service returned status=failed"),
                "recoverable": True,
                "raw": data,
            }

        observation = data.get("observation")
        obs_np = self._decode_obs(observation)
        self.current_obs = obs_np
        self.current_steps += 1

        score = float(data.get("score", 0.0) or 0.0)
        mobile_done = bool(data.get("done", False))
        terminate = mobile_done or (self.current_steps >= self.max_steps)

        info = {
            "reason": data.get("reason"),
            "score": score,
            "is_success": 1.0 if score >= self.success_threshold else 0.0,
            "task": data.get("task"),
            "action_type": data.get("action_type"),
            "done": mobile_done,
        }
        if data.get("execution_time") is not None:
            info["execution_time"] = data.get("execution_time")
        if isinstance(observation, dict):
            if observation.get("ask_user_response") is not None:
                info["ask_user_response"] = observation.get("ask_user_response")
            if observation.get("tool_call") is not None:
                info["tool_call"] = observation.get("tool_call")

        self._save_step_data(self.current_steps, obs_np, action)

        if terminate:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            success = info.get("is_success", 0)
            completion_payload = {
                "task": self.assigned_task,
                "success": success > 0.5,
                "steps": self.current_steps,
                "time": float(elapsed_time),
            }
            requests.post(f"{self.task_manager_url}/complete_task", json=completion_payload)

            if self.current_steps >= self.max_steps:
                completion_payload["termination_reason"] = "max_steps"
            elif isinstance(action, dict) and action.get("action") == "terminate":
                completion_payload["termination_reason"] = "active terminate"
            else:
                completion_payload["termination_reason"] = "success"

            with open(self.current_task_dir / "result.json", "w") as f:
                json.dump(completion_payload, f, indent=4)

        return obs_np, score, terminate, None, info

    def render(self, mode="rgb_array"):
        return self.current_obs

    def close(self):
        try:
            requests.post(f"{self.service_url}/close", json={"console_port": self.console_port})
        except Exception:
            pass
