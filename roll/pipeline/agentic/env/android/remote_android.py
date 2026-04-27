import base64
import requests
import numpy as np
import os
from gem import Env
import time
from roll.utils.logging import get_logger
from pathlib import Path
import json
from PIL import Image
from .tasks import TASK_LIST , TRAIN_TASK_LIST , FAIL_TASK_LIST , Information_Retrieval_TASK_LIST
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


class RemoteAndroidEnv(TaskManagerUtilsMixin, Env):
    def __init__(
        self,
        adb_path: str = "/root/android-sdk/platform-tools/adb",
        console_ports: list[int] | str = [],
        grpc_ports: list[int] | str = [],
        task: str | None = None,
        task_family: str = "android_world",
        max_steps: int = 50,
        group_seed: int = 0,
        max_image_tokens: int = 600,
        save_dir: str = "trajectories",
        group_size: int = 1,
        mode: str = "train",
        n_task: int = 20 ,
        **kwargs,
    ):
        # 基础配置解析
        self.service_url = kwargs.get("service_url", os.environ.get("ANDROID_ENV_SERVICE", "http://localhost:18000")).rstrip("/")
        self.env_id = kwargs.get("android_env_id", 0)
        # self.task_manager_url = kwargs.get("task_manager_url", "http://localhost:5001")
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

        if task == "all_task":
            task_list = TASK_LIST + Information_Retrieval_TASK_LIST if self.mode == "val" else TRAIN_TASK_LIST
            task_list = ["FilesDeleteFile", "OsmAndFavorite"] # "RecipeDeleteDuplicateRecipes","MarkorEditNote"
            # task_list = TRAIN_TASK_LIST
            # print(f"Using all tasks for mode=train, total {len(task_list)} tasks.")
        else:
            task_list = task.split(",") if task else []

        if self.mode == "train":
            n_task = int(1e9) # 无穷多任务
            
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
        if not data or not isinstance(data, dict):
            return True
        
        # 核心判断：FastAPI 的标准 500 错误
        if data.get("detail") is not None: #data.get("detail") == "Internal Server Error" or 
            return True
        
        # 保留你原来的其他失败判断
        return (
            data.get("status") == "failed" or
            (isinstance(data.get("info"), dict) and data["info"].get("error") is True)
        )

    
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
        retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError))  # 捕获网络异常和JSON解析异常
    )
    @log_time_on_error
    def call_init(self,payload):
        return requests.post(f"{self.service_url}/init", json=payload, timeout=TIMEOUT)
    
    @retry(
        stop=stop_after_attempt(3),  # 最多尝试3次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避等待
        retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError))  # 捕获网络异常和JSON解析异常
    )
    @log_time_on_error
    def call_reset(self, payload):
        resp = requests.post(f"{self.service_url}/reset", json=payload, timeout=TIMEOUT)
        return resp.json()
    
    @retry(
        stop=stop_after_attempt(3),  # 最多尝试3次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避等待
        retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError))  # 捕获网络异常和JSON解析异常
    )
    @log_time_on_error
    def call_step(self, payload):
        resp = requests.post(f"{self.service_url}/step", json=payload, timeout=TIMEOUT)
        return resp.json()

    def _init_server(self, adb_path, max_image_tokens):
        payload = {
            "console_port": self.console_port,
            "grpc_port": self.grpc_port,
            "max_steps": self.max_steps_forall,
            "adb_path": adb_path,
            "max_image_tokens": max_image_tokens
        }
        # resp = requests.post(f"{self.service_url}/init", json=payload, timeout=60)
        resp = self.call_init(payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Server init failed: {resp.text}")

    def _decode_obs(self, resp_data):
        """解析观测数据：返回渲染用的 numpy 数组"""
        np_bytes = base64.b64decode(resp_data["observation_np_b64"])
        dtype = np.dtype(resp_data["observation_dtype"])
        shape = tuple(resp_data["observation_shape"])
        obs_np = np.frombuffer(np_bytes, dtype=dtype).reshape(shape)
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
            "timestamp": time.strftime("%Y-%m-%d_%H%M%S")
        }
        
        with open(self.current_task_dir / "steps.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(step_info, ensure_ascii=False) + "\n")




    def reset(self, go_home: bool = True, seed: int = 42, target_task: str | None = None):
        super().reset(seed=seed)

        if target_task == "finish":
            while True:
                time.sleep(60)

        self.current_steps = 0
        self.start_time = time.time()
        self.assigned_task = target_task
        self._task_returned_for_current_episode = False

        payload = {
            "console_port": self.console_port,
            "go_home": go_home,
            "task": target_task,
            "task_family": self.task_family,
            "seed": seed
        }
        
        if self.env_id == 0:
            print(f"Reset seed is fixed to 42")

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
                "failure_reason": data.get("error", "service returned status=failed"),
                "recoverable": True,
                "raw": data,
            }

        try:
            self.current_obs = self._decode_obs(data)
        except Exception as e:
            logger.warning(f"Failed to decode observation: {e}, data is {data}")
            assert False
            
        try:
            self.task = data["task"]
        except Exception as e:
            logger.warning(f"Failed to get task: {e}, data is {data}")
            assert False
        self.max_steps = self.task.get("max_steps", self.max_steps)

        task_name = self.task.get("name", "unknown_task")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_task_dir = self.save_dir / f"{task_name}" / f"{timestamp}"
        self.current_task_dir.mkdir(parents=True, exist_ok=True)

        with open(self.current_task_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.task, f, ensure_ascii=False, indent=4)

        self._save_step_data(0, self.current_obs, action="RESET")
        return self.current_obs, data["info"]

    def step(self, action: str | dict):
        payload = {
            "console_port": self.console_port,
            "action": action
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
                "failure_reason": data.get("error", "service returned status=failed"),
                "recoverable": True,
                "raw": data,
            }

        obs_np = self._decode_obs(data)
        self.current_obs = obs_np
        self.current_steps += 1
        terminate = data.get("terminate", False) or (self.current_steps >= self.max_steps)

        self._save_step_data(self.current_steps, obs_np, action)

        if terminate:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            success = data["info"].get("is_success", 0)
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

        return obs_np, data["reward"], terminate, None, data["info"]

    def render(self, mode="rgb_array"):
        return self.current_obs

    def close(self):
        try:
            requests.post(f"{self.service_url}/close", json={"console_port": self.console_port})
        except:
            pass

    # ============== Exploration Methods ==============

    def explore_reset(
        self,
        go_home: bool = True,
        seed: int = 42,
        target_task: str | None = None,
        exploration_id: str | None = None,
    ):
        """Reset for exploration mode without task_manager integration.

        This method differs from reset() in that it:
        - Does not interact with task_manager
        - Does not save to the standard trajectory directory
        - Uses exploration-specific output directory
        - Can work with arbitrary task names for free exploration
        """
        super().reset(seed=seed)
        self.current_steps = 0
        self.start_time = time.time()
        self.assigned_task = target_task
        self._task_returned_for_current_episode = False

        payload = {
            "console_port": self.console_port,
            "go_home": go_home,
            "task": target_task,
            "task_family": self.task_family,
            "seed": seed,
        }

        try:
            data = self.call_reset(payload)
        except Exception as e:
            self._need_recover = True
            self._try_recover()
            return None, {
                "env_failed": True,
                "failed_stage": "explore_reset",
                "failure_reason": str(e),
                "recoverable": True,
            }

        if self._is_failed_payload(data):
            self._need_recover = True
            self._try_recover()
            return None, {
                "env_failed": True,
                "failed_stage": "explore_reset",
                "failure_reason": data.get("error", "service returned status=failed"),
                "recoverable": True,
                "raw": data,
            }

        try:
            self.current_obs = self._decode_obs(data)
        except Exception as e:
            logger.warning(f"Failed to decode observation: {e}")
            self.current_obs = None

        return self.current_obs, data.get("info", {})

    def explore_step(self, action: str | dict, disable_judge: bool = True):
        """Execute exploration step without reward evaluation.

        This method differs from step() in that it:
        - Does not evaluate task success (is_successful)
        - Does not call task_manager completion
        - Focuses on recording trajectory for curriculum generation

        Args:
            action: The action to execute
            disable_judge: If True, skip success evaluation (default True for exploration)

        Returns:
            Tuple of (observation, reward, terminate, info)
        """
        payload = {
            "console_port": self.console_port,
            "action": action,
        }

        try:
            data = self.call_step(payload)
        except Exception as e:
            self._need_recover = True
            self._try_recover()
            return self.current_obs, 0.0, True, None, {
                "env_failed": True,
                "failed_stage": "explore_step",
                "failure_reason": str(e),
                "recoverable": True,
            }

        if self._is_failed_payload(data):
            self._need_recover = True
            self._try_recover()
            return self.current_obs, 0.0, True, None, {
                "env_failed": True,
                "failed_stage": "explore_step",
                "failure_reason": data.get("error", "service returned status=failed"),
                "recoverable": True,
                "raw": data,
            }

        obs_np = self._decode_obs(data)
        self.current_obs = obs_np
        self.current_steps += 1

        # For exploration, we don't automatically terminate unless max_steps reached
        terminate = self.current_steps >= self.max_steps

        # Still track terminate if explicitly requested
        if isinstance(action, dict):
            action_name = action.get("arguments", {}).get("action", "")
        elif isinstance(action, str):
            action_name = action
        else:
            action_name = ""

        if action_name in ["terminate", "done"]:
            terminate = True

        # Get success info but don't use it for termination in exploration mode
        is_success = data.get("info", {}).get("is_success", 0)

        return obs_np, data.get("reward", 0.0), terminate, None, {
            "is_success": is_success,
            "action_name": action_name,
            "current_step": self.current_steps,
            "max_steps": self.max_steps,
            "exploration_mode": True,
        }

    def get_current_app(self) -> str | None:
        """Get the currently focused/foreground app name.

        This is used during exploration to track which apps are discovered.

        Returns:
            App name string if determinable, None otherwise
        """
        # Try to infer from current observation/task context
        if hasattr(self, "task") and self.task:
            task_name = self.task.get("name", "")
            if task_name:
                # Extract app name from task name (e.g., "ContactsAddContact" -> "Contacts")
                for prefix in ["Contacts", "Calendar", "SimpleCalendar", "SMS", "SimpleSms",
                              "Expense", "Files", "Markor", "OsmAnd", "Recipe", "Retro",
                              "VLC", "System", "Browser", "Camera", "Clock", "AudioRecorder"]:
                    if task_name.startswith(prefix):
                        return prefix
        return None

    def explore_loop(
        self,
        max_steps: int = 50,
        action_generator=None,
        save_screenshots: bool = True,
        output_dir: str = "./exploration_output",
    ) -> dict:
        """Run a complete exploration loop.

        This is a convenience method that combines explore_reset and explore_step
        for easy exploration execution.

        Args:
            max_steps: Maximum number of exploration steps
            action_generator: Callable that takes (observation, step) and returns action
            save_screenshots: Whether to save screenshots at each step
            output_dir: Directory to save exploration output

        Returns:
            Dictionary containing exploration results and trajectory
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        screenshots_dir = output_path / "screenshots"
        if save_screenshots:
            screenshots_dir.mkdir(parents=True, exist_ok=True)

        self.max_steps = max_steps

        # Reset for exploration
        obs, info = self.explore_reset(go_home=True, target_task=None)
        if obs is None:
            return {"success": False, "error": "Exploration reset failed", "trajectory": []}

        trajectory = []
        discovered_apps = set()
        discovered_actions = set()

        # Save initial screenshot
        if save_screenshots and obs is not None:
            Image.fromarray(obs).save(screenshots_dir / "step_000_init.png")

        # Get initial app
        current_app = self.get_current_app()
        if current_app:
            discovered_apps.add(current_app)

        for step_idx in range(max_steps):
            # Generate or get action
            if action_generator is not None:
                action = action_generator(obs, step_idx)
            else:
                # Default: random action
                import random
                actions = [
                    {"arguments": {"action": "click", "coordinate": [300, 400]}},
                    {"arguments": {"action": "swipe", "coordinate": [300, 600], "coordinate2": [300, 200]}},
                    {"arguments": {"action": "wait"}},
                    {"arguments": {"action": "navigate_back"}},
                ]
                action = random.choice(actions)

            # Execute step
            obs, reward, terminate, _, step_info = self.explore_step(action)

            # Record step
            if isinstance(action, dict):
                action_type = action.get("arguments", {}).get("action", "unknown")
            else:
                action_type = "text"

            discovered_actions.add(action_type)

            step_record = {
                "step": step_idx,
                "action": action,
                "action_type": action_type,
                "reward": reward,
                "terminate": terminate,
                "info": step_info,
            }

            # Save screenshot
            if save_screenshots and obs is not None:
                screenshot_path = screenshots_dir / f"step_{step_idx + 1:03d}.png"
                Image.fromarray(obs).save(screenshot_path)
                step_record["screenshot_path"] = str(screenshot_path)

            trajectory.append(step_record)

            # Track app changes
            current_app = self.get_current_app()
            if current_app:
                discovered_apps.add(current_app)

            if terminate:
                break

        return {
            "success": True,
            "total_steps": len(trajectory),
            "discovered_apps": list(discovered_apps),
            "discovered_actions": list(discovered_actions),
            "trajectory": trajectory,
            "output_dir": str(output_path),
            "screenshots_dir": str(screenshots_dir) if save_screenshots else None,
        }