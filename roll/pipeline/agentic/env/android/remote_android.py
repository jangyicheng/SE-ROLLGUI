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
from datetime import datetime
from .tasks import TASK_LIST

logger = get_logger()


class RemoteAndroidEnv(Env):
    def __init__(
        self,
        adb_path: str = "/root/android-sdk/platform-tools/adb",
        console_ports: list[int] | str = [],
        grpc_ports: list[int] | str = [],
        task: str | None = None,
        task_family: str = "android_world",
        max_steps: int = 10,
        group_seed: int = 0,
        max_image_tokens: int = 600,
        envs_num: int | None = None,
        save_dir: str = "trajectories",
        group_size: int = 1,
        **kwargs,
    ):
        # 基础配置解析
        self.service_url = kwargs.get("service_url", os.environ.get("ANDROID_ENV_SERVICE", "http://localhost:18000")).rstrip("/")
        self.env_id = kwargs.get("android_env_id", 0)
        self.task_manager_url = kwargs.get("task_manager_url", "http://localhost:5001")
        
        # 端口解析 (保持原逻辑并增加轻微健壮性)
        self.console_ports = eval(console_ports) if isinstance(console_ports, str) else console_ports
        self.grpc_ports = eval(grpc_ports) if isinstance(grpc_ports, str) else grpc_ports
        if not self.console_ports or not self.grpc_ports:
            raise ValueError("console_ports and grpc_ports must be provided")

        self.console_port = self.console_ports[self.env_id % len(self.console_ports)]
        self.grpc_port = self.grpc_ports[self.env_id % len(self.grpc_ports)]
        
        self.max_steps = max_steps
        self.task_family = task_family
        time_str = datetime.now().strftime("%Y-%m-%d_%H%M")
        self.save_dir = Path(save_dir) / time_str
        self.current_task_dir = None
        
        # 任务初始化
        if task == "all_task":
            num = eval(envs_num) if isinstance(envs_num, str) else envs_num
            task_list = TASK_LIST[:num]
        else:
            task_list = task.split(",") if task else []

        requests.post(f"{self.task_manager_url}/initialize", json={"task_list": task_list , "group_size": group_size})
        
        # 远程 Server 初始化
        self._init_server(adb_path, max_image_tokens)

        self.current_obs = None
        self.current_steps = 0
        self.start_time = None
        self.task = None
        

    def _init_server(self, adb_path, max_image_tokens):
        payload = {
            "console_port": self.console_port,
            "grpc_port": self.grpc_port,
            "max_steps": self.max_steps,
            "adb_path": adb_path,
            "max_image_tokens": max_image_tokens
        }
        resp = requests.post(f"{self.service_url}/init", json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Server init failed: {resp.text}")

    def _decode_obs(self, resp_data):
        """解析观测数据：返回渲染用的 numpy 数组"""
        np_bytes = base64.b64decode(resp_data["observation_np_b64"])
        dtype = np.dtype(resp_data["observation_dtype"])
        shape = tuple(resp_data["observation_shape"])
        obs_np = np.frombuffer(np_bytes, dtype=dtype).reshape(shape)
        return obs_np

    def _save_step_data(self, step_idx, obs_np, action, response_text=None):
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
            "ai_response": response_text,
            "timestamp": time.time()
        }
        
        with open(self.current_task_dir / "steps.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(step_info, ensure_ascii=False) + "\n")

    def reset(self, go_home: bool = True, seed: int | None = None , target_task: str | None = None):
        super().reset(seed=seed)
        # 从 TaskManager 获取新任务
        # resp = requests.get(f"{self.task_manager_url}/get_task")
        # target_task = resp.json().get("task")
        
        if target_task == "finish":
            while True: time.sleep(60)

        # 环境状态重置
        self.current_steps = 0
        self.start_time = time.time()
        
        payload = {
            "console_port": self.console_port,
            "go_home": go_home,
            "task": target_task,
            "task_family": self.task_family
        }
        
        resp = requests.post(f"{self.service_url}/reset", json=payload)
        data = resp.json()
        
        self.current_obs = self._decode_obs(data)
        self.task = data["task"] # 包含任务名、目标等
        self.max_steps = self.task.get("max_steps", self.max_steps)

        # --- 初始化轨迹目录 ---
        task_name = self.task.get("name", "unknown_task")
        timestamp = time.strftime("%Y%m%d_%H%M")
        self.current_task_dir = self.save_dir / f"{task_name}" / f"{timestamp}"
        self.current_task_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储任务元信息
        with open(self.current_task_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.task, f, ensure_ascii=False, indent=4)

        # 记录初始状态
        self._save_step_data(0, self.current_obs, action="RESET")
        
        return self.current_obs, data["info"]

    def step(self, action: str | dict, ai_response: str = None):
        """
        action: 执行的动作
        ai_response: 可选，AI 生成的原始推理文本
        """
        payload = {
            "console_port": self.console_port,
            "action": action
        }
        
        resp = requests.post(f"{self.service_url}/step", json=payload)
        if resp.status_code != 200:
            return None, 0.0, True, None, {"error": resp.text}
            
        data = resp.json()
        obs_np = self._decode_obs(data)
        self.current_obs = obs_np
        self.current_steps += 1
        
        terminate = data.get("terminate", False) or (self.current_steps >= self.max_steps)
        
        # --- 存储轨迹 ---
        self._save_step_data(self.current_steps, obs_np, action, ai_response)
        
        if terminate:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            success = data["info"].get("is_success", 0)
            
            # 通知 TaskManager
            completion_payload = {
                "task": self.task["name"],
                "success": success > 0.5,
                "steps": self.current_steps,
                "time": float(elapsed_time)
            }
            requests.post(f"{self.task_manager_url}/complete_task", json=completion_payload)
            
            # 存入最终状态到 meta
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