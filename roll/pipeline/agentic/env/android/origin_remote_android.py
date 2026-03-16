import base64
import requests
import numpy as np
import os
from gem import Env
import time
from roll.utils.logging import get_logger


logger = get_logger()

TASK_LIST = [
    "AudioRecorderRecordAudio",
    "BrowserDraw",
    # "CameraTakePhoto",    
    # "ClockStopWatchPausedVerify",
    "ContactsAddContact",
    "SimpleCalendarAddOneEvent",
    "OsmAndFavorite",
    # "ExpenseAddMultiple",
    "FilesDeleteFile",    
    # "MarkorCreateFolder",
    "RecipeAddMultipleRecipes",
    # "SystemBluetoothTurnOff",   
    "VlcCreatePlaylist", 
]



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
        envs_num:int | None = None,
        **kwargs,
    ):
        # 获取 Server 地址，优先从 kwargs 获取，否则环境变量，否则默认
        self.service_url = kwargs.get("service_url", os.environ.get("ANDROID_ENV_SERVICE", "http://localhost:18000")).rstrip("/")
        self.env_id = kwargs.get("android_env_id", 0)
        self.task_manager_url = kwargs.get("task_manager_url", "http://localhost:5001")  
        # --- 保持原有接口的解析逻辑 ---
        # 支持字符串形式的列表 (兼容原有 eval 写法) 或 直接列表
        c_ports_list = eval(console_ports) if isinstance(console_ports, str) else console_ports
        g_ports_list = eval(grpc_ports) if isinstance(grpc_ports, str) else grpc_ports
        envs_num = eval(envs_num)
        
        
        if not c_ports_list or not g_ports_list:
             raise ValueError("console_ports and grpc_ports must be provided")

        # 唯一确定当前 Client 对应的 Server 端虚拟机端口
        # 通过取模或直接索引来确定,环境数量应该与虚拟机数量相同，即与端口数量相同
        self.console_port = c_ports_list[self.env_id % len(c_ports_list)]
        self.grpc_port = g_ports_list[self.env_id % len(g_ports_list)]
        self.group_seed = group_seed
        self.max_steps = max_steps        


        # 解析任务 (处理 task="task1,task2" 的情况)
        if task == "all_task":
            task_list = TASK_LIST[:envs_num]
        else:
            task_list = task.split(",") if task else []

        # 在task_list解析后添加
        payload = {"task_list": task_list}
        requests.post(f"{self.task_manager_url}/initialize", json=payload)              
        
        logger.info(f"Group-{kwargs.get('android_group_id', 0)}[Env-{self.env_id}] Remote Init on port {self.console_port}")
        
        self.task_family = task_family
        self.task = None
        
        # --- 远程初始化 ---
        payload = {
            "console_port": self.console_port,
            "grpc_port": self.grpc_port,
            "max_steps": max_steps,
            "adb_path": adb_path,
            "max_image_tokens": max_image_tokens
        }
        
        try:
            resp = requests.post(f"{self.service_url}/init", json=payload, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"Server init failed: {resp.text}")
            resp_data = resp.json()
            
        except Exception as e:
            raise RuntimeError(f"Could not connect to AndroidEnv Server at {self.service_url}: {e}")

        self.current_obs = None
        self.name = f"remote-emulator-{self.console_port}"
        self.current_steps = 0
        self.start_time = None

    def _decode_obs(self, resp_data):
        """解码观测数据"""
        # Server 已经做好了 Base64 编码
        np_bytes = base64.b64decode(resp_data["observation_np_b64"])
        dtype = np.dtype(resp_data["observation_dtype"])
        shape = tuple(resp_data["observation_shape"])
        obs_np = np.frombuffer(np_bytes, dtype=dtype).reshape(shape)
        
        # 返回 VLM 需要的 base64 字符串 和 渲染需要的 numpy
        return resp_data["observation_b64"], obs_np

    def step(self, action: str | dict):
        payload = {
            "console_port": self.console_port,
            "action": action
        }
        
        resp = requests.post(f"{self.service_url}/step", json=payload)
        if resp.status_code != 200:
            # 简单容错
            return None, 0.0, True, None, {"error": resp.text}
            
        data = resp.json()
        obs_b64, obs_np = self._decode_obs(data)
        self.current_obs = obs_np # 保存 numpy 用于 render
        self.current_steps += 1
        
        if self.current_steps >= self.max_steps:
            data["terminate"] = True
        
        
        if data["terminate"]:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            success = data["info"]["is_success"] 
            payload = {
                "task": self.task["name"],
                "success": success > 0.5,
                "steps": self.current_steps,
                "time": float(elapsed_time)
            }
            requests.post(f"{self.task_manager_url}/complete_task", json=payload)
        
        return obs_np , data["reward"], data["terminate"], None, data["info"]

    def reset(self, go_home: bool = True, seed: int | None = None) -> tuple[np.ndarray, dict]:
        
        resp = requests.get(f"{self.task_manager_url}/get_task")
        if resp.status_code != 200:
            raise ValueError("Failed to get task from TaskManager")
        target_task = resp.json()["task"]
        
        if target_task == "finish":
            while True:
                time.sleep(60)
        
        self.current_steps = 0
        self.start_time = time.time()
        
        payload = {
            "console_port": self.console_port,
            "go_home": go_home,
            "task": target_task,
            "task_family": self.task_family
        }
        
        resp = requests.post(f"{self.service_url}/reset", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Reset failed: {resp.text}")
            
        data = resp.json()
        obs_b64, obs_np = self._decode_obs(data)
        self.current_obs = obs_np
        self.task = data["task"]
        self.max_steps = self.task["max_steps"]
        
        # 注意：AndroidWorld 原版 reset 返回 (obs, info)
        # 这里 obs 返回 base64 字符串给 Agent 使用
        return obs_np , data["info"]

    def render(self, mode="rgb_array") -> np.ndarray:
        if self.current_obs is None:
             # 如果未开始，尝试 reset 获取初始帧
             _, _ = self.reset()
        return self.current_obs

    def close(self):
        try:
            requests.post(f"{self.service_url}/close", json={"console_port": self.console_port})
        except:
            pass
    
    
    