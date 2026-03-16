import os
import ast
import time
import json
import requests
from typing import List
from dataclasses import dataclass
from pathlib import Path
from .remote_android import RemoteAndroidEnv


@dataclass
class env_config:
    envs_num: int
    service_url: str
    console_ports: List[int] | str
    grpc_ports: List[int] | str
    adb_path: str = "/root/android-sdk/platform-tools/adb" # 默认值

class RemoteMultiAndroidEnv(RemoteAndroidEnv):
    def __init__(
        self,
        task: str | None = None,
        task_family: str = "android_world",
        max_steps: int = 10,
        group_seed: int = 0,
        max_image_tokens: int = 600,
        env_configs: List[env_config] = None,
        save_dir: str = "trajectories",
        group_size: int = 1,
        **kwargs,
    ):
        if not env_configs:
            raise ValueError("env_configs must be provided")

        self.env_id = kwargs.get("android_env_id", 0)
        
        # --------------------------------------------------
        # 1. 计算当前 env_id 属于哪个 Server 节点
        # --------------------------------------------------
        server_index = None
        local_env_id = None
        start = 0

        for i, cfg in enumerate(env_configs):
            end = start + cfg.envs_num
            if start <= self.env_id < end:
                server_index = i
                local_env_id = self.env_id - start
                break
            start = end

        if server_index is None:
            total_expected = sum(c.envs_num for c in env_configs)
            raise ValueError(f"env_id {self.env_id} exceeds total configured envs {total_expected}")

        target_cfg = env_configs[server_index]

        # --------------------------------------------------
        # 2. 解析该 Server 节点的端口列表
        # --------------------------------------------------
        def _parse_ports(ports):
            if isinstance(ports, list):
                return ports

            if isinstance(ports, str):
                ports = ports.strip()

                try:
                    parsed = ast.literal_eval(ports)
                    if isinstance(parsed, (list, tuple)):
                        return list(parsed)
                except Exception:
                    pass

                if ports.startswith("range"):
                    return list(eval(ports))

                if ports.startswith("list(range"):
                    return list(eval(ports))

            raise ValueError(f"Unsupported ports format: {ports}")
        c_ports = _parse_ports(target_cfg.console_ports)
        g_ports = _parse_ports(target_cfg.grpc_ports)

        if len(c_ports) != len(g_ports):
            raise ValueError(f"Server {server_index} console/grpc ports length mismatch")

        # 映射到具体的端口
        # 使用取模以防 envs_num 大于实际端口数（实现复用）或者直接索引
        idx = local_env_id % len(c_ports)
        selected_console = c_ports[idx]
        selected_grpc = g_ports[idx]

        # --------------------------------------------------
        # 3. 调用父类初始化 (父类负责轨迹存储逻辑)
        # --------------------------------------------------
        # 注意：这里 console_ports 传的是单元素列表，因为子类已经算好了确切端口
        super().__init__(
            adb_path=target_cfg.adb_path,  # 从 config 传入
            console_ports=[selected_console],
            grpc_ports=[selected_grpc],
            task=task,
            task_family=task_family,
            max_steps=max_steps,
            group_seed=group_seed,
            max_image_tokens=max_image_tokens,
            service_url=target_cfg.service_url,
            save_dir=save_dir,
            group_size=group_size,
            **kwargs,
        )