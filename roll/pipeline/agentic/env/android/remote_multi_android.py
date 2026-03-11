import os
import ast
from typing import List
from dataclasses import dataclass

@dataclass
class env_config:
    envs_num: int
    service_url: str
    console_ports: List[int] | str
    grpc_ports: List[int] | str

# env_configs = [
#     env_config(
#         envs_num=256,
#         service_url="http://server1:18000",
#         console_ports=list(range(5554, 6066, 2)),
#         grpc_ports=list(range(8554, 9066, 2)),
#     ),
#     env_config(
#         envs_num=256,
#         service_url="http://server2:18000",
#         console_ports=list(range(5554, 6066, 2)),
#         grpc_ports=list(range(8554, 9066, 2)),
#     ),
# ]
class RemoteMultiAndroidEnv(RemoteAndroidEnv):

    def __init__(
        self,
        adb_path: str = "/root/android-sdk/platform-tools/adb",
        task: str | None = None,
        task_family: str = "android_world",
        max_steps: int = 10,
        group_seed: int = 0,
        max_image_tokens: int = 600,
        env_configs: List[env_config] = None,
        **kwargs,
    ):

        if env_configs is None or len(env_configs) == 0:
            raise ValueError("env_configs must be provided")

        self.env_configs = env_configs
        self.env_id = kwargs.get("android_env_id", 0)

        # --------------------------------------------------
        # 1 计算每个 server 的环境范围
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
            raise ValueError(
                f"env_id {self.env_id} exceeds total envs "
                f"{sum(c.envs_num for c in env_configs)}"
            )

        cfg = env_configs[server_index]

        # --------------------------------------------------
        # 2 解析端口
        # --------------------------------------------------

        console_ports = (
            ast.literal_eval(cfg.console_ports)
            if isinstance(cfg.console_ports, str)
            else cfg.console_ports
        )

        grpc_ports = (
            ast.literal_eval(cfg.grpc_ports)
            if isinstance(cfg.grpc_ports, str)
            else cfg.grpc_ports
        )

        if not console_ports or not grpc_ports:
            raise ValueError("console_ports and grpc_ports must be provided")

        if len(console_ports) != len(grpc_ports):
            raise ValueError("console_ports and grpc_ports length mismatch")

        if local_env_id >= len(console_ports):
            raise ValueError(
                f"local_env_id {local_env_id} exceeds available ports "
                f"{len(console_ports)}"
            )

        # --------------------------------------------------
        # 3 选择当前 env 的端口
        # --------------------------------------------------

        console_port = console_ports[local_env_id]
        grpc_port = grpc_ports[local_env_id]

        service_url = cfg.service_url.rstrip("/")

        # --------------------------------------------------
        # 4 调用父类初始化
        # --------------------------------------------------

        super().__init__(
            adb_path=adb_path,
            console_ports=[console_port],
            grpc_ports=[grpc_port],
            task=task,
            task_family=task_family,
            max_steps=max_steps,
            group_seed=group_seed,
            max_image_tokens=max_image_tokens,
            service_url=service_url,
            **kwargs,
        )