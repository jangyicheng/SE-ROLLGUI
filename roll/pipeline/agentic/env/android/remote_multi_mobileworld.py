import ast
from dataclasses import dataclass
from typing import List

from .remote_mobileworld import RemoteMobileEnv


@dataclass
class env_config:
    envs_num: int
    service_url: str
    console_ports: List[int] | str
    grpc_ports: List[int] | str
    adb_path: str = "/root/android-sdk/platform-tools/adb"


class RemoteMultiMobileWorldEnv(RemoteMobileEnv):
    def __init__(
        self,
        task: str | None = None,
        task_family: str = "mobile_world",
        max_steps: int = 10,
        group_seed: int = 0,
        max_image_tokens: int = 600,
        env_configs: List[env_config] | None = None,
        save_dir: str = "trajectories",
        group_size: int = 1,
        mode: str = "train",
        n_task: int = 3,
        **kwargs,
    ):
        if not env_configs:
            raise ValueError("env_configs must be provided")

        self.env_id = kwargs.get("android_env_id", 0)
        
        server_index = None
        local_env_id = None
        start = 0

        for i, cfg in enumerate(env_configs):
            cfg_envs_num = int(getattr(cfg, "envs_num", cfg["envs_num"]))
            end = start + cfg_envs_num
            if start <= self.env_id < end:
                server_index = i
                local_env_id = self.env_id - start
                break
            start = end

        if server_index is None:
            total_expected = sum(int(getattr(c, "envs_num", c["envs_num"])) for c in env_configs)
            raise ValueError(f"env_id {self.env_id} exceeds total configured envs {total_expected}")

        target_cfg = env_configs[server_index]

        def _cfg_get(cfg_obj, key):
            if isinstance(cfg_obj, dict):
                return cfg_obj[key]
            return getattr(cfg_obj, key)

        def _parse_ports(ports):
            if isinstance(ports, list):
                return ports

            if isinstance(ports, str):
                raw = ports.strip()

                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, (list, tuple)):
                        return list(parsed)
                except Exception:
                    pass

                if raw.startswith("range") or raw.startswith("list(range"):
                    return list(eval(raw))

            raise ValueError(f"Unsupported ports format: {ports}")

        c_ports = _parse_ports(_cfg_get(target_cfg, "console_ports"))
        g_ports = _parse_ports(_cfg_get(target_cfg, "grpc_ports"))

        if len(c_ports) != len(g_ports):
            raise ValueError(f"Server {server_index} console/grpc ports length mismatch")

        idx = local_env_id % len(c_ports)
        selected_console = c_ports[idx]
        selected_grpc = g_ports[idx]

        super().__init__(
            adb_path=_cfg_get(target_cfg, "adb_path"),
            console_ports=[selected_console],
            grpc_ports=[selected_grpc],
            task=task,
            task_family=task_family,
            max_steps=max_steps,
            group_seed=group_seed,
            max_image_tokens=max_image_tokens,
            service_url=_cfg_get(target_cfg, "service_url"),
            save_dir=save_dir,
            group_size=group_size,
            mode=mode,
            n_task=n_task,
            **kwargs,
        )
