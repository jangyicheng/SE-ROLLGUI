import ray
from typing import Dict, Optional, Any
import numpy as np
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

logger = get_logger()

@ray.remote
class GlobalTrajectoryCache:
    def __init__(self):
        self.cache: Dict[str, DataProto] = {}  # key: task_name, value: best DataProto

    async def save_trajectory(self, task: str, rollout: DataProto, mode: str):
        """
        保存 DataProto 到全局缓存中（仅在 train 模式）。
        只保留 episode_score > 0.5 的轨迹，且优先保留 step 更少的轨迹（更短的成功轨迹）。
        """
        if mode == "val":
            return

        try:
            episode_score = 0.0
            try:
                ep = rollout.non_tensor_batch.get("episode_scores", None)
                if ep is not None:
                    # ep 可能是 numpy 数组或列表，取第一个元素
                    episode_score = float(np.array(ep).tolist()[0])
            except Exception:
                episode_score = 0.0

            if episode_score > 0.5:
                if task not in self.cache:
                    self.cache[task] = rollout
                else:
                    current_best = self.cache[task]
                    try:
                        curr_step = int(np.array(current_best.non_tensor_batch.get("step", [0])).tolist()[-1])
                        new_step = int(np.array(rollout.non_tensor_batch.get("step", [0])).tolist()[-1])
                        if new_step < curr_step:
                            self.cache[task] = rollout
                    except Exception:
                        # 如果比较失败，保守替换
                        self.cache[task] = rollout

        except Exception as e:
            logger.info(f"save_trajectory error for task {task}: {e}")

    async def get_best_trajectory(self, task: str) -> Optional[DataProto]:
        return self.cache.get(task)

    async def reset(self):
        self.cache.clear()


@ray.remote
class GlobalTrajectoryCacheManager:
    def __init__(self):
        self.global_cache_dict: Dict[str, Any] = {}

    async def register(self, cache_name: str, cache_ref):
        self.global_cache_dict[cache_name] = cache_ref

    async def reset(self):
        refs = []
        for cache_ref in self.global_cache_dict.values():
            refs.append(cache_ref.reset.remote())
        ray.get(refs)