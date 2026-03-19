import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray._private import profiling
from tqdm import tqdm

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.utils.functionals import append_to_dict, GenerateRequestType
from roll.utils.import_utils import safe_import_class
from roll.utils.logging import get_logger
from roll.datasets.global_trajectory_cache import GlobalTrajectoryCacheManager,GlobalTrajectoryCache


logger = get_logger()

@dataclass
class GroupData:
    group_id: int
    episode_id: int
    create_step: int
    task: str = ""
    rollouts: List[DataProto] = field(default_factory=list)
    running_rollouts: int = 0


# 新增：放在 GroupData 后
class GroupQueue:
    def __init__(
        self,
        group_id: int,
        task: str,
        progress_bar: tqdm,
        group_size: int,
        group_size_redundancy: int,
        max_traj_per_env: Optional[int],
        async_generation_ratio: int,
        group_filter,
    ):
        self.group_id = group_id
        self.task = task
        self.progress_bar = progress_bar
        self.group_size = group_size
        self.group_size_redundancy = group_size_redundancy
        self.max_traj_per_env = max_traj_per_env
        self.async_generation_ratio = async_generation_ratio
        self.group_filter = group_filter
        self.group_filter_count = 0

        self.current_step = None
        self.next_episode_id = 0
        self.groups: Dict[int, GroupData] = {}

        self.progress = asyncio.Event()
        self.complete = asyncio.Event()
        self.quit = False

    def clear(self):
        self.current_step = None
        self.next_episode_id = 0
        self.groups.clear()
        self.progress = asyncio.Event()
        self.complete = asyncio.Event()

    def shutdown(self):
        self.quit = True
        self.groups.clear()
        self.progress.set()
        self.complete.set()

    def advance_group(self, create_step: int):
        assert not self.quit
        self.groups[self.next_episode_id] = GroupData(
            group_id=self.group_id,
            episode_id=self.next_episode_id,
            create_step=create_step,
            task=self.task,
        )
        self.next_episode_id += 1

    def _advance_step(self, create_step: int):
        if self.max_traj_per_env is None:
            return
        for _ in range(self.max_traj_per_env):
            self.advance_group(create_step)

    def advance_step(self, step: int):
        if self.current_step is None:
            for _ in range(self.async_generation_ratio):
                self._advance_step(step)
        else:
            expired = []
            for episode_id, group in self.groups.items():
                if step - group.create_step > self.async_generation_ratio:
                    expired.append(episode_id)
            for episode_id in expired:
                self.groups.pop(episode_id, None)

        self.current_step = step
        self._advance_step(step)
        self.progress.set()

    async def get_episode_id(self) -> Optional[int]:
        while not self.quit:
            for episode_id, group in self.groups.items():
                if group.running_rollouts < self.group_size + self.group_size_redundancy:
                    group.running_rollouts += 1
                    return episode_id

            if self.max_traj_per_env is None:
                while self.current_step is None and not self.quit:
                    self.progress.clear()
                    await self.progress.wait()
                if self.quit:
                    return None
                self.advance_group(self.current_step)
                continue
            else:
                self.progress.clear()
                await self.progress.wait()
        return None

    def put(self, episode_id: int, start_step: int, rollout: DataProto):
        if episode_id not in self.groups:
            return
        group = self.groups[episode_id]
        assert start_step >= group.create_step, f"{start_step=} {group.create_step=}"
        group.rollouts.append(rollout)

        if len(group.rollouts) == self.group_size:
            if all(r is None for r in group.rollouts):
                self.complete.set()
            elif self.group_filter.filter(group_id=self.group_id, episode_id=episode_id, group=group.rollouts):
                logger.info(f"filter rollout group {group.group_id} episode {group.episode_id}")
                self.group_filter_count += 1
                self.groups.pop(episode_id, None)
                if self.current_step is not None:
                    self.advance_group(create_step=self.current_step)
            else:
                self.complete.set()
                self.progress_bar.update(self.group_size)

    async def get(self) -> GroupData:
        while True:
            while not self.groups:
                self.complete.clear()
                await self.complete.wait()
            episode_id = next(iter(self.groups))
            group = self.groups[episode_id]
            if len(group.rollouts) >= self.group_size:
                self.groups.pop(episode_id, None)
                return group
            self.complete.clear()
            await self.complete.wait()
            

@ray.remote
class GroupQueueManager:
    def __init__(self, config, env_manager_config: EnvManagerConfig, mode):
        self.mode = mode
        self.env_manager_config = env_manager_config
        self.group_size = env_manager_config.group_size
        self.group_size_redundancy = env_manager_config.group_size_redundancy
        self.progress_bar = tqdm(desc=f"{self.mode} rollout progress(trajectory)", mininterval=1)

        group_filter_cls = safe_import_class(env_manager_config.group_filter_cls)
        assert group_filter_cls
        self.group_filter = group_filter_cls(config, env_manager_config, mode)

        if self.mode == "train":
            self.async_generation_ratio = config.async_generation_ratio
            self.max_traj_per_env = env_manager_config.max_traj_per_env if config.rollout_batch_size > 0 else None
        else:
            self.async_generation_ratio = 0
            self.max_traj_per_env = env_manager_config.max_traj_per_env if config.val_batch_size > 0 else None

        self.task_manager_url = getattr(env_manager_config, "task_manager_url", "http://localhost:5001")

        self.next_group_id = 0
        self.task_to_group_id: Dict[str, int] = {}
        self.group_queues: Dict[int, GroupQueue] = {}

        self.pending_gets = set()
        self.rollout_complete = {}

        self.current_step = None
        self.progress = asyncio.Event()
        self.quit = False


    def _get_or_create_group_queue(self, task: str) -> Tuple[int, GroupQueue]:
        if task in self.task_to_group_id:
            gid = self.task_to_group_id[task]
            return gid, self.group_queues[gid]

        gid = self.next_group_id
        self.next_group_id += 1
        self.task_to_group_id[task] = gid

        q = GroupQueue(
            group_id=gid,
            task=task,
            progress_bar=self.progress_bar,
            group_size=self.group_size,
            group_size_redundancy=self.group_size_redundancy,
            max_traj_per_env=self.max_traj_per_env,
            async_generation_ratio=self.async_generation_ratio,
            group_filter=self.group_filter,
        )
        self.group_queues[gid] = q
        if self.current_step is not None:
            q.advance_step(self.current_step)
        return gid, q

    def collect_metrics(self):
        cnt = 0
        for q in self.group_queues.values():
            cnt += q.group_filter_count
            q.group_filter_count = 0
        return {"scheduler/group_filter_count": cnt}

    def clear(self):
        self.rollout_complete = {}
        for q in self.group_queues.values():
            q.clear()

        self.current_step = None
        self.progress = asyncio.Event()

    def advance_step(self, step):
        self.current_step = step
        for q in self.group_queues.values():
            q.advance_step(step)
        self.progress.set()

    async def get_episode_id(self, group_id=None) -> Optional[Dict]:
        while self.current_step is None and not self.quit:
            self.progress.clear()
            await self.progress.wait()

        if self.quit:
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{self.task_manager_url}/get_task")
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error(f"Failed to get task from TaskManager: {e}")
            return None

        task = data["task"]
        if task == "finish":
            return None

        gid, q = self._get_or_create_group_queue(task)
        episode_id = await q.get_episode_id()
        if episode_id is None:
            return None
        return {"group_id": gid, "episode_id": episode_id, "task": task}

    def shutdown(self):
        self.quit = True
        for q in self.group_queues.values():
            q.shutdown()
        self.progress.set()

    def put(self, group_id, episode_id, start_step, rollout: DataProto):
        if rollout is None:
            return
        q = self.group_queues.get(group_id)
        if q is None:
            return
        q.put(episode_id, start_step, rollout)

    async def get_batch(self, batch_size, current_step) -> List[DataProto]:
        """
        返回已完成的 rollouts，按 group 收集。
        batch_size < 0 表示收集所有直到所有环境退出。
        """
        ret: List[DataProto] = []

        while batch_size < 0 or len(ret) < batch_size:
            # 所有环境都退出时，允许尽快结束
            if self.all_done:
                has_active_group = any(len(q.groups) > 0 for q in self.group_queues.values())
                if not has_active_group:
                    break

            # 直接等待“任意一个 group queue 完成一个 group”
            tasks = {
                asyncio.create_task(q.get(), name=str(gid))
                for gid, q in self.group_queues.items()
            }

            if not tasks:
                if self.all_done:
                    break
                await asyncio.sleep(0.05)
                continue

            done, pending = await asyncio.wait(
                tasks,
                # timeout=0.2,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # 本轮其余未完成任务取消，避免悬挂
            for p in pending:
                p.cancel()

            if not done:
                # 无已完成 group：对 batch_size<0 直接返回已收集内容；
                # 对 batch_size>0 继续等下一轮
                if batch_size < 0:
                    break
                continue

            for d in done:
                group = await d
                group_rollout = [r for r in group.rollouts if r is not None]
                if len(group_rollout) == 0:
                    continue

                # 过时数据过滤：基于 async_generation_ratio
                if current_step - group.create_step > self.async_generation_ratio:
                    continue

                if batch_size > 0:
                    remain = batch_size - len(ret)
                    if remain <= 0:
                        break
                    ret.extend(group_rollout[:remain])
                else:
                    ret.extend(group_rollout)

                if batch_size > 0 and len(ret) >= batch_size:
                    break

        get_batch_return_start_time = time.time()
        for d in ret:
            d.meta_info["get_batch_return_start_time"] = get_batch_return_start_time
        return ret


class RolloutScheduler:
    """
    Usage:
        actor_infer
        train_rollout_scheduler = RolloutScheduler(actor_infer)
        val_rollout_scheduler = RolloutScheduler(actor_infer)
        while True:
            ray.get(train_rollout_scheduler.suspend.remote())
            model_update()
            if val:
                ray.get(val_rollout_scheduler.get_batch.remote())
            ray.get(train_rollout_scheduler.get_batch.remote())
            rollout()
        ray.get(train_rollout_scheduler.shutdown.remote())
    """
    def __init__(self, config, env_manager_config: EnvManagerConfig, resource_manager, infer_cluster, mode, collator=None):
        self.config = config
        self.env_manager_config = env_manager_config
        self.resource_manager = resource_manager
        self.infer_cluster = infer_cluster
        self.mode = mode

        env_num = self.env_manager_config.world_size * self.env_manager_config.max_env_num_per_worker

        self.env_output_queue = GroupQueueManager.options(
            max_concurrency = env_num + 1
        ).remote(
            self.config,
            self.env_manager_config,
            mode
        )

        self.generate_scheduler = RequestScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
                max_concurrency = env_num + 1
            ).remote(infer_cluster=self.infer_cluster, pipeline_config=config)

        self.es_manager: Any = Cluster(
            name=self.env_manager_config.name,
            worker_cls=self.env_manager_config.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.env_manager_config,
        )
        self.es_manager.initialize(
            pipeline_config=self.config,
            generate_scheduler=self.generate_scheduler,
            output_queue=self.env_output_queue,
            collator=collator,
            mode=self.mode,
        )

        self.rollout_task = None
        self.manager = GlobalTrajectoryCacheManager.options(name="global_traj_manager").remote()
        self.cache = GlobalTrajectoryCache.options(name="global_traj_cache").remote()


    async def shutdown(self):
        if self.rollout_task is None:
            return
        await asyncio.gather(*self.es_manager.stop(blocking=False))
        await self.env_output_queue.shutdown.remote()
        await self.generate_scheduler.abort_request.remote()
        await self.rollout_task
        self.rollout_task = None

    async def suspend(self):
        await self.generate_scheduler.suspend.remote()

    async def _run_rollout_loop(self, seed):
        await asyncio.gather(*self.es_manager.run_rollout_loop(seed, blocking=False))

    async def _get_batch(self, batch_size, global_step):
        return await self.env_output_queue.get_batch.remote(batch_size, global_step)

    async def get_batch(self, data: DataProto, batch_size):
        global_step = data.meta_info["global_step"]

        if self.rollout_task is None:
            seed = random.randint(0, 1000000) if self.mode == "train" else self.config.seed
            self.rollout_task = asyncio.create_task(self._run_rollout_loop(seed))

        await asyncio.gather(*self.es_manager.update_step(global_step, blocking=False))
        await self.env_output_queue.advance_step.remote(global_step)
        await self.generate_scheduler.resume.remote()

        get_task = asyncio.create_task(self._get_batch(batch_size, global_step))
        await asyncio.wait({get_task, self.rollout_task}, return_when=asyncio.FIRST_COMPLETED)
        if self.rollout_task.done() and self.rollout_task.exception() is not None:
            await self.rollout_task
        data_batch = await get_task
        if batch_size <= 0:
            await self.rollout_task
            self.rollout_task = None
            await self.env_output_queue.clear.remote()

        if len(data_batch) == 0:
            return None

        metrics = {}
        get_batch_return_start_time = None
        for d_item in data_batch:
            get_batch_return_start_time = d_item.meta_info.pop("get_batch_return_start_time", None)
            append_to_dict(metrics, d_item.meta_info["metrics"])
        if get_batch_return_start_time is not None:
            metrics["time/get_batch_cost_gqm"] = time.time() - get_batch_return_start_time
        metrics.update(await self.env_output_queue.collect_metrics.remote())
        batch = DataProto.concat(data_batch)
        batch.meta_info["metrics"] = metrics
        batch.meta_info["get_batch_return_start_time"] = time.time()
        return batch