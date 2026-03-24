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


# 可以实现评测的版本，但是没有实现保证相同的任务分配进同一个groupQueue的功能，以及根据 async_generation_ratio 管理数据过时性的功能。
# 请你恢复使用GroupQueue的方式，但仍然保持动态分配 group_id 的 GroupQueueManager，仅在现有的代码上加入下面几点功能：
# 1. 恢复功能：为每个group对应的任务均分配一个GroupQueue，并为其中的GroupData赋予准确的episode_id
# 2. 恢复功能：使得GroupQueueManager能够根据async_generation_ratio去管理数据的过时性

@ray.remote
class GroupQueueManager:
    """
    动态任务分配的 GroupQueueManager。
    环境不再绑定固定 group_id，而是每次请求任务时动态分配。
    同一任务的 group_size 条轨迹填入同一个 group。
    """

    def __init__(self, config, env_manager_config: EnvManagerConfig, mode):
        self.mode = mode
        self.env_manager_config = env_manager_config
        self.group_size = self.env_manager_config.group_size
        self.group_size_redundancy = env_manager_config.group_size_redundancy
        self.progress_bar = tqdm(desc=f"{self.mode} rollout progress(trajectory)", mininterval=1)
        self.pending_gets = set()
        self.rollout_complete = {}

        group_filter_cls = safe_import_class(env_manager_config.group_filter_cls)
        assert group_filter_cls
        self.group_filter = group_filter_cls(config, env_manager_config, mode)
        self.group_filter_count = 0

        if self.mode == "train":
            self.async_generation_ratio = config.async_generation_ratio
        else:
            self.async_generation_ratio = 0

        # 动态 group 管理
        self.next_group_id = 0
        self.groups: Dict[int, GroupData] = {}  # group_id -> GroupData
        self.task_to_pending_group: Dict[str, int] = {}  # task -> 当前未满的 group_id
        self.task_to_fixed_group: Dict[str, int] = {}    # task -> 永久固定 group_id（新增）
        self.completed_groups: List[GroupData] = []  # 已完成待消费的 groups

        # TaskManager URL（需要在 env_manager_config 中配置 task_manager_url）
        self.task_manager_url = getattr(env_manager_config, 'task_manager_url', 'http://localhost:5001')

        self.current_step = None
        self.progress = asyncio.Event()
        self.complete = asyncio.Event()
        self.quit = False

        # 统计所有 env 数量，用于判断所有 env 退出
        self.total_env_count = sum(
            len(rank_envs) for rank_envs in env_manager_config.env_configs.values()
        )
        self.env_exit_count = 0
        self.all_done = False

        # for debug
        self.total = 0
        self.waiting = 0
        self.current_group_id = None
        self.current_task = ''


    def _allocate_group(self, task: str) -> int:
        """为指定任务分配 group；同一 task 永久复用同一 group_id。"""
        if task in self.task_to_fixed_group:
            group_id = self.task_to_fixed_group[task]
        else:
            group_id = self.next_group_id
            self.next_group_id += 1
            self.task_to_fixed_group[task] = group_id

        # 若该 group 当前不在活跃组中，则创建一个新的轮次容器
        if group_id not in self.groups:
            self.groups[group_id] = GroupData(
                group_id=group_id,
                episode_id=group_id,  # 保持你当前逻辑，最小改动
                create_step=self.current_step or 0,
                task=task,
            )

        self.task_to_pending_group[task] = group_id
        logger.info(f"Allocated group {group_id} for task: {task}")
        return group_id

    async def get_episode_id(self, group_id=None) -> Optional[Dict]:
        """
        环境请求任务分配。返回 {'group_id': ..., 'episode_id': ..., 'task': ...} 或 None。
        忽略传入的 group_id 参数（保持向后兼容）。
        """
        # 等待 advance_step 被调用
        while self.current_step is None and not self.quit:
            self.progress.clear()
            await self.progress.wait()

        if self.quit:
            return None

        # 调用 TaskManager 获取任务
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
            logger.info("TaskManager returned 'finish', no more tasks")
            return None

        # 查找或创建该任务的 group
        if task in self.task_to_pending_group:
            gid = self.task_to_pending_group[task]
            if gid in self.groups:
                group = self.groups[gid]
                group.running_rollouts += 1
                # 如果已分配足够多的 env，移除 pending 映射
                if group.running_rollouts >= self.group_size + self.group_size_redundancy:
                    del self.task_to_pending_group[task]
            else:
                # group 已被消费或清理，重新分配
                gid = self._allocate_group(task)
                self.groups[gid].running_rollouts = 1
        else:
            gid = self._allocate_group(task)
            self.groups[gid].running_rollouts = 1

        return {
            'group_id': gid,
            'episode_id': self.groups[gid].episode_id,
            'task': task,
        }

    def collect_metrics(self):
        group_filter_count = self.group_filter_count
        self.group_filter_count = 0
        return {"scheduler/group_filter_count": group_filter_count}

    def clear(self):
        self.rollout_complete = {}
        for get_task in self.pending_gets:
            get_task.cancel()
        self.pending_gets = set()
        self.groups.clear()
        self.task_to_pending_group.clear()
        self.completed_groups.clear()
        self.current_step = None
        self.env_exit_count = 0
        self.all_done = False
        self.progress = asyncio.Event()
        self.complete = asyncio.Event()

    def advance_step(self, step):
        self.current_step = step
        self.progress.set()

    def shutdown(self):
        self.quit = True
        self.groups.clear()
        self.task_to_pending_group.clear()
        self.completed_groups.clear()
        self.progress.set()
        self.complete.set()
        for get_task in self.pending_gets:
            get_task.cancel()
        self.pending_gets = set()

    def put(self, group_id, episode_id, start_step, rollout: DataProto):
        # 处理环境退出信号
        if rollout is None:
            self.env_exit_count += 1
            logger.info(f"Env exit signal received ({self.env_exit_count}/{self.total_env_count})")
            if self.env_exit_count >= self.total_env_count:
                self.all_done = True
                self.complete.set()
            return

        if group_id not in self.groups:
            logger.warning(f"Received rollout for unknown group_id {group_id}, ignoring")
            return

        self.waiting += 1
        group = self.groups[group_id]
        group.rollouts.append(rollout)

        if len(group.rollouts) == self.group_size:
            task = group.task
            if all(r is None for r in group.rollouts):
                logger.info(f"GroupQueueManager: group {group_id} all None, skip")
                self.groups.pop(group_id)
                if task in self.task_to_pending_group and self.task_to_pending_group[task] == group_id:
                    del self.task_to_pending_group[task]
            elif self.group_filter.filter(group_id=group_id, episode_id=episode_id, group=group.rollouts):
                logger.info(f"filter rollout group {group.group_id} episode {group.episode_id}")
                self.group_filter_count += 1
                candidate_idxs = [i for i, r in enumerate(group.rollouts) if r is not None]
                if not candidate_idxs:
                    logger.info(f"no valid rollouts to replace for group {group.group_id} episode {group.episode_id}, drop group")
                    self.groups.pop(episode_id)
                    self.advance_group(create_step=self.current_step)
                    return
                replacement_idx = random.choice(candidate_idxs)

                try:
                    # 尝试从已有 rollouts 中提取 task 标识
                    task = None
                    for r in group.rollouts:
                        if r is None:
                            continue
                        meta = getattr(r, "meta_info", None)
                        if isinstance(meta, dict):
                            task = meta.get("task") 
                            if task:
                                break
                except Exception as e:
                    logger.info(f"failed to extract task from rollouts: {e}")
                    task = None
                            
                replaced = False
                if task is not None:
                    try:
                        traj_cache_actor = ray.get_actor("global_traj_cache")
                        cached = ray.get(traj_cache_actor.get_best_trajectory.remote(task))
                        if cached:
                            if isinstance(cached, DataProto):
                                group.rollouts[replacement_idx] = cached
                                replaced = True
                            else:
                                logger.info("global cache returned a rollout but cannot convert to DataProto")
                    except Exception as e:
                        logger.info(f"failed to fetch/convert cached trajectory: {e}")

                if replaced:
                        # 替换成功：保留该组（已替换），并补一个新的 group 占位，唤醒等待者
                        self.advance_group(create_step=self.current_step)
                        self.complete.set()
                        self.progress_bar.update(self.group_size)
                else:
                    # 缓存返回失败或无 task：仍然按 filter 行为过滤掉该组（丢弃）
                    logger.info(f"drop filtered group {group.group_id} episode {group.episode_id} due to cache miss")
                    self.groups.pop(episode_id)
                    self.advance_group(create_step=self.current_step)
                if task in self.task_to_pending_group and self.task_to_pending_group[task] == group_id:
                    del self.task_to_pending_group[task]
            else:
                # group 完成，移入已完成队列
                self.groups.pop(group_id)
                if task in self.task_to_pending_group and self.task_to_pending_group[task] == group_id:
                    del self.task_to_pending_group[task]
                self.completed_groups.append(group)
                self.complete.set()
                self.progress_bar.update(self.group_size)

        self.waiting -= 1
        self.total += 1

    async def get_batch(self, batch_size, current_step) -> List[DataProto]:
        """
        返回已完成的 rollouts，按 group 收集。
        batch_size < 0 表示收集所有直到所有环境退出。
        """
        ret: List[DataProto] = []

        while batch_size < 0 or len(ret) < batch_size:
            # 检查是否全部完成
            if self.all_done and not self.completed_groups:
                break

            # 等待有已完成的 group
            while not self.completed_groups:
                if self.all_done:
                    break
                self.complete.clear()
                await self.complete.wait()

            if not self.completed_groups:
                break

            group = self.completed_groups.pop(0)
            for rollout in group.rollouts:
                if rollout is not None:
                    ret.append(rollout)

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