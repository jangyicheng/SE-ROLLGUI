import asyncio
import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from codetiming import Timer
from transformers import PreTrainedTokenizer, ProcessorMixin

from roll.utils.context_managers import local_profiler

from roll.pipeline.agentic.env_manager.base_env_manager import BaseEnvManager
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider, default_processor_provider, get_extra_data_provider
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.pipeline.agentic.memory_probe import collect_gui_traj_memory_snapshot
from roll.utils.checkpoint_manager import download_model
from roll.utils.import_utils import safe_import_class
from roll.utils.logging import get_logger


logger = get_logger()


class EnvironmentWorker(Worker):
    """
      Within a group, all environments share identical states by using the same seed.
      To reduce the overhead of dedicating one process per environment, parallelism is redesigned as **process + threads** :
      - One `EnvironmentWorker` holds multiple `EnvStateManager`s.
      - Each `EnvStateManager` manages the rollout loop for a single environment.
      - `EnvStateManager.run_rollout_loop` runs inside dedicated threads.
        TODO: GiGPO: https://arxiv.org/abs/2505.10978
    """

    def __init__(self, worker_config: EnvManagerConfig):
        super().__init__(worker_config)
        self.worker_config: EnvManagerConfig = worker_config
        self.env_managers: Dict[int, BaseEnvManager] = {}
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.processor: Optional[ProcessorMixin] = None
        self.env_configs: Dict[int, Dict] = worker_config.env_configs[self.rank]
        self.thread_lock = threading.Lock()
        self.output_queue = None
        self.mode = "train"

        # Lightweight memory monitor (log-only)
        self._monitor_thread = None
        self._monitor_stop = threading.Event()
        self._monitor_interval_sec = float(os.environ.get("ROLLOUT_CACHE_MONITOR_INTERVAL", "30"))
        self._group_snapshot_every_n = max(1, int(os.environ.get("ROLLOUT_CACHE_GROUP_SNAPSHOT_EVERY_N", "1")))
        self._obj_probe_every_n = max(1, int(os.environ.get("ROLLOUT_OBJ_PROBE_EVERY_N", "2")))
        self._monitor_tick = 0
        self._prev_rss_mb = None
        self._prev_cache_mb = None
        self._cache_logger = logger


    def _log_object_distribution_probe(self):
        # 低频采样，避免对主流程造成可见影响
        if self._monitor_tick % self._obj_probe_every_n != 0:
            return

        try:
            import gc
            import sys

            torch_mod = sys.modules.get("torch")
            np_mod = sys.modules.get("numpy")
            tensor_type = getattr(torch_mod, "Tensor", None)
            ndarray_type = getattr(np_mod, "ndarray", None)

            max_scan = int(os.environ.get("ROLLOUT_OBJ_PROBE_MAX_SCAN", "200000"))
            scanned = 0

            data_proto_count = 0
            data_proto_bytes = 0

            tensor_count = 0
            tensor_bytes = 0

            ndarray_count = 0
            ndarray_bytes = 0

            def _bytes_of(x):
                try:
                    nbytes = getattr(x, "nbytes", None)
                    if isinstance(nbytes, int):
                        return int(nbytes)

                    if tensor_type is not None and isinstance(x, tensor_type):
                        return int(x.element_size() * x.numel())
                except Exception:
                    return 0
                return 0

            for obj in gc.get_objects():
                scanned += 1
                if scanned > max_scan:
                    break

                try:
                    if isinstance(obj, DataProto):
                        data_proto_count += 1

                        batch = getattr(obj, "batch", None)
                        if batch is not None:
                            try:
                                for _, t in batch.items():
                                    data_proto_bytes += _bytes_of(t)
                            except Exception:
                                pass

                        non_tensor = getattr(obj, "non_tensor_batch", None)
                        if isinstance(non_tensor, dict):
                            try:
                                for v in non_tensor.values():
                                    data_proto_bytes += _bytes_of(v)
                            except Exception:
                                pass

                    if tensor_type is not None and isinstance(obj, tensor_type):
                        tensor_count += 1
                        tensor_bytes += _bytes_of(obj)

                    if ndarray_type is not None and isinstance(obj, ndarray_type):
                        ndarray_count += 1
                        ndarray_bytes += _bytes_of(obj)

                except ReferenceError:
                    # 弱引用对象在扫描时失效，直接跳过
                    continue
                except Exception:
                    # 单对象异常不影响整轮探针
                    continue

            self._cache_logger.info(
                "obj_probe scanned=%s dataproto_count=%s dataproto_mb=%.2f "
                "tensor_count=%s tensor_mb=%.2f ndarray_count=%s ndarray_mb=%.2f",
                scanned,
                data_proto_count,
                data_proto_bytes / (1024 * 1024),
                tensor_count,
                tensor_bytes / (1024 * 1024),
                ndarray_count,
                ndarray_bytes / (1024 * 1024),
            )
        except Exception as e:
            self._cache_logger.debug("object distribution probe failed: %s", e)

    @staticmethod
    def _get_process_rss_mb() -> float:
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        return float(parts[1]) / 1024.0
        except Exception:
            pass
        return 0.0

    def _collect_group_queue_memory_snapshot(self):
        if self.output_queue is None:
            return {}

        try:
            import ray

            if not hasattr(self.output_queue, "get_memory_snapshot"):
                return {}

            return ray.get(self.output_queue.get_memory_snapshot.remote())
        except Exception as e:
            self._cache_logger.debug("group queue memory snapshot failed: %s", e)
            return {}

    def _monitor_rollout_cache_loop(self):
        while not self._monitor_stop.is_set():
            try:
                rss_mb = self._get_process_rss_mb()
                gui_stats = collect_gui_traj_memory_snapshot(self.env_managers)

                cache_mb = float(gui_stats.get("memory/gui/cache_estimated_mb", 0.0))
                delta_rss = 0.0 if self._prev_rss_mb is None else rss_mb - self._prev_rss_mb
                delta_cache = 0.0 if self._prev_cache_mb is None else cache_mb - self._prev_cache_mb
                self._prev_rss_mb = rss_mb
                self._prev_cache_mb = cache_mb

                group_stats = {}
                if self._monitor_tick % self._group_snapshot_every_n == 0:
                    group_stats = self._collect_group_queue_memory_snapshot()
                self._monitor_tick += 1

                detail = gui_stats.get("memory/gui/top_envs", "")
                non_tensor_top = gui_stats.get("memory/gui/cache_non_tensor_top_keys", "")

                self._cache_logger.info(
                    "worker_rank=%s rss_mb=%.2f delta_rss_mb=%.2f rollout_cache_mb=%.2f delta_cache_mb=%.2f "
                    "obs_mb=%.2f msg_img_mb=%.2f msg_txt_mb=%.2f non_tensor_mb=%.2f frames_mb=%.2f non_tensor_top=%s "
                    "gui_envs=%s active_envs=%s gui_history=%s gui_frames=%s group_queues=%s "
                    "pending_groups=%s buffered_rollouts=%s group_payload_mb=%.2f %s",
                    self.rank,
                    rss_mb,
                    delta_rss,
                    cache_mb,
                    delta_cache,
                    float(gui_stats.get("memory/gui/cache_observation_mb", 0.0)),
                    float(gui_stats.get("memory/gui/cache_messages_image_mb", 0.0)),
                    float(gui_stats.get("memory/gui/cache_messages_text_mb", 0.0)),
                    float(gui_stats.get("memory/gui/cache_non_tensor_mb", 0.0)),
                    float(gui_stats.get("memory/gui/cache_frames_mb", 0.0)),
                    non_tensor_top,
                    gui_stats.get("memory/gui/env_count", 0),
                    gui_stats.get("memory/gui/active_envs", 0),
                    gui_stats.get("memory/gui/history_total", 0),
                    gui_stats.get("memory/gui/frames_total", 0),
                    group_stats.get("memory/group_queue/group_queues_count", 0),
                    group_stats.get("memory/group_queue/pending_groups_total", 0),
                    group_stats.get("memory/group_queue/buffered_rollouts_total", 0),
                    float(group_stats.get("memory/group_queue/rollout_payload_estimated_mb", 0.0)),
                    detail,
                )
                # if os.environ.get("ROLLOUT_OBJ_PROBE_ENABLE", "0") == "1":
                #     self._log_object_distribution_probe()
            except Exception as e:
                self._cache_logger.warning("rollout cache monitor error: %s", e)

            self._monitor_stop.wait(self._monitor_interval_sec)

    def _start_rollout_cache_monitor(self):
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_rollout_cache_loop,
            name=f"rollout-cache-monitor-{self.rank}",
            daemon=True,
        )
        self._monitor_thread.start()
        self._cache_logger.info(
            "rollout cache monitor started: worker_rank=%s interval_sec=%s",
            self.rank,
            self._monitor_interval_sec,
        )

    def _stop_rollout_cache_monitor(self):
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
        self._cache_logger.info("rollout cache monitor stopped: worker_rank=%s", self.rank)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def initialize(
        self,
        pipeline_config,
        generate_scheduler,
        output_queue,
        collator: Optional[callable] = None,
        mode: str = "train",
    ):
        super().initialize(pipeline_config)

        self.output_queue = output_queue
        self.mode = mode
        model_name_or_path = download_model(self.worker_config.model_args.model_name_or_path)
        self.tokenizer = default_tokenizer_provider(self.worker_config.model_args, model_name_or_path)
        self.processor = default_processor_provider(self.worker_config.model_args, model_name_or_path)

        def create_env_manager(env_id, env_config):
            if env_id == 0:
                self.logger.info(f"use env_manager_cls: {env_config['env_manager_cls']}")
            env_manager_cls = safe_import_class(env_config["env_manager_cls"])

            assert env_manager_cls is not None
            tokenizer = copy.deepcopy(self.tokenizer)
            processor = copy.deepcopy(self.processor)
            extra_data_provider = None
            if processor is not None and isinstance(processor, ProcessorMixin):
                extra_data_provider = get_extra_data_provider(model_name_or_path, processor=processor)
            return (
                env_id,
                env_manager_cls(
                    worker_config=self.worker_config,
                    pipeline_config=pipeline_config,
                    env_config=env_config,
                    tokenizer=tokenizer,  # https://github.com/huggingface/tokenizers/issues/537
                    processor=processor,
                    generate_scheduler=generate_scheduler,
                    output_queue=output_queue,
                    thread_lock=self.thread_lock,
                    mode=mode,
                    extra_data_provider=extra_data_provider,
                ),
            )

        with ThreadPoolExecutor(max_workers=min(len(self.env_configs), 64)) as executor:
            futures = [
                executor.submit(create_env_manager, env_id, env_config)
                for env_id, env_config in self.env_configs.items()
            ]
            for future in as_completed(futures):
                try:
                    env_id, env_manager = future.result()
                    self.env_managers[env_id] = env_manager
                except Exception as e:
                    self.logger.error(f"Failed to initialize env_manager: {e}", exc_info=True)
                    raise e

        # self._start_rollout_cache_monitor()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def run_rollout_loop(self, seed):
        # Set environment variables for profiler context
        os.environ["roll_EXEC_FUNC_NAME"] = "run_rollout_loop"
        os.environ["WORKER_NAME"] = f"EnvironmentWorker_{self.rank}"

        loop = asyncio.get_event_loop()
        pool = ThreadPoolExecutor(max_workers=len(self.env_managers))

        def run_with_profiler(env_manager, data_proto):
            with local_profiler():
                return env_manager.run_rollout_loop(data_proto)

        def run_without_profiler(env_manager, data_proto):
            return env_manager.run_rollout_loop(data_proto)

        tasks = []
        for env_id, env_manager in self.env_managers.items():
            # Only profile the first env_manager (env_id=0) on rank=0
            run_func = run_without_profiler
            if self.rank == 0 and env_id == 0:
                run_func = run_with_profiler
            tasks.append(loop.run_in_executor(pool, run_func, env_manager, DataProto(meta_info={"seed": seed})))

        await asyncio.gather(*tasks)
        pool.shutdown()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def update_step(self, global_step):
        for env_manager in self.env_managers.values():
            env_manager.update_step(global_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    async def stop(self):
        self._stop_rollout_cache_monitor()
        for env_manager in self.env_managers.values():
            env_manager.stop()
