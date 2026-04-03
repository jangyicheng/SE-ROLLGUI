import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from roll.distributed.scheduler.protocol import DataProto


def extract_episode_score(rollout: DataProto) -> Optional[float]:
    try:
        non_tensor = getattr(rollout, "non_tensor_batch", None) or {}
        arr = non_tensor.get("episode_scores")
        if arr is None or len(arr) == 0:
            return None
        return float(arr[0])
    except Exception:
        return None


def extract_task_name(rollout: DataProto) -> str:
    try:
        meta = getattr(rollout, "meta_info", None) or {}
        task = meta.get("task")
        if task:
            return str(task)
    except Exception:
        pass
    return "__unknown_task__"


def extract_step_count(rollout: DataProto) -> Optional[int]:
    try:
        non_tensor = getattr(rollout, "non_tensor_batch", None) or {}
        step_scores = non_tensor.get("step_scores")
        if step_scores is None:
            return None
        return int(len(step_scores))
    except Exception:
        return None


def aggregate_batch_task_metrics(ret: List[DataProto]) -> Dict[str, Dict]:
    per_task: Dict[str, Dict] = {}
    for rollout in ret:
        task = extract_task_name(rollout)
        score = extract_episode_score(rollout)
        steps = extract_step_count(rollout)

        if task not in per_task:
            per_task[task] = {
                "count": 0,
                "score_count": 0,
                "success_count": 0,
                "sum_episode_score": 0.0,
                "step_count": 0,
                "sum_steps": 0,
            }

        m = per_task[task]
        m["count"] += 1

        if score is not None:
            m["score_count"] += 1
            m["sum_episode_score"] += float(score)
            if score > 0.0:
                m["success_count"] += 1

        if steps is not None:
            m["step_count"] += 1
            m["sum_steps"] += int(steps)

    for m in per_task.values():
        score_count = m["score_count"]
        step_count = m["step_count"]
        m["success_rate"] = (m["success_count"] / score_count) if score_count > 0 else None
        m["avg_episode_score"] = (m["sum_episode_score"] / score_count) if score_count > 0 else None
        m["avg_steps"] = (m["sum_steps"] / step_count) if step_count > 0 else None

    return per_task


async def fetch_scheduler_snapshot(task_manager_url: str, logger) -> Dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            info_resp = await client.get(f"{task_manager_url}/info")
            info_resp.raise_for_status()
            stats_resp = await client.get(f"{task_manager_url}/stats")
            stats_resp.raise_for_status()
        return {"info": info_resp.json(), "stats": stats_resp.json()}
    except Exception as e:
        logger.warning(f"failed to fetch scheduler snapshot: {e}")
        return {}


def resolve_running_dir(scheduler_info: Dict, trajectory_root: Path, mode: str) -> Optional[Path]:
    info = scheduler_info.get("info", {})
    log_file = info.get("log_file")
    ts = info.get("timestamp")

    if log_file:
        base_dir = Path(log_file).parent
    elif ts:
        base_dir = trajectory_root / ts
    else:
        return None

    mode_dir = "train" if mode == "train" else "val"
    running_dir = base_dir / "running" / mode_dir
    running_dir.mkdir(parents=True, exist_ok=True)
    return running_dir


def build_queue_state(group_queues) -> Dict[str, int]:
    return {
        "group_queue_num": len(group_queues),
        "pending_episode_num": int(sum(len(q.groups) for q in group_queues.values())),
        "running_rollouts": int(
            sum(g.running_rollouts for q in group_queues.values() for g in q.groups.values())
        ),
    }


async def dump_running_batch_stats(
    ret: List[DataProto],
    current_step: int,
    batch_size: int,
    group_queues,
    mode: str,
    trajectory_root: Path,
    task_manager_url: str,
    logger,
):
    if not ret:
        return

    scheduler_snapshot = await fetch_scheduler_snapshot(task_manager_url=task_manager_url, logger=logger)
    running_dir = resolve_running_dir(
        scheduler_info=scheduler_snapshot,
        trajectory_root=trajectory_root,
        mode=mode,
    )
    if running_dir is None:
        return

    batch_task_metrics = aggregate_batch_task_metrics(ret)
    total_score_count = sum(v["score_count"] for v in batch_task_metrics.values())
    total_success_count = sum(v["success_count"] for v in batch_task_metrics.values())

    record = {
        "ts": time.time(),
        "mode": mode,
        "current_step": current_step,
        "requested_batch_size": batch_size,
        "returned_rollouts": len(ret),
        "batch_success_rate": (total_success_count / total_score_count) if total_score_count > 0 else None,
        "batch_task_metrics": batch_task_metrics,
        "scheduler_info": scheduler_snapshot.get("info", {}),
        "scheduler_tasks": scheduler_snapshot.get("stats", {}).get("tasks", {}),
        "queue_state": build_queue_state(group_queues),
    }

    out_file = running_dir / "batch_stats.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
