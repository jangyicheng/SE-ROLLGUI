from __future__ import annotations

import json
from pathlib import Path


def discover_batches(traj_root: Path) -> list[Path]:
    return sorted([d for d in traj_root.iterdir() if d.is_dir()], reverse=True)


def collect_attempts(selected_batch: Path, subset_tasks: list[str]) -> list[dict]:
    subset_set = set(subset_tasks)
    all_attempts = []

    for task_dir in selected_batch.iterdir():
        if not task_dir.is_dir():
            continue
        if subset_set and task_dir.name not in subset_set:
            continue

        for attempt_dir in task_dir.iterdir():
            if not attempt_dir.is_dir():
                continue

            result_path = attempt_dir / "result.json"
            finished = result_path.exists()
            success = None

            if finished:
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        res = json.load(f)
                    success = res.get("success")
                except Exception:
                    success = None

            all_attempts.append({"path": attempt_dir, "finished": finished, "success": success})

    return all_attempts


def sort_attempts(all_attempts: list[dict], sort_mode: str) -> list[dict]:
    if sort_mode == "按时间（新→旧）":
        all_attempts.sort(key=lambda x: x["path"].stat().st_mtime, reverse=True)
    else:
        all_attempts.sort(key=lambda x: (x["path"].parent.name.lower(), x["path"].name.lower()))
    return all_attempts


def filter_attempts(all_attempts: list[dict], filter_mode: str) -> list[Path]:
    if filter_mode == "只显示成功轨迹":
        return [x["path"] for x in all_attempts if x["finished"] and x["success"] is True]
    if filter_mode == "只显示失败轨迹":
        return [x["path"] for x in all_attempts if x["finished"] and x["success"] is False]
    if filter_mode == "显示所有完成的轨迹":
        return [x["path"] for x in all_attempts if x["finished"]]
    return [x["path"] for x in all_attempts]
