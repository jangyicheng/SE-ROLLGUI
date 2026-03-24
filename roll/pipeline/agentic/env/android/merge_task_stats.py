#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple


# 示例（合并到新目录）:
# python merge_task_stats.py \
#   --input-dirs \
#   /HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL/trajectories/2026-03-23_120554 \
#   /HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL/trajectories/2026-03-24_055558 \
#   --output-dir /HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL/trajectories/merged_Qwen3-VL-reflection

# 示例（就地合并 A + B -> A）:
# python merge_task_stats.py \
#   --input-dirs /path/to/A /path/to/B \
#   --output-dir /path/to/A \
#   --allow-inplace


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_stats_json_file(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".json":
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and "tasks" in data and "global_stats" in data
    except Exception:
        return False


def find_stats_json(source_dir: Path) -> Path:
    # 优先使用“目录同名json”
    preferred = source_dir / f"{source_dir.name}.json"
    if is_stats_json_file(preferred):
        return preferred

    # 兜底：找目录下所有顶层json
    candidates = [p for p in sorted(source_dir.glob("*.json")) if is_stats_json_file(p)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"{source_dir} 下存在多个可用统计JSON，请只保留一个或使用同名JSON。候选: {candidates}"
        )
    raise FileNotFoundError(f"{source_dir} 下未找到统计JSON（需包含 tasks/global_stats）")


def compute_global_stats(task_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    attempted_tasks = 0
    sum_success_rates = 0.0
    total_success = 0
    total_complete = 0

    for stats in task_stats.values():
        complete_attempts = int(stats.get("complete_attempts", 0))
        if complete_attempts > 0:
            attempted_tasks += 1
            sum_success_rates += float(stats.get("average_success_rate", 0.0))
            total_success += int(stats.get("success_count", 0))
            total_complete += complete_attempts

    return {
        "total_success_count": total_success,
        "total_complete_count": total_complete,
        "total_task_types": len(task_stats),
        "attempted_task_types": attempted_tasks,
        "global_success_rate": (total_success / total_complete) if total_complete > 0 else 0.0,
        "average_success_rate": (sum_success_rates / attempted_tasks) if attempted_tasks > 0 else 0.0,
        "average_completions_per_task": (total_complete / attempted_tasks) if attempted_tasks > 0 else 0.0,
    }


def merge_stats_from_files(stats_files: List[Path]) -> Dict[str, Any]:
    merged_raw: Dict[str, Dict[str, Any]] = {}

    for file_path in stats_files:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        tasks = data.get("tasks", {})
        if not isinstance(tasks, dict):
            raise ValueError(f"{file_path} 的 tasks 不是对象(dict)")

        for task_name, s in tasks.items():
            complete_attempts = int(s.get("complete_attempts", 0))
            success_count = int(s.get("success_count", 0))
            failure_count = int(s.get("failure_count", 0))
            avg_steps = float(s.get("average_steps", 0.0))
            avg_time = float(s.get("average_time", 0.0))

            if task_name not in merged_raw:
                merged_raw[task_name] = {
                    "complete_attempts": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "sum_steps": 0.0,
                    "sum_time": 0.0,
                }

            merged_raw[task_name]["complete_attempts"] += complete_attempts
            merged_raw[task_name]["success_count"] += success_count
            merged_raw[task_name]["failure_count"] += failure_count
            merged_raw[task_name]["sum_steps"] += avg_steps * complete_attempts
            merged_raw[task_name]["sum_time"] += avg_time * complete_attempts

    merged_tasks = {}
    for task_name, acc in sorted(merged_raw.items()):
        complete_attempts = acc["complete_attempts"]
        success_count = acc["success_count"]

        average_steps = (acc["sum_steps"] / complete_attempts) if complete_attempts > 0 else 0.0
        average_time = (acc["sum_time"] / complete_attempts) if complete_attempts > 0 else 0.0
        average_success_rate = (success_count / complete_attempts) if complete_attempts > 0 else 0.0

        merged_tasks[task_name] = {
            "complete_attempts": complete_attempts,
            "success_count": success_count,
            "failure_count": acc["failure_count"],
            "average_steps": average_steps,
            "average_time": average_time,
            "average_success_rate": average_success_rate,
        }

    return {
        "last_saved": now_utc_iso(),
        "tasks": merged_tasks,
        "global_stats": compute_global_stats(merged_tasks),
    }


def make_unique_tag(base: str, used: set) -> str:
    if base not in used:
        used.add(base)
        return base
    idx = 2
    while f"{base}_{idx}" in used:
        idx += 1
    tag = f"{base}_{idx}"
    used.add(tag)
    return tag


def _unique_run_dir(dst_task_dir: Path, run_name: str, source_tag: str) -> Path:
    candidate = dst_task_dir / run_name
    if not candidate.exists():
        return candidate

    idx = 2
    while True:
        candidate = dst_task_dir / f"{run_name}__{source_tag}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def move_trajectories_by_task(
    source_dir: Path,
    stats_json_path: Path,
    dst_root: Path,
    source_tag: str,
    inplace_mode: bool,
) -> Tuple[int, List[str]]:
    """
    将 source_dir 下每个子任务目录中的“每次运行目录”移动到 dst_root/<task>/ 下。
    返回: (移动的运行目录数量, 目标目录样例列表)
    """
    moved_runs = 0
    moved_samples: List[str] = []

    # 就地模式且 source==dst 时，不移动自身轨迹（已经在目标目录中）
    if inplace_mode and source_dir.resolve() == dst_root.resolve():
        return 0, []

    for task_dir in sorted(source_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        # 防御：避免把输出目录本身当作任务目录
        if task_dir.resolve() == dst_root.resolve():
            continue

        dst_task_dir = dst_root / task_dir.name
        dst_task_dir.mkdir(parents=True, exist_ok=True)

        for run_dir in sorted(task_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            target_run_dir = _unique_run_dir(dst_task_dir, run_dir.name, source_tag)
            shutil.move(str(run_dir), str(target_run_dir))
            moved_runs += 1

            if len(moved_samples) < 20:
                moved_samples.append(str(target_run_dir.relative_to(dst_root).as_posix()))

        # 尝试清理已空的任务目录
        try:
            task_dir.rmdir()
        except OSError:
            pass

    return moved_runs, moved_samples

def load_inherited_sources(source_dir: Path, manifest_name: str) -> List[Dict[str, Any]]:
    manifest_path = source_dir / manifest_name
    if not manifest_path.is_file():
        return []
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sources = data.get("sources", [])
        if isinstance(sources, list):
            return [x for x in sources if isinstance(x, dict)]
    except Exception:
        pass
    return []


def dedupe_source_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result: List[Dict[str, Any]] = []
    for r in records:
        key = (
            str(r.get("source_dir", "")),
            str(r.get("stats_json", "")),
            str(r.get("source_tag", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(r)
    return result

def main():
    parser = argparse.ArgumentParser(
        description="按目录合并任务统计JSON，并将轨迹按子任务合并到输出目录"
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="输入批次目录路径（可多个），每个目录应包含统计JSON与轨迹目录",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录（合并后的JSON与轨迹都写到这里）",
    )
    parser.add_argument(
        "--stats-output-name",
        default="merged_stats.json",
        help="合并统计JSON文件名，默认 merged_stats.json",
    )
    parser.add_argument(
        "--manifest-name",
        default="merge_manifest.json",
        help="合并来源记录文件名，默认 merge_manifest.json",
    )
    parser.add_argument(
        "--allow-inplace",
        action="store_true",
        help="允许输出目录与某个输入目录相同（例如 A+B->A）",
    )
    args = parser.parse_args()

    input_dirs = [Path(p).resolve() for p in args.input_dirs]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_dir in input_dirs and not args.allow_inplace:
        raise ValueError("检测到就地合并场景，请显式加 --allow-inplace")

    # 1) 找每个目录的统计json
    source_infos = []
    for d in input_dirs:
        if not d.exists() or not d.is_dir():
            raise NotADirectoryError(f"输入目录不存在或不是目录: {d}")
        stats_json = find_stats_json(d)
        inherited_sources = load_inherited_sources(d, args.manifest_name)
        source_infos.append(
            {
                "source_dir": d,
                "stats_json": stats_json,
                "inherited_sources": inherited_sources,
            }
        )
    # 2) 合并统计
    merged_stats = merge_stats_from_files([x["stats_json"] for x in source_infos])

    stats_output_path = output_dir / args.stats_output_name
    with stats_output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_stats, f, ensure_ascii=False, indent=2)

    # 3) 合并轨迹（按子任务聚合）+ 记录来源
    used_tags = set()
    manifest_sources = []
    total_moved_runs = 0

    for info in source_infos:
        src_dir = info["source_dir"]
        stats_json = info["stats_json"]
        source_tag = make_unique_tag(src_dir.name, used_tags)

        moved_count, moved_samples = move_trajectories_by_task(
            source_dir=src_dir,
            stats_json_path=stats_json,
            dst_root=output_dir,
            source_tag=source_tag,
            inplace_mode=args.allow_inplace,
        )
        total_moved_runs += moved_count

        manifest_sources.append(
            {
                "source_dir": str(src_dir),
                "source_tag": source_tag,
                "stats_json": str(stats_json),
                "moved_run_dir_count": moved_count,
                "moved_run_dirs_sample": moved_samples,
            }
        )

    inherited_all: List[Dict[str, Any]] = []
    for info in source_infos:
        inherited_all.extend(info.get("inherited_sources", []))

    inherited_all = dedupe_source_records(inherited_all)
    all_sources = dedupe_source_records(inherited_all + manifest_sources)

    manifest = {
    "merged_at": now_utc_iso(),
    "output_dir": str(output_dir),
    "merged_stats_file": str(stats_output_path),
    "source_dir_count": len(source_infos),
    "total_moved_run_dirs": total_moved_runs,
    "inplace_mode": bool(args.allow_inplace),
    "inherited_source_record_count": len(inherited_all),
    "direct_source_record_count": len(manifest_sources),
    "sources": all_sources,
    }

    manifest_path = output_dir / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"统计已合并: {stats_output_path}")
    print(f"轨迹已按子任务合并到: {output_dir}")
    print(f"来源记录已写入: {manifest_path}")


if __name__ == "__main__":
    main()