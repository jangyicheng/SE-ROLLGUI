from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def is_stats_json_file(path: Path) -> bool:
    if path.name in {"meta.json", "result.json"} or path.name.endswith(".jsonl"):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and "tasks" in data and "global_stats" in data
    except Exception:
        return False


def discover_stats_files(traj_root: Path) -> list[Path]:
    preferred = []
    for batch_dir in traj_root.iterdir():
        if batch_dir.is_dir():
            candidate = batch_dir / f"{batch_dir.name}.json"
            if candidate.exists() and is_stats_json_file(candidate):
                preferred.append(candidate)

    others = []
    for path in traj_root.rglob("*.json"):
        if path in preferred:
            continue
        if is_stats_json_file(path):
            others.append(path)

    all_files = list({x.resolve(): x for x in (preferred + others)}.values())
    all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return all_files


def load_stats_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tasks_to_df(stats_data: dict) -> pd.DataFrame:
    rows = []
    for task_name, info in stats_data.get("tasks", {}).items():
        if not isinstance(info, dict):
            continue
        rows.append(
            {
                "task": task_name,
                "complete_attempts": info.get("complete_attempts", 0),
                "success_count": info.get("success_count", 0),
                "failure_count": info.get("failure_count", 0),
                "average_steps": info.get("average_steps", 0.0),
                "average_time": info.get("average_time", 0.0),
                "average_success_rate": info.get("average_success_rate", 0.0),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "complete_attempts",
        "success_count",
        "failure_count",
        "average_steps",
        "average_time",
        "average_success_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["average_success_rate"] = df["average_success_rate"].clip(0, 1)
    return df


def apply_task_subset_to_df(df: pd.DataFrame, subset_tasks: list[str]) -> pd.DataFrame:
    if not subset_tasks:
        return df
    return df[df["task"].isin(subset_tasks)].copy()


def calc_metrics(df: pd.DataFrame, global_stats: dict, use_subset: bool) -> tuple[int, int, float, float, int]:
    if use_subset:
        if df.empty:
            return 0, 0, 0.0, 0.0, 0

        total_complete = int(df["complete_attempts"].sum())
        total_success = int(df["success_count"].sum())
        global_success_rate = (total_success / total_complete) if total_complete > 0 else 0.0
        average_success_rate = float(df["average_success_rate"].mean())
        total_task_types = int(len(df))
        return total_complete, total_success, global_success_rate, average_success_rate, total_task_types

    return (
        int(global_stats.get("total_complete_count", 0)),
        int(global_stats.get("total_success_count", 0)),
        float(global_stats.get("global_success_rate", 0)),
        float(global_stats.get("average_success_rate", 0)),
        int(global_stats.get("total_task_types", 0)),
    )


def make_hist_df(series: pd.Series, bins: int, x_name: str = "bin", y_name: str = "count") -> pd.DataFrame:
    clean = series.dropna()
    if clean.empty:
        return pd.DataFrame(columns=[x_name, y_name, "_sort"])

    counts, edges = np.histogram(clean, bins=bins)
    labels = [f"{edges[i]:.2f}~{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]

    return pd.DataFrame(
        {
            x_name: labels,
            y_name: counts,
            "_sort": edges[:-1],
        }
    )


def build_ranked_tables(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = df[df["complete_attempts"] > 0].copy()
    ranked.sort_values(by=["average_success_rate", "task"], ascending=[False, True], inplace=True)

    top_df = ranked.head(top_n).copy().reset_index(drop=True)
    top_df["_order"] = range(len(top_df))

    bottom_df = ranked.tail(top_n).copy()
    bottom_df.sort_values(by=["average_success_rate", "task"], ascending=[True, True], inplace=True)
    bottom_df = bottom_df.reset_index(drop=True)
    bottom_df["_order"] = range(len(bottom_df))

    return top_df, bottom_df
