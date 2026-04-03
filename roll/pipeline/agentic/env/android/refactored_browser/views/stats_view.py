from __future__ import annotations

from pathlib import Path

import altair as alt
import streamlit as st

from refactored_browser.services.stats_service import (
    apply_task_subset_to_df,
    build_ranked_tables,
    calc_metrics,
    discover_stats_files,
    load_stats_json,
    make_hist_df,
    tasks_to_df,
)


def _plot_sorted_hist(hist_df, x_col: str, y_col: str):
    chart = (
        alt.Chart(hist_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:N", sort=alt.SortField(field="_sort", order="ascending"), title=x_col),
            y=alt.Y(f"{y_col}:Q", title=y_col),
            tooltip=[alt.Tooltip(f"{x_col}:N", title=x_col), alt.Tooltip(f"{y_col}:Q", title=y_col)],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def _plot_rank_chart(rank_df, title: str):
    st.markdown(f"**{title}**")
    chart = (
        alt.Chart(rank_df)
        .mark_bar()
        .encode(
            x=alt.X("task:N", sort=alt.SortField(field="_order", order="ascending"), title="任务"),
            y=alt.Y("average_success_rate:Q", title="平均成功率"),
            tooltip=[
                alt.Tooltip("task:N", title="任务"),
                alt.Tooltip("average_success_rate:Q", title="平均成功率"),
            ],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def render_stats_mode(traj_root: Path, subset_name: str, subset_tasks: list[str]):
    def format_stats_display(p: Path) -> str:
        # 尝试获取父目录和祖父目录名称
        parent = p.parent
        grandparent = parent.parent if parent.parent else None

        # 如果祖父目录存在且其名称不是 "trajectories"，则显示更完整路径
        if grandparent and grandparent.name != "trajectories":
            return f"{grandparent.name}/{p.name}"
        else:
            return f"{parent.name}/{p.name}"
    st.header("📈 轨迹统计可视化")

    stats_files = discover_stats_files(traj_root)
    if not stats_files:
        st.warning("未找到包含 tasks/global_stats 的统计JSON文件")
        return

    selected_stats = st.sidebar.selectbox("统计文件", stats_files, format_func=format_stats_display)

    stats_data = load_stats_json(selected_stats)
    df = tasks_to_df(stats_data)
    global_stats = stats_data.get("global_stats", {})

    use_subset = bool(subset_tasks)
    df = apply_task_subset_to_df(df, subset_tasks)

    st.caption(f"当前统计文件：{selected_stats}")
    st.caption(f"当前任务子集：{subset_name}")

    total_complete, total_success, global_success_rate, avg_success_rate, total_task_types = calc_metrics(
        df, global_stats, use_subset=use_subset
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总完成数", total_complete)
    c2.metric("总成功数", total_success)
    c3.metric("全局成功率", f"{global_success_rate * 100:.1f}%")
    c4.metric("任务平均成功率", f"{avg_success_rate * 100:.1f}%")
    c5.metric("任务类型数", total_task_types)

    if df.empty:
        st.info("当前子集下 tasks 为空")
        return

    st.subheader("1. 任务成功率分布")
    success_hist = make_hist_df(df["average_success_rate"], bins=20, x_name="成功率区间", y_name="任务数")
    st.bar_chart(success_hist.set_index("成功率区间"))

    st.subheader("2. 完成步数与耗时分布")
    col_steps, col_time = st.columns(2)
    with col_steps:
        step_hist = make_hist_df(df["average_steps"], bins=15, x_name="平均步数区间", y_name="任务数")
        _plot_sorted_hist(step_hist, "平均步数区间", "任务数")
    with col_time:
        time_hist = make_hist_df(df["average_time"], bins=15, x_name="平均耗时区间(秒)", y_name="任务数")
        _plot_sorted_hist(time_hist, "平均耗时区间(秒)", "任务数")

    st.subheader("3. 成功率最高与最低的N个任务")
    n_tasks = 40
    top_df, bottom_df = build_ranked_tables(df, top_n=n_tasks)

    t1, t2 = st.columns(2)
    with t1:
        _plot_rank_chart(top_df, f"Top {n_tasks}")
        st.dataframe(
            top_df[["task", "average_success_rate", "complete_attempts", "average_steps", "average_time"]],
            use_container_width=True,
        )
    with t2:
        _plot_rank_chart(bottom_df, f"Bottom {n_tasks}")
        st.dataframe(
            bottom_df[["task", "average_success_rate", "complete_attempts", "average_steps", "average_time"]],
            use_container_width=True,
        )
