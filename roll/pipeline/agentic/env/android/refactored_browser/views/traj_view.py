from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from refactored_browser.services.traj_service import collect_attempts, discover_batches, filter_attempts, sort_attempts
from refactored_browser.utils.action_overlay import get_image_path_for_step, get_overlay_image_path
from refactored_browser.utils.inline_image import render_image_inline


@st.cache_data(show_spinner=False, max_entries=1024)
def _load_steps_cached(steps_path_str: str, mtime_ns: int) -> list[dict]:
    _ = mtime_ns
    steps_data: list[dict] = []
    with open(steps_path_str, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                steps_data.append(json.loads(line))
    return steps_data


def _load_steps(selected_attempt: Path) -> list[dict]:
    steps_path = selected_attempt / "steps.jsonl"

    if steps_path.exists():
        raw_steps = _load_steps_cached(str(steps_path), steps_path.stat().st_mtime_ns)
        steps_data = [dict(x) for x in raw_steps]
    else:
        steps_data = []

    if steps_data and steps_data[0].get("step") == 0:
        steps_data.pop(0)

    last_step = steps_data[-1]["step"] if steps_data else 0
    steps_data.append(
        {
            "step": int(last_step) + 1,
            "action": "Terminate",
            "timestamp": "None",
            "image_step": int(last_step) + 1,
            "is_terminal": True,
        }
    )
    return steps_data


def render_traj_mode(traj_root: Path, subset_name: str, subset_tasks: list[str]):
    st.sidebar.header("📂 轨迹筛选")
    st.sidebar.caption(f"当前任务子集：{subset_name}")

    batches = discover_batches(traj_root)
    if not batches:
        st.info("暂无轨迹记录")
        return

    selected_batch = st.sidebar.selectbox("1. 选择运行批次", batches, format_func=lambda x: x.name)

    all_attempts = collect_attempts(selected_batch, subset_tasks)

    sort_mode = st.sidebar.selectbox("2. 排序方式", ["按时间（新→旧）", "按任务名（A→Z）"])
    all_attempts = sort_attempts(all_attempts, sort_mode)

    filter_mode = st.sidebar.selectbox(
        "3. 轨迹筛选",
        ["只显示成功轨迹", "只显示失败轨迹", "显示所有完成的轨迹", "显示所有轨迹"],
    )

    filtered_attempts = filter_attempts(all_attempts, filter_mode)
    if not filtered_attempts:
        st.info("当前筛选条件下暂无轨迹记录")
        return

    selected_attempt = st.sidebar.selectbox(
        "4. 选择具体任务记录",
        filtered_attempts,
        format_func=lambda x: f"{x.parent.name} ({x.name})",
    )

    view_mode = st.sidebar.radio("视图模式", ["📜 详细步骤", "📊 缩略图视图"])

    if view_mode == "📊 缩略图视图":
        max_render_count = st.sidebar.slider("5. 缩略图最大显示数", min_value=8, max_value=200, value=48, step=4)
    else:
        max_render_count = st.sidebar.slider("5. 详细步骤最大显示数", min_value=10, max_value=300, value=50, step=10)

    meta_path = selected_attempt / "meta.json"
    result_path = selected_attempt / "result.json"

    screenshot_count = len(list(selected_attempt.glob("step_*.png")))

    st.markdown(f"<h3>当前路径：{selected_attempt}</h3>", unsafe_allow_html=True)

    col_meta, col_res = st.columns([2, 1])

    with col_meta:
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.markdown(f"<h2>🎯 目标: {meta.get('goal', 'N/A')}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h2>任务 ID: <code>{meta.get('name', 'N/A')}</code></h2>", unsafe_allow_html=True)

    with col_res:
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                res = json.load(f)
            success = res.get("success")
            color = "green" if success else "red"
            result_text = "成功" if success else "失败"
            steps_text = res.get("steps")
            time_text = f"{res.get('time', 0):.1f}s"
            completion_reason = res.get("termination_reason", "未知")
        else:
            color = "orange"
            result_text = "未完成"
            steps_text = screenshot_count
            time_text = "-"
            completion_reason = "-"

        st.markdown(f"<h2>结果: <span style='color:{color}'>{result_text}</span></h2>", unsafe_allow_html=True)
        st.markdown(f"<h2>步数: {steps_text}<br>耗时: {time_text}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2>结束原因: {completion_reason}</h2>", unsafe_allow_html=True)

    st.divider()

    steps_data = _load_steps(selected_attempt)

    valid_steps = []
    for step in steps_data:
        step_num = step.get("step")
        if isinstance(step_num, int) and step_num > 0:
            valid_steps.append(step)

    if len(valid_steps) > max_render_count:
        st.caption(f"为提升速度，仅渲染最近 {max_render_count} 步（共 {len(valid_steps)} 步）")
        valid_steps = valid_steps[-max_render_count:]

    if view_mode == "📊 缩略图视图":
        st.subheader("📊 截图总览")
        cols = st.columns(4)

        rendered_images = set()
        thumb_items = []

        for step in valid_steps:
            step_num = step["step"]
            image_step = step.get("image_step", step_num)
            img_path = get_image_path_for_step(selected_attempt, image_step)
            if not img_path.exists():
                continue

            img_key = img_path.name
            if img_key in rendered_images and not step.get("is_terminal", False):
                continue

            rendered_images.add(img_key)
            thumb_items.append((step_num, step, img_path))

        thumb_items.sort(key=lambda x: x[0])

        for idx, (step_num, step, img_path) in enumerate(thumb_items):
            with cols[idx % 4]:
                overlay_path = get_overlay_image_path(img_path, step)
                render_image_inline(overlay_path, use_container_width=True)
                st.caption(f"Step {step_num}")
    else:
        st.subheader("🚀 执行序列")

        for step in valid_steps:
            step_num = step["step"]
            with st.container():
                c1, c2 = st.columns([1, 2])
                image_step = step.get("image_step", step_num)
                img_path = get_image_path_for_step(selected_attempt, image_step)

                with c1:
                    if img_path.exists():
                        overlay_path = get_overlay_image_path(img_path, step)
                        render_image_inline(overlay_path, use_container_width=True)
                    else:
                        st.caption("🖼️ 无截图")

                with c2:
                    st.markdown(
                        f'<span style="font-size: 2.4em; font-weight: bold;">Step {step_num}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='font-size:32px'><b>Action:</b> {step['action']}</div>",
                        unsafe_allow_html=True,
                    )
                    if "timestamp" in step:
                        st.caption(f"🕒 {step['timestamp']}")

                st.divider()

    st.sidebar.divider()
    st.sidebar.caption(f"总计批次数: {len(batches)}")
