import streamlit as st
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import altair as alt
import re
# cd ~/HDD_POOL/ROLL/roll/pipeline/agentic/env/android && conda activate roll && streamlit run app.py
# ssh -L 8501:localhost:8501 超算
# ssh -L 8501:localhost:8501 hitsz_xdeng_2@121.46.19.4

TRAJ_ROOT = Path("../../../../../trajectories")
# 仅对给定训练任务子集进行筛选/统计时使用
TRAIN_TASK_LIST = [
    'SimpleSmsReply',
    'MarkorEditNote',
    'ExpenseDeleteMultiple2',
    'SystemWifiTurnOn',
    'FilesDeleteFile',
    'SystemBrightnessMin',
    'SimpleCalendarAddOneEventInTwoWeeks',
    'SystemBrightnessMax',
    'ClockTimerEntry',
    'SystemBluetoothTurnOn',
    'SimpleCalendarAddOneEventRelativeDay',
    'RecipeDeleteMultipleRecipesWithNoise',
    'MarkorDeleteAllNotes',
    'SimpleCalendarAddOneEventTomorrow',
    'SimpleSmsSendClipboardContent',
    'FilesMoveFile',
    'RecipeDeleteMultipleRecipesWithConstraint',
    'SimpleCalendarAddOneEvent',
    'CameraTakeVideo',
    'MarkorMoveNote',
    'SystemWifiTurnOff',
    'MarkorCreateNoteFromClipboard',
    'SaveCopyOfReceiptTaskEval',
    'RecipeDeleteDuplicateRecipes',
    'TurnOnWifiAndOpenApp',
    'ExpenseDeleteDuplicates2'
]

st.set_page_config(page_title="Android Trajectory Browser", layout="wide")
st.title("📱 Android 任务轨迹可视化管理器")

# =========================
# 🎯 工具函数
# =========================

def rescale_coordinates(point, width, height):
    return [
        round(point[0] / 999 * width),
        round(point[1] / 999 * height)
    ]

def parse_args(raw_args):
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}

def pick_point(args, keys):
    for key in keys:
        if key not in args:
            continue
        value = args[key]
        # [x, y]
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return [float(value[0]), float(value[1])]
            except Exception:
                continue
        # {"x":..., "y":...}
        if isinstance(value, dict) and "x" in value and "y" in value:
            try:
                return [float(value["x"]), float(value["y"])]
            except Exception:
                continue
    return None

def get_image_path_for_step(attempt_dir, step_num):
    # 第0张截图对应 step 1 动作
    idx = max(int(step_num) - 1, 0)
    p = attempt_dir / f"step_{idx:03d}.png"
    # 兼容旧数据命名：如果不存在则回退到 step 同号
    if not p.exists():
        p = attempt_dir / f"step_{int(step_num):03d}.png"
    return p



def extract_action_and_args(step_data):
    # 1) 兼容原始结构: {"action":"click","args":{...}}
    raw_action = step_data.get("action", "")
    raw_args = parse_args(step_data.get("args"))
    if isinstance(raw_action, str) and raw_action.lower() in {"click", "tap", "long_press", "swipe"}:
        return raw_action.lower(), raw_args

    # 2) 兼容你给的结构: action 是大段文本，tool_call 内嵌 JSON
    text = str(raw_action)
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S)
    if not m:
        return "", {}

    try:
        payload = json.loads(m.group(1))
    except Exception:
        return "", {}

    arguments = payload.get("arguments", {}) if isinstance(payload, dict) else {}
    action = str(arguments.get("action", "")).lower()
    return action, arguments if isinstance(arguments, dict) else {}

def draw_action_overlay(img_path, step_data):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    action, args = extract_action_and_args(step_data)
    p1 = pick_point(args, ["coordinate", "coord", "position", "start", "from"])
    p2 = pick_point(args, ["coordinate2", "coord2", "end", "to"])

    if action in ["click", "tap"] and p1 is not None:
        x, y = rescale_coordinates(p1, w, h)
        r = 10
        draw.ellipse((x-r, y-r, x+r, y+r), fill="red")

    elif action == "long_press" and p1 is not None:
        x, y = rescale_coordinates(p1, w, h)
        r = 15
        draw.ellipse((x-r, y-r, x+r, y+r), outline="yellow", width=4)

    elif action == "swipe" and p1 is not None and p2 is not None:
        x1, y1 = rescale_coordinates(p1, w, h)
        x2, y2 = rescale_coordinates(p2, w, h)
        draw.line((x1, y1, x2, y2), fill="blue", width=5)
        draw.ellipse((x2-6, y2-6, x2+6, y2+6), fill="blue")

    return img

def is_stats_json_file(p: Path) -> bool:
    if p.name in {"meta.json", "result.json"} or p.name.endswith(".jsonl"):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and "tasks" in data and "global_stats" in data
    except Exception:
        return False

def discover_stats_files(traj_root: Path):
    # 优先批次目录下“同名json”
    preferred = []
    for batch_dir in traj_root.iterdir():
        if batch_dir.is_dir():
            p = batch_dir / f"{batch_dir.name}.json"
            if p.exists() and is_stats_json_file(p):
                preferred.append(p)

    # 兜底：递归扫描所有符合格式的统计json
    others = []
    for p in traj_root.rglob("*.json"):
        if p in preferred:
            continue
        if is_stats_json_file(p):
            others.append(p)

    # 去重并按时间倒序
    all_files = list({x.resolve(): x for x in (preferred + others)}.values())
    all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return all_files

def load_stats_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def tasks_to_df(stats_data: dict) -> pd.DataFrame:
    rows = []
    for task_name, info in stats_data.get("tasks", {}).items():
        if not isinstance(info, dict):
            continue
        rows.append({
            "task": task_name,
            "complete_attempts": info.get("complete_attempts", 0),
            "success_count": info.get("success_count", 0),
            "failure_count": info.get("failure_count", 0),
            "average_steps": info.get("average_steps", 0.0),
            "average_time": info.get("average_time", 0.0),
            "average_success_rate": info.get("average_success_rate", 0.0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["complete_attempts", "success_count", "failure_count", "average_steps", "average_time", "average_success_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["average_success_rate"] = df["average_success_rate"].clip(0, 1)
    return df

def make_hist_df(series: pd.Series, bins: int, x_name: str = "bin", y_name: str = "count"):
    clean = series.dropna()
    if clean.empty:
        return pd.DataFrame(columns=[x_name, y_name, "_sort"])

    counts, edges = np.histogram(clean, bins=bins)
    labels = [f"{edges[i]:.2f}~{edges[i+1]:.2f}" for i in range(len(edges) - 1)]

    return pd.DataFrame({
        x_name: labels,
        y_name: counts,
        "_sort": edges[:-1],   # 仅用于排序，不展示
    })

def plot_sorted_hist(hist_df: pd.DataFrame, x_col: str, y_col: str):
    chart = (
        alt.Chart(hist_df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{x_col}:N",
                sort=alt.SortField(field="_sort", order="ascending"),
                title=x_col,
            ),
            y=alt.Y(f"{y_col}:Q", title=y_col),
            tooltip=[
                alt.Tooltip(f"{x_col}:N", title=x_col),
                alt.Tooltip(f"{y_col}:Q", title=y_col),
            ],
        )
    )
    st.altair_chart(chart, use_container_width=True)
    
def render_stats_mode(traj_root: Path):
    st.header("📈 轨迹统计可视化")

    stats_files = discover_stats_files(traj_root)
    if not stats_files:
        st.warning("未找到包含 tasks/global_stats 的统计JSON文件")
        return

    selected_stats = st.sidebar.selectbox(
        "统计文件",
        stats_files,
        format_func=lambda p: f"{p.parent.name}/{p.name}"
    )

    stats_data = load_stats_json(selected_stats)
    df = tasks_to_df(stats_data)
    gs = stats_data.get("global_stats", {})
    # Sidebar: 是否仅统计训练任务子集
    only_train = st.sidebar.checkbox("仅统计 TRAIN_TASK_LIST 子集", value=False)
    if only_train:
        df = df[df["task"].isin(TRAIN_TASK_LIST)].copy()

    st.caption(f"当前统计文件：{selected_stats}")

    # 计算展示用的统计指标：如果启用 only_train 则基于过滤后的 df 重新计算
    if only_train:
        if not df.empty:
            total_complete = int(df["complete_attempts"].sum())
            total_success = int(df["success_count"].sum())
            global_success_rate = (total_success / total_complete) if total_complete > 0 else 0.0
            avg_success_rate = float(df["average_success_rate"].mean())
            total_task_types = int(len(df))
        else:
            total_complete = 0
            total_success = 0
            global_success_rate = 0.0
            avg_success_rate = 0.0
            total_task_types = 0
    else:
        total_complete = int(gs.get("total_complete_count", 0))
        total_success = int(gs.get("total_success_count", 0))
        global_success_rate = float(gs.get("global_success_rate", 0))
        avg_success_rate = float(gs.get("average_success_rate", 0))
        total_task_types = int(gs.get("total_task_types", 0))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总完成数", total_complete)
    c2.metric("总成功数", total_success)
    c3.metric("全局成功率", f"{global_success_rate*100:.1f}%")
    c4.metric("任务平均成功率", f"{avg_success_rate*100:.1f}%")
    c5.metric("任务类型数", total_task_types)

    if df.empty:
        st.info("统计文件中 tasks 为空")
        return

    # 1) 成功率分布
    st.subheader("1. 任务成功率分布")
    success_hist = make_hist_df(df["average_success_rate"], bins=20, x_name="成功率区间", y_name="任务数")
    st.bar_chart(success_hist.set_index("成功率区间"))

    # 2) 步数和耗时分布
    st.subheader("2. 完成步数与耗时分布")
    col_steps, col_time = st.columns(2)

    with col_steps:
        step_hist = make_hist_df(df["average_steps"], bins=15, x_name="平均步数区间", y_name="任务数")
        plot_sorted_hist(step_hist, "平均步数区间", "任务数")

    with col_time:
        time_hist = make_hist_df(df["average_time"], bins=15, x_name="平均耗时区间(秒)", y_name="任务数")
        plot_sorted_hist(time_hist, "平均耗时区间(秒)", "任务数")
    
    # 3) 成功率最高/最低N个任务
    st.subheader("3. 成功率最高与最低的N个任务")
    # 仅包含有完成尝试的任务
    ranked = df[df["complete_attempts"] > 0].copy()
    # 稳定排序：先按成功率降序，再按任务名升序（相同成功率按音序）
    ranked.sort_values(by=["average_success_rate", "task"], ascending=[False, True], inplace=True)
    N = 40
    topN = ranked.head(N).copy()
    bottomN = ranked.tail(N).copy()
    # bottomN 我们希望按成功率从低到高展示，并在相同成功率下按任务名升序
    bottomN.sort_values(by=["average_success_rate", "task"], ascending=[True, True], inplace=True)

    # 为确保图表严格按照“难度”顺序绘制，创建显式排序字段
    topN = topN.reset_index(drop=True)
    topN["_order"] = range(len(topN))
    bottomN = bottomN.reset_index(drop=True)
    bottomN["_order"] = range(len(bottomN))

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"**Top {N}**")
        chart_top = (
            alt.Chart(topN)
            .mark_bar()
            .encode(
                x=alt.X("task:N", sort=alt.SortField(field="_order", order="ascending"), title="任务"),
                y=alt.Y("average_success_rate:Q", title="平均成功率"),
                tooltip=[alt.Tooltip("task:N", title="任务"), alt.Tooltip("average_success_rate:Q", title="平均成功率")]
            )
        )
        st.altair_chart(chart_top, use_container_width=True)
        # 表格：按成功率降序、相同成功率按任务名升序（已由 ranked 保证）
        st.dataframe(
            topN[["task", "average_success_rate", "complete_attempts", "average_steps", "average_time"]],
            use_container_width=True
        )

    with t2:
        st.markdown(f"**Bottom {N}**")
        chart_bottom = (
            alt.Chart(bottomN)
            .mark_bar()
            .encode(
                x=alt.X("task:N", sort=alt.SortField(field="_order", order="ascending"), title="任务"),
                y=alt.Y("average_success_rate:Q", title="平均成功率"),
                tooltip=[alt.Tooltip("task:N", title="任务"), alt.Tooltip("average_success_rate:Q", title="平均成功率")]
            )
        )
        st.altair_chart(chart_bottom, use_container_width=True)
        st.dataframe(
            bottomN[["task", "average_success_rate", "complete_attempts", "average_steps", "average_time"]],
            use_container_width=True
        )

# =========================
# 📂 目录检查
# =========================
if not TRAJ_ROOT.exists():
    st.error(f"未找到轨迹根目录: {TRAJ_ROOT.absolute()}")
    st.stop()
    
mode = st.sidebar.radio("工作模式", ["📱 轨迹浏览", "📈 统计分析"])
if mode == "📈 统计分析":
    render_stats_mode(TRAJ_ROOT)
    st.stop()
# =========================
# 📂 Sidebar
# =========================
st.sidebar.header("📂 轨迹筛选")

batches = sorted([d for d in TRAJ_ROOT.iterdir() if d.is_dir()], reverse=True)
if not batches:
    st.info("暂无轨迹记录")
    st.stop()

selected_batch = st.sidebar.selectbox("1. 选择运行批次", batches, format_func=lambda x: x.name)
# Sidebar: 仅显示训练任务子集轨迹
only_train_attempts = st.sidebar.checkbox("仅显示 TRAIN_TASK_LIST 子集轨迹", value=False)

all_attempts = []
for task_dir in selected_batch.iterdir():
    if not task_dir.is_dir():
        continue
    # 可选：只保留训练子集的任务
    if only_train_attempts and task_dir.name not in TRAIN_TASK_LIST:
        continue
    for attempt_dir in task_dir.iterdir():
        if attempt_dir.is_dir():
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

            all_attempts.append({
                "path": attempt_dir,
                "finished": finished,
                "success": success
            })

sort_mode = st.sidebar.selectbox(
    "2. 排序方式",
    ["按时间（新→旧）", "按任务名（A→Z）"]
)
if sort_mode == "按时间（新→旧）":
    all_attempts.sort(key=lambda x: x["path"].stat().st_mtime, reverse=True)
else:
    all_attempts.sort(key=lambda x: (x["path"].parent.name.lower(), x["path"].name.lower()))

filter_mode = st.sidebar.selectbox(
    "3. 轨迹筛选",
    ["只显示成功轨迹", "只显示失败轨迹", "显示所有完成的轨迹", "显示所有轨迹"]
)

if filter_mode == "只显示成功轨迹":
    filtered_attempts = [x["path"] for x in all_attempts if x["finished"] and x["success"] is True]
elif filter_mode == "只显示失败轨迹":
    filtered_attempts = [x["path"] for x in all_attempts if x["finished"] and x["success"] is False]
elif filter_mode == "显示所有完成的轨迹":
    filtered_attempts = [x["path"] for x in all_attempts if x["finished"]]
else:  # 显示所有轨迹
    filtered_attempts = [x["path"] for x in all_attempts]

if not filtered_attempts:
    st.info("当前筛选条件下暂无轨迹记录")
    st.stop()

selected_attempt = st.sidebar.selectbox(
    "4. 选择具体任务记录",
    filtered_attempts,
    format_func=lambda x: f"{x.parent.name} ({x.name})"
)

view_mode = st.sidebar.radio(
    "视图模式",
    ["📜 详细步骤", "📊 缩略图视图"]
)

# =========================
# 🧠 主界面
# =========================
if selected_attempt:
    meta_path = selected_attempt / "meta.json"
    result_path = selected_attempt / "result.json"
    steps_path = selected_attempt / "steps.jsonl"

    screenshot_count = len(list(selected_attempt.glob("step_*.png")))

    st.markdown(f"<h3>当前路径：{selected_attempt}</h3>",unsafe_allow_html=True)

    col_meta, col_res = st.columns([2, 1])

    with col_meta:
        if meta_path.exists():
            meta = json.load(open(meta_path, "r", encoding="utf-8"))

            st.markdown(
                f"<h2>🎯 目标: {meta.get('goal', 'N/A')}</h2>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<h2>任务 ID: <code>{meta.get('name', 'N/A')}</code></h2>",
                unsafe_allow_html=True
            )

    with col_res:
        if result_path.exists():
            res = json.load(open(result_path, "r", encoding="utf-8"))
            success = res.get("success")
            color = "green" if success else "red"
            result_text = "成功" if success else "失败"
            steps_text = res.get("steps")
            time_text = f"{res.get('time', 0):.1f}s"
            completion_reason = res.get("termination_reason", "未知")
        else:
            # 无结果文件 -> 视为未完成，步数用截图数量
            color = "orange"
            result_text = "未完成"
            steps_text = screenshot_count
            time_text = "-"
            completion_reason = "-"

        st.markdown(
            f"<h2>结果: <span style='color:{color}'>{result_text}</span></h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<h2>步数: {steps_text}<br>耗时: {time_text}</h2>",
            unsafe_allow_html=True
        )
        
        
        st.markdown(
            f"<h2>结束原因: {completion_reason}</h2>",
            unsafe_allow_html=True
        )

    st.divider()

    steps_data = []
    if steps_path.exists():
        with open(steps_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    steps_data.append(json.loads(line))
        # steps_data.pop(0)  # 去掉第一行的初始状态

    # 读取后
    if steps_data and steps_data[0].get("step") == 0:
        steps_data.pop(0)  # 保持 step 从 1 开始，与 step-1 截图映射一致

    last_step = steps_data[-1]["step"] if steps_data else 0
    steps_data.append({
        "step": int(last_step) + 1,      # 展示用步号
        "action": "Terminate",
        "timestamp": "None",
        "image_step": int(last_step) + 1,    # 终态复用最后一张截图
        "is_terminal": True,
    })
        
    if view_mode == "📊 缩略图视图":
        st.subheader("📊 截图总览")
        cols = st.columns(4)

        rendered_images = set()
        thumb_items = []

        for step in steps_data:
            step_num = step.get("step")
            if not isinstance(step_num, int):
                continue
            if step_num <= 0:
                continue

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

        for i, (step_num, step, img_path) in enumerate(thumb_items):
            with cols[i % 4]:
                img = draw_action_overlay(img_path, step)
                st.image(img, use_container_width=True)
                st.caption(f"Step {step_num}")

    else:
        st.subheader("🚀 执行序列")

        for step in steps_data:
            step_num = step.get("step")
            if not isinstance(step_num, int):
                continue
            if step_num <= 0:
                continue
            with st.container():
                c1, c2 = st.columns([1, 2])
                image_step = step.get("image_step", step_num)
                img_path = get_image_path_for_step(selected_attempt, image_step)
                with c1:
                    if img_path.exists():
                        img = draw_action_overlay(img_path, step)
                        st.image(img, use_container_width=True)
                    else:
                        st.caption("🖼️ 无截图")

                with c2:
                    st.markdown(
                        f'<span style="font-size: 2.4em; font-weight: bold;">Step {step["step"]}</span>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"<div style='font-size:32px'><b>Action:</b> {step['action']}</div>",
                        unsafe_allow_html=True
                    )


                    if "timestamp" in step:
                        # t = time.strftime('%H:%M:%S', time.localtime(step['timestamp']))
                        st.caption(f"🕒 {step['timestamp']}")

                st.divider()

st.sidebar.divider()
st.sidebar.caption(f"总计批次数: {len(batches)}")