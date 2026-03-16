import streamlit as st
import json
import time
from pathlib import Path
from PIL import Image

# cd ~/HDD_POOL/ROLL/roll/pipeline/agentic/env/android && conda activate roll && streamlit run app.py
# ssh -L 8501:localhost:8501 超算
# --- 配置 ---
TRAJ_ROOT = Path("../../../../../trajectories")

st.set_page_config(page_title="Android Trajectory Browser", layout="wide")
st.title("📱 Android 任务轨迹可视化管理器")

# 检查根目录
if not TRAJ_ROOT.exists():
    st.error(f"未找到轨迹根目录: {TRAJ_ROOT.absolute()}")
    st.stop()

# --- 侧边栏：多级任务选择 ---
st.sidebar.header("📂 轨迹筛选")

# 1. 选择批次 (例如 2026-03-15_1658)
batches = sorted([d for d in TRAJ_ROOT.iterdir() if d.is_dir()], reverse=True)
if not batches:
    st.info("暂无轨迹记录")
    st.stop()

selected_batch = st.sidebar.selectbox("1. 选择运行批次", batches, format_func=lambda x: x.name)

# 2. 选择任务与尝试次数
# 遍历当前批次下的 任务/时间戳 结构
all_attempts = []
if selected_batch:
    for task_dir in selected_batch.iterdir():
        if task_dir.is_dir():
            for attempt_dir in task_dir.iterdir():
                if attempt_dir.is_dir():
                    all_attempts.append(attempt_dir)

# 按时间倒序排列尝试记录
all_attempts.sort(key=lambda x: x.name, reverse=True)

selected_attempt = st.sidebar.selectbox(
    "2. 选择具体任务记录",
    all_attempts,
    format_func=lambda x: f"{x.parent.name} ({x.name})"
)

# --- 主界面：展示逻辑 ---
if selected_attempt:
    meta_path = selected_attempt / "meta.json"
    result_path = selected_attempt / "result.json"
    steps_path = selected_attempt / "steps.jsonl"

    # 1. 顶部状态栏
    st.caption(f"当前路径: `{selected_attempt}`")
    
    col_meta, col_res = st.columns([2, 1])
    
    with col_meta:
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.markdown(f"### 🎯 目标: {meta.get('goal', 'N/A')}")
            st.write(f"**任务 ID:** `{meta.get('name', 'N/A')}`")
        else:
            st.warning("缺少 meta.json")

    with col_res:
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                res = json.load(f)
            color = "green" if res.get("success") else "red"
            st.markdown(f"### 结果: :{color}[{'成功' if res.get('success') else '失败'}]")
            st.write(f"步数: {res.get('steps')} | 耗时: {res.get('time', 0):.1f}s")
        else:
            st.info("任务正在进行中或未生成结果文件")

    st.divider()

    # 2. 轨迹详情
    st.subheader("🚀 执行序列")

    if steps_path.exists():
        steps_data = []
        with open(steps_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    steps_data.append(json.loads(line))
        
        # 记录是否存在步骤
        if not steps_data:
            st.write("步骤文件为空")
        
        for step in steps_data:
            with st.container():
                c1, c2 = st.columns([1, 2])
                
                # 图片路径补齐：step_000.png
                img_path = selected_attempt / f"step_{step['step']:03d}.png"
                
                with c1:
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.empty()
                        st.caption("🖼️ 无截图")
                
                with c2:
                    st.markdown(f"**Step {step['step']}**")
                    st.code(f"Action: {step['action']}", language="python")
                    
                    if step.get("ai_response"):
                        with st.expander("查看 AI 推理原文", expanded=True):
                            st.write(step["ai_response"])
                    
                    t = time.strftime('%H:%M:%S', time.localtime(step['timestamp']))
                    st.caption(f"🕒 时间戳: {t}")
                
                st.divider()
    else:
        st.error("未发现 steps.jsonl 文件")

# --- 侧边栏底部统计 ---
st.sidebar.divider()
st.sidebar.caption(f"总计批次数: {len(batches)}")