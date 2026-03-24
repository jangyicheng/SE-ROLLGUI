from __future__ import annotations

from pathlib import Path

import streamlit as st

from refactored_browser.config.task_subsets import load_task_subsets
from refactored_browser.views.stats_view import render_stats_mode
from refactored_browser.views.traj_view import render_traj_mode

# cd ~/HDD_POOL/ROLL/roll/pipeline/agentic/env/android && conda activate roll && streamlit run app_modular.py
# ssh -L 8501:localhost:8501 超算

TRAJ_ROOT = Path("../../../../../trajectories")
SUBSET_CONFIG_PATH = Path(__file__).parent / "task_subsets.json"

st.set_page_config(page_title="Android Trajectory Browser", layout="wide")
st.title("📱 Android 任务轨迹可视化管理器")

if not TRAJ_ROOT.exists():
    st.error(f"未找到轨迹根目录: {TRAJ_ROOT.absolute()}")
    st.stop()

subsets = load_task_subsets(SUBSET_CONFIG_PATH)
subset_name = st.sidebar.selectbox("任务子集", list(subsets.keys()))
subset_tasks = subsets[subset_name]

mode = st.sidebar.radio("工作模式", ["📱 轨迹浏览", "📈 统计分析"])
if mode == "📈 统计分析":
    render_stats_mode(TRAJ_ROOT, subset_name, subset_tasks)
else:
    render_traj_mode(TRAJ_ROOT, subset_name, subset_tasks)
