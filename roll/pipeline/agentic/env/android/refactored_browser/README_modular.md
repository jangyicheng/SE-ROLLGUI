# Android Trajectory Browser（模块化版）

不修改原有 `app.py`，新增入口 `app_modular.py`。

## 运行

```bash
cd ~/HDD_POOL/ROLL/roll/pipeline/agentic/env/android
conda activate roll
streamlit run app_modular.py
```

## 子集扩展

- 内置子集在 `refactored_browser/config/task_subsets.py`
- 外部扩展子集在 `task_subsets.json`
- 选择 “全部任务” 时不过滤；其他子集会同时作用于轨迹浏览和统计分析

## 模块说明

- `refactored_browser/config/task_subsets.py`：任务子集配置和加载
- `refactored_browser/services/stats_service.py`：统计数据加载/过滤/指标计算
- `refactored_browser/services/traj_service.py`：轨迹扫描、排序和过滤
- `refactored_browser/utils/action_overlay.py`：动作点位解析和截图叠加
- `refactored_browser/views/stats_view.py`：统计页面渲染
- `refactored_browser/views/traj_view.py`：轨迹页面渲染
