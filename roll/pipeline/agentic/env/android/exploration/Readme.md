# AndroidWorld/MobileWorld 环境探索模块

本模块实现 AndroidWorld 和 MobileWorld 环境的自主探索功能，是 ROLL-GUI 自进化训练循环的关键组成部分。

## 架构概述

本模块遵循自进化循环设计（详见 `docs/self-evolve_zh.md`）：

```
┌──────────────────────────────────────────────────────────────────┐
│  环境探索 (Exploration)                                          │
│    └─► 探索轨迹 (actions + screenshots)                          │
│    └─► 发现的 App 列表、UI 状态                                │
│                                                                  │
│  任务初始化 (Task Initialization)                                │
│    └─► Params 持久化 + App 快照恢复                            │
│    └─► 替换原 Context Review                                    │
│                                                                  │
│  课程任务生成 (Curriculum Task Generation)                        │
│    └─► 基于探索动作序列 + 初始化结果生成任务                   │
│                                                                  │
│  训练 (Training)                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
exploration/
├── __init__.py                    # 模块入口
├── explorer.py                    # 自由探索器
├── params_manager.py              # Params 持久化管理
├── task_initializer.py            # 任务初始化探索器
├── trajectory_formatter.py        # 轨迹格式化工具
├── templates/
│   └── androidworld_exploration.json   # 探索模板
└── scripts/
    ├── run_exploration.py        # 探索 CLI 脚本
    ├── run_task_init.py           # 任务初始化 CLI 脚本
    ├── run_exploration.sh         # 探索 Shell 脚本
    └── run_context_review.sh      # 上下文审查 Shell 脚本
```

## 核心组件

### 1. Explorer（探索器）

`explorer.py` 实现了 `AndroidWorldExplorer` 和 `MobileWorldExplorer` 类，负责在移动环境中进行自由探索。

**主要功能**：
- 复用现有 `RemoteAndroidEnv` 的 HTTP API 循环
- 记录动作轨迹和截图
- 追踪发现的 App 和动作类型
- 支持自定义探索指令

**关键方法**：
```python
# 创建探索器
explorer = AndroidWorldExplorer(
    server_url="http://localhost:8000",
    max_steps=50,
    output_dir="./exploration_output",
)

# 运行探索
result = explorer.run()

# 保存结果
explorer.save_trajectory()
result_path = explorer.save_result()
```

### 2. ParamsManager（参数管理器）

`params_manager.py` 实现了 `AndroidWorldParamsManager` 和 `MobileWorldParamsManager` 类，负责确定性参数生成和持久化。

**主要功能**：
- 通过 sha256 派生确定性 seed
- 持久化 params 到 pickle 文件
- 构建 params 索引
- 兼容 OpenMobile 的 params 格式

**关键方法**：
```python
# 创建管理器
params_manager = AndroidWorldParamsManager(
    params_dir="./params",
    task_random_seed=42,
)

# 生成并保存 params
params, params_path = params_manager.generate_and_save(
    task_name="ContactsAddContact",
    instance_id=0,
)

# 加载已保存的 params
loaded_params = params_manager.load_params(
    task_name="ContactsAddContact",
    instance_id=0,
)

# 构建 params 索引
index = params_manager.build_params_index()
```

### 3. TaskInitializer（任务初始化器）

`task_initializer.py` 实现了 `AndroidWorldTaskInitializer` 和 `MobileWorldTaskInitializer` 类，替代原 Context Review 功能。

**主要功能**：
- 对任务池中每个任务执行初始化探索
- 调用 `reset_with_params` 端点进行确定性初始化
- 获取初始截图验证任务可成功初始化
- 可选：执行少量验证步确保 App 功能正常

**关键方法**：
```python
# 创建初始化器
initializer = AndroidWorldTaskInitializer(
    server_url="http://localhost:8000",
    params_manager=params_manager,
    task_pool=TRAIN_TASK_LIST,
    output_dir="./init_output",
)

# 运行初始化
results = initializer.run(num_instances=1)

# 加载已保存的结果
loaded_results = initializer.load_results()
```

### 4. TrajectoryFormatter（轨迹格式化器）

`trajectory_formatter.py` 提供轨迹格式化工具，用于将探索数据转换为课程生成器可用的格式。

**主要功能**：
- 加载和解析探索结果
- 提取动作序列和截图
- 格式化课程生成器输入
- 生成上下文审查文本

**关键方法**：
```python
# 创建格式化器
formatter = TrajectoryFormatter(
    exploration_dir="./exploration_output",
    init_output_dir="./init_output",
)

# 格式化探索数据
formatted = formatter.format_for_curriculum(
    exploration_result,
    trajectory,
    init_results,
)

# 便捷函数
formatted = format_trajectory_for_curriculum(
    exploration_dir="./exploration_output",
)
```

## 使用方法

### 快速开始

#### 1. 环境探索

**Python API**：
```python
from roll.pipeline.agentic.env.android.exploration import AndroidWorldExplorer

explorer = AndroidWorldExplorer(
    server_url="http://localhost:8000",
    max_steps=50,
    console_port=5554,
)
result = explorer.run()
```

**CLI 脚本**：
```bash
# 使用 Python
python -m roll.pipeline.agentic.env.android.exploration.scripts.run_exploration \
    --server_url http://localhost:8000 \
    --max_steps 50 \
    --output_dir ./exploration_output

# 使用 Shell 脚本
bash roll/pipeline/agentic/env/android/exploration/scripts/run_exploration.sh \
    --max_steps 50 \
    --output_dir ./exploration_output
```

#### 2. 任务初始化

**Python API**：
```python
from roll.pipeline.agentic.env.android.exploration import (
    AndroidWorldTaskInitializer,
    AndroidWorldParamsManager,
)
from roll.pipeline.agentic.env.android.tasks import TRAIN_TASK_LIST

params_manager = AndroidWorldParamsManager()
initializer = AndroidWorldTaskInitializer(
    server_url="http://localhost:8000",
    params_manager=params_manager,
    task_pool=TRAIN_TASK_LIST,
)
results = initializer.run(num_instances=1)
```

**CLI 脚本**：
```bash
python -m roll.pipeline.agentic.env.android.exploration.scripts.run_task_init \
    --server_url http://localhost:8000 \
    --task_pool train \
    --output_dir ./init_output \
    --num_instances 1

# 使用 Shell 脚本
bash roll/pipeline/agentic/env/android/exploration/scripts/run_context_review.sh \
    --task_pool train \
    --num_instances 1
```

#### 3. 课程任务生成

```python
from roll.pipeline.agentic.env.android.mobile.curriculum_task_generator import MobileSpecificTaskGenerator

generator = MobileSpecificTaskGenerator(
    openai_api_key="your-api-key",
    model="gpt-4o",
)

# 从探索数据生成任务
generated_files = generator.generate_tasks_from_exploration(
    exploration_dir="./exploration_output",
    init_output_dir="./init_output",
    output_dir="./generated_tasks",
    index_dir="./task_index",
)
```

## 与 OpenMobile 的关系

本模块参考 OpenMobile 的确定性重放机制：

| 机制 | OpenMobile | ROLL-GUI Exploration |
|------|-----------|------------------------|
| Seed 派生 | sha256 hash | 完全一致 |
| Params 存储 | pickle 文件 | 完全一致 |
| App 快照恢复 | `restore_snapshot()` | `_init_task()` 中自动调用 |
| Rollout 初始化 | 加载 pickle | `reset_with_params` 端点 |

## 输出数据结构

### 探索结果

```json
{
    "exploration_id": "exp_androidworld_001",
    "timestamp": "2026-04-26T10:00:00Z",
    "environment": "AndroidWorld",
    "max_steps": 50,
    "actual_steps": 47,
    "discovered_apps": ["Contacts", "Settings", "Chrome"],
    "discovered_action_types": ["click", "type", "swipe", "long_press"],
    "trajectory_file": "trajectory.jsonl",
    "trajectory_dir": "trajectory/",
    "screenshots_dir": "screenshots/",
    "success": true
}
```

### 任务初始化结果

```json
{
    "task_name": "ContactsAddContact",
    "instance_id": 0,
    "seed": 12345678,
    "params_path": "./params/ContactsAddContact_0_params.pkl",
    "app_snapshot_restored": ["Contacts"],
    "initialization": {
        "success": true,
        "steps_used": 0
    },
    "init_screenshot": "init_screenshot.png",
    "verification": {
        "enabled": true,
        "steps": 5,
        "all_successful": true
    },
    "timestamp": "2026-04-26T10:00:00Z",
    "blockers": []
}
```

## 集成到训练流程

### 1. 数据准备

生成的课程任务 JSON 包含 `params_path` 字段：

```json
{
    "id": "ContactsAddContact_task_001",
    "task_name": "ContactsAddContact",
    "params_path": "./init_output/params/ContactsAddContact_0_params.pkl",
    "instruction": "Add a new contact named Alice Smith...",
    "source": "self_evolving"
}
```

### 2. Rollout 集成

`remote_android.py` 新增的 `explore_reset()` 和 `explore_step()` 方法可在训练流程中复用探索逻辑。

## 注意事项

1. **服务器依赖**：探索模块依赖 `androidworld-server` 运行的 HTTP API 服务器
2. **Params 兼容性**：生成的 pickle 文件与 OpenMobile 格式兼容
3. **截图存储**：大量探索会产生大量截图，注意磁盘空间管理
4. **随机种子**：使用固定的 `task_random_seed` 确保可重复性

## 扩展阅读

- [自进化模式设计](./docs/self-evolve_zh.md)
- [AndroidWorld 训练框架](./docs/android_mobileworld_roll_qwen_framework.md)
- [AndroidWorld 服务器文档](./androidworld-server/README.md)
- [OpenMobile 任务重置逻辑](./E:/code/GUI/OpenMobile-Code/docs/reset_task.md)

---

*最后更新: 2026-04-26*
