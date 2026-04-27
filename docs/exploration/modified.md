# 文件修改清单

本文档记录探索模块实现过程中所有修改的文件及其改动位置。

---

## 新建文件

| 文件路径 | 描述 |
|---------|------|
| `roll/pipeline/agentic/env/android/exploration/__init__.py` | 探索模块入口，导出核心类 |
| `roll/pipeline/agentic/env/android/exploration/explorer.py` | AndroidWorld/MobileWorld 自由探索器 |
| `roll/pipeline/agentic/env/android/exploration/params_manager.py` | Params 持久化管理器 |
| `roll/pipeline/agentic/env/android/exploration/task_initializer.py` | 任务初始化探索器 |
| `roll/pipeline/agentic/env/android/exploration/trajectory_formatter.py` | 轨迹格式化工具 |
| `roll/pipeline/agentic/env/android/exploration/templates/androidworld_exploration.json` | AndroidWorld 探索模板 |
| `roll/pipeline/agentic/env/android/exploration/scripts/run_exploration.py` | 探索 CLI 脚本 |
| `roll/pipeline/agentic/env/android/exploration/scripts/run_task_init.py` | 任务初始化 CLI 脚本 |
| `roll/pipeline/agentic/env/android/exploration/scripts/run_exploration.sh` | 探索 Shell 脚本 |
| `roll/pipeline/agentic/env/android/exploration/scripts/run_context_review.sh` | 上下文审查 Shell 脚本 |
| `roll/pipeline/agentic/env/android/exploration/Readme.md` | 探索模块中文 README |
| `docs/exploration/modified.md` | 本文档 |

---

## 修改文件

### 1. `roll/pipeline/agentic/env/android/remote_android.py`

**改动类型**: 新增方法

**改动位置**: 文件末尾

**新增方法**:

#### `explore_reset()` (行 359-422)
```python
def explore_reset(
    self,
    go_home: bool = True,
    seed: int = 42,
    target_task: str | None = None,
    exploration_id: str | None = None,
):
```

用于探索模式的 reset，与标准 `reset()` 的区别：
- 不与 task_manager 交互
- 不保存到标准轨迹目录
- 使用探索专用输出目录
- 可以使用任意任务名进行自由探索

#### `explore_step()` (行 424-494)
```python
def explore_step(self, action: str | dict, disable_judge: bool = True):
```

用于探索模式的 step，与标准 `step()` 的区别：
- 不评估任务成功（is_successful）
- 不调用 task_manager 完成通知
- 专注于记录课程生成的轨迹

#### `get_current_app()` (行 496-515)
```python
def get_current_app(self) -> str | None:
```

获取当前焦点应用的名称，用于探索过程中追踪发现的 App。

#### `explore_loop()` (行 517-620)
```python
def explore_loop(
    self,
    max_steps: int = 50,
    action_generator=None,
    save_screenshots: bool = True,
    output_dir: str = "./exploration_output",
) -> dict:
```

便捷方法，组合 `explore_reset` 和 `explore_step` 进行完整探索执行。

---

### 2. `roll/pipeline/agentic/env/android/mobile/curriculum_task_generator.py`

**改动类型**: 新增方法

**改动位置**: `MobileSpecificTaskGenerator` 类

**新增方法**:

#### `load_exploration_data()` (行 973-999)
```python
def load_exploration_data(self, exploration_dir: str) -> Dict[str, Any]:
```

从探索输出目录加载探索数据。

#### `load_init_data()` (行 1001-1028)
```python
def load_init_data(self, init_output_dir: str) -> Dict[str, Dict[str, Any]]:
```

从初始化输出目录加载任务初始化结果。

#### `format_context_review_for_prompt()` (行 1030-1067)
```python
def format_context_review_for_prompt(
    self,
    init_results: Dict[str, Dict[str, Any]],
    max_tasks: int = 10,
) -> str:
```

将初始化结果格式化为上下文审查文本。

#### `generate_tasks_from_exploration()` (行 1069-1175)
```python
def generate_tasks_from_exploration(
    self,
    exploration_dir: str,
    init_output_dir: str,
    output_dir: str,
    index_dir: str,
    context_filename: str = "exploration_context",
    existing_task_files: Optional[List[str]] = None,
    feedback_file: Optional[str] = None,
    iteration: int = 0,
) -> List[str]:
```

从探索和初始化数据生成课程任务。

#### `get_exploration_screenshots()` (行 1177-1196)
```python
def get_exploration_screenshots(self, exploration_dir: str) -> List[Dict[str, Any]]:
```

从探索输出目录获取探索截图。

---

## 已存在且被复用的文件

### `androidworld-server/android_backend.py`

**已有方法**: `reset_with_params()` (行 260-276)

此方法在 `androidworld-server/android_backend.py` 中已存在，用于确定性重放初始化任务。探索模块直接调用此方法。

```python
def reset_with_params(self, task_name: str, params: dict) -> dict:
    """使用预生成 params 确定性重放初始化任务"""
```

### `androidworld-server/android_server_fullyasync.py`

**已有端点**: `/reset_with_params` (行 190-225)

此端点已在 `android_server_fullyasync.py` 中存在，暴露 `reset_with_params` 功能给 HTTP 客户端。

---

## 数据流

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1: 环境探索 (Explorer)                                     │
│   explorer.py: AndroidWorldExplorer / MobileWorldExplorer        │
│   └─► exploration_output/{exploration_id}/                       │
│       ├─ exploration_result.json                                  │
│       └─ screenshots/                                           │
│           └─ trajectory.jsonl                                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Phase 2: 任务初始化 (Task Initializer)                           │
│   task_initializer.py: AndroidWorldTaskInitializer              │
│   └─► init_output/{task_name}/instance_{id}/                    │
│       ├─ task_init_result.json                                  │
│       └─ init_screenshot.png                                    │
│                                                                  │
│   params_manager.py: AndroidWorldParamsManager                   │
│   └─► init_output/params/androidworld/                          │
│       └─ {task_name}_{instance}_params.pkl                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Phase 3: 课程生成 (Curriculum Generator)                         │
│   curriculum_task_generator.py: 新增方法                         │
│   ├─ load_exploration_data()                                    │
│   ├─ load_init_data()                                           │
│   └─ generate_tasks_from_exploration()                          │
│   └─► generated_tasks/{task_id}.json                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 修改优先级

| 优先级 | 改动 | 影响范围 |
|--------|------|----------|
| P0 | `explorer.py`, `params_manager.py`, `task_initializer.py` | 核心新功能 |
| P0 | `remote_android.py` 新增探索方法 | 环境集成 |
| P1 | `curriculum_task_generator.py` 新增方法 | 课程生成集成 |
| P2 | CLI 脚本和模板文件 | 使用便捷性 |

---

*最后更新: 2026-04-26*
