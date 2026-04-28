# 自进化模式（Self-Evolving Mode）

## 一、概述

自进化模式是 ROLL 训练框架（GUI Agent）为 AndroidWorld/MobileWorld 环境新增的一种闭环训练范式。与常规训练不同，自进化模式不依赖环境的内置奖励信号，而是由 **MobileJudge（LLM 裁判）** 提供多维奖励与文本反馈，驱动 **Curriculum Task Generator（课程任务生成器）** 自主生成和更新任务池，实现"探索 → 学习 → 反馈 → 新任务"的闭环迭代。

### 1.1 核心设计约束

- **不修改奖励模块**：`mobilejudge.py` 保持原样，自进化逻辑通过外部 JSON 文件中转与 judge 交互
- **探索数据来源**：课程生成器的初始化数据来自 `exploration/` 模块（`TrajectoryFormatter` 输出）
- **环境快速恢复**：每个生成任务包含 `snapshot` 属性，环境初始化通过该标识符快速恢复
- **零侵入兼容**：所有新增代码路径均受 `self_evolve.enabled = false` 保护，常规训练完全不受影响

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        探索阶段（离线/初始化）                                   │
│  Explorer ──▶ TaskInitializer ──▶ TrajectoryFormatter                        │
│                          │                   │                                 │
│                     params.pkl        formatted_exploration_data             │
│                     init_screenshot         (via JSON)                       │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ consume
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         训练主干（不做核心修改）                                  │
│  AgenticPipeline ──▶ RolloutScheduler ──▶ AndroidStepEnvManager              │
│                                              │                                │
│                                     RemoteMobileEnv                          │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ episode_end signal
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      自进化控制平面（新增）                                      │
│                                                                              │
│  AndroidStepEnvManager.formulate_rollouts()                                 │
│      │                                                                       │
│      │ writes judge_input.json ──▶ SelfEvolveCoordinator.on_episode_end()   │
│      │                               │                                       │
│      │                               │ calls MobileJudge (via evaluate_trajectory)
│      │                               │                                       │
│      │                               ▼                                       │
│      │                         writes judge_result.json                     │
│      │                         returns JudgeEpisodeResult                   │
│      │                         overrides last-step reward                    │
│      │                                                                       │
│  AgenticPipeline.run()                                                       │
│      │ round_end signal                                                      │
│      │                                                                       │
│      ▼                                                                       │
│  SelfEvolveCoordinator.on_round_end()                                        │
│      │                                                                       │
│      ├── FeedbackAggregator.build_round_feedback()                           │
│      │       aggregates judge_result.json ──▶ round_feedback.json           │
│      │                                                                       │
│      ├── CurriculumTaskGenerator.generate_tasks_from_exploration()           │
│      │       consumes: exploration_data + round_feedback.json                │
│      │       outputs: generated task JSONs ──▶ validate snapshot           │
│      │                                                                       │
│      └── prepare_data.py --self_evolve_mode                                  │
│              generates train.parquet ──▶ consumed by train_dataloader        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心组件

### 3.1 `SelfEvolveCoordinator`

**文件**：`roll/pipeline/agentic/self_evolve_coordinator.py`

自进化系统的中央编排器，提供两个核心钩子：

#### Episode 级钩子：`on_episode_end(episode_artifacts, task_id, episode_id)`

- 由 `AndroidStepEnvManager.formulate_rollouts()` 在 episode 结束时调用
- 组装 `JudgeEpisodeInput`，写入 `judge_input.json`
- 调用 `MobileJudgeEvaluator.evaluate_trajectory()` 获取 judge 结果
- 将 judge 结果写入 `judge_result.json`
- 返回 `JudgeEpisodeResult`（含 reward、success、feedback_summary）
- **关键**：返回的 `judge_reward` 覆盖 `rollout_cache.history[-1]["reward"]`

#### Round 级钩子：`on_round_end(exploration_dirs, context_template_dir, ...)`

- 由 `AgenticPipeline.run()` 在每个 round 边界调用
- 调用 `build_round_feedback()` 聚合本轮所有 `judge_result.json`
- 调用 `CurriculumTaskGenerator.generate_tasks_from_exploration()` 生成新任务
- 调用 `prepare_data.py --self_evolve_mode` 生成新 parquet
- 返回 `RoundUpdateResult`

### 3.2 `SelfEvolveIO`

**文件**：`roll/pipeline/agentic/env/android/mobile/self_evolve_io.py`

负责 JSON 文件 I/O 和 exploration 数据加载：

| 函数 | 职责 |
|---|---|
| `atomic_write_json(path, data)` | 原子写入（写 `.tmp` 后 `os.rename`） |
| `write_judge_input()` / `read_judge_input()` | Episode judge 输入输出 |
| `write_judge_result()` / `read_judge_result()` | Episode judge 结果读写 |
| `scan_judge_results(round_dir)` | 扫描目录获取所有 judge 结果 |
| `write_round_feedback()` / `read_round_feedback()` | Round 反馈文件读写 |
| `build_exploration_task_context()` | 从 TrajectoryFormatter 输出构建标准化 ExplorationTaskContext |
| `build_exploration_task_context_from_formatted()` | 从已格式化的 TrajectoryFormatter 数据构建 ExplorationTaskContext |

### 3.3 `SelfEvolveFeedback`

**文件**：`roll/pipeline/agentic/env/android/mobile/self_evolve_feedback.py`

```python
def build_round_feedback(
    round_dir: Path,
    output_path: Optional[Path] = None,
    task_instructions: Optional[Dict[str, str]] = None,
    task_snapshots: Optional[Dict[str, str]] = None,
) -> RoundFeedback:
```

- 扫描 `round_dir/episodes/*/judge_result.json`
- 按 `task_id` 分组，计算每任务的 `success_rate`、`avg_reward`
- 提取所有 `failure_reasons` 和 `feedback_strings` 并去重
- 输出 `RoundFeedback`（含 `overall_success_rate`、`overall_avg_reward`、`task_performances`）

### 3.4 `CurriculumTaskGenerator` 扩展

**文件**：`roll/pipeline/agentic/env/android/mobile/curriculum_task_generator.py`

| 新增方法 | 职责 |
|---|---|
| `generate_tasks_from_exploration(exploration_contexts, feedback_file, ...)` | **自进化主入口**：接收 ExplorationTaskContext 列表 + round_feedback.json，调用 LLM 生成任务 |
| `load_round_feedback(feedback_file)` | 加载 `round_feedback.json` |
| `format_feedback_summary_v2(feedback_data)` | 生成含成功率分层（≥80%/30-80%/<30%）和失败原因的分析文本 |
| `load_exploration_data(exploration_dir)` | 从 exploration 输出目录加载数据 |
| `load_init_data(init_output_dir)` | 从 TaskInitializer 输出加载初始化结果 |
| `format_context_review_for_prompt(init_results, max_tasks)` | 将 init 结果格式化为 prompt 文本 |

**关键校验**：`create_task_config_file()` 现在要求任务 JSON 必须包含 `snapshot` 字段，缺失时抛出 `ValueError`。

---

## 四、数据结构

### 4.1 Episode 级数据结构

```python
# JudgeEpisodeInput（写入 judge_input.json）
{
    "task_id": "settings_main_v3_task_abc123",
    "instruction": "打开 Wi-Fi 设置页面并确认当前已看到可用网络列表",
    "snapshot": "settings_main_v3",
    "current_screenshot_paths": ["exp_001/screenshots/step_000.png", ...],
    "final_screenshot_path": "exp_001/screenshots/step_010.png",
    "reference_screenshot_paths": [],
    "reference_instruction": "",
    "action_history": ["tap_settings", "tap_wifi", ...],
    "env_signals": {"env_raw_score": 0.0, "env_raw_success": False}
}

# JudgeEpisodeResult（从 judge_result.json 读取）
{
    "task_id": "...",
    "reward": 1.0,          # MobileJudge 给出的数值奖励
    "success": True,
    "feedback_summary": "目标已达成...",
    "feedback_strings": ["关键点：Wi-Fi 设置页面已打开..."],
    "failure_reasons": [],
    "dimension_scores": {"goal_completion": 1.0, "ui_state_match": 1.0},
    "raw_judge_text": "..."
}
```

### 4.2 Round 级数据结构

```python
# RoundFeedback（round_feedback.json）
{
    "round_id": 3,
    "total_tasks": 5,
    "overall_success_rate": 0.45,
    "overall_avg_reward": 0.62,
    "exploration_data_paths": ["./exploration_output/exp_001", ...],
    "task_performances": [
        {
            "task_id": "settings_wifi_task_001",
            "snapshot": "settings_main_v3",
            "instruction": "打开 Wi-Fi 设置页面...",
            "success_rate": 0.8,
            "avg_reward": 0.85,
            "total_episodes": 10,
            "success_episodes": 8,
            "feedback_strings": ["完成度高", "操作流畅"],
            "failure_reasons": []
        },
        ...
    ]
}
```

### 4.3 生成任务 JSON 格式

```json
{
    "id": "mobile_app.exp_001_task_a1b2c3d4",
    "snapshot": "SettingsMain",
    "instruction": "在设置中打开 Wi-Fi 并确认已看到可用网络列表",
    "source": "self_evolve",
    "params_path": "init_output/exp_001/SettingsMain/instance_0/params.pkl",
    "metadata": {
        "parent_exploration_id": "exp_001",
        "round_id": 3,
        "generator": "curriculum_task_generator"
    }
}
```

---

## 五、目录结构

```
trajectories/
  self_evolve/
    round_0001/
      episodes/
        settings_wifi_task_abc123/
          episode_0/
            judge_input.json       # Episode 评估输入
            judge_result.json     # Episode 评估结果
          episode_1/
            judge_input.json
            judge_result.json
        another_task_xyz/
          ...
      round_feedback.json         # 回合聚合反馈
      generated_tasks/
        mobile/
          mobile_app.exp_001_task_xxx.json  # 新生成的任务
          mobile_app.exp_001_task_yyy.json
      parquet/
        train.parquet             # 供 train_dataloader 消费的 parquet

exploration_output/              # 来自 exploration/ 模块
  exp_001/
    exp_001_result.json
    screenshots/
      step_000_init.png
      step_001.png
    trajectory/
      trajectory.jsonl
```

---

## 六、启用方式

### 6.1 配置文件

在训练 YAML 配置的 `self_evolve` 节点中启用：

```yaml
# 在 agentic pipeline 配置中添加
self_evolve:
  enabled: true
  feedback_root: "./trajectories/self_evolve"
  generated_task_root: "./data/tasks/generated"
  parquet_root: "./data/self_evolve_parquet"
  round_update_interval: 1    # 每 1 轮触发一次任务更新
```

### 6.2 环境管理器配置

```yaml
train_env_manager:
  env_config:
    self_evolve:
      enabled: true
      feedback_root: "./trajectories/self_evolve"
```

### 6.3 探索模块输出配置

确保 `exploration/` 模块已运行，输出路径传入 `on_round_end()`：

```python
self_evolve_coordinator.on_round_end(
    exploration_dirs=["./exploration_output"],
    context_template_dir="./data/tasks/examples/mobile",
)
```

---

## 七、降级与异常处理

| 场景 | 处理方式 |
|---|---|
| `self_evolve.enabled = false` | 所有新增代码路径完全旁路，零影响 |
| Judge 调用失败 | 回退到环境原始 reward，记录 warning，继续训练 |
| `round_feedback.json` 生成失败 | 保持当前任务池，记录 error |
| 任务生成失败 | 回退到上一轮任务池 |
| `params_path` 文件不存在 | 回退到仅用 `snapshot` 字段 reset |
| `snapshot` 字段缺失 | `create_task_config_file()` 抛出 `ValueError`，拒绝生成该任务 |

---

## 十二、vLLM 模型后端配置

自进化模式的探索阶段和课程任务生成阶段均依赖 VLM（视觉语言模型）提供决策和生成能力。框架提供了统一的 `VLMModelFactory`，支持本地 vLLM 服务、OpenAI 云端 API 和 HuggingFace TGI 三种后端。

### 12.1 支持的后端

| 后端 | 说明 | 典型场景 |
|---|---|---|
| `vllm` | 本地 vLLM 服务，OpenAI-compatible API | 自托管，推荐用于探索阶段 |
| `openai` | OpenAI 官方 API 或第三方兼容端点 | 快速验证，有网络条件 |
| `huggingface` | HuggingFace TGI 或 vLLM HF 推理端点 | HF 模型推理 |
| `none` | 不使用模型，随机动作 | 调试/基准测试 |

### 12.2 启动本地 vLLM 服务

```bash
# 示例：启动 GUI-Owl-1.5-8B-Instruct vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model Models/GUI-Owl-1.5-8B-Instruct \
    --served-model-name GUI-Owl-1.5-8B-Instruct \
    --trust-remote-code \
    --dtype half \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096
```

> **注意**：部署时请根据实际 GPU 显存选择合适的模型（7B/14B/30B），确保 `port` 与后续配置中的 `--vllm_base_url` 一致。

### 12.3 探索阶段模型配置

#### Python API

```python
from roll.pipeline.agentic.env.android.exploration.model_client import VLMModelFactory

# 方式一：直接指定后端
model_client = VLMModelFactory.create(
    backend="vllm",
    model_name="Models/GUI-Owl-1.5-8B-Instruct",
    base_url="http://localhost:8000/v1",
    temperature=1.0,
    max_tokens=256,
)

# 方式二：从 URL 自动推断后端
model_client = VLMModelFactory.from_url(
    base_url="http://localhost:8000/v1",
    model_name="Models/GUI-Owl-1.5-8B-Instruct",
)
```

#### CLI 脚本

```bash
# vLLM 后端（本地）
python roll/pipeline/agentic/env/android/exploration/scripts/run_exploration.py \
    --env mobileworld \
    --server_url http://localhost:18000 \
    --model_backend vllm \
    --model_name Models/GUI-Owl-1.5-8B-Instruct \
    --vllm_base_url http://localhost:8000/v1 \
    --model_temperature 1.0 \
    --model_max_tokens 256 \
    --num_episodes 20 \
    --max_steps 30 \
    --console_port 5554 \
    --grpc_port 8554 \
    --output_dir ./exploration_output

# OpenAI API 后端
python roll/pipeline/agentic/env/android/exploration/scripts/run_exploration.py \
    --env mobileworld \
    --server_url http://localhost:18000 \
    --model_backend openai \
    --model_name gpt-4o \
    --model_temperature 1.0 \
    --num_episodes 20 \
    --console_port 5554 \
    --grpc_port 8554

# 随机动作（不使用模型）
python roll/pipeline/agentic/env/android/exploration/scripts/run_exploration.py \
    --env mobileworld \
    --server_url http://localhost:18000 \
    --model_backend none \
    --console_port 5554 \
    --grpc_port 8554
```

#### Shell 脚本

```bash
# 默认使用 vLLM
export MODEL_BACKEND=vllm
export MODEL_NAME=Models/GUI-Owl-1.5-8B-Instruct
export VLLM_BASE_URL=http://localhost:8000/v1

bash jyc/scripts/run_exploration.sh
```

### 12.4 课程生成器模型配置

课程任务生成器（`CurriculumTaskGenerator`）在 `SelfEvolveCoordinator._generate_tasks()` 中初始化，需要传入有效的 OpenAI API Key：

```python
from roll.pipeline.agentic.env.android.mobile.curriculum_task_generator import MobileSpecificTaskGenerator

generator = MobileSpecificTaskGenerator(
    openai_api_key="sk-...",      # OpenAI API Key（vLLM 不可用）
    model="gpt-4o",               # 生成模型
    enable_diversity=True,
)
```

如果使用 OpenAI-compatible 端点替代 OpenAI 云端 API，可以将 API Key 替换为 vLLM 地址：

```python
import openai

client = openai.OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",  # 本地 vLLM 端点
)
```

### 12.5 Judge 模型配置

`MobileJudgeEvaluator`（`mobilejudge.py`）同样支持 vLLM 后端：

```python
from roll.pipeline.agentic.env.android.mobile.mobilejudge import MobileJudgeEvaluator

evaluator = MobileJudgeEvaluator(
    key_identification_screenshot_model="gpt-4o",   # 或本地 vLLM 模型名
    key_points_outcome_model="gpt-4o",
    max_image=50,
    use_vllm_for_key_screenshot=True,
    vllm_base_url="http://localhost:8000/v1",
)
```

### 12.6 完整使用流程（vLLM）

```bash
# 1. 启动 MobileWorld 后端服务（环境侧）
python roll/pipeline/agentic/env/android/GuiTaskEvalManager.py \
    > roll/pipeline/agentic/env/android/GuiTaskEvalManager.log 2>&1 &

# 2. 启动 vLLM 模型服务（模型侧）
#    在另一个终端运行：
python -m vllm.entrypoints.openai.api_server \
    --model Models/GUI-Owl-1.5-8B-Instruct \
    --port 8000 \
    --trust-remote-code \
    --dtype float32 \ 
    --gpu-memory-utilization 0.9

#    后台运行 dtype：'auto', 'bfloat16', 'float', 'float16', 'float32', 'half'
nohup python -m vllm.entrypoints.openai.api_server \
    --model /HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/Models/GUI-Owl-1.5-8B-Instruct \
    --served-model-name GUI-Owl-1.5-8B-Instruct \
    --trust-remote-code \
    --dtype float32 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    > vllm.log 2>&1 &

curl http://localhost:8000/v1/models

# 3. 运行探索阶段（使用 vLLM）
export MODEL_BACKEND=vllm
export MODEL_NAME=Models/GUI-Owl-1.5-8B-Instruct
export VLLM_BASE_URL=http://localhost:8000/v1
export NUM_EPISODES=5
bash jyc/scripts/run_exploration.sh

# 4. 启动自进化训练
export MODEL_BACKEND=vllm
bash jyc/scripts/run_self_evolve_pipeline.sh
```

### 12.7 常见问题

| 问题 | 原因 | 解决方案 |
|---|---|---|
| `APIConnectionError` | vLLM 服务未启动或地址错误 | 确认 `http://localhost:8000/v1` 可访问：`curl http://localhost:8000/v1/models` |
| `RateLimitError` | 请求频率过高 | 减小 `num_episodes` 并行数；在 vLLM 启动参数中添加 `--max-num-batched-tokens` |
| 模型输出非 JSON 格式 | 模型格式遵从度不足 | 降低 `temperature`（如 0.3）；在 prompt 中强调 JSON 格式要求 |
| 显存不足（OOM） | 模型过大 | 使用更小的模型（如 Qwen2.5-VL-3B-Instruct）；降低 `gpu-memory-utilization` |
| `OPENAI_API_KEY` 未设置 | 使用 OpenAI 后端但未配置 Key | 切换到 `--model_backend vllm`；或 `export OPENAI_API_KEY=sk-...` |

1. **Reward Override 而非替换**：Judge reward **覆盖**末步 reward 而非替换环境 reward，保证训练梯度来自 judge 信号，同时环境 reward 为 0 不影响训练
2. **懒加载**：`SelfEvolveCoordinator` 在 `AndroidStepEnvManager` 和 `AgenticPipeline` 中均为懒加载属性，首次访问时才导入，禁用时零开销
3. **原子写入**：所有 JSON 文件写入使用 `.tmp` + `os.rename()`，避免并发写入导致文件损坏
4. **TypedDict 强制类型**：所有跨模块数据结构使用 TypedDict，IDE 自动补全 + 类型检查
5. **兼容性优先**：不做任何破坏性修改，所有现有功能保持不变

---

## 九、快速上手

### 9.1 环境准备

#### 步骤 1：确保 Python 依赖已安装

```bash
pip install tenacity openai pillow datasets pandas pyarrow
```

#### 步骤 2：启动 MobileWorld 后端服务（已有则跳过）

```bash
# 在服务端机器上启动 MobileWorld API 服务
python roll/pipeline/agentic/env/android/GuiTaskEvalManager.py \
    > roll/pipeline/agentic/env/android/GuiTaskEvalManager.log 2>&1 &
```

#### 步骤 3：运行探索模块生成初始化数据（首次运行必须）

```bash
# 详细步骤参见 docs/exploration/exploration_androidworld_plan.md
python roll/pipeline/agentic/env/android/exploration/explorer.py \
    --output_dir ./exploration_output \
    --env_type mobileworld \
    --num_episodes 20 \
    --max_steps 30

python roll/pipeline/agentic/env/android/exploration/task_initializer.py \
    --exploration_dir ./exploration_output \
    --init_output_dir ./init_output \
    --env_type mobileworld
```

探索模块会生成：
- `exploration_output/exp_xxx/` — 探索轨迹和截图
- `init_output/TaskName/instance_N/` — 任务初始化参数（`params.pkl`、`init_screenshot.png`）

### 9.2 配置文件

完整配置示例见 `jyc/agent_val_self_evolve.yaml`，关键配置项说明：

```yaml
# =============================================
# 自进化模式配置（新增）
# =============================================
self_evolve:
  enabled: true
  # judge 结果与生成任务的存储根目录
  feedback_root: "./trajectories/self_evolve"
  # 生成任务的 JSON 文件存放目录
  generated_task_root: "./data/tasks/generated"
  # 自进化 parquet 文件存放目录
  parquet_root: "./data/self_evolve_parquet"
  # 每隔多少 global_step 触发一次任务生成
  round_update_interval: 1

# =============================================
# 训练流程配置
# =============================================
max_steps: 32          # 总训练步数，自进化模式下建议 16-64
save_steps: 8           # 每隔多少步保存一次 checkpoint
logging_steps: 1        # 每隔多少步记录一次日志
eval_steps: -1          # 评估间隔（-1 表示不评估）

rollout_batch_size: 32  # 每步参与 rollout 的任务数

# =============================================
# 训练环境管理器配置（自进化关键）
# =============================================
train_env_manager:
  max_env_num_per_worker: 32
  num_env_groups: 4
  group_size: ${group_size}
  tags: [AndroidEnv-Train]
  num_groups_partition: [4]
  group_filter_cls: roll.pipeline.agentic.env_manager.gui_traj_env_manager.AndroidEnvGroupFilter

# =============================================
# 自进化专用环境配置（env_config 中嵌入）
# =============================================
custom_envs:
  AndroidEnv-Train:
    env_type: RemoteMobileEnv
    env_manager_cls: roll.pipeline.agentic.env_manager.android_step_env_manager.AndroidStepEnvManager
    env_config:
      env_type: RemoteMobileEnv
      env_id: null
      group_id: null
      max_steps: 40
      max_tokens_per_step: 512
      rollout_data_type: trajectory
      keep_last_k: 8
      # 自进化模式开关（优先级高于顶层 self_evolve 配置）
      self_evolve:
        enabled: true
        feedback_root: "./trajectories/self_evolve"
```

> **注意**：`self_evolve` 配置同时出现在顶层（`AgenticPipeline` 级）和 `env_config` 中（`AndroidStepEnvManager` 级），两者取并集，`env_config` 中的值优先级更高。

### 9.3 启动训练

```bash
# 方式一：直接运行脚本（推荐）
bash jyc/scripts/run_self_evolve_pipeline.sh

# 方式二：手动分步启动
# 1. 启动 MobileWorld 后端（如果尚未启动）
python roll/pipeline/agentic/env/android/GuiTaskEvalManager.py \
    > roll/pipeline/agentic/env/android/GuiTaskEvalManager.log 2>&1 &

# 2. 运行训练
python roll/pipeline/agentic/start_agentic_pipeline.py \
    --config_name agent_val_self_evolve \
    rollout_batch_size=32 \
    max_steps=32
```

### 9.4 运行后观察

训练启动后，可通过以下方式验证自进化是否正常运行：

#### 检查点 1：Judge 结果文件生成

```bash
# 每完成一个 episode，目录下会生成 judge_input.json 和 judge_result.json
ls -la trajectories/self_evolve/round_0001/episodes/
```

示例输出：
```
settings_wifi_task_abc123/
  episode_0/
    judge_input.json   # 约 1-5KB
    judge_result.json  # 约 500B-2KB
  episode_1/
    judge_input.json
    judge_result.json
```

#### 检查点 2：Round 反馈文件

```bash
# 第一个 round 结束后会生成 round_feedback.json
cat trajectories/self_evolve/round_0001/round_feedback.json | python -m json.tool
```

#### 检查点 3：生成的任务 JSON

```bash
ls -la trajectories/self_evolve/round_0001/generated_tasks/mobile/
# 应看到形如 mobile_app.exp_001_task_xxx.json 的文件
```

#### 检查点 4：Parquet 文件

```bash
ls -lh trajectories/self_evolve/round_0001/parquet/train.parquet
```

### 9.5 日志关键词

在训练日志中搜索以下关键词确认自进化运行状态：

| 关键词 | 含义 |
|---|---|
| `[SelfEvolve] AndroidStepEnvManager self-evolve mode enabled` | EnvManager 自进化已启用 |
| `[SelfEvolve] AgenticPipeline self-evolve mode enabled` | Pipeline 自进化已启用 |
| `[SelfEvolve] Round N: X tasks, SR=Y.Y%` | Round N 反馈已生成 |
| `[SelfEvolve] Round N update complete` | Round N 任务生成完成 |
| `[SelfEvolve] on_episode_end failed` | Judge 调用失败（不影响训练，降级处理） |

---

## 十、配置参考

### 10.1 完整 YAML 配置

参见 `jyc/agent_val_self_evolve.yaml`。

### 10.2 启动脚本

参见 `jyc/scripts/run_self_evolve_pipeline.sh`。

### 10.3 `self_evolve` 配置节点完整字段

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | `bool` | `false` | 总开关，设为 `true` 启用自进化模式 |
| `feedback_root` | `str` | `"./trajectories/self_evolve"` | judge 结果存储根目录 |
| `generated_task_root` | `str` | `"./data/tasks/generated"` | 生成的任务 JSON 文件存储目录 |
| `parquet_root` | `str` | `"./data/self_evolve_parquet"` | 自进化 parquet 输出目录 |
| `round_update_interval` | `int` | `1` | 每隔多少 global_step 触发一次 `on_round_end` |

### 10.4 `env_config.self_evolve` 字段

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | `bool` | `false` | 环境管理器级开关（优先级更高） |
| `feedback_root` | `str` | `"./trajectories/self_evolve"` | 环境管理器使用的 feedback 目录 |

---

## 十一、故障排查

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `[SelfEvolve] on_episode_end failed` | Judge API 调用超时或失败 | 检查 `OPENAI_API_KEY` 环境变量；确认 MobileJudgeEvaluator 的 model 配置正确 |
| `round_feedback.json` 未生成 | 无 `judge_result.json` 文件 | 检查 `feedback_root` 路径是否正确；确认 episode 已正常结束 |
| 生成任务 JSON 数量为 0 | Exploration 输出目录不存在 | 确认探索模块已运行；`exploration_dirs` 路径正确 |
| `params_path` 相关错误 | pickle 文件不存在 | 确认 TaskInitializer 已运行；探索和初始化是同一批次数据 |
| `snapshot` 字段缺失错误 | `create_task_config_file` 校验失败 | 确保 `context_template` 中包含 `snapshot` 字段，或 `init_result.task_name` 非空 |
| 训练 reward 始终为 0 | judge_reward 未覆盖末步 | 确认 `self_evolve.enabled=true`；检查 `on_episode_end` 是否被调用 |
