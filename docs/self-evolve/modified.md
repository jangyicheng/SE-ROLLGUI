# 自进化模式代码改动清单

> 本文档记录本次自进化模式（Self-Evolving Mode）实现中，所有新增文件及对现有文件的改动位置与内容摘要。

## 一、新增文件

| 文件路径 | 职责说明 |
|---|---|
| `roll/pipeline/agentic/env/android/mobile/self_evolve_types.py` | 集中定义所有 TypedDict 类型（JudgeEpisodeInput、JudgeEpisodeResult、RoundFeedback、TaskPerformance、ExplorationTaskContext 等） |
| `roll/pipeline/agentic/env/android/mobile/self_evolve_io.py` | JSON 文件原子读写（`.tmp` + `os.rename`）、目录管理、exploration 数据加载（桥接 TrajectoryFormatter） |
| `roll/pipeline/agentic/env/android/mobile/self_evolve_feedback.py` | `build_round_feedback()` 聚合 episode 级 judge 结果为 round_feedback.json |
| `roll/pipeline/agentic/self_evolve_coordinator.py` | `SelfEvolveCoordinator`：回合级编排、judge 中转（JSON 文件）、任务生成触发 |
| `roll/pipeline/agentic/env/android/exploration/model_client.py` | `VLMModelFactory`（支持 openai / vllm / huggingface 后端）+ `ExplorerModelWrapper`（桥接 BaseExplorer 接口与 chat-message API） |

---

## 二、修改文件

### 1. `roll/pipeline/agentic/env/android/mobile/curriculum_task_generator.py`

**类：`MobileSpecificTaskGenerator`**

| 改动位置 | 改动内容 |
|---|---|
| `load_feedback_data()` (L251-268) | 兼容旧格式（`accuracy`）与自进化格式（`success_rate`）；新增 `load_round_feedback()` 方法（L274） |
| `format_feedback_summary()` (L276-310) | 增加 `accuracy → success_rate` 自动归一化 |
| **`format_feedback_summary_v2()`** (L312-385) | **新增方法**：按成功率分层（高≥80%、中30-80%、低<30%）生成反馈摘要，包含失败原因提取 |
| **`generate_tasks_from_exploration()`** (L705-878) | **新增方法（自进化主入口）**：接收 `ExplorationTaskContext` 列表 + `round_feedback.json` 路径，调用 LLM 生成任务 |
| **`create_task_config_file()`** (L580-630) | **新增 `snapshot` 必填校验**（L599-616），无 snapshot 时抛出 `ValueError` |
| `load_exploration_data()` (L948-971) | 从 exploration 结果 JSON 加载数据；现已移至类内 |
| `load_init_data()` (L973-1007) | 加载 TaskInitializer 输出；现已移至类内 |
| `format_context_review_for_prompt()` (L1009-1065) | 将 init 结果格式化为 prompt 文本；现已移至类内 |
| `get_exploration_screenshots()` (L1126-1145) | 从 exploration 目录获取截图；现已移至类内 |
| `create_task_config_file` 末尾赋值 | `config["source"]` 默认值改为 `"self_evolve"`（L634） |

**说明**：原有在类外定义的 `generate_tasks_from_exploration()`、`load_exploration_data()` 等方法已移除重复定义，统一移入类内。

---

### 2. `roll/pipeline/agentic/env_manager/android_step_env_manager.py`

**类：`AndroidStepEnvManager`**

| 改动位置 | 改动内容 |
|---|---|
| `__init__` (L146-152) | 新增 `self.self_evolve_config`、`self.self_evolve_enabled` 属性（从 `env_config["self_evolve"]` 读取，默认 `False`） |
| **`self_evolve_coordinator` 属性** (L154-165) | **新增懒加载属性**：首次访问时延迟导入并初始化 `SelfEvolveCoordinator` |
| `formulate_rollouts()` (L302-460) | **主要改动**：<br>1. 方法开头（L302-333）新增 episode-end judge 调用分支<br>2. 末步 reward 覆盖逻辑（L367-371）：`score_tensor[-1] = judge_reward_override`<br>3. `episode_scores` 改用 `effective_episode_score`（L384）<br>4. 末步注入 `judge_feedback` 和 `judge_reward` 元数据（L391-393）<br>5. 轨迹缓存保存阈值使用 judge_reward（L453-458） |
| 类型注解 | `Optional` 已导入 |

---

### 3. `roll/pipeline/agentic/env_manager/gui_traj_env_manager.py`

**类：`GuiTrajEnvManager`**

| 改动位置 | 改动内容 |
|---|---|
| **`on_episode_end()`** (L403-411) | **新增虚方法**：默认透传 `rollout_cache`；子类（如 `AndroidStepEnvManager`）可覆写以实现 judge 调用等逻辑 |

---

### 3. `roll/pipeline/agentic/agentic_config.py`

**类：`AgenticConfig`**

| 改动位置 | 改动内容 |
|---|---|
| 类属性定义 (L193-207) | 新增 `self_evolve: Dict[str, Any]` 字段，包含 `enabled`、`feedback_root`、`generated_task_root`、`parquet_root`、`round_update_interval` 等子项，默认 `enabled=False` |

---

### 4. `roll/pipeline/agentic/agentic_pipeline.py`

**类：`AgenticPipeline`**

| 改动位置 | 改动内容 |
|---|---|
| `__init__` (L131-145) | 新增 `self.self_evolve_config`、`self.self_evolve_enabled` 属性（从 `pipeline_config.self_evolve` 读取，默认 `False`） |
| **`self_evolve_coordinator` 属性** (L135-144) | **新增懒加载属性**：首次访问时延迟导入并初始化 `SelfEvolveCoordinator` |
| `run()` (L312-318) | 在 `do_checkpoint()` 后新增 `on_round_end()` 调用（受 `round_update_interval` 控制，默认每轮执行） |

---

### 5. `roll/pipeline/agentic/env/android/mobile/prepare_data.py`

| 改动位置 | 改动内容 |
|---|---|
| CLI 参数解析 (L123-127) | 新增 `--self_evolve_mode` 布尔参数（`action='store_true'`） |
| `load_data()` 分支 (L189、L219) | 当 `args.self_evolve_mode=True` 时，`val_tasks` 设为空列表，跳过 UUID 格式过滤 |
| 日志输出 (L206-208、L233-235) | 自进化模式下打印 "Self-evolve mode: all tasks go to train split" |

---

### 6. `roll/pipeline/agentic/env/android/remote_mobileworld.py`

**类：`RemoteMobileEnv`**

| 改动位置 | 改动内容 |
|---|---|
| `_call_reset_with_params()` (L163-169) | **新增方法**：调用 `call_reset_with_params` 的错误处理封装 |
| `call_reset_with_params()` (L199-206) | **新增方法**：带重试装饰器的 `/reset_with_params` 端点调用 |
| `reset()` (L243-348) | **主要改动**：<br>1. 从 `target_task` dict 中识别 `params_path` 和 `snapshot` 字段（L256-260）<br>2. 若 `params_path` 存在：加载 pickle 文件，构建含 `params` 的 payload，调用 `_call_reset_with_params`（L284-298）<br>3. `params_path` 文件不存在时自动回退到标准 `reset`（L299-318）<br>4. 无 `params_path` 时走原有 `call_reset` 逻辑（L329-347） |

---

### 7. `roll/pipeline/agentic/env/android/exploration/scripts/run_exploration.py`

| 改动位置 | 改动内容 |
|---|---|
| 文件头部 docstring (L1-14) | 更新使用示例，增加 vLLM 和 OpenAI 后端的命令行示例 |
| 新增导入 (L22-24) | 导入 `VLMModelFactory` |
| **`_build_model_client()`** (L27-50) | **新增函数**：根据 CLI 参数构造 `ExplorerModelWrapper`；支持 `none`（随机动作）模式 |
| argparse 模型参数组 (L103-142) | 新增 `--model_backend`、`--model_name`、`--vllm_base_url`、`--api_key`、`--model_temperature`、`--model_max_tokens` 参数 |
| `main()` 调用 (L167, L189) | Explorer 初始化时传入 `model_client=model_client`（替换原有 `model_client=None`）；`model_type` 参数移除 |
| 结果打印 (L212) | 新增打印 `result.model` 字段，显示实际使用的模型名称 |

---

### 8. `roll/pipeline/agentic/env/android/exploration/__init__.py`

| 改动位置 | 改动内容 |
|---|---|
| 新增导入 (L20) | 导入 `ExplorerModelWrapper` 和 `VLMModelFactory` |
| `__all__` (L26-27) | 新增 `"ExplorerModelWrapper"`、`"VLMModelFactory"` 到导出列表 |

---

### 9. `jyc/scripts/run_exploration.sh`

| 改动位置 | 改动内容 |
|---|---|
| 文件头注释 (L8-18) | 新增使用示例（vLLM、OpenAI、随机模式） |
| 环境变量配置 (L22-31) | 新增 `MODEL_BACKEND`、`MODEL_NAME`、`VLLM_BASE_URL`、`MODEL_TEMPERATURE`、`MODEL_MAX_TOKENS` 环境变量（默认值：`vllm`、`Qwen/Qwen2.5-VL-7B-Instruct`、`http://localhost:8000/v1`） |
| 命令行参数解析 (L33-64) | 新增 `while` 循环解析 `--model_backend`、`--model_name`、`--vllm_base_url`、`--num_episodes`、`--max_steps`、`--exploration_output_dir`、`--init_output_dir` 参数 |
| 步骤 1 Explorer 调用 (L76-118) | 根据 `MODEL_BACKEND` 拼接不同 Python 命令行：vLLM 传入 `--model_backend vllm --model_name --vllm_base_url`；OpenAI 传入 `--model_backend openai --model_name`；none 传入 `--model_backend none` |

---

## 三、兼容性保证

所有自进化相关代码均受 `self_evolve.enabled = False` 配置项保护：

- `AndroidStepEnvManager.__init__`：`self.self_evolve_enabled = False` 默认为假
- `AgenticPipeline.__init__`：`self.self_evolve_enabled = False` 默认为假
- `SelfEvolveCoordinator.on_episode_end()`：若 `not self.enabled` 直接返回空结果
- `SelfEvolveCoordinator.on_round_end()`：若 `not self.enabled` 直接返回 `{"success": False}`
- `AndroidStepEnvManager.formulate_rollouts()`：所有 judge 相关逻辑均在 `if self.self_evolve_enabled` 分支内
- `AgenticPipeline.run()`：round-end hook 同样在 `if self.self_evolve_enabled` 分支内

---

## 四、配置说明

在训练 YAML 配置中添加以下节点即可启用自进化模式：

```yaml
self_evolve:
  enabled: true
  feedback_root: "./trajectories/self_evolve"         # judge 结果存储根目录
  generated_task_root: "./data/tasks/generated"         # 生成任务存储根目录
  parquet_root: "./data/self_evolve_parquet"           # Parquet 输出根目录
  round_update_interval: 1                            # 每 N 轮触发一次任务更新
```

在 `env_config`（环境管理器配置）中同样可设置：

```yaml
train_env_manager:
  env_config:
    self_evolve:
      enabled: true
      feedback_root: "./trajectories/self_evolve"
```
