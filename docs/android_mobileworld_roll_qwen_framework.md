# ROLL 在 AndroidWorld / MobileWorld 上的 Qwen GUI Agent 训练与评测框架解读

## 1. 文档范围
本文围绕以下核心代码，梳理项目在 **mobile GUI agent 在线训练/评测** 的完整链路：

- 训练入口：`jyc/scripts/run_agentic_pipeline.sh`
- 评测入口：`jyc/scripts/evaluate_agentic_pipeline.sh`
- 训练主干：`roll/pipeline/agentic/agentic_pipeline.py`
- rollout 调度：`roll/distributed/scheduler/rollout_scheduler.py`
- 环境与轨迹管理：
  - `roll/pipeline/agentic/env_manager/android_step_env_manager.py`
  - `roll/pipeline/agentic/env_manager/gui_traj_env_manager.py`
  - `roll/pipeline/agentic/env/android/remote_android.py`
  - `roll/pipeline/agentic/env/android/remote_mobileworld.py`
  - `roll/pipeline/agentic/env/android/GuiTaskEvalManager.py`

同时补充了框架运行所必需的关联模块（`EnvironmentWorker`、`RequestScheduler`、`AgenticRolloutPipeline`、`agentic/utils.py` 等）。

> 说明：代码中部分 `/HOME/...` 的绝对路径是迁移遗留，不影响本框架逻辑理解，本文默认忽略路径不匹配问题。

---

## 2. 总体架构（从脚本到训练）

### 2.1 两条入口链路

1. 训练链路
- `run_agentic_pipeline.sh` -> `jyc/start_agentic_pipeline.py` -> `AgenticPipeline.run()`
- 默认配置：`agent_val_multiandroid_grpo`（可切到 `..._lora`）

2. 评测链路
- `evaluate_agentic_pipeline.sh` -> `jyc/evaluate_agentic_pipeline.py` -> `AgenticRolloutPipeline.run()`
- 默认 `android_world` 走 `agent_val_multiandroid_grpo_evaluate`
- `mobile_world` 走 `agent_val_multimobileworld_evaluate`
- `MODEL_PARAM` 可切到 `voyager/guiowl/reflection` 对应不同 env_manager 模板

### 2.2 在线训练关键参与者

- **AgenticPipeline**：RL 主循环（模型更新、取 rollout、奖励/优势、训练、评测）。
- **RolloutScheduler**：把“环境 rollout 系统”与“推理系统”拼接成可取 batch 的统一接口。
- **EnvironmentWorker + EnvManager**：每个 worker 内多线程跑多个环境实例。
- **GroupQueueManager**：按 task/group 管理 rollout 收集与出队。
- **RequestScheduler**：单条请求级推理调度（支持 suspend/resume/abort）。
- **RemoteAndroidEnv / RemoteMobileEnv**：对接远端 Android 服务，执行 reset/step、产出观测与奖励。
- **GuiTaskEvalManager(FastAPI)**：任务分配与统计服务（train/eval 各一套 URL）。

---

## 3. Qwen 模型在本框架中的接入方式

### 3.1 配置层
`jyc/agent_val_multiandroid_grpo*.yaml` 与 `jyc/agent_val_multimobileworld*.yaml` 中通过：
- `Model`/`pretrain` 指定模型（常见为 `Qwen3-VL-*`）
- `actor_train`（训练）
- `actor_infer`（环境交互推理，常用 `vllm`）
- `reference`（KL 参考策略）

训练配置通常是 GRPO：
- `adv_estimator: grpo`
- `enable_reference: true`

### 3.2 推理代理层
`android_step_env_manager.py` 中的 LLM 调用统一经 `create_llm_proxy()`：
- 默认 `LLMProxyConfig.proxy_type = policy`
- `PolicyProxy` 内部调用 `RequestScheduler.generate_one_request`
- 即：环境线程并不直接调用模型，而是走统一请求调度器

### 3.3 多模态输入
- `EnvironmentWorker.initialize()` 为每个 env_manager 创建 tokenizer + processor
- `AndroidStepEnvManager.format_messages()` 构造 Qwen-VL 风格 messages（文本+截图 base64 image）
- 再由 `DataCollatorWithPaddingForMM` 打包为 `DataProto` 输入

---

## 4. 环境管理：AndroidWorld 与 MobileWorld

### 4.1 gem 环境注册
`roll/pipeline/agentic/env/__init__.py` 中注册了：
- `remote_multi_android`
- `remote_multi_mobileworld`

配置里的 `env_type` 最终通过 `gem.make()` 实例化。

### 4.2 多 server 封装
- `remote_multi_android.py`
- `remote_multi_mobileworld.py`

逻辑一致：
1. 根据 `android_env_id` 判断落在哪个 server 配置分片
2. 解析 console/grpc ports
3. 选出当前 env 对应端口
4. 调父类（`RemoteAndroidEnv` / `RemoteMobileEnv`）完成真实环境初始化

### 4.3 RemoteAndroidEnv / RemoteMobileEnv 职责

共同点：
- 与远端服务交互：`/init`、`/reset`、`/step`、`/close`
- 失败重试（tenacity）+ 故障恢复（close -> re-init）
- 异常时调用 task manager `/return_task`
- 终止时上报 `/complete_task`
- 按 task 保存轨迹截图与元数据到本地目录

差异点：
- AndroidWorld：观测来自 `observation_np_b64` + dtype + shape。
- MobileWorld：观测来自 `screenshot_b64(PNG)`。
- MobileWorld 成功判定用 `score >= success_threshold`（默认 0.99）。

---

## 5. Agent 上下文管理（核心在 AndroidStepEnvManager）

### 5.1 RolloutCache 状态
每条轨迹使用 `RolloutCache` 维护：
- `history`: 每步 observation / response / reward / metrics / hash 等
- `step`, `terminated`, `truncated`

### 5.2 Prompt 与上下文构建
`AndroidStepEnvManager.format_messages()`：

1. system prompt：内置函数调用规范（`mobile_use` tool schema）
2. 从历史中提取最近 `keep_last_k` 步动作摘要（默认 8）
3. 注入当前任务目标（`self.env.task["goal"]`）
4. 注入当前截图（base64 image）
5. 可选注入 skill memory（`android_utils.get_skill`）
6. 生成 `input_ids/attention_mask/position_ids`
7. 将 `prompt_ids/state_hash/messages` 写回当前 step cache

### 5.3 决策与执行
- `make_decision()`：调用 llm_proxy 生成 response token ids
- `step()`（继承 `GuiTrajEnvManager`）：把解码文本动作交给 env.step
- 环境返回新的 observation/reward/done 并追加到历史

---

## 6. Agent 与环境的交互循环

执行入口在 `GuiTrajEnvManager.run_rollout_loop()`（`AndroidStepEnvManager` 复用）。

单环境线程循环：
1. `reset()`：向 `GroupQueueManager.get_episode_id()` 申请任务分配（含 `task/group_id/episode_id`）
2. 环境 reset 到目标 task
3. `make_decision()` 生成动作
4. `step()` 与环境交互
5. 终止后 `formulate_rollouts()` 转成训练样本 `DataProto`
6. 通过 `output_queue.put()` 上交给 GroupQueue
7. 继续下一 episode

异常 reset/step 会产出 `None rollout`，并回收 episode slot / return task。

---

## 7. 任务调度与 rollout 调度

### 7.1 任务调度（GuiTaskEvalManager）
`GuiTaskEvalManager.py` 提供 HTTP API：
- `/initialize`：初始化 task 列表、group_size、n_task、seed
- `/get_task`：分发任务
- `/complete_task`：记录成功率、平均步数、平均时间
- `/return_task`：环境失败回退配额

调度策略要点：
- 以 `GROUP_SIZE` 为单位连续分配同一任务
- 任务选择优先 `total_attempts` 最少，再看 `assigned`
- 同分候选用 `seed + step` 的确定性哈希打散
- 到达 `N_TASK * GROUP_SIZE` 后返回 `finish`

### 7.2 rollout 调度（RolloutScheduler + GroupQueueManager）

`RolloutScheduler.get_batch()` 每轮做：
1. 首次启动环境 rollout loop
2. 向 env workers 广播 `global_step`
3. `GroupQueueManager.advance_step(step)`
4. `RequestScheduler.resume()` 放开推理
5. 等待 `GroupQueueManager.get_batch(batch_size, step)` 收集完成组

`GroupQueueManager` 关键点：
- task -> group queue 映射（动态创建）
- 每次 `get_episode_id()` 会向 task manager `/get_task`
- 每个 group 收到 `group_size` 条 rollout 才算完成
- 可按 `group_filter_cls` 过滤低质量组
- 过滤后可尝试从 `GlobalTrajectoryCache` 替换一条历史最佳轨迹

---

## 8. rollout 生成（DataProto 结构）

`AndroidStepEnvManager.formulate_rollouts()` 会把轨迹拆成 step-level 样本（训练时多步，val 通常仅保留首步样本以减内存）：

`batch` 常见字段：
- `input_ids`
- `attention_mask`
- `position_ids`
- `response_mask`
- `prompt_mask`
- `scores`（每条样本在末 token 写入 step reward）
- `infer_logprobs`（可选）

`non_tensor_batch` 常见字段：
- `episode_scores`
- `step_scores`
- `tags`
- `env_ids`
- `group_ids`
- `state_hash`
- `step`
- `traj_group_id`
- `traj_id`

`meta_info`：
- `task`
- `metrics`（env侧聚合统计）

---

## 9. rollout 系统与算法框架交互（训练主干）

`AgenticPipeline.run()` 的每个 global step 可概括为：

1. **暂停 rollout / 推理切换**
- `train_rollout_scheduler.suspend()`
- 模型更新前可停 actor_infer server

2. **模型同步**
- `model_update()` 由 `ModelUpdateGroup` 把 `actor_train` 参数同步到 `actor_infer`

3. **拉取 rollout batch**
- `train_rollout_scheduler.get_batch(..., rollout_batch_size)`
- 写入 trajectory dump

4. **奖励与优势构建**
- `compute_discounted_returns`（gigpo/step_reinforce 时）
- `compute_response_level_rewards`（group/batch 归一化）
- `compute_token_reward`（把 response reward 展开到 token，并可加 KL）
- `agentic_compute_advantage`（GRPO 走 reinforce-return）

5. **训练**
- `actor_train.train_step(batch)`
- 若 `adv_estimator=gae` 再训练 critic

6. **记录与 checkpoint**
- 各类 env/critic/token/system 指标
- checkpoint 与 tracker log

### 9.1 评测主干（无参数更新）
`AgenticRolloutPipeline.run()`：
- 仅启动 `actor_infer + rollout_scheduler(val_env_manager)`
- 持续收集 rollout 并统计 `val/*` 指标
- 不做 actor/critic 反向更新

---

## 10. AndroidWorld / MobileWorld 在该框架中的统一与差异

统一点：
- 统一 EnvManager（AndroidStepEnvManager）
- 统一调度（RolloutScheduler + GroupQueueManager + RequestScheduler）
- 统一 RL 训练主干（AgenticPipeline）

差异点：
- `env_type` 不同：`remote_multi_android` vs `remote_multi_mobileworld`
- 远端服务返回 payload 格式不同（观测解析、done/success 语义）
- task 集合不同（android tasks vs mobileworld tasks）

本质上：**算法层完全复用，环境差异被封装在 RemoteEnv 层与少量配置中**。

---

## 11. 关键实现注意事项（阅读/二次开发时建议关注）

1. `GroupQueueManager.__init__` 中将 `max_traj_per_env` 强制设为 `None`，实际不再按该配置限制轨迹数，更多依赖 group/filter 控制。
2. `remote_android.py` 中 `task == all_task` 分支存在固定任务覆盖（`FilesDeleteFile`, `OsmAndFavorite`），会影响任务覆盖面。
3. `AndroidStepEnvManager` 的 `scores` 写的是 step reward；评测若要严格 episode 指标，建议优先看 `non_tensor_batch["episode_scores"]`。
4. 轨迹缓存（`GlobalTrajectoryCache`）链路已接入，但 `GuiTrajEnvManager.save_trajectory` 当前实现对 `ObjectRef` 的比较逻辑建议再核查。

---

## 12. 一句话总结

这个项目把 **Qwen-VL GUI agent 的在线 RL** 拆成了三层解耦：
- **环境层**（Android/MobileWorld 远端执行 + task manager）
- **rollout层**（多线程环境交互 + GroupQueue 聚合 + RequestScheduler 推理调度）
- **算法层**（GRPO/PPO 训练循环）

因此你可以在基本不改训练算法的前提下，替换环境实现或 agent prompt/策略实现，并保持统一的 rollout->训练数据接口。

