# ROLL 框架技术文档

> 本文档基于 `e:/code/GUI/Roll-GUI` 仓库中 `roll/` 目录下的源码（截至 2026-04-24）编写，包含目录树分析、核心逻辑详解、配置说明和注意事项。
>
> **注**：本仓库包含 AndroidWorld/MobileWorld GUI Agent 训练框架的扩展代码（`roll/pipeline/agentic/env/android/`），但主体框架基于阿里巴巴 ROLL 开源项目（[arXiv:2506.06122](https://arxiv.org/abs/2506.06122)）。

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构与模块职责](#2-目录结构与模块职责)
3. [核心逻辑详解](#3-核心逻辑详解)
   - [3.1 数据流：从输入到输出的完整链路](#31-数据流从输入到输出的完整链路)
   - [3.2 模型/算法架构：组件关系图](#32-模型算法架构组件关系图)
   - [3.3 关键函数与实现细节](#33-关键函数与实现细节)
   - [3.4 主训练循环](#34-主训练循环)
   - [3.5 优势估计与奖励处理](#35-优势估计与奖励处理)
   - [3.6 Agentic 训练架构](#36-agentic-训练架构)
   - [3.7 AndroidWorld / MobileWorld GUI Agent 训练详解](#37-androidworld--mobileworld-gui-agent-训练详解)
4. [配置系统与超参数](#4-配置系统与超参数)
5. [快速启动](#5-快速启动)
6. [注意事项与陷阱](#6-注意事项与陷阱)

---

## 1. 项目概述

ROLL（Reinforcement Learning Optimization for Large-Scale Learning）是阿里巴巴淘天未来生活实验室与阿里巴巴 AI 引擎团队联合开发的高效大模型 RL 训练框架。核心能力：

- **多任务 RLVR 训练**：覆盖数学、代码、推理、通用问答、IFEval 等多领域，通过 `domain_interleave_probs` 动态采样
- **Agentic RL**：基于环境的 RL 训练（游戏、多轮对话、工具调用），支持 TrajectoryWise（StarPO）和 StepWise（GiGPO）两种训练范式
- **多后端策略抽象**：统一接口支持 vLLM、SGLang、Megatron-Core、DeepSpeed、HuggingFace 等推理/训练后端
- **多角色分布式架构**：基于 Ray 实现 Actor、Critic、Reference、Reward 等角色的灵活资源分配与异构任务调度
- **异步并行**：支持异步 Rollout、异步奖励计算、异步训练（async_generation_ratio > 0）
- **AndroidWorld / MobileWorld GUI Agent 训练**：基于远端 Android 设备的 GUI Agent 在线 RL 训练与评测，支持 Qwen-VL 多模态模型、MobileJudge 评判和课程任务自进化

---

## 2. 目录结构与模块职责

```
roll/                          (共 313 个 Python 文件)
├── configs/                   (7 文件) 配置类定义
│   ├── base_config.py         PPOConfig + BaseConfig（exp_name, seed, max_steps, rollout_batch_size 等全局参数）
│   ├── worker_config.py       WorkerConfig（device_mapping, strategy_args, training_args 等角色配置）
│   ├── data_args.py          数据加载参数
│   ├── generating_args.py     生成参数（temperature, top_p, max_new_tokens 等）
│   └── model_args.py         模型参数（dtype, attn_implementation 等）
│
├── datasets/                  (7 文件) 数据加载与处理
│   ├── dataset.py             Dataset 基类
│   ├── global_dataset.py      GlobalDatasetManager（Ray actor）
│   ├── global_trajectory_cache.py  GlobalTrajectoryCacheManager / GlobalTrajectoryCache（轨迹缓存）
│   ├── loader.py              DataLoader 封装
│   ├── collator.py            DataCollatorWithPaddingForPaddedKeys（多模态数据整理）
│   └── sampler.py             采样器
│
├── distributed/               (27 文件) Ray 分布式系统核心
│   ├── executor/
│   │   ├── cluster.py         Cluster 类：管理多个 Worker ray actor，绑定 Dispatch 方法，模型更新组
│   │   ├── worker.py           Worker 基类（ray.remote）
│   │   ├── model_update_group.py  ModelUpdateGroup：从 actor_train -> actor_infer 的参数同步
│   │   └── cluster.py          节点亲和调度，worker rank 管理
│   ├── scheduler/
│   │   ├── generate_scheduler.py     GenerateScheduler（ray.remote，生成请求调度，支持动态采样）
│   │   ├── async_generate_scheduler.py  AsyncDynamicSamplingScheduler（异步版本）
│   │   ├── reward_scheduler.py         RewardScheduler（奖励计算调度）
│   │   ├── rollout_scheduler.py       RolloutScheduler（环境 rollout 调度，GroupQueueManager）
│   │   ├── resource_manager.py        ResourceManager（GPU 资源管理）
│   │   ├── protocol.py                DataProto（跨函数数据交换标准协议）
│   │   └── storage.py / running_stats_utils.py
│   └── strategy/              (9 文件) 推理/训练后端统一抽象
│       ├── strategy.py             InferenceStrategy 基类（forward_step, generate, add_request 等接口）
│       ├── vllm_strategy.py         vLLM 后端（支持 0.8.4 / 0.10.0 / 0.10.2 / 0.11.0 四个版本）
│       ├── sglang_strategy.py       SGLang 后端
│       ├── deepspeed_strategy.py    DeepSpeed ZeRO 后端
│       ├── megatron_strategy.py      Megatron-Core 后端（TP/PP/CP/EP）
│       ├── hf_strategy.py           HuggingFace 后端
│       └── factory.py               策略工厂
│
├── models/                   (3 文件) 模型抽象层
│   ├── model_providers.py     default_tokenizer_provider / default_processor_provider
│   ├── func_providers.py      函数提供者
│   └── trl_patches.py         TRL 库补丁
│
├── pipeline/                  (117 文件) 训练管道定义
│   ├── base_pipeline.py       BasePipeline（run, model_update, do_checkpoint, set_model_update_pair）
│   ├── base_worker.py         ActorWorker / CriticWorker 基类
│   │
│   ├── rlvr/                 (RLVR 主管道，RL with Verifiable Rewards)
│   │   ├── rlvr_pipeline.py        RLVRPipeline 主循环（820+ 行）
│   │   ├── rlvr_config.py          RLVRConfig（继承 PPOConfig）
│   │   ├── actor_worker.py          ActorWorker（PPO loss, GRPO loss, KL penalty）
│   │   ├── actor_pg_worker.py
│   │   └── rewards/                 (8 个奖励计算 Worker)
│   │       ├── math_rule_reward_worker.py       MathRuleRewardWorker
│   │       ├── code_sandbox_reward_worker.py    CodeSandboxRewardWorker（沙箱执行）
│   │       ├── llm_judge_reward_worker.py      LLMJudgeRewardWorker（LLM 作为 judge）
│   │       ├── ifeval_rule_reward_worker.py    GeneralRuleRewardWorker
│   │       ├── crossthinkqa_rule_reward_worker.py
│   │       ├── general_val_rule_reward_worker.py
│   │       ├── detection_reward_worker.py
│   │       └── multiple_choice_boxed_rule_reward_worker.py
│   │
│   ├── agentic/              (Agentic RL 管道)
│   │   ├── agentic_pipeline.py     AgenticPipeline 主循环
│   │   ├── agentic_rollout_pipeline.py  AgenticRolloutPipeline（仅评测，无参数更新）
│   │   ├── agentic_config.py       AgenticConfig
│   │   ├── agentic_actor_worker.py
│   │   ├── environment_worker.py    EnvironmentWorker（多线程环境管理）
│   │   ├── utils.py               compute_discounted_returns, compute_response_level_rewards, agentic_compute_advantage
│   │   ├── llm_proxy/             (LLM 调用代理)
│   │   │   ├── policy_proxy.py     PolicyProxy（经 RequestScheduler 调度）
│   │   │   ├── openai_proxy.py
│   │   │   └── random_proxy.py
│   │   ├── tools/                 (工具调用封装)
│   │   │   ├── python_code_tool.py
│   │   │   ├── mcp_tool.py
│   │   │   └── tool_env_wrapper.py
│   │   ├── env/                   (环境定义)
│   │   │   ├── frozen_lake/      FrozenLake 网格世界
│   │   │   ├── sokoban/           Sokoban 推箱子
│   │   │   ├── webshop/           WebShop 购物环境
│   │   │   ├── gem/               General Embedding Model 环境（math/code/qa）
│   │   │   ├── mcp/               MCP 工具环境
│   │   │   └── android/            Android 设备控制环境
│   │   │       ├── remote_android.py        RemoteAndroidEnv（远端 Android 执行）
│   │   │       ├── remote_mobileworld.py    RemoteMobileEnv（MobileWorld）
│   │   │       ├── remote_multi_android.py   多 server 封装
│   │   │       ├── remote_multi_mobileworld.py
│   │   │       ├── GuiTaskEvalManager.py     HTTP 任务分配与统计服务
│   │   │       ├── android.py
│   │   │       ├── app_modular.py
│   │   │       └── mobile/
│   │   │           ├── mobilejudge.py         MobileJudge（多模态轨迹评判）
│   │   │           ├── curriculum_task_generator.py  课程任务生成器
│   │   │           └── prepare_data.py       Parquet 数据准备
│   │   └── env_manager/           (环境状态与任务管理)
│   │       ├── gui_traj_env_manager.py   GuiTrajEnvManager（轨迹级环境管理基类）
│   │       ├── android_step_env_manager.py  AndroidStepEnvManager（Android 步级管理）
│   │       ├── step_env_manager.py       StepEnvManager 基类
│   │       ├── base_env_manager.py        BaseEnvManager + RolloutCache
│   │       └── token_mask_utils.py
│   │
│   ├── distill/               (知识蒸馏管道)
│   │   ├── distill_pipeline.py
│   │   ├── distill_worker.py
│   │   └── various_divergence.py  KL / reverse KL / JS 散度
│   │
│   ├── dpo/                  (Direct Preference Optimization)
│   │   ├── dpo_pipeline.py
│   │   └── actor_worker.py
│   │
│   ├── sft/                  (Supervised Fine-Tuning)
│   │   ├── sft_pipeline.py
│   │   └── sft_worker.py
│   │
│   └── diffusion/            (Diffusion / Reward FL)
│       └── reward_fl/
│           ├── reward_fl_pipeline.py
│           └── actor_worker.py
│
├── platforms/                (7 文件) 硬件平台抽象
│   ├── platform.py           Platform 基类
│   ├── cuda.py / rocm.py / npu.py / cpu.py
│   └── unknown.py
│
├── third_party/             (107 文件) 厂商特定补丁
│   ├── vllm/                (4 个版本 × 各 6 个核心文件 = 24 文件)
│   │   ├── vllm_0_8_4/
│   │   ├── vllm_0_10_0/
│   │   ├── vllm_0_10_2/
│   │   └── vllm_0_11_0/
│   │       └── v1/          (vLLM v1 engine 支持)
│   │           ├── llm_engine.py
│   │           ├── worker.py
│   │           └── ray_distributed_executor.py
│   ├── vllm-xyq/            (镜像版，支持自定义 patch)
│   ├── sglang/              (4 个版本：v0410post2 / v046post4 / v052 / v054)
│   ├── deepspeed/           (offload_states 补丁)
│   └── megatron/            (tensor_parallel / optimizer 补丁)
│
└── utils/                   (40 文件) 共享工具
    ├── functionals.py        核心算法函数（compute_advantage, compute_token_reward, compute_approx_kl, RunningMoments 等）
    ├── kl_controller.py      AdaptiveKLController
    ├── checkpoint_manager.py  CheckpointManager（save / load / resume）
    ├── collectives/          分布式通信原语
    ├── context_parallel/     Ulysses Attention / AllToAll（CP 并行）
    ├── local_code/           沙箱代码执行（evaluator, execute_utils, pass_k_utils）
    ├── dynamic_batching.py   动态批处理
    ├── sequence_packing.py   序列打包
    ├── metrics/              MetricsManager（性能指标追踪）
    ├── tracking.py           SwanLab / WandB / TensorBoard 集成
    └── logging.py / hash_utils.py / ray_utils.py 等
```

---

## 3. 核心逻辑详解

### 3.1 数据流：从输入到输出的完整链路

#### RLVR Pipeline 数据流

```
JSON数据集 (data/*.jsonl)
    │
    ▼
datasets.load_dataset("json")          [rlvr_pipeline.py:143]
    │
    ▼
preprocess_dataset()                   [rlvr_pipeline.py:157-174]
  - chat_template 编码
  - prompt_len 过滤
  - update_dataset_domain (tag -> domain 映射)
    │
    ▼
domain_datasets[domain]                [rlvr_pipeline.py:171-178]
  - 按 domain 过滤: math_rule / code_sandbox / llm_judge / ifeval / crossthinkqa
    │
    ▼
DynamicSamplingScheduler.remote()      [rlvr_pipeline.py:260-279]
  - 每个 domain 一个 scheduler
  - 接收 actor_cluster + reward_cluster + dataset
    │
    ▼
┌── RLVRPipeline.run() 主循环 ──────────────────────────────────────────────────┐
│  每个 global_step:                                                              │
│  1. model_update(): ModelUpdateGroup.model_update()                              │
│     - actor_train 权重同步到 actor_infer                                        │
│  2. scheduler.get_batch(): DataProto (prompts)                                  │
│     - vLLM/SGLang 推理生成 responses                                           │
│  3. reward 计算: RewardWorker.compute_reward()                                   │
│     - math: MathRuleRewardWorker                                                │
│     - code: CodeSandboxRewardWorker (本地沙箱)                                   │
│     - llm_judge: LLMJudgeRewardWorker                                           │
│  4. ref_log_probs: Reference.compute_log_probs()                                 │
│     - KL divergence 计算                                                         │
│  5. old_log_probs: ActorTrain.compute_log_probs()                                │
│     - 在 enable_old_logprobs_recompute=True 时触发                               │
│  6. compute_token_reward(): token 级奖励展开 + KL penalty                        │
│  7. compute_advantage(): 优势估计 (GAE / Reinforce / GRPO / TOPR 等)          │
│  8. ActorWorker.train_step(): PPO/PG loss 反向更新                               │
│  9. do_checkpoint(): checkpoint 保存 + tracker 日志                                │
└─────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
输出: Checkpoint + TensorBoard/WandB/SwanLab 日志
```

#### Agentic Pipeline 数据流

```
Task JSON files (data/tasks/)
    │
    ▼
prepare_data.py -> train.parquet / val.parquet
    │
    ▼
GuiTaskEvalManager (FastAPI HTTP 服务)     [roll/pipeline/agentic/env/android/GuiTaskEvalManager.py]
  - /initialize: 初始化任务列表
  - /get_task: 按 GROUP_SIZE 分发烧配任务
    │
    ▼
RolloutScheduler / GroupQueueManager       [roll/distributed/scheduler/rollout_scheduler.py:39-79]
  - task -> group queue 映射
  - 每个 group 收集 group_size 条 rollout 后才算完成
    │
    ▼
EnvironmentWorker (Ray actor)               [roll/pipeline/agentic/environment_worker.py]
  - 多 EnvStateManager (线程) 管理多个环境实例
  - 通过 EnvManager 与远端服务交互
    │
    ▼
EnvManager (GuiTrajEnvManager / AndroidStepEnvManager)
  - reset(): 从 task manager 获取 episode_id，初始化环境
  - make_decision(): LLM 推理生成动作
  - step(): 与环境交互
  - formulate_rollouts(): 轨迹 -> DataProto
    │
    ▼
DataProto batch:
  - batch: input_ids, attention_mask, response_mask, scores, infer_logprobs
  - non_tensor_batch: episode_scores, step_scores, env_ids, group_ids, state_hash, traj_id
  - meta_info: task, metrics
    │
    ▼
AgenticPipeline.run() 主循环:
  1. train_rollout_scheduler.get_batch(): 拉取 rollout
  2. compute_discounted_returns() (gigpo / step_reinforce)
  3. compute_response_level_rewards(): batch 归一化
  4. compute_token_reward(): token 级奖励展开
  5. agentic_compute_advantage(): GRPO/Reinforce
  6. ActorWorker.train_step(): 反向更新
  7. eval: val_rollout_scheduler.get_batch(): 评测统计
```

### 3.2 模型/算法架构：组件关系图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RLVRPipeline / AgenticPipeline                     │
│                     (roll/pipeline/rlvr/rlvr_pipeline.py)                   │
│                     (roll/pipeline/agentic/agentic_pipeline.py)              │
│                                                                             │
│  self.actor_train (Cluster)   ──model_update()──▶  self.actor_infer (Cluster)│
│  self.actor_train (Cluster)   ──train_step()──▶   更新权重                   │
│  self.reference (Cluster)     ──compute_log_probs()──▶ ref_log_probs         │
│  self.critic (Cluster)        ──compute_values()──▶ values (GAE 时)         │
│  self.rewards[domain] (Cluster) ──compute_reward()──▶ response_level_rewards │
│  self.kl_ctrl (AdaptiveKLController)                                         │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ 共享同一套角色框架
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Cluster                                          │
│                   (roll/distributed/executor/cluster.py)                      │
│                                                                             │
│  - 管理 N 个 Worker ray actor                                                │
│  - _bind_worker_method(): 将 Worker 方法通过 Dispatch 绑定到 Cluster 级别      │
│  - dp_size / tp_size / pp_size 属性                                         │
│  - model_update_groups: 权重同步映射                                         │
│  - rank2devices / worker2nodes                                             │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ 持有同一个 strategy 实例
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       InferenceStrategy (基类)                               │
│              (roll/distributed/strategy/strategy.py:19-79)                  │
│                                                                             │
│  抽象接口:                                                                   │
│    initialize(*args, **kwargs)         初始化模型                             │
│    forward_step(batch, forward_func)   训练前向                               │
│    generate(*args, **kwargs)          生成推理                               │
│    add_request(command, data)          添加推理请求                           │
│    start_server() / stop_server()      vLLM server 管理                      │
│    save_checkpoint() / load_checkpoint()                                     │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ 策略工厂创建
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
┌──────────────────┐ ┌──────────────────┐ ┌────────────────────────┐
│ vLLMStrategy     │ │ SGLangStrategy   │ │ DeepSpeedStrategy      │
│ (third_party/vllm│ │ (third_party/    │ │ (deepspeed_strategy.py)│
│  各版本补丁)      │ │  sglang/)        │ │                        │
│                  │ │                  │ │                        │
│ GPU: 推理专用     │ │ GPU: 推理专用     │ │ GPU: 训练 + ZeRO 优化  │
└──────────────────┘ └──────────────────┘ └────────────────────────┘
                                              ┌────────────────────────┐
                                              │ MegatronStrategy       │
                                              │ (megatron_strategy.py) │
                                              │                        │
                                              │ TP / PP / CP / EP      │
                                              └────────────────────────┘
```

### 3.3 关键函数与实现细节

#### 3.3.1 DataProto 协议 (`roll/distributed/scheduler/protocol.py`)

DataProto 是整个框架的数据交换标准协议，定义在 `protocol.py:162-858`，包含三个核心字段：

```python
@dataclass
class DataProto:
    batch: TensorDict           # torch.Tensor 字典，按 batch_size[0] 组织
    non_tensor_batch: Dict      # numpy object 数组（字符串、列表等非 tensor 数据）
    meta_info: Dict             # 元信息（global_step, metrics 等）
```

**关键方法**：

| 方法 | 位置 | 说明 |
|------|------|------|
| `__getitem__(item)` | protocol.py:191 | 支持 int/slice/list/np.ndarray/torch.Tensor 索引 |
| `select()` / `pop()` | protocol.py:365/483 | 按 key 提取/弹出子集 |
| `reorder(indices)` | protocol.py:716 | in-place 重排序 |
| `group_by(keys)` | protocol.py:726 | 按字段分组返回 dict[DataProto] |
| `concat(data_list)` | protocol.py:646 | 拼接多个 DataProto |
| `materialize_concat()` | protocol.py:816 | 从 Ray ObjectRef 拉取并 concat |
| `make_iterator()` | protocol.py:564 | 生成 DataLoader 迭代器 |
| `chunk(chunks)` | protocol.py:603 | 按 dim=0 分块 |
| `repeat(repeat_times)` | protocol.py:772 | 复制数据 |

#### 3.3.2 优势估计 (`roll/utils/functionals.py`)

`compute_advantage()` 函数（`functionals.py`，约 850+ 行）支持多种优势估计器：

```python
def compute_advantage(
    data: DataProto,
    gamma: float = 1.0,          # 折扣因子
    lambd: float = 0.95,         # GAE lambda
    adv_estimator: str,          # gae / reinforce / grpo / gigpo / step_reinforce
    advantage_clip: float,
    whiten_advantages: bool,
    whiten_rewards: bool,
    response_mask: Tensor,
) -> DataProto
```

**支持的 adv_estimator**：

- `gae`：Generalized Advantage Estimation，结合 value function
- `reinforce`：标准策略梯度（`roll/pipeline/rlvr/rlvr_pipeline.py` 默认）
- `grpo`：Group Relative Policy Optimization（同 prompt 组内归一化）
- `gigpo`：Stepwise Generalized Policy Optimization（Agentic 场景，`agentic/utils.py:59`）
- `step_reinforce`：逐步强化（Agentic 场景）
- `agentic_reinforce`：Agentic 专用强化

#### 3.3.3 Token 级奖励计算 (`roll/utils/functionals.py`)

`compute_token_reward()` 将 response 级奖励展开到每个 token，并应用 KL penalty：

```python
def compute_token_reward(
    data: DataProto,
    config: RLVRConfig,
    kl_ctrl: AdaptiveKLController,
) -> Tuple[DataProto, Dict]
```

关键步骤：
1. `reward_clip` 裁剪极端奖励值
2. 按 `norm_mean_type`（batch / group / running）和 `norm_std_type` 归一化
3. 将 response reward 填充到 response 区间所有 token
4. `add_token_level_kl=True` 时在 token 级加入 KL 惩罚
5. `kl_ctrl.update(kl)` 自适应调整 KL 系数

#### 3.3.4 Actor Worker Loss (`roll/pipeline/rlvr/actor_worker.py`)

`ActorWorker.loss_func()` 是 RL 策略更新的核心（`actor_worker.py:11-154`）：

```python
def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
    log_probs = self.strategy.op_compute_log_probs(logits, input_ids, attention_mask)
    old_log_probs = self.get_old_log_probs_with_cache(data, log_probs)
    infer_log_probs = data.batch.get("infer_logprobs", old_log_probs)

    # KL penalty: policy vs reference
    kl_loss = compute_approx_kl(log_probs, ref_log_probs, mask, kl_penalty="k3")

    # PPO clipped surrogate loss
    ratio = (log_probs - old_log_probs).exp()
    clipped_ratio = torch.clamp(ratio, 1-pg_clip, 1+pg_clip)
    pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # Dual clip (PPO枯叶)
    if self.pipeline_config.dual_clip_loss:
        pg_loss = torch.where(advantages > 0, pg_loss, torch.clamp(pg_loss, clip_low))

    # 可选: train/infer ratio mask / difficulty mask / length penalty
    loss = (pg_loss * final_response_mask).mean()
```

#### 3.3.5 Agentic Advantage 计算 (`roll/pipeline/agentic/utils.py`)

`agentic_compute_advantage()` 支持 GRPO 和 Reinforce 两种模式：

```python
def agentic_compute_advantage(batch: DataProto, config: AgenticConfig) -> DataProto:
    # GRPO: group 内归一化
    if config.adv_estimator == "grpo":
        group_mean = batch.batch["response_level_rewards"].group_by("traj_group_id").mean()
        group_std = batch.batch["response_level_rewards"].group_by("traj_group_id").std()
        advantages = (rewards - group_mean) / (group_std + eps)
    # Reinforce: 基础策略梯度
    advantages = rewards  # 无 baseline
```

### 3.4 主训练循环

#### RLVRPipeline.run() (`rlvr_pipeline.py:426-769`)

每个 global_step 的完整流程（约 350 行核心循环体）：

```
for global_step in range(max_steps):
    1. [tps_timer] ──── 计时开始 ──────────────────────────────────────────
    │
    2. [step_stop_server] ── 停止 actor_infer server (async_pipeline 时)
    │
    3. [step_model_update] ── ModelUpdateGroup.model_update()
       │  - 从 src (actor_train) 广播到 tgt (actor_infer)
       │  - 按 pp_rank 分组并发广播
       │  - device_mapping 映射
       │
    4. [optional] val(): 评测（eval_steps 周期）
    │
    5. [step_generate] ──── 生成阶段 ────────────────────────────────────────
       │
       5.1. actor_infer.start_server(): 启动 vLLM/SGLang server
       │
       5.2. scheduler.start_sampling(): 每个 domain scheduler 开始采样
       │      (async_pipeline 时异步启动)
       │
       5.3. scheduler.get_batch(): 收集 rollout
       │      - vLLM batch generate
       │      - reward 计算 (math/code/llm_judge)
       │      - 返回 DataProto (prompts + responses + rewards)
       │
       5.4. DataProto.concat(): 合并多 domain batch
       │
       5.5. dump_rollout_to_specific_path(): 保存 rollout (可选)
       │
    6. [cal_ref_log_probs] ── Reference.compute_log_probs() (KL 计算)
       │  - ref_log_probs 存入 batch["ref_log_probs"]
       │
    7. [cal_old_log_probs] ── ActorTrain.compute_log_probs()
       │  - old_log_probs 存入 batch["old_log_probs"]
       │  - critic.compute_values() (GAE 时)
       │
    8. batch_by_domain ──── 分 domain 处理 ───────────────────────────────────
       │
       8.1. get_sample_level_mask(): 难度 mask / 长度 mask
       │
       8.2. reward_postprocess(): 奖励后处理（clip, norm, 难度权重）
       │
       8.3. compute_token_reward(): token 级奖励展开 + KL penalty
       │
       8.4. compute_advantage(): 优势估计 (GAE/Reinforce/GRPO/TOPR)
       │
    9. [step_train] ──── 训练阶段 ───────────────────────────────────────────
       │
       9.1. critic.train_step(): 训练 value function (adv_estimator=gae)
       │
       9.2. actor_train.train_step(): PPO/PG loss 反向更新
       │
    10. do_checkpoint(): checkpoint 保存
    │
    11. tracker.log(): TensorBoard / WandB / SwanLab 记录
    │
    12. [tps_timer] ──── 计算 tokens/sec 吞吐 ──────────────────────────────
```

#### AgenticPipeline.run() (`agentic_pipeline.py:134-319`)

与 RLVRPipeline 类似，但核心差异在 rollout 阶段——数据来源于环境交互而非文本生成：

```
for global_step in range(max_steps):
    1. critic.offload_states() + actor_train.offload_states()
    │
    2. train_rollout_scheduler.suspend(): 暂停推理
    │
    3. model_update(): actor_train -> actor_infer
    │
    4. actor_infer.start_server(): 启动 vLLM/SGLang server
    │
    5. [rollout_timer] ──── 环境交互阶段 ─────────────────────────────────
       │
       5.1. train_rollout_scheduler.get_batch(batch_size): 拉取 rollout
       │      - EnvironmentWorker 多线程执行
       │      - GroupQueueManager 聚合 group_size 条轨迹
       │      - 返回 DataProto (input_ids + responses + scores)
       │
       5.2. dump_rollout_trajectories(): 保存轨迹（可选）
       │
    6. compute_discounted_returns(): gigpo / step_reinforce 时折扣回报
       │  - 将 step_scores 展开为 discounted_returns
       │
    7. adjust_batch(): 过滤/调整 batch
       │
    8. [cal_ref_log_probs] ── Reference.compute_log_probs() (KL 计算)
    │
    9. [cal_old_log_probs] ── ActorTrain.compute_log_probs()
       │  - old_log_probs 存入 batch["old_log_probs"]
       │  - critic.compute_values() (GAE 时)
       │
    10. get_agentic_response_level_mask(): 计算 response 级 mask
    │
    11. compute_response_level_rewards(): 奖励归一化（grouping by traj_group_id）
        │  - reward_normalization.grouping: traj_group_id / batch / state
        │  - norm_mean_type / norm_std_type: group / batch / None
        │
    12. compute_token_reward(): token 级奖励展开
    │
    13. agentic_compute_advantage(): GRPO / Reinforce / GAE 优势估计
        │
    14. [train_timer] ──── 训练阶段 ────────────────────────────────────────
        │
        14.1. critic.train_step(): 训练 value function (adv_estimator=gae)
        │
        14.2. actor_train.train_step(): PPO/PG loss 反向更新
        │
    15. compute_data_metrics(): 计算数据质量指标
    │
    16. do_checkpoint() + tracker.log()
```

### 3.5 优势估计与奖励处理

#### 奖励归一化配置 (`roll/pipeline/agentic/agentic_config.py`)

```python
@dataclass
class RewardNormalizationConfig:
    grouping: str      # "state" / "batch" / "inductive"
    norm_mean_type: Literal["batch", "group"] | None  # 归一化均值来源
    norm_std_type: Literal["batch", "group"] | None   # 归一化标准差来源
```

**grouping 选项说明**：

| 值 | 含义 | 典型用途 |
|----|------|----------|
| `"traj_group_id"` | 按轨迹分组归一化 | Agentic 场景（每组内同任务） |
| `"batch"` | 全 batch 归一化 | 标准 RLVR |
| `"state"` | 按状态归一化 | 进阶用法 |

**method 已废弃**：请直接配置 `norm_mean_type` 和 `norm_std_type`，旧字段仅作向后兼容。

#### RLVR 中的 TOPR 支持 (`roll/pipeline/rlvr/rlvr_config.py`)

```python
# rlvr_config.py:141-144
use_topr_loss: bool = field(default=False)  # http://arxiv.org/abs/2503.14286
# TOPR Neg loss
use_topr_neg_loss_coef: float = field(default=0.0)
```

#### KL Controller (`roll/utils/kl_controller.py`)

```python
class AdaptiveKLController:
    def update(self, kl: float) -> None:
        # Adaptive KL 控制: kl_horizon 内逼近 target_kl
```

### 3.6 Agentic 训练架构

#### EnvironmentWorker (`roll/pipeline/agentic/environment_worker.py:28-79`)

```python
class EnvironmentWorker(Worker):
    """
      Within a group, all environments share identical states by using the same seed.
      To reduce the overhead of dedicating one process per environment, parallelism is redesigned as **process + threads** :
      - One `EnvironmentWorker` holds multiple `EnvStateManager`s.
      - Each `EnvStateManager` manages the rollout loop for a single environment.
      - `EnvStateManager.run_rollout_loop` runs inside dedicated threads.
        TODO: GiGPO: https://arxiv.org/abs/2505.10978
    """

    env_managers: Dict[int, BaseEnvManager]  # rank -> EnvStateManager
    thread_lock: threading.Lock               # 线程安全锁
    output_queue: GroupQueueManager           # rollout 输出队列

    # 轻量级内存监控（仅日志，不影响主流程）
    _monitor_thread: Thread                   # 低频采样线程
    _monitor_interval_sec: float = 30.0        # 采样间隔（环境变量 ROLLOUT_CACHE_MONITOR_INTERVAL）
    _obj_probe_every_n: int = 2              # 对象分布采样频率（ROLLOUT_OBJ_PROBE_EVERY_N）
```

**多线程并行策略**：
- 每个 `EnvironmentWorker` 是一个 Ray actor 进程（GPU 无关，CPU 密集）
- 内部维护多个 `EnvStateManager`（子线程），每个管理一个环境实例
- 同一 group 内的所有环境共享相同的 `group_seed`，确保任务配置一致
- 可通过 `max_env_step_concurrent` 控制环境 step 的并发上限（`env_action_limiter`）

#### RolloutScheduler + GroupQueueManager (`roll/distributed/scheduler/rollout_scheduler.py:39-79`)

```python
@dataclass
class GroupData:
    group_id: int
    episode_id: int
    create_step: int
    task: str = ""
    rollouts: List[DataProto] = field(default_factory=list)
    running_rollouts: int = 0


class GroupQueue:
    """
    管理一个 group 的 rollout 收集。
    每个 group 对应一个 task，同一 group 内的 rollout 共享 task 配置。
    """
    group_id: int
    task: str
    group_size: int              # 收集多少条 rollout 才算完成
    group_size_redundancy: int    # 额外冗余数（允许部分失败）
    max_traj_per_env: Optional[int]  # 每个环境最大轨迹数（实际被强制设为 None）
    async_generation_ratio: int   # 异步轮次比例
    groups: Dict[int, GroupData]  # episode_id -> GroupData

    async_generation_ratio > 1 时支持异步：
    - advance_step() 每次推进 async_generation_ratio 个 episode slot
    - 过期 episode（step - create_step > async_generation_ratio）自动清理
    - get_episode_id() 返回仍在等待的 slot


class GroupQueueManager:
    # task -> group queue 映射
    # 每个 task 首次被请求时动态创建 GroupQueue
    # advance_step(): 广播 global_step 到所有活跃 queue
    # get_batch(): 等待所有 group 收集完成，返回聚合 DataProto
    # 支持 group_filter_cls 过滤低质量组，过滤后可从 GlobalTrajectoryCache 替换
```

**关键调度流程**：

```
RolloutScheduler.get_batch():
  1. 首次启动：启动所有 env workers 的 rollout loop
  2. 广播 global_step（触发各 queue advance_step）
  3. GroupQueueManager.advance_step(step)
  4. RequestScheduler.resume() 放开推理
  5. 等待 GroupQueueManager.get_batch(group_size) 收集完成
```

#### GuiTrajEnvManager (`roll/pipeline/agentic/env_manager/gui_traj_env_manager.py`)

轨迹级环境管理器基类，通过 `gem.make()` 工厂实例化环境：

```python
class GuiTrajEnvManager(BaseEnvManager):
    # 核心状态
    rollout_cache: Optional[RolloutCache] = None
    group_seed: int = None
    episode_id: int = None

    # 环境并发控制
    use_thread_lock: bool        # CPU 密集操作时加锁
    max_env_step_concurrent: int  # 环境 step 并发上限（0=无限制）

    def __init__():
        # 通过 gem.make() 工厂创建环境（支持 frozen_lake / sokoban / android 等）
        self.env = gem.make(env_id=self.env_config["env_type"], **self.env_config['config'])

        # 工具调用包装（可选）
        if "tool_wrapper" in self.env_config:
            self.env = tool_wrapper(self.env, ...)

        # LLM 调用代理（PolicyProxy / OpenAIProxy / RandomProxy）
        self.llm_proxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env
        )

    def run_rollout_loop(self):
        # 单环境线程的主循环
        # 1. reset(): 向 GroupQueueManager.get_episode_id() 申请任务
        # 2. make_decision(): LLM 推理生成动作
        # 3. step(): 与环境交互
        # 4. 终止后 formulate_rollouts() 转 DataProto
        # 5. output_queue.put() 上传结果

    def save_trajectory(self, rollout: RolloutCache):
        # 写入 GlobalTrajectoryCache（Ray actor）
        # 仅在训练模式执行
        # 筛选条件：step 数最少且 episode_score > 0.5
```

### 3.7 AndroidWorld / MobileWorld GUI Agent 训练详解

本节详细梳理 AndroidWorld / MobileWorld 上 Qwen GUI Agent 在线 RL 训练与评测的完整链路，是框架在 `roll/pipeline/agentic/env/android/` 下的扩展实现。

#### 3.7.1 两条入口链路

框架提供训练和评测两条独立的入口脚本：

**训练链路**（`AgenticPipeline`）：

```
jyc/scripts/run_agentic_pipeline.sh
  → jyc/start_agentic_pipeline.py
    → AgenticPipeline.run()
      默认配置: agent_val_multiandroid_grpo（可切到 _lora）
```

**评测链路**（`AgenticRolloutPipeline`，无参数更新）：

```
jyc/scripts/evaluate_agentic_pipeline.sh
  → jyc/evaluate_agentic_pipeline.py
    → AgenticRolloutPipeline.run()
      默认配置:
        android_world → agent_val_multiandroid_grpo_evaluate
        mobile_world → agent_val_multimobileworld_evaluate
      MODEL_PARAM 可切换不同 env_manager 模板：
        voyager / guiowl / reflection
```

#### 3.7.2 在线训练关键参与者

| 角色 | 文件 | 职责 |
|------|------|------|
| **AgenticPipeline** | `agentic_pipeline.py:40` | RL 主循环（模型更新、取 rollout、奖励/优势、训练、评测） |
| **RolloutScheduler** | `rollout_scheduler.py` | 把"环境 rollout 系统"与"推理系统"拼接成统一 batch 接口 |
| **EnvironmentWorker** | `environment_worker.py:28` | 每个 worker 内多线程跑多个环境实例 |
| **GroupQueueManager** | `rollout_scheduler.py:115+` | 按 task/group 管理 rollout 收集与出队 |
| **RequestScheduler** | `generate_scheduler.py` | 单条请求级推理调度（支持 suspend/resume/abort） |
| **RemoteAndroidEnv** | `remote_android.py:31` | 对接远端 Android 服务，执行 reset/step、产出观测与奖励 |
| **RemoteMobileEnv** | `remote_mobileworld.py:36` | 对接 MobileWorld 服务（成功判定用 score >= 0.99） |
| **GuiTaskEvalManager** | `GuiTaskEvalManager.py` | FastAPI 任务分配与统计服务（train/eval 各一套 URL） |

#### 3.7.3 Qwen 模型在本框架中的接入方式

**配置层**（`jyc/agent_val_multiandroid_grpo*.yaml` / `jyc/agent_val_multimobileworld*.yaml`）：

```yaml
Model/pretrain: Qwen3-VL-*   # 常用 Qwen3-VL 模型
actor_train:   # 训练集群
actor_infer:   # 环境交互推理（常用 vllm）
reference:     # KL 参考策略（enable_reference: true）
```

训练配置通常使用 GRPO：
```yaml
adv_estimator: grpo
enable_reference: true
```

**推理代理层**（`android_step_env_manager.py` 中的 LLM 调用）：

```python
# 环境线程中的 LLM 调用统一经 create_llm_proxy()
self.llm_proxy = create_llm_proxy(
    generate_scheduler=self.generate_scheduler,  # RequestScheduler
    llm_proxy_config=self.worker_config.llm_proxy,
    tokenizer=self.tokenizer,
    env=self.env
)
# PolicyProxy 内部调用 RequestScheduler.generate_one_request
# 即：环境线程并不直接调用模型，而是走统一请求调度器
```

**多模态输入处理**：

```python
# EnvironmentWorker.initialize() 为每个 env_manager 创建 tokenizer + processor
self.tokenizer = default_tokenizer_provider(model_args=...)
self.processor = default_processor_provider(model_args=...)

# AndroidStepEnvManager.format_messages() 构造 Qwen-VL 风格 messages
# - 文本：system prompt（mobile_use tool schema）+ 历史动作摘要 + 当前任务目标
# - 图片：当前截图 base64 PNG
# - DataCollatorWithPaddingForMM 打包为 DataProto
```

#### 3.7.4 环境管理：AndroidWorld 与 MobileWorld

**GEM 环境注册**（`roll/pipeline/agentic/env/__init__.py`）：

```python
# 注册了以下 env_type：
"remote_multi_android"    # 多 server 封装的 AndroidWorld
"remote_multi_mobileworld"  # 多 server 封装的 MobileWorld
```

配置中的 `env_type` 通过 `gem.make()` 实例化。

**多 server 封装**（`remote_multi_android.py` / `remote_multi_mobileworld.py`）：

```python
# 根据 android_env_id 判断落在哪个 server 配置分片
self.console_port = self.console_ports[self.env_id % len(self.console_ports)]
self.grpc_port = self.grpc_ports[self.env_id % len(self.grpc_ports)]
# 解析 console/grpc ports
# 调父类（RemoteAndroidEnv / RemoteMobileEnv）完成真实环境初始化
```

**RemoteAndroidEnv / RemoteMobileEnv 共同职责**：

| 功能 | 实现 |
|------|------|
| HTTP 通信 | `/init` / `/reset` / `/step` / `/close` |
| 失败重试 | `tenacity` 装饰器（`@retry(stop=stop_after_attempt(3), wait=wait_exponential)`） |
| 故障恢复 | `close -> re-init` 循环 |
| 异常处理 | 调用 task manager `/return_task` 回收配额 |
| 终止上报 | 调用 task manager `/complete_task` |
| 轨迹保存 | 按 task 保存截图与元数据到本地目录 |

**差异点**：

| 方面 | AndroidWorld (`remote_android.py`) | MobileWorld (`remote_mobileworld.py`) |
|------|-----------------------------------|-------------------------------------|
| 观测来源 | `observation_np_b64` + dtype + shape | `screenshot_b64` (PNG base64) |
| 成功判定 | `done` + `is_success` | `score >= success_threshold`（默认 0.99） |
| 任务集合 | `TASK_LIST` / `TRAIN_TASK_LIST` / `FAIL_TASK_LIST` | `MOBILEWORLD_TASK_LIST` |

#### 3.7.5 Agent 上下文管理（核心：AndroidStepEnvManager）

**RolloutCache 状态维护**（`roll/pipeline/agentic/env_manager/base_env_manager.py`）：

```python
@dataclass
class RolloutCache:
    history: List[Dict]     # 每步 observation/response/reward/metrics/state_hash
    step: int               # 当前步数
    terminated: bool
    truncated: bool
    state_hash: str         # 状态哈希（用于去重/缓存）
```

**Prompt 与上下文构建**（`android_step_env_manager.py:38-49` 中的 system prompt 定义）：

```python
# 内置 mobile_use tool schema（Qwen-VL 风格）
SYSTEM_PROMPT = """
# Tools
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {
  "name": "mobile_use",
  "description": "Use a touchscreen to interact with a mobile device..."
  // actions: click, long_press, swipe, type, answer, system_button, wait, terminate
  // coordinates: (x, y) from top-left, resolution 999x999
}}
</tools>

# Response format
Response format for every step:
1) Thought: one concise sentence explaining the next move
2) Action: a short imperative describing what to do in the UI
3) <tool_call>{"name": "mobile_use", "arguments": {...}}</tool_call>

Rules:
- Output exactly in the order: Thought, Action, <tool_call>
- Be brief: one sentence for Thought, one for Action
- If finishing, use action=terminate
"""

# format_messages() 构造流程：
# 1. system prompt = SYSTEM_PROMPT
# 2. 从 history 提取最近 keep_last_k 步动作摘要（默认 8）
# 3. 注入当前任务目标 (self.env.task["goal"])
# 4. 注入当前截图 (base64 image)
# 5. 可选注入 skill memory (android_utils.get_skill)
# 6. 生成 input_ids/attention_mask/position_ids
# 7. 写回当前 step cache
```

**决策与执行流程**：

```python
# make_decision(): 调用 llm_proxy 生成 response token ids
# step(): 把解码文本动作交给 env.step
#   - 环境返回新的 observation/reward/done
#   - 追加到 history
#   - 若 terminated/truncated，触发 rollout 完成
```

#### 3.7.6 Agent 与环境的交互循环

执行入口在 `GuiTrajEnvManager.run_rollout_loop()`（`AndroidStepEnvManager` 继承复用）：

```
单环境线程循环（run_rollout_loop）:
  1. GroupQueueManager.get_episode_id(): 申请任务分配
     - 返回 task / group_id / episode_id
  2. reset(): 环境 reset 到目标 task
  3. while not terminated:
     3.1. make_decision(): LLM 推理生成动作
     3.2. step(): 与环境交互
     3.3. 检查终止条件（terminated / truncated / max_steps）
  4. formulate_rollouts(): 轨迹转 DataProto
     - step-level: 每步拆成独立样本（训练用）
     - val 时可仅保留首步样本（减内存）
  5. output_queue.put(): 上传给 GroupQueue
  6. 继续下一 episode

异常 reset/step 产出 None rollout，并回收 episode slot / 调用 /return_task
```

#### 3.7.7 任务调度与 rollout 调度

**任务调度**（`GuiTaskEvalManager.py`，FastAPI HTTP 服务）：

| API 端点 | 说明 |
|----------|------|
| `POST /initialize` | 初始化 task 列表、GROUP_SIZE、n_task、seed |
| `GET /get_task` | 按 GROUP_SIZE 为单位分发烧配任务 |
| `POST /complete_task` | 记录成功率、平均步数、平均时间 |
| `POST /return_task` | 环境失败回退配额 |
| `GET /stats` | 汇总统计 |
| `GET /batch_stats` | 批次级统计 |

**调度策略要点**：
- 任务选择优先级：`total_attempts` 最少优先，再看 `assigned`
- 同分候选用 `seed + step` 的确定性哈希打散
- 到达 `N_TASK * GROUP_SIZE` 后返回 `finish`
- 支持定期持久化 `task_stats` 到 JSON（`SAVE_INTERVAL_SECONDS = 30s`）

**rollout 调度**（`RolloutScheduler.get_batch()`）：

```
RolloutScheduler.get_batch() 每轮:
  1. 首次启动：启动所有 env workers 的 rollout loop
  2. 广播 global_step
  3. GroupQueueManager.advance_step(step)
  4. RequestScheduler.resume() 放开推理
  5. 等待 GroupQueueManager.get_batch(batch_size, step) 收集完成

GroupQueueManager 关键点：
  - task -> group queue 映射（动态创建）
  - 每次 get_episode_id() 向 task manager /get_task
  - 每个 group 收到 group_size 条 rollout 才算完成
  - 支持 group_filter_cls 过滤低质量组
  - 过滤后可尝试从 GlobalTrajectoryCache 替换历史最佳轨迹
```

#### 3.7.8 rollout 生成数据结构（DataProto）

`AndroidStepEnvManager.formulate_rollouts()` 把轨迹拆成 step-level 样本：

**batch 字段（TensorDict）**：

| 字段 | 说明 |
|------|------|
| `input_ids` | tokenized 输入（含 system prompt + 历史 + 截图 + 当前目标） |
| `attention_mask` | attention mask |
| `position_ids` | position ids |
| `response_mask` | response 区间的 mask |
| `scores` | **step reward**（末 token 写入当前步奖励） |
| `infer_logprobs` | 可选：推理 log probabilities |

**non_tensor_batch 字段（np.object 数组）**：

| 字段 | 说明 |
|------|------|
| `episode_scores` | episode 级最终奖励 |
| `step_scores` | 每步的奖励列表 |
| `tags` | 环境类型标签 |
| `env_ids` | 环境 ID |
| `group_ids` | group ID |
| `state_hash` | 状态哈希 |
| `step` | 步数 |
| `traj_group_id` | 轨迹组 ID |
| `traj_id` | 轨迹 ID |
| `sample_uuid` | 全局唯一样本标识 |

**meta_info 字段**：

| 字段 | 说明 |
|------|------|
| `task` | 任务配置（含 goal、instruction） |
| `metrics` | 环境侧聚合统计（env_raw_score、env_raw_success 等） |

#### 3.7.9 rollout 系统与算法框架交互（训练主干）

`AgenticPipeline.run()` 的每个 global step 可概括为：

```
1. [suspend] ── 暂停 rollout / 推理切换
   - train_rollout_scheduler.suspend()
   - 模型更新前可停止 actor_infer server

2. [model_update] ── 模型同步
   - ModelUpdateGroup.model_update() 把 actor_train 参数同步到 actor_infer

3. [rollout] ── 拉取 rollout batch
   - train_rollout_scheduler.get_batch(rollout_batch_size)
   - 写入 trajectory dump

4. [compute_discounted_returns] ── 折扣回报（gigpo / step_reinforce）
   - step_scores 展开为 discounted_returns

5. [cal_response_norm_rewards] ── 奖励归一化
   - compute_response_level_rewards()（grouping by traj_group_id）
   - norm_mean_type / norm_std_type 配置生效

6. [cal_token_reward] ── Token 级奖励
   - compute_token_reward()（展开到 token，并可加 KL）

7. [compute_advantage] ── 优势估计
   - agentic_compute_advantage()（GRPO 走 reinforce-return）

8. [train] ── 训练
   - actor_train.train_step(batch)
   - 若 adv_estimator=gae 再训练 critic

9. [log + ckpt] ── 记录与 checkpoint
   - 各类 env/critic/token/system 指标
   - checkpoint 与 tracker log
```

**评测主干（无参数更新）**：

```python
# AgenticRolloutPipeline.run()
# 仅启动 actor_infer + val_env_manager
# 持续收集 rollout 并统计 val/* 指标
# 不做 actor/critic 反向更新
```

#### 3.7.10 AndroidWorld / MobileWorld 统一与差异

**统一点**：
- 统一 EnvManager（`AndroidStepEnvManager`）
- 统一调度（`RolloutScheduler` + `GroupQueueManager` + `RequestScheduler`）
- 统一 RL 训练主干（`AgenticPipeline`）

**差异点**：

| 维度 | AndroidWorld | MobileWorld |
|------|-------------|-------------|
| `env_type` | `remote_multi_android` | `remote_multi_mobileworld` |
| 远端服务返回 payload | `observation_np_b64` + dtype/shape | `screenshot_b64` (PNG) |
| 成功语义 | `done` + `is_success` | `score >= success_threshold` |
| 任务集合 | `TASK_LIST` | `MOBILEWORLD_TASK_LIST` |

**本质**：算法层完全复用，环境差异被封装在 RemoteEnv 层与少量配置中。

#### 3.7.11 MobileJudge（多模态轨迹评判）

`roll/pipeline/agentic/env/android/mobile/mobilejudge.py` 是轨迹级奖励评判模块：

```python
class MobileJudgeEvaluator:
    async def evaluate_episode(
        self,
        episode_input: JudgeEpisodeInput,
    ) -> JudgeEpisodeResult:
        """
        输入：轨迹截图序列 + 参考截图 + 动作历史 + 任务元信息
        输出：reward + success + feedback_strings + failure_reasons
        """
        # 多模态分析（当前截图 + 参考截图 + 参考指令）
        # 输出 response_text, reward, details
        # details 包含 key_points / thoughts / usage 等
```

**标准化输入**（`JudgeEpisodeInput` TypedDict）：

```python
class JudgeEpisodeInput(TypedDict, total=False):
    task_id: str
    instruction: str
    snapshot: str
    current_screenshot_paths: List[str]    # 轨迹截图序列
    final_screenshot_path: str             # 最终状态图（显式单列）
    reference_screenshot_paths: List[str]  # 参考截图
    reference_instruction: str              # 参考指令文本
    action_history: List[str]              # 动作历史
    action_thoughts: List[str]            # 思考过程
    env_signals: Dict[str, Any]           # 环境原始信号（辅助诊断）
    metadata: Dict[str, Any]
```

**标准化输出**（`JudgeEpisodeResult` TypedDict）：

```python
class JudgeEpisodeResult(TypedDict, total=False):
    task_id: str
    reward: float                          # 训练用奖励（建议归一化到 [0, 1]）
    success: bool                          # 给 task manager 统计用
    feedback_summary: str                  # 一句话总结
    feedback_strings: List[str]            # 详细文本反馈（供生成器消费）
    failure_reasons: List[str]            # 结构化失败原因
    dimension_scores: Dict[str, float]    # 多维奖励
    raw_judge_text: str
    usage: Dict[str, int]
```

#### 3.7.12 自进化模式（Self-Evolving Mode）

自进化模式是框架设计文档（`docs/self-evolve_zh.md`）中的规划功能，旨在训练过程中形成"任务执行 -> 奖励判定 -> 反馈聚合 -> 新任务生成 -> 新一轮训练"的闭环。

**核心设计原则**：
- 保持 `AgenticPipeline` / `RolloutScheduler` / `EnvManager` 主接口不变
- 将"任务是否成功"的最终裁决权从环境侧转移到 `MobileJudge`
- 课程任务生成器消费上一轮的数值成功率与文本反馈，自动产出下一批任务

**建议配置**：

```yaml
self_evolve:
  enabled: false
  round_update_interval: 1
  feedback_root: ./trajectories/self_evolve
  generated_task_root: ./data/tasks/generated/mobile_self_evolve
  parquet_root: ./data/self_evolve_parquet
  judge:
    enabled: true
    reward_scale: 1.0
    write_json: true
  generator:
    enabled: true
    use_text_feedback: true
    use_success_rate: true
```

**自进化循环拓扑**：

```
Round_t Task Pool
  → Task Manager / RolloutScheduler
  → EnvManager → RemoteMobileEnv / RemoteAndroidEnv
  → Agent rollout trajectory
  → Self-Evolve Recorder → 写 judge_input.json
  → MobileJudge → 输出 judge_result.json
  → Feedback Aggregator → 汇总为 round_feedback.json
  → CurriculumTaskGenerator → 生成 next task json files
  → prepare_data.py → 生成 train.parquet
  → 下一轮 Round_(t+1) Task Pool
```

**推荐实施顺序**：
1. 第一步：奖励标准化（改 `mobilejudge.py`）
2. 第二步：episode 级接入（在 `android_step_env_manager.py` 接入 judge）
3. 第三步：round 级任务生成（完成 `round_feedback.json` + 改 `curriculum_task_generator.py`）
4. 第四步：Parquet 自动刷新与任务池切换（接通 `prepare_data.py` + pipeline 轮次更新）

---

## 4. 配置系统与超参数

### 4.1 配置继承链

```
RLVRConfig (rlvr_config.py:81)
  └── PPOConfig (base_config.py:306)
        └── BaseConfig (base_config.py:41)
              └── @dataclass 字段展开
```

### 4.2 关键超参数说明

#### 全局参数 (`BaseConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_steps` | 500 | 训练总步数 |
| `save_steps` | 50 | checkpoint 保存周期 |
| `eval_steps` | 10 | 评测周期 |
| `rollout_batch_size` | 128 | 每批次采样的 prompt 数 |
| `prompt_length` | 1024 | prompt 最大长度（padding） |
| `response_length` | None | response 最大长度 |
| `sequence_length` | prompt + response | 序列总长度 |
| `seed` | 42 | 随机种子 |
| `track_with` | tensorboard | 日志后端（wandb / swanlab / stdout） |

#### PPO/GRPO 算法参数 (`PPOConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `adv_estimator` | "gae" | 优势估计器：gae/reinforce/grpo/gigpo/step_reinforce |
| `gamma` | 1.0 | 折扣因子 |
| `lambd` | 0.95 | GAE lambda |
| `pg_clip` | 0.2 | PPO clip 范围 |
| `dual_clip_loss` | False | PPO枯叶（从 0.2 变为 0.1 的技巧） |
| `value_clip` | None | value 裁剪 |
| `kl_penalty` | "kl" | KL 惩罚类型：kl/abs/mse/full |
| `init_kl_coef` | 0.2 | KL 系数初始值 |
| `target_kl` | None | 自适应 KL 目标值 |
| `whiten_advantages` | False | 优势白化 |
| `reward_clip` | None | 奖励裁剪 |
| `use_reward_scaling` | False | 奖励缩放 |
| `entropy_loss_coef` | 0 | 熵正则化系数 |
| `ppo_epochs` | 1 | 每批数据的更新轮数 |
| `max_grad_norm` | 1.0 | 梯度裁剪 |

#### RLVR 特有参数 (`RLVRConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_return_sequences_in_group` | 1 | 同 prompt 的采样数（用于方差缩减） |
| `is_num_return_sequences_expand` | False | 是否将 num_return_sequences 复制 prompt |
| `async_generation_ratio` | 0 | 异步管道比率（> 0 启用异步） |
| `add_token_level_kl` | False | token 级 KL 惩罚 |
| `difficulty_loss_weight` | False | 难度加权 loss |
| `length_loss_weight` | False | 长度加权 loss |
| `use_topr_loss` | False | TOPR 算法 |
| `use_policy_loss_type` | "PPO" | PPO 或 PG |
| `importance_sampling` | "token" | token 级或 seq 级重要性采样 |

#### Agentic 参数 (`AgenticConfig`)

| 参数 | 说明 |
|------|------|
| `adv_estimator` | "grpo" / "gigpo" / "step_reinforce" |
| `train_env_manager` | `EnvManagerConfig`（max_env_num_per_worker, num_env_groups, group_size, tags） |
| `val_env_manager` | 同上 |
| `enable_reference` | 是否启用 reference cluster（计算 KL） |
| `reward_normalization.grouping` | "traj_group_id" / "batch" / "state" |

#### AndroidWorld/MobileWorld 特有配置

```yaml
# train_env_manager 配置示例
train_env_manager:
  max_env_num_per_worker: 16        # 每个 worker 最大环境数
  num_env_groups: 128               # 环境分组数
  group_size: 8                     # 每组 rollout 数（GRPO 方差缩减）
  tags: [FrozenLake, AndroidWorld] # 可用环境标签
  num_groups_partition: [128]      # 各标签的分组数分配

# val_env_manager 配置示例
val_env_manager:
  max_env_num_per_worker: 32
  num_env_groups: 1024
  group_size: 1                     # 评测时必须为 1（greedy 输出相同）

# 环境配置示例（custom_envs）
custom_envs:
  AndroidWorldTask:
    env_type: remote_multi_android
    max_steps: 50
    max_tokens_per_step: 128
    env_manager_cls: AndroidStepEnvManager
    group_size: 8
    service_url: http://localhost:18000
    task_manager_train_url: http://localhost:5001
    task_manager_eval_url: http://localhost:5002
    console_ports: [16384, 16385, ...]
    grpc_ports: [16384, 16385, ...]
```

### 4.3 角色配置示例（YAML）

```yaml
# actor_train: 训练用 Actor (DeepSpeed / Megatron)
actor_train:
  model_args:
    dtype: bf16
    attn_implementation: fa2
  training_args:
    learning_rate: 1.0e-6
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32
  strategy_args:
    strategy_name: megatron_train  # 或 deepspeed_train
    strategy_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      use_distributed_optimizer: true
  device_mapping: list(range(0,8))

# actor_infer: 推理用 Actor (vLLM / SGLang)
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      max_model_len: 8000
  device_mapping: list(range(0,8))

# reference: KL 参考策略
reference:
  strategy_args:
    strategy_name: hf_infer  # 无需 GPU 密集推理
  device_mapping: list(range(0,8))

# rewards: 多域奖励
rewards:
  math_rule:
    worker_cls: roll.pipeline.rlvr.rewards.math_rule_reward_worker.MathRuleRewardWorker
    tag_included: [deepmath_103k, aime]
  code_sandbox:
    worker_cls: roll.pipeline.rlvr.rewards.code_sandbox_reward_worker.CodeSandboxRewardWorker
    tag_included: [KodCode]
```

---

## 5. 快速启动

### 5.1 环境准备

```bash
# 克隆仓库
git clone https://github.com/alibaba/ROLL.git
cd ROLL

# 安装依赖（推荐 conda）
pip install -e ".[dev]"

# 或使用 Docker
docker pull alibaba/roll:latest
```

### 5.2 RLVR 训练

```bash
# Qwen2.5-7B RLVR 训练（Megatron + vLLM）
python examples/start_rlvr_pipeline.py \
    --config_name qwen2.5-7B-rlvr_megatron/rlvr_config

# 覆盖配置参数
python examples/start_rlvr_pipeline.py \
    rollout_batch_size=128 max_steps=1000 adv_estimator=reinforce

# 使用 GRPO（无需 critic）
python examples/start_rlvr_pipeline.py \
    adv_estimator=grpo ppo_epochs=4
```

### 5.3 Agentic 训练

```bash
# Sokoban 环境（GiGPO 训练）
python examples/start_agentic_pipeline.py \
    --config_name qwen2.5-0.5B-agentic/agentic_val_sokoban_gigpo

# FrozenLake 环境（GRPO）
python examples/start_agentic_pipeline.py \
    --config_name qwen2.5-0.5B-agentic/agent_val_frozen_lake

# 异步 Agentic 训练
python examples/start_agentic_pipeline.py \
    --config_name qwen2.5-0.5B-agentic/agent_val_frozen_lake_async
```

### 5.4 AndroidWorld / MobileWorld GUI Agent 训练

```bash
# AndroidWorld 训练（GRPO）
python jyc/start_agentic_pipeline.py \
    --config_name agent_val_multiandroid_grpo

# AndroidWorld 评测（无参数更新）
python jyc/evaluate_agentic_pipeline.py \
    --config_name agent_val_multiandroid_grpo_evaluate \
    --env_type android_world

# MobileWorld 评测
python jyc/evaluate_agentic_pipeline.py \
    --config_name agent_val_multimobileworld_evaluate \
    --env_type mobile_world

# 使用不同 agent 策略（voyager / guiowl / reflection）
python jyc/evaluate_agentic_pipeline.py \
    --config_name agent_val_multimobileworld_evaluate \
    --env_type mobile_world \
    --model_param voyager
```

### 5.5 评测（无需训练）

```bash
# Agentic Rollout 评测
python examples/start_agentic_rollout_pipeline.py \
    --config_name qwen2.5-0.5B-agentic/agentic_rollout_sokoban
```

### 5.6 配置系统说明

框架使用 Hydra 进行分层 YAML 配置，支持：
- 配置继承：`defaults: - ../config/deepspeed_zero3@_here_`
- 变量引用：`${deepspeed_zero3}` / `${response_length}`
- CLI 覆盖：`key=value` 格式
- 多环境配置：`actor_train` / `actor_infer` / `reference` / `rewards[domain]`

---

## 6. 注意事项与陷阱

### 6.1 依赖与版本

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.10+ | 类型注解使用 |
| PyTorch | 最新稳定版 | 基础深度学习框架 |
| Ray | 2.x | 分布式执行引擎 |
| Transformers | 最新 | tokenizer / model |
| vLLM | 0.8.4 / 0.10.0 / 0.10.2 / 0.11.0 | 推理后端 |
| SGLang | 0.4.10+ / 0.46 / 0.52 / 0.54 | 推理后端 |
| Megatron-LM | mcore adapter 版本 | 模型并行训练 |
| DeepSpeed | ZeRO 支持 | 分布式优化器 |
| hydra-core | 最新 | 配置管理 |
| datasets | 最新 | 数据集加载 |
| FastAPI | 最新 | GuiTaskEvalManager HTTP 服务 |
| tenacity | 最新 | RemoteAndroidEnv / RemoteMobileEnv 失败重试 |
| PIL | 最新 | 截图处理 |
| gem | 最新 | 环境抽象层（AndroidWorld/MobileWorld/Sokoban/FrozenLake 等） |

> **重要**：third_party 目录下包含多个版本的 vLLM 和 SGLang patch 文件（`roll/third_party/vllm/vllm_0_8_4/` 等），需根据实际安装的版本选择对应 patch。

### 6.2 常见陷阱

#### 6.2.1 设备映射错误

```yaml
# 错误：device_mapping 总和不等于 num_gpus_per_node
actor_train:
  device_mapping: list(range(0,16))  # 申请 16 GPU
num_gpus_per_node: 8                 # 但机器只有 8 GPU

# 正确：确保总设备数匹配
num_gpus_per_node: 16
```

#### 6.2.2 Megatron batch_size 可整除性

`roll/configs/base_config.py:250-268` 中，Megatron 策略会校验：
```python
validate_megatron_batch_size(
    batch_size=rollout_batch_size,
    num_gpus=len(actor_train.device_mapping),
    strategy_config=strategy_config,  # 包含 TP/PP/CP 配置
)
```
`rollout_batch_size` 必须能被 `num_gpus * TP * PP * CP` 整除。

#### 6.2.3 async_pipeline 必须 is_num_return_sequences_expand

```python
# rlvr_config.py:226-229
assert self.generate_opt_level == 1, "AsyncRLVRPipeline only support generate_opt_level 1"
if self.num_return_sequences_in_group > 1 and not self.is_num_return_sequences_expand:
    self.is_num_return_sequences_expand = True  # 自动修正
```

#### 6.2.4 old_logprobs_recompute 逻辑

`roll/configs/base_config.py:464-496` 中有复杂的自动决策逻辑：
- `backward_steps_per_rank > 1` → 自动启用 `enable_old_logprobs_recompute = True`
- `init_kl_coef > 0` → 自动启用 `enable_old_logprobs_recompute = True`

禁用缓存会导致每次训练需要重新计算 `old_log_probs`，增加显存和时间开销。

#### 6.2.5 Agentic group_size 与 val

```yaml
# 训练时：group_size > 1 用于 GRPO 方差缩减
train_env_manager:
  group_size: 8
  num_env_groups: 128

# 评测时：group_size 必须为 1（greedy 生成，相同 prompt 导致相同输出）
val_env_manager:
  group_size: 1
```

#### 6.2.6 AndroidWorld/MobileWorld 特有陷阱

1. **GroupQueueManager 中 max_traj_per_env 被强制设为 None**（`rollout_scheduler.py:94`）：
   ```python
   if self.max_traj_per_env is None:
       return  # 直接跳过，不限制轨迹数
   ```
   轨迹数限制依赖 group/filter 控制，而非配置。

2. **`remote_android.py` 中 task == all_task 分支**存在固定任务覆盖（`FilesDeleteFile`, `OsmAndFavorite`），可能影响任务覆盖面。

3. **`scores` 写入的是 step reward**：评测时若要严格 episode 指标，应查看 `non_tensor_batch["episode_scores"]` 而非 `batch["scores"]`。

4. **轨迹缓存（GlobalTrajectoryCache）链路**已接入，但 `GuiTrajEnvManager.save_trajectory` 对 `ObjectRef` 的比较逻辑建议二次核查。

5. **MobileWorld 成功判定阈值**：默认 `success_threshold = 0.99`，需确认远端服务返回的 score 是否满足此阈值。

6. **远端服务依赖**：训练和评测均依赖远端 Android 服务（`http://localhost:18000` 等），服务不可用时环境会持续重试（`tenacity`）。

7. **自进化模式**（`docs/self-evolve_zh.md`）仍为设计文档，代码实现需按四步实施。

### 6.3 未完成点

根据 README.md 的 Upcoming Features 和代码状态：

| 功能 | 状态 | 位置 |
|------|------|------|
| Async RLVR pipeline | 实现中 | `AsyncDynamicSamplingScheduler` 已存在 |
| FSDP2 | 实现中 | `fsdp_strategy.py` 待添加 |
| DeepSeekV3 支持 | 计划中 | - |
| SFT Pipeline | 实现中 | `roll/pipeline/sft/` |
| Distill Pipeline | 实现中 | `roll/pipeline/distill/` |
| VLM Agentic | 实现中 | `examples/qwen2.5-vl-3B-agentic/` |
| 自进化模式 | 设计阶段 | `docs/self-evolve_zh.md` |

### 6.4 性能调优建议

1. **显存不足**：优先降低 `rollout_batch_size`，而非 `per_device_train_batch_size`（梯度累积可弥补）
2. **推理吞吐低**：增大 `max_running_requests`（默认 128）和 `actor_infer` 的 `infer_batch_size`
3. **训练不稳定**：尝试 `dual_clip_loss: true`（PPO枯叶保护）和 `whiten_advantages: true`
4. **长序列OOM**：启用 `sequence_packing`（仅 student/teacher/sft_train 角色支持）和 `remove_padding`
5. **多域负载不均**：调整 `domain_interleave_probs` 比例，避免某些 reward worker 成为瓶颈
6. **AndroidWorld 训练**：确保远端 Android 服务稳定，可通过 `console_ports` / `grpc_ports` 列表配置多 server 分片提升并发
7. **AndroidWorld 评测**：调整 `group_size`（建议 1）和 `max_env_num_per_worker` 平衡吞吐与内存

---

## 附录：关键文件索引

| 文件 | 行数 | 核心内容 |
|------|------|----------|
| `roll/pipeline/rlvr/rlvr_pipeline.py` | ~820 | RLVRPipeline.run() 主循环 |
| `roll/pipeline/agentic/agentic_pipeline.py` | ~580 | AgenticPipeline.run() 主循环 |
| `roll/distributed/scheduler/protocol.py` | ~860 | DataProto 定义 |
| `roll/utils/functionals.py` | ~950 | compute_advantage / compute_token_reward 等 |
| `roll/pipeline/rlvr/actor_worker.py` | ~200 | ActorWorker.loss_func() |
| `roll/distributed/executor/model_update_group.py` | ~120 | ModelUpdateGroup 权重同步 |
| `roll/distributed/scheduler/rollout_scheduler.py` | ~620 | RolloutScheduler + GroupQueueManager |
| `roll/distributed/strategy/strategy.py` | ~300 | InferenceStrategy 基类 |
| `roll/configs/base_config.py` | ~500 | BaseConfig + PPOConfig |
| `roll/pipeline/agentic/env_manager/gui_traj_env_manager.py` | ~500 | GuiTrajEnvManager |
| `roll/pipeline/agentic/env_manager/android_step_env_manager.py` | ~400 | AndroidStepEnvManager |
| `roll/pipeline/agentic/environment_worker.py` | ~350 | EnvironmentWorker |
| `roll/pipeline/agentic/utils.py` | ~350 | agentic_compute_advantage 等 |
| `roll/pipeline/agentic/env/android/remote_android.py` | ~350 | RemoteAndroidEnv |
| `roll/pipeline/agentic/env/android/remote_mobileworld.py` | ~350 | RemoteMobileEnv |
| `roll/pipeline/agentic/env/android/GuiTaskEvalManager.py` | ~450 | FastAPI 任务分配服务 |
| `roll/pipeline/agentic/env/android/mobile/mobilejudge.py` | ~300 | MobileJudge 多模态评判 |
| `roll/pipeline/agentic/env/android/mobile/curriculum_task_generator.py` | ~300 | 课程任务生成器 |
| `roll/pipeline/agentic/env/android/mobile/prepare_data.py` | ~200 | Parquet 数据准备 |
