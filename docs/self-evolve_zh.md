# Self-Evolving Mode 架构设计

## 1. 目标与设计原则

本文面向当前 ROLL 在 AndroidWorld / MobileWorld 上的 GUI Agent 训练框架，设计一套与现有训练主干兼容的“自进化模式（Self-Evolving Mode）”。

核心目标有四个：

1. 在训练过程中形成“任务执行 -> 奖励判定 -> 反馈聚合 -> 新任务生成 -> 新一轮训练”的闭环。
2. 保持现有 `AgenticPipeline`、`RolloutScheduler`、`EnvManager`、`prepare_data.py`、`train_dataloader` 的主接口稳定，避免重写训练主干。
3. 将“任务是否成功”的最终裁决权从环境侧转移到奖励模块 `MobileJudge`，同时保留环境原始信号用于诊断。
4. 让课程任务生成器能够消费上一轮的数值成功率与文本反馈，自动产出下一批任务，并写回现有训练数据入口。

本文推荐采用“模式开关 + 最小侵入式扩展”的方案：

- 常规模式：沿用现有环境奖励、现有任务池与现有训练逻辑。
- 自进化模式：只在 episode 结束、任务统计、任务生成和数据刷新几个关键节点插入新链路。

建议新增统一配置段：

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

默认 `enabled: false`，以确保非自进化模式完全不受影响。

---

## 2. 总体架构设计

### 2.1 闭环拓扑

推荐把自进化模式设计为“回合制任务池更新”而不是“每条轨迹立刻改写任务池”。原因是当前框架的调度、统计和 `prepare_data.py` 都更适合按批次或回合更新。

一个自进化回合的拓扑如下：

```text
Round_t Task Pool
    ->
Task Manager / RolloutScheduler
    ->
EnvManager -> RemoteMobileEnv / RemoteAndroidEnv
    ->
Agent rollout trajectory
    ->
Self-Evolve Recorder
    -> 写 judge_input.json
MobileJudge
    -> 输出 judge_result.json
Feedback Aggregator
    -> 汇总为 round_feedback.json
Curriculum Task Generator
    -> 生成 next task json files
prepare_data.py
    -> 生成 train.parquet
下一轮 Round_(t+1) Task Pool
```

### 2.2 角色职责划分

建议按以下边界拆分：

1. 环境层
   负责执行动作、返回观测、记录原始环境信号、按 `snapshot` 恢复环境。

2. 奖励判定层 `MobileJudge`
   负责读取完整轨迹信息与参考信息，输出：
   - 数值奖励 `reward`
   - 成功标记 `success`
   - 文本反馈 `feedback_strings`
   - 结构化失败原因 `failure_reasons`

3. 反馈聚合层
   负责把 episode 级 judge 输出整理成 task 级统计，形成课程生成器可直接消费的本地 JSON。

4. 课程生成层 `CurriculumTaskGenerator`
   负责根据 task 级成功率与文本反馈生成下一轮任务，并保留 `snapshot`。

5. 数据准备层
   负责把新任务目录转换为 Parquet，保持 `train_dataloader` 无感接入。

### 2.3 与现有框架的兼容原则

为了兼容现有训练代码，推荐保持以下不变：

1. `AgenticPipeline.run()` 仍然消费 `RolloutScheduler.get_batch()` 返回的 `DataProto`。
2. `compute_discounted_returns`、`compute_response_level_rewards`、`compute_token_reward`、`agentic_compute_advantage` 不改主签名。
3. `prepare_data.py` 仍然是 Parquet 的唯一标准生成入口。
4. 任务配置文件仍然是 JSON，环境 reset 仍然从任务配置恢复。

关键兼容策略是：

- 自进化模式下，不要求训练主干理解“文本反馈”。
- 文本反馈只在“任务统计 -> 新任务生成”链路中使用。
- 训练主干继续只消费数值奖励。

---

## 3. 自进化循环设计

### 3.1 回合边界定义

推荐把“上一轮任务”定义为“当前 task pool 完成一次完整调度周期”。

与现有任务管理器对齐后，可用以下条件作为回合结束信号：

1. `GuiTaskEvalManager` 中所有任务达到当前轮次的 `N_TASK * GROUP_SIZE` 分配目标。
2. `/get_task` 开始返回 `finish`。
3. 当前 `RolloutScheduler.get_batch()` 收完最后一个 batch。

此时触发：

1. 汇总本轮 judge 结果。
2. 生成下一轮任务 JSON。
3. 运行 `prepare_data.py` 生成新的 `train.parquet`。
4. 重新初始化 task manager 的任务列表。
5. 开启下一轮训练。

### 3.2 Episode 内奖励流

当前框架默认把环境返回的 reward 写入 `history["reward"]`，再由 `formulate_rollouts()` 写入：

- `non_tensor_batch["step_scores"]`
- `non_tensor_batch["episode_scores"]`
- `batch["scores"]`

为了最小化改动，自进化模式建议采用“末步覆盖”方案：

1. 环境交互阶段：
   - 中间步 `reward` 固定记为 `0.0`。
   - 环境原始分数、done reason、是否成功等信息写入 `info["env_raw_score"]`、`info["env_raw_success"]`、`info["env_raw_reason"]`。

2. Episode 终止后：
   - 由 `MobileJudge` 读取轨迹。
   - 输出 `judge_reward`、`judge_success`、`feedback_strings`。
   - 将最后一步 `history[-1]["reward"] = judge_reward`。
   - 将 `episode_score = judge_reward`。

3. 训练阶段：
   - 现有算法仍按 `episode_scores/step_scores` 训练。
   - 非自进化模式下仍沿用环境 reward。

该方案的优点是：

1. `AgenticPipeline` 基本不需要理解新的 reward 来源。
2. `AndroidStepEnvManager.formulate_rollouts()` 的输出结构可以保持不变。
3. 训练奖励来自 judge，但环境原始信号仍可保留用于分析。

### 3.3 任务更新流

在回合结束后，推荐按以下顺序执行：

1. 读取本轮 `judge_result.json` 文件集合。
2. 聚合成 `round_feedback.json`。
3. 调用 `CurriculumTaskGenerator.generate_tasks_from_feedback(...)`。
4. 为每个新任务生成标准 JSON 配置文件。
5. 调用 `prepare_data.py` 生成下一轮 Parquet。
6. 重新指向新的 task pool。

---

## 4. 流水线改动方案

### 4.1 训练主干接入方式

建议不要在 `AgenticPipeline` 内部直接塞入大段生成逻辑，而是新增一个轻量的“自进化协调器”概念，例如：

```python
class SelfEvolveCoordinator:
    def on_episode_end(self, episode_artifacts) -> JudgeResult: ...
    def on_round_end(self, round_dir) -> RoundUpdateResult: ...
```

训练主干只负责在两个时机调用它：

1. episode 结束时调用 `on_episode_end`
2. 回合结束时调用 `on_round_end`

这样可以避免 `agentic_pipeline.py` 被任务生成逻辑污染。

### 4.2 环境、智能体、奖励模型、生成器的串联方式

推荐串联顺序如下：

#### 阶段 A：环境执行

1. `RolloutScheduler` 从 task manager 获取任务。
2. `GuiTrajEnvManager.reset()` 把 `target_task` 传给环境。
3. 环境仅根据任务里的 `snapshot` 快速恢复环境。
4. agent 完成一条轨迹。

#### 阶段 B：奖励判定

1. episode 结束后，EnvManager 组装 judge 输入：
   - 当前轨迹截图序列
   - 最终截图
   - 参考截图
   - 参考指令
   - 动作历史
   - 任务元信息
2. 写本地 `judge_input.json`。
3. 调用 `MobileJudge` 评估。
4. 写本地 `judge_result.json`。
5. 用 `judge_reward/judge_success` 回填 rollout 与 task manager。

#### 阶段 C：反馈聚合

1. 回合结束后扫描所有 `judge_result.json`。
2. 按 task 聚合：
   - success_rate
   - avg_reward
   - failure_reasons_topk
   - feedback_strings
   - representative examples
3. 输出 `round_feedback.json`。

#### 阶段 D：课程生成

1. `CurriculumTaskGenerator` 读取 `round_feedback.json`。
2. 对每个 snapshot 或每个上下文模板生成下一批任务。
3. 写出任务 JSON 文件。
4. 更新任务索引 JSON。

#### 阶段 E：数据准备

1. 调用 `prepare_data.py` 扫描新任务目录。
2. 生成新的 `train.parquet`。
3. 新一轮训练继续。

### 4.3 本地 JSON 中转设计

为满足“任务生成器与奖励模型之间的数据交互通过本地 JSON 文件进行中转”的约束，建议目录结构如下：

```text
trajectories/<timestamp>/self_evolve/
  round_0001/
    episodes/
      <task_id>/
        <episode_id>/
          judge_input.json
          judge_result.json
    round_feedback.json
    generated_tasks/
      <category>/<task_json_files...>
    parquet/
      train.parquet
      val.parquet
```

这样有几个好处：

1. 便于复盘每轮如何从反馈生成任务。
2. 奖励模块与任务生成器完全通过文件解耦。
3. 后续可以离线回放和重跑。

---

## 5. 环境初始化与任务格式规范

### 5.1 snapshot 作为唯一快速恢复入口

每个生成任务都必须包含 `snapshot` 字段，且环境初始化阶段只依赖该字段完成快速恢复。

推荐任务 JSON 最小格式：

```json
{
  "id": "settings_task_0001",
  "snapshot": "settings_main_v3",
  "instruction": "打开 Wi-Fi 设置页面并确认当前已看到可用网络列表",
  "source": "self_evolve",
  "metadata": {
    "parent_task_id": "settings_task_0000",
    "round_id": 3,
    "generator": "curriculum_task_generator"
  }
}
```

要求：

1. `snapshot` 必填。
2. `snapshot` 应是稳定的任务标识符，而不是临时文件路径。
3. 生成器改写任务时应尽量继承原始任务的 `snapshot`，除非明确切换到新上下文。

### 5.2 任务配置兼容现有环境

当前 `curriculum_task_generator.py` 已有：

- `config["id"]`
- `config["snapshot"]`
- `config["instruction"]`

建议在自进化模式下继续保持这三个字段为核心，不修改旧字段语义，只增加可选扩展字段，例如：

```json
{
  "original_task": "在搜索页输入关键词并查看结果",
  "reasoning": "上一轮成功率过高，因此提升任务难度并加入排序约束",
  "self_evolve": {
    "round_id": 4,
    "success_rate": 0.92,
    "avg_reward": 0.88,
    "feedback_ref": "round_0003/round_feedback.json"
  }
}
```

---

## 6. 奖励模块设计

### 6.1 MobileJudge 的职责升级

`roll/pipeline/agentic/env/android/mobile/mobilejudge.py` 当前已经具备：

1. 读取截图
2. 多模态分析
3. 输出 `response_text, reward, details`

自进化模式下建议把它升级为标准化 judge 服务接口，输出两类结果：

1. 数值奖励
2. 详细反馈文本

### 6.2 推荐的统一输入接口

建议定义一个通用输入对象 `JudgeEpisodeInput`：

```python
from typing import Any, Dict, List, Optional, TypedDict


class JudgeEpisodeInput(TypedDict, total=False):
    task_id: str
    instruction: str
    snapshot: str
    current_screenshot_paths: List[str]
    final_screenshot_path: str
    reference_screenshot_paths: List[str]
    reference_instruction: str
    action_history: List[str]
    action_thoughts: List[str]
    env_signals: Dict[str, Any]
    metadata: Dict[str, Any]
```

说明：

1. `current_screenshot_paths`
   轨迹过程中供 judge 使用的截图序列。

2. `final_screenshot_path`
   最终状态图，建议显式单列，避免 judge 自行猜测最后一张。

3. `reference_screenshot_paths`
   参考状态截图，用于支持“当前状态截图 + 参考截图 + 参考文本”的多模态输入要求。

4. `reference_instruction`
   参考指令文本，可来自人工模板、context review 模板或任务生成器保留字段。

5. `env_signals`
   记录环境原始 `score`、`done`、`reason`、`terminate_type` 等，仅用于辅助诊断，不作为最终成功判定来源。

### 6.3 推荐的统一输出接口

建议定义 `JudgeEpisodeResult`：

```python
class JudgeEpisodeResult(TypedDict, total=False):
    task_id: str
    reward: float
    success: bool
    feedback_summary: str
    feedback_strings: List[str]
    failure_reasons: List[str]
    dimension_scores: Dict[str, float]
    raw_judge_text: str
    usage: Dict[str, int]
    metadata: Dict[str, Any]
```

字段建议：

1. `reward`
   训练使用的最终数值奖励，范围建议固定在 `[0, 1]`。

2. `success`
   给 task manager 统计成功率使用的布尔值。

3. `feedback_summary`
   一句话总结，便于课程生成 prompt 压缩。

4. `feedback_strings`
   详细文本反馈列表，适合被生成器直接消费。

5. `failure_reasons`
   尽量结构化，如：
   - `"wrong_page"`
   - `"missing_confirmation"`
   - `"constraint_not_satisfied"`
   - `"hallucinated_completion"`

6. `dimension_scores`
   支持多维奖励，例如：
   - `goal_completion`
   - `ui_state_match`
   - `constraint_satisfaction`
   - `action_efficiency`

### 6.4 建议的 JSON 输入输出格式

`judge_input.json`：

```json
{
  "task_id": "mobile.settings_task_0001",
  "instruction": "打开 Wi-Fi 设置页面并确认当前已看到可用网络列表",
  "snapshot": "settings_main_v3",
  "current_screenshot_paths": [
    "step_000.png",
    "step_001.png",
    "step_002.png"
  ],
  "final_screenshot_path": "step_002.png",
  "reference_screenshot_paths": [
    "ref_0.png"
  ],
  "reference_instruction": "目标页面应展示 Wi-Fi 开关与网络列表",
  "action_history": [
    "open settings",
    "tap Wi-Fi"
  ],
  "env_signals": {
    "env_raw_score": 0.0,
    "env_raw_success": false,
    "done_reason": "agent_terminate"
  }
}
```

`judge_result.json`：

```json
{
  "task_id": "mobile.settings_task_0001",
  "reward": 1.0,
  "success": true,
  "feedback_summary": "任务已完成，目标页面与参考要求一致。",
  "feedback_strings": [
    "最终截图中已出现 Wi-Fi 页面。",
    "可见网络列表，满足目标要求。"
  ],
  "failure_reasons": [],
  "dimension_scores": {
    "goal_completion": 1.0,
    "ui_state_match": 1.0,
    "constraint_satisfaction": 1.0
  }
}
```

### 6.5 奖励注入策略

为了兼容现有 `formulate_rollouts()`，建议：

1. 中间步骤 reward 为 `0.0`
2. 最后一步 reward 为 `judge_result["reward"]`
3. `episode_scores = judge_result["reward"]`
4. `step_scores = [0.0, ..., judge_reward]`

这样：

- GRPO/PPO 侧不需要知道奖励来自环境还是 judge。
- 文本反馈不会污染训练 batch。
- judge 输出失败原因时仍可通过 `meta_info` 或 JSON 文件保留。

---

## 7. 课程任务生成器设计

### 7.1 生成器输入要求

课程任务生成器必须支持同时接收：

1. 数值成功率
2. 文本反馈
3. 原始任务内容
4. snapshot
5. 可选参考截图或参考上下文

推荐输入对象 `GeneratorRoundInput`：

```python
class GeneratorRoundInput(TypedDict, total=False):
    round_id: int
    feedback_file: str
    context_templates: List[Dict[str, Any]]
    existing_task_files: List[str]
    target_output_dir: str
    index_dir: str
```

其中 `feedback_file` 指向聚合后的 `round_feedback.json`。

### 7.2 反馈聚合文件格式

建议定义 `round_feedback.json`：

```json
{
  "round_id": 3,
  "task_performances": [
    {
      "task_id": "mobile.settings_task_0001",
      "snapshot": "settings_main_v3",
      "instruction": "打开 Wi-Fi 设置页面并确认当前已看到可用网络列表",
      "success_rate": 0.25,
      "avg_reward": 0.31,
      "feedback_strings": [
        "多次停留在设置首页，未进入 Wi-Fi 子页面。",
        "最终截图没有出现目标网络列表。"
      ],
      "failure_reasons": [
        "wrong_page",
        "missing_target_state"
      ]
    }
  ]
}
```

### 7.3 生成策略建议

课程生成器可按成功率区间进行演化：

1. `success_rate >= 0.8`
   生成更难、更具约束的新任务。

2. `0.3 <= success_rate < 0.8`
   生成同难度但不同场景任务，增强泛化。

3. `success_rate < 0.3`
   拆分为更基础的 prerequisite 任务，显式针对失败原因补课。

文本反馈的作用：

1. 用于识别失败集中点。
2. 用于限制下一轮不要重复同一歧义。
3. 用于生成更具体的“脚手架式任务”。

### 7.4 生成任务的输出要求

生成器输出仍应是现有任务 JSON，而不是直接输出 Parquet。

推荐流程：

1. 为每个新任务写出 `<output_dir>/<category>/<task_id>.json`
2. 保证包含 `snapshot`
3. 更新索引文件
4. 再由 `prepare_data.py` 统一生成 Parquet

原因：

1. 与现有 `prepare_data.py` 对齐。
2. 便于回溯每条任务的自然语言与元信息。
3. 更适合后续人工抽检。

---

## 8. 数据流转与 Parquet 兼容方案

### 8.1 与 prepare_data.py 的对接原则

`prepare_data.py` 当前的关键逻辑是：

1. 扫描 `data_dir` 下的各 category 子目录。
2. 递归读取 category 内 `.json` 文件名。
3. 构造 `task_name = "<category>.<filename>"`。
4. 写出 `train.parquet` / `val.parquet`。

因此，自进化任务只要遵守以下规则，就能无缝接入：

1. 每轮生成任务放进一个明确的 category 目录。
2. 任务文件扩展名仍为 `.json`。
3. 训练时通过 `prepare_data.py` 生成标准 Parquet。

推荐目录：

```text
data/tasks/generated/mobile_self_evolve/
  mobile/
    settings_task_0001.json
    settings_task_0002.json
```

### 8.2 推荐的对接方式

推荐保留 `prepare_data.py` 主逻辑不变，只新增一层调度调用：

```bash
python roll/pipeline/agentic/env/android/mobile/prepare_data.py \
  --mode visual \
  --data_dir data/tasks/generated/mobile_self_evolve \
  --train_category mobile \
  --train_category_dir data/tasks/generated/mobile_self_evolve/mobile \
  --local_dir trajectories/<timestamp>/self_evolve/round_0003/parquet
```

若训练集只使用新生成任务，可通过：

- `--train_category_dir`
- `--train_task_pool_file`

来显式指定。

### 8.3 train_dataloader 无缝读取的关键点

需要确保 Parquet 结构仍与原训练一致，即保留：

```json
{
  "prompt": [{"role": "user", "content": "<image>"}],
  "task_name": "mobile.settings_task_0001",
  "images": "...",
  "ability": "agent",
  "extra_info": {
    "split": "train",
    "index": 0
  }
}
```

因此不建议在 dataloader 层增加自进化特例，而是继续让任务 JSON -> `prepare_data.py` -> Parquet 这一链路承担兼容责任。

---

## 9. 接口定义规范

### 9.1 奖励模型通用 API

推荐 `MobileJudge` 增加以下通用 API：

```python
class MobileJudgeEvaluator:
    async def evaluate_episode(
        self,
        episode_input: JudgeEpisodeInput,
    ) -> JudgeEpisodeResult:
        ...

    async def evaluate_from_json(
        self,
        input_json_path: str,
        output_json_path: Optional[str] = None,
    ) -> JudgeEpisodeResult:
        ...
```

输入参数要求：

1. 支持当前状态截图
2. 支持参考截图
3. 支持参考文本指令
4. 支持动作历史
5. 支持环境原始信号

输出参数要求：

1. `reward: float`
2. `success: bool`
3. `feedback_strings: List[str]`
4. `failure_reasons: List[str]`

### 9.2 任务生成器通用 API

推荐 `CurriculumTaskGenerator` 增加以下通用 API：

```python
class MobileSpecificTaskGenerator:
    def generate_tasks_from_feedback(
        self,
        feedback_file: str,
        context_template_dir: str,
        output_dir: str,
        index_dir: str,
        iteration: int,
    ) -> List[str]:
        ...
```

输入参数要求：

1. `feedback_file`
   任务级成功率与文本反馈汇总文件。

2. `context_template_dir`
   包含 snapshot 与参考上下文模板。

3. `output_dir`
   任务 JSON 输出目录。

4. `index_dir`
   任务索引输出目录。

5. `iteration`
   当前自进化轮次。

输出：

1. 生成的任务 JSON 文件列表
2. 更新后的 task index 路径

### 9.3 反馈聚合器 API

建议新增中间层 API，而不是让生成器直接逐个扫描 judge 文件：

```python
def build_round_feedback(
    judge_result_dir: str,
    output_json_path: str,
) -> str:
    ...
```

职责：

1. episode 级结果 -> task 级统计
2. 合并成功率与文本反馈
3. 输出统一的 `round_feedback.json`

这样能让生成器接口稳定，不依赖 judge 原始文件布局。

---

## 10. 代码改动清单

本节优先覆盖用户指定的三个文件，并补充必需的关联改动点。

### 10.1 `roll/pipeline/agentic/env/android/mobile/mobilejudge.py`

当前状态：

1. 已支持多图输入。
2. 已返回 `response_text, reward, details`。
3. `details` 中已有 `key_points`、`thoughts`、`usage`。

建议改动：

1. 新增标准化输入输出结构
   - 增加 `JudgeEpisodeInput`
   - 增加 `JudgeEpisodeResult`

2. 新增 episode 级接口
   - `evaluate_episode(...)`
   - `evaluate_from_json(...)`

3. 扩展输入模态
   - 增加 `reference_screenshot_paths`
   - 增加 `reference_instruction`
   - 增加 `env_signals`

4. 扩展输出
   - `success`
   - `feedback_summary`
   - `feedback_strings`
   - `failure_reasons`
   - `dimension_scores`

5. 增加本地 JSON 读写辅助函数
   - `load_judge_input_json(path)`
   - `save_judge_result_json(path, result)`

逻辑建议：

1. 保留现有 `cuajudge_general_eval()` 作为底层 evaluator。
2. 在其外层封装统一结果格式，避免上层直接依赖松散的 `details` 字段。
3. `reward` 建议固定归一化到 `[0, 1]`。
4. `success` 建议由 `reward >= threshold` 或显式 judge 决策确定，而不是继续依赖环境。

### 10.2 `roll/pipeline/agentic/env/android/mobile/curriculum_task_generator.py`

当前状态：

1. 已具备 iteration 0 与 iteration 1+ 两套 prompt。
2. 已支持读取反馈文件。
3. 已在 `create_task_config_file()` 中写入 `snapshot`。

建议改动：

1. 明确区分两类反馈输入
   - 旧格式：仅 `accuracy`
   - 新格式：`success_rate + avg_reward + feedback_strings + failure_reasons`

2. 新增专门的自进化入口
   - `generate_tasks_from_feedback(...)`
   - `load_round_feedback(...)`
   - `format_feedback_summary_v2(...)`

3. Prompt 升级
   - 显式告诉模型读取数值成功率
   - 显式告诉模型读取失败原因文本
   - 明确要求保留 `snapshot`

4. 任务输出增强
   - 写入 `self_evolve.round_id`
   - 写入 `parent_task_id`
   - 写入 `feedback_ref`

5. 输出目录调整
   - 支持直接输出到 `<generated_root>/<category>/`
   - 方便 `prepare_data.py` 直接扫描

逻辑建议：

1. 低成功率任务优先生成 prerequisite task。
2. 高成功率任务优先升难度与增加约束。
3. 新任务默认继承原任务 `snapshot`。
4. 如果反馈文件缺失，自动回退到 iteration 0 逻辑，保证健壮性。

### 10.3 `roll/pipeline/agentic/env/android/mobile/prepare_data.py`

当前状态：

1. 通过目录扫描 JSON 文件名构造 `task_name`。
2. 输出标准 `train.parquet` 和 `val.parquet`。

建议改动：

1. 保持主逻辑不变
   - 这是兼容性的关键，不建议重写。

2. 新增可选便捷参数
   - `--generated_task_dir`
   - `--self_evolve_round_dir`
   - 仅作为路径糖，不改变输出格式。

3. 增加日志提示
   - 明确打印当前使用的是哪一轮生成任务。

4. 可选增加 train-only 模式说明
   - 因为自进化生成任务不一定满足当前 val 的 UUID 过滤逻辑。

逻辑建议：

1. Parquet 结构保持不变。
2. 自进化任务建议主要进入 train split。
3. 若后续需要 self-evolve val，可单独定义验证集命名规范，而不是在当前版本侵入修改 val 过滤规则。

### 10.4 建议同步改动的关联文件

虽然题目重点点名了三个文件，但要真正跑通闭环，以下文件也建议一并扩展。

#### `roll/pipeline/agentic/env_manager/android_step_env_manager.py`

建议改动：

1. 增加 `self_evolve.enabled` 分支。
2. episode 结束时收集轨迹并写 `judge_input.json`。
3. 调用 `MobileJudge.evaluate_from_json()`。
4. 用 judge 奖励覆盖最后一步 `reward`。
5. 将 `judge_result_path`、`judge_feedback_summary` 写入 `meta_info` 或 `non_tensor_batch` 的非训练字段。

这是最关键的接入点，因为它直接控制 `formulate_rollouts()` 里的 reward 写入。

#### `roll/pipeline/agentic/env_manager/gui_traj_env_manager.py`

建议改动：

1. 在通用轨迹生命周期中预留 `on_episode_end` 钩子。
2. 让子类可以在不改主循环的前提下插入 judge 逻辑。

推荐形式：

```python
def on_episode_end(self, rollout_cache: RolloutCache) -> RolloutCache:
    return rollout_cache
```

然后由 `AndroidStepEnvManager` 覆写。

#### `roll/pipeline/agentic/env/android/GuiTaskEvalManager.py`

建议改动：

1. `/complete_task` 增加可选字段：
   - `reward`
   - `feedback_summary`
   - `feedback_path`
   - `failure_reasons`

2. `task_stats` 增加：
   - `average_reward`
   - `last_feedback_summary`

3. `/stats` 输出扩展后的聚合结果。

这样任务生成器既可以读 round_feedback，也可以必要时直接读 task manager 统计。

#### `roll/pipeline/agentic/env/android/remote_mobileworld.py`

建议改动：

1. 在 self-evolve 模式下，保留环境原始 `score/is_success/reason`，但不直接作为最终训练 reward。
2. `reset()` 时明确按任务里的 `snapshot` 恢复上下文。
3. `complete_task` 上报改为接受 judge 后的 success，而不是环境即时 success。

可选做法：

1. 由环境先不调用 `/complete_task`
2. 改为由 EnvManager 在 judge 完成后统一上报

这样最符合“环境中的任务成功结果反馈由奖励模块代替环境给出”的要求。

#### `roll/pipeline/agentic/env/android/remote_android.py`

建议同步采取与 `remote_mobileworld.py` 一致的策略，避免两个环境家族行为不一致。

#### `roll/pipeline/agentic/agentic_pipeline.py`

建议改动：

1. 增加自进化回合边界检查。
2. 在回合结束时调用：
   - 反馈聚合
   - 任务生成
   - `prepare_data.py`
   - task manager 重初始化

注意：

1. 训练 batch 计算逻辑尽量不动。
2. 只增加 round 级控制逻辑。

---

## 11. 推荐新增的辅助模块

为了避免把现有文件改得过重，建议新增几个轻量模块：

1. `roll/pipeline/agentic/env/android/mobile/self_evolve_types.py`
   放标准 TypedDict / dataclass。

2. `roll/pipeline/agentic/env/android/mobile/self_evolve_io.py`
   放 JSON 读写与目录组织逻辑。

3. `roll/pipeline/agentic/env/android/mobile/self_evolve_feedback.py`
   放 `build_round_feedback()` 聚合逻辑。

4. `roll/pipeline/agentic/self_evolve_coordinator.py`
   放 round 级编排逻辑。

这样可以把：

- 奖励模型
- 任务生成器
- 训练主干

之间的耦合降到最低。

---

## 12. 兼容性与风险控制

### 12.1 非自进化模式稳定性

必须保证默认行为不变：

1. 默认 `self_evolve.enabled = false`
2. 所有新增字段均为可选字段
3. judge 逻辑仅在自进化模式触发
4. 任务生成与 Parquet 刷新仅在自进化模式触发

### 12.2 失败降级策略

建议加入以下降级：

1. judge 失败
   - 记录错误
   - 回退到环境原始 reward 或 0.0

2. round_feedback 生成失败
   - 保持当前任务池继续训练

3. 任务生成失败
   - 回退到上一轮任务池

4. `prepare_data.py` 失败
   - 不切换 task pool

### 12.3 一致性风险

最需要注意的三个风险：

1. reward 来源切换后，环境和训练端的 success 定义可能不一致。
   建议始终以 judge 为准，并把环境结果只作为辅助字段。

2. 任务生成后若丢失 `snapshot`，环境将无法快速恢复。
   建议在写任务 JSON 时强校验 `snapshot` 非空。

3. 生成任务若不通过 `prepare_data.py`，dataloader 可能出现字段不一致。
   建议坚持“JSON 任务 -> prepare_data.py -> Parquet”的唯一入口。

---

## 13. 推荐实施顺序

建议按四步落地，避免一次性大改：

1. 第一步：奖励标准化
   - 先改 `mobilejudge.py`
   - 能稳定输出 `reward + feedback_strings + json`

2. 第二步：episode 级接入
   - 在 `android_step_env_manager.py` 接入 judge
   - 让训练 reward 来自 judge

3. 第三步：round 级任务生成
   - 完成 `round_feedback.json`
   - 改 `curriculum_task_generator.py`

4. 第四步：Parquet 自动刷新与任务池切换
   - 接通 `prepare_data.py`
   - 在 pipeline 中加入 round 更新

这样可以先验证“reward 闭环”，再验证“任务闭环”。

---

## 14. 一句话总结

最推荐的实现路径是：在现有框架上新增一个“自进化控制平面”，让环境继续负责执行、让 `MobileJudge` 负责最终奖励与文本反馈、让 `CurriculumTaskGenerator` 负责基于本地 JSON 反馈生成下一轮任务，并始终通过现有 `prepare_data.py` 写入 Parquet。这样既能满足闭环自进化目标，也能最大程度保持常规训练链路稳定。
