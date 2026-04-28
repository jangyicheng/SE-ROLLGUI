# `agent_val_multiandroid_grpo` 当前训练异常诊断

本文基于以下内容做诊断：

- 当前配置：[jyc/agent_val_multiandroid_grpo.yaml](/e:/code/GUI/Roll-GUI/jyc/agent_val_multiandroid_grpo.yaml)
- 当前训练指标：`output/metrics/step_0` 到 `step_4`
- 训练主循环：[agentic_pipeline.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_pipeline.py)
- actor loss：[agentic_actor_worker.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_actor_worker.py)
- 奖励和 KL 处理：[functionals.py](/e:/code/GUI/Roll-GUI/roll/utils/functionals.py)

结论先说：这次训练里确实同时存在“日志口径误导”和“真实训练配置/实现问题”两类情况。

- `actor/lr` 始终为 0 不是优化器没建起来，而是 scheduler 的总步数在第 0 个 pipeline step 内就被跑完了，日志记录的是 step 结束时的 LR。
- `critic/kl_coef` 始终为 0 不代表 KL 完全没参与训练。它只对应 reward-side token KL 的日志口径；当前 actor-side `kl_loss` 已经通过 `kl_loss_coef=0.001` 加进 `total_loss`，但 reward-side KL penalty 最终没有真正留在 `token_level_rewards` 里。
- 当前更危险的问题是：训练样本是 step-wise 展开的，`max_steps` 却按 `rollout_batch_size=16` 这类轨迹数来估算 scheduler 总步数，导致 LR 计划严重错配；同时 reward-side KL penalty 被后续逻辑覆盖，batch 还会被大量复制补齐，策略很容易在单个 pipeline step 内漂移过快。

## 现象概览

从 `output/metrics` 的前 5 个 step 可以直接看到：

- `actor/lr`：step 0 到 step 4 全是 `0.0`
- `critic/kl_coef`：step 0 到 step 4 全是 `0`
- `critic/kl`：从 `-0.00043` 下降到 `-0.24257`
- `actor/ratio_max`：`126.0 -> 9.8 -> 51.4 -> 89.7 -> 400.0`
- `actor/ratio_min`：最小到 `1e-6` 量级
- `actor/approxkl`：`0.0107 -> 0.0517 -> 0.1301 -> 0.0642 -> 0.1208`
- `actor_train/grad_norm`：在 `3.16` 到 `4.49` 间波动，并没有稳定下降
- `system/batch_add_count`：`80, 98, 68, 45, 71`
- `actor/samples_total`：每个 step 都是 `256`

这些数值说明两件事：

1. 第 0 个 step 看到的 actor 指标并不是“训练前状态”，而是当前 batch 经过完整 PPO 更新后的聚合结果。
2. 当前训练里每个 pipeline step 实际发生的 optimizer update 数量，远大于 `max_steps` 估算时默认假设的数量。

## 训练链路与指标来源

当前 agentic 训练主循环的关键顺序是：

1. rollout 采样
2. `adjust_batch()`
3. 计算 `ref_log_probs`
4. 计算 `old_log_probs`
5. 计算 response reward
6. `apply_kl_penalty()`
7. `compute_token_reward()`
8. `agentic_compute_advantage()`
9. `actor_train.train_step()`

对应代码：

- [agentic_pipeline.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_pipeline.py):212
- [agentic_pipeline.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_pipeline.py):218
- [agentic_pipeline.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_pipeline.py):229
- [agentic_pipeline.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_pipeline.py):278
- [agentic_pipeline.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_pipeline.py):300

actor 指标来源：

- `ratio = exp(log_probs - old_log_probs)`
- `approxkl = 0.5 * (log_probs - old_log_probs)^2`
- `kl_loss = compute_approx_kl(log_probs, ref_log_probs, kl_penalty="k3")`
- `total_loss = pg_loss + kl_loss * kl_loss_coef`

对应代码：

- [agentic_actor_worker.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_actor_worker.py):30
- [agentic_actor_worker.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_actor_worker.py):46
- [agentic_actor_worker.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_actor_worker.py):50
- [agentic_actor_worker.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_actor_worker.py):60

reward-side KL 指标来源：

- `apply_kl_penalty()` 用 `kl_ctrl.value` 计算 token-level KL penalty
- `compute_token_reward()` 也会再算一次 KL，并把 `critic/kl` / `critic/kl_coef` 写进日志

对应代码：

- [functionals.py](/e:/code/GUI/Roll-GUI/roll/utils/functionals.py):630
- [functionals.py](/e:/code/GUI/Roll-GUI/roll/utils/functionals.py):757

## 逐项诊断

### 1. `actor/lr` 为什么始终为 0

这次不是简单的 warmup 现象，而是 scheduler 总步数和真实训练步数严重错配。

当前配置：

```yaml
max_steps: 16
rollout_batch_size: 16
ppo_epochs: 4
per_device_train_batch_size: 4
gradient_accumulation_steps: 16
learning_rate: 1.0e-6
warmup_steps: 4
```

`PPOConfig.set_max_steps()` 会按下面的公式设置 actor train 的总步数：

```python
actor_train.training_args.max_steps =
    max_steps * (rollout_batch_size * num_return_sequences * ppo_epochs // actor_backward_batch_size)
```

在当前配置下：

```text
actor_backward_batch_size = 4 * 16 = 64
rollout_batch_size * ppo_epochs = 16 * 4 = 64
所以 actor_train.training_args.max_steps = 16 * (64 // 64) = 16
```

随后 DeepSpeed 初始化又会除以 `dp_size=2`：

```text
worker train max_steps = 16 // 2 = 8
```

也就是说，每个 DP rank 的 scheduler 总共只有 `8` 步。

问题在于，这个估算是按 `rollout_batch_size=16` 条轨迹来的，但当前 Android agentic 训练并不是“一条轨迹对应一个训练样本”。[android_step_env_manager.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/env_manager/android_step_env_manager.py:302) 会把一条轨迹拆成多个 step-wise 样本，再 `DataProto.concat(samples)`。

因此日志里我们看到的真实训练样本数是：

- step 0：`actor/samples_total = 256`
- step 1：`actor/samples_total = 256`
- step 2：`actor/samples_total = 256`
- step 3：`actor/samples_total = 256`
- step 4：`actor/samples_total = 256`

以 step 0 为例：

- `system/batch_add_count = 80`
- 所以 `adjust_batch()` 前原始样本数是 `256 - 80 = 176`
- DP=2 时，每个 rank 收到 `128` 条样本
- `backward_batch_size = 64`
- 每个 epoch 每个 rank 有 `128 / 64 = 2` 个 outer batch
- `ppo_epochs = 4`
- 所以单个 pipeline step 内每个 rank 实际会执行 `2 * 4 = 8` 次 optimizer step

这正好把 scheduler 的 `8` 个总步数在 `global_step=0` 一次性耗尽。于是：

- step 0 内部虽然不一定从头到尾都用 `lr=0`
- 但 [base_worker.py](/e:/code/GUI/Roll-GUI/roll/pipeline/base_worker.py:113) 记录的是 `train_step()` 结束后的 `get_last_lr()`
- 所以写进日志时已经是 `0.0`
- 后续 step 因 scheduler 已跑完，继续保持 `0.0`

这也是为什么现在看到的是“LR 始终为 0”，而不是“前几步 warmup 很小”。

### 2. `critic/kl_coef` 为什么始终为 0

这里同时存在“日志口径”和“实际应用”两个层面。

先看日志口径。

当前 pipeline 里 reward/KL 的顺序是：

```python
batch, kl_metrics = apply_kl_penalty(...)
batch, token_level_metrics = compute_token_reward(...)
metrics.update(token_level_metrics)
```

注意最后只 `update(token_level_metrics)`，没有 `update(kl_metrics)`。

而在 `compute_token_reward()` 里：

```python
beta = 0
if pipeline_config.add_token_level_kl:
    beta = kl_ctrl.value
    token_level_rewards = token_level_rewards - beta * kld
metrics = {"critic/kl": current_kl, "critic/kl_coef": beta}
```

当前配置没有设置 `add_token_level_kl: true`，默认就是 `false`，所以这里日志里的 `beta` 永远是 `0`，`critic/kl_coef` 自然始终是 `0`。

但这不代表 KL 完全没参与训练。实际要分两条路径看：

1. reward-side KL penalty：
   - `apply_kl_penalty()` 里确实会用 `kl_ctrl.value`
   - 但随后 `compute_token_reward()` 会重新 `expand_to_token_level(data)`，并把先前的 `token_level_rewards` 重命名为 `token_level_scores`
   - 最终新的 `token_level_rewards` 会覆盖前一阶段的 reward-side KL penalty
   - 所以 reward-side KL penalty 最后基本没有保留到 advantage 计算里

2. actor-side KL loss：
   - 当前配置已经显式设置了 `kl_loss_coef: 0.001`
   - 所以 `total_loss = pg_loss + 0.001 * kl_loss`
   - 这条 KL loss 是真的在 actor 训练里生效的

这可以直接从日志验证。以 step 4 为例：

```text
actor/pg_loss   = 0.1464010573
actor/kl_loss   = 0.3958202958
actor/total_loss= 0.1467968770
```

两者差值约为：

```text
0.1467968770 - 0.1464010573 = 0.0003958197
≈ 0.001 * 0.3958202958
```

所以当前状态不是“KL 没生效”，而是：

- actor-side `kl_loss_coef` 已经生效
- reward-side `critic/kl_coef` 日志始终为 0
- reward-side KL penalty 还被后续逻辑覆盖了

### 3. `critic/kl` 为什么为负，且越来越负

当前 `critic/kl` 不是严格意义上的 KL 散度。

默认 `kl_penalty="kl"` 时，[functionals.py](/e:/code/GUI/Roll-GUI/roll/utils/functionals.py:164) 的定义是：

```python
log_ratio = log_probs - log_probs_base
```

在 reward/KL 逻辑里传入的是：

```python
log_probs=data.batch["old_log_probs"]
log_probs_base=data.batch["ref_log_probs"]
```

所以这里记录的其实是：

```text
critic/kl = mean(old_log_probs - ref_log_probs)
```

它是“采样 token 上的平均 log-prob 差值”，不是全分布上的严格非负 KL。

当前日志：

- step 0: `-0.00043`
- step 1: `-0.03593`
- step 2: `-0.08543`
- step 3: `-0.15328`
- step 4: `-0.24257`

这说明 old policy 在采样 token 上相对 reference 越来越“更不相信”这些动作了，也就是 reference 和当前训练策略之间的偏离在累积。由于 reward-side KL penalty 最终没被保留，这个趋势并不意外。

### 4. `actor/ratio_max`、`actor/ratio_min`、`actor/approxkl` 为什么第 0 个 step 就很怪

这类指标的第一层原因是日志聚合口径，第二层原因才是真实漂移。

先看口径：

- [base_worker.py](/e:/code/GUI/Roll-GUI/roll/pipeline/base_worker.py:94) 和 [base_worker.py](/e:/code/GUI/Roll-GUI/roll/pipeline/base_worker.py:104) 会按 `ppo_epochs=4` 构造 dataloader
- 每个 outer batch 的 actor 指标会被 `append_to_dict(metrics, pg_metrics)`
- `reduce_metrics()` 对 `_max` 后缀取 `np.max`，对 `_min` 后缀取 `np.min`

因此：

- `actor/ratio_max` 代表整个 train step 内所有 micro-batch 的最大极值
- `actor/ratio_min` 代表整个 train step 内所有 micro-batch 的最小极值
- `actor/approxkl` 是 step 内多次更新后的均值，不是“训练前对 old policy 的一次比较”

再看真实漂移。

当前 5 个 step 的 `actor/ratio_max` 是：

- `126.0, 9.8, 51.4, 89.7, 400.0`

`actor/ratio_min` 最小到 `1e-6`，`actor/approxkl` 最高到 `0.1301`。这已经不是纯口径问题，而是模型在一个 pipeline step 内对 old policy 发生了很强的偏移。

促成这个偏移的因素包括：

- step-wise 样本多，单 step 实际 optimizer steps 很多
- `ppo_epochs=4`
- batch 还会被复制补齐
- reward-side KL penalty 没最终留下
- `pg_clip_high: 0.28` 当前其实没有生效，因为 `use_pg_clip_range` 默认是 `false`，actor 代码仍然会回落到默认 `pg_clip`

也就是说，现在看到的 ratio 异常既有“第 0 step 不是 pre-update 快照”的成分，也有“当前 step 内确实更新过猛”的成分。

### 5. `system/batch_add_count` 和 `actor/samples_total` 说明了什么

这组指标很关键，它解释了为什么训练强度远超表面上的 `rollout_batch_size=16`。

当前是 step-wise agentic rollout：

- [android_step_env_manager.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/env_manager/android_step_env_manager.py:345) 会对轨迹中的每个 step 构造一个 `DataProto`
- 最后 `DataProto.concat(samples)` 得到真正的训练 batch

所以 `rollout_batch_size=16` 表示的是 rollout 轨迹数，不是 actor 训练时看到的样本数。

从日志反推，`adjust_batch()` 前的真实样本数分别是：

- step 0: `256 - 80 = 176`
- step 1: `256 - 98 = 158`
- step 2: `256 - 68 = 188`
- step 3: `256 - 45 = 211`
- step 4: `256 - 71 = 185`

也就是说，当前训练不是“16 条轨迹补到 128”，而是“约 158 到 211 条 step-wise 样本补到 256”。这比上一次诊断更接近真实情况。

但问题仍然存在：

- 补齐样本是复制来的
- `adjust_batch(copy)` 里的 `TODO: set dup_proto response_mask to 0` 还没做
- 所以复制样本仍然参与 loss、KL、reward、advantage

当前复制比例大约在：

- `45 / 256 = 17.6%`
- 到
- `98 / 256 = 38.3%`

之间。

这已经足以放大 batch 内过拟合和极端 token ratio。

## 根因链路

### 根因 1：scheduler 总步数按轨迹数估算，但真实训练按 step-wise 样本更新

这是这次 `actor/lr=0` 的直接原因，也是当前最明确的配置-实现错位。

- `set_max_steps()` 用的是 `rollout_batch_size`
- 实际 actor 更新强度取决于 step-wise 样本数、`adjust_batch()` 后的大小、DP 分片和 `ppo_epochs`
- 当前这几个量完全不在一个尺度上

结果就是：scheduler 以为自己有 8 步可走，真实却在第 0 个 pipeline step 里就跑满了 8 步。

### 根因 2：reward-side KL penalty 被覆盖，日志中的 `critic/kl_coef` 也无法反映真实路径

当前 reward/KL 路径里同时存在三件事：

- `apply_kl_penalty()` 会写 `token_level_rewards`
- `compute_token_reward()` 会重新展开 `token_level_rewards`
- pipeline 只记录 `compute_token_reward()` 的 `token_level_metrics`

因此最终表现是：

- 日志里的 `critic/kl_coef = 0`
- reward-side KL penalty 最终没进入 advantage
- 但 actor-side `kl_loss` 仍然在 loss 中生效

这会让人误以为“KL 完全没开”，实际上是“reward-side 和 actor-side 走了两条不同的 KL 路径，而且日志只看见了其中一条的 0 值”。

### 根因 3：batch 复制和多 epoch 叠加，放大单 step 内策略漂移

当前每个 step 都会：

- 用 17% 到 38% 的复制样本补齐到 256
- 对同一批样本跑 `ppo_epochs=4`
- 让 `ratio_max` 按全 step 极值记录

这使得第 0 个 step 的 actor 指标天然就不是“初始状态”，而是“单 step 内多次更新之后的结果”。

### 根因 4：`pg_clip_high: 0.28` 当前并没有按你看到的名字生效

配置里虽然写了：

```yaml
pg_clip_high: 0.28
```

但 actor 代码是：

```python
pg_clip_high = self.pipeline_config.pg_clip_high if self.pipeline_config.use_pg_clip_range else self.pipeline_config.pg_clip
```

如果 `use_pg_clip_range` 没开，最终仍然回到 `pg_clip`。所以如果你原本希望上限 clip 用 `0.28`，当前并没有按这个值工作。

## 建议的验证点

1. 验证 scheduler 是否在 step 0 内耗尽：
   - 临时增加日志，记录每个 optimizer step 后的 `scheduler.get_last_lr()`
   - 记录 `self.worker_config.training_args.max_steps`
   - 记录每个 pipeline step 内真实发生的 optimizer step 数

2. 验证 reward-side KL penalty 是否被覆盖：
   - 在 `apply_kl_penalty()` 后打印 `token_level_rewards.sum()`
   - 在 `compute_token_reward()` 后再打印一次
   - 对比两者是否发生重置

3. 区分 reward-side KL 和 actor-side KL：
   - 单独记录 `reward/kl_coef_applied`
   - 单独记录 `actor/kl_loss_coef_applied`
   - 不要再让 `critic/kl_coef` 混用两条语义

4. 验证 `pg_clip_high` 是否实际生效：
   - 打印 `pg_clip_low` / `pg_clip_high` 的最终运行值

## 建议修复方向

### 1. 修正 LR scheduler 步数估算

这是优先级最高的修复之一。

更合理的方向：

- 不要再用 `rollout_batch_size` 估算 actor scheduler 总步数
- 改为按真实 step-wise 样本数估算
- 至少应接近：

```text
optimizer_steps_per_pipeline_step
= ceil(adjusted_samples_per_rank / backward_batch_size) * ppo_epochs
```

短期配置规避办法：

- 先把 `warmup_steps` 设为 `0`
- 同时把 `actor_train.training_args.max_steps` 的估算显著放大
- 或降低 `ppo_epochs`

但这只是缓解。根本方案还是让 scheduler 基于真实更新步数，而不是轨迹数。

### 2. 合并 reward-side KL 路径

当前不应同时保留“先 `apply_kl_penalty()`，后 `compute_token_reward()` 覆盖”的流程。

建议二选一：

- 方案 A：只保留 `compute_token_reward()`，并开启 `add_token_level_kl: true`
- 方案 B：只保留 `apply_kl_penalty()`，让 `compute_token_reward()` 不再覆盖 `token_level_rewards`

无论选哪种，都需要：

- `kl_ctrl.update()` 只调用一次
- 日志里明确区分 reward-side KL 和 actor-side KL

### 3. 保留 actor-side KL loss，但把口径写清楚

当前 `kl_loss_coef: 0.001` 是生效的，这一点从 `actor/total_loss` 已经能验证。

建议：

- 保留 `kl_loss_coef`
- 新增明确日志，例如 `actor/kl_loss_coef_applied`
- 不要再把 `critic/kl_coef` 当成“整个系统的 KL 是否开启”的判断依据

### 4. 降低单 step 内更新强度

可以优先试这几个配置方向：

- `ppo_epochs: 4 -> 1` 或 `2`
- 降低 `gradient_accumulation_steps`
- 尽量让 `adjust_batch()` 后的复制比例下降

当前 `batch_add_count` 已经说明，复制样本比例不低。即便不是之前以为的“16 复制到 128”，现在的“158 到 211 补到 256”依然会明显干扰训练。

### 5. 让 `pg_clip_high` 真正生效

如果你想用非对称 clip 范围，需要显式打开：

```yaml
use_pg_clip_range: true
```

否则当前 `pg_clip_high: 0.28` 只是写在配置里，实际运行并不会按这个上限裁剪。

## 结论

这次训练里最关键的两个异常是：

1. `actor/lr` 不是“优化器坏了”，而是 scheduler 总步数按轨迹数估算，结果在第 0 个 pipeline step 内就被 step-wise 样本训练耗尽了。
2. `critic/kl_coef` 不是“KL 完全没开”，而是 reward-side 日志口径始终返回 0；与此同时 actor-side `kl_loss_coef=0.001` 已经生效，但 reward-side KL penalty 又被后续逻辑覆盖掉了。

如果只看当前最值得优先修的部分，顺序建议是：

1. 修 `max_steps` 和 scheduler 步数估算
2. 合并 reward-side KL 路径，避免 `token_level_rewards` 被覆盖
3. 降低单 step 内更新强度，减少复制样本和 PPO epoch
4. 补清晰日志，明确区分 reward-side KL、actor-side KL、step 内真实 optimizer step 数

只要前两项修正到位，`actor/lr=0` 和 `critic/kl_coef=0` 这两个最迷惑人的现象都会变得可解释，`ratio_max` / `approxkl` 这类真正反映策略漂移的指标也更容易收敛到可分析的范围。

## 建议修正参数与目标值

这一节只列建议直接修改的参数和值。分成两类：

- 第一类：不改代码时也建议先改，用来止血。
- 第二类：只有在代码路径修正后，这些参数值才真正符合预期。

### 一、不改代码时，建议先改的参数

建议先把当前配置改成下面这一版：

```yaml
advantage_clip: 0.5
ppo_epochs: 1
use_pg_clip_range: true
pg_clip_low: 0.2
pg_clip_high: 0.28
use_kl_loss: true
kl_loss_coef: 0.01
add_token_level_kl: true

actor_train:
  training_args:
    learning_rate: 1.0e-6
    gradient_accumulation_steps: 8
    warmup_steps: 0
```

各参数的建议值和原因如下：

- `warmup_steps: 0`
  原因：在当前代码里，scheduler 总步数被严重低估，`warmup_steps: 4` 只会让日志里的 `actor/lr` 更快看起来像“始终为 0”。在修代码前，先取消 warmup 更稳妥。

- `ppo_epochs: 1`
  原因：当前单个 pipeline step 内真实 optimizer step 已经很多，`ppo_epochs: 4` 会显著放大策略漂移。先降到 `1` 最直接。

- `gradient_accumulation_steps: 8`
  原因：当前是 `16`，更新粒度太粗，同时会放大 batch 对齐和 scheduler 步数错配问题。先降到 `8`，让单 step 的训练强度下降一档。

- `use_pg_clip_range: true`
  原因：否则你配置里的 `pg_clip_high: 0.28` 实际不会生效。

- `pg_clip_low: 0.2`
  原因：与当前默认 `pg_clip` 下界保持一致，避免同时引入额外变量。

- `pg_clip_high: 0.28`
  原因：保留你现在想要的非对称上界，但前提是 `use_pg_clip_range: true`。

- `kl_loss_coef: 0.01`
  原因：当前 `0.001` 已经生效，但太弱。以 step 4 为例，KL 对 `total_loss` 的增量只有约 `0.000396`，约等于 `pg_loss` 的千分之几，约束力偏弱。先提高到 `0.01` 更合理。

- `add_token_level_kl: true`
  原因：在当前代码不改的前提下，这是让最终 `token_level_rewards` 至少带上一份 KL penalty 的最直接办法，也能让日志里的 `critic/kl_coef` 不再始终是 `0`。

- `advantage_clip: 0.5`
  原因：当前是 `1.0`，对 step-wise GUI RL 来说偏宽。先收紧到 `0.5`，减少高方差优势对 `ratio` 尾部的放大。

### 二、代码修正后，建议保留或调整成的参数

如果后续修正了以下代码问题：

- scheduler 总步数按真实 step-wise 样本数估算
- reward-side KL 只保留一条路径
- `token_level_rewards` 不再被重复覆盖
- `kl_ctrl.update()` 不再一轮调用两次

那么建议参数目标值如下：

```yaml
advantage_clip: 0.5
ppo_epochs: 2
use_pg_clip_range: true
pg_clip_low: 0.2
pg_clip_high: 0.28
use_kl_loss: true
kl_loss_coef: 0.01
add_token_level_kl: true

actor_train:
  training_args:
    learning_rate: 1.0e-6
    gradient_accumulation_steps: 8
    warmup_steps: 2
```

这里和“止血版”相比，只有两点变化：

- `ppo_epochs: 1 -> 2`
  原因：代码修好后，可以尝试恢复一点 PPO 重复利用率，但不建议直接回到 `4`。

- `warmup_steps: 0 -> 2`
  原因：当 scheduler 步数估算修正后，可以重新引入一个很短的 warmup；`2` 比当前的 `4` 更保守。

### 三、暂时不建议直接改动的参数

下面这些参数这次不建议直接一起动：

- `learning_rate`
  建议先保持 `1.0e-6`。当前最主要的问题不是 base LR 过大，而是 scheduler 步数错配和单 step 更新过猛。

- `init_kl_coef`
  建议先保持 `0.001`。当前真正有问题的是 reward-side KL 路径和日志口径，不是初始 KL controller 数值本身。

- `rollout_batch_size`
  这次先不建议直接改，因为当前训练是 step-wise 样本展开，真正影响训练强度的是展开后的样本数，而不是表面上的 `16` 条轨迹。

### 四、推荐优先落地顺序

如果只想按最小代价先改一轮，建议顺序是：

1. `warmup_steps: 0`
2. `ppo_epochs: 1`
3. `use_pg_clip_range: true`
4. `kl_loss_coef: 0.01`
5. `add_token_level_kl: true`
6. `gradient_accumulation_steps: 8`

这组参数改完后，最值得立刻观察的指标是：

- `actor/lr` 是否仍然始终为 `0`
- `critic/kl_coef` 是否仍然始终为 `0`
- `actor/ratio_max`
- `actor/approxkl`
- `actor_train/grad_norm`
- `system/batch_add_count`
