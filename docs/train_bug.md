# `agent_val_multiandroid_grpo` 当前训练诊断

本文基于以下内容分析：

- 当前运行状态：[output/worker_state_pipeline.json](/e:/code/GUI/Roll-GUI/output/worker_state_pipeline.json)
- 当前配置文件：[jyc/agent_val_multiandroid_grpo.yaml](/e:/code/GUI/Roll-GUI/jyc/agent_val_multiandroid_grpo.yaml)
- 主训练链路：[roll/pipeline/agentic/agentic_pipeline.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_pipeline.py)
- actor loss：[roll/pipeline/agentic/agentic_actor_worker.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_actor_worker.py)
- actor scheduler 初始化：[roll/distributed/strategy/deepspeed_strategy.py](/e:/code/GUI/Roll-GUI/roll/distributed/strategy/deepspeed_strategy.py)
- `max_steps` 估算：[roll/configs/base_config.py](/e:/code/GUI/Roll-GUI/roll/configs/base_config.py)
- step-wise rollout 展开：[roll/pipeline/agentic/env_manager/android_step_env_manager.py](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/env_manager/android_step_env_manager.py)

先说结论。

这轮训练比更早那轮健康一些，但最关键的问题还没有消失：

1. `critic/rewards/mean` 在涨，但 `critic/score/mean` 没有稳定跟着涨，说明优化目标有改善，真实任务成功率还没有形成持续正反馈。
2. `actor/approxkl` 总体不高，但 `actor/ratio_max` 在 step4 冲到 `26.95`，说明没有全面发散，但已经有 tail token 漂移。
3. `actor/lr` 仍然表现异常，根因还是 scheduler 和 step-wise 模式的尺度错配。
4. 更重要的是：按当前 yaml 静态计算，`actor_train.training_args.max_steps` 会被整除成 `0`。这意味着当前 `cosine` scheduler 在 step-wise 模式下不是“有点不准”，而是配置逻辑本身已经不成立。

## 现象概览

`worker_state_pipeline.json` 中 step0 到 step4 的主要指标如下：

| 指标 | step0 | step1 | step2 | step3 | step4 | 趋势 |
|------|-------|-------|-------|-------|-------|------|
| `actor/lr` | `1e-6` | `1e-6` | `0` | `1e-6` | `0` | 不稳定 |
| `critic/score/mean` | `0.4375` | `0.5` | `0.75` | `0.5625` | `0.5625` | 波动，无明显上升 |
| `critic/rewards/mean` | `0.000007` | `0.00105` | `0.00264` | `0.00609` | `0.00727` | 持续上升 |
| `actor/ratio_max` | `9.18` | `6.76` | `7.96` | `6.71` | `26.95` | step4 飙高 |
| `actor/approxkl` | `0.0040` | `0.0065` | `0.0123` | `0.00286` | `0.00759` | 总体低位 |
| `critic/entropy/mean` | `0.0806` | `0.0825` | `0.0798` | `0.0648` | `0.0626` | 缓慢下降 |
| `actor_train/grad_norm` | `3.19` | `2.75` | `5.10` | `3.57` | `3.31` | step2 有尖峰 |
| `critic/kl` | `-0.00022` | `-0.01024` | `-0.02596` | `-0.05883` | `-0.07066` | 越来越负 |
| `critic/kl_coef` | `0.001` | `0.001` | `0.001` | `0.001` | `0.001` | 已生效 |
| `actor/samples_total` | `256` | `256` | `192` | `192` | `192` | step-wise 展开后样本数 |
| `system/batch_add_count` | `48` | `53` | `8` | `26` | `47` | 复制补齐样本数 |

这组数据反映出三点：

1. reward 在持续变好，但 score 没有稳定变好。
2. 大多数 token 更新仍然温和，但少量 token 已经在 step4 出现明显极值。
3. `lr` 的问题不是单纯 warmup，而是 scheduler 时间轴本身不对。

## 当前 yaml 和这轮日志的关键信息

当前 yaml 里的关键配置是：

```yaml
max_steps: 16
rollout_batch_size: 16
ppo_epochs: 1

actor_train:
  training_args:
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 8
    learning_rate: 1.0e-6
    warmup_steps: 0
    lr_scheduler_type: cosine
```

同时还有三点值得注意：

1. `add_token_level_kl: true` 已经开启，所以当前 `critic/kl_coef = 0.001` 是真实生效值，不再是旧文档里那种“日志假象”。
2. `kl_loss_coef: 0.01` 已经生效。从 step4 可验证：

```text
actor/total_loss - actor/pg_loss
= 0.1499057103 - 0.1481004519
= 0.0018052584

actor/kl_loss = 0.1805258179

0.0018052584 / 0.1805258179 ≈ 0.01
```

3. `use_pg_clip_range` 目前没有显式设为 `true`。因此虽然 yaml 里写了 `pg_clip_high: 0.28`，实际仍会回退到默认 `pg_clip`，因为 actor 代码是这样取值的：[agentic_actor_worker.py:35](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_actor_worker.py:35)

## 逐项诊断

### 1. 这轮日志怎么看

#### `critic/rewards/mean` 持续上升，但 `critic/score/mean` 不跟

这说明模型确实在往优化目标更高的方向走，但这些 reward 改善还没有稳定转化为任务成功率。当前更应该把：

- `critic/score/mean`

作为主目标，把：

- `critic/rewards/mean`
- `actor/approxkl`
- `actor/ratio_max`

作为辅助诊断指标。

#### `actor/approxkl` 不高，但 `actor/ratio_max` 在 step4 冲高

这不是矛盾，而是典型的“整体没炸，尾部有问题”。

- `approxkl` 是均值口径
- `ratio_max` 是整轮 train step 内的极值口径

所以现在的状态不是全面发散，而是少量 token 已经开始有较大的更新幅度。复制补齐样本和 step-wise 重复训练会放大这个问题。

#### `critic/kl` 越来越负

这里的 `critic/kl` 不是严格非负的 full KL，而是采样 token 上的 `old_log_probs - ref_log_probs` 口径，所以为负本身不奇怪。更值得关注的是它从 `-0.00022` 一路走到 `-0.07066`，说明策略在持续偏离 reference。

#### `system/batch_add_count` 说明复制样本仍然不少

根据 `actor/samples_total - system/batch_add_count` 可以反推出补齐前原始 step-wise 样本数：

- step0: `256 - 48 = 208`
- step1: `256 - 53 = 203`
- step2: `192 - 8 = 184`
- step3: `192 - 26 = 166`
- step4: `192 - 47 = 145`

复制比例分别约为：

- step0: `18.8%`
- step1: `20.7%`
- step2: `4.2%`
- step3: `15.7%`
- step4: `32.4%`

step4 的复制比例已经很高，这和它的 `ratio_max = 26.95` 同时出现，不是巧合。

### 2. `ppo_epochs` 的作用是什么

`ppo_epochs` 的作用非常直接：

它表示同一批 rollout 得到的 on-policy 样本，会被重复训练多少轮。

在当前 actor worker 里，`ppo_epochs` 直接决定 dataloader 重复次数：[base_worker.py:102](/e:/code/GUI/Roll-GUI/roll/pipeline/base_worker.py:102)

```python
dataloader = data.make_iterator(
    mini_batch_size=backward_batch_size,
    epochs=self.pipeline_config.ppo_epochs,
    ...
)
```

所以它会同时影响四件事：

1. 同一批样本会被重复利用多少次。
2. actor optimizer step 数会被线性放大。
3. scheduler 消耗速度会被线性放大。
4. 如果 batch 里有复制补齐样本，这些重复样本也会被重复训练更多次。

在 step-wise 模式下，这个参数尤其敏感。因为这里训练的不是 `16` 条轨迹，而是展开后的 `145 ~ 208` 条原始 step 样本，再被补齐到 `192/256`。

换句话说：

- 在普通轨迹级 PPO 里，`ppo_epochs=4` 只是“四次复用同一批轨迹”
- 在 step-wise GUI RL 里，`ppo_epochs=4` 往往意味着“对已经展开、还可能被复制过的 step 样本再多轮反复训练”

所以 step-wise 模式里，`ppo_epochs` 比普通 PPO 更容易推高 `ratio_max` 和 scheduler 消耗速度。

### 3. 为什么当前 step-wise 模式下 scheduler 会异常

#### 3.1 当前代码如何算 actor scheduler 总步数

当前 `set_max_steps()` 的核心公式在这里：[base_config.py:437](/e:/code/GUI/Roll-GUI/roll/configs/base_config.py:437)

```python
self.actor_train.training_args.max_steps = max_steps * (
    self.rollout_batch_size
    * self.actor_infer.generating_args.num_return_sequences
    * self.ppo_epochs
    // actor_backward_batch_size
)
```

其中：

```text
actor_backward_batch_size
= per_device_train_batch_size * gradient_accumulation_steps
```

随后 DeepSpeed 初始化时还会再除以 `dp_size`：[deepspeed_strategy.py:373](/e:/code/GUI/Roll-GUI/roll/distributed/strategy/deepspeed_strategy.py:373)

```python
self.worker_config.training_args.max_steps =
    self.worker_config.training_args.max_steps // self.worker.rank_info.dp_size
```

#### 3.2 按你当前 yaml 静态代入，会发生什么

当前配置：

```text
max_steps = 16
rollout_batch_size = 16
num_return_sequences = 1
ppo_epochs = 1
per_device_train_batch_size = 4
gradient_accumulation_steps = 8
```

先算：

```text
actor_backward_batch_size = 4 * 8 = 32
rollout_batch_size * num_return_sequences * ppo_epochs = 16 * 1 * 1 = 16
16 // 32 = 0
```

所以：

```text
actor_train.training_args.max_steps = 16 * 0 = 0
```

再除以 `dp_size=2` 后，仍然是：

```text
worker train max_steps = 0
```

这意味着一个非常关键的事实：

在当前 yaml 下，任何依赖 `num_training_steps` 的 scheduler，比如：

- `cosine`
- `linear`
- `polynomial`

都已经失去了合理的时间轴基础。

这比“总步数太短”更严重，因为这里是“静态推导直接为 0”。

#### 3.3 为什么这和日志里 `actor/lr = 1e-6 / 0` 交替看起来不完全一致

这说明至少有一个情况成立：

1. 当前 `worker_state_pipeline.json` 不是严格由这份 yaml 完整跑出来的。
2. 或者 scheduler 在 `num_training_steps=0` 时出现了不直观行为。
3. 或者训练过程中某处实际生效参数和 yaml 文本不完全一致。

无论是哪一种，都指向同一个结论：

当前 step-wise 模式下，scheduler 配置已经不可靠，不能再单纯依赖 `actor/lr` 的日志表象来判断训练是否正常。

### 4. step-wise 模式下真实的 actor update 数应该怎么看

step-wise 样本是在这里被展开并拼接的：

- [android_step_env_manager.py:302](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/env_manager/android_step_env_manager.py:302)
- [android_step_env_manager.py:441](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/env_manager/android_step_env_manager.py:441)

训练前又会经过 `adjust_batch()` 补齐到可整除大小：[agentic_pipeline.py:489](/e:/code/GUI/Roll-GUI/roll/pipeline/agentic/agentic_pipeline.py:489)

当前 actor 的全局训练批大小是：

```text
actor_global_backward_batch_size
= per_device_train_batch_size * gradient_accumulation_steps * dp_size
= 4 * 8 * 2
= 64
```

所以对这轮日志来说，每个 pipeline step 的真实 actor update 数更接近：

```text
actor_updates_per_pipeline_step
= ceil(actor/samples_total / 64) * ppo_epochs
```

由于当前 `ppo_epochs = 1`，于是：

- step0: `256 / 64 = 4`
- step1: `256 / 64 = 4`
- step2: `192 / 64 = 3`
- step3: `192 / 64 = 3`
- step4: `192 / 64 = 3`

如果整轮训练总共还是 `16` 个 pipeline step，那么 actor scheduler 更合理的总步数应该在：

```text
16 * 3 = 48
到
16 * 4 = 64
```

这个量级，而不是当前静态公式推导出来的 `0`。

如果未来把 `ppo_epochs` 改成 `2`，那合理量级就会变成：

```text
96 ~ 128
```

## Step-wise 模式下如何正确配置 scheduler

这里分成“短期不改代码”和“长期改代码”两类方案。

### 方案 A：不改代码时，怎样防止 scheduler 异常

在当前 `set_max_steps()` 逻辑不变的前提下，最稳妥的做法不是继续用 `cosine`，而是：

```yaml
actor_train:
  training_args:
    lr_scheduler_type: constant
    warmup_steps: 0
```

原因：

1. `constant` 不依赖精确的 `num_training_steps`。
2. 当前 step-wise 模式下，`num_training_steps` 已经被静态整除到 `0`，继续用 `cosine` 没有意义。
3. `warmup_steps: 0` 可以避免在错误时间轴上再叠加 warmup 干扰。

这是最推荐的止血方案。

如果你坚持不改代码、但还想继续用 `cosine`，至少需要满足：

```text
rollout_batch_size * num_return_sequences * ppo_epochs
>= per_device_train_batch_size * gradient_accumulation_steps
```

也就是要让：

```text
16 * 1 * ppo_epochs >= 4 * 8
```

当前等价于：

```text
ppo_epochs >= 2
```

或者把：

```text
gradient_accumulation_steps <= 4
```

否则 `set_max_steps()` 就会直接给出 `0`。

但要注意：这只能避免“静态等于 0”，并不能解决“step-wise 样本数远大于 rollout_batch_size，导致总步数估算仍然偏离真实更新次数”的根问题。

### 方案 B：改代码时，step-wise 模式下正确的 scheduler 思路

真正合理的做法是：

不要再用 `rollout_batch_size` 这种轨迹数去估 actor scheduler 总步数，而应该按真实 step-wise 样本数估算。

更接近实际的公式应该是：

```text
actor_updates_per_pipeline_step
= ceil(adjusted_step_samples_total / actor_global_backward_batch_size) * ppo_epochs

actor_scheduler_total_steps
= pipeline_max_steps * actor_updates_per_pipeline_step
```

对当前配置，可直接按日志估一个工作量级：

```text
actor_scheduler_total_steps ≈ 48 ~ 64
```

若 `ppo_epochs = 2`：

```text
actor_scheduler_total_steps ≈ 96 ~ 128
```

也就是说，step-wise 模式下的 scheduler 正确配置不是“把 `warmup_steps` 改来改去”，而是先把：

- `num_training_steps`

和真实 actor update 数对齐。

### 方案 C：step-wise 模式下推荐的参数思路

如果你现在不准备改代码，建议优先使用这一组：

```yaml
ppo_epochs: 1
use_pg_clip_range: true
pg_clip_low: 0.2
pg_clip_high: 0.28
add_token_level_kl: true
use_kl_loss: true
init_kl_coef: 0.001
kl_loss_coef: 0.01

actor_train:
  training_args:
    learning_rate: 1.0e-6
    gradient_accumulation_steps: 8
    warmup_steps: 0
    lr_scheduler_type: constant
```

原因：

1. `ppo_epochs: 1`
   先控制单个 pipeline step 内的重复更新强度。

2. `lr_scheduler_type: constant`
   绕开错误的 `num_training_steps` 估算。

3. `warmup_steps: 0`
   避免在错误 scheduler 时间轴上再叠加 warmup。

4. `use_pg_clip_range: true`
   让 `pg_clip_high: 0.28` 真正生效。

如果后续你愿意改 `set_max_steps()`，再考虑恢复到：

```yaml
ppo_epochs: 2

actor_train:
  training_args:
    lr_scheduler_type: cosine
    warmup_steps: 2
```

前提是：

- actor scheduler 总步数已经改成基于真实 step-wise 样本数估算

否则 `cosine` 只是在一条错误时间轴上工作。

## 本轮最重要的结论

这轮 run 里最需要记住的是三件事：

1. `ppo_epochs` 的本质是“同一批 on-policy 样本重复训练多少轮”。在 step-wise 模式下，它会线性放大真实 actor update 数。
2. 当前 step-wise 模式下，scheduler 的问题已经不只是“总步数偏短”，而是按当前 yaml 静态计算会直接变成 `0`。
3. 在不改代码的前提下，最稳妥的方式是把 actor scheduler 先改成 `constant`，而不是继续调 `cosine + warmup_steps`。

## 建议立刻补充到配置文档里的结论

建议在配置说明里补充下面这段：

```text
step-wise 模式下，actor scheduler 不能再按 rollout_batch_size 估算总步数，
因为真实训练对象是展开后的 step-level 样本，而不是轨迹数。

当前实现里：
actor_train.training_args.max_steps
= max_steps * (
    rollout_batch_size * num_return_sequences * ppo_epochs
    // (per_device_train_batch_size * gradient_accumulation_steps)
)

当 rollout_batch_size * num_return_sequences * ppo_epochs
小于 per_device_train_batch_size * gradient_accumulation_steps 时，
max_steps 会被整除成 0，导致 cosine/linear scheduler 异常。

因此：
1. 不改代码时，优先使用 lr_scheduler_type=constant, warmup_steps=0
2. 需要 cosine 时，必须先把 num_training_steps 改为基于真实 step-wise 样本数估算
3. ppo_epochs 会线性放大 scheduler 消耗速度，step-wise 模式下建议从 1 开始
```
