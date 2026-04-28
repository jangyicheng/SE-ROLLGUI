# Modified Files

This document records all changes made to fix swanlab metrics logging.

---

## Summary

Training metrics were not being uploaded to swanlab due to two issues:

1. The `SwanlabTracker.log()` method passed the full `metrics` dict to `swanlab.log()`, which contains non-numeric values (strings, lists, dicts, tensor objects). Swanlab's API only accepts scalar numeric types (`int`, `float`) and its own `BaseType` wrappers. When unsupported types are passed, swanlab may silently discard the log call or raise an exception.
2. The YAML config `agent_val_multiandroid_grpo.yaml` used incorrect swanlab API parameter names (`log_dir` instead of `logdir`, `name` instead of `experiment_name`), causing swanlab to initialize with wrong or missing configuration.

---

## Change 1: Filter non-numeric values in `SwanlabTracker.log()`

**File:** `roll/utils/tracking.py`

**Before:**

```python
class SwanlabTracker(BaseTracker):
    ...
    def log(self, values: dict, step: Optional[int], **kwargs):
        self.run.log(values, step=step, **kwargs)
```

**After:**

```python
class SwanlabTracker(BaseTracker):
    ...
    def log(self, values: dict, step: Optional[int], **kwargs):
        filtered = {k: v for k, v in values.items() if isinstance(v, (int, float))}
        self.run.log(filtered, step=step, **kwargs)
```

**Reasoning:** The `metrics` dict assembled in `agentic_pipeline.py` contains a wide variety of values:
- Numeric scalars (e.g., `critic/score/mean`, `time/step_rollout`) -- these are correctly logged
- Nested structures (e.g., `log_res`, a list of dicts with `"prompt"`/`"response"` strings) -- these cause swanlab to fail
- Non-serializable tensor objects that may slip through as raw Python objects

By filtering to only `int` and `float`, we ensure swanlab receives valid data on every step. This mirrors the pattern used by `TensorBoardTracker.log()`, which also skips non-scalar types.

---

## Change 2: Clear metrics dump directory before training

**File:** `roll/pipeline/agentic/agentic_pipeline.py`

**Before:**

```python
@torch.no_grad()
def run(self):
    # Calculate tokens-per-second system throughput
    tps_timer = _Timer(window_size=5)

    for global_step in range(self.pipeline_config.max_steps):
```

**After:**

```python
@torch.no_grad()
def run(self):
    # Calculate tokens-per-second system throughput
    tps_timer = _Timer(window_size=5)

    if self.pipeline_config.metrics_dump_dir:
        import shutil
        if os.path.exists(self.pipeline_config.metrics_dump_dir):
            shutil.rmtree(self.pipeline_config.metrics_dump_dir)
        os.makedirs(self.pipeline_config.metrics_dump_dir, exist_ok=True)

    for global_step in range(self.pipeline_config.max_steps):
```

**Reasoning:** The training loop writes per-step metrics to `output/metrics/step_N/metrics.jsonl`. If a training run is resumed or re-run, old metric files from previous runs would accumulate alongside new ones. By clearing and recreating the directory at the start of `run()`, we guarantee that only metrics from the current training run are stored.

The `if self.pipeline_config.metrics_dump_dir:` guard ensures the block is skipped safely when the field is `null` or unset.

---

## Change 3: Fix swanlab YAML config parameter names

**File:** `jyc/agent_val_multiandroid_grpo.yaml`

**Before:**

```yaml
track_with: swanlab
tracker_kwargs:
  api_key: 4FsBwIEyUhCvOwlPJyr7g
  project: GUIAgent-Online-RL
  name: ${exp_name}
  tags: 
    - online_rl
    - grpo
  group: training
  description: GUI online RL
  log_dir: roll_exp/android
```

**After:**

```yaml
track_with: swanlab
tracker_kwargs:
  login_kwargs:
    api_key: 4FsBwIEyUhCvOwlPJyr7g
  project: GUIAgent-Online-RL
  experiment_name: ${exp_name}
  tags:
    - online_rl
    - grpo
  description: GUI online RL
  logdir: roll_exp/android
```

**Reasoning:** The Swanlab Python API (`swanlab.init()`) uses:
- `logdir` (no underscore) for the offline log directory -- `log_dir` is silently ignored
- `experiment_name` for the experiment name -- `name` is not a recognized kwarg and passes through to `**kwargs` unused
- `login_kwargs` (a dict) to pass authentication credentials -- `api_key` at the top level is not a valid swanlab.init kwarg and is also silently ignored

The corrected config passes the API key inside `login_kwargs`, uses the correct parameter names, and removes the invalid `group` key.

---

## Swanlab Metrics Logged

After these changes, the following categories of numeric metrics are uploaded to swanlab every training step:

| Category | Example Metrics |
|---|---|
| **Score / Reward** | `critic/score/mean`, `critic/score/max`, `critic/score/min`, `critic/rewards/mean`, `critic/rewards/max`, `critic/rewards/min` |
| **Advantage / Returns** | `critic/advantages/mean`, `critic/advantages/max`, `critic/advantages/min`, `critic/returns/mean`, `critic/returns/max`, `critic/returns/min` |
| **Token Lengths** | `tokens/response_length/mean`, `tokens/response_length/max`, `tokens/response_length/min`, `tokens/prompt_length/mean`, `tokens/prompt_length/max`, `tokens/prompt_length/min`, `tokens/non_prompt_length/mean` |
| **Environment Timing** | `env/traj_rollout_time/mean`, `env/traj_rollout_time/max`, `env/traj_rollout_time/min`, `env/traj_env_time/mean`, `env/traj_env_time/max`, `env/traj_env_time/min` |
| **Optional** (if present in batch) | `critic/values/mean`, `critic/values/max`, `critic/values/min`, `critic/episode_rewards_norm/mean`, `critic/step_rewards_norm/mean` |
| **Actor Training** | `actor/loss`, `actor/pg_loss`, `actor/kl_loss`, `actor/entropy_loss`, `actor/approxkl`, `actor/clipfrac` |
| **Critic Training** | `critic/loss`, `critic/error` |
| **Timing Overhead** | `time/step_total`, `time/step_model_update`, `time/step_rollout`, `time/step_train`, `time/step_compute_data_metrics`, `time/step_log`, `time/step_val` |
| **System** | `system/tps`, `system/samples` |
| **Validation** | `val/score/mean`, `val/score/max`, `val/score/min` |
