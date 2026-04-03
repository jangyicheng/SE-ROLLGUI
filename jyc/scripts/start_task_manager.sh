#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   ./start_task_manager.sh train
#   ./start_task_manager.sh eval
#   ./start_task_manager.sh both
MODE="${1:-both}"

ROOT="/HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANAGER_PY="$ROOT/ROLL/roll/pipeline/agentic/env/android/GuiTaskEvalManager.py"
TRAJ_ROOT="$ROOT/ROLL/trajectories"

TRAIN_PORT="${TRAIN_PORT:-5001}"
EVAL_PORT="${EVAL_PORT:-5002}"
HOST="${HOST:-0.0.0.0}"

RUN_DIR="$TRAJ_ROOT/task_manager_runtime"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

# 两个调度器共享同一个 timestamp
SHARED_TS="$(date +%Y-%m-%d_%H%M%S)"

start_one () {
  local m="$1"
  local p="$2"
  local log_file="$LOG_DIR/${m}.log"
  local pid_file="$PID_DIR/${m}.pid"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "[skip] $m 已在运行, pid=$(cat "$pid_file"), port=$p"
    return 0
  fi

  echo "[start] mode=$m port=$p ts=$SHARED_TS"
  nohup env \
    TASK_MANAGER_MODE="$m" \
    TRAJECTORY_ROOT="$TRAJ_ROOT" \
    TASK_MANAGER_TIMESTAMP="$SHARED_TS" \
    "$PYTHON_BIN" "$MANAGER_PY" \
      --host "$HOST" \
      --port "$p" \
      --mode "$m" \
      --trajectory_root "$TRAJ_ROOT" \
    > "$log_file" 2>&1 &

  echo $! > "$pid_file"
  echo "[ok] $m pid=$! log=$log_file"
}

case "$MODE" in
  train)
    start_one train "$TRAIN_PORT"
    ;;
  eval)
    start_one eval "$EVAL_PORT"
    ;;
  both)
    start_one train "$TRAIN_PORT"
    start_one eval "$EVAL_PORT"
    ;;
  *)
    echo "错误: MODE 只能是 train / eval / both"
    exit 1
    ;;
esac

echo "[done] 启动完成。共享 timestamp: $SHARED_TS"
echo "查看日志: tail -f $LOG_DIR/train.log 或 tail -f $LOG_DIR/eval.log"
