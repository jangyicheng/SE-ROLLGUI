#!/bin/bash
#
# run_exploration.sh
#
# 自进化训练前的探索数据准备脚本。
# 在启动自进化训练之前，必须先运行此脚本生成初始化数据。
#
# 使用方法:
#   bash jyc/scripts/run_exploration.sh
#
# 自定义模型后端（默认使用 vLLM 本地服务）:
#   bash jyc/scripts/run_exploration.sh \
#       --model_backend vllm \
#       --model_name Qwen/Qwen2.5-VL-7B-Instruct \
#       --vllm_base_url http://localhost:8000/v1
#
# 使用 OpenAI API:
#   bash jyc/scripts/run_exploration.sh \
#       --model_backend openai \
#       --model_name gpt-4o
#
set +x

# =============================================
# 基础路径配置（请根据实际环境修改）
# =============================================
export ROLL_PATH="/HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# =============================================
# 输出目录配置
# =============================================
EXPLORATION_OUTPUT_DIR="${EXPLORATION_OUTPUT_DIR:-./exploration_output}"
INIT_OUTPUT_DIR="${INIT_OUTPUT_DIR:-./init_output}"

# =============================================
# 探索参数配置
# =============================================
NUM_EPISODES="${NUM_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-30}"

# =============================================
# MobileWorld 端口配置
# =============================================
CONSOLE_PORT="${CONSOLE_PORT:-5554}"
GRPC_PORT="${GRPC_PORT:-8554}"

# =============================================
# VLM 模型后端配置
# =============================================
# 可选值: none | vllm | openai | huggingface
MODEL_BACKEND="${MODEL_BACKEND:-vllm}"

# 模型名称（根据实际部署的模型修改）
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"

# vLLM / OpenAI-compatible 服务地址
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"

# 采样参数
MODEL_TEMPERATURE="${MODEL_TEMPERATURE:-1.0}"
MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS:-256}"

# =============================================
# 解析命令行参数
# =============================================
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_backend)
            MODEL_BACKEND="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --vllm_base_url)
            VLLM_BASE_URL="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --exploration_output_dir)
            EXPLORATION_OUTPUT_DIR="$2"
            shift 2
            ;;
        --init_output_dir)
            INIT_OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}"

echo "============================================="
echo "  Self-Evolve Exploration Phase"
echo "============================================="
echo "  Exploration output: $EXPLORATION_OUTPUT_DIR"
echo "  Init output:        $INIT_OUTPUT_DIR"
echo "  Model backend:      $MODEL_BACKEND"
echo "  Model name:         $MODEL_NAME"
echo "  vLLM base URL:      $VLLM_BASE_URL"
echo "  Episodes:           $NUM_EPISODES"
echo "  Max steps/ep:       $MAX_STEPS"
echo "  Console port:       $CONSOLE_PORT"
echo "  gRPC port:          $GRPC_PORT"
echo "============================================="

cd "$ROLL_PATH"

# =============================================
# 步骤 1：运行 Explorer（带模型后端）
# =============================================
echo ""
echo "[Step 1/2] Running Explorer with model backend: $MODEL_BACKEND"
echo "Command: python roll/pipeline/agentic/env/android/exploration/scripts/run_exploration.py"
echo "         --env mobileworld"
echo "         --server_url http://localhost:18000"
echo "         --model_backend $MODEL_BACKEND"
echo "         --model_name $MODEL_NAME"
echo "         --output_dir $EXPLORATION_OUTPUT_DIR"
echo ""

if [ "$MODEL_BACKEND" = "vllm" ]; then
    python roll/pipeline/agentic/env/android/exploration/scripts/run_exploration.py \
        --env mobileworld \
        --server_url http://localhost:18000 \
        --model_backend vllm \
        --model_name "$MODEL_NAME" \
        --vllm_base_url "$VLLM_BASE_URL" \
        --model_temperature "$MODEL_TEMPERATURE" \
        --model_max_tokens "$MODEL_MAX_TOKENS" \
        --output_dir "$EXPLORATION_OUTPUT_DIR" \
        --num_episodes "$NUM_EPISODES" \
        --max_steps "$MAX_STEPS" \
        --console_port "$CONSOLE_PORT" \
        --grpc_port "$GRPC_PORT"
elif [ "$MODEL_BACKEND" = "openai" ]; then
    python roll/pipeline/agentic/env/android/exploration/scripts/run_exploration.py \
        --env mobileworld \
        --server_url http://localhost:18000 \
        --model_backend openai \
        --model_name "$MODEL_NAME" \
        --model_temperature "$MODEL_TEMPERATURE" \
        --model_max_tokens "$MODEL_MAX_TOKENS" \
        --output_dir "$EXPLORATION_OUTPUT_DIR" \
        --num_episodes "$NUM_EPISODES" \
        --max_steps "$MAX_STEPS" \
        --console_port "$CONSOLE_PORT" \
        --grpc_port "$GRPC_PORT"
else
    # none / 随机动作
    python roll/pipeline/agentic/env/android/exploration/scripts/run_exploration.py \
        --env mobileworld \
        --server_url http://localhost:18000 \
        --model_backend none \
        --output_dir "$EXPLORATION_OUTPUT_DIR" \
        --num_episodes "$NUM_EPISODES" \
        --max_steps "$MAX_STEPS" \
        --console_port "$CONSOLE_PORT" \
        --grpc_port "$GRPC_PORT"
fi

EXPLORER_EXIT=$?
if [ $EXPLORER_EXIT -ne 0 ]; then
    echo ""
    echo "[ERROR] Explorer failed with exit code $EXPLORER_EXIT. Please check logs above."
    exit 1
fi

echo ""
echo "[Step 1/2] Explorer completed successfully."

# =============================================
# 步骤 2：运行 Task Initializer
# =============================================
echo ""
echo "[Step 2/2] Running Task Initializer..."
echo "Command: python roll/pipeline/agentic/env/android/exploration/scripts/run_task_init.py"
echo "         --exploration_dir $EXPLORATION_OUTPUT_DIR"
echo "         --init_output_dir $INIT_OUTPUT_DIR"
echo "         --env mobileworld"
echo ""

python roll/pipeline/agentic/env/android/exploration/scripts/run_task_init.py \
    --exploration_dir "$EXPLORATION_OUTPUT_DIR" \
    --init_output_dir "$INIT_OUTPUT_DIR" \
    --env mobileworld

INIT_EXIT=$?
if [ $INIT_EXIT -ne 0 ]; then
    echo ""
    echo "[ERROR] Task Initializer failed with exit code $INIT_EXIT. Please check logs above."
    exit 1
fi

echo ""
echo "[Step 2/2] Task Initializer completed successfully."

# =============================================
# 步骤 3：验证输出
# =============================================
echo ""
echo "============================================="
echo "  Exploration Output Verification"
echo "============================================="
if [ -d "$EXPLORATION_OUTPUT_DIR" ]; then
    EXP_COUNT=$(find "$EXPLORATION_OUTPUT_DIR" -maxdepth 1 -type d | wc -l)
    EXP_COUNT=$((EXP_COUNT - 1))
    echo "  Exploration dirs found: $EXP_COUNT"
    echo "  Directory: $EXPLORATION_OUTPUT_DIR"
else
    echo "  [WARNING] Exploration output directory not found."
fi

if [ -d "$INIT_OUTPUT_DIR" ]; then
    INIT_COUNT=$(find "$INIT_OUTPUT_DIR" -name "task_init_result.json" 2>/dev/null | wc -l)
    echo "  Init results found: $INIT_COUNT"
    echo "  Directory: $INIT_OUTPUT_DIR"
else
    echo "  [WARNING] Init output directory not found."
fi
echo "============================================="

echo ""
echo "[DONE] Exploration phase complete."
echo "Now you can start self-evolve training with:"
echo "  bash jyc/scripts/run_self_evolve_pipeline.sh"
echo ""
