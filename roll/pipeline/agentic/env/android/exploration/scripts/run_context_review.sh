#!/bin/bash

# Task Initialization (Context Review) Launch Script for AndroidWorld
# Usage: bash run_context_review.sh [OPTIONS]

set -e

# Default values
SERVER_URL="${ANDROID_SERVER_URL:-http://localhost:8000}"
OUTPUT_DIR="${INIT_OUTPUT_DIR:-./androidworld_init_output}"
PARAMS_DIR="${PARAMS_DIR:-}"
TASK_POOL="${INIT_TASK_POOL:-train}"
NUM_INSTANCES="${NUM_INSTANCES:-1}"
TASK_RANDOM_SEED="${TASK_RANDOM_SEED:-42}"
CONSOLE_PORT="${ANDROID_CONSOLE_PORT:-5554}"
GRPC_PORT="${ANDROID_GRPC_PORT:-8554}"
ADB_PATH="${ADB_PATH:-/root/android-sdk/platform-tools/adb}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server_url)
            SERVER_URL="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --params_dir)
            PARAMS_DIR="$2"
            shift 2
            ;;
        --task_pool)
            TASK_POOL="$2"
            shift 2
            ;;
        --num_instances)
            NUM_INSTANCES="$2"
            shift 2
            ;;
        --task_random_seed)
            TASK_RANDOM_SEED="$2"
            shift 2
            ;;
        --console_port)
            CONSOLE_PORT="$2"
            shift 2
            ;;
        --grpc_port)
            GRPC_PORT="$2"
            shift 2
            ;;
        --enable_vlm_verification)
            VLM_VERIFY="--enable_vlm_verification"
            shift
            ;;
        --help|-h)
            echo "Usage: bash run_context_review.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --server_url URL              Server URL (default: $SERVER_URL)"
            echo "  --output_dir DIR              Output directory (default: $OUTPUT_DIR)"
            echo "  --params_dir DIR             Params directory"
            echo "  --task_pool POOL             Task pool: auto, all, train, info_retrieval, or comma-separated (default: $TASK_POOL)"
            echo "  --num_instances N            Instances per task (default: $NUM_INSTANCES)"
            echo "  --task_random_seed N         Random seed (default: $TASK_RANDOM_SEED)"
            echo "  --console_port PORT           Console port (default: $CONSOLE_PORT)"
            echo "  --grpc_port PORT              gRPC port (default: $GRPC_PORT)"
            echo "  --enable_vlm_verification     Enable VLM verification"
            echo "  --help, -h                   Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  ANDROID_SERVER_URL    Server URL"
            echo "  INIT_OUTPUT_DIR      Output directory"
            echo "  PARAMS_DIR           Params directory"
            echo "  INIT_TASK_POOL       Task pool selection"
            echo "  NUM_INSTANCES        Instances per task"
            echo "  TASK_RANDOM_SEED     Random seed"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build params
PARAMS_ARGS=""
if [[ -n "$PARAMS_DIR" ]]; then
    PARAMS_ARGS="--params_dir $PARAMS_DIR"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPLORATION_MODULE_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo "AndroidWorld Task Initialization (Context Review)"
echo "================================================"
echo "Server URL:       $SERVER_URL"
echo "Output dir:       $OUTPUT_DIR"
echo "Params dir:       ${PARAMS_DIR:-<auto>}"
echo "Task pool:        $TASK_POOL"
echo "Num instances:    $NUM_INSTANCES"
echo "Random seed:      $TASK_RANDOM_SEED"
echo "Console port:     $CONSOLE_PORT"
echo "gRPC port:        $GRPC_PORT"
echo "================================================"
echo ""

# Run task initialization
python "$EXPLORATION_MODULE_DIR/scripts/run_task_init.py" \
    --env androidworld \
    --server_url "$SERVER_URL" \
    --output_dir "$OUTPUT_DIR" \
    --task_pool "$TASK_POOL" \
    --num_instances "$NUM_INSTANCES" \
    --task_random_seed "$TASK_RANDOM_SEED" \
    --console_port "$CONSOLE_PORT" \
    --grpc_port "$GRPC_PORT" \
    $PARAMS_ARGS \
    $VLM_VERIFY

echo ""
echo "Task initialization finished!"
