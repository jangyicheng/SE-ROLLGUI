#!/bin/bash

# Environment Exploration Launch Script for AndroidWorld
# Usage: bash run_exploration.sh [OPTIONS]

set -e

# Default values
SERVER_URL="${ANDROID_SERVER_URL:-http://localhost:8000}"
MAX_STEPS="${EXPLORATION_MAX_STEPS:-50}"
OUTPUT_DIR="${EXPLORATION_OUTPUT_DIR:-./androidworld_exploration_output}"
CONSOLE_PORT="${ANDROID_CONSOLE_PORT:-5554}"
GRPC_PORT="${ANDROID_GRPC_PORT:-8554}"
ADB_PATH="${ADB_PATH:-/root/android-sdk/platform-tools/adb}"
EXPLORATION_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server_url)
            SERVER_URL="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
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
        --exploration_id)
            EXPLORATION_ID="--exploration_id $2"
            shift 2
            ;;
        --no_save_screenshots)
            NO_SCREENSHOTS="--no_save_screenshots"
            shift
            ;;
        --no_log_trajectory)
            NO_TRAJECTORY="--no_log_trajectory"
            shift
            ;;
        --help|-h)
            echo "Usage: bash run_exploration.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --server_url URL         Server URL (default: $SERVER_URL)"
            echo "  --max_steps N            Maximum exploration steps (default: $MAX_STEPS)"
            echo "  --output_dir DIR         Output directory (default: $OUTPUT_DIR)"
            echo "  --console_port PORT      Console port (default: $CONSOLE_PORT)"
            echo "  --grpc_port PORT         gRPC port (default: $GRPC_PORT)"
            echo "  --exploration_id ID      Custom exploration ID"
            echo "  --no_save_screenshots    Disable screenshot saving"
            echo "  --no_log_trajectory      Disable trajectory logging"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  ANDROID_SERVER_URL       Server URL"
            echo "  EXPLORATION_MAX_STEPS    Maximum steps"
            echo "  EXPLORATION_OUTPUT_DIR   Output directory"
            echo "  ANDROID_CONSOLE_PORT     Console port"
            echo "  ANDROID_GRPC_PORT        gRPC port"
            echo "  ADB_PATH                 Path to ADB"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPLORATION_MODULE_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo "AndroidWorld Environment Exploration"
echo "================================================"
echo "Server URL:     $SERVER_URL"
echo "Max steps:      $MAX_STEPS"
echo "Output dir:     $OUTPUT_DIR"
echo "Console port:   $CONSOLE_PORT"
echo "gRPC port:      $GRPC_PORT"
echo "ADB path:       $ADB_PATH"
echo "================================================"
echo ""

# Run exploration
python "$EXPLORATION_MODULE_DIR/scripts/run_exploration.py" \
    --env androidworld \
    --server_url "$SERVER_URL" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --console_port "$CONSOLE_PORT" \
    --grpc_port "$GRPC_PORT" \
    --adb_path "$ADB_PATH" \
    $EXPLORATION_ID \
    $NO_SCREENSHOTS \
    $NO_TRAJECTORY

echo ""
echo "Exploration finished!"
