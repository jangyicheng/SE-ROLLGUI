#!/bin/bash
# set +x
export ROLL_PATH="/HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL"
# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# export CONFIG_PATH="$SCRIPT_DIR"
CONFIG_PATH=$(basename $(dirname $0))
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"
echo "当前路径: $(pwd)"
echo "ROLL_PATH: $ROLL_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "CONFIG_PATH: $CONFIG_PATH"
# echo "检查配置目录是否存在: $(ls -la $CONFIG_PATH 2>/dev/null || echo "目录不存在")"
python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agent_val_frozen_lake_single_node_demo_qwen3
