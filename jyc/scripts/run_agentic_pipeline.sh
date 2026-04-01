#!/bin/bash
set +x

export ROLL_PATH="/HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL" #要改动
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"
# debug mode for ray
# export RAY_DEBUG="legacy"
export MASTER_PORT=6422
export DASHBOARD_PORT=8299

# bash emulator_stop.sh
# bash emulator_start.sh
MODEL_PARAM=$1

CONFIG_NAME="agent_val_multiandroid_grpo"
if [ "$MODEL_PARAM" = "lora" ]; then
    CONFIG_NAME="agent_val_multiandroid_grpo_lora"
fi



python jyc/start_agentic_pipeline.py \
    --config_path "." \
    --config_name $CONFIG_NAME
