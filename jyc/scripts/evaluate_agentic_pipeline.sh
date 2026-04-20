#!/bin/bash
set +x
# source /app/bin/proxy.sh
export ROLL_PATH="/HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL" #要改动
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"
# debug mode for ray
# export RAY_DEBUG="legacy"
export MASTER_PORT=6422
export DASHBOARD_PORT=8299

# bash emulator_stop.sh
# bash emulator_start.sh
# ssh -fN -L 18000:localhost:8000 -p 30115 root@121.46.19.2 

MODEL_PARAM=$1
# 环境默认值为android_world
ENV=$2
if [ -z "$ENV" ]; then
    ENV="android_world"
fi

# 如果环境为android_world，则使用agent_val_multiandroid_grpo_evaluate配置，否则使用agent_val_multimobileworld_grpo系列配置
if [ "$ENV" = "android_world" ]; then
    CONFIG_NAME="agent_val_multiandroid_grpo_evaluate"
    if [ "$MODEL_PARAM" = "voyager" ]; then
        CONFIG_NAME="agent_val_multiandroid_grpo_evaluate_voyager"
    fi
    if [ "$MODEL_PARAM" = "guiowl" ]; then
        CONFIG_NAME="agent_val_multiandroid_grpo_evaluate_guiowl"
    fi
    if [ "$MODEL_PARAM" = "reflection" ]; then
        CONFIG_NAME="agent_val_multiandroid_grpo_evaluate_reflection"
    fi
fi
if [ "$ENV" = "mobile_world" ]; then
    CONFIG_NAME="agent_val_multimobileworld_evaluate"
    if [ "$MODEL_PARAM" = "voyager" ]; then
        CONFIG_NAME="agent_val_multimobileworld_evaluate_voyager"
    fi
fi


python jyc/evaluate_agentic_pipeline.py \
    --config_path "." \
    --config_name $CONFIG_NAME