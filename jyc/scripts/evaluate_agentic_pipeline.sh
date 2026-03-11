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

python jyc/evaluate_agentic_pipeline.py \
    --config_path "." \
    --config_name agent_val_android_grpo_evaluate



