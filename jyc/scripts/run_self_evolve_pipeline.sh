#!/bin/bash
#
# run_self_evolve_pipeline.sh
#
# 启动自进化模式训练脚本。
# 使用方法:
#   bash jyc/scripts/run_self_evolve_pipeline.sh
#
# 可选参数（覆盖 Hydra 配置）:
#   bash jyc/scripts/run_self_evolve_pipeline.sh rollout_batch_size=64 max_steps=64
#
set +x

# =============================================
# 基础路径配置（请根据实际环境修改）
# =============================================
export ROLL_PATH="/HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# Ray 和分布式训练端口配置
export MASTER_PORT=6422
export DASHBOARD_PORT=8299

# =============================================
# MobileWorld 服务配置
# 如已有运行中的服务，跳过以下 ssh -fN 命令
# =============================================
# 转发 MobileWorld API 服务端口（需要按实际地址修改）
# ssh -fN -L 18000:localhost:8000 -p 30115 root@121.46.19.2
# ssh -fN -L 18001:localhost:8000 -p 30116 root@121.46.19.2

# =============================================
# 训练配置
# =============================================
# 自进化训练使用专用配置（jyc/agent_val_self_evolve.yaml）
CONFIG_NAME="agent_val_self_evolve"

# 默认覆盖参数（可命令行传入）
# rollout_batch_size: 每步任务数量，建议 16-64
# max_steps: 总训练步数，建议 32-128
# self_evolve.round_update_interval: 每隔多少步触发一次任务生成

OVERRIDES=""
for arg in "$@"; do
    OVERRIDES="$OVERRIDES $arg"
done

echo "============================================="
echo "  Self-Evolving Mode Training"
echo "============================================="
echo "  Config:       $CONFIG_NAME"
echo "  Overrides:   ${OVERRIDES:-none}"
echo "  Feedback:     ./trajectories/self_evolve"
echo "  Tasks root:   ./data/tasks/generated"
echo "============================================="

cd "$ROLL_PATH"

python jyc/start_agentic_pipeline.py \
    --config_path "." \
    --config_name $CONFIG_NAME \
    $OVERRIDES
