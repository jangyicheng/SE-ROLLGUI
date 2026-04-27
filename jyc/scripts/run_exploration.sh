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
set +x

# =============================================
# 基础路径配置（请根据实际环境修改）
# =============================================
export ROLL_PATH="/HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# =============================================
# 输出目录配置
# =============================================
EXPLORATION_OUTPUT_DIR="./exploration_output"
INIT_OUTPUT_DIR="./init_output"
EXPLORATION_MODEL="gpt-4o"          # 探索任务生成模型（可改为 gpt-4o-mini 等）
INIT_MODEL="gpt-4o"                  # 任务初始化模型

# =============================================
# 探索参数配置
# =============================================
NUM_EPISODES=20          # 探索 episode 数量（建议 10-50）
MAX_STEPS=30             # 每个 episode 最大步数

echo "============================================="
echo "  Self-Evolve Exploration Phase"
echo "============================================="
echo "  Exploration output: $EXPLORATION_OUTPUT_DIR"
echo "  Init output:        $INIT_OUTPUT_DIR"
echo "  Model:              $EXPLORATION_MODEL"
echo "  Episodes:           $NUM_EPISODES"
echo "  Max steps/ep:       $MAX_STEPS"
echo "============================================="

cd "$ROLL_PATH"

# =============================================
# 步骤 1：运行 Explorer
# =============================================
echo ""
echo "[Step 1/2] Running Explorer..."
echo "Command: python roll/pipeline/agentic/env/android/exploration/explorer.py"
echo "         --output_dir $EXPLORATION_OUTPUT_DIR"
echo "         --env_type mobileworld"
echo "         --num_episodes $NUM_EPISODES"
echo "         --max_steps $MAX_STEPS"
echo ""

python roll/pipeline/agentic/env/android/exploration/explorer.py \
    --output_dir "$EXPLORATION_OUTPUT_DIR" \
    --env_type mobileworld \
    --num_episodes "$NUM_EPISODES" \
    --max_steps "$MAX_STEPS"

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Explorer failed. Please check logs above."
    exit 1
fi

echo ""
echo "[Step 1/2] Explorer completed successfully."

# =============================================
# 步骤 2：运行 Task Initializer
# =============================================
echo ""
echo "[Step 2/2] Running Task Initializer..."
echo "Command: python roll/pipeline/agentic/env/android/exploration/task_initializer.py"
echo "         --exploration_dir $EXPLORATION_OUTPUT_DIR"
echo "         --init_output_dir $INIT_OUTPUT_DIR"
echo "         --env_type mobileworld"
echo ""

python roll/pipeline/agentic/env/android/exploration/task_initializer.py \
    --exploration_dir "$EXPLORATION_OUTPUT_DIR" \
    --init_output_dir "$INIT_OUTPUT_DIR" \
    --env_type mobileworld

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Task Initializer failed. Please check logs above."
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
