#!/bin/bash
# 后台运行3DMAE预训练脚本

set -e

# 设置环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 创建日志目录
mkdir -p logs/background_runs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/background_runs/pretrain_${TIMESTAMP}.log"
PID_FILE="logs/background_runs/pretrain_${TIMESTAMP}.pid"

echo "Starting 3DMAE pretraining in background..."
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"

# 使用nohup在后台运行
nohup python training/pretrain.py \
    --experiment-name "mae_pretrain_${TIMESTAMP}" \
    > "$LOG_FILE" 2>&1 &

# 保存进程ID
PRETRAIN_PID=$!
echo $PRETRAIN_PID > "$PID_FILE"

echo "Training started with PID: $PRETRAIN_PID"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To stop training: kill $PRETRAIN_PID"