#!/bin/bash
# 使用screen在后台运行3DMAE预训练

set -e

# 设置环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_NAME="3dmae_pretrain_${TIMESTAMP}"

echo "Starting 3DMAE pretraining in screen session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  To attach to session: screen -r $SESSION_NAME"
echo "  To detach: Ctrl+A, then D"
echo "  To list sessions: screen -ls"
echo ""

# 在screen会话中运行训练
screen -S "$SESSION_NAME" -dm bash -c "
    echo 'Starting 3DMAE pretraining...';
    echo 'Session: $SESSION_NAME';
    echo 'Timestamp: $TIMESTAMP';
    echo '';
    python training/pretrain.py --experiment-name 'mae_pretrain_${TIMESTAMP}';
    echo '';
    echo 'Training completed. Press Enter to exit.';
    read
"

echo "Screen session created."
echo "To attach and monitor: screen -r $SESSION_NAME"
echo "To list all sessions: screen -ls"