#!/bin/bash
# 使用tmux在后台运行3DMAE预训练

set -e

# 设置环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_NAME="3dmae_pretrain_${TIMESTAMP}"

echo "Starting 3DMAE pretraining in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  To attach to session: tmux attach -t $SESSION_NAME"
echo "  To detach: Ctrl+B, then D"
echo "  To list sessions: tmux ls"
echo ""

# 检查tmux是否已安装
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# 创建新的tmux会话并运行训练
tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && python training/pretrain.py --experiment-name 'mae_pretrain_${TIMESTAMP}'"

echo "Tmux session created."
echo "To attach and monitor: tmux attach -t $SESSION_NAME"
echo "To list all sessions: tmux ls"