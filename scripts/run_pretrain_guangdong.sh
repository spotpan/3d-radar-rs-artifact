#!/bin/bash
# 3DMAE 预训练启动脚本 - 广东雷达数据
# 数据路径: /path/to/radar_station_dataset/
# GPU: CUDA-compatible GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_NAME="3dmae_pretrain_gd_${TIMESTAMP}"

# 检测 tmux
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

echo "=========================================="
echo " 3DMAE Pretraining - Guangdong Data"
echo "=========================================="
echo "Project:    $PROJECT_ROOT"
echo "Data:       /path/to/radar_station_dataset/"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ' | sed 's/,$//')"
echo "Session:    $SESSION_NAME"
echo "Timestamp:  $TIMESTAMP"
echo ""
echo "Commands:"
echo "  Attach:   tmux attach -t $SESSION_NAME"
echo "  Detach:   Ctrl+B, D"
echo "  List:     tmux ls"
echo "  Monitor:  tail -f logs/mae_pretrain_guangdong/*.log"
echo "  TB:       tensorboard --logdir=logs/mae_pretrain_guangdong --port=6006"
echo "=========================================="
echo ""

# 创建 tmux 会话并运行训练
tmux new-session -d -s "$SESSION_NAME" bash -c "
    cd '$PROJECT_ROOT'

    echo '[$(date)] Starting 3DMAE pretraining...'
    echo '[$(date)] Data: /path/to/radar_station_dataset/'
    echo '[$(date)] Config: configs/pretrain_guangdong_config.py'
    echo '[$(date)] Logs: logs/mae_pretrain_guangdong/'
    echo '[$(date)] Checkpoints: checkpoints/mae_pretrain_guangdong/'
    echo ''

    python training/pretrain_guangdong.py \
        --experiment-name 'mae_pretrain_guangdong_${TIMESTAMP}'

    echo ''
    echo '[$(date)] Training completed.'
    echo 'Press Enter to exit.'
    read
"

echo "Training started in tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
