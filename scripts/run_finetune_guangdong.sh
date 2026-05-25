#!/bin/bash
# 3DMAE 降水微调启动脚本 - 广东雷达数据
# 数据路径: /path/to/radar_station_dataset/
# GPU: CUDA-compatible GPU
# 说明: 使用3帧30分钟间隔的雷达序列，预测降水量

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_NAME="3dmae_finetune_gd_${TIMESTAMP}"

# 默认 MAE 预训练 checkpoint（最近的）
DEFAULT_CKPT=$(ls -t checkpoints/mae_pretrain_guangdong/*/best_model.pt 2>/dev/null | head -1)
if [ -z "$DEFAULT_CKPT" ]; then
    DEFAULT_CKPT=$(ls -t checkpoints/mae_pretrain_guangdong/*/checkpoint_epoch_*.pt 2>/dev/null | head -1)
fi

if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

echo "=========================================="
echo " 3DMAE Precipitation Fine-tuning"
echo "  Guangdong Radar Data"
echo "=========================================="
echo "Project:    $PROJECT_ROOT"
echo "Data:       /path/to/radar_station_dataset/"
echo "Config:     configs/finetune_guangdong_config.py"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ' | sed 's/,$//')"
echo "MAE ckpt:   ${DEFAULT_CKPT:-"(not found - use --mae-checkpoint)"}"
echo "Session:    $SESSION_NAME"
echo "Timestamp:  $TIMESTAMP"
echo ""
echo "Commands:"
echo "  Attach:   tmux attach -t $SESSION_NAME"
echo "  Detach:   Ctrl+B, D"
echo "  Monitor:  tail -f logs/precipitation_guangdong/*.log"
echo "  TB:       tensorboard --logdir=logs/precipitation_guangdong --port=6006"
echo "=========================================="
echo ""

if [ -z "$DEFAULT_CKPT" ]; then
    echo "WARNING: No pretrained MAE checkpoint found!"
    echo "Please train MAE first using: bash scripts/run_pretrain_guangdong.sh"
    echo "Then provide checkpoint: --mae-checkpoint <path>"
    echo ""
fi

# Build command
CMD="python training/finetune_guangdong.py --experiment-name 'precip_guangdong_${TIMESTAMP}'"
if [ -n "$DEFAULT_CKPT" ]; then
    CMD="$CMD --mae-checkpoint '$DEFAULT_CKPT'"
fi

tmux new-session -d -s "$SESSION_NAME" bash -c "
    cd '$PROJECT_ROOT'

    echo '[$(date)] Starting fine-tuning...'
    echo '[$(date)] Data: /path/to/radar_station_dataset/'
    echo '[$(date)] Logs: logs/precipitation_guangdong/'
    echo '[$(date)] Checkpoints: checkpoints/precipitation_guangdong/'
    echo ''

    $CMD

    echo ''
    echo '[$(date)] Fine-tuning completed.'
    echo 'Press Enter to exit.'
    read
"

echo "Training started in tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo ""
echo "To specify a different MAE checkpoint:"
echo "  bash scripts/run_finetune_guangdong.sh --mae-checkpoint /path/to/mae/best_model.pt"
