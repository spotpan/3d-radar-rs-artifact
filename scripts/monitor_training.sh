#!/bin/bash
# 监控训练状态

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== 3DMAE Training Monitor ==="
echo "Project: $PROJECT_ROOT"
echo "Timestamp: $(date)"
echo ""

# 检查GPU状态
echo "--- GPU Status ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
else
    echo "nvidia-smi not available"
fi

echo ""
echo "--- Training Processes ---"

# 检查Python训练进程
echo "Python training processes:"
ps aux | grep -E "python.*pretrain\.py|python.*finetune\.py" | grep -v grep || echo "No training processes found"

echo ""
echo "--- Screen Sessions ---"
if command -v screen &> /dev/null; then
    screen -ls 2>/dev/null | grep -i "3dmae" || echo "No 3DMAE screen sessions found"
else
    echo "screen not installed"
fi

echo ""
echo "--- Tmux Sessions ---"
if command -v tmux &> /dev/null; then
    tmux ls 2>/dev/null | grep -i "3dmae" || echo "No 3DMAE tmux sessions found"
else
    echo "tmux not installed"
fi

echo ""
echo "--- Recent Log Files ---"
find logs -name "*.log" -type f -mtime -1 2>/dev/null | head -5 || echo "No recent log files found"

echo ""
echo "--- Checkpoint Files ---"
find checkpoints -name "*.pt" -type f -mtime -1 2>/dev/null | head -5 || echo "No recent checkpoint files found"

echo ""
echo "=== Monitoring Commands ==="
echo "1. Watch GPU usage: watch -n 1 nvidia-smi"
echo "2. Follow latest log: tail -f logs/mae_pretrain/latest_training.log"
echo "3. Check tensorboard: tensorboard --logdir logs/ --port 6006"
echo "4. Check disk space: df -h ."
echo "5. Kill training: pkill -f 'python.*pretrain\.py'"