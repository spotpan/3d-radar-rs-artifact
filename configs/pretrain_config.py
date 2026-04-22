import torch
from datetime import datetime

# Data configuration
DATA_CONFIG = {
    'data_paths': [
        # '/mnt/md1/hxc/guangdong/train_select/time_radar_rain_2022.h5',
        # '/mnt/md1/hxc/guangdong/train_select/time_radar_rain_2023.h5',
        '/mnt/md1/hxc/guangdong/train_select/time_radar_rain_2025.h5',
    ],
    'radar_height_layers': [0, 1, 2, 3, 4, 5],  # First 6 layers
    'spatial_size': (700, 900),
    'batch_size': 4,  # From paper
    'num_workers': 4,
    'pin_memory': True,
    'shuffle': True,
}

# Model configuration
MODEL_CONFIG = {
    'in_channels': 6,
    'img_size': (700, 900),
    'patch_size': 16,
    'encoder_dim': 768,
    'encoder_depth': 16,
    'encoder_num_heads': 12,
    'decoder_dim': 512,
    'decoder_depth': 8,
    'decoder_num_heads': 8,
    'mask_ratio': 0.8,
}

# Training configuration
TRAIN_CONFIG = {
    'num_epochs': 200,
    'learning_rate': 1.5e-4,
    'weight_decay': 0.05,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'use_amp': True,

    # Learning rate scheduler (cosine annealing)
    'scheduler': 'cosine',
    'warmup_epochs': 10,
    'min_lr': 1e-6,

    # Checkpoint and logging
    'checkpoint_dir': './checkpoints/mae_pretrain',
    'log_dir': './logs/mae_pretrain',
    'experiment_name': f'mae_pretrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Validation
    'val_split': 0.1,
    'val_batch_size': 32,
}

# Hardware
HARDWARE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'gpu_ids': [0, 1] if torch.cuda.device_count() > 1 else [0],
}


def get_config() -> dict:
    """Get complete configuration."""
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'train': TRAIN_CONFIG,
        'hardware': HARDWARE_CONFIG,
    }