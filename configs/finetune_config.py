import torch
from datetime import datetime
import numpy as np

# Data configuration
DATA_CONFIG = {
    'data_paths': [
        '/mnt/md1/guangdong/train/time_radar_rain_2022.h5',
        '/mnt/md1/guangdong/train/time_radar_rain_2023.h5',
        '/mnt/md1/guangdong/train/time_radar_rain_2024.h5',
    ],
    'radar_height_layers': [0, 1, 2, 3, 4, 5],  # First 6 layers
    'spatial_size': (700, 900),

    # Finetune-specific
    'target_minutes': [0, 30],  # Precipitation target minutes
    'history_frames': 6,        # Number of radar frames
    'frame_interval': 12,       # Minutes between frames

    'batch_size': 8,  # From paper (smaller due to sequence)
    'num_workers': 4,
    'pin_memory': True,
    'shuffle': True,
}

# Model configuration
MODEL_CONFIG = {
    # MAE encoder (loaded from checkpoint)
    'mae_checkpoint': './checkpoints/mae_pretrain/best_model.pt',

    # Precipitation decoder
    'num_frames': 6,
    'temporal_dim': 768,
    'temporal_depth': 2,
    'temporal_num_heads': 8,
    'decoder_channels': [512, 256, 128, 64, 32],
    'use_skip_connections': True,

    # Activation (changed in later epochs)
    'final_activation_reg': 'linear',  # Changed to 'softplus' in last 30% epochs
    'final_activation_cls': 'sigmoid',
}

# Loss configuration
LOSS_CONFIG = {
    'quantile': 0.9,
    'rain_threshold': 0.1,  # mm/h

    # Loss weights (from paper)
    'lambda_global': 1.0,
    'lambda_point': 0.5,
    'lambda_cls': 0.3,

    # Station data (optional)
    'station_coords_path': None,  # Path to station coordinates file
    'neighborhood_size': 5,
    'sigma': 1.0,
}

# Training configuration
TRAIN_CONFIG = {
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-3,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'use_amp': True,

    # Learning rate scheduler (cosine annealing)
    'scheduler': 'cosine',
    'warmup_epochs': 5,
    'min_lr': 1e-6,

    # Activation function change
    'softplus_start_epoch': 70,  # Start using softplus in last 30% of epochs

    # Checkpoint and logging
    'checkpoint_dir': './checkpoints/precipitation',
    'log_dir': './logs/precipitation',
    'experiment_name': f'precipitation_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Validation
    'val_split': 0.1,
    'val_batch_size': 4,
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
        'loss': LOSS_CONFIG,
        'train': TRAIN_CONFIG,
        'hardware': HARDWARE_CONFIG,
    }


def load_station_coords(station_coords_path: str) -> np.ndarray:
    """Load station coordinates from file.

    Expected format: (S, 2) array of [latitude, longitude] or [y, x] indices.
    """
    if station_coords_path is None:
        return None

    # Try different file formats
    if station_coords_path.endswith('.npy'):
        coords = np.load(station_coords_path)
    elif station_coords_path.endswith('.csv'):
        coords = np.loadtxt(station_coords_path, delimiter=',')
    else:
        raise ValueError(f"Unknown file format: {station_coords_path}")

    print(f"Loaded {coords.shape[0]} station coordinates")
    return coords