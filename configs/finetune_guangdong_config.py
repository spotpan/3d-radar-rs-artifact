"""
Guangdong fine-tuning/evaluation config for 3DMAE precipitation retrieval.

This file is required by:
  - training/finetune_guangdong.py
  - evaluation/evaluate_5class.py
  - evaluation/evaluate_reg_event_patch.py
  - evaluation/compare_physical_rs_uncertainty.py
  - evaluation/rs_patch_distribution_demo.py

Recovered for reproducibility.
"""

def get_config():
    return {
        "model": {
            # MAE checkpoint used in the successful previous evaluation logs.
            "mae_checkpoint": "/path/to/mae_checkpoint.pt",

            # # In the current MAE encoder, one 700x900 radar frame produces 2508 spatial tokens. 
            # The decoder reshapes fused tokens into 44 x 57 x 768.
            "num_patches_h": 44,
            "num_patches_w": 57,

            # Multi-frame temporal module.
            "num_frames": 6,
            "temporal_dim": 768,
            "temporal_depth": 2,
            "temporal_num_heads": 8,

            # Must match the fine-tuned checkpoint head shape:
            # reg_head / cls_head expect 32 input channels.
            "decoder_channels": [512, 256, 128, 64, 32],
            "use_skip_connections": True,

            # Output activations used by the binary/regression-first checkpoint.
            "final_activation_reg": "relu",
            "final_activation_cls": "sigmoid",
        },

        "data": {
            "data_paths": [
                "/path/to/radar_station_dataset/time_radar_rain_2022.h5",
                "/path/to/radar_station_dataset/time_radar_rain_2023.h5",
            ],
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": True,

            # Successful logs show: Sequence: 6 frames @ 12-min intervals.
            "history_frames": 6,
            "frame_interval": 12,

            # IMPORTANT: dataloader checks:
            #   if target_time.minute not in self.target_minutes
            # so this must be iterable, not an int.
            # The data timestamps appear at 6-minute resolution.
            "target_minutes": [0, 6, 12, 18, 24, 30, 36, 42, 48, 54],

            # Radar input shape in previous logs: (6, 6, 700, 900).
            # Must be iterable for dataloader indexing.
            "radar_height_layers": [0, 1, 2, 3, 4, 5],
            "spatial_size": (700, 900),
        },

        "train": {
            "num_epochs": 1,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 0,
            "scheduler": "cosine",
            "val_split": 0.1,
            "val_batch_size": 1,
            "use_weighted_sampler": False,
        },

        "loss": {
            "lambda_global": 1.0,
            "lambda_point": 0.0,
            "lambda_cls": 0.1,
            "lambda_multi_cls": 0.0,
            "rain_threshold": 0.1,
            "quantile": 0.9,
        },
    }


def load_station_coords(*args, **kwargs):
    """
    Compatibility function for training/finetune_guangdong.py.

    Current evaluation / figure scripts do not require explicit station
    coordinates. If future training needs station supervision, this function
    should be replaced by the original station-coordinate loader.
    """
    return None
