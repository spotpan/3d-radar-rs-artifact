from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from configs.finetune_guangdong_config import get_config
from data.dataloader import FinetuneDatasetGuangdong


def main():
    config = get_config()
    data_config = config["data"]
    train_config = config["train"]

    cache_path = Path(train_config.get(
        "sampler_cache_path",
        "data/cache/guangdong_sampler_weights_013575.npy"
    ))
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = FinetuneDatasetGuangdong(
        data_paths=data_config["data_paths"],
        radar_height_layers=data_config["radar_height_layers"],
        spatial_size=data_config["spatial_size"],
        target_minutes=data_config["target_minutes"],
        history_frames=data_config["history_frames"],
        frame_interval=data_config["frame_interval"],
    )

    weights = dataset.compute_sample_weights(
        thresholds=tuple(train_config.get("sampler_thresholds", [0.1, 1.0, 3.5, 7.5])),
        area_rules=train_config.get("sampler_area_rules", None),
        max_weight=train_config.get("sampler_max_weight", 4.0),
        verbose=True,
    )

    np.save(cache_path, weights)
    print(f"\nSaved sampler cache: {cache_path}")
    print(f"shape={weights.shape}, min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")

    dataset.close()


if __name__ == "__main__":
    main()
