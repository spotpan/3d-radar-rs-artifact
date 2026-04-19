import sys
sys.path.append('.')

from data.dataloader import RadarRainDataset, PretrainDataset, FinetuneDataset
import torch
from torch.utils.data import DataLoader

def test_dataset():
    # Use small subset for testing
    data_path = "/mnt/md1/hxc/guangdong/train_select/time_radar_rain_2022.h5"

    print("=== Testing RadarRainDataset ===")
    dataset = RadarRainDataset([data_path], use_valid_only=True)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Radar shape: {sample['radar'].shape}")
        print(f"Rain shape: {sample['rain'].shape}")
        print(f"Time: {sample['time']}")
        print(f"Radar dtype: {sample['radar'].dtype}")
        print(f"Rain dtype: {sample['rain'].dtype}")

        # Check values
        print(f"Radar min/max: {sample['radar'].min():.2f}/{sample['radar'].max():.2f}")
        print(f"Rain min/max: {sample['rain'].min():.2f}/{sample['rain'].max():.2f}")

    dataset.close()

    print("\n=== Testing PretrainDataset ===")
    pretrain_dataset = PretrainDataset([data_path])
    print(f"Pretrain dataset size: {len(pretrain_dataset)}")

    if len(pretrain_dataset) > 0:
        sample = pretrain_dataset[0]
        print(f"Radar shape: {sample['radar'].shape}")

    pretrain_dataset.close()

    print("\n=== Testing FinetuneDataset ===")
    finetune_dataset = FinetuneDataset([data_path])
    print(f"Finetune dataset size: {len(finetune_dataset)}")

    if len(finetune_dataset) > 0:
        sample = finetune_dataset[0]
        print(f"Radar sequence shape: {sample['radar_sequence'].shape}")
        print(f"Rain shape: {sample['rain'].shape}")
        print(f"Target time: {sample['target_time']}")
        print(f"Number of history frames: {len(sample['history_times'])}")
        print(f"First history time: {sample['history_times'][0]}")

    finetune_dataset.close()

    print("\n=== Testing DataLoader ===")
    dataset = RadarRainDataset([data_path], use_valid_only=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for i, batch in enumerate(dataloader):
        if i >= 2:  # Just check first 2 batches
            break
        print(f"Batch {i}:")
        print(f"  Radar batch shape: {batch['radar'].shape}")
        print(f"  Rain batch shape: {batch['rain'].shape}")
        print(f"  Times: {batch['time'][:2]}...")

    dataset.close()

if __name__ == "__main__":
    test_dataset()