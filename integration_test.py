#!/usr/bin/env python3
"""
Integration test for the 3DMAE precipitation estimation system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np

from data.dataloader import PretrainDataset, FinetuneDataset
from models.mae import MAE3D, MAE3DConfig
from models.rain_decoder import create_precipitation_decoder
from models.losses import QuantileLoss, CombinedLoss
from utils.metrics import calculate_all_metrics


def test_data_loading():
    """Test data loading functionality."""
    print("=== Testing Data Loading ===")

    # Use a small test file or mock data
    # For now, just test if the dataset classes can be instantiated
    try:
        # Test PretrainDataset
        print("Testing PretrainDataset...")
        pretrain_dataset = PretrainDataset(
            data_paths=['/mnt/md1/hxc/guangdong/train_select/time_radar_rain_2022.h5'],
            radar_height_layers=[0, 1, 2, 3, 4, 5],
            spatial_size=(700, 900),
        )
        print(f"  Pretrain dataset size: {len(pretrain_dataset)}")

        if len(pretrain_dataset) > 0:
            sample = pretrain_dataset[0]
            print(f"  Sample radar shape: {sample['radar'].shape}")
            print(f"  Sample time: {sample['time']}")

        pretrain_dataset.close()

        # Test FinetuneDataset
        print("\nTesting FinetuneDataset...")
        finetune_dataset = FinetuneDataset(
            data_paths=['/mnt/md1/hxc/guangdong/train_select/time_radar_rain_2022.h5'],
            radar_height_layers=[0, 1, 2, 3, 4, 5],
            spatial_size=(700, 900),
            target_minutes=[0, 30],
            history_frames=6,
            frame_interval=12,
        )
        print(f"  Finetune dataset size: {len(finetune_dataset)}")

        if len(finetune_dataset) > 0:
            sample = finetune_dataset[0]
            print(f"  Radar sequence shape: {sample['radar_sequence'].shape}")
            print(f"  Rain shape: {sample['rain'].shape}")
            print(f"  Target time: {sample['target_time']}")

        finetune_dataset.close()

        return True

    except Exception as e:
        print(f"Error in data loading test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mae_model():
    """Test MAE model forward pass."""
    print("\n=== Testing MAE Model ===")

    try:
        # Create config with smaller dimensions for testing
        config = MAE3DConfig(
            in_channels=6,
            img_size=(704, 896),  # Divisible by 16 for testing
            patch_size=16,
            encoder_dim=768,
            encoder_depth=2,  # Smaller for testing
            encoder_num_heads=12,
            decoder_dim=512,
            decoder_depth=2,  # Smaller for testing
            decoder_num_heads=8,
            mask_ratio=0.8,
        )

        model = MAE3D(config)

        # Test with random input
        B, C, H, W = 2, 6, 704, 896
        x = torch.randn(B, C, H, W)

        print(f"Input shape: {x.shape}")
        print(f"Number of patches: {config.num_patches}")

        # Forward pass
        pred, mask = model(x)

        print(f"Output shape: {pred.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask ratio: {mask.mean().item():.3f}")

        # Test feature extraction
        features = model.get_latent_features(x)
        print(f"Feature shape: {features.shape}")

        return True

    except Exception as e:
        print(f"Error in MAE model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_precipitation_decoder():
    """Test precipitation decoder forward pass."""
    print("\n=== Testing Precipitation Decoder ===")

    try:
        # Create MAE model (small for testing)
        mae_config = MAE3DConfig(
            in_channels=6,
            img_size=(704, 896),  # Divisible by 16
            patch_size=16,
            encoder_dim=768,
            encoder_depth=2,
            encoder_num_heads=12,
            decoder_dim=512,
            decoder_depth=2,
            decoder_num_heads=8,
            mask_ratio=0.8,
        )
        mae = MAE3D(mae_config)

        # Create precipitation decoder
        decoder = create_precipitation_decoder(
            mae_encoder=mae,
            num_frames=6,
            temporal_dim=768,
            temporal_depth=2,
            temporal_num_heads=8,
            decoder_channels=[512, 256, 128, 64, 32],
        )

        # Test with random input
        B, T, C, H, W = 2, 6, 6, 704, 896
        radar_seq = torch.randn(B, T, C, H, W)

        print(f"Input shape: {radar_seq.shape}")

        # Forward pass
        outputs = decoder(radar_seq)

        print(f"Regression output shape: {outputs['reg_output'].shape}")
        print(f"Classification output shape: {outputs['cls_output'].shape}")
        print(f"Regression range: [{outputs['reg_output'].min():.2f}, {outputs['reg_output'].max():.2f}]")
        print(f"Classification range: [{outputs['cls_output'].min():.2f}, {outputs['cls_output'].max():.2f}]")

        return True

    except Exception as e:
        print(f"Error in precipitation decoder test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test loss functions."""
    print("\n=== Testing Loss Functions ===")

    try:
        B, H, W = 2, 704, 896

        # Test QuantileLoss
        print("Testing QuantileLoss...")
        quantile_loss = QuantileLoss(quantile=0.9)
        pred = torch.randn(B, 1, H, W)
        target = torch.randn(B, 1, H, W)
        loss = quantile_loss(pred, target)
        print(f"  Quantile loss: {loss.item():.4f}")

        # Test CombinedLoss (without station data)
        print("\nTesting CombinedLoss...")
        combined_loss = CombinedLoss(
            station_coords=None,
            img_size=(H, W),
            quantile=0.9,
            rain_threshold=0.1,
            lambda_global=1.0,
            lambda_point=0.5,
            lambda_cls=0.3,
        )

        pred_reg = torch.randn(B, 1, H, W)
        pred_cls = torch.sigmoid(torch.randn(B, 1, H, W))
        target_grid = torch.randn(B, 1, H, W).abs()  # Positive precipitation

        losses = combined_loss(pred_reg, pred_cls, target_grid)
        for name, loss_val in losses.items():
            print(f"  {name}: {loss_val.item():.4f}")

        return True

    except Exception as e:
        print(f"Error in loss functions test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test evaluation metrics."""
    print("\n=== Testing Metrics ===")

    try:
        B, H, W = 2, 100, 100
        pred = torch.randn(B, 1, H, W).abs()
        target = torch.randn(B, 1, H, W).abs()

        metrics = calculate_all_metrics(pred, target)
        print("Metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        return True

    except Exception as e:
        print(f"Error in metrics test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("Running integration tests for 3DMAE precipitation estimation system...")
    print("=" * 70)

    # Run tests
    tests = [
        ("Data Loading", test_data_loading),
        ("MAE Model", test_mae_model),
        ("Precipitation Decoder", test_precipitation_decoder),
        ("Loss Functions", test_loss_functions),
        ("Metrics", test_metrics),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{' ' + test_name + ' ':-^70}")
        success = test_func()
        results.append((test_name, success))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY:")
    print("=" * 70)

    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:30} {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\nAll tests passed! ✅")
    else:
        print("\nSome tests failed! ❌")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)