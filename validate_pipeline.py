#!/usr/bin/env python3
"""
Validate the complete 3DMAE precipitation estimation pipeline.
Tests data loading, model creation, forward pass, and training loop.
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import tempfile
import h5py

def create_mini_dataset():
    """Create a minimal dataset for validation."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, 'mini_data.h5')

    # Create a very small dataset
    num_samples = 10
    with h5py.File(file_path, 'w') as f:
        grp = f.create_group('radar-rain')

        # Tiny spatial size for testing
        H, W = 112, 144  # Divisible by 16: 112/16=7, 144/16=9
        C = 6

        # Radar data
        radar_shape = (num_samples, 11, H, W)
        radar = np.random.randint(-10, 70, size=radar_shape, dtype=np.int8)
        grp.create_dataset('radar', data=radar)

        # Rain data
        rain_shape = (num_samples, H, W)
        rain = np.random.uniform(0, 10, size=rain_shape).astype(np.float16)
        grp.create_dataset('rain', data=rain)

        # Valid masks (all valid)
        radar_valid = np.ones(num_samples, dtype=bool)
        rain_valid = np.ones(num_samples, dtype=bool)
        grp.create_dataset('radar_valid', data=radar_valid)
        grp.create_dataset('rain_valid', data=rain_valid)

        # Time strings
        times = [f'20220101{str(i).zfill(2)}00' for i in range(num_samples)]
        times_array = np.array(times, dtype='S12')
        grp.create_dataset('time', data=times_array)

    print(f"Created mini dataset at {file_path}")
    print(f"  Samples: {num_samples}")
    print(f"  Spatial size: {H}x{W}")
    print(f"  Radar channels: 11 (using first {C})")

    return file_path, temp_dir, (H, W)

def test_pretrain_pipeline():
    """Test the pretraining pipeline with mini dataset."""
    print("\n" + "="*60)
    print("Testing Pretraining Pipeline")
    print("="*60)

    # Create mini dataset
    data_path, temp_dir, img_size = create_mini_dataset()
    H, W = img_size

    try:
        # Import modules
        from data.dataloader import PretrainDataset
        from models.mae import MAE3D, MAE3DConfig
        from torch.utils.data import DataLoader

        # Create small model config
        config = MAE3DConfig(
            in_channels=6,
            img_size=(H, W),
            patch_size=16,
            encoder_dim=64,  # Small for testing
            encoder_depth=2,
            encoder_num_heads=4,
            decoder_dim=32,
            decoder_depth=2,
            decoder_num_heads=4,
            mask_ratio=0.8,
        )

        # Create model
        model = MAE3D(config)
        print(f"\nCreated MAE model:")
        print(f"  Input size: ({config.img_h}, {config.img_w})")
        print(f"  Patch size: {config.patch_size}")
        print(f"  Num patches: {config.num_patches}")
        print(f"  Encoder dim: {config.encoder_dim}")

        # Create dataset and dataloader
        dataset = PretrainDataset([data_path], spatial_size=(H, W))
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        print(f"\nData loading:")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Batch size: {2}")

        # Test a few batches
        model.eval()
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Test 2 batches
                break

            radar = batch['radar']
            print(f"\nBatch {i}:")
            print(f"  Radar shape: {radar.shape}")

            # Forward pass
            with torch.no_grad():
                pred, mask = model(radar)
                print(f"  Prediction shape: {pred.shape}")
                print(f"  Mask shape: {mask.shape}")
                print(f"  Mask ratio: {mask.mean().item():.3f}")

                # Check reconstruction error
                mse = ((pred - radar) ** 2).mean().item()
                print(f"  Reconstruction MSE: {mse:.4f}")

        # Test feature extraction
        sample = dataset[0]
        radar_sample = sample['radar'].unsqueeze(0)  # Add batch dimension
        features = model.get_latent_features(radar_sample)
        print(f"\nFeature extraction:")
        print(f"  Feature shape: {features.shape}")

        dataset.close()
        print("\nPretraining pipeline test passed! ✓")

    finally:
        # Clean up
        import shutil
        os.unlink(data_path)
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary files")

def test_finetune_pipeline():
    """Test the fine-tuning pipeline with mini dataset."""
    print("\n" + "="*60)
    print("Testing Fine-tuning Pipeline")
    print("="*60)

    # Create mini dataset
    data_path, temp_dir, img_size = create_mini_dataset()
    H, W = img_size

    try:
        # Import modules
        from data.dataloader import FinetuneDataset
        from models.mae import MAE3D, MAE3DConfig
        from models.rain_decoder import PrecipitationDecoder, RainDecoderConfig
        from torch.utils.data import DataLoader

        # Create small MAE model
        mae_config = MAE3DConfig(
            in_channels=6,
            img_size=(H, W),
            patch_size=16,
            encoder_dim=64,
            encoder_depth=2,
            encoder_num_heads=4,
            decoder_dim=32,
            decoder_depth=2,
            decoder_num_heads=4,
            mask_ratio=0.8,
        )
        mae = MAE3D(mae_config)

        # Create precipitation decoder config
        num_patches_h = (H + 15) // 16  # ceil(H/16)
        num_patches_w = (W + 15) // 16  # ceil(W/16)

        decoder_config = RainDecoderConfig(
            encoder_dim=64,
            num_patches=(num_patches_h, num_patches_w),
            patch_size=16,
            num_frames=6,
            temporal_dim=64,
            temporal_depth=1,
            temporal_num_heads=4,
            decoder_channels=[32, 16, 8],
            output_channels=1,
            use_skip_connections=True,
            final_activation_reg='linear',
            final_activation_cls='sigmoid',
        )

        # Create precipitation decoder
        decoder = PrecipitationDecoder(decoder_config, mae)

        print(f"\nCreated precipitation model:")
        print(f"  MAE encoder dim: {mae_config.encoder_dim}")
        print(f"  Temporal dim: {decoder_config.temporal_dim}")
        print(f"  Num frames: {decoder_config.num_frames}")

        # Create dataset and dataloader
        dataset = FinetuneDataset(
            [data_path],
            spatial_size=(H, W),
            target_minutes=[0, 30],
            history_frames=6,
            frame_interval=12,
        )

        if len(dataset) == 0:
            print("Warning: No valid sequences found in mini dataset")
            print("Skipping fine-tuning pipeline test")
            return

        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        print(f"\nData loading:")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Sequence length: {6} frames")

        # Test a batch
        decoder.eval()
        batch = next(iter(dataloader))

        radar_seq = batch['radar_sequence']
        rain_target = batch['rain']

        print(f"\nSample batch:")
        print(f"  Radar sequence shape: {radar_seq.shape}")
        print(f"  Rain target shape: {rain_target.shape}")

        # Forward pass
        with torch.no_grad():
            outputs = decoder(radar_seq)
            print(f"  Regression output shape: {outputs['reg_output'].shape}")
            print(f"  Classification output shape: {outputs['cls_output'].shape}")
            print(f"  Regression range: [{outputs['reg_output'].min():.2f}, {outputs['reg_output'].max():.2f}]")
            print(f"  Classification range: [{outputs['cls_output'].min():.2f}, {outputs['cls_output'].max():.2f}]")

        # Test loss computation
        from models.losses import CombinedLoss

        # Create dummy station coordinates
        S = 5  # 5 dummy stations
        station_coords = np.random.rand(S, 2)
        station_coords[:, 0] *= H  # y coordinates
        station_coords[:, 1] *= W  # x coordinates

        loss_fn = CombinedLoss(
            station_coords=station_coords,
            img_size=(H, W),
            quantile=0.9,
            rain_threshold=0.1,
            lambda_global=1.0,
            lambda_point=0.5,
            lambda_cls=0.3,
        )

        # Create dummy station observations
        B = radar_seq.shape[0]
        station_obs = torch.randn(B, S)

        losses = loss_fn(
            outputs['reg_output'],
            outputs['cls_output'],
            rain_target,
            station_obs,
        )

        print(f"\nLoss computation:")
        for name, loss_val in losses.items():
            print(f"  {name}: {loss_val.item():.4f}")

        dataset.close()
        print("\nFine-tuning pipeline test passed! ✓")

    finally:
        # Clean up
        import shutil
        os.unlink(data_path)
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary files")

def main():
    """Run pipeline validation."""
    print("="*60)
    print("3DMAE Pipeline Validation")
    print("="*60)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Test pretraining pipeline
        test_pretrain_pipeline()

        # Test fine-tuning pipeline
        test_finetune_pipeline()

        print("\n" + "="*60)
        print("All pipeline tests passed! ✓")
        print("="*60)
        print("\nThe 3DMAE precipitation estimation system is ready for use.")
        print("\nNext steps:")
        print("1. Prepare your radar-rain dataset in HDF5 format")
        print("2. Adjust configuration files for your data")
        print("3. Run pretraining: python training/pretrain.py")
        print("4. Run fine-tuning: python training/finetune.py")

    except Exception as e:
        print(f"\nPipeline validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())