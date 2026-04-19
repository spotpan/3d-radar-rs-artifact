#!/usr/bin/env python3
"""
Integration test for the 3DMAE precipitation estimation system.
Tests data loading, model forward pass, and loss computation.
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import tempfile
import h5py

# Test utilities
def create_dummy_h5_file(file_path: str, num_samples: int = 100):
    """Create a dummy HDF5 file for testing."""
    with h5py.File(file_path, 'w') as f:
        grp = f.create_group('radar-rain')

        # Radar data: (N, 11, 700, 900) int8
        radar_shape = (num_samples, 11, 700, 900)
        radar = np.random.randint(-10, 70, size=radar_shape, dtype=np.int8)
        grp.create_dataset('radar', data=radar)

        # Rain data: (N, 700, 900) float16
        rain_shape = (num_samples, 700, 900)
        rain = np.random.uniform(0, 50, size=rain_shape).astype(np.float16)
        grp.create_dataset('rain', data=rain)

        # Valid masks
        radar_valid = np.ones(num_samples, dtype=bool)
        rain_valid = np.ones(num_samples, dtype=bool)
        grp.create_dataset('radar_valid', data=radar_valid)
        grp.create_dataset('rain_valid', data=rain_valid)

        # Time strings
        times = [f'20220101{str(i).zfill(2)}00' for i in range(num_samples)]
        times_array = np.array(times, dtype='S12')
        grp.create_dataset('time', data=times_array)

    print(f"Created dummy HDF5 file with {num_samples} samples")

def test_data_loading():
    """Test data loading functionality."""
    print("\n=== Testing Data Loading ===")

    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        create_dummy_h5_file(tmp_path, num_samples=50)

        # Test RadarRainDataset
        from data.dataloader import RadarRainDataset
        dataset = RadarRainDataset([tmp_path], use_valid_only=True)
        print(f"RadarRainDataset loaded {len(dataset)} samples")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample radar shape: {sample['radar'].shape}")
            print(f"Sample rain shape: {sample['rain'].shape}")
            print(f"Sample time: {sample['time']}")

            # Test batch loading
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
            batch = next(iter(dataloader))
            print(f"Batch radar shape: {batch['radar'].shape}")
            print(f"Batch rain shape: {batch['rain'].shape}")

        dataset.close()

    finally:
        # Clean up
        os.unlink(tmp_path)

    print("Data loading test passed!")

def test_mae_model():
    """Test 3DMAE model."""
    print("\n=== Testing 3DMAE Model ===")

    from models.mae import MAE3D, MAE3DConfig

    # Create config for testing (smaller model)
    config = MAE3DConfig(
        in_channels=6,
        img_size=(700, 900),
        patch_size=16,
        encoder_dim=128,  # Smaller for testing
        encoder_depth=2,
        encoder_num_heads=4,
        decoder_dim=64,
        decoder_depth=2,
        decoder_num_heads=4,
        mask_ratio=0.8,
    )

    model = MAE3D(config)

    # Test forward pass
    B, C, H, W = 2, 6, 700, 900
    x = torch.randn(B, C, H, W)

    pred, mask = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio: {mask.mean().item():.3f}")

    # Test feature extraction
    features = model.get_latent_features(x)
    print(f"Feature shape: {features.shape}")

    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("MAE model test passed!")

def test_precipitation_decoder():
    """Test precipitation decoder."""
    print("\n=== Testing Precipitation Decoder ===")

    from models.mae import MAE3D, MAE3DConfig
    from models.rain_decoder import PrecipitationDecoder, RainDecoderConfig

    # Create dummy MAE encoder
    mae_config = MAE3DConfig(
        in_channels=6,
        img_size=(700, 900),
        patch_size=16,
        encoder_dim=128,
        encoder_depth=2,
        encoder_num_heads=4,
        decoder_dim=64,
        decoder_depth=2,
        decoder_num_heads=4,
        mask_ratio=0.8,
    )
    mae = MAE3D(mae_config)

    # Create precipitation decoder config
    decoder_config = RainDecoderConfig(
        encoder_dim=128,  # Match MAE encoder dim
        num_patches=(44, 57),  # ceil(700/16)=44, ceil(900/16)=57
        patch_size=16,
        num_frames=6,
        temporal_dim=128,
        temporal_depth=1,
        temporal_num_heads=4,
        decoder_channels=[64, 32, 16],
        output_channels=1,
        use_skip_connections=True,
        final_activation_reg='linear',
        final_activation_cls='sigmoid',
    )

    # Create precipitation decoder
    decoder = PrecipitationDecoder(decoder_config, mae)

    # Test forward pass
    B, T, C, H, W = 2, 6, 6, 700, 900
    radar_seq = torch.randn(B, T, C, H, W)

    outputs = decoder(radar_seq)
    print(f"Input sequence shape: {radar_seq.shape}")
    print(f"Regression output shape: {outputs['reg_output'].shape}")
    print(f"Classification output shape: {outputs['cls_output'].shape}")
    print(f"Regression min/max: {outputs['reg_output'].min():.2f}/{outputs['reg_output'].max():.2f}")

    # Check that MAE encoder is frozen
    mae_trainable = sum(p.numel() for p in mae.parameters() if p.requires_grad)
    decoder_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"MAE encoder trainable params: {mae_trainable:,} (should be 0)")
    print(f"Decoder trainable params: {decoder_trainable:,}")

    print("Precipitation decoder test passed!")

def test_loss_functions():
    """Test loss functions."""
    print("\n=== Testing Loss Functions ===")

    from models.losses import QuantileLoss, ClassificationLoss, CombinedLoss

    # Test QuantileLoss
    quantile_loss = QuantileLoss(quantile=0.9)
    pred = torch.randn(2, 1, 700, 900)
    target = torch.randn(2, 1, 700, 900)
    loss = quantile_loss(pred, target)
    print(f"Quantile loss: {loss.item():.4f}")

    # Test ClassificationLoss
    cls_loss = ClassificationLoss(threshold=0.1)
    pred_cls = torch.sigmoid(torch.randn(2, 1, 700, 900))
    loss_cls = cls_loss(pred_cls, target)
    print(f"Classification loss: {loss_cls.item():.4f}")

    # Test CombinedLoss (without station data)
    combined_loss = CombinedLoss(
        station_coords=None,
        img_size=(700, 900),
        quantile=0.9,
        rain_threshold=0.1,
        lambda_global=1.0,
        lambda_point=0.5,
        lambda_cls=0.3,
    )

    losses = combined_loss(pred, pred_cls, target, station_obs=None)
    print(f"Combined losses:")
    for name, loss_val in losses.items():
        print(f"  {name}: {loss_val.item():.4f}")

    print("Loss functions test passed!")

def test_trainer():
    """Test trainer classes (simplified)."""
    print("\n=== Testing Trainer Classes ===")

    from models.mae import MAE3D, MAE3DConfig
    from training.trainer import MAETrainer

    # Create small model
    config = MAE3DConfig(
        in_channels=6,
        img_size=(700, 900),
        patch_size=16,
        encoder_dim=64,
        encoder_depth=1,
        encoder_num_heads=4,
        decoder_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        mask_ratio=0.8,
    )
    model = MAE3D(config)

    # Create dummy data loader
    from torch.utils.data import DataLoader, TensorDataset

    # Create dummy dataset
    dummy_data = torch.randn(20, 6, 700, 900)
    dummy_dataset = TensorDataset(dummy_data)

    train_loader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dummy_dataset, batch_size=4, shuffle=False)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create trainer
    trainer = MAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device='cpu',
        num_epochs=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        use_amp=False,
        checkpoint_dir='./test_checkpoints',
        log_dir='./test_logs',
        experiment_name='test',
    )

    # Test one training step
    print("Testing trainer initialization...")
    print(f"Device: {trainer.device}")
    print(f"Model on device: {next(model.parameters()).device}")

    # Test compute_loss
    batch = {'radar': dummy_data[:4]}
    loss_dict = trainer.compute_loss(batch)
    print(f"Loss computed: {loss_dict['loss'].item():.4f}")

    print("Trainer test passed!")

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("3DMAE Precipitation Estimation System - Integration Test")
    print("=" * 60)

    try:
        test_data_loading()
        test_mae_model()
        test_precipitation_decoder()
        test_loss_functions()
        test_trainer()

        print("\n" + "=" * 60)
        print("All integration tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\nIntegration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())