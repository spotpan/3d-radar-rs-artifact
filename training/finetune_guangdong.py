#!/usr/bin/env python3
"""
Precipitation estimation fine-tuning for Guangdong radar data.
Uses 6 frames x 12-min intervals (1 hour radar history).
Data valid rain only at minutes 00 and 30; radar data available at 6-min
resolution for all timesteps via FinetuneDatasetGuangdong.
Data: /path/to/radar_station_dataset/
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
from datetime import datetime
import argparse
from typing import Dict, Tuple

from data.dataloader import FinetuneDatasetGuangdong
from models.mae import MAE3D, MAE3DConfig
from models.rain_decoder import (
    PrecipitationDecoder, RainDecoderConfig,
)
from models.losses import CombinedLoss
from training.trainer import PrecipitationTrainer
from configs.finetune_guangdong_config import get_config, load_station_coords


def load_pretrained_mae(checkpoint_path: str, config: dict) -> MAE3D:
    """Load pretrained MAE model from checkpoint."""
    print(f"Loading pretrained MAE from {checkpoint_path}")

    mae_config = MAE3DConfig(
        in_channels=6,
        img_size=(700, 900),
        patch_size=16,
        encoder_dim=768,
        encoder_depth=16,
        encoder_num_heads=12,
        decoder_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mask_ratio=0.8,
    )

    mae = MAE3D(mae_config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle DataParallel wrapper prefix
    state_dict = checkpoint['model_state_dict']
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    mae.load_state_dict(state_dict)
    print(f"  Loaded MAE from epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss={checkpoint.get('val_loss', '?'):.4f}")

    return mae


def create_model(config: dict) -> PrecipitationDecoder:
    """Create precipitation estimation model with correct num_patches."""
    model_config = config['model']

    # Load pretrained MAE encoder
    mae = load_pretrained_mae(model_config['mae_checkpoint'], config)

    # Build decoder config with correct patch dimensions for (700, 900)
    decoder_config = RainDecoderConfig(
        encoder_dim=768,
        num_patches=(model_config['num_patches_h'],
                     model_config['num_patches_w']),
        patch_size=16,
        num_frames=model_config['num_frames'],
        temporal_dim=model_config['temporal_dim'],
        temporal_depth=model_config['temporal_depth'],
        temporal_num_heads=model_config['temporal_num_heads'],
        decoder_channels=model_config['decoder_channels'],
        output_channels=1,
        use_skip_connections=model_config['use_skip_connections'],
        final_activation_reg=model_config['final_activation_reg'],
        final_activation_cls=model_config['final_activation_cls'],
    )

    decoder = PrecipitationDecoder(decoder_config, mae_encoder=mae)

    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    mae_params = sum(p.numel() for p in mae.parameters())
    mae_trainable = sum(p.numel() for p in mae.parameters() if p.requires_grad)

    print(f"\nModel created:")
    print(f"  MAE encoder parameters: {mae_params:,}")
    print(f"  MAE encoder trainable: {mae_trainable:,} (should be 0)")
    print(f"  Decoder parameters: {total_params:,}")
    print(f"  Decoder trainable: {trainable_params:,}")
    print(f"  Total trainable: {trainable_params:,}")
    print(f"  Sequence: {model_config['num_frames']} frames x 30-min intervals")

    return decoder


def create_data_loaders(config: dict):
    """Create data loaders for fine-tuning."""
    data_config = config['data']
    train_config = config['train']

    dataset = FinetuneDatasetGuangdong(
        data_paths=data_config['data_paths'],
        radar_height_layers=data_config['radar_height_layers'],
        spatial_size=data_config['spatial_size'],
        target_minutes=data_config['target_minutes'],
        history_frames=data_config['history_frames'],
        frame_interval=data_config['frame_interval'],
    )

    val_size = int(len(dataset) * train_config['val_split'])
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    use_weighted_sampler = train_config.get('use_weighted_sampler', False)

    if use_weighted_sampler:
        cache_path = train_config.get('sampler_cache_path', None)
        cache_refresh = train_config.get('sampler_cache_refresh', False)

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if cache_path is not None and os.path.exists(cache_path) and not cache_refresh:
            print(f"\nLoading sample weights from cache: {cache_path}")
            dataset.sample_weights = np.load(cache_path).astype(np.float32)

            if len(dataset.sample_weights) != len(dataset):
                raise ValueError(
                    f"Cached sample weights length {len(dataset.sample_weights)} "
                    f"!= dataset length {len(dataset)}. Set sampler_cache_refresh=True."
                )
        else:
            dataset.compute_sample_weights(
                thresholds=tuple(train_config.get('sampler_thresholds', [0.1, 1.0, 3.5, 7.5])),
                area_rules=train_config.get('sampler_area_rules', None),
                max_weight=train_config.get('sampler_max_weight', 4.0),
                verbose=True,
            )

            if cache_path is not None:
                np.save(cache_path, dataset.sample_weights)
                print(f"Saved sample weights cache to: {cache_path}")

        train_indices = np.asarray(train_dataset.indices, dtype=np.int64)
        train_weights = dataset.sample_weights[train_indices]

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(train_weights, dtype=torch.double),
            num_samples=len(train_weights),
            replacement=True,
        )

        print("\nWeighted sampler enabled:")
        print(f"  train weights min/max/mean: "
              f"{train_weights.min():.4f}/"
              f"{train_weights.max():.4f}/"
              f"{train_weights.mean():.4f}")
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['val_batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        drop_last=False,
    )

    print(f"\nData loaded:")
    print(f"  Total sequences: {len(dataset)}")
    print(f"  Training sequences: {len(train_dataset)}")
    print(f"  Validation sequences: {len(val_dataset)}")
    print(f"  Batch size: {data_config['batch_size']}")
    print(f"  Steps per epoch: {len(train_loader)}")
    print(f"  Sequence: {data_config['history_frames']} frames @ "
          f"{data_config['frame_interval']}-min intervals")

    return train_loader, val_loader, dataset


def create_loss_fn(config: dict):
    """Create loss function."""
    loss_config = config['loss']
    data_config = config['data']

    station_coords = load_station_coords(loss_config['station_coords_path'])

    loss_fn = CombinedLoss(
        station_coords=station_coords,
        img_size=data_config['spatial_size'],
        quantile=loss_config['quantile'],
        rain_threshold=loss_config['rain_threshold'],
        lambda_global=loss_config['lambda_global'],
        lambda_point=loss_config['lambda_point'],
        lambda_cls=loss_config['lambda_cls'],
        lambda_multi_cls=loss_config.get('lambda_multi_cls', 0.3),
        rain_class_thresholds=loss_config.get('rain_class_thresholds', None),
        multi_cls_weights=loss_config.get('multi_cls_weights', None),
    )

    print(f"\nLoss function created:")
    print(f"  Quantile: {loss_config['quantile']}")
    print(f"  Rain threshold: {loss_config['rain_threshold']} mm/h")
    print(f"  Weights: global={loss_config['lambda_global']}, "
          f"point={loss_config['lambda_point']}, "
          f"cls={loss_config['lambda_cls']}, "
          f"multi_cls={loss_config.get('lambda_multi_cls', 0.3)}")
    if station_coords is not None:
        print(f"  Stations: {station_coords.shape[0]}")
    else:
        print(f"  Stations: not available (point loss = 0)")

    return loss_fn


def create_optimizer(model: nn.Module, config: dict, train_loader):
    """Create optimizer and scheduler."""
    train_config = config['train']

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        betas=(0.9, 0.95),
    )

    if train_config['scheduler'] == 'cosine':
        total_steps = train_config['num_epochs'] * len(train_loader)
        warmup_steps = train_config['warmup_epochs'] * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler


class PrecipitationTrainerWithActivationChange(PrecipitationTrainer):
    """Trainer that switches regression activation to softplus mid-training."""

    def __init__(self, softplus_start_epoch: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softplus_start_epoch = softplus_start_epoch

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        if epoch >= self.softplus_start_epoch:
            self.model.set_final_activation('softplus')
            if epoch == self.softplus_start_epoch:
                self.logger.info(
                    f"Switched regression activation to softplus at epoch {epoch+1}"
                )
        return super().train_epoch(epoch)


def main(args):
    """Main training function."""
    config = get_config()

    # Override with command line arguments
    if args.num_epochs:
        config['train']['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['train']['learning_rate'] = args.learning_rate
    if args.checkpoint_dir:
        config['train']['checkpoint_dir'] = args.checkpoint_dir
    if args.experiment_name:
        config['train']['experiment_name'] = args.experiment_name
    if args.mae_checkpoint:
        config['model']['mae_checkpoint'] = args.mae_checkpoint
    if args.max_train_steps is not None:
        config['train']['max_train_steps'] = args.max_train_steps
    if args.max_val_steps is not None:
        config['train']['max_val_steps'] = args.max_val_steps

    # Set device
    device = torch.device(config['hardware']['device'])
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        torch.cuda.set_device(config['hardware']['gpu_ids'][0])

    # Create model
    model = create_model(config)

    # Multi-GPU
    if device.type == 'cuda' and len(config['hardware']['gpu_ids']) > 1:
        print(f"Using {len(config['hardware']['gpu_ids'])} GPUs: "
              f"{config['hardware']['gpu_ids']}")
        model = nn.DataParallel(model, device_ids=config['hardware']['gpu_ids'])

    # Data loaders
    train_loader, val_loader, dataset = create_data_loaders(config)

    # Loss function
    loss_fn = create_loss_fn(config)

    # Optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, config, train_loader)

    # Trainer
    trainer = PrecipitationTrainerWithActivationChange(
        softplus_start_epoch=config['train']['softplus_start_epoch'],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        num_epochs=config['train']['num_epochs'],
        gradient_accumulation_steps=config['train']['gradient_accumulation_steps'],
        max_grad_norm=config['train']['max_grad_norm'],
        max_train_steps=config['train'].get('max_train_steps', None),
        max_val_steps=config['train'].get('max_val_steps', None),
        use_amp=config['train']['use_amp'],
        checkpoint_dir=config['train']['checkpoint_dir'],
        log_dir=config['train']['log_dir'],
        experiment_name=config['train']['experiment_name'],
    )

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint(
            trainer.current_epoch,
            trainer.best_val_loss,
            is_best=False
        )
    finally:
        dataset.close()

    print("Fine-tuning completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precipitation fine-tuning for Guangdong radar data"
    )
    parser.add_argument("--num-epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--mae-checkpoint", type=str,
                        help="Path to pretrained MAE checkpoint")
    parser.add_argument("--resume-from", type=str,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max-train-steps", type=int,
                        help="Maximum number of training batches per epoch for debugging/smoke tests")
    parser.add_argument("--max-val-steps", type=int,
                        help="Maximum number of validation batches per epoch for debugging/smoke tests")
    args = parser.parse_args()
    main(args)
