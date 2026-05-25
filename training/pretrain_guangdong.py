#!/usr/bin/env python3
"""
MAE pretraining for Guangdong radar data.
Data: /path/to/radar_station_dataset/
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from datetime import datetime
import argparse

from data.dataloader import PretrainDataset
from models.mae import MAE3D, MAE3DConfig
from training.trainer import MAETrainer
from configs.pretrain_guangdong_config import get_config


def create_model(config: dict) -> MAE3D:
    """Create MAE model."""
    model_config = config['model']
    mae_config = MAE3DConfig(**model_config)
    model = MAE3D(mae_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Mask ratio: {model_config['mask_ratio']}")

    return model


def create_data_loaders(config: dict):
    """Create data loaders for pretraining."""
    data_config = config['data']
    train_config = config['train']

    dataset = PretrainDataset(
        data_paths=data_config['data_paths'],
        radar_height_layers=data_config['radar_height_layers'],
        spatial_size=data_config['spatial_size'],
    )

    val_size = int(len(dataset) * train_config['val_split'])
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
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

    print(f"Data loaded:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {data_config['batch_size']}")
    print(f"  Training steps per epoch: {len(train_loader)}")

    return train_loader, val_loader, dataset


def create_optimizer(model: nn.Module, config: dict, train_loader: DataLoader):
    """Create optimizer and scheduler."""
    train_config = config['train']

    optimizer = optim.AdamW(
        model.parameters(),
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

    # Set device
    device = torch.device(config['hardware']['device'])
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        torch.cuda.set_device(config['hardware']['gpu_ids'][0])

    # Create model
    model = create_model(config)

    # Multi-GPU training
    if device.type == 'cuda' and len(config['hardware']['gpu_ids']) > 1:
        print(f"Using {len(config['hardware']['gpu_ids'])} GPUs: {config['hardware']['gpu_ids']}")
        model = nn.DataParallel(model, device_ids=config['hardware']['gpu_ids'])

    # Create data loaders
    train_loader, val_loader, dataset = create_data_loaders(config)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, config, train_loader)

    # Create trainer
    trainer = MAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config['train']['num_epochs'],
        gradient_accumulation_steps=config['train']['gradient_accumulation_steps'],
        max_grad_norm=config['train']['max_grad_norm'],
        use_amp=config['train']['use_amp'],
        checkpoint_dir=config['train']['checkpoint_dir'],
        log_dir=config['train']['log_dir'],
        experiment_name=config['train']['experiment_name'],
    )

    # Load checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        trainer.save_checkpoint(
            trainer.current_epoch,
            trainer.best_val_loss,
            is_best=False
        )
    finally:
        dataset.close()

    print("Pretraining completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DMAE pretraining for Guangdong radar data")
    parser.add_argument("--num-epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--resume-from", type=str, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    main(args)
