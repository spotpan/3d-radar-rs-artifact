import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
import math

from .mae import MAE3D, MAE3DConfig


class RainDecoderConfig:
    """Configuration for precipitation decoder."""

    def __init__(
        self,
        # Input from MAE encoder
        encoder_dim: int = 768,
        num_patches: Tuple[int, int] = (43, 56),  # (H_p, W_p)
        patch_size: int = 16,

        # Temporal fusion
        num_frames: int = 6,
        temporal_dim: int = 768,
        temporal_depth: int = 2,
        temporal_num_heads: int = 8,
        temporal_mlp_ratio: float = 4.0,

        # Decoder
        decoder_channels: List[int] = [512, 256, 128, 64, 32],
        output_channels: int = 1,  # For regression
        use_skip_connections: bool = True,

        # Activation
        final_activation_reg: str = 'linear',  # 'linear', 'softplus', 'relu'
        final_activation_cls: str = 'sigmoid',
    ):
        # MAE encoder output
        self.encoder_dim = encoder_dim
        self.num_patches_h, self.num_patches_w = num_patches
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_size = patch_size

        # Temporal fusion
        self.num_frames = num_frames
        self.temporal_dim = temporal_dim
        self.temporal_depth = temporal_depth
        self.temporal_num_heads = temporal_num_heads
        self.temporal_mlp_ratio = temporal_mlp_ratio

        # Decoder
        self.decoder_channels = decoder_channels
        self.output_channels = output_channels
        self.use_skip_connections = use_skip_connections

        # Activation
        self.final_activation_reg = final_activation_reg
        self.final_activation_cls = final_activation_cls


class TemporalFusion(nn.Module):
    """Temporal fusion module using pointwise temporal Transformer."""

    def __init__(self, config: RainDecoderConfig):
        super().__init__()
        self.config = config

        # Time position encoding
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, config.num_frames, config.temporal_dim)
        )
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

        # Transformer for temporal fusion
        self.temporal_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.temporal_dim,
                nhead=config.temporal_num_heads,
                dim_feedforward=int(config.temporal_dim * config.temporal_mlp_ratio),
                batch_first=True,
                activation='gelu',
            )
            for _ in range(config.temporal_depth)
        ])

        self.norm = nn.LayerNorm(config.temporal_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Fuse temporal features using pointwise temporal attention.

        Args:
            features: (B, T, N, D) where T=num_frames, N=num_patches, D=temporal_dim

        Returns:
            fused: (B, N, D) fused features at target time
        """
        B, T, N, D = features.shape

        # Add time position encoding
        features = features + self.time_pos_embed.unsqueeze(2)  # (B, T, N, D)

        # Reshape for transformer: treat each spatial position independently
        # (B, T, N, D) -> (B*N, T, D)
        features = features.permute(0, 2, 1, 3).contiguous()  # (B, N, T, D)
        features = features.view(B * N, T, D)

        # Apply temporal transformer
        for layer in self.temporal_transformer:
            features = layer(features)

        features = self.norm(features)

        # Get features at target time (last frame)
        # (B*N, T, D) -> take last timestep -> (B*N, D)
        fused = features[:, -1, :]

        # Reshape back: (B*N, D) -> (B, N, D)
        fused = fused.view(B, N, D)

        return fused


class UpsampleBlock(nn.Module):
    """Upsampling block with optional skip connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        use_skip: bool = True,
    ):
        super().__init__()
        self.use_skip = use_skip

        # Upsampling
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        )

        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # Skip connection projection (if needed)
        if use_skip:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)

        # Apply skip connection if provided
        if self.use_skip and skip is not None:
            # Project skip to match channels if needed
            if skip.shape[1] != x.shape[1]:
                skip = self.skip_proj(skip)
            x = x + skip

        # First conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second conv
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class PrecipitationDecoder(nn.Module):
    """Dual-head decoder for precipitation estimation."""

    def __init__(
        self,
        config: RainDecoderConfig,
        mae_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config

        # MAE encoder (frozen)
        if mae_encoder is not None:
            self.mae_encoder = mae_encoder
            # Freeze encoder parameters
            for param in self.mae_encoder.parameters():
                param.requires_grad = False
        else:
            self.mae_encoder = None

        # Projection from MAE encoder to temporal dimension
        self.enc_proj = nn.Linear(config.encoder_dim, config.temporal_dim)

        # Temporal fusion
        self.temporal_fusion = TemporalFusion(config)

        # Initial convolution to reduce channel dimension
        self.init_conv = nn.Conv2d(
            config.temporal_dim,
            config.decoder_channels[0],
            kernel_size=1
        )
        self.init_bn = nn.BatchNorm2d(config.decoder_channels[0])
        self.init_relu = nn.ReLU(inplace=True)

        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        for i in range(len(config.decoder_channels) - 1):
            in_ch = config.decoder_channels[i]
            out_ch = config.decoder_channels[i + 1]
            block = UpsampleBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                scale_factor=2,
                use_skip=config.use_skip_connections,
            )
            self.upsample_blocks.append(block)

        # Final upsampling to original resolution
        final_scale = config.patch_size // (2 ** (len(config.decoder_channels) - 1))
        if final_scale > 1:
            self.final_upsample = nn.Upsample(
                scale_factor=final_scale,
                mode='bilinear',
                align_corners=False
            )
        else:
            self.final_upsample = nn.Identity()

        # Regression head (precipitation intensity)
        self.reg_head = nn.Conv2d(
            config.decoder_channels[-1],
            config.output_channels,
            kernel_size=1
        )

        # Classification head (rain probability)
        self.cls_head = nn.Conv2d(
            config.decoder_channels[-1],
            config.output_channels,
            kernel_size=1
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_frame_features(self, radar_frame: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a single radar frame using MAE encoder.

        Args:
            radar_frame: (B, C, H, W) radar data for one frame

        Returns:
            features: (B, N, D_enc) encoded patches
        """
        if self.mae_encoder is None:
            raise ValueError("MAE encoder not provided")

        # Use MAE encoder to get features
        # Note: MAE3D.get_latent_features returns (B, N, D_enc)
        features = self.mae_encoder.get_latent_features(radar_frame)
        return features

    def forward(
        self,
        radar_sequence: torch.Tensor,  # (B, T, C, H, W)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through precipitation decoder.

        Args:
            radar_sequence: (B, T, C, H, W) sequence of radar frames

        Returns:
            dict containing:
                - reg_output: (B, 1, H, W) precipitation intensity
                - cls_output: (B, 1, H, W) rain probability
        """
        B, T, C, H, W = radar_sequence.shape

        # Extract features for each frame
        frame_features = []
        for t in range(T):
            frame = radar_sequence[:, t, :, :, :]  # (B, C, H, W)
            features = self.extract_frame_features(frame)  # (B, N, D_enc)
            frame_features.append(features)

        # Stack features: (B, T, N, D_enc)
        features = torch.stack(frame_features, dim=1)

        # Project to temporal dimension
        features = self.enc_proj(features)  # (B, T, N, D_temp)

        # Temporal fusion
        fused = self.temporal_fusion(features)  # (B, N, D_temp)

        # Reshape to 2D feature map: (B, N, D_temp) -> (B, D_temp, H_p, W_p)
        fused_2d = fused.view(
            B,
            self.config.num_patches_h,
            self.config.num_patches_w,
            self.config.temporal_dim
        )
        fused_2d = fused_2d.permute(0, 3, 1, 2).contiguous()  # (B, D_temp, H_p, W_p)

        # Initial convolution
        x = self.init_conv(fused_2d)
        x = self.init_bn(x)
        x = self.init_relu(x)

        # Upsampling blocks
        for block in self.upsample_blocks:
            x = block(x)

        # Final upsampling to intermediate resolution
        x = self.final_upsample(x)

        # Apply heads
        reg_output = self.reg_head(x)
        cls_output = self.cls_head(x)

        # Apply final activations
        if self.config.final_activation_reg == 'softplus':
            reg_output = F.softplus(reg_output)
        elif self.config.final_activation_reg == 'relu':
            reg_output = F.relu(reg_output)
        # 'linear' does nothing

        if self.config.final_activation_cls == 'sigmoid':
            cls_output = torch.sigmoid(cls_output)

        # Final upsampling to original resolution if needed
        current_h, current_w = x.shape[2:]
        target_h, target_w = H, W
        if current_h != target_h or current_w != target_w:
            scale_h = target_h / current_h
            scale_w = target_w / current_w
            reg_output = F.interpolate(
                reg_output,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
            cls_output = F.interpolate(
                cls_output,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )

        return {
            'reg_output': reg_output,  # (B, 1, H, W)
            'cls_output': cls_output,  # (B, 1, H, W)
        }

    def set_final_activation(self, activation: str):
        """Set final activation for regression output."""
        self.config.final_activation_reg = activation


def create_precipitation_decoder(
    mae_encoder: nn.Module,
    num_frames: int = 6,
    **kwargs
) -> PrecipitationDecoder:
    """Create precipitation decoder with default parameters."""
    config = RainDecoderConfig(
        encoder_dim=768,
        num_patches=(43, 56),
        patch_size=16,
        num_frames=num_frames,
        temporal_dim=768,
        temporal_depth=2,
        temporal_num_heads=8,
        decoder_channels=[512, 256, 128, 64, 32],
        output_channels=1,
        use_skip_connections=True,
        final_activation_reg='linear',
        final_activation_cls='sigmoid',
        **kwargs
    )

    return PrecipitationDecoder(config, mae_encoder)


# Test the model
if __name__ == "__main__":
    # Create a dummy MAE encoder for testing
    mae_config = MAE3DConfig(
        in_channels=6,
        img_size=(700, 900),
        patch_size=16,
        encoder_dim=768,
        encoder_depth=2,  # Small for testing
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
    )

    # Test with random input
    B, T, C, H, W = 2, 6, 6, 700, 900
    radar_seq = torch.randn(B, T, C, H, W)

    print(f"Input shape: {radar_seq.shape}")
    print(f"Number of frames: {T}")

    # Forward pass
    outputs = decoder(radar_seq)

    print(f"Regression output shape: {outputs['reg_output'].shape}")
    print(f"Classification output shape: {outputs['cls_output'].shape}")
    print(f"Regression min/max: {outputs['reg_output'].min():.2f}/{outputs['reg_output'].max():.2f}")
    print(f"Classification min/max: {outputs['cls_output'].min():.2f}/{outputs['cls_output'].max():.2f}")

    # Check parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nDecoder parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Check MAE encoder is frozen
    mae_trainable = sum(p.numel() for p in mae.parameters() if p.requires_grad)
    print(f"  MAE encoder trainable: {mae_trainable:,} (should be 0)")