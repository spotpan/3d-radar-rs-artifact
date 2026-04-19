import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math


class MAE3DConfig:
    """Configuration for 3DMAE model."""

    def __init__(
        self,
        # Input dimensions
        in_channels: int = 6,           # Number of height layers
        img_size: Tuple[int, int] = (700, 900),
        patch_size: int = 16,

        # Encoder
        encoder_dim: int = 768,
        encoder_depth: int = 16,
        encoder_num_heads: int = 12,
        encoder_mlp_ratio: float = 4.0,

        # Decoder
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        decoder_mlp_ratio: float = 4.0,

        # Masking
        mask_ratio: float = 0.8,

        # Other
        norm_layer: nn.Module = nn.LayerNorm,
        activation: nn.Module = nn.GELU,
    ):
        self.in_channels = in_channels
        self.img_h, self.img_w = img_size
        self.patch_size = patch_size

        # Calculate patch grid (use ceil to handle non-divisible dimensions)
        self.num_patches_h = (self.img_h + patch_size - 1) // patch_size
        self.num_patches_w = (self.img_w + patch_size - 1) // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Calculate patch dimension
        self.patch_dim = in_channels * patch_size * patch_size

        # Encoder
        self.encoder_dim = encoder_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.encoder_mlp_ratio = encoder_mlp_ratio

        # Decoder
        self.decoder_dim = decoder_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.decoder_mlp_ratio = decoder_mlp_ratio

        # Masking
        self.mask_ratio = mask_ratio

        # Other
        self.norm_layer = norm_layer
        self.activation = activation


class PatchEmbed3D(nn.Module):
    """Split 3D radar data into patches and embed them."""

    def __init__(self, config: MAE3DConfig):
        super().__init__()
        self.config = config

        # Project patches to encoder dimension
        self.proj = nn.Linear(config.patch_dim, config.encoder_dim)

        # Learnable position embeddings for patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches, config.encoder_dim)
        )

        # Learnable class token (for global representation)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.encoder_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        Args:
            x: (B, C, H, W) where C=in_channels (height layers), H=img_h, W=img_w

        Returns:
            x_embed: (B, N+1, D) where N=num_patches, D=encoder_dim
            padding: (pad_top, pad_bottom, pad_left, pad_right) - all zeros in our case
        """
        B, C, H, W = x.shape
        patch_size = self.config.patch_size

        # Pad if necessary to make divisible by patch_size
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # Calculate actual padded dimensions
        H_padded = H + pad_h
        W_padded = W + pad_w
        num_patches_h = H_padded // patch_size
        num_patches_w = W_padded // patch_size
        num_patches = num_patches_h * num_patches_w

        # Reshape to patches: (B, C, H_padded, W_padded) -> (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        x = x.view(B, C,
                   num_patches_h, patch_size,
                   num_patches_w, patch_size)

        # Permute and flatten: -> (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()

        # Flatten patches: -> (B, num_patches, C * patch_size * patch_size)
        x = x.view(B, num_patches, -1)

        # Project to embedding dimension
        x = self.proj(x)  # (B, N, D)

        # Add position embeddings (use only first num_patches positions)
        x = x + self.pos_embed[:, :num_patches, :]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Return padding info (we only pad at bottom and right)
        padding = (0, pad_h, 0, pad_w)  # (pad_top, pad_bottom, pad_left, pad_right)

        return x, padding


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            activation(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with pre-norm
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks."""

    def __init__(self, config: MAE3DConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.encoder_dim,
                num_heads=config.encoder_num_heads,
                mlp_ratio=config.encoder_mlp_ratio,
                norm_layer=config.norm_layer,
                activation=config.activation,
            )
            for _ in range(config.encoder_depth)
        ])
        self.norm = config.norm_layer(config.encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """Stack of Transformer blocks for decoding."""

    def __init__(self, config: MAE3DConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.decoder_dim,
                num_heads=config.decoder_num_heads,
                mlp_ratio=config.decoder_mlp_ratio,
                norm_layer=config.norm_layer,
                activation=config.activation,
            )
            for _ in range(config.decoder_depth)
        ])
        self.norm = config.norm_layer(config.decoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class MAE3D(nn.Module):
    """3D Masked Autoencoder for radar data."""

    def __init__(self, config: MAE3DConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbed3D(config)

        # Encoder
        self.encoder = TransformerEncoder(config)

        # Projection from encoder to decoder dimension
        self.enc_to_dec = nn.Linear(config.encoder_dim, config.decoder_dim)

        # Mask token (learnable token for masked patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder position embeddings (no class token)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches, config.decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Decoder
        self.decoder = TransformerDecoder(config)

        # Prediction head: reconstruct pixels in each patch
        self.pred_head = nn.Linear(config.decoder_dim, config.patch_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches.

        Args:
            x: (B, N, D) patch embeddings (without class token)

        Returns:
            x_masked: (B, N_keep, D) visible patches
            mask: (B, N) binary mask where 1 indicates masked
            ids_restore: (B, N) indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.config.mask_ratio))

        # Generate random noise for each patch
        noise = torch.rand(B, N, device=x.device)  # (B, N)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # (B, N) ascending
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # (B, N) to restore

        # Keep the first len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]

        # Gather visible patches
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Generate binary mask: 0 is keep, 1 is mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get mask in original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]:
        """
        Forward pass through encoder.

        Args:
            x: (B, C, H, W) input radar data

        Returns:
            latent: (B, N_keep, D_enc) encoded visible patches
            mask: (B, N) binary mask
            ids_restore: (B, N) indices to restore original order
            padding: (pad_top, pad_bottom, pad_left, pad_right)
        """
        # Patch embedding
        x, padding = self.patch_embed(x)  # (B, N+1, D_enc)

        # Separate class token and patches
        cls_token = x[:, :1, :]
        patches = x[:, 1:, :]  # (B, N, D_enc)

        # Random masking
        patches_visible, mask, ids_restore = self.random_masking(patches)

        # Add class token back
        x_encoder = torch.cat([cls_token, patches_visible], dim=1)

        # Apply encoder
        latent = self.encoder(x_encoder)  # (B, N_keep+1, D_enc)

        # Remove class token for decoder
        latent = latent[:, 1:, :]  # (B, N_keep, D_enc)

        return latent, mask, ids_restore, padding

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            latent: (B, N_keep, D_enc) encoded visible patches
            ids_restore: (B, N) indices to restore original order

        Returns:
            pred: (B, N, patch_dim) reconstructed patches
        """
        B = latent.shape[0]
        N = self.config.num_patches
        D_dec = self.config.decoder_dim

        # Project to decoder dimension
        x = self.enc_to_dec(latent)  # (B, N_keep, D_dec)

        # Append mask tokens to the sequence
        mask_tokens = self.mask_token.repeat(B, N - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)  # (B, N, D_dec)

        # Unshuffle to original order
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D_dec))

        # Add decoder position embeddings
        x = x + self.decoder_pos_embed

        # Apply decoder
        x = self.decoder(x)

        # Prediction head
        pred = self.pred_head(x)  # (B, N, patch_dim)

        return pred

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through entire MAE.

        Args:
            x: (B, C, H, W) input radar data

        Returns:
            pred: (B, C, H, W) reconstructed radar data
            mask: (B, N) binary mask
        """
        # Encoder
        latent, mask, ids_restore, padding = self.forward_encoder(x)
        pad_top, pad_bottom, pad_left, pad_right = padding

        # Decoder
        pred_patches = self.forward_decoder(latent, ids_restore)

        # Reshape patches back to image
        B, N, patch_dim = pred_patches.shape
        C = self.config.in_channels
        patch_size = self.config.patch_size

        # Calculate actual patch grid dimensions (after padding)
        H_padded = self.config.img_h + pad_bottom  # pad_bottom is total pad in height
        W_padded = self.config.img_w + pad_right   # pad_right is total pad in width
        num_patches_h = H_padded // patch_size
        num_patches_w = W_padded // patch_size

        # Reshape: (B, N, C * patch_size * patch_size) -> (B, N, C, patch_size, patch_size)
        pred_patches = pred_patches.view(B, N, C, patch_size, patch_size)

        # Reshape to image: (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
        pred = pred_patches.view(B,
                                num_patches_h,
                                num_patches_w,
                                C,
                                patch_size,
                                patch_size)

        # Permute: -> (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        pred = pred.permute(0, 3, 1, 4, 2, 5).contiguous()

        # Reshape: -> (B, C, H_padded, W_padded)
        pred = pred.view(B, C, H_padded, W_padded)

        # Remove padding
        if pad_bottom > 0 or pad_right > 0:
            pred = pred[:, :, :self.config.img_h, :self.config.img_w]

        return pred, mask

    def get_latent_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent features from encoder (for downstream tasks).

        Args:
            x: (B, C, H, W) input radar data

        Returns:
            features: (B, N, D_enc) encoded patches (without class token)
        """
        # Patch embedding
        x_embed, _ = self.patch_embed(x)  # (B, N+1, D_enc)

        # Separate class token and patches
        cls_token = x_embed[:, :1, :]
        patches = x_embed[:, 1:, :]  # (B, N, D_enc)

        # No masking for feature extraction
        x_encoder = torch.cat([cls_token, patches], dim=1)

        # Apply encoder
        latent = self.encoder(x_encoder)  # (B, N+1, D_enc)

        # Return patches only (no class token)
        return latent[:, 1:, :]


def mae3d_base(**kwargs) -> MAE3D:
    """Create base 3DMAE model with default parameters from paper."""
    config = MAE3DConfig(
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
    return MAE3D(config)


# Test the model
if __name__ == "__main__":
    # Create config and model
    config = MAE3DConfig(
        in_channels=6,
        img_size=(700, 900),
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
    B, C, H, W = 2, 6, 700, 900
    x = torch.randn(B, C, H, W)

    print(f"Input shape: {x.shape}")
    print(f"Number of patches: {config.num_patches}")
    print(f"Patch dimension: {config.patch_dim}")

    # Forward pass
    pred, mask = model(x)

    print(f"Output shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio (actual): {mask.mean().item():.3f}")

    # Test feature extraction
    features = model.get_latent_features(x)
    print(f"Feature shape: {features.shape}")

    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")