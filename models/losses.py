import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
import math


class QuantileLoss(nn.Module):
    """Quantile loss for regression tasks.

    L_q(y, y_hat) = max(q * (y - y_hat), (1 - q) * (y_hat - y))
    """

    def __init__(self, quantile: float = 0.9, reduction: str = 'mean'):
        super().__init__()
        self.quantile = quantile
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, ...) predictions
            target: (B, ...) targets

        Returns:
            loss: scalar or tensor depending on reduction
        """
        errors = target - pred
        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class StationWeightedLoss(nn.Module):
    """Loss that weights predictions around station locations.

    For each station, compute a Gaussian-weighted average of predictions
    in a neighborhood, then compare with station observation.
    """

    def __init__(
        self,
        station_coords: np.ndarray,  # (S, 2) latitude, longitude or (x, y) indices
        img_size: Tuple[int, int] = (700, 900),
        neighborhood_size: int = 5,  # Square neighborhood side length (odd number)
        sigma: float = 1.0,
        quantile: float = 0.9,
    ):
        """
        Args:
            station_coords: (S, 2) coordinates of stations in image coordinates
            img_size: (H, W) size of the image
            neighborhood_size: size of square neighborhood (must be odd)
            sigma: standard deviation for Gaussian weighting
            quantile: quantile for quantile loss
        """
        super().__init__()
        self.img_h, self.img_w = img_size
        self.neighborhood_size = neighborhood_size
        self.sigma = sigma
        self.quantile = quantile
        self.quantile_loss = QuantileLoss(quantile, reduction='none')

        # Convert station coordinates to integer indices if needed
        # Assume coordinates are already in pixel coordinates
        self.station_coords = torch.from_numpy(station_coords.astype(np.float32))

        # Pre-compute Gaussian kernel
        self._precompute_gaussian_kernel()

    def _precompute_gaussian_kernel(self):
        """Pre-compute Gaussian kernel for neighborhood weighting."""
        half_size = self.neighborhood_size // 2
        x = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
        y = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

        # 2D Gaussian
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()  # Normalize to sum to 1

        self.gaussian_kernel = kernel  # (neighborhood_size, neighborhood_size)

    def _get_neighborhood_predictions(
        self,
        pred: torch.Tensor,  # (B, 1, H, W)
        station_idx: int,
    ) -> torch.Tensor:
        """Get Gaussian-weighted predictions around a station."""
        B, _, H, W = pred.shape
        station_y, station_x = self.station_coords[station_idx]

        # Convert to integer coordinates
        center_y = int(round(station_y.item()))
        center_x = int(round(station_x.item()))

        half_size = self.neighborhood_size // 2

        # Calculate bounds (clamp to image boundaries)
        y_start = max(0, center_y - half_size)
        y_end = min(H, center_y + half_size + 1)
        x_start = max(0, center_x - half_size)
        x_end = min(W, center_x + half_size + 1)

        # Extract neighborhood
        neighborhood = pred[:, :, y_start:y_end, x_start:x_end]  # (B, 1, h, w)

        # Extract corresponding part of Gaussian kernel
        ky_start = half_size - (center_y - y_start)
        ky_end = half_size + (y_end - center_y - 1) + 1
        kx_start = half_size - (center_x - x_start)
        kx_end = half_size + (x_end - center_x - 1) + 1

        kernel_part = self.gaussian_kernel[ky_start:ky_end, kx_start:kx_end]
        kernel_part = kernel_part / kernel_part.sum()  # Renormalize for partial kernel

        # Apply weighting
        kernel_part = kernel_part.view(1, 1, kernel_part.shape[0], kernel_part.shape[1])
        kernel_part = kernel_part.to(pred.device)

        weighted_neighborhood = neighborhood * kernel_part
        weighted_sum = weighted_neighborhood.sum(dim=(2, 3))  # (B, 1)

        return weighted_sum.squeeze(1)  # (B,)

    def forward(
        self,
        pred: torch.Tensor,  # (B, 1, H, W)
        target_grid: torch.Tensor,  # (B, 1, H, W) not used here
        station_obs: torch.Tensor,  # (B, S) station observations
    ) -> torch.Tensor:
        """
        Compute station-weighted loss.

        Args:
            pred: predicted precipitation field
            target_grid: grid target (not used, but kept for API consistency)
            station_obs: station observations for each sample

        Returns:
            loss: scalar
        """
        B, _, H, W = pred.shape
        S = self.station_coords.shape[0]

        if station_obs.shape != (B, S):
            raise ValueError(f"station_obs must have shape (B, S)=({B}, {S}), got {station_obs.shape}")

        total_loss = 0.0
        valid_stations = 0

        # For each station
        for s in range(S):
            # Get weighted predictions for this station
            station_preds = self._get_neighborhood_predictions(pred, s)  # (B,)

            # Get station observations for this station
            station_targets = station_obs[:, s]  # (B,)

            # Compute quantile loss
            loss = self.quantile_loss(station_preds, station_targets)  # (B,)

            # Only include valid observations (non-NaN)
            valid_mask = ~torch.isnan(station_targets)
            if valid_mask.any():
                valid_loss = loss[valid_mask].mean()
                total_loss += valid_loss
                valid_stations += 1

        if valid_stations == 0:
            return torch.tensor(0.0, device=pred.device)

        return total_loss / valid_stations


class ClassificationLoss(nn.Module):
    """Binary classification loss for rain/no-rain."""

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        pred_prob: torch.Tensor,  # (B, 1, H, W) rain probabilities
        target_grid: torch.Tensor,  # (B, 1, H, W) precipitation values
    ) -> torch.Tensor:
        """
        Compute binary classification loss.

        Args:
            pred_prob: predicted rain probabilities [0, 1]
            target_grid: precipitation values (mm/h)

        Returns:
            loss: scalar
        """
        # Create binary mask: 1 if precipitation >= threshold
        target_mask = (target_grid >= self.threshold).float()

        # Ensure pred_prob is in [0, 1] (should be from sigmoid)
        pred_prob = torch.clamp(pred_prob, 0, 1)

        # Compute BCE loss
        loss = self.bce_loss(pred_prob, target_mask)

        return loss


class CombinedLoss(nn.Module):
    """Combined loss for precipitation estimation.

    L_total = λ_global * L_global + λ_point * L_point + λ_cls * L_cls
    """

    def __init__(
        self,
        station_coords: Optional[np.ndarray] = None,
        img_size: Tuple[int, int] = (700, 900),
        quantile: float = 0.9,
        rain_threshold: float = 0.1,
        lambda_global: float = 1.0,
        lambda_point: float = 0.5,
        lambda_cls: float = 0.3,
    ):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_point = lambda_point
        self.lambda_cls = lambda_cls

        # Global regression loss (quantile loss on grid)
        self.global_loss = QuantileLoss(quantile)

        # Point (station) loss
        if station_coords is not None:
            self.point_loss = StationWeightedLoss(
                station_coords=station_coords,
                img_size=img_size,
                quantile=quantile,
            )
        else:
            self.point_loss = None

        # Classification loss
        self.cls_loss = ClassificationLoss(threshold=rain_threshold)

    def forward(
        self,
        pred_reg: torch.Tensor,  # (B, 1, H, W) regression predictions
        pred_cls: torch.Tensor,  # (B, 1, H, W) classification probabilities
        target_grid: torch.Tensor,  # (B, 1, H, W) grid targets
        station_obs: Optional[torch.Tensor] = None,  # (B, S) station observations
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred_reg: regression predictions (precipitation values)
            pred_cls: classification predictions (rain probabilities)
            target_grid: grid precipitation targets
            station_obs: station observations (optional)

        Returns:
            dict with individual losses and total loss
        """
        losses = {}

        # Global regression loss
        losses['loss_global'] = self.global_loss(pred_reg, target_grid)

        # Point (station) loss
        if self.point_loss is not None and station_obs is not None:
            losses['loss_point'] = self.point_loss(pred_reg, target_grid, station_obs)
        else:
            losses['loss_point'] = torch.tensor(0.0, device=pred_reg.device)

        # Classification loss
        losses['loss_cls'] = self.cls_loss(pred_cls, target_grid)

        # Weighted total
        losses['loss_total'] = (
            self.lambda_global * losses['loss_global'] +
            self.lambda_point * losses['loss_point'] +
            self.lambda_cls * losses['loss_cls']
        )

        return losses


# Test the losses
if __name__ == "__main__":
    B, H, W = 2, 700, 900

    # Test QuantileLoss
    print("=== Testing QuantileLoss ===")
    quantile_loss = QuantileLoss(quantile=0.9)
    pred = torch.randn(B, 1, H, W)
    target = torch.randn(B, 1, H, W)
    loss = quantile_loss(pred, target)
    print(f"Quantile loss: {loss.item():.4f}")

    # Test StationWeightedLoss (with dummy stations)
    print("\n=== Testing StationWeightedLoss ===")
    # Create dummy station coordinates (10 stations)
    S = 10
    station_coords = np.random.rand(S, 2)
    station_coords[:, 0] *= H  # y coordinates
    station_coords[:, 1] *= W  # x coordinates

    point_loss = StationWeightedLoss(
        station_coords=station_coords,
        img_size=(H, W),
        neighborhood_size=5,
        sigma=1.0,
        quantile=0.9,
    )

    # Dummy station observations
    station_obs = torch.randn(B, S)

    loss_point = point_loss(pred, target, station_obs)
    print(f"Station weighted loss: {loss_point.item():.4f}")

    # Test ClassificationLoss
    print("\n=== Testing ClassificationLoss ===")
    cls_loss = ClassificationLoss(threshold=0.1)
    pred_cls = torch.sigmoid(torch.randn(B, 1, H, W))  # Probabilities
    loss_cls = cls_loss(pred_cls, target)
    print(f"Classification loss: {loss_cls.item():.4f}")

    # Test CombinedLoss
    print("\n=== Testing CombinedLoss ===")
    combined_loss = CombinedLoss(
        station_coords=station_coords,
        img_size=(H, W),
        quantile=0.9,
        rain_threshold=0.1,
        lambda_global=1.0,
        lambda_point=0.5,
        lambda_cls=0.3,
    )

    losses = combined_loss(pred, pred_cls, target, station_obs)
    for name, loss_val in losses.items():
        print(f"{name}: {loss_val.item():.4f}")