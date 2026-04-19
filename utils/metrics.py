import torch
import numpy as np
from typing import Tuple, Dict, Optional
import sklearn.metrics as sk_metrics


def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return torch.abs(pred - target).mean().item()


def root_mean_squared_error(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def correlation_coefficient(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Calculate correlation coefficient (CC)."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    if pred.numel() == 0 or target.numel() == 0:
        return 0.0

    # Convert to numpy for correlation calculation
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    # Use numpy's corrcoef
    corr_matrix = np.corrcoef(pred_np, target_np)
    if corr_matrix.shape[0] < 2:
        return 0.0

    return float(corr_matrix[0, 1])


def critical_success_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.1,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate Critical Success Index (CSI) at given threshold.

    CSI = hits / (hits + false_alarms + misses)
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    if pred.numel() == 0:
        return 0.0

    # Create binary masks
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()

    # Calculate contingency table
    hits = (pred_binary * target_binary).sum().item()
    false_alarms = (pred_binary * (1 - target_binary)).sum().item()
    misses = ((1 - pred_binary) * target_binary).sum().item()

    denominator = hits + false_alarms + misses
    if denominator == 0:
        return 0.0

    return hits / denominator


def calculate_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: Tuple[float, ...] = (0.1, 1.0, 5.0, 10.0),
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate all metrics for precipitation estimation.

    Args:
        pred: predicted precipitation (B, H, W) or (B, 1, H, W)
        target: target precipitation (same shape as pred)
        thresholds: precipitation thresholds for CSI
        mask: optional mask for valid pixels

    Returns:
        Dictionary of metrics
    """
    # Ensure tensors are 2D for metric calculation
    if pred.dim() == 4:
        pred = pred.squeeze(1)  # (B, H, W)
    if target.dim() == 4:
        target = target.squeeze(1)  # (B, H, W)

    # Flatten batch and spatial dimensions
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    if mask is not None:
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        mask_flat = mask.flatten()
        pred_flat = pred_flat[mask_flat]
        target_flat = target_flat[mask_flat]

    # Calculate metrics
    metrics = {
        'MAE': mean_absolute_error(pred_flat, target_flat),
        'RMSE': root_mean_squared_error(pred_flat, target_flat),
        'CC': correlation_coefficient(pred_flat, target_flat),
    }

    # Calculate CSI at different thresholds
    for threshold in thresholds:
        csi = critical_success_index(pred_flat, target_flat, threshold)
        metrics[f'CSI_{threshold}'] = csi

    return metrics


def calculate_precipitation_bias(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Calculate precipitation bias metrics.

    Returns:
        Dictionary with bias metrics
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    pred_flat = pred.flatten()
    target_flat = target.flatten()

    if mask is not None:
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        mask_flat = mask.flatten()
        pred_flat = pred_flat[mask_flat]
        target_flat = target_flat[mask_flat]

    # Calculate mean error (bias)
    bias = (pred_flat - target_flat).mean().item()

    # Calculate relative bias
    target_mean = target_flat.mean().item()
    if target_mean > 0:
        relative_bias = bias / target_mean
    else:
        relative_bias = 0.0

    # Calculate bias for different intensity ranges
    intensity_ranges = {
        'light': (0.1, 1.0),
        'moderate': (1.0, 5.0),
        'heavy': (5.0, 10.0),
        'extreme': (10.0, float('inf')),
    }

    range_biases = {}
    for range_name, (low, high) in intensity_ranges.items():
        if high == float('inf'):
            range_mask = (target_flat >= low)
        else:
            range_mask = (target_flat >= low) & (target_flat < high)

        if range_mask.any():
            pred_range = pred_flat[range_mask]
            target_range = target_flat[range_mask]
            range_bias = (pred_range - target_range).mean().item()
            range_biases[f'bias_{range_name}'] = range_bias
        else:
            range_biases[f'bias_{range_name}'] = 0.0

    return {
        'bias': bias,
        'relative_bias': relative_bias,
        **range_biases,
    }


# Test the metrics
if __name__ == "__main__":
    # Create test data
    B, H, W = 2, 100, 100
    pred = torch.randn(B, 1, H, W).abs()  # Positive values
    target = torch.randn(B, 1, H, W).abs()  # Positive values

    print("Test metrics:")
    metrics = calculate_all_metrics(pred, target)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    bias_metrics = calculate_precipitation_bias(pred, target)
    print("\nBias metrics:")
    for name, value in bias_metrics.items():
        print(f"  {name}: {value:.4f}")