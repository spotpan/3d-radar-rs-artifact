#!/usr/bin/env python3
"""
Model evaluation tool for precipitation estimation.

This script evaluates model performance using the metrics from the paper:
- ME (Mean Error/Bias)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- CC (Correlation Coefficient)
- CSI (Critical Success Index) at thresholds: 0.1, 1.0, 5.0, 10.0 mm/h
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
import json
from datetime import datetime
import warnings

from utils.metrics import calculate_all_metrics, calculate_precipitation_bias


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    thresholds: Tuple[float, ...] = (0.1, 1.0, 5.0, 10.0)
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for precipitation estimation.

    Args:
        predictions: (B, H, W) or (B, 1, H, W) predicted precipitation
        targets: (B, H, W) or (B, 1, H, W) target precipitation
        mask: optional mask for valid pixels
        thresholds: precipitation thresholds for CSI

    Returns:
        Dictionary of metrics
    """
    # Ensure tensors are 2D for metric calculation
    if predictions.dim() == 4:
        predictions = predictions.squeeze(1)  # (B, H, W)
    if targets.dim() == 4:
        targets = targets.squeeze(1)  # (B, H, W)

    # Flatten batch and spatial dimensions
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    if mask is not None:
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        mask_flat = mask.flatten()
        pred_flat = pred_flat[mask_flat]
        target_flat = target_flat[mask_flat]

    # Calculate basic metrics
    metrics = calculate_all_metrics(predictions, targets, thresholds, mask)

    # Calculate bias metrics
    bias_metrics = calculate_precipitation_bias(predictions, targets, mask)
    metrics.update(bias_metrics)

    # Calculate ME (mean error)
    me = (pred_flat - target_flat).mean().item()
    metrics['ME'] = me

    return metrics


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    prediction_fn: Optional[Callable] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a model on a test dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader for test data
        device: torch device
        prediction_fn: Optional function to extract predictions from model output.
                     If None, assumes model returns a dict with 'reg_output' key.
        verbose: Whether to print progress

    Returns:
        Dictionary of metrics averaged over the test set
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_masks = []

    if verbose:
        print(f"Evaluating model on {len(data_loader)} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if verbose and batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(data_loader)}")

            # Move data to device
            if isinstance(batch, dict):
                # Assuming batch contains 'radar_sequence' and 'rain'
                radar_seq = batch['radar_sequence'].to(device)
                targets = batch['rain'].to(device)
                masks = batch.get('mask', None)
                if masks is not None:
                    masks = masks.to(device)
            else:
                # Assume batch is a tuple (inputs, targets)
                inputs, targets = batch
                radar_seq = inputs.to(device)
                targets = targets.to(device)
                masks = None

            # Forward pass
            if prediction_fn is not None:
                outputs = prediction_fn(model, radar_seq)
            else:
                outputs = model(radar_seq)
                if isinstance(outputs, dict):
                    predictions = outputs['reg_output']
                else:
                    predictions = outputs

            # Store predictions and targets
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            if masks is not None:
                all_masks.append(masks.cpu())

    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    if all_masks:
        masks = torch.cat(all_masks, dim=0)
    else:
        masks = None

    if verbose:
        print(f"Total samples evaluated: {predictions.shape[0]}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {targets.shape}")

    # Compute metrics
    metrics = compute_metrics(predictions, targets, masks)

    if verbose:
        print("\nEvaluation Results:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    return metrics


class ModelEvaluator:
    """Class for evaluating multiple models and generating comparison tables."""

    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Store results for each model
        self.results = {}
        self.model_names = []

    def evaluate(
        self,
        model_name: str,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        prediction_fn: Optional[Callable] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate a single model and store results."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating model: {model_name}")
            print(f"{'='*60}")

        metrics = evaluate_model(model, data_loader, device, prediction_fn, verbose)

        # Store results
        self.results[model_name] = metrics
        if model_name not in self.model_names:
            self.model_names.append(model_name)

        # Save individual model results
        self._save_model_results(model_name, metrics)

        return metrics

    def compare_models(self, save_to_file: bool = True) -> pd.DataFrame:
        """Compare all evaluated models and generate a results table."""
        if not self.results:
            warnings.warn("No results to compare. Evaluate models first.")
            return pd.DataFrame()

        # Create DataFrame
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())

        # Sort metrics for consistent output
        metric_order = ['ME', 'MAE', 'RMSE', 'CC',
                       'CSI_0.1', 'CSI_1.0', 'CSI_5.0', 'CSI_10.0',
                       'bias', 'relative_bias',
                       'bias_light', 'bias_moderate', 'bias_heavy', 'bias_extreme']

        # Filter to only include metrics that exist
        metric_order = [m for m in metric_order if m in all_metrics]

        # Create comparison table
        comparison_data = {}
        for model_name in self.model_names:
            metrics = self.results[model_name]
            row = {}
            for metric in metric_order:
                if metric in metrics:
                    row[metric] = metrics[metric]
                else:
                    row[metric] = np.nan
            comparison_data[model_name] = row

        df = pd.DataFrame(comparison_data).T
        df = df[metric_order]  # Reorder columns

        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(self.output_dir, f"model_comparison_{timestamp}.csv")
            latex_path = os.path.join(self.output_dir, f"model_comparison_{timestamp}.tex")
            json_path = os.path.join(self.output_dir, f"model_comparison_{timestamp}.json")

            # Save as CSV
            df.to_csv(csv_path)
            print(f"Comparison table saved to {csv_path}")

            # Save as LaTeX
            self._save_latex_table(df, latex_path)

            # Save as JSON
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Detailed results saved to {json_path}")

        return df

    def ablation_study(self, base_model_name: str, ablation_models: Dict[str, nn.Module],
                      data_loader: torch.utils.data.DataLoader, device: torch.device,
                      save_to_file: bool = True) -> pd.DataFrame:
        """Perform ablation study comparing base model with ablated versions."""
        if base_model_name not in self.results:
            warnings.warn(f"Base model {base_model_name} not evaluated. Evaluating now...")
            # This would need the base model to be provided
            raise ValueError("Base model must be evaluated first or provided")

        print(f"\n{'='*60}")
        print(f"Performing ablation study")
        print(f"{'='*60}")

        # Evaluate all ablation models
        for name, model in ablation_models.items():
            if name not in self.results:
                self.evaluate(name, model, data_loader, device, verbose=True)

        # Create ablation comparison table
        ablation_names = [base_model_name] + list(ablation_models.keys())
        ablation_df = self._create_ablation_table(ablation_names)

        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(self.output_dir, f"ablation_study_{timestamp}.csv")
            latex_path = os.path.join(self.output_dir, f"ablation_study_{timestamp}.tex")

            ablation_df.to_csv(csv_path)
            self._save_latex_table(ablation_df, latex_path, is_ablation=True)

            print(f"Ablation study saved to {csv_path}")

        return ablation_df

    def _save_model_results(self, model_name: str, metrics: Dict[str, float]):
        """Save individual model results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        results = {
            'model_name': model_name,
            'timestamp': timestamp,
            'metrics': metrics
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

    def _save_latex_table(self, df: pd.DataFrame, filepath: str, is_ablation: bool = False):
        """Save DataFrame as LaTeX table."""
        # Format numbers for LaTeX
        def format_number(x):
            if isinstance(x, (int, np.integer)):
                return str(x)
            elif isinstance(x, (float, np.floating)):
                if abs(x) < 0.001:
                    return "0.000"
                elif abs(x) < 1:
                    return f"{x:.3f}"
                else:
                    return f"{x:.3f}"
            else:
                return str(x)

        # Create LaTeX table
        latex_lines = []
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        if is_ablation:
            latex_lines.append("\\caption{Ablation study results}")
        else:
            latex_lines.append("\\caption{Model comparison results}")
        latex_lines.append("\\label{tab:model_comparison}")
        latex_lines.append("\\begin{tabular}{l" + "c" * len(df.columns) + "}")
        latex_lines.append("\\toprule")

        # Header
        header = "Model & " + " & ".join(df.columns) + " \\\\"
        latex_lines.append(header)
        latex_lines.append("\\midrule")

        # Rows
        for idx, row in df.iterrows():
            formatted_values = [format_number(val) for val in row.values]
            row_str = f"{idx} & " + " & ".join(formatted_values) + " \\\\"
            latex_lines.append(row_str)

        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        with open(filepath, 'w') as f:
            f.write("\n".join(latex_lines))

    def _create_ablation_table(self, model_names: List[str]) -> pd.DataFrame:
        """Create table showing relative changes for ablation study."""
        if len(model_names) < 2:
            return pd.DataFrame()

        base_model = model_names[0]
        if base_model not in self.results:
            raise ValueError(f"Base model {base_model} not found in results")

        base_metrics = self.results[base_model]

        # Create DataFrame with relative changes
        ablation_data = {}
        for name in model_names:
            if name not in self.results:
                continue

            metrics = self.results[name]
            row = {}
            for metric, value in metrics.items():
                if metric in base_metrics:
                    base_value = base_metrics[metric]
                    if base_value != 0:
                        rel_change = (value - base_value) / abs(base_value)
                        row[metric] = value
                        row[f"{metric}_rel"] = rel_change
                    else:
                        row[metric] = value
                        row[f"{metric}_rel"] = 0.0
                else:
                    row[metric] = value
            ablation_data[name] = row

        return pd.DataFrame(ablation_data).T


# Example usage
if __name__ == "__main__":
    # Example of how to use the evaluator
    print("Example usage of ModelEvaluator")

    # This is a template - users should fill in their own models and data
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ModelEvaluator()

    # Load your test data loader
    # test_loader = ...

    # Define your models
    models = {
        '3DMAEPP': your_3dmaepp_model,
        'RandomForest': random_forest_model,
        'UNet': unet_model,
        'PVT': pvt_model,
        'DPT': dpt_model,
        '3DQPE': threedqpe_model,
        'OpenSTL': openstl_model,
    }

    # Evaluate each model
    for name, model in models.items():
        evaluator.evaluate(name, model, test_loader, device)

    # Compare all models
    comparison_df = evaluator.compare_models()

    # Print comparison table
    print("\nModel Comparison:")
    print(comparison_df.to_string())

    # Ablation study (example)
    ablation_models = {
        'w/o 3DMAE': model_without_pretraining,
        'w/o Temporal': model_without_temporal_fusion,
        'w/o Classification': model_without_classification_head,
        'w/o StationLoss': model_without_station_loss,
    }

    ablation_df = evaluator.ablation_study(
        base_model_name='3DMAEPP',
        ablation_models=ablation_models,
        data_loader=test_loader,
        device=device
    )

    print("\nAblation Study:")
    print(ablation_df.to_string())
    """