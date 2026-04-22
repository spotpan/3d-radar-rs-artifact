# Model Evaluation Tool

This tool evaluates precipitation estimation models using the metrics from the paper "基于三维掩码自编码器的定量降水估计方法" (3DMAE for Quantitative Precipitation Estimation).

## Features

- Computes all metrics from the paper:
  - ME (Mean Error/Bias)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - CC (Correlation Coefficient)
  - CSI (Critical Success Index) at thresholds: 0.1, 1.0, 5.0, 10.0 mm/h
  - Additional bias metrics for different precipitation intensity ranges

- Supports comparison of multiple models
- Generates ablation study tables
- Exports results in multiple formats: CSV, LaTeX, JSON
- Easy integration with PyTorch models

## Installation

1. Ensure you have the required dependencies:
```bash
pip install torch numpy pandas scikit-learn
```

2. The evaluation tool is included in the `evaluation` directory.

## Quick Start

### 1. Prepare Your Test Data

Create a PyTorch DataLoader for your test dataset. The data loader should return batches in one of these formats:

**Option A: Dictionary format (recommended)**
```python
batch = {
    'radar_sequence': radar_data,  # (B, T, C, H, W)
    'rain': precipitation_target,  # (B, 1, H, W) or (B, H, W)
    'mask': optional_mask,         # (B, H, W) binary mask for valid pixels
}
```

**Option B: Tuple format**
```python
batch = (radar_data, precipitation_target)
```

### 2. Prepare Your Models

Your models should be PyTorch modules that accept radar sequence input and return predictions. The evaluator supports two output formats:

**Option A: Dictionary output (recommended)**
```python
outputs = model(radar_sequence)
predictions = outputs['reg_output']  # (B, 1, H, W) or (B, H, W)
```

**Option B: Direct tensor output**
```python
predictions = model(radar_sequence)  # (B, 1, H, W) or (B, H, W)
```

If your model uses a different output format, you can provide a custom `prediction_fn`.

### 3. Run Evaluation

Here's a complete example:

```python
import torch
from torch.utils.data import DataLoader
from evaluation.evaluate import ModelEvaluator

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize evaluator
evaluator = ModelEvaluator(output_dir="./evaluation_results")

# Load your test data
# test_dataset = YourTestDataset(...)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define your models (example - fill in with your actual models)
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
    model.to(device)
    model.eval()
    
    evaluator.evaluate(
        model_name=name,
        model=model,
        data_loader=test_loader,
        device=device,
        verbose=True
    )

# Compare all models
comparison_df = evaluator.compare_models()
print("\nModel Comparison Results:")
print(comparison_df.to_string())

# Save to file (optional)
comparison_df.to_csv("model_comparison_results.csv")
```

### 4. Run Ablation Study

To compare the base model with ablated versions:

```python
# Define ablation models
ablation_models = {
    'w/o 3DMAE': model_without_pretraining,
    'w/o Temporal': model_without_temporal_fusion,
    'w/o Classification': model_without_classification_head,
    'w/o StationLoss': model_without_station_loss,
}

# Run ablation study
ablation_df = evaluator.ablation_study(
    base_model_name='3DMAEPP',
    ablation_models=ablation_models,
    data_loader=test_loader,
    device=device,
    save_to_file=True
)

print("\nAblation Study Results:")
print(ablation_df.to_string())
```

## Advanced Usage

### Custom Prediction Function

If your model doesn't follow the standard output format, provide a custom prediction function:

```python
def custom_prediction_fn(model, inputs):
    # Your model might return a tuple or different structure
    outputs = model(inputs)
    # Extract the precipitation predictions
    if isinstance(outputs, tuple):
        predictions = outputs[0]  # First element is predictions
    else:
        predictions = outputs['prediction']  # Different key name
    
    return predictions

# Use custom function in evaluation
evaluator.evaluate(
    model_name='CustomModel',
    model=custom_model,
    data_loader=test_loader,
    device=device,
    prediction_fn=custom_prediction_fn
)
```

### Using Masks for Evaluation

If your data has invalid pixels (e.g., outside region of interest), provide a mask:

```python
# Your dataset should return masks
batch = {
    'radar_sequence': radar_data,
    'rain': precipitation_target,
    'mask': valid_pixel_mask,  # 1 for valid, 0 for invalid
}

# The evaluator will automatically use the mask
```

### Evaluating on Subset of Data

To evaluate on a subset of test data:

```python
from torch.utils.data import Subset

# Create subset
subset_indices = list(range(100))  # First 100 samples
test_subset = Subset(test_dataset, subset_indices)
test_loader_subset = DataLoader(test_subset, batch_size=4, shuffle=False)

# Evaluate on subset
evaluator.evaluate('ModelName', model, test_loader_subset, device)
```

## Output Files

The evaluator generates several output files in the specified `output_dir`:

1. **Individual model results**: `{model_name}_{timestamp}.json`
   - Contains all metrics for a single model
   - JSON format for easy parsing

2. **Model comparison**: `model_comparison_{timestamp}.csv`
   - CSV table comparing all models
   - Suitable for Excel or pandas analysis

3. **Model comparison (LaTeX)**: `model_comparison_{timestamp}.tex`
   - LaTeX table for paper publication
   - Includes proper formatting and caption

4. **Detailed results**: `model_comparison_{timestamp}.json`
   - Complete results in JSON format
   - Includes all metrics for all models

5. **Ablation study**: `ablation_study_{timestamp}.csv` and `.tex`
   - Similar to model comparison but for ablation study
   - Shows relative changes from base model

## Metrics Explanation

### ME (Mean Error / Bias)
- Average difference between predictions and targets
- Positive values indicate overestimation, negative values indicate underestimation
- Formula: `ME = mean(pred - target)`

### MAE (Mean Absolute Error)
- Average absolute difference between predictions and targets
- More robust to outliers than RMSE
- Formula: `MAE = mean(|pred - target|)`

### RMSE (Root Mean Squared Error)
- Square root of average squared differences
- More sensitive to large errors than MAE
- Formula: `RMSE = sqrt(mean((pred - target)^2))`

### CC (Correlation Coefficient)
- Measures linear relationship between predictions and targets
- Range: -1 to 1, higher is better
- Formula: `CC = corr(pred, target)`

### CSI (Critical Success Index)
- Measures accuracy of binary classification (rain/no-rain)
- Formula: `CSI = hits / (hits + false_alarms + misses)`
- Calculated at thresholds: 0.1, 1.0, 5.0, 10.0 mm/h

### Bias by Intensity Range
- ME calculated for different precipitation intensity ranges:
  - Light: 0.1-1.0 mm/h
  - Moderate: 1.0-5.0 mm/h
  - Heavy: 5.0-10.0 mm/h
  - Extreme: >10.0 mm/h

## Tips for Accurate Evaluation

1. **Use consistent units**: Ensure predictions and targets are in the same units (mm/h).

2. **Handle missing data**: Use masks to exclude invalid pixels from evaluation.

3. **Sufficient sample size**: Ensure test set is large enough for statistically significant results.

4. **Random seed**: Set random seeds for reproducible evaluation:
   ```python
   import random
   import numpy as np
   import torch
   
   random.seed(42)
   np.random.seed(42)
   torch.manual_seed(42)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(42)
   ```

5. **Batch size**: Use appropriate batch size based on GPU memory.

## Troubleshooting

### Model outputs wrong shape
- Ensure model returns predictions with shape `(B, H, W)` or `(B, 1, H, W)`
- Use `prediction_fn` to extract correct predictions

### Memory issues
- Reduce batch size
- Use CPU evaluation for large models: `device=torch.device('cpu')`

### Slow evaluation
- Use GPU if available
- Increase batch size (within memory limits)
- Disable verbose mode: `verbose=False`

### Missing metrics
- Check that predictions and targets have the same shape
- Ensure mask (if provided) has correct shape

## Example Results

The evaluator produces tables similar to those in the paper:

| Model | ME | MAE | RMSE | CC | CSI_0.1 | CSI_1.0 | CSI_5.0 | CSI_10.0 |
|-------|-----|-----|------|-----|---------|---------|---------|----------|
| 3DMAEPP | -0.302 | 0.925 | 2.581 | 0.723 | 0.732 | 0.658 | 0.607 | 0.411 |
| RandomForest | -0.415 | 1.128 | 3.215 | 0.681 | 0.698 | 0.621 | 0.568 | 0.385 |
| UNet | -0.378 | 1.052 | 2.894 | 0.702 | 0.715 | 0.642 | 0.589 | 0.398 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Citation

If you use this evaluation tool in your research, please cite the original paper:

```
@article{3dmae2025,
  title={基于三维掩码自编码器的定量降水估计方法},
  author={Author Names},
  journal={Journal Name},
  year={2025}
}
```

## License

This evaluation tool is provided as part of the 3DMAE precipitation estimation project.