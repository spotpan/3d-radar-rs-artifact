#!/usr/bin/env python3
"""
Example script for running model evaluation.

This script demonstrates how to use the evaluation utilities to:
1. Load a trained 3DMAEPP model.
2. Evaluate model performance on a test set.
3. Generate the model comparison table corresponding to Table 1.
4. Generate the ablation study table corresponding to Table 2.

Users need to implement the following parts:
1. Load the target 3DMAEPP model.
2. Prepare the test data loader.
3. Define comparison models and ablation variants.

Note:
This is a template script for reproducing Table 1 and Table 2 when
the required checkpoints and authorized data are available.
"""

import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.evaluate import ModelEvaluator


def load_your_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load the target 3DMAEPP model.

    Users should implement this function according to their own model
    definition and checkpoint format.

    Args:
        model_path: Path to the model checkpoint.
        device: Torch device.

    Returns:
        Loaded PyTorch model.
    """
    print(f"Loading model from: {model_path}")

    # Example implementation:
    # from models.rain_decoder import PrecipitationDecoder
    #
    # model = PrecipitationDecoder(...)
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.eval()
    # model.to(device)
    # return model

    raise NotImplementedError("Please implement load_your_model().")


def load_comparison_models(device: torch.device) -> dict:
    """
    Load comparison models, such as RandomForest, UNet, PVT, DPT,
    3DQPE, and OpenSTL.

    Users should implement this function according to the available
    checkpoints and model definitions.

    Args:
        device: Torch device.

    Returns:
        A dictionary mapping model names to model instances.
    """
    models = {}

    # Example:
    # models["RandomForest"] = load_random_forest_model()
    # models["UNet"] = load_unet_model()
    # models["PVT"] = load_pvt_model()
    # models["DPT"] = load_dpt_model()
    # models["3DQPE"] = load_3dqpe_model()
    # models["OpenSTL"] = load_openstl_model()
    #
    # for _, model in models.items():
    #     if hasattr(model, "to"):
    #         model.to(device)
    #     if hasattr(model, "eval"):
    #         model.eval()
    #
    # return models

    raise NotImplementedError("Please implement load_comparison_models().")


def load_ablation_models(base_model: nn.Module, device: torch.device) -> dict:
    """
    Load ablation variants of the base 3DMAEPP model.

    Users should implement this function according to the available
    ablation checkpoints and model definitions.

    Args:
        base_model: Base model instance.
        device: Torch device.

    Returns:
        A dictionary mapping ablation names to model instances.
    """
    ablation_models = {}

    # Example:
    # ablation_models["w/o 3DMAE"] = create_model_without_pretraining()
    # ablation_models["w/o temporal fusion"] = create_model_without_temporal_fusion()
    # ablation_models["w/o dual-head decoder"] = create_model_without_dual_head()
    # ablation_models["w/o station loss"] = create_model_without_station_loss()
    #
    # for _, model in ablation_models.items():
    #     model.to(device)
    #     model.eval()
    #
    # return ablation_models

    raise NotImplementedError("Please implement load_ablation_models().")


def create_test_data_loader(batch_size: int = 4) -> DataLoader:
    """
    Create the test data loader.

    Users should implement this function according to the local data
    organization and preprocessing pipeline.

    Args:
        batch_size: Batch size.

    Returns:
        PyTorch DataLoader.
    """
    # Example:
    # from data.dataloader import FinetuneDataset
    #
    # test_dataset = FinetuneDataset(
    #     data_paths=["/path/to/radar_station_dataset/test_data.h5"],
    #     radar_height_layers=[0, 1, 2, 3, 4, 5],
    #     spatial_size=(700, 900),
    #     target_minutes=[0, 30],
    #     history_frames=6,
    #     frame_interval=12,
    # )
    #
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=2,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    #
    # return test_loader

    raise NotImplementedError("Please implement create_test_data_loader().")


def custom_prediction_fn(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Extract the precipitation prediction from a model output.

    Implement this function if the model output is not directly a tensor
    with shape (B, H, W) or (B, 1, H, W).

    Args:
        model: PyTorch model.
        inputs: Model inputs.

    Returns:
        Prediction tensor with shape (B, H, W) or (B, 1, H, W).
    """
    outputs = model(inputs)

    if isinstance(outputs, dict):
        if "reg_output" in outputs:
            return outputs["reg_output"]
        if "prediction" in outputs:
            return outputs["prediction"]

        for _, value in outputs.items():
            if isinstance(value, torch.Tensor):
                return value

    if isinstance(outputs, tuple):
        return outputs[0]

    return outputs


def main() -> None:
    """Run the full model evaluation workflow."""
    print("=" * 70)
    print("3DMAEPP model evaluation utility")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = f"./evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    evaluator = ModelEvaluator(output_dir=output_dir)

    try:
        print("\n" + "=" * 70)
        print("Step 1: Preparing test data")
        print("=" * 70)

        test_loader = create_test_data_loader(batch_size=4)
        print("Test data loader created.")
        print(f"Number of test batches: {len(test_loader)}")

        print("\n" + "=" * 70)
        print("Step 2: Loading the 3DMAEPP model")
        print("=" * 70)

        model_path = "./checkpoints/precipitation/best_model.pt"
        model_3dmaepp = load_your_model(model_path, device)

        evaluator.evaluate(
            model_name="3DMAEPP",
            model=model_3dmaepp,
            data_loader=test_loader,
            device=device,
            prediction_fn=custom_prediction_fn,
            verbose=True,
        )

        print("\n" + "=" * 70)
        print("Step 3: Evaluating comparison models")
        print("=" * 70)

        comparison_models = load_comparison_models(device)

        for name, model in comparison_models.items():
            evaluator.evaluate(
                model_name=name,
                model=model,
                data_loader=test_loader,
                device=device,
                prediction_fn=custom_prediction_fn,
                verbose=True,
            )

        print("\n" + "=" * 70)
        print("Step 4: Generating model comparison table")
        print("=" * 70)

        comparison_df = evaluator.compare_models(save_to_file=True)

        print("\nModel comparison results:")
        print("-" * 70)
        print(comparison_df.to_string())

        print("\n" + "=" * 70)
        print("Step 5: Running ablation study")
        print("=" * 70)

        ablation_models = load_ablation_models(model_3dmaepp, device)

        ablation_df = evaluator.ablation_study(
            base_model_name="3DMAEPP",
            ablation_models=ablation_models,
            data_loader=test_loader,
            device=device,
            save_to_file=True,
        )

        print("\nAblation study results:")
        print("-" * 70)
        print(ablation_df.to_string())

        print("\n" + "=" * 70)
        print("Step 6: Generating LaTeX tables")
        print("=" * 70)

        comparison_latex = comparison_df.to_latex(
            float_format="%.3f",
            caption=(
                "Model comparison results. ME, MAE, and RMSE are in mm/h; "
                "CC denotes correlation coefficient; CSI denotes critical "
                "success index."
            ),
            label="tab:comparison_results",
        )

        ablation_latex = ablation_df.to_latex(
            float_format="%.3f",
            caption=(
                "Ablation study results. Values in parentheses denote "
                "relative changes with respect to the full model."
            ),
            label="tab:ablation_results",
        )

        latex_file = os.path.join(output_dir, "results_tables.tex")
        with open(latex_file, "w", encoding="utf-8") as f:
            f.write("\\subsection{Model Comparison Results}\n")
            f.write(comparison_latex)
            f.write("\n\\subsection{Ablation Study Results}\n")
            f.write(ablation_latex)

        print(f"LaTeX tables saved to: {latex_file}")

        print("\n" + "=" * 70)
        print("Step 7: Writing result summary")
        print("=" * 70)

        key_metrics = [
            "ME",
            "MAE",
            "RMSE",
            "CC",
            "CSI_0.1",
            "CSI_1.0",
            "CSI_5.0",
            "CSI_10.0",
        ]

        summary_file = os.path.join(output_dir, "results_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("3DMAEPP model evaluation summary\n")
            f.write("=" * 50 + "\n\n")

            f.write("1. Model comparison results\n")
            f.write("-" * 50 + "\n")
            for metric in key_metrics:
                if metric in comparison_df.columns:
                    best_value = (
                        comparison_df[metric].max()
                        if metric == "CC"
                        else comparison_df[metric].min()
                    )
                    best_model = comparison_df[
                        comparison_df[metric] == best_value
                    ].index[0]
                    f.write(
                        f"{metric}: best model = {best_model} "
                        f"({best_value:.3f})\n"
                    )

            f.write("\n2. Ablation study results\n")
            f.write("-" * 50 + "\n")
            for model_name in ablation_df.index:
                if model_name != "3DMAEPP":
                    f.write(f"\n{model_name}:\n")
                    for metric in key_metrics:
                        if metric in ablation_df.columns:
                            value = ablation_df.loc[model_name, metric]
                            rel_col = f"{metric}_rel"
                            if rel_col in ablation_df.columns:
                                rel_change = ablation_df.loc[model_name, rel_col]
                                if rel_change > 0:
                                    change_str = f"(+{rel_change * 100:.1f}%)"
                                else:
                                    change_str = f"(-{abs(rel_change) * 100:.1f}%)"
                                f.write(f"  {metric}: {value:.3f} {change_str}\n")
                            else:
                                f.write(f"  {metric}: {value:.3f}\n")

        print(f"Result summary saved to: {summary_file}")

        print("\n" + "=" * 70)
        print("Evaluation completed.")
        print("=" * 70)
        print(f"All results saved to: {output_dir}")
        print(f"1. Model comparison table: {output_dir}/model_comparison_*.csv")
        print(f"2. Ablation study table: {output_dir}/ablation_study_*.csv")
        print(f"3. LaTeX tables: {output_dir}/results_tables.tex")
        print(f"4. Result summary: {output_dir}/results_summary.txt")

    except NotImplementedError as exc:
        print(f"\nError: {exc}")
        print("Please implement the required functions indicated in this script:")
        print("  1. load_your_model()")
        print("  2. load_comparison_models()")
        print("  3. load_ablation_models()")
        print("  4. create_test_data_loader()")
    except Exception as exc:
        print(f"\nAn error occurred during evaluation: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
