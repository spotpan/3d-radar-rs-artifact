import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.finetune_guangdong_config import get_config
from training.finetune_guangdong import create_model, create_data_loaders


def strip_module_prefix(state_dict):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def patch_target_event(y, patch_size=128, rain_threshold=0.1, area_ratio=0.001):
    """
    y: (B,1,H,W)
    target event = fraction(y >= rain_threshold) >= area_ratio
    """
    y_bin = (y >= rain_threshold).float()
    frac = F.avg_pool2d(y_bin, kernel_size=patch_size, stride=patch_size)
    return (frac >= area_ratio), frac


def patch_pred_event(pred, patch_size=128, pred_threshold=0.0005, mode="max"):
    """
    pred: (B,1,H,W)
    """
    if mode == "max":
        score = F.max_pool2d(pred, kernel_size=patch_size, stride=patch_size)
        return score >= pred_threshold, score

    if mode == "mean":
        score = F.avg_pool2d(pred, kernel_size=patch_size, stride=patch_size)
        return score >= pred_threshold, score

    raise ValueError(f"Unsupported mode: {mode}")


def event_metrics(target_event, pred_event):
    """
    target_event / pred_event: bool tensors, same shape
    """
    t = target_event.bool()
    p = pred_event.bool()

    tp = int((t & p).sum().item())
    fp = int((~t & p).sum().item())
    fn = int((t & ~p).sum().item())
    tn = int((~t & ~p).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "target_count": int(t.sum().item()),
        "pred_count": int(p.sum().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/binary_regfirst_relu_cls01_5000t200v/binary_regfirst_relu_cls01_5000t200v/best_model.pt",
    )
    parser.add_argument("--max-batches", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--target-rain-threshold", type=float, default=0.1)
    parser.add_argument("--target-area-ratio", type=float, default=0.001)
    parser.add_argument("--pred-event-threshold", type=float, default=0.0005)
    parser.add_argument("--pred-mode", type=str, default="max", choices=["max", "mean"])

    parser.add_argument("--min-target-count", type=int, default=2)
    parser.add_argument("--max-target-count", type=int, default=40)
    parser.add_argument("--min-pred-count", type=int, default=2)
    parser.add_argument("--max-pred-count", type=int, default=60)

    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--out-csv", type=str, default="evaluation_results/rs_demo_sample_candidates.csv")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config()
    config["data"]["batch_size"] = args.batch_size
    config["data"]["num_workers"] = args.num_workers
    config["train"]["use_weighted_sampler"] = False

    model = create_model(config)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = strip_module_prefix(ckpt["model_state_dict"])
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    _, val_loader, dataset = create_data_loaders(config)

    rows = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.max_batches:
                break

            if i % 20 == 0:
                print(f"Scanning sample {i}/{args.max_batches}")

            x = batch["radar_sequence"].to(device)
            y = batch["rain"].to(device)

            out = model(x)
            pred = out["reg_output"].clamp(min=0)

            # Safety: if model returns larger batch than input, keep first B
            if pred.shape[0] != x.shape[0]:
                pred = pred[:x.shape[0]]

            target_event, target_frac = patch_target_event(
                y,
                patch_size=args.patch_size,
                rain_threshold=args.target_rain_threshold,
                area_ratio=args.target_area_ratio,
            )

            pred_event, pred_score = patch_pred_event(
                pred,
                patch_size=args.patch_size,
                pred_threshold=args.pred_event_threshold,
                mode=args.pred_mode,
            )

            m = event_metrics(target_event, pred_event)

            target_mean = float(y.mean().item())
            target_max = float(y.max().item())
            pred_mean = float(pred.mean().item())
            pred_max = float(pred.max().item())
            mae = float((pred - y).abs().mean().item())
            rmse = float(torch.sqrt(((pred - y) ** 2).mean()).item())

            row = {
                "sample_index": i,
                **m,
                "target_mean": target_mean,
                "target_max": target_max,
                "pred_mean": pred_mean,
                "pred_max": pred_max,
                "mae": mae,
                "rmse": rmse,
            }

            # A balanced visualization score:
            # prefer non-empty target/pred, overlap, moderate counts, and not-too-large local error.
            count_ok = (
                args.min_target_count <= m["target_count"] <= args.max_target_count
                and args.min_pred_count <= m["pred_count"] <= args.max_pred_count
            )

            row["count_ok"] = int(count_ok)
            row["viz_score"] = (
                2.0 * m["f1"]
                + 1.5 * m["iou"]
                + 0.1 * min(m["target_count"], 20) / 20.0
                + 0.1 * min(m["pred_count"], 20) / 20.0
                - 0.05 * mae
            )

            if not count_ok:
                row["viz_score"] -= 1.0

            rows.append(row)

    rows_sorted = sorted(rows, key=lambda d: d["viz_score"], reverse=True)

    print("\n==================== Top RS demo sample candidates ====================")
    print(
        f"{'rank':>4} {'idx':>6} {'score':>9} {'iou':>8} {'f1':>8} "
        f"{'prec':>8} {'rec':>8} {'tar':>5} {'pred':>5} "
        f"{'tp':>5} {'fp':>5} {'fn':>5} {'mae':>8} {'tmax':>8} {'pmax':>8}"
    )

    for rank, r in enumerate(rows_sorted[:args.top_k], start=1):
        print(
            f"{rank:4d} {r['sample_index']:6d} {r['viz_score']:9.4f} "
            f"{r['iou']:8.4f} {r['f1']:8.4f} {r['precision']:8.4f} {r['recall']:8.4f} "
            f"{r['target_count']:5d} {r['pred_count']:5d} "
            f"{r['tp']:5d} {r['fp']:5d} {r['fn']:5d} "
            f"{r['mae']:8.4f} {r['target_max']:8.3f} {r['pred_max']:8.3f}"
        )

    out_path = PROJECT_ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        pd.DataFrame(rows_sorted).to_csv(out_path, index=False)
        print(f"\nSaved candidates to: {out_path}")
    except Exception as e:
        print(f"\nCould not save CSV: {e}")

    dataset.close()


if __name__ == "__main__":
    main()
