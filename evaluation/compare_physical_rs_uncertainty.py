import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.finetune_guangdong_config import get_config
from training.finetune_guangdong import create_model, create_data_loaders


def strip_module_prefix(state_dict):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perturb_gaussian(x, sigma=0.03):
    return x + torch.randn_like(x) * sigma


def perturb_level_dropout(x, drop_prob=0.15):
    x2 = x.clone()
    b, t, c, h, w = x2.shape
    mask = torch.ones((b, t, c, 1, 1), device=x2.device, dtype=x2.dtype)
    drop = torch.rand((b, t, c, 1, 1), device=x2.device) < drop_prob
    mask = mask.masked_fill(drop, 0.0)
    return x2 * mask


def perturb_level_scaling(x, scale_std=0.10):
    b, t, c, h, w = x.shape
    scale = 1.0 + torch.randn((b, t, c, 1, 1), device=x.device, dtype=x.dtype) * scale_std
    scale = torch.clamp(scale, 0.5, 1.5)
    return x * scale


def perturb_temporal_dropout(x, drop_prob=0.15):
    x2 = x.clone()
    b, t, c, h, w = x2.shape
    mask = torch.ones((b, t, 1, 1, 1), device=x2.device, dtype=x2.dtype)
    drop = torch.rand((b, t, 1, 1, 1), device=x2.device) < drop_prob
    mask = mask.masked_fill(drop, 0.0)
    return x2 * mask


def perturb_block_mask(x, block_size=64, num_blocks=4):
    x2 = x.clone()
    b, t, c, h, w = x2.shape
    for _ in range(num_blocks):
        y0 = torch.randint(0, max(h - block_size, 1), (1,), device=x2.device).item()
        x0 = torch.randint(0, max(w - block_size, 1), (1,), device=x2.device).item()
        x2[..., y0:y0 + block_size, x0:x0 + block_size] = 0.0
    return x2


def apply_perturbation(x, mode, args):
    if mode == "gaussian":
        return perturb_gaussian(x, sigma=args.gaussian_sigma)
    if mode == "level_dropout":
        return perturb_level_dropout(x, drop_prob=args.level_drop_prob)
    if mode == "level_scaling":
        return perturb_level_scaling(x, scale_std=args.level_scale_std)
    if mode == "temporal_dropout":
        return perturb_temporal_dropout(x, drop_prob=args.temporal_drop_prob)
    if mode == "block_mask":
        return perturb_block_mask(x, block_size=args.block_size, num_blocks=args.num_blocks)
    if mode == "mixed":
        x2 = perturb_gaussian(x, sigma=args.gaussian_sigma)
        x2 = perturb_level_scaling(x2, scale_std=args.level_scale_std)
        x2 = perturb_block_mask(x2, block_size=args.block_size, num_blocks=args.num_blocks)
        return x2
    raise ValueError(f"Unknown perturbation mode: {mode}")


def patch_target_event(y, patch_size, rain_threshold, area_ratio):
    """
    y: (1,1,H,W)
    """
    y_bin = (y >= rain_threshold).float()
    frac = F.avg_pool2d(y_bin, kernel_size=patch_size, stride=patch_size)
    return (frac[0, 0] >= area_ratio), frac[0, 0]


def patch_scores_from_pred(pred, patch_size):
    """
    pred: (1,1,H,W)
    return: (hp,wp)
    """
    score = F.max_pool2d(pred, kernel_size=patch_size, stride=patch_size)
    return score[0, 0]


def boundary_mask(binary_mask):
    """
    binary_mask: bool tensor (hp,wp)
    Boundary = cells whose 8-neighborhood is not homogeneous.
    """
    x = binary_mask.float()[None, None]
    dil = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)[0, 0]
    ero = 1.0 - F.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)[0, 0]
    bnd = (dil - ero) > 0
    return bnd


def safe_mean(values, mask=None):
    if mask is not None:
        values = values[mask]
    if values.numel() == 0:
        return float("nan")
    return float(values.float().mean().item())


def update_confusion(stats, target_event, pred_event):
    t = target_event.bool()
    p = pred_event.bool()

    stats["tp"] += int((t & p).sum().item())
    stats["fp"] += int((~t & p).sum().item())
    stats["fn"] += int((t & ~p).sum().item())
    stats["tn"] += int((~t & ~p).sum().item())


def compute_metrics_from_confusion(tp, fp, fn, tn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return precision, recall, f1, acc, iou


def physical_meaning(mode):
    mapping = {
        "gaussian": "intensity noise / calibration uncertainty",
        "level_dropout": "missing or unreliable vertical radar layers",
        "level_scaling": "vertical-profile amplitude uncertainty",
        "temporal_dropout": "missing historical radar frames",
        "block_mask": "local echo missing / occlusion",
        "mixed": "combined structured radar uncertainty",
    }
    return mapping.get(mode, "")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/binary_regfirst_relu_cls01_5000t200v/binary_regfirst_relu_cls01_5000t200v/best_model.pt",
    )
    parser.add_argument("--perturbations", nargs="+",
                        default=["gaussian", "level_dropout", "level_scaling", "temporal_dropout", "block_mask", "mixed"])
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--target-rain-threshold", type=float, default=0.1)
    parser.add_argument("--target-area-ratio", type=float, default=0.001)
    parser.add_argument("--pred-event-threshold", type=float, default=0.0005)
    parser.add_argument("--smooth-event-threshold", type=float, default=0.5)
    parser.add_argument("--high-u-threshold", type=float, default=0.5)

    parser.add_argument("--gaussian-sigma", type=float, default=0.03)
    parser.add_argument("--level-drop-prob", type=float, default=0.15)
    parser.add_argument("--level-scale-std", type=float, default=0.10)
    parser.add_argument("--temporal-drop-prob", type=float, default=0.15)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)

    parser.add_argument("--out-csv", type=str, default="evaluation_results/physical_rs_uncertainty_comparison.csv")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Physical RS comparison")
    print(f"  perturbations: {args.perturbations}", flush=True)
    print(f"  num_samples: {args.num_samples}", flush=True)
    print(f"  max_batches: {args.max_batches}", flush=True)
    print(f"  seed: {args.seed}", flush=True)

    config = get_config()
    config["data"]["batch_size"] = args.batch_size
    config["data"]["num_workers"] = args.num_workers
    config["train"]["use_weighted_sampler"] = False

    print("Creating model...", flush=True)
    model = create_model(config)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = strip_module_prefix(ckpt["model_state_dict"])
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    print("Creating data loaders...", flush=True)
    _, val_loader, dataset = create_data_loaders(config)

    accum = {}
    for mode in args.perturbations:
        accum[mode] = defaultdict(float)
        accum[mode]["tp"] = 0
        accum[mode]["fp"] = 0
        accum[mode]["fn"] = 0
        accum[mode]["tn"] = 0
        accum[mode]["baseline_tp"] = 0
        accum[mode]["baseline_fp"] = 0
        accum[mode]["baseline_fn"] = 0
        accum[mode]["baseline_tn"] = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.max_batches:
                break

            if i % 5 == 0:
                print(f"Processing validation sample {i}/{args.max_batches}", flush=True)

            x = batch["radar_sequence"].to(device)
            y = batch["rain"].to(device)

            # Force first sample for single-sample patch evaluation
            if y.shape[0] != 1:
                y = y[:1]
            target_event, target_frac = patch_target_event(
                y,
                patch_size=args.patch_size,
                rain_threshold=args.target_rain_threshold,
                area_ratio=args.target_area_ratio,
            )
            bnd = boundary_mask(target_event)
            non_target = ~target_event

            baseline = model(x)["reg_output"].clamp(min=0)
            baseline = baseline[:1]
            baseline_score = patch_scores_from_pred(baseline, args.patch_size)
            baseline_event = baseline_score >= args.pred_event_threshold

            for mode in args.perturbations:
                if i % 5 == 0:
                    print(f"  perturbation={mode}", flush=True)
                stats = accum[mode]

                # Baseline confusion is stored separately for paired comparison.
                # Do NOT add baseline confusion into RS tp/fp/fn/tn.
                t = target_event.bool()
                bp = baseline_event.bool()
                stats["baseline_tp"] += int((t & bp).sum().item())
                stats["baseline_fp"] += int((~t & bp).sum().item())
                stats["baseline_fn"] += int((t & ~bp).sum().item())
                stats["baseline_tn"] += int((~t & ~bp).sum().item())

                score_list = []
                event_list = []

                for n in range(args.num_samples):
                    x_pert = apply_perturbation(x, mode, args)
                    pred = model(x_pert)["reg_output"].clamp(min=0)
                    pred = pred[:1]
                    score = patch_scores_from_pred(pred, args.patch_size)
                    score_list.append(score)
                    event_list.append((score >= args.pred_event_threshold).float())

                scores = torch.stack(score_list, dim=0)      # N,hp,wp
                events = torch.stack(event_list, dim=0)      # N,hp,wp

                prob = events.mean(dim=0)
                rs_event = prob >= args.smooth_event_threshold
                U = 4.0 * prob * (1.0 - prob)
                score_std = scores.std(dim=0)

                # RS event metrics
                update_confusion(stats, target_event, rs_event)

                high_u = U >= args.high_u_threshold

                # accumulate uncertainty stats
                stats["n_samples_eval"] += 1
                stats["n_patches"] += int(U.numel())
                stats["n_target_patches"] += int(target_event.sum().item())
                stats["n_boundary_patches"] += int(bnd.sum().item())
                stats["n_high_u_patches"] += int(high_u.sum().item())

                stats["U_sum"] += float(U.sum().item())
                stats["U_target_sum"] += float(U[target_event].sum().item()) if target_event.any() else 0.0
                stats["U_nontarget_sum"] += float(U[non_target].sum().item()) if non_target.any() else 0.0
                stats["U_boundary_sum"] += float(U[bnd].sum().item()) if bnd.any() else 0.0

                stats["score_std_sum"] += float(score_std.sum().item())
                stats["score_std_target_sum"] += float(score_std[target_event].sum().item()) if target_event.any() else 0.0
                stats["score_std_boundary_sum"] += float(score_std[bnd].sum().item()) if bnd.any() else 0.0

                stats["high_u_boundary"] += int((high_u & bnd).sum().item())
                stats["high_u_target"] += int((high_u & target_event).sum().item())

    rows = []
    for mode in args.perturbations:
        s = accum[mode]

        precision, recall, f1, acc, iou = compute_metrics_from_confusion(
            s["tp"], s["fp"], s["fn"], s["tn"]
        )

        bprec, brec, bf1, bacc, biou = compute_metrics_from_confusion(
            s["baseline_tp"], s["baseline_fp"], s["baseline_fn"], s["baseline_tn"]
        )

        n_p = max(s["n_patches"], 1)
        n_t = max(s["n_target_patches"], 1)
        n_nt = max(n_p - s["n_target_patches"], 1)
        n_b = max(s["n_boundary_patches"], 1)
        n_high = max(s["n_high_u_patches"], 1)

        row = {
            "perturbation": mode,
            "physical_meaning": physical_meaning(mode),
            "num_samples": args.num_samples,
            "num_eval_cases": int(s["n_samples_eval"]),
            "baseline_precision": bprec,
            "baseline_recall": brec,
            "baseline_f1": bf1,
            "rs_precision": precision,
            "rs_recall": recall,
            "rs_f1": f1,
            "rs_iou": iou,
            "delta_f1_vs_baseline": f1 - bf1,
            "mean_uncertainty_U": s["U_sum"] / n_p,
            "target_U": s["U_target_sum"] / n_t,
            "non_target_U": s["U_nontarget_sum"] / n_nt,
            "boundary_U": s["U_boundary_sum"] / n_b,
            "high_U_area_ratio": s["n_high_u_patches"] / n_p,
            "high_U_boundary_fraction": s["high_u_boundary"] / n_high,
            "high_U_target_fraction": s["high_u_target"] / n_high,
            "mean_score_std": s["score_std_sum"] / n_p,
            "target_score_std": s["score_std_target_sum"] / n_t,
            "boundary_score_std": s["score_std_boundary_sum"] / n_b,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # pretty print main table
    cols = [
        "perturbation",
        "rs_f1",
        "rs_precision",
        "rs_recall",
        "mean_uncertainty_U",
        "boundary_U",
        "high_U_area_ratio",
        "high_U_boundary_fraction",
        "mean_score_std",
        "boundary_score_std",
    ]

    print("\n==================== Physical RS uncertainty comparison ====================")
    print(df[cols].round(4).to_string(index=False))

    out_path = PROJECT_ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved CSV: {out_path}")

    dataset.close()


if __name__ == "__main__":
    main()
