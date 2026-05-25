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


def update_conf(conf, target_bin, pred_bin):
    for t in [0, 1]:
        for p in [0, 1]:
            conf[t, p] += int(((target_bin == t) & (pred_bin == p)).sum().item())


def metrics_from_conf(conf):
    tn, fp = conf[0, 0], conf[0, 1]
    fn, tp = conf[1, 0], conf[1, 1]

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(conf.sum(), 1)

    return precision, recall, f1, accuracy, tn, fp, fn, tp


def patch_score_map(x, patch_size, mode="mean", topk_ratio=0.05):
    """
    x: (B, 1, H, W)
    returns patch-level continuous score map.
    """
    if mode == "mean":
        return F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)

    if mode == "max":
        return F.max_pool2d(x, kernel_size=patch_size, stride=patch_size)

    if mode == "topk":
        # unfold patches: (B, C*ps*ps, L)
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        # patches: (B, ps*ps, L) since C=1
        b, n, l = patches.shape
        k = max(1, int(n * topk_ratio))
        topk_vals = torch.topk(patches, k=k, dim=1).values
        scores = topk_vals.mean(dim=1)  # (B, L)

        h_out = x.shape[-2] // patch_size
        w_out = x.shape[-1] // patch_size
        return scores.view(b, 1, h_out, w_out)

    raise ValueError(f"Unknown score mode: {mode}")


def patch_area_event(x, patch_size, rain_threshold=0.1, area_ratio=0.01):
    bin_map = (x >= rain_threshold).float()
    frac = F.avg_pool2d(bin_map, kernel_size=patch_size, stride=patch_size)
    return frac >= area_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/binary_regfirst_relu_cls01_5000t200v/binary_regfirst_relu_cls01_5000t200v/best_model.pt",
    )
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--target-rain-threshold", type=float, default=0.1)
    parser.add_argument("--target-area-ratios", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.02])
    parser.add_argument("--pred-thresholds", type=float, nargs="+",
                        default=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--topk-ratios", type=float, nargs="+", default=[0.01, 0.05, 0.10])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config()
    config["data"]["batch_size"] = args.batch_size
    config["data"]["num_workers"] = args.num_workers
    config["train"]["use_weighted_sampler"] = False

    model = create_model(config)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    _, val_loader, dataset = create_data_loaders(config)

    # key: (patch_size, target_area_ratio, pred_mode, pred_threshold, topk_ratio)
    confs = {}

    for ps in args.patch_sizes:
        for tar in args.target_area_ratios:
            for mode in ["mean", "max"]:
                for th in args.pred_thresholds:
                    confs[(ps, tar, mode, th, None)] = np.zeros((2, 2), dtype=np.int64)

            # area mode: pred event also uses area ratio over pred >= target_rain_threshold
            for par in args.target_area_ratios:
                confs[(ps, tar, "area", par, None)] = np.zeros((2, 2), dtype=np.int64)

            for topk in args.topk_ratios:
                for th in args.pred_thresholds:
                    confs[(ps, tar, "topk", th, topk)] = np.zeros((2, 2), dtype=np.int64)

    mae_sum = 0.0
    rmse_sum = 0.0
    n_pix = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.max_batches:
                break
            if i % 10 == 0:
                print(f"Processed {i}/{args.max_batches} batches")

            x = batch["radar_sequence"].to(device)
            y = batch["rain"].to(device)

            out = model(x)
            pred = out["reg_output"].clamp(min=0)

            diff = pred - y
            mae_sum += diff.abs().sum().item()
            rmse_sum += (diff ** 2).sum().item()
            n_pix += diff.numel()

            for ps in args.patch_sizes:
                # cache pred score maps
                pred_mean = patch_score_map(pred, ps, mode="mean")
                pred_max = patch_score_map(pred, ps, mode="max")
                pred_area_events = {
                    par: patch_area_event(
                        pred,
                        patch_size=ps,
                        rain_threshold=args.target_rain_threshold,
                        area_ratio=par,
                    ).long()
                    for par in args.target_area_ratios
                }
                pred_topk_scores = {
                    topk: patch_score_map(pred, ps, mode="topk", topk_ratio=topk)
                    for topk in args.topk_ratios
                }

                for tar in args.target_area_ratios:
                    target_patch = patch_area_event(
                        y,
                        patch_size=ps,
                        rain_threshold=args.target_rain_threshold,
                        area_ratio=tar,
                    ).long()

                    for th in args.pred_thresholds:
                        update_conf(
                            confs[(ps, tar, "mean", th, None)],
                            target_patch,
                            (pred_mean >= th).long(),
                        )
                        update_conf(
                            confs[(ps, tar, "max", th, None)],
                            target_patch,
                            (pred_max >= th).long(),
                        )

                    for par in args.target_area_ratios:
                        update_conf(
                            confs[(ps, tar, "area", par, None)],
                            target_patch,
                            pred_area_events[par],
                        )

                    for topk, score in pred_topk_scores.items():
                        for th in args.pred_thresholds:
                            update_conf(
                                confs[(ps, tar, "topk", th, topk)],
                                target_patch,
                                (score >= th).long(),
                            )

    print("\n==================== Regression ====================")
    print(f"MAE:  {mae_sum / n_pix:.6f}")
    print(f"RMSE: {(rmse_sum / n_pix) ** 0.5:.6f}")

    rows = []
    for key, conf in confs.items():
        ps, tar, mode, th_or_area, topk = key
        p, r, f1, acc, tn, fp, fn, tp = metrics_from_conf(conf)
        rows.append({
            "patch": ps,
            "target_area": tar,
            "mode": mode,
            "pred_threshold_or_area": th_or_area,
            "topk": topk if topk is not None else -1,
            "precision": p,
            "recall": r,
            "f1": f1,
            "accuracy": acc,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

    rows = sorted(rows, key=lambda d: d["f1"], reverse=True)

    print("\n==================== Top patch-event settings by F1 ====================")
    print(
        f"{'rank':>4} {'patch':>6} {'tar_area':>9} {'mode':>8} "
        f"{'pred_th/area':>12} {'topk':>8} {'prec':>10} {'recall':>10} {'f1':>10} {'acc':>10} {'tp':>8} {'fp':>8} {'fn':>8}"
    )
    for rank, row in enumerate(rows[:50], start=1):
        topk_str = "-" if row["topk"] < 0 else f"{row['topk']:.2f}"
        print(
            f"{rank:4d} {row['patch']:6d} {row['target_area']:9.4f} {row['mode']:>8s} "
            f"{row['pred_threshold_or_area']:12.4f} {topk_str:>8s} "
            f"{row['precision']:10.6f} {row['recall']:10.6f} {row['f1']:10.6f} {row['accuracy']:10.6f} "
            f"{row['tp']:8d} {row['fp']:8d} {row['fn']:8d}"
        )

    out_dir = PROJECT_ROOT / "evaluation_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "patch_event_sweep_results.csv"

    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\nSaved full sweep results to: {out_path}")
    except Exception as e:
        print(f"\nCould not save CSV: {e}")

    dataset.close()


if __name__ == "__main__":
    main()
