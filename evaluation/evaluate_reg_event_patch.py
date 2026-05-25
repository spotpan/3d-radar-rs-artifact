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

    return precision, recall, f1, accuracy


def patch_event_map(x, patch_size, mode="mean", threshold=0.1, area_ratio=0.01):
    """
    x: (B, 1, H, W)

    target event:
      target patch has rain if fraction of pixels >= threshold exceeds area_ratio

    pred event:
      if mode == mean: patch mean >= threshold
      if mode == max: patch max >= threshold
      if mode == area: fraction of pixels >= threshold exceeds area_ratio
    """
    if mode == "mean":
        pooled = F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)
        return pooled >= threshold

    if mode == "max":
        pooled = F.max_pool2d(x, kernel_size=patch_size, stride=patch_size)
        return pooled >= threshold

    if mode == "area":
        bin_map = (x >= threshold).float()
        frac = F.avg_pool2d(bin_map, kernel_size=patch_size, stride=patch_size)
        return frac >= area_ratio

    raise ValueError(f"Unknown mode: {mode}")


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
    parser.add_argument("--rain-thresholds", type=float, nargs="+",
                        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--target-rain-threshold", type=float, default=0.1)
    parser.add_argument("--target-area-ratio", type=float, default=0.01)
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

    pixel_confs = {
        float(th): np.zeros((2, 2), dtype=np.int64)
        for th in args.rain_thresholds
    }

    patch_confs = {}
    for ps in args.patch_sizes:
        for mode in ["mean", "max", "area"]:
            patch_confs[(ps, mode)] = np.zeros((2, 2), dtype=np.int64)

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

            target_pixel = (y >= args.target_rain_threshold).long()

            for th, conf in pixel_confs.items():
                pred_pixel = (pred >= th).long()
                update_conf(conf, target_pixel, pred_pixel)

            for ps in args.patch_sizes:
                target_patch = patch_event_map(
                    y,
                    patch_size=ps,
                    mode="area",
                    threshold=args.target_rain_threshold,
                    area_ratio=args.target_area_ratio,
                ).long()

                for mode in ["mean", "max", "area"]:
                    pred_patch = patch_event_map(
                        pred,
                        patch_size=ps,
                        mode=mode,
                        threshold=args.target_rain_threshold,
                        area_ratio=args.target_area_ratio,
                    ).long()
                    update_conf(patch_confs[(ps, mode)], target_patch, pred_patch)

    print("\n==================== Regression ====================")
    print(f"MAE:  {mae_sum / n_pix:.6f}")
    print(f"RMSE: {(rmse_sum / n_pix) ** 0.5:.6f}")

    print("\n==================== Pixel-level reg_output threshold sweep ====================")
    print("threshold".rjust(10), "precision".rjust(12), "recall".rjust(12), "f1".rjust(12), "accuracy".rjust(12))
    for th in args.rain_thresholds:
        p, r, f1, acc = metrics_from_conf(pixel_confs[float(th)])
        print(f"{th:10.4f} {p:12.6f} {r:12.6f} {f1:12.6f} {acc:12.6f}")

    print("\n==================== Patch-level event evaluation ====================")
    print("patch".rjust(8), "mode".rjust(8), "precision".rjust(12), "recall".rjust(12), "f1".rjust(12), "accuracy".rjust(12))
    for ps in args.patch_sizes:
        for mode in ["mean", "max", "area"]:
            p, r, f1, acc = metrics_from_conf(patch_confs[(ps, mode)])
            print(f"{ps:8d} {mode:>8s} {p:12.6f} {r:12.6f} {f1:12.6f} {acc:12.6f}")

    dataset.close()


if __name__ == "__main__":
    main()
