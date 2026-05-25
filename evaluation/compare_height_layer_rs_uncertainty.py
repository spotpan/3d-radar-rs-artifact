import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.finetune_guangdong_config import get_config
from training.finetune_guangdong import create_model, create_data_loaders


def strip_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    state = strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint. Missing={len(missing)}, Unexpected={len(unexpected)}", flush=True)
    return model


def get_reg_output(model, x):
    out = model(x)
    if isinstance(out, dict):
        pred = out["reg_output"]
    else:
        pred = out
    if pred.shape[0] != x.shape[0]:
        pred = pred[:x.shape[0]]
    return pred.clamp(min=0)


def get_mode_levels(mode):
    if mode.startswith("drop_L"):
        return [int(mode.replace("drop_L", ""))]
    if mode == "drop_low":
        return [0, 1]
    if mode == "drop_mid":
        return [2, 3]
    if mode == "drop_high":
        return [4, 5]
    raise ValueError(f"Unknown mode: {mode}")


def apply_height_dropout(x, mode, drop_prob):
    """
    x shape: (B, T, C, H, W)
    With probability drop_prob, set selected height layers to zero.
    This gives randomized samples for event probability and instability.
    """
    y = x.clone()
    levels = get_mode_levels(mode)

    if torch.rand((), device=x.device).item() < drop_prob:
        y[:, :, levels, :, :] = 0.0

    return y


def patch_score_map(pred, patch_size):
    """
    pred: torch.Tensor, shape (B, 1, H, W)
    return: np.ndarray, shape (B, R, C)
    Use floor division to match previous 5x7 maps for 700x900 with patch=128.
    """
    pred = pred.detach().cpu().float().numpy()
    B, _, H, W = pred.shape
    R = H // patch_size
    C = W // patch_size
    out = np.zeros((B, R, C), dtype=np.float32)

    for b in range(B):
        img = np.maximum(pred[b, 0], 0)
        for r in range(R):
            for c in range(C):
                y0, y1 = r * patch_size, (r + 1) * patch_size
                x0, x1 = c * patch_size, (c + 1) * patch_size
                out[b, r, c] = np.max(img[y0:y1, x0:x1])
    return out


def target_event_map(target, patch_size, rain_threshold, area_ratio):
    """
    target: torch.Tensor, shape (B, 1, H, W)
    return: np.ndarray, shape (B, R, C)
    """
    target = target.detach().cpu().float().numpy()
    B, _, H, W = target.shape
    R = H // patch_size
    C = W // patch_size
    out = np.zeros((B, R, C), dtype=np.int64)

    for b in range(B):
        img = target[b, 0]
        for r in range(R):
            for c in range(C):
                y0, y1 = r * patch_size, (r + 1) * patch_size
                x0, x1 = c * patch_size, (c + 1) * patch_size
                block = img[y0:y1, x0:x1]
                ratio = np.mean(block >= rain_threshold)
                out[b, r, c] = int(ratio >= area_ratio)
    return out


def boundary_mask(event_map):
    """
    event_map: np.ndarray, shape (B, R, C)
    A patch is boundary if at least one 8-neighbor has different target label.
    """
    B, R, C = event_map.shape
    bd = np.zeros_like(event_map, dtype=bool)

    for b in range(B):
        for r in range(R):
            for c in range(C):
                v = event_map[b, r, c]
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < R and 0 <= cc < C:
                            if event_map[b, rr, cc] != v:
                                bd[b, r, c] = True
                                break
                    if bd[b, r, c]:
                        break
    return bd


def update_conf(tp_fp_fn, pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp_fp_fn["tp"] += int(np.logical_and(pred, target).sum())
    tp_fp_fn["fp"] += int(np.logical_and(pred, ~target).sum())
    tp_fp_fn["fn"] += int(np.logical_and(~pred, target).sum())


def prf(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=[
            "drop_L0", "drop_L1", "drop_L2", "drop_L3", "drop_L4", "drop_L5",
            "drop_low", "drop_mid", "drop_high",
        ],
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--drop-prob", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--target-rain-threshold", type=float, default=0.1)
    parser.add_argument("--target-area-ratio", type=float, default=0.001)
    parser.add_argument("--pred-event-threshold", type=float, default=0.0005)
    parser.add_argument("--smooth-event-threshold", type=float, default=0.5)
    parser.add_argument("--high-u-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Height-layer RS comparison", flush=True)
    print(f"  modes: {args.modes}", flush=True)
    print(f"  num_samples: {args.num_samples}", flush=True)
    print(f"  drop_prob: {args.drop_prob}", flush=True)
    print(f"  max_batches: {args.max_batches}", flush=True)

    config = get_config()
    config["data"]["batch_size"] = args.batch_size
    config["data"]["num_workers"] = args.num_workers
    config["train"]["use_weighted_sampler"] = False

    print("Creating model...", flush=True)
    model = create_model(config)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()

    print("Creating data loaders...", flush=True)
    _, val_loader, dataset = create_data_loaders(config)

    stats = {}
    for mode in args.modes:
        stats[mode] = {
            "tp": 0, "fp": 0, "fn": 0,
            "U_sum": 0.0, "U_count": 0,
            "boundary_U_sum": 0.0, "boundary_U_count": 0,
            "high_U_count": 0,
            "score_std_sum": 0.0, "score_std_count": 0,
            "boundary_score_std_sum": 0.0, "boundary_score_std_count": 0,
        }

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            if batch_idx % 5 == 0:
                print(f"Processing validation sample {batch_idx}/{args.max_batches}", flush=True)

            x = batch["radar_sequence"].to(device).float()
            target = batch["rain"].to(device).float()

            target_event = target_event_map(
                target,
                patch_size=args.patch_size,
                rain_threshold=args.target_rain_threshold,
                area_ratio=args.target_area_ratio,
            )
            bd_mask = boundary_mask(target_event)

            for mode in args.modes:
                print(f"  mode={mode}", flush=True)
                scores = []

                for _ in range(args.num_samples):
                    xp = apply_height_dropout(x, mode, drop_prob=args.drop_prob)
                    pred = get_reg_output(model, xp)
                    scores.append(patch_score_map(pred, args.patch_size))

                scores = np.stack(scores, axis=0)  # N, B, R, C
                events = (scores >= args.pred_event_threshold).astype(np.float32)
                p_event = events.mean(axis=0)  # B, R, C
                pred_event = (p_event >= args.smooth_event_threshold).astype(np.int64)

                U = 4.0 * p_event * (1.0 - p_event)
                score_std = scores.std(axis=0)

                s = stats[mode]
                update_conf(s, pred_event, target_event)

                s["U_sum"] += float(U.sum())
                s["U_count"] += int(U.size)
                s["high_U_count"] += int((U >= args.high_u_threshold).sum())

                s["score_std_sum"] += float(score_std.sum())
                s["score_std_count"] += int(score_std.size)

                if bd_mask.any():
                    s["boundary_U_sum"] += float(U[bd_mask].sum())
                    s["boundary_U_count"] += int(bd_mask.sum())
                    s["boundary_score_std_sum"] += float(score_std[bd_mask].sum())
                    s["boundary_score_std_count"] += int(bd_mask.sum())

    rows = []
    for mode in args.modes:
        s = stats[mode]
        precision, recall, f1 = prf(s["tp"], s["fp"], s["fn"])
        rows.append({
            "mode": mode,
            "drop_levels": str(get_mode_levels(mode)),
            "drop_prob": args.drop_prob,
            "rs_f1": f1,
            "rs_precision": precision,
            "rs_recall": recall,
            "mean_uncertainty_U": s["U_sum"] / max(s["U_count"], 1),
            "boundary_U": s["boundary_U_sum"] / max(s["boundary_U_count"], 1),
            "high_U_area_ratio": s["high_U_count"] / max(s["U_count"], 1),
            "mean_score_std": s["score_std_sum"] / max(s["score_std_count"], 1),
            "boundary_score_std": s["boundary_score_std_sum"] / max(s["boundary_score_std_count"], 1),
        })

    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n==================== Height-layer RS uncertainty comparison ====================")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved CSV: {out.resolve()}", flush=True)

    if hasattr(dataset, "close"):
        dataset.close()


if __name__ == "__main__":
    main()
