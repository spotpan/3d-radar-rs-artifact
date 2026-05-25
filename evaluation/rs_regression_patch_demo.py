import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib import gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.finetune_guangdong_config import get_config
from training.finetune_guangdong import create_model, create_data_loaders


# -----------------------------
# Utilities
# -----------------------------
def strip_module_prefix(state_dict):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def to_2d(x):
    """
    Convert tensor/array to 2D numpy safely.

    Accepts shapes like:
      (1,1,H,W), (B,1,H,W), (1,H,W), (H,W)

    If a non-singleton leading dimension exists, use the first slice.
    This avoids infinite loops when shape is e.g. (4,1,H,W).
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        while x.ndim > 2:
            if x.shape[0] == 1:
                x = x.squeeze(0)
            else:
                x = x[0]
        return x.float().numpy()

    x = np.asarray(x)
    while x.ndim > 2:
        if x.shape[0] == 1:
            x = np.squeeze(x, axis=0)
        else:
            x = x[0]
    return x


def robust_vmax(*arrays, q=99.5, minimum=1.0):
    vals = []
    for arr in arrays:
        a = np.asarray(arr)
        a = a[np.isfinite(a)]
        if a.size > 0:
            vals.append(a)
    if not vals:
        return minimum
    vals = np.concatenate(vals)
    vmax = float(np.percentile(vals, q))
    return max(vmax, minimum)


def patch_event_prob_from_preds(
    preds,
    patch_size=128,
    pred_threshold=0.0005,
):
    """
    preds: (N, 1, H, W), non-negative regression outputs.
    Returns:
      patch_prob: (1, 1, H_patch, W_patch)
      patch_event_each: (N, 1, H_patch, W_patch)
    """
    patch_max = F.max_pool2d(preds, kernel_size=patch_size, stride=patch_size)
    patch_event = (patch_max >= pred_threshold).float()
    patch_prob = patch_event.mean(dim=0, keepdim=True)
    return patch_prob, patch_event


def target_patch_event(
    target,
    patch_size=128,
    target_threshold=0.1,
    target_area_ratio=0.001,
):
    """
    target: (1,1,H,W)
    target patch has rain if fraction(target >= threshold) >= area_ratio.
    """
    target_bin = (target >= target_threshold).float()
    frac = F.avg_pool2d(target_bin, kernel_size=patch_size, stride=patch_size)
    return (frac >= target_area_ratio).float(), frac


def upsample_patch_map(patch_map, target_hw):
    """
    patch_map: (1,1,h,w)
    return: (H,W) nearest upsampled numpy map
    """
    up = F.interpolate(patch_map, size=target_hw, mode="nearest")
    return to_2d(up)


# -----------------------------
# Physical perturbations
# -----------------------------
def perturb_gaussian(x, sigma=0.03):
    """
    Intensity perturbation:
    simulates radar measurement noise / calibration uncertainty.
    """
    noise = torch.randn_like(x) * sigma
    return x + noise


def perturb_level_dropout(x, drop_prob=0.15):
    """
    Vertical-level perturbation:
    randomly suppresses one or more height layers.

    x shape: (B,T,C,H,W)
    """
    x2 = x.clone()
    b, t, c, h, w = x2.shape

    mask = torch.ones((b, t, c, 1, 1), device=x2.device, dtype=x2.dtype)
    drop = torch.rand((b, t, c, 1, 1), device=x2.device) < drop_prob
    mask = mask.masked_fill(drop, 0.0)

    return x2 * mask


def perturb_level_scaling(x, scale_std=0.10):
    """
    Vertical-level scaling:
    each height layer is slightly amplified or attenuated.
    """
    b, t, c, h, w = x.shape
    scale = 1.0 + torch.randn((b, t, c, 1, 1), device=x.device, dtype=x.dtype) * scale_std
    scale = torch.clamp(scale, 0.5, 1.5)
    return x * scale


def perturb_temporal_dropout(x, drop_prob=0.15):
    """
    Temporal-frame perturbation:
    randomly suppresses historical frames.
    """
    x2 = x.clone()
    b, t, c, h, w = x2.shape

    mask = torch.ones((b, t, 1, 1, 1), device=x2.device, dtype=x2.dtype)
    drop = torch.rand((b, t, 1, 1, 1), device=x2.device) < drop_prob
    mask = mask.masked_fill(drop, 0.0)

    return x2 * mask


def perturb_block_mask(x, block_size=64, num_blocks=4):
    """
    Spatial block perturbation:
    simulates local echo missing / occlusion / clutter removal.
    """
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


# -----------------------------
# Visualization
# -----------------------------
def draw_main_panel(
    target,
    baseline,
    rs_mean,
    rs_std,
    event_prob_up,
    instability_up,
    target_event_up,
    out_path,
    title,
):
    """
    High-level academic multipanel visualization.
    """
    target_np = np.clip(target, 0, None)
    baseline_np = np.clip(baseline, 0, None)
    mean_np = np.clip(rs_mean, 0, None)
    std_np = np.clip(rs_std, 0, None)

    rain_vmax = robust_vmax(target_np, baseline_np, mean_np, q=99.7, minimum=1.0)
    std_vmax = robust_vmax(std_np, q=99.5, minimum=0.1)

    fig = plt.figure(figsize=(18, 10), dpi=120)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.18, hspace=0.22)

    panels = [
        ("(a) Target precipitation", target_np, "rain"),
        ("(b) Baseline prediction", baseline_np, "rain"),
        ("(c) RS mean prediction", mean_np, "rain"),
        ("(d) RS standard deviation", std_np, "std"),
        ("(e) Patch rain-event probability", event_prob_up, "prob"),
        ("(f) Perturbation-sensitive regions", instability_up, "instability"),
    ]

    axes = []
    for i, (label, arr, kind) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        axes.append(ax)

        if kind == "rain":
            im = ax.imshow(
                arr,
                origin="upper",
                cmap="turbo",
                norm=PowerNorm(gamma=0.45, vmin=0, vmax=rain_vmax),
            )
            cb_label = "Rain rate (mm h$^{-1}$)"
        elif kind == "std":
            im = ax.imshow(
                arr,
                origin="upper",
                cmap="magma",
                norm=PowerNorm(gamma=0.55, vmin=0, vmax=std_vmax),
            )
            cb_label = "Std. of prediction"
        elif kind == "prob":
            im = ax.imshow(
                arr,
                origin="upper",
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            cb_label = "Event probability"
        else:
            im = ax.imshow(
                arr,
                origin="upper",
                cmap="inferno",
                vmin=0,
                vmax=1,
            )
            cb_label = "Instability score"

        # Target contour disabled for speed in first demo

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(cb_label, fontsize=8)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    footer = (
        "White contours denote target patch-level rain-event regions. "
        "Event probability is estimated under physically structured perturbations."
    )
    fig.text(0.5, 0.02, footer, ha="center", fontsize=10)

    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    # fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_event_only_panel(
    event_prob_up,
    instability_up,
    target_event_up,
    out_path,
    title,
):
    fig = plt.figure(figsize=(13, 5.2), dpi=120)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(event_prob_up, origin="upper", cmap="viridis", vmin=0, vmax=1)
    # ax1.contour(target_event_up, levels=[0.5], colors="white", linewidths=0.9)
    ax1.set_title("(a) Smoothed patch-event probability", fontsize=12, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cb1.set_label("P(event=1)", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(instability_up, origin="upper", cmap="inferno", vmin=0, vmax=1)
    # ax2.contour(target_event_up, levels=[0.5], colors="white", linewidths=0.9)
    ax2.set_title("(b) Boundary-sensitive instability", fontsize=12, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cb2.set_label("4p(1-p)", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    # fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/binary_regfirst_relu_cls01_5000t200v/binary_regfirst_relu_cls01_5000t200v/best_model.pt",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=32)

    parser.add_argument(
        "--perturbation",
        type=str,
        default="level_dropout",
        choices=[
            "gaussian",
            "level_dropout",
            "level_scaling",
            "temporal_dropout",
            "block_mask",
            "mixed",
        ],
    )

    parser.add_argument("--gaussian-sigma", type=float, default=0.03)
    parser.add_argument("--level-drop-prob", type=float, default=0.15)
    parser.add_argument("--level-scale-std", type=float, default=0.10)
    parser.add_argument("--temporal-drop-prob", type=float, default=0.15)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)

    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--target-rain-threshold", type=float, default=0.1)
    parser.add_argument("--target-area-ratio", type=float, default=0.001)
    parser.add_argument("--pred-event-threshold", type=float, default=0.0005)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="figures/rs_patch_demo")
    parser.add_argument("--no-plot", action="store_true", help="Only save arrays, skip matplotlib plotting")

    args = parser.parse_args()

    print("RS demo arguments:")
    print(f"  sample_index={args.sample_index}")
    print(f"  num_samples={args.num_samples}")
    print(f"  perturbation={args.perturbation}")
    print(f"  patch_size={args.patch_size}")
    print(f"  pred_event_threshold={args.pred_event_threshold}")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Fetch one sample
    batch = None
    for i, b in enumerate(val_loader):
        if i == args.sample_index:
            batch = b
            break

    if batch is None:
        raise RuntimeError(f"Could not fetch sample-index={args.sample_index}")

    x = batch["radar_sequence"].to(device)
    y = batch["rain"].to(device)

    with torch.no_grad():
        baseline = model(x)["reg_output"].clamp(min=0)
        if baseline.shape[0] != x.shape[0]:
            print(f"Warning: baseline batch dim {baseline.shape[0]} != input batch dim {x.shape[0]}; using first {x.shape[0]} sample(s).")
            baseline = baseline[:x.shape[0]]

    preds = []
    with torch.no_grad():
        for n in range(args.num_samples):
            if n % 4 == 0:
                print(f"RS sample {n}/{args.num_samples}")
            x_pert = apply_perturbation(x, args.perturbation, args)
            out = model(x_pert)
            pred = out["reg_output"].clamp(min=0)
            if pred.shape[0] != x.shape[0]:
                if n == 0:
                    print(f"Warning: pred batch dim {pred.shape[0]} != input batch dim {x.shape[0]}; using first {x.shape[0]} sample(s).")
                pred = pred[:x.shape[0]]
            preds.append(pred.detach())
        print(f"RS sample {args.num_samples}/{args.num_samples}")

    print('Concatenating RS predictions...')
    preds = torch.cat(preds, dim=0)  # (N,1,H,W)
    print('preds shape:', tuple(preds.shape))
    print('Computing RS mean/std...')
    rs_mean = preds.mean(dim=0, keepdim=True)
    rs_std = preds.std(dim=0, keepdim=True)

    print('Computing patch event probability...')
    patch_prob, patch_events = patch_event_prob_from_preds(
        preds,
        patch_size=args.patch_size,
        pred_threshold=args.pred_event_threshold,
    )

    target_patch, target_frac = target_patch_event(
        y,
        patch_size=args.patch_size,
        target_threshold=args.target_rain_threshold,
        target_area_ratio=args.target_area_ratio,
    )

    h, w = y.shape[-2], y.shape[-1]
    print('Upsampling patch maps...')
    event_prob_up = upsample_patch_map(patch_prob, (h, w))
    target_event_up = upsample_patch_map(target_patch, (h, w))

    print('Computing instability map...')
    instability = 4.0 * patch_prob * (1.0 - patch_prob)
    instability_up = upsample_patch_map(instability, (h, w))
    print('Patch maps upsampled:', event_prob_up.shape, target_event_up.shape, instability_up.shape)

    print('Converting tensors to numpy arrays...')
    target_np = to_2d(y)
    baseline_np = to_2d(baseline)
    mean_np = to_2d(rs_mean)
    std_np = to_2d(rs_std)

    print('Preparing plot title and output stem...')
    title = (
        f"Physics-guided RS on 3D Radar Volumes | "
        f"perturbation={args.perturbation}, N={args.num_samples}, "
        f"patch={args.patch_size}"
    )

    stem = (
        f"sample{args.sample_index:04d}_"
        f"{args.perturbation}_N{args.num_samples}_"
        f"patch{args.patch_size}"
    )

    print('Drawing main panel...')
    if not args.no_plot:
        draw_main_panel(
            target=target_np,
            baseline=baseline_np,
            rs_mean=mean_np,
            rs_std=std_np,
            event_prob_up=event_prob_up,
            instability_up=instability_up,
            target_event_up=target_event_up,
            out_path=out_dir / f"{stem}_main_panel",
            title=title,
        )

        print('Drawing event panel...')
        draw_event_only_panel(
            event_prob_up=event_prob_up,
            instability_up=instability_up,
            target_event_up=target_event_up,
            out_path=out_dir / f"{stem}_event_panel",
            title=title,
        )
    else:
        print('Skipping plotting; saving arrays only.')

    # Save arrays for later paper-quality replotting
    print('Saving arrays...')
    np.savez_compressed(
        out_dir / f"{stem}_arrays.npz",
        target=target_np,
        baseline=baseline_np,
        rs_mean=mean_np,
        rs_std=std_np,
        event_prob=event_prob_up,
        instability=instability_up,
        target_event=target_event_up,
        perturbation=args.perturbation,
        num_samples=args.num_samples,
        patch_size=args.patch_size,
        pred_event_threshold=args.pred_event_threshold,
        target_rain_threshold=args.target_rain_threshold,
        target_area_ratio=args.target_area_ratio,
    )

    print("\nSaved outputs:")
    print(f"  {out_dir / f'{stem}_main_panel.png'}")
    print(f"  {out_dir / f'{stem}_main_panel.pdf'}")
    print(f"  {out_dir / f'{stem}_event_panel.png'}")
    print(f"  {out_dir / f'{stem}_event_panel.pdf'}")
    print(f"  {out_dir / f'{stem}_arrays.npz'}")

    dataset.close()


if __name__ == "__main__":
    main()
