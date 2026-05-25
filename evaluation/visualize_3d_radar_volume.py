import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import PowerNorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.finetune_guangdong_config import get_config
from training.finetune_guangdong import create_data_loaders


def robust_vmax(*arrays, q=99.5, minimum=1.0):
    vals = []
    for arr in arrays:
        a = np.asarray(arr)
        a = a[np.isfinite(a)]
        if a.size > 0:
            vals.append(a.reshape(-1))
    if not vals:
        return minimum
    vals = np.concatenate(vals)
    return max(float(np.percentile(vals, q)), minimum)


def get_sample(val_loader, sample_index):
    for i, batch in enumerate(val_loader):
        if i == sample_index:
            return batch
    raise RuntimeError(f"sample_index={sample_index} not found")


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def prepare_sequence(x):
    """
    Expected radar_sequence shapes:
      (B,T,C,H,W)
    Fallback:
      (B,C,H,W) -> add T dimension
    """
    x = to_numpy(x)

    if x.ndim == 5:
        # B,T,C,H,W
        return x[0]

    if x.ndim == 4:
        # B,C,H,W
        return x[0][None, ...]

    raise ValueError(f"Unsupported radar_sequence shape: {x.shape}")


def prepare_target(y):
    y = to_numpy(y)
    while y.ndim > 2:
        y = y[0]
    return y


def draw_layer_montage(volume, target, out_path, title):
    """
    volume: (C,H,W), one time frame.
    """
    c, h, w = volume.shape
    ncols = 3
    nrows = int(np.ceil(c / ncols))

    vmax = robust_vmax(volume, q=99.7, minimum=1.0)

    fig = plt.figure(figsize=(14, 4.2 * nrows), dpi=180)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.18)

    for k in range(c):
        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        im = ax.imshow(
            volume[k],
            origin="upper",
            cmap="turbo",
            norm=PowerNorm(gamma=0.55, vmin=0, vmax=vmax),
        )

        try:
            ax.contour(target >= 0.1, levels=[0.5], colors="white", linewidths=0.5, alpha=0.8)
        except Exception:
            pass

        ax.set_title(f"Height layer {k}", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty panels
    for k in range(c, nrows * ncols):
        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        ax.axis("off")

    cax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Radar echo intensity", fontsize=10)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def draw_vertical_sections(volume, target, out_path, title):
    """
    Draw vertical cross-sections through the location of maximum target rain.
    volume: (C,H,W)
    target: (H,W)
    """
    c, h, w = volume.shape

    if np.nanmax(target) > 0:
        y0, x0 = np.unravel_index(np.nanargmax(target), target.shape)
    else:
        y0, x0 = h // 2, w // 2

    # Window-averaged vertical sections for smoother physical profiles
    win = 12
    x1 = max(0, x0 - win)
    x2 = min(w, x0 + win + 1)
    y1 = max(0, y0 - win)
    y2 = min(h, y0 + win + 1)

    # z-y section averaged over a local x-window
    zy = volume[:, :, x1:x2].mean(axis=2)  # C,H

    # z-x section averaged over a local y-window
    zx = volume[:, y1:y2, :].mean(axis=1)  # C,W

    vmax = robust_vmax(zy, zx, q=99.5, minimum=1.0)

    fig = plt.figure(figsize=(14, 6), dpi=180)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        zy,
        origin="lower",
        aspect="auto",
        cmap="turbo",
        norm=PowerNorm(gamma=0.55, vmin=0, vmax=vmax),
    )
    ax1.set_title(f"Vertical section Z-Y at x={x0}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Y coordinate")
    ax1.set_ylabel("Height layer index")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        zx,
        origin="lower",
        aspect="auto",
        cmap="turbo",
        norm=PowerNorm(gamma=0.55, vmin=0, vmax=vmax),
    )
    ax2.set_title(f"Vertical section Z-X at y={y0}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Height layer index")

    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    cb = fig.colorbar(im2, cax=cax)
    cb.set_label("Radar echo intensity", fontsize=10)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def draw_3d_echo_cloud(volume, target, out_path, title, stride=8, quantile=99.5, max_points=60000):
    """
    Sparse 3D point cloud of high radar echo values.
    volume: (C,H,W)
    """
    c, h, w = volume.shape

    vol = volume[:, ::stride, ::stride]
    thr = np.percentile(vol[np.isfinite(vol)], quantile)

    zz, yy, xx = np.where(vol >= thr)
    vals = vol[zz, yy, xx]

    if vals.size > max_points:
        idx = np.random.choice(vals.size, size=max_points, replace=False)
        zz, yy, xx, vals = zz[idx], yy[idx], xx[idx], vals[idx]

    xx_full = xx * stride
    yy_full = yy * stride

    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        xx_full,
        yy_full,
        zz,
        c=vals,
        cmap="turbo",
        s=4,
        alpha=0.75,
        depthshade=False,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_zlabel("Height layer")
    ax.view_init(elev=28, azim=-58)

    cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08)
    cb.set_label("Radar echo intensity")

    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def draw_time_height_signature(seq, out_path, title):
    """
    seq: (T,C,H,W)
    Show max and mean echo over space for each time-height pair.
    """
    max_sig = np.nanmax(seq, axis=(-1, -2))   # T,C
    mean_sig = np.nanmean(seq, axis=(-1, -2)) # T,C

    vmax = robust_vmax(max_sig, q=99.5, minimum=1.0)

    fig = plt.figure(figsize=(12, 5), dpi=180)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(max_sig.T, origin="lower", aspect="auto", cmap="turbo", vmin=0, vmax=vmax)
    ax1.set_title("(a) Spatial maximum echo", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Historical frame")
    ax1.set_ylabel("Height layer")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(mean_sig.T, origin="lower", aspect="auto", cmap="viridis")
    ax2.set_title("(b) Spatial mean echo", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Historical frame")
    ax2.set_ylabel("Height layer")

    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cb1.set_label("Max echo")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cb2.set_label("Mean echo")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=int, default=100)
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--quantile", type=float, default=99.5)
    parser.add_argument("--out-dir", type=str, default="figures/rs_patch_demo/volume3d")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = get_config()
    config["data"]["batch_size"] = args.batch_size
    config["data"]["num_workers"] = args.num_workers
    config["train"]["use_weighted_sampler"] = False

    _, val_loader, dataset = create_data_loaders(config)
    batch = get_sample(val_loader, args.sample_index)

    seq = prepare_sequence(batch["radar_sequence"])  # T,C,H,W
    target = prepare_target(batch["rain"])           # H,W

    t = args.frame_index
    volume = seq[t]  # C,H,W

    print("radar_sequence shape:", seq.shape)
    print("selected volume shape:", volume.shape)
    print("target shape:", target.shape)
    print("target max/mean:", float(np.max(target)), float(np.mean(target)))
    print("volume max/mean:", float(np.max(volume)), float(np.mean(volume)))

    stem = f"sample{args.sample_index:04d}_frame{t}_3d"

    print('Drawing height-layer montage...')
    draw_layer_montage(
        volume,
        target,
        out_dir / f"{stem}_height_layers.png",
        title=f"3D radar volume height-layer montage | sample={args.sample_index}, frame={t}",
    )

    print('Drawing vertical sections...')
    draw_vertical_sections(
        volume,
        target,
        out_dir / f"{stem}_vertical_sections.png",
        title=f"Vertical structure through precipitation maximum | sample={args.sample_index}",
    )

    print('Drawing 3D echo cloud...')
    draw_3d_echo_cloud(
        volume,
        target,
        out_dir / f"{stem}_echo_cloud.png",
        title=f"3D sparse radar echo cloud | sample={args.sample_index}, frame={t}",
        stride=args.stride,
        quantile=args.quantile,
    )

    print('Drawing time-height signature...')
    draw_time_height_signature(
        seq,
        out_dir / f"{stem}_time_height_signature.png",
        title=f"Temporal-height radar signature | sample={args.sample_index}",
    )

    print('All figures generated.')
    print("\nSaved 3D radar visualizations:")
    print(out_dir / f"{stem}_height_layers.png")
    print(out_dir / f"{stem}_vertical_sections.png")
    print(out_dir / f"{stem}_echo_cloud.png")
    print(out_dir / f"{stem}_time_height_signature.png")

    dataset.close()


if __name__ == "__main__":
    main()
