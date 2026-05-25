import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# Standalone config shim for this framework-figure script only.
# We do NOT create or modify configs/finetune_guangdong_config.py.
# Instead, we register an in-memory module so that
# training/finetune_guangdong.py can import it safely.
# ---------------------------------------------------------------------
import types

def get_config():
    config = {}

    config["model"] = {
        "mae_checkpoint": "/path/to/mae_checkpoint.pt",
        "num_patches_h": 14,
        "num_patches_w": 18,
        "num_frames": 6,
        "temporal_dim": 768,
        "temporal_depth": 2,
        "temporal_num_heads": 8,
        # Match the checkpoint head shape: reg/cls heads expect 32 channels.
        "decoder_channels": [512, 256, 128, 64, 32],
        "use_skip_connections": True,
        "final_activation_reg": "relu",
        "final_activation_cls": "sigmoid",
    }

    config["data"] = {
        "data_paths": [
            "/path/to/radar_station_dataset/time_radar_rain_2022.h5",
            "/path/to/radar_station_dataset/time_radar_rain_2023.h5",
        ],
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": True,
        "history_frames": 6,
        "frame_interval": 12,
        "target_minutes": 0,
        "radar_height_layers": [0, 1, 2, 3, 4, 5],
        "spatial_size": (700, 900),
    }

    config["train"] = {
        "num_epochs": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 0,
        "scheduler": "cosine",
        "val_split": 0.1,
        "val_batch_size": 1,
        "use_weighted_sampler": False,
    }

    config["loss"] = {
        "lambda_global": 1.0,
        "lambda_point": 0.0,
        "lambda_cls": 0.1,
        "lambda_multi_cls": 0.0,
        "rain_threshold": 0.1,
        "quantile": 0.9,
    }

    return config


def load_station_coords(*args, **kwargs):
    return None


_cfg_mod = types.ModuleType("configs.finetune_guangdong_config")
_cfg_mod.get_config = get_config
_cfg_mod.load_station_coords = load_station_coords
sys.modules["configs.finetune_guangdong_config"] = _cfg_mod

from training.finetune_guangdong import create_model, create_data_loaders


# ---------------------------
# Basic helpers
# ---------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _first_existing(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def unpack_batch(batch):
    """Robustly unpack different dataloader return formats."""
    if isinstance(batch, dict):
        x = _first_existing(batch, ["radar_sequence", "radar", "input", "x"])
        y = _first_existing(batch, ["target", "rain", "y", "target_grid"])
        if x is None or y is None:
            raise KeyError(f"Cannot find x/y in batch keys: {list(batch.keys())}")
        return x, y

    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch tuple/list has fewer than 2 elements.")
        return batch[0], batch[1]

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def get_selected_sample(val_loader, sample_index: int):
    """Fetch one validation sample by index."""
    for i, batch in enumerate(val_loader):
        if i == sample_index:
            x, y = unpack_batch(batch)
            return x, y
    raise IndexError(f"sample_index={sample_index} exceeds validation loader length.")


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}", flush=True)


def model_predict_reg(model, x):
    """Return non-negative regression output with batch dimension aligned."""
    out = model(x)
    if isinstance(out, dict):
        pred = _first_existing(out, ["reg_output", "retrieval", "prediction", "pred"])
    else:
        pred = out

    if pred is None:
        raise ValueError("Cannot find regression output from model.")

    if pred.shape[0] != x.shape[0]:
        print(f"Warning: pred batch dim {pred.shape[0]}; using pred[:{x.shape[0]}].", flush=True)
        pred = pred[: x.shape[0]]

    return pred.clamp(min=0)


def to_numpy_2d(a):
    """Convert tensor/array to 2D HxW."""
    if torch.is_tensor(a):
        a = a.detach().cpu().float().numpy()
    a = np.asarray(a)

    # Remove batch/channel dimensions robustly
    while a.ndim > 2:
        a = a[0]
    return a


def robust_vmax(*arrays, q=99.5, minimum=1.0):
    vals = []
    for a in arrays:
        aa = np.asarray(a)
        aa = aa[np.isfinite(aa)]
        if aa.size:
            vals.append(aa.reshape(-1))
    if not vals:
        return minimum
    vals = np.concatenate(vals)
    vmax = float(np.percentile(vals, q))
    return max(vmax, minimum)


# ---------------------------
# Perturbations
# ---------------------------

def perturb_gaussian(x, rng, std=0.05):
    # std is relative to input dynamic range
    xr = x.detach()
    scale = torch.nan_to_num(xr.std(), nan=torch.tensor(1.0, device=x.device)).item()
    noise = torch.randn_like(x) * (std * max(scale, 1e-6))
    return x + noise


def perturb_level_scaling(x, rng, sigma=0.12, min_scale=0.65, max_scale=1.35):
    # x: B,T,C,H,W
    if x.ndim != 5:
        return x
    B, T, C, H, W = x.shape
    scales = torch.empty((B, 1, C, 1, 1), device=x.device).normal_(mean=1.0, std=sigma)
    scales = scales.clamp(min_scale, max_scale)
    return x * scales


def perturb_level_dropout(x, rng, drop_prob=0.15):
    if x.ndim != 5:
        return x
    B, T, C, H, W = x.shape
    mask = torch.ones((B, 1, C, 1, 1), device=x.device)
    rand = torch.rand((B, 1, C, 1, 1), device=x.device)
    mask = (rand > drop_prob).float()
    # avoid all-zero levels
    if mask.sum() == 0:
        mask[:, :, rng.integers(0, C), :, :] = 1
    return x * mask


def perturb_temporal_dropout(x, rng, drop_prob=0.15):
    if x.ndim != 5:
        return x
    B, T, C, H, W = x.shape
    mask = torch.ones((B, T, 1, 1, 1), device=x.device)
    rand = torch.rand((B, T, 1, 1, 1), device=x.device)
    mask = (rand > drop_prob).float()
    # keep the latest frame to avoid fully destroying the case
    mask[:, -1] = 1
    return x * mask


def perturb_block_mask(x, rng, block_ratio=0.18):
    if x.ndim != 5:
        return x
    B, T, C, H, W = x.shape
    y = x.clone()
    bh = max(8, int(H * block_ratio))
    bw = max(8, int(W * block_ratio))
    top = int(rng.integers(0, max(1, H - bh)))
    left = int(rng.integers(0, max(1, W - bw)))
    y[:, :, :, top:top + bh, left:left + bw] = 0
    return y


def apply_perturbation(x, mode, rng):
    if mode == "gaussian":
        return perturb_gaussian(x, rng)
    if mode == "level_scaling":
        return perturb_level_scaling(x, rng)
    if mode == "level_dropout":
        return perturb_level_dropout(x, rng)
    if mode == "temporal_dropout":
        return perturb_temporal_dropout(x, rng)
    if mode == "block_mask":
        return perturb_block_mask(x, rng)
    if mode == "mixed":
        y = perturb_gaussian(x, rng, std=0.035)
        y = perturb_level_scaling(y, rng, sigma=0.10)
        y = perturb_level_dropout(y, rng, drop_prob=0.10)
        y = perturb_block_mask(y, rng, block_ratio=0.12)
        return y
    raise ValueError(f"Unknown perturbation mode: {mode}")


# ---------------------------
# Patch maps
# ---------------------------

def patch_score_map(pred2d, patch_size):
    H, W = pred2d.shape
    nr = int(np.ceil(H / patch_size))
    nc = int(np.ceil(W / patch_size))
    score = np.zeros((nr, nc), dtype=np.float32)
    for r in range(nr):
        for c in range(nc):
            yy0, yy1 = r * patch_size, min(H, (r + 1) * patch_size)
            xx0, xx1 = c * patch_size, min(W, (c + 1) * patch_size)
            score[r, c] = float(np.nanmax(np.maximum(pred2d[yy0:yy1, xx0:xx1], 0)))
    return score


def target_event_map(target2d, patch_size, rain_threshold, area_ratio):
    H, W = target2d.shape
    nr = int(np.ceil(H / patch_size))
    nc = int(np.ceil(W / patch_size))
    event = np.zeros((nr, nc), dtype=np.float32)
    for r in range(nr):
        for c in range(nc):
            yy0, yy1 = r * patch_size, min(H, (r + 1) * patch_size)
            xx0, xx1 = c * patch_size, min(W, (c + 1) * patch_size)
            block = target2d[yy0:yy1, xx0:xx1]
            ratio = np.mean(block >= rain_threshold)
            event[r, c] = 1.0 if ratio >= area_ratio else 0.0
    return event


def compute_rs_maps(model, x, rng, mode, num_samples, patch_size, pred_event_threshold):
    scores = []
    preds_for_mean = []

    with torch.no_grad():
        for k in range(num_samples):
            if k % max(1, num_samples // 4) == 0:
                print(f"RS sample {k}/{num_samples}", flush=True)
            xp = apply_perturbation(x, mode, rng)
            pred = model_predict_reg(model, xp)
            pred2d = to_numpy_2d(pred)
            preds_for_mean.append(pred2d)
            scores.append(patch_score_map(pred2d, patch_size))
        print(f"RS sample {num_samples}/{num_samples}", flush=True)

    scores = np.stack(scores, axis=0)  # N, R, C
    event_prob = (scores >= pred_event_threshold).mean(axis=0)
    instability = 4.0 * event_prob * (1.0 - event_prob)
    score_std = scores.std(axis=0)
    pred_mean = np.mean(np.stack(preds_for_mean, axis=0), axis=0)
    return event_prob, instability, score_std, pred_mean


# ---------------------------
# Plot helpers
# ---------------------------

def add_panel_box(fig, xywh, title, edgecolor="#4C78A8", facecolor="#F8FBFF"):
    x, y, w, h = xywh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=fig.transFigure,
        zorder=0
    )
    fig.add_artist(patch)
    fig.text(x + w/2, y + h - 0.018, title,
             ha="center", va="top", fontsize=10.5, fontweight="bold", color=edgecolor)
    return patch


def add_arrow(fig, p1, p2, color="#333333", lw=1.5):
    arr = FancyArrowPatch(
        p1, p2, transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, zorder=5
    )
    fig.add_artist(arr)


def add_text_box(fig, xywh, title, lines, edgecolor="#666666", facecolor="#FFFFFF", fontsize=8.2):
    add_panel_box(fig, xywh, title, edgecolor=edgecolor, facecolor=facecolor)
    x, y, w, h = xywh
    txt = "\n".join(lines)
    fig.text(x + 0.015, y + h - 0.055, txt,
             ha="left", va="top", fontsize=fontsize, color="#111111", linespacing=1.25)


def draw_3d_volume(ax, vol, title="", stride=18, q=99.3, cmap_name="turbo",
                   alpha=0.80, selected_levels=None, mask_block=None, drop_level=None):
    """
    Draw stacked horizontal slices from a 3D volume C,H,W.
    """
    C, H, W = vol.shape
    if selected_levels is None:
        selected_levels = [0, C // 2, C - 1] if C >= 3 else list(range(C))

    vmax = robust_vmax(vol, q=q, minimum=1.0)
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    yy = np.arange(0, H, stride)
    xx = np.arange(0, W, stride)
    Xg, Yg = np.meshgrid(xx, yy)

    for idx, c in enumerate(selected_levels):
        layer = np.maximum(vol[c], 0).copy()

        if mask_block is not None:
            y0, y1, x0, x1 = mask_block
            layer[y0:y1, x0:x1] = np.nan

        Zg = np.full_like(Xg, idx, dtype=float)
        vals = layer[::stride, ::stride]
        colors = cmap(norm(np.nan_to_num(vals, nan=0.0)))
        if drop_level is not None and c == drop_level:
            colors[..., :3] = 0.82
            colors[..., 3] = 0.25
        else:
            colors[..., 3] = alpha

        ax.plot_surface(
            Xg, Yg, Zg,
            facecolors=colors,
            rstride=1, cstride=1,
            linewidth=0, antialiased=False, shade=False
        )

    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(-0.1, len(selected_levels) - 0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((W, H, 0.45 * max(len(selected_levels), 1)))
    ax.grid(False)


def add_patch_grid(ax, patch_map, color="white", lw=0.8):
    nr, nc = patch_map.shape
    for r in range(nr + 1):
        ax.axhline(r - 0.5, color=color, linewidth=lw, alpha=0.65)
    for c in range(nc + 1):
        ax.axvline(c - 0.5, color=color, linewidth=lw, alpha=0.65)


def draw_small_map(ax, data, title, cmap="turbo", vmin=None, vmax=None, grid=False):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    if grid:
        add_patch_grid(ax, data, color="white", lw=0.8)
    return im


# ---------------------------
# Main figure
# ---------------------------

def make_framework_figure(
    x,
    target,
    baseline_pred,
    rs_event_prob,
    rs_instability,
    pred_mean,
    out_path_pdf,
    out_path_png,
    patch_size=128,
    title="Physics-guided RS for 3D Radar Precipitation Event Uncertainty Diagnosis",
):
    # Convert data
    x_np = x.detach().cpu().float().numpy()
    # Expected B,T,C,H,W
    if x_np.ndim == 5:
        vol = x_np[0, -1]  # latest frame, C,H,W
    elif x_np.ndim == 4:
        vol = x_np[-1]
    else:
        raise ValueError(f"Unexpected x shape: {x_np.shape}")

    target2d = to_numpy_2d(target)
    base2d = to_numpy_2d(baseline_pred)
    pred_mean2d = pred_mean

    H, W = target2d.shape
    vmax_radar = robust_vmax(vol, q=99.5, minimum=1.0)
    vmax_rain = robust_vmax(target2d, base2d, pred_mean2d, q=99.5, minimum=1.0)

    base_patch = patch_score_map(base2d, patch_size)

    fig = plt.figure(figsize=(18, 10.2), dpi=220)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.985)

    # Panel positions
    p_input = (0.02, 0.55, 0.18, 0.36)
    p_backbone = (0.215, 0.55, 0.34, 0.36)
    p_event = (0.57, 0.55, 0.14, 0.36)
    p_rs = (0.725, 0.55, 0.12, 0.36)
    p_out = (0.86, 0.55, 0.12, 0.36)

    p_pert = (0.02, 0.31, 0.76, 0.19)
    p_metrics = (0.80, 0.31, 0.18, 0.19)
    p_products = (0.02, 0.055, 0.58, 0.20)
    p_insights = (0.62, 0.055, 0.36, 0.20)

    # Boxes
    add_panel_box(fig, p_input, "1. 3D Radar Volume Input", "#2C5AA0", "#F8FBFF")
    add_panel_box(fig, p_backbone, "2. 3DMAEPP Backbone", "#2E7D32", "#F7FFF7")
    add_panel_box(fig, p_event, "3. Patch-level Event Definition", "#2C5AA0", "#F8FBFF")
    add_panel_box(fig, p_rs, "4. Physics-guided RS", "#D2691E", "#FFF9F2")
    add_panel_box(fig, p_out, "5. Smoothed Event & Uncertainty Outputs", "#5E35B1", "#FBF8FF")
    add_panel_box(fig, p_pert, "Physics-guided Local Perturbation Families", "#4C4C4C", "#FBFBFB")
    add_panel_box(fig, p_metrics, "Physical Meaning", "#4C4C4C", "#FFFFFF")
    add_panel_box(fig, p_products, "RS Spatial Products from Real Validation Sample", "#5E35B1", "#FBF8FF")
    add_panel_box(fig, p_insights, "Quantitative Diagnosis", "#5E35B1", "#FBF8FF")

    # Arrows top
    ymid = 0.73
    add_arrow(fig, (p_input[0] + p_input[2] + 0.006, ymid), (p_backbone[0] - 0.006, ymid))
    add_arrow(fig, (p_backbone[0] + p_backbone[2] + 0.006, ymid), (p_event[0] - 0.006, ymid))
    add_arrow(fig, (p_event[0] + p_event[2] + 0.006, ymid), (p_rs[0] - 0.006, ymid))
    add_arrow(fig, (p_rs[0] + p_rs[2] + 0.006, ymid), (p_out[0] - 0.006, ymid))

    # Input 3D volume
    ax_vol = fig.add_axes((p_input[0] + 0.02, p_input[1] + 0.055, p_input[2] - 0.04, p_input[3] - 0.11), projection="3d")
    draw_3d_volume(ax_vol, vol, title="Multi-level radar reflectivity", stride=20, q=99.2, alpha=0.80)
    fig.text(p_input[0] + p_input[2]/2, p_input[1] + 0.035,
             "multi-frame sequence  |  vertical layers  |  reflectivity (dBZ)",
             ha="center", va="center", fontsize=7.2)

    # Backbone internal mini-blocks
    x0, y0, w0, h0 = p_backbone
    bx_w = w0 / 3.15
    sub_y = y0 + 0.08
    sub_h = h0 - 0.14
    sub_xs = [x0 + 0.018, x0 + 0.018 + bx_w + 0.012, x0 + 0.018 + 2*(bx_w + 0.012)]
    sub_titles = [
        "2.1 3DMAE pretraining\nrandom mask $\\rightarrow$ reconstruction",
        "2.2 temporal fusion\npoint-wise Transformer",
        "2.3 dual-head decoder\nretrieval + rain probability",
    ]
    for sx, st in zip(sub_xs, sub_titles):
        rect = Rectangle((sx, sub_y), bx_w, sub_h, transform=fig.transFigure,
                         facecolor="white", edgecolor="#A5D6A7", linewidth=1.0)
        fig.add_artist(rect)
        fig.text(sx + bx_w/2, sub_y + sub_h - 0.015, st,
                 ha="center", va="top", fontsize=7.8, fontweight="bold", color="#1B5E20")

    # Backbone visual mini maps
    # 3DMAE mask thumbnail
    ax_m = fig.add_axes((sub_xs[0] + 0.02, sub_y + 0.08, bx_w - 0.04, sub_h - 0.16))
    mid_layer = np.maximum(vol[min(vol.shape[0]-1, vol.shape[0]//2)], 0)
    thumb = mid_layer.copy()
    # overlay random gray masked blocks
    ax_m.imshow(thumb, cmap="turbo", vmin=0, vmax=vmax_radar, aspect="auto")
    rng_local = np.random.default_rng(3)
    for _ in range(10):
        yy = rng_local.integers(0, max(1, H-90))
        xx = rng_local.integers(0, max(1, W-90))
        ax_m.add_patch(Rectangle((xx, yy), 80, 80, color="lightgray", alpha=0.72))
    ax_m.set_xticks([]); ax_m.set_yticks([])

    # Temporal fusion schematic
    ax_t = fig.add_axes((sub_xs[1] + 0.018, sub_y + 0.08, bx_w - 0.036, sub_h - 0.16))
    ax_t.axis("off")
    for i in range(4):
        ax_t.add_patch(Rectangle((0.1 + i*0.12, 0.60), 0.07, 0.12, facecolor="#DCEBFF", edgecolor="#4C78A8"))
        ax_t.add_patch(Rectangle((0.1 + i*0.12, 0.38), 0.07, 0.12, facecolor="#DCEBFF", edgecolor="#4C78A8"))
        ax_t.plot([0.135 + i*0.12, 0.135 + i*0.12], [0.50, 0.60], color="#555", lw=1)
    ax_t.text(0.50, 0.25, "temporal attention\n$f_{t-T+1},...,f_t$", ha="center", va="center", fontsize=8)
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)

    # Decoder thumbnails
    ax_d1 = fig.add_axes((sub_xs[2] + 0.015, sub_y + 0.16, (bx_w - 0.045)/2, sub_h - 0.23))
    draw_small_map(ax_d1, base2d, "retrieval", cmap="turbo", vmin=0, vmax=vmax_rain)
    ax_d2 = fig.add_axes((sub_xs[2] + 0.030 + (bx_w - 0.045)/2, sub_y + 0.16, (bx_w - 0.045)/2, sub_h - 0.23))
    binary = (base2d > np.percentile(base2d, 99.7)).astype(float)
    draw_small_map(ax_d2, binary, "rain prob.", cmap="Blues", vmin=0, vmax=1)

    # Event definition panel
    ax_event = fig.add_axes((p_event[0] + 0.025, p_event[1] + 0.16, p_event[2] - 0.05, 0.12))
    draw_small_map(ax_event, base_patch, "patch score", cmap="turbo", vmin=0, vmax=max(1e-6, np.percentile(base_patch, 99.5)), grid=True)
    fig.text(p_event[0] + p_event[2]/2, p_event[1] + 0.115,
             "$s_j=\\max_{u\\in\\Omega_j}\\hat{Y}_u$\n$\\hat{e}_j=\\mathbf{1}[s_j\\geq\\tau_p]$",
             ha="center", va="top", fontsize=9)

    # RS panel: perturbed volume thumbnails
    for k in range(3):
        axp = fig.add_axes((p_rs[0] + 0.035, p_rs[1] + 0.235 - k*0.075, p_rs[2] - 0.07, 0.062), projection="3d")
        if k == 0:
            volp = vol
            ttl = "$k=1$"
        elif k == 1:
            volp = vol.copy()
            ttl = "$k=2$"
        else:
            volp = vol.copy()
            ttl = "$k=N$"
        draw_3d_volume(axp, volp, title=ttl, stride=32, q=99.0, alpha=0.75,
                       drop_level=(vol.shape[0]//2 if k == 1 else None))
    fig.text(p_rs[0] + p_rs[2]/2, p_rs[1] + 0.08,
             "$X^{(k)}\\sim\\mathcal{P}_m(X)$\nrepeat sampling",
             ha="center", va="center", fontsize=8.5)

    # Outputs
    ax_o1 = fig.add_axes((p_out[0] + 0.035, p_out[1] + 0.205, p_out[2] - 0.07, 0.105))
    im1 = draw_small_map(ax_o1, rs_event_prob, "$p_j$", cmap="turbo", vmin=0, vmax=1)
    ax_o2 = fig.add_axes((p_out[0] + 0.035, p_out[1] + 0.075, p_out[2] - 0.07, 0.105))
    im2 = draw_small_map(ax_o2, rs_instability, "$U_j$", cmap="magma", vmin=0, vmax=1)
    fig.text(p_out[0] + p_out[2]/2, p_out[1] + 0.335,
             "$p_j=\\frac{1}{N}\\sum_k\\mathbf{1}[s_j^{(k)}\\geq\\tau_p]$",
             ha="center", va="center", fontsize=7.6)
    fig.text(p_out[0] + p_out[2]/2, p_out[1] + 0.055,
             "$U_j=4p_j(1-p_j)$",
             ha="center", va="center", fontsize=8.2)

    # Perturbation families row
    px, py, pw, ph = p_pert
    names = [
        "(1) Gaussian\nnoise",
        "(2) Level\nscaling",
        "(3) Level\ndropout",
        "(4) Temporal\nframe dropout",
        "(5) Local block\nmasking",
        "(6) Mixed\nstructured",
    ]
    desc = [
        "measurement /\ncalibration error",
        "layer-wise\nintensity bias",
        "missing vertical\nlayers",
        "incomplete\nhistory",
        "local echo\nmissing",
        "compound\nuncertainty",
    ]
    n = 6
    cell_w = (pw - 0.03) / n
    for i in range(n):
        cx = px + 0.015 + i * cell_w
        # separator
        if i > 0:
            fig.add_artist(Rectangle((cx - 0.006, py + 0.03), 0.001, ph - 0.06,
                                     transform=fig.transFigure, facecolor="#CCCCCC", edgecolor="none"))
        fig.text(cx + cell_w/2, py + ph - 0.04, names[i],
                 ha="center", va="top", fontsize=7.7, fontweight="bold")
        axmini = fig.add_axes((cx + 0.018, py + 0.062, cell_w - 0.036, 0.075), projection="3d")
        if i == 4:
            y0 = int(H*0.35); y1 = int(H*0.58); x0b = int(W*0.35); x1b = int(W*0.58)
            draw_3d_volume(axmini, vol, stride=36, q=99.0, alpha=0.65, mask_block=(y0, y1, x0b, x1b))
        elif i == 2:
            draw_3d_volume(axmini, vol, stride=36, q=99.0, alpha=0.65, drop_level=vol.shape[0]//2)
        else:
            draw_3d_volume(axmini, vol, stride=36, q=99.0, alpha=0.55)
        fig.text(cx + cell_w/2, py + 0.035, desc[i], ha="center", va="center", fontsize=7.2)

    # Physical meaning box
    fig.text(p_metrics[0] + 0.02, p_metrics[1] + p_metrics[3] - 0.055,
             "• Vertical structure uncertainty\n"
             "• Temporal evolution uncertainty\n"
             "• Local observation incompleteness\n"
             "• Intensity calibration uncertainty\n"
             "• Processing / clutter artifacts",
             ha="left", va="top", fontsize=8.3, linespacing=1.45)

    # RS spatial products row
    sx, sy, sw, sh = p_products
    map_w = (sw - 0.08) / 4
    maps = [
        (target2d, "(a) Target\nprecipitation", "turbo", 0, vmax_rain),
        (base2d, "(b) Baseline\nretrieval", "turbo", 0, vmax_rain),
        (rs_event_prob, "(c) Event\nprobability", "turbo", 0, 1),
        (rs_instability, "(d) Instability", "magma", 0, 1),
    ]
    for i, (dat, ttl, cmap, vmin, vmax) in enumerate(maps):
        ax = fig.add_axes((sx + 0.025 + i*(map_w + 0.012), sy + 0.055, map_w, sh - 0.105))
        draw_small_map(ax, dat, ttl, cmap=cmap, vmin=vmin, vmax=vmax, grid=(dat.ndim == 2 and dat.shape == rs_event_prob.shape))

    # Quantitative diagnosis
    ix, iy, iw, ih = p_insights
    fig.text(ix + 0.02, iy + ih - 0.055,
             "Main empirical findings from physical RS:\n"
             "• Temporal dropout produces the highest instability and score variability.\n"
             "• Vertical-level dropout induces much larger score variability than Gaussian noise.\n"
             "• Block masking increases boundary uncertainty, indicating sensitivity to local echo completeness.\n"
             "• Gaussian noise yields low uncertainty and may underestimate structured radar sensitivity.",
             ha="left", va="top", fontsize=8.8, linespacing=1.45)

    # Footer
    fig.text(0.5, 0.018,
             "Random mask is used for self-supervised representation learning; physical perturbations are applied only at inference time for uncertainty diagnosis.",
             ha="center", va="center", fontsize=8.3,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", lw=0.8))

    # Save
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--rs-perturbation", type=str, default="level_dropout",
                        choices=["gaussian", "level_scaling", "level_dropout", "temporal_dropout", "block_mask", "mixed"])
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--target-rain-threshold", type=float, default=0.1)
    parser.add_argument("--target-area-ratio", type=float, default=0.001)
    parser.add_argument("--pred-event-threshold", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=str, default="figures/aaai_framework")
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("Creating config...", flush=True)
    config = get_config()
    config["data"]["batch_size"] = args.batch_size
    config["data"]["num_workers"] = args.num_workers
    config["train"]["use_weighted_sampler"] = False

    # Override MAE checkpoint for the Guangdong fine-tuned model used in this project.
    # The default finetune_config.py may point to ./checkpoints/mae_pretrain/best_model.pt,
    # which does not exist in the current workspace.
    if "model" in config:
        config["model"]["mae_checkpoint"] = "/path/to/mae_checkpoint.pt"
        print(f"Using MAE checkpoint: {config['model']['mae_checkpoint']}", flush=True)

    print("Creating model...", flush=True)
    model = create_model(config)
    load_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()

    print("Creating data loaders...", flush=True)
    _, val_loader, dataset = create_data_loaders(config)
    print("Fetching selected validation sample...", flush=True)
    x, target = get_selected_sample(val_loader, args.sample_index)
    print("Selected sample fetched.", flush=True)

    x = x.to(device).float()
    if torch.is_tensor(target):
        target = target.to(device).float()

    print(f"x shape: {tuple(x.shape)}", flush=True)
    if torch.is_tensor(target):
        print(f"target shape: {tuple(target.shape)}", flush=True)

    print("Running baseline prediction...", flush=True)
    with torch.no_grad():
        baseline_pred = model_predict_reg(model, x)

    print("Computing RS maps...", flush=True)
    rs_event_prob, rs_instability, score_std, pred_mean = compute_rs_maps(
        model=model,
        x=x,
        rng=rng,
        mode=args.rs_perturbation,
        num_samples=args.num_samples,
        patch_size=args.patch_size,
        pred_event_threshold=args.pred_event_threshold,
    )

    print("Drawing framework figure...", flush=True)
    out_png = out_dir / f"aaai_framework_sample{args.sample_index:04d}_{args.rs_perturbation}_N{args.num_samples}.png"
    out_pdf = out_dir / f"aaai_framework_sample{args.sample_index:04d}_{args.rs_perturbation}_N{args.num_samples}.pdf"

    make_framework_figure(
        x=x,
        target=target,
        baseline_pred=baseline_pred,
        rs_event_prob=rs_event_prob,
        rs_instability=rs_instability,
        pred_mean=pred_mean,
        out_path_pdf=out_pdf,
        out_path_png=out_png,
        patch_size=args.patch_size,
    )

    print(f"Saved PNG: {out_png.resolve()}", flush=True)
    print(f"Saved PDF: {out_pdf.resolve()}", flush=True)

    if hasattr(dataset, "close"):
        dataset.close()


if __name__ == "__main__":
    main()
