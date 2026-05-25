import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib import gridspec
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.finetune_guangdong_config import get_config
from training.finetune_guangdong import create_model, create_data_loaders


def strip_module_prefix(state_dict):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def get_sample(val_loader, sample_index):
    for i, batch in enumerate(val_loader):
        if i == sample_index:
            return batch
    raise RuntimeError(f"sample_index={sample_index} not found")


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
    if mode == "block_mask":
        return perturb_block_mask(x, block_size=args.block_size, num_blocks=args.num_blocks)
    if mode == "mixed":
        x2 = perturb_gaussian(x, sigma=args.gaussian_sigma)
        x2 = perturb_level_scaling(x2, scale_std=args.level_scale_std)
        x2 = perturb_block_mask(x2, block_size=args.block_size, num_blocks=args.num_blocks)
        return x2
    raise ValueError(f"Unknown perturbation mode: {mode}")


def patch_max_scores(preds, patch_size):
    """
    preds: (N,1,H,W)
    returns scores: (N,hp,wp)
    """
    scores = F.max_pool2d(preds, kernel_size=patch_size, stride=patch_size)
    return scores[:, 0]


def patch_event_probability(scores, threshold):
    """
    scores: (N,hp,wp)
    """
    events = (scores >= threshold).float()
    prob = events.mean(dim=0)
    return prob, events


def target_patch_event(y, patch_size, rain_threshold, area_ratio):
    y_bin = (y >= rain_threshold).float()
    frac = F.avg_pool2d(y_bin, kernel_size=patch_size, stride=patch_size)
    return (frac[0, 0] >= area_ratio).float(), frac[0, 0]


def choose_abc_patches(prob, instability, target_event):
    """
    prob/instability/target_event: (hp,wp) tensors.
    Return patch coordinates (row, col) for A/B/C.
    """
    p = prob.detach().cpu().numpy()
    u = instability.detach().cpu().numpy()
    t = target_event.detach().cpu().numpy()

    hp, wp = p.shape

    # A: stable rain, prefer high p and inside target event if possible
    mask_a = t > 0.5
    score_a = p - 0.25 * u
    if mask_a.any():
        score_a = np.where(mask_a, score_a, -999)
    a_idx = np.unravel_index(np.argmax(score_a), p.shape)

    # B: uncertain boundary, p near 0.5 or high instability
    score_b = u - 0.15 * np.abs(p - 0.5)
    b_idx = np.unravel_index(np.argmax(score_b), p.shape)

    # C: stable no-rain, low p and outside target event
    mask_c = t < 0.5
    score_c = (1.0 - p) - 0.25 * u
    if mask_c.any():
        score_c = np.where(mask_c, score_c, -999)
    c_idx = np.unravel_index(np.argmax(score_c), p.shape)

    return {"A": a_idx, "B": b_idx, "C": c_idx}


def decision_from_prob(mean_event_prob, low=0.2, high=0.8):
    if mean_event_prob >= high:
        return "RAIN"
    if mean_event_prob <= low:
        return "NO-RAIN"
    return "UNCERTAIN"


def draw_distribution_panel(
    target,
    baseline,
    event_prob,
    instability,
    target_event,
    scores_np,
    events_np,
    abc,
    patch_size,
    pred_threshold,
    out_path,
    title,
):
    """
    target/baseline: H,W numpy
    event_prob/instability/target_event: hp,wp numpy
    scores_np/events_np: N,hp,wp numpy
    abc: dict label -> (row,col)
    """
    hp, wp = event_prob.shape
    H, W = target.shape

    # Upsample patch maps for spatial display
    prob_up = np.kron(event_prob, np.ones((patch_size, patch_size)))[:H, :W]
    inst_up = np.kron(instability, np.ones((patch_size, patch_size)))[:H, :W]
    target_event_up = np.kron(target_event, np.ones((patch_size, patch_size)))[:H, :W]

    # Baseline patch score map: max retrieval in each patch.
    # This is more consistent with the patch-score distributions below than raw pixel-level retrieval.
    bh = H // patch_size
    bw = W // patch_size
    baseline_crop = baseline[:bh * patch_size, :bw * patch_size]
    baseline_patch = baseline_crop.reshape(bh, patch_size, bw, patch_size).max(axis=(1, 3))
    baseline_patch_up = np.kron(baseline_patch, np.ones((patch_size, patch_size)))[:H, :W]

    fig = plt.figure(figsize=(22.5, 15.0), dpi=200)
    gs = gridspec.GridSpec(
        4, 4,
        figure=fig,
        width_ratios=[1.18, 1.18, 1.18, 1.28],
        height_ratios=[1.28, 1.22, 1.22, 1.22],
        wspace=0.28,
        hspace=0.70,
    )

    # Spatial maps
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(target, origin="upper", cmap="turbo", vmin=0, vmax=max(1.0, np.percentile(target, 99.5)))
    ax0.set_title("(a) Target", fontweight="bold", fontsize=15, pad=8)
    ax0.set_xticks([]); ax0.set_yticks([])
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.050, pad=0.025)
    cb0.ax.tick_params(labelsize=11)

    ax1 = fig.add_subplot(gs[0, 1])
    score_vmax = max(float(np.percentile(baseline_patch, 99.5)), pred_threshold * 10, 0.01)
    im1 = ax1.imshow(
        baseline_patch_up,
        origin="upper",
        cmap="turbo",
        vmin=0,
        vmax=score_vmax,
    )
    ax1.set_title("(b) Baseline patch score", fontweight="bold", fontsize=15, pad=8)
    ax1.set_xticks([]); ax1.set_yticks([])
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cb1.set_label("Patch score", fontsize=12)
    cb1.ax.tick_params(labelsize=11)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(prob_up, origin="upper", cmap="viridis", vmin=0, vmax=1)
    ax2.contour(target_event_up, levels=[0.5], colors="white", linewidths=0.8)
    ax2.set_title("(c) Event probability", fontweight="bold", fontsize=15, pad=8)
    ax2.set_xticks([]); ax2.set_yticks([])
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.050, pad=0.025)
    cb2.ax.tick_params(labelsize=11)

    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(inst_up, origin="upper", cmap="inferno", vmin=0, vmax=1)
    ax3.contour(target_event_up, levels=[0.5], colors="white", linewidths=0.8)
    ax3.set_title("(d) Instability", fontweight="bold", fontsize=15, pad=8)
    ax3.set_xticks([]); ax3.set_yticks([])
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.050, pad=0.025)
    cb3.ax.tick_params(labelsize=11)

    # Mark A/B/C on maps
    colors = {"A": "red", "B": "orange", "C": "cyan"}
    names = {"A": "Stable rain", "B": "Uncertain", "C": "Stable no-rain"}

    for label, (r, c) in abc.items():
        y0 = r * patch_size
        x0 = c * patch_size
        for ax in [ax0, ax1, ax2, ax3]:
            rect = Rectangle((x0, y0), patch_size, patch_size, fill=False,
                             edgecolor=colors[label], linewidth=2.0)
            ax.add_patch(rect)
            ax.text(x0 + 8, y0 + 28, label, color=colors[label],
                    fontsize=17, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=2.2))

    # Histograms for A/B/C
    for row_id, label in enumerate(["A", "B", "C"], start=1):
        r, c = abc[label]
        score_values = scores_np[:, r, c]
        event_values = events_np[:, r, c]
        event_prob_i = float(event_values.mean())
        score_mean = float(score_values.mean())
        score_std = float(score_values.std())
        q025, q975 = np.quantile(score_values, [0.025, 0.975])
        decision = decision_from_prob(event_prob_i)

        axh = fig.add_subplot(gs[row_id, 0:3])
        # Adaptive non-negative x-axis. Keep the distribution natural but avoid negative ranges.
        finite_scores = score_values[np.isfinite(score_values)]
        if finite_scores.size == 0 or float(np.max(finite_scores)) <= 1e-10:
            x_max = 0.01
            bins = np.linspace(0, x_max, 24)
        else:
            smax = float(np.max(finite_scores))
            q975 = float(np.percentile(finite_scores, 97.5))
            x_max = max(smax * 1.05, q975 * 1.15, pred_threshold * 8.0, 0.01)
            bins = 24

        axh.hist(score_values, bins=bins, color=colors[label], alpha=0.36, edgecolor=colors[label], linewidth=1.0)
        axh.axvline(pred_threshold, color="black", linestyle="--", linewidth=2.0, label=f"threshold={pred_threshold:g}")
        axh.axvline(score_mean, color=colors[label], linestyle="-", linewidth=2.8, label="mean")
        axh.set_xlim(0, x_max)

        axh.set_title(
            f"({chr(100 + row_id)}) Patch {label}: {names[label]}   "
            f"P={event_prob_i:.2f}, {decision}",
            loc="left",
            fontweight="bold",
            color=colors[label],
            fontsize=16,
            pad=12,
        )
        axh.set_xlabel("Patch score: maximum retrieval within the patch", fontsize=13, labelpad=7)
        axh.set_ylabel("Frequency", fontsize=13, labelpad=6)
        axh.legend(fontsize=11, loc="upper right", frameon=True)
        axh.grid(alpha=0.22)
        axh.tick_params(labelsize=12)

        axt = fig.add_subplot(gs[row_id, 3])
        axt.axis("off")
        text = (
            f"Patch {label}: {names[label]}\n\n"
            f"Patch index: ({r}, {c})\n"
            f"P(event): {event_prob_i:.3f}\n"
            f"Mean score: {score_mean:.5f}\n"
            f"Std score: {score_std:.5f}\n"
            f"95% interval:\n[{q025:.5f}, {q975:.5f}]\n\n"
            f"Smoothed decision:\n{decision}"
        )
        axt.text(
            0.03, 0.95, text,
            va="top",
            fontsize=14,
            linespacing=1.22,
            bbox=dict(boxstyle="round,pad=0.70", facecolor="white",
                      edgecolor=colors[label], linewidth=2.4),
        )

    fig.suptitle(title, fontsize=21, fontweight="bold", y=0.992)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=320)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/binary_regfirst_relu_cls01_5000t200v/binary_regfirst_relu_cls01_5000t200v/best_model.pt",
    )
    parser.add_argument("--sample-index", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--perturbation", type=str, default="level_dropout",
                        choices=["gaussian", "level_dropout", "level_scaling", "block_mask", "mixed"])

    parser.add_argument("--gaussian-sigma", type=float, default=0.03)
    parser.add_argument("--level-drop-prob", type=float, default=0.15)
    parser.add_argument("--level-scale-std", type=float, default=0.10)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)

    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--target-rain-threshold", type=float, default=0.1)
    parser.add_argument("--target-area-ratio", type=float, default=0.001)
    parser.add_argument("--pred-event-threshold", type=float, default=0.0005)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="figures/rs_patch_demo/distribution")

    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Distribution demo arguments:", flush=True)
    print(f"  sample_index={args.sample_index}", flush=True)
    print(f"  num_samples={args.num_samples}", flush=True)
    print(f"  perturbation={args.perturbation}", flush=True)
    print(f"  patch_size={args.patch_size}", flush=True)
    print(f"  pred_event_threshold={args.pred_event_threshold}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config()
    config["data"]["batch_size"] = args.batch_size
    config["data"]["num_workers"] = args.num_workers
    config["train"]["use_weighted_sampler"] = False

    print("Creating model...", flush=True)
    model = create_model(config)
    print("Model created.", flush=True)
    print("Loading checkpoint...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print("Checkpoint loaded.", flush=True)
    state = strip_module_prefix(ckpt["model_state_dict"])
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    print("Creating data loaders...", flush=True)
    _, val_loader, dataset = create_data_loaders(config)
    print("Data loaders created.", flush=True)
    print("Fetching selected validation sample...", flush=True)
    batch = get_sample(val_loader, args.sample_index)
    print("Selected sample fetched.", flush=True)

    x = batch["radar_sequence"].to(device)
    y = batch["rain"].to(device)

    with torch.no_grad():
        baseline = model(x)["reg_output"].clamp(min=0)
        if baseline.shape[0] != 1:
            print(f"Warning: baseline batch dim {baseline.shape[0]}; using baseline[:1].", flush=True)
        baseline = baseline[:1]

    preds = []
    with torch.no_grad():
        for n in range(args.num_samples):
            if n % 4 == 0:
                print(f"RS distribution sample {n}/{args.num_samples}", flush=True)
            x_pert = apply_perturbation(x, args.perturbation, args)
            pred = model(x_pert)["reg_output"].clamp(min=0)
            if pred.shape[0] != 1 and n == 0:
                print(f"Warning: pred batch dim {pred.shape[0]}; using pred[:1] for distribution demo.", flush=True)
            pred = pred[:1]
            preds.append(pred.detach())
        print(f"RS distribution sample {args.num_samples}/{args.num_samples}", flush=True)

    print("Concatenating predictions...", flush=True)
    preds = torch.cat(preds, dim=0)  # N,1,H,W
    print("preds shape:", tuple(preds.shape), flush=True)
    print("Computing patch scores/probabilities...", flush=True)
    scores = patch_max_scores(preds, args.patch_size)
    prob, events = patch_event_probability(scores, args.pred_event_threshold)
    instability = 4.0 * prob * (1.0 - prob)

    target_event, target_frac = target_patch_event(
        y,
        patch_size=args.patch_size,
        rain_threshold=args.target_rain_threshold,
        area_ratio=args.target_area_ratio,
    )

    print("Choosing A/B/C representative patches...", flush=True)
    abc = choose_abc_patches(prob, instability, target_event)

    print("Selected patches:")
    for k, v in abc.items():
        r, c = v
        print(f"  {k}: row={r}, col={c}, p={float(prob[r,c]):.3f}, instability={float(instability[r,c]):.3f}, target={float(target_event[r,c]):.0f}")

    target_np = y[0, 0].detach().cpu().float().numpy()
    baseline_np = baseline[0, 0].detach().cpu().float().numpy()

    prob_np = prob.detach().cpu().float().numpy()
    instability_np = instability.detach().cpu().float().numpy()
    target_event_np = target_event.detach().cpu().float().numpy()
    scores_np = scores.detach().cpu().float().numpy()
    events_np = events.detach().cpu().float().numpy()

    stem = (
        f"sample{args.sample_index:04d}_"
        f"{args.perturbation}_N{args.num_samples}_"
        f"patch{args.patch_size}_distribution"
    )

    title = (
        "Patch-level prediction distributions under vertical-level perturbations"
        if args.perturbation == "level_dropout"
        else f"Patch-level prediction distributions under {args.perturbation} perturbations"
    )

    print("Drawing distribution panel...", flush=True)
    draw_distribution_panel(
        target=target_np,
        baseline=baseline_np,
        event_prob=prob_np,
        instability=instability_np,
        target_event=target_event_np,
        scores_np=scores_np,
        events_np=events_np,
        abc=abc,
        patch_size=args.patch_size,
        pred_threshold=args.pred_event_threshold,
        out_path=out_dir / stem,
        title=title,
    )

    print("Saving arrays...", flush=True)
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        target=target_np,
        baseline=baseline_np,
        event_prob=prob_np,
        instability=instability_np,
        target_event=target_event_np,
        scores=scores_np,
        events=events_np,
        abc=np.array([[k, abc[k][0], abc[k][1]] for k in ["A", "B", "C"]], dtype=object),
        patch_size=args.patch_size,
        pred_event_threshold=args.pred_event_threshold,
        perturbation=args.perturbation,
        num_samples=args.num_samples,
    )

    print("\nSaved distribution figure:")
    print(out_dir / f"{stem}.png")
    print(out_dir / f"{stem}.npz")

    dataset.close()


if __name__ == "__main__":
    main()
