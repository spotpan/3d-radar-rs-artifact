import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle


def load_abc(npz):
    """
    abc saved as object array like [['A', r, c], ...]
    """
    abc_arr = npz["abc"]
    out = {}
    for item in abc_arr:
        label = str(item[0])
        r = int(item[1])
        c = int(item[2])
        out[label] = (r, c)
    return out


def decision_from_prob(p, low=0.2, high=0.8):
    if p >= high:
        return "RAIN"
    if p <= low:
        return "NO-RAIN"
    return "UNCERTAIN"


def upsample_patch_map(patch_map, patch_size, H, W):
    return np.kron(patch_map, np.ones((patch_size, patch_size)))[:H, :W]


def baseline_patch_score_map(baseline, patch_size):
    H, W = baseline.shape
    bh = H // patch_size
    bw = W // patch_size
    crop = baseline[:bh * patch_size, :bw * patch_size]
    patch = crop.reshape(bh, patch_size, bw, patch_size).max(axis=(1, 3))
    up = upsample_patch_map(patch, patch_size, H, W)
    return patch, up


def draw(npz_path, out_path):
    data = np.load(npz_path, allow_pickle=True)

    target = data["target"]
    baseline = data["baseline"]
    event_prob = data["event_prob"]
    instability = data["instability"]
    target_event = data["target_event"]
    scores = data["scores"]
    events = data["events"]
    abc = load_abc(data)

    patch_size = int(data["patch_size"])
    pred_threshold = float(data["pred_event_threshold"])
    perturbation = str(data["perturbation"])
    num_samples = int(data["num_samples"])

    H, W = target.shape

    prob_up = upsample_patch_map(event_prob, patch_size, H, W)
    inst_up = upsample_patch_map(instability, patch_size, H, W)
    target_event_up = upsample_patch_map(target_event, patch_size, H, W)
    baseline_patch, baseline_patch_up = baseline_patch_score_map(baseline, patch_size)

    colors = {"A": "red", "B": "orange", "C": "cyan"}
    names = {"A": "Stable rain", "B": "Uncertain", "C": "Stable no-rain"}

    # Big, dense, readable layout
    fig = plt.figure(figsize=(24, 16.2), dpi=220)
    gs = gridspec.GridSpec(
        4, 4,
        figure=fig,
        height_ratios=[1.95, 1.42, 1.42, 1.42],
        width_ratios=[1.25, 1.25, 1.25, 1.32],
        wspace=0.24,
        hspace=0.55,
    )

    # -------------------------
    # Top row maps
    # -------------------------
    axes = []

    ax0 = fig.add_subplot(gs[0, 0])
    vmax_target = max(1.0, float(np.percentile(target, 99.5)))
    im0 = ax0.imshow(target, origin="upper", cmap="turbo", vmin=0, vmax=vmax_target)
    ax0.set_title("(a) Target precipitation", fontsize=20, fontweight="bold", pad=10)
    ax0.set_xticks([]); ax0.set_yticks([])
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.050, pad=0.025)
    cb0.ax.tick_params(labelsize=13)
    axes.append(ax0)

    ax1 = fig.add_subplot(gs[0, 1])
    vmax_score = max(float(np.percentile(baseline_patch, 99.5)), pred_threshold * 10, 0.01)
    im1 = ax1.imshow(baseline_patch_up, origin="upper", cmap="turbo", vmin=0, vmax=vmax_score)
    ax1.set_title("(b) Baseline patch score", fontsize=20, fontweight="bold", pad=10)
    ax1.set_xticks([]); ax1.set_yticks([])
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.050, pad=0.025)
    cb1.set_label("Patch score", fontsize=13)
    cb1.ax.tick_params(labelsize=13)
    axes.append(ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(prob_up, origin="upper", cmap="viridis", vmin=0, vmax=1)
    ax2.contour(target_event_up, levels=[0.5], colors="white", linewidths=1.2)
    ax2.set_title("(c) Event probability", fontsize=20, fontweight="bold", pad=10)
    ax2.set_xticks([]); ax2.set_yticks([])
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.050, pad=0.025)
    cb2.ax.tick_params(labelsize=13)
    axes.append(ax2)

    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(inst_up, origin="upper", cmap="inferno", vmin=0, vmax=1)
    ax3.contour(target_event_up, levels=[0.5], colors="white", linewidths=1.2)
    ax3.set_title("(d) Instability", fontsize=20, fontweight="bold", pad=10)
    ax3.set_xticks([]); ax3.set_yticks([])
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.050, pad=0.025)
    cb3.ax.tick_params(labelsize=13)
    axes.append(ax3)

    # A/B/C boxes on all top maps
    for label, (r, c) in abc.items():
        y0 = r * patch_size
        x0 = c * patch_size
        for ax in axes:
            rect = Rectangle(
                (x0, y0), patch_size, patch_size,
                fill=False,
                edgecolor=colors[label],
                linewidth=3.0,
            )
            ax.add_patch(rect)
            ax.text(
                x0 + 10, y0 + 32, label,
                color=colors[label],
                fontsize=20,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2.2),
            )

    # -------------------------
    # Histogram rows
    # -------------------------
    row_map = {"A": 1, "B": 2, "C": 3}
    panel_letters = {"A": "e", "B": "f", "C": "g"}

    for label in ["A", "B", "C"]:
        r, c = abc[label]
        score_values = scores[:, r, c]
        event_values = events[:, r, c]

        event_prob_i = float(event_values.mean())
        score_mean = float(score_values.mean())
        score_std = float(score_values.std())
        q025, q975 = np.quantile(score_values, [0.025, 0.975])
        decision = decision_from_prob(event_prob_i)

        finite_scores = score_values[np.isfinite(score_values)]
        if finite_scores.size == 0 or float(np.max(finite_scores)) <= 1e-10:
            x_max = 0.01
            bins = np.linspace(0, x_max, 24)
        else:
            smax = float(np.max(finite_scores))
            q975_score = float(np.percentile(finite_scores, 97.5))
            x_max = max(smax * 1.05, q975_score * 1.15, pred_threshold * 8.0, 0.01)
            bins = 28

        axh = fig.add_subplot(gs[row_map[label], 0:3])
        axh.hist(
            score_values,
            bins=bins,
            color=colors[label],
            alpha=0.38,
            edgecolor=colors[label],
            linewidth=1.1,
        )
        axh.axvline(pred_threshold, color="black", linestyle="--", linewidth=2.4, label=f"threshold={pred_threshold:g}")
        axh.axvline(score_mean, color=colors[label], linestyle="-", linewidth=3.0, label="mean")
        axh.set_xlim(0, x_max)

        axh.set_title(
            f"({panel_letters[label]}) Patch {label}: {names[label]}    "
            f"P={event_prob_i:.2f}, {decision}",
            loc="left",
            color=colors[label],
            fontsize=18,
            fontweight="bold",
            pad=10,
        )
        axh.set_xlabel("Patch score: maximum retrieval within the patch", fontsize=14, labelpad=6)
        axh.set_ylabel("Frequency", fontsize=14, labelpad=6)
        axh.tick_params(labelsize=13)
        axh.grid(alpha=0.22)
        axh.legend(fontsize=13, loc="upper right", frameon=True)

        axt = fig.add_subplot(gs[row_map[label], 3])
        axt.axis("off")
        box_text = (
            f"Patch {label}: {names[label]}\n"
            f"Index: ({r}, {c})\n"
            f"P(event): {event_prob_i:.3f}\n"
            f"Mean: {score_mean:.5f}\n"
            f"Std: {score_std:.5f}\n"
            f"95% CI: [{q025:.5f}, {q975:.5f}]\n"
            f"Decision: {decision}"
        )
        axt.text(
            0.03, 0.96, box_text,
            va="top",
            fontsize=15,
            linespacing=1.18,
            bbox=dict(
                boxstyle="round,pad=0.65",
                facecolor="white",
                edgecolor=colors[label],
                linewidth=2.6,
            ),
        )

    title = "Patch-level prediction distributions under vertical-level perturbations"
    fig.suptitle(title, fontsize=25, fontweight="bold", y=0.988)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=320)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(out_path.with_suffix(".png"))
    print(out_path.with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    draw(args.npz, args.out)


if __name__ == "__main__":
    main()
