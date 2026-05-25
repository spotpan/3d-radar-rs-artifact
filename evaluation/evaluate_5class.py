import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make project root importable when running as: python evaluation/evaluate_5class.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.finetune_guangdong_config import get_config
from training.finetune_guangdong import create_model, create_data_loaders


def strip_module_prefix(state_dict):
    """Handle checkpoints saved from nn.DataParallel."""
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_finetuned_checkpoint(model, checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Some checkpoints may directly be a state_dict-like dict.
            state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded checkpoint.")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("  First missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("  First unexpected keys:", unexpected[:10])

    return model


def update_confusion_matrix(conf_mat, target, pred, num_classes):
    """
    target, pred: torch.Tensor with shape (B, H, W), values in [0, num_classes-1]
    """
    target = target.reshape(-1).to(torch.int64)
    pred = pred.reshape(-1).to(torch.int64)

    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    target = target[valid]
    pred = pred[valid]

    idx = target * num_classes + pred
    bincount = torch.bincount(idx, minlength=num_classes * num_classes)
    conf_mat += bincount.reshape(num_classes, num_classes).cpu().numpy()


def metrics_from_confusion(conf_mat):
    conf = conf_mat.astype(np.float64)
    tp = np.diag(conf)
    support = conf.sum(axis=1)
    pred_sum = conf.sum(axis=0)

    precision = tp / np.maximum(pred_sum, 1.0)
    recall = tp / np.maximum(support, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    accuracy = tp.sum() / np.maximum(conf.sum(), 1.0)
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support).sum() / np.maximum(support.sum(), 1.0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def print_confusion_and_metrics(title, conf_mat, class_names):
    print(f"\n==================== {title} ====================")
    print("Confusion matrix: rows=true, cols=pred")
    header = "true\\pred".ljust(16) + "".join([name.rjust(14) for name in class_names])
    print(header)
    for i, row in enumerate(conf_mat):
        print(class_names[i].ljust(16) + "".join([f"{int(x):14d}" for x in row]))

    m = metrics_from_confusion(conf_mat)
    print("\nPer-class metrics:")
    print("class".ljust(16), "precision".rjust(12), "recall".rjust(12), "f1".rjust(12), "support".rjust(14))
    for name, p, r, f, s in zip(class_names, m["precision"], m["recall"], m["f1"], m["support"]):
        print(name.ljust(16), f"{p:12.6f}", f"{r:12.6f}", f"{f:12.6f}", f"{int(s):14d}")

    print("\nSummary:")
    print(f"accuracy:    {m['accuracy']:.6f}")
    print(f"macro_f1:    {m['macro_f1']:.6f}")
    print(f"weighted_f1: {m['weighted_f1']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument(
        "--sweep-binary-thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Thresholds for binary rain/no-rain sweep.",
    )
    parser.add_argument(
        "--rain-thresholds",
        type=float,
        nargs="+",
        default=[0.1, 2.5, 8.0, 16.0],
        help="Four thresholds for five rain classes.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    config = get_config()

    # Keep evaluation light and deterministic.
    config["data"]["batch_size"] = args.batch_size
    config["data"]["num_workers"] = args.num_workers
    config["train"]["val_batch_size"] = args.batch_size

    # Create model and load fine-tuned checkpoint.
    model = create_model(config)
    model = load_finetuned_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    # Data loaders.
    train_loader, val_loader, dataset = create_data_loaders(config)

    thresholds = torch.tensor(args.rain_thresholds, device=device, dtype=torch.float32)

    binary_conf = np.zeros((2, 2), dtype=np.int64)
    sweep_binary_confs = {
        float(th): np.zeros((2, 2), dtype=np.int64)
        for th in args.sweep_binary_thresholds
    }
    five_conf = np.zeros((5, 5), dtype=np.int64)

    abs_err_sum = 0.0
    sq_err_sum = 0.0
    reg_count = 0

    class_names_binary = ["no_rain", "rain"]
    class_names_five = [
        f"<{args.rain_thresholds[0]}",
        f"{args.rain_thresholds[0]}-{args.rain_thresholds[1]}",
        f"{args.rain_thresholds[1]}-{args.rain_thresholds[2]}",
        f"{args.rain_thresholds[2]}-{args.rain_thresholds[3]}",
        f">={args.rain_thresholds[3]}",
    ]

    print(f"\nEvaluating max_batches={args.max_batches}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            radar_sequence = batch["radar_sequence"].to(device)
            target = batch["rain"].to(device)  # (B, 1, H, W)

            outputs = model(radar_sequence)

            pred_reg = outputs["reg_output"].clamp(min=0)
            pred_bin_prob = outputs["cls_output"]
            pred_multi_logits = outputs["multi_cls_logits"]

            # Binary labels.
            target_bin = (target >= args.rain_thresholds[0]).long().squeeze(1)
            pred_bin = (pred_bin_prob >= args.binary_threshold).long().squeeze(1)
            update_confusion_matrix(binary_conf, target_bin, pred_bin, 2)

            for th, conf in sweep_binary_confs.items():
                pred_bin_sweep = (pred_bin_prob >= th).long().squeeze(1)
                update_confusion_matrix(conf, target_bin, pred_bin_sweep, 2)

            # Five-class labels.
            target_five = torch.bucketize(target.squeeze(1).float(), thresholds)
            pred_five = pred_multi_logits.argmax(dim=1)
            update_confusion_matrix(five_conf, target_five, pred_five, 5)

            # Regression metrics.
            valid = torch.isfinite(target) & torch.isfinite(pred_reg)
            err = (pred_reg - target)[valid].float()
            abs_err_sum += err.abs().sum().item()
            sq_err_sum += (err ** 2).sum().item()
            reg_count += err.numel()

            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1} batches")

    print_confusion_and_metrics("Binary rain/no-rain", binary_conf, class_names_binary)

    print("\n==================== Binary threshold sweep ====================")
    print("threshold".rjust(10), "precision".rjust(12), "recall".rjust(12), "f1".rjust(12), "accuracy".rjust(12))
    for th in sorted(sweep_binary_confs.keys()):
        m = metrics_from_confusion(sweep_binary_confs[th])
        # class 1 is rain
        p = m["precision"][1]
        r = m["recall"][1]
        f = m["f1"][1]
        a = m["accuracy"]
        print(f"{th:10.3f} {p:12.6f} {r:12.6f} {f:12.6f} {a:12.6f}")

    print_confusion_and_metrics("Five-class rain level", five_conf, class_names_five)

    mae = abs_err_sum / max(reg_count, 1)
    rmse = np.sqrt(sq_err_sum / max(reg_count, 1))
    print("\n==================== Regression ====================")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    dataset.close()


if __name__ == "__main__":
    main()
