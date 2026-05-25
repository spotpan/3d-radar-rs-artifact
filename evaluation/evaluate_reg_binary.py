import torch
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.finetune_guangdong_config import get_config
from training.finetune_guangdong import create_model, create_data_loaders

ckpt_path = "checkpoints/binary_regfirst_relu_cls01_5000t200v/binary_regfirst_relu_cls01_5000t200v/best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = get_config()
config["data"]["batch_size"] = 1
config["data"]["num_workers"] = 0
config["train"]["use_weighted_sampler"] = False

model = create_model(config)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state = ckpt["model_state_dict"]
if all(k.startswith("module.") for k in state.keys()):
    state = {k[7:]: v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.to(device).eval()

_, val_loader, dataset = create_data_loaders(config)

conf = np.zeros((2, 2), dtype=np.int64)
mae_sum = 0.0
rmse_sum = 0.0
n_pix = 0

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i >= 200:
            break
        if i % 10 == 0:
            print(f"Processed {i}/200 batches")

        x = batch["radar_sequence"].to(device)
        y = batch["rain"].to(device)

        out = model(x)
        pred = out["reg_output"].clamp(min=0)

        pred_bin = (pred >= 0.1).long()
        target_bin = (y >= 0.1).long()

        for t in [0, 1]:
            for p in [0, 1]:
                conf[t, p] += int(((target_bin == t) & (pred_bin == p)).sum().item())

        diff = pred - y
        mae_sum += diff.abs().sum().item()
        rmse_sum += (diff ** 2).sum().item()
        n_pix += diff.numel()

tn, fp = conf[0, 0], conf[0, 1]
fn, tp = conf[1, 0], conf[1, 1]

precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-12)
accuracy = (tp + tn) / max(conf.sum(), 1)

print("\nRegression-derived binary rain/no-rain, threshold=0.1")
print("confusion matrix rows=true cols=pred")
print(conf)
print(f"precision={precision:.6f}")
print(f"recall={recall:.6f}")
print(f"f1={f1:.6f}")
print(f"accuracy={accuracy:.6f}")
print(f"MAE={mae_sum / n_pix:.6f}")
print(f"RMSE={(rmse_sum / n_pix) ** 0.5:.6f}")

dataset.close()
