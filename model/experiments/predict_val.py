"""Save validation-set probabilities of the three trained models (seed-42 checkpoints).

Why: a decision threshold for the worklist should be tuned on validation data,
not on the test set. The format is the same as data/predictions/<model>_test_probs.csv.

Run from the repository root:
    python model/experiments/predict_val.py

Output (local, not in GitHub; share through Drive like the test files):
    data/predictions/<model>_val_probs.csv

Note: the best epoch of each model was chosen on this same validation set, so
validation numbers are slightly optimistic. They are still the cleaner choice for
tuning a threshold, because the test set stays untouched for the final report.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # avoids "OpenBLAS memory allocation failed" on Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import SequentialSampler

from data import CLASSES, get_loaders
from train import build_model

MODELS = ["efficientnet_b0", "resnet50", "densenet121"]


@torch.no_grad()
def val_probs(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(x)
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_loader, _ = get_loaders(batch_size=32, num_workers=0)   # no worker processes: lower memory use
    val_df = pd.read_csv("data/splits/val.csv")
    if not isinstance(val_loader.sampler, SequentialSampler):
        sys.exit("The validation loader is shuffled; the row order would not match val.csv.")
    if len(val_loader.dataset) != len(val_df):
        sys.exit(f"Size mismatch: loader {len(val_loader.dataset)} vs val.csv {len(val_df)}")
    if not {"path", "label"} <= set(val_df.columns):
        sys.exit(f"val.csv columns are {list(val_df.columns)}; expected 'path' and 'label'.")

    out_dir = Path("data/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        ckpt = Path("model/checkpoints") / f"{name}_8ep_best.pt"
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        probs = val_probs(model, val_loader, device)

        df = val_df[["path", "label"]].copy()
        for i, c in enumerate(CLASSES):
            df[f"p_{c}"] = probs[:, i]
        df["pred"] = [CLASSES[i] for i in probs.argmax(1)]
        df.to_csv(out_dir / f"{name}_val_probs.csv", index=False)

        acc = accuracy_score(df["label"], df["pred"])
        f1 = f1_score(df["label"], df["pred"], average="macro")
        print(f"{name:16} val accuracy {acc:.3f} | macro-F1 {f1:.3f} | saved {len(df)} rows")


if __name__ == "__main__":
    main()