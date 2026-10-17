import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data import CLASSES, compute_class_weights, get_loaders, get_transforms
from train import build_model, run_epoch

MODELS = ["efficientnet_b0", "resnet50", "densenet121"]
RUNS_CSV = Path("analysis/seed_runs.csv")
EXT_ROOT = Path("data/external/chest_xray")
EXT = {".jpeg", ".jpg", ".png"}
NORMAL_IDX = CLASSES.index("Normal")


def clean_external():
    """External images that do NOT appear in our data (same rule as before: corr >= 0.98)."""
    ov = pd.read_csv("data/external_overlap.csv")
    seen = {Path(p).as_posix() for p in ov.loc[ov["corr"] >= 0.98, "external_path"]}
    rows = []
    for split in ["train", "val", "test"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            for p in (EXT_ROOT / split / label).iterdir():
                if p.suffix.lower() in EXT and p.as_posix() not in seen:
                    rows.append((str(p), label))
    return pd.DataFrame(rows, columns=["path", "label"])


class ExtDS(Dataset):
    def __init__(self, df):
        self.paths = df["path"].tolist()
        self.tf = get_transforms(train=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("L").convert("RGB"))


@torch.no_grad()
def predict_labels(model, loader, device):
    model.eval()
    out = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            out.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(out)


def train_one(name, seed, args, device, ext_df, ext_loader):
    torch.manual_seed(seed)
    train_loader, val_loader, test_loader = get_loaders(batch_size=args.batch, num_workers=2)
    weights = compute_class_weights("data/splits/train.csv").to(device)
    model = build_model(name).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    ckpt = Path("model/checkpoints/seeds") / f"{name}_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        vl, va, vf, _ = run_epoch(model, val_loader, criterion, device)
        print(f"  [{name} seed {seed}] epoch {ep}/{args.epochs} ({time.time()-t0:.0f}s) "
              f"train acc {ta:.3f} | val acc {va:.3f} macroF1 {vf:.3f}")
        scheduler.step()
        if vf > best:
            best = vf
            torch.save(model.state_dict(), ckpt)

    # Best-validation weights, then ONE evaluation on test (results are recorded, never used to choose)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    _, acc, f1, rec = run_epoch(model, test_loader, criterion, device)

    pred = predict_labels(model, ext_loader, device)
    pneu_true = (ext_df["label"] == "PNEUMONIA").values
    pneu_pred = pred != NORMAL_IDX
    sens = float((pneu_pred & pneu_true).sum() / pneu_true.sum())
    spec = float((~pneu_pred & ~pneu_true).sum() / (~pneu_true).sum())

    row = {"model": name, "seed": seed, "best_val_macro_f1": best, "test_acc": acc,
           "test_macro_f1": f1, "ext_sens": sens, "ext_spec": spec}
    row.update({f"recall_{c}": float(r) for c, r in zip(CLASSES, rec)})
    return row


def summarize():
    runs = pd.read_csv(RUNS_CSV)
    try:  # the original runs (seed 42)
        mc = pd.read_csv("analysis/model_comparison_test.csv").set_index("model")
        ex = pd.read_csv("analysis/external_clean_generalization.csv")
        ex = ex[ex["subset"] == "clean_not_seen"].set_index("model")
        base = [{"model": m, "seed": 42, "test_acc": mc.loc[m, "accuracy"],
                 "test_macro_f1": mc.loc[m, "macro_f1"],
                 "ext_sens": ex.loc[m, "sensitivity"], "ext_spec": ex.loc[m, "specificity"]}
                for m in MODELS]
        runs = pd.concat([runs, pd.DataFrame(base)], ignore_index=True)
    except Exception as e:
        print("Could not add the seed-42 results:", e)

    print("\nMean +- std over runs (std needs at least 2 runs):")
    for m in MODELS:
        d = runs[runs["model"] == m]
        if len(d) == 0:
            continue
        s = " | ".join(f"{c} {d[c].mean():.3f}+-{d[c].std():.3f}"
                       for c in ["test_acc", "test_macro_f1", "ext_sens", "ext_spec"])
        print(f"{m:16} runs={len(d)} | {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if not args.summary_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ext_df = clean_external()
        print("Clean external images:", len(ext_df), ext_df["label"].value_counts().to_dict())
        ext_loader = DataLoader(ExtDS(ext_df), batch_size=32, shuffle=False,
                                num_workers=2, pin_memory=True)
        done = pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame(columns=["model", "seed"])
        for name in args.models:
            for seed in args.seeds:
                if ((done["model"] == name) & (done["seed"] == seed)).any():
                    print(f"skip {name} seed {seed} (already done)")
                    continue
                row = train_one(name, seed, args, device, ext_df, ext_loader)
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                done.to_csv(RUNS_CSV, index=False)   # saved after every run
                print(f"  -> test acc {row['test_acc']:.3f}, macroF1 {row['test_macro_f1']:.3f}, "
                      f"external clean sens {row['ext_sens']:.3f} spec {row['ext_spec']:.3f}")
    summarize()


if __name__ == "__main__":
    main()