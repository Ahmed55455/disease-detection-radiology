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

from data import CLASSES, CLASS_TO_IDX, compute_class_weights, get_transforms
from train import build_model, run_epoch


class MaskedDS(Dataset):
    def __init__(self, csv_path, train, masked=True):
        self.df = pd.read_csv(csv_path)
        self.masked = masked
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        p = Path(row["path"])
        img = Image.open(p).convert("L")
        if self.masked:
            m = Image.open(p.parent.parent / "masks" / p.name).convert("L")
            m = m.resize(img.size, Image.NEAREST)
            img = Image.fromarray((np.asarray(img) * (np.asarray(m) > 127)).astype(np.uint8))
        return self.tf(img.convert("RGB")), CLASS_TO_IDX[row["label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mk = lambda csv, train, masked=True: DataLoader(
        MaskedDS(f"data/splits/{csv}.csv", train, masked), batch_size=args.batch,
        shuffle=train, num_workers=2, pin_memory=True)
    train_loader, val_loader = mk("train", True), mk("val", False)
    test_masked, test_orig = mk("test", False), mk("test", False, masked=False)

    weights = compute_class_weights("data/splits/train.csv").to(device)
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    ckpt = f"model/checkpoints/{args.model}_masked_best.pt"
    best = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        vl, va, vf, vr = run_epoch(model, val_loader, criterion, device)
        print(f"Epoch {ep}/{args.epochs} ({time.time()-t0:.0f}s) | train loss {tl:.3f} acc {ta:.3f} "
              f"| val loss {vl:.3f} acc {va:.3f} macroF1 {vf:.3f}")
        print("  val recall:", {c: round(float(r), 3) for c, r in zip(CLASSES, vr)})
        scheduler.step()
        if vf > best:
            best = vf
            torch.save(model.state_dict(), ckpt)
            print("  saved best model")

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    rows = []
    for name, loader in [("test_masked", test_masked), ("test_original", test_orig)]:
        _, acc, f1, rec = run_epoch(model, loader, criterion, device)
        rows.append({"model": f"{args.model}_masked_trained", "eval_on": name, "accuracy": acc,
                     "macro_f1": f1, **{f"recall_{c}": r for c, r in zip(CLASSES, rec)}})
    out = pd.DataFrame(rows)
    out.to_csv("analysis/mask_trained_eval.csv", index=False)
    print()
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()