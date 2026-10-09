import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from torchvision import models
from tqdm import tqdm

from data import CLASSES, compute_class_weights, get_loaders


def build_model(name: str, num_classes: int = 4):
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(name)
    return m


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            preds += out.argmax(1).cpu().tolist()
            targets += y.cpu().tolist()
    n = len(loader.dataset)
    acc = sum(int(p == t) for p, t in zip(preds, targets)) / n
    macro_f1 = f1_score(targets, preds, average="macro")
    recalls = recall_score(targets, preds, average=None, labels=list(range(len(CLASSES))))
    return total_loss / n, acc, macro_f1, recalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "| Model:", args.model)

    train_loader, val_loader, _ = get_loaders(batch_size=args.batch, num_workers=2)
    weights = compute_class_weights("data/splits/train.csv").to(device)

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    Path("model/checkpoints").mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va_loss, va_acc, va_f1, va_rec = run_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} ({time.time()-t0:.0f}s) "
              f"| train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f} macroF1 {va_f1:.3f}")
        print("  val recall per class:",
              {c: round(float(r), 3) for c, r in zip(CLASSES, va_rec)})
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), f"model/checkpoints/{args.model}_best.pt")
            print("  saved best model")


if __name__ == "__main__":
    main()