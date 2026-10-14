from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def feats(path):
    p = Path(path)
    m = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(
        p.parent.parent / "masks" / p.name).convert("L")) > 127
    h, w = m.shape
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return [0.0] * 12
    left, right = m[:, : w // 2], m[:, w // 2:]
    return [m.mean(), left.mean(), right.mean(),
            ys.min() / h, ys.max() / h, xs.min() / w, xs.max() / w,
            (ys.max() - ys.min()) / max(xs.max() - xs.min(), 1),
            ys.mean() / h, xs.mean() / w, ys.std() / h, xs.std() / w]


def load(split):
    df = pd.read_csv(f"data/splits/{split}.csv")
    X = np.array([feats(p) for p in tqdm(df["path"], desc=split)])
    return X, df["label"].map({c: i for i, c in enumerate(CLASSES)}).values


Xtr, ytr = load("train")
Xte, yte = load("test")

clf = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)

print("\nMajority-class baseline accuracy:", round(float(np.mean(yte == np.bincount(ytr).argmax())), 3))
print("Mask-shape-only classifier: accuracy", round(float((pred == yte).mean()), 3),
      "| macro-F1", round(f1_score(yte, pred, average="macro"), 3))
rec = recall_score(yte, pred, average=None, labels=list(range(4)))
print({c: round(float(r), 3) for c, r in zip(CLASSES, rec)})