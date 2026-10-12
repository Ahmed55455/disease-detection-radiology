from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import CLASSES, get_transforms
from train import build_model

ROOT = Path("data/external/chest_xray")
EXT = {".jpeg", ".jpg", ".png"}
NORMAL_IDX = CLASSES.index("Normal")

MODELS = {
    "resnet50": "model/checkpoints/resnet50_8ep_best.pt",
    "efficientnet_b0": "model/checkpoints/efficientnet_b0_8ep_best.pt",
    "densenet121": "model/checkpoints/densenet121_8ep_best.pt",
}


def collect():
    rows = []
    for split in ["train", "val", "test"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            for p in (ROOT / split / label).iterdir():
                if p.suffix.lower() in EXT:
                    rows.append({"path": str(p), "label": label, "split": split})
    return pd.DataFrame(rows)


class ExtDataset(Dataset):
    def __init__(self, df):
        self.paths = df["path"].tolist()
        self.tf = get_transforms(train=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # same preprocessing as predict(): grayscale -> 3 channels
        img = Image.open(self.paths[i]).convert("L").convert("RGB")
        return self.tf(img)


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    out = []
    for x in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(x)
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = collect()
    print("Images:", len(df))
    print(df["label"].value_counts().to_string())

    # sanity check: are these images grayscale like the training data?
    diffs = []
    for p in df.sample(300, random_state=0)["path"]:
        a = np.asarray(Image.open(p).convert("RGB").resize((64, 64)), dtype=np.float32)
        diffs.append(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean())
    print(f"Color check: {sum(d > 10 for d in diffs)} of 300 sampled images above threshold 10\n")

    y_true = (df["label"] == "PNEUMONIA").astype(int).values  # 1 = pneumonia
    loader = DataLoader(ExtDataset(df), batch_size=32, shuffle=False,
                        num_workers=2, pin_memory=True)

    Path("analysis").mkdir(exist_ok=True)
    Path("data/predictions").mkdir(parents=True, exist_ok=True)
    summary = []

    for name, ckpt in MODELS.items():
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        probs = predict_all(model, loader, device)

        pred4 = probs.argmax(1)
        y_pred = (pred4 != NORMAL_IDX).astype(int)
        p_pneu = 1.0 - probs[:, NORMAL_IDX]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        prec = tp / (tp + fp)
        bal = balanced_accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, p_pneu)

        print(f"===== {name} =====")
        print(f"TN {tn} | FP {fp} | FN {fn} | TP {tp}")
        print(f"Sensitivity (pneumonia found): {sens:.3f}")
        print(f"Specificity (normal found)   : {spec:.3f}")
        print(f"Precision (pneumonia)        : {prec:.3f}")
        print(f"Balanced accuracy            : {bal:.3f}")
        print(f"ROC-AUC (1 - p_Normal)       : {auc:.4f}")
        print("What the 4-class model predicted:")
        pred_names = pd.Series([CLASSES[i] for i in pred4], name="predicted")
        print(pd.crosstab(df["label"], pred_names).to_string(), "\n")

        summary.append({"model": name, "sensitivity": sens, "specificity": spec,
                        "precision": prec, "balanced_acc": bal, "auc": auc,
                        "TN": tn, "FP": fp, "FN": fn, "TP": tp})

        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["NORMAL", "PNEUMONIA"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["NORMAL", "PNEUMONIA"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"External: {name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        plt.savefig(f"analysis/external_confusion_{name}.png", dpi=130)
        plt.close()

        res = df.copy()
        for i, c in enumerate(CLASSES):
            res[f"p_{c}"] = probs[:, i]
        res["pred4"] = [CLASSES[i] for i in pred4]
        res["pred_binary"] = np.where(y_pred == 1, "PNEUMONIA", "NORMAL")
        res.to_csv(f"data/predictions/external_{name}_probs.csv", index=False)

    pd.DataFrame(summary).to_csv("analysis/external_generalization.csv", index=False)
    print(pd.DataFrame(summary).round(3).to_string())


if __name__ == "__main__":
    main()