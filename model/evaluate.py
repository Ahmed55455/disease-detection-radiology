from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from tqdm import tqdm

from data import CLASSES, CLASS_TO_IDX, get_loaders
from train import build_model

MODELS = {
    "resnet50": "model/checkpoints/resnet50_8ep_best.pt",
    "efficientnet_b0": "model/checkpoints/efficientnet_b0_8ep_best.pt",
    "densenet121": "model/checkpoints/densenet121_8ep_best.pt",
}


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    probs = []
    for x, _ in tqdm(loader, leave=False, desc="test"):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            out = model(x)
        probs.append(torch.softmax(out.float(), dim=1).cpu().numpy())
    return np.concatenate(probs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader = get_loaders(batch_size=32, num_workers=2)
    test_df = pd.read_csv("data/splits/test.csv")
    y_true = test_df["label"].map(CLASS_TO_IDX).values

    Path("analysis").mkdir(exist_ok=True)
    Path("data/predictions").mkdir(parents=True, exist_ok=True)
    summary = []

    for name, ckpt in MODELS.items():
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        probs = predict_all(model, test_loader, device)
        y_pred = probs.argmax(1)

        print(f"\n===== {name} =====")
        print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        acc = (y_pred == y_true).mean()
        print(f"Accuracy {acc:.3f} | macro-F1 {macro_f1:.3f} | macro ROC-AUC {auc:.4f}")
        summary.append({"model": name, "accuracy": acc, "macro_f1": macro_f1, "macro_auc": auc})

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(name)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        plt.savefig(f"analysis/confusion_{name}.png", dpi=130)
        plt.close()

        # Save probabilities for the Industrial Engineering teammate
        out = test_df[["path", "label"]].copy()
        for i, c in enumerate(CLASSES):
            out[f"p_{c}"] = probs[:, i]
        out["pred"] = [CLASSES[i] for i in y_pred]
        out.to_csv(f"data/predictions/{name}_test_probs.csv", index=False)

    pd.DataFrame(summary).to_csv("analysis/model_comparison_test.csv", index=False)
    print("\n", pd.DataFrame(summary).round(4))


if __name__ == "__main__":
    main()