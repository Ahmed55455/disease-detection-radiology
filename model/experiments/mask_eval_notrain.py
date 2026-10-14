import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import CLASSES, CLASS_TO_IDX, get_transforms
from train import build_model

MODELS = {
    "resnet50": "model/checkpoints/resnet50_8ep_best.pt",
    "efficientnet_b0": "model/checkpoints/efficientnet_b0_8ep_best.pt",
    "densenet121": "model/checkpoints/densenet121_8ep_best.pt",
}


class DS(Dataset):
    def __init__(self, df, masked):
        self.paths = df["path"].tolist()
        self.masked = masked
        self.tf = get_transforms(train=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = Path(self.paths[i])
        img = Image.open(p).convert("L")
        if self.masked:
            m = Image.open(p.parent.parent / "masks" / p.name).convert("L")
            m = m.resize(img.size, Image.NEAREST)
            img = Image.fromarray((np.asarray(img) * (np.asarray(m) > 127)).astype(np.uint8))
        return self.tf(img.convert("RGB"))


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    out = []
    for x in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            out.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv("data/splits/test.csv")
    y = df["label"].map(CLASS_TO_IDX).values
    rows = []
    for name, ckpt in MODELS.items():
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        for masked in (False, True):
            loader = DataLoader(DS(df, masked), batch_size=32, num_workers=2, pin_memory=True)
            pred = run(model, loader, device)
            rec = recall_score(y, pred, average=None, labels=list(range(4)))
            rows.append({"model": name, "input": "masked" if masked else "original",
                         "accuracy": (pred == y).mean(), "macro_f1": f1_score(y, pred, average="macro"),
                         **{f"recall_{c}": r for c, r in zip(CLASSES, rec)}})
    out = pd.DataFrame(rows)
    out.to_csv("analysis/mask_eval_notrain.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()