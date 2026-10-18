"""Pick demo images for the presentation and run predict() on them.

Run from the repository root:
    python model/tests/make_demo_images.py

Output (local only, add demo_images/ to .gitignore):
    demo_images/<name>.png         the images
    demo_images/<name>_cam.png     Grad-CAM overlay from predict()
    demo_images/summary.csv        what was picked and what the model said

Selection uses the TEST split predictions only (no tuning involved):
  - one confident, correct example per class (random among confident correct ones)
  - one real error (the most confident wrong prediction), to show honestly
  - a blue-tinted copy of the COVID example (shows the color warning)
  - a random noise image (shows the model has no "not an X-ray" option)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model"))
from predict import predict  # noqa: E402

CSV = REPO_ROOT / "data" / "predictions" / "efficientnet_b0_test_probs.csv"
OUT = REPO_ROOT / "demo_images"
OUT.mkdir(exist_ok=True)
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
rng = np.random.default_rng(0)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() and p.exists() else REPO_ROOT / p


df = pd.read_csv(CSV)
pcols = [f"p_{c}" for c in CLASSES]
df["conf"] = df[pcols].max(axis=1)

picked = []  # (name, PIL image, true label, source path)

for c in CLASSES:
    ok = df[(df["label"] == c) & (df["pred"] == c) & (df["conf"] > 0.9)]
    row = ok.iloc[int(rng.integers(len(ok)))]
    picked.append((f"{c.replace(' ', '_')}_correct", Image.open(resolve(row["path"])), c, row["path"]))

wrong = df[df["label"] != df["pred"]].sort_values("conf", ascending=False)
row = wrong.iloc[0]
picked.append((f"error_true_{row['label'].replace(' ', '_')}_pred_{row['pred'].replace(' ', '_')}",
               Image.open(resolve(row["path"])), row["label"], row["path"]))

# blue-tinted copy of the COVID example
covid_img = picked[0][1].convert("RGB")
arr = np.asarray(covid_img).astype(np.float32)
arr[..., 0] *= 0.6
arr[..., 1] *= 0.8
arr[..., 2] = np.clip(arr[..., 2] * 1.2 + 20, 0, 255)
picked.append(("COVID_blue_tint", Image.fromarray(arr.astype(np.uint8)), "COVID", "synthetic tint"))

# random colored noise
noise = rng.integers(0, 256, size=(299, 299, 3), dtype=np.uint8)
picked.append(("random_noise", Image.fromarray(noise), "none", "synthetic noise"))

rows = []
for name, img, true_label, src in picked:
    img_path = OUT / f"{name}.png"
    img.save(img_path)
    res = predict(img_path)
    Image.fromarray(res["heatmap"]).save(OUT / f"{name}_cam.png")
    rows.append({
        "name": name,
        "true_label": true_label,
        "predicted": res["class"],
        "confidence": round(res["confidence"], 3),
        "warnings": " | ".join(res["warnings"]),
        "source": src,
    })
    print(f"{name:45s} true={true_label:16s} pred={res['class']:16s} conf={res['confidence']:.3f} warn={res['warnings']}")

pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
print(f"\nSaved to {OUT}")