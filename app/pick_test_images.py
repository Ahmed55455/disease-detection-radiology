"""Copy a few TEST-set images into a demo folder (one sub-folder per class).

Uses the test predictions CSV from Drive (column "path" has the test image paths and
"label" the true class) and the Kaggle "COVID-19 Radiography Database" on your laptop.
Only test images are copied, never training images, so the demo is honest.

Run from the repository root:
    python app/pick_test_images.py --csv "C:\\...\\efficientnet_b0_test_probs.csv" ^
        --dataset "C:\\...\\COVID-19_Radiography_Dataset" --out "C:\\...\\demo_images" --per-class 5
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Test predictions CSV (has 'path' and 'label')")
    ap.add_argument("--dataset", required=True, help="Unzipped Kaggle dataset folder")
    ap.add_argument("--out", required=True, help="Output folder for demo images")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not {"path", "label"} <= set(df.columns):
        sys.exit(f"CSV must have 'path' and 'label' columns. Found: {list(df.columns)}")

    # Index every image in the dataset by file name (e.g. COVID-123.png)
    dataset = Path(args.dataset)
    # The dataset also has lung masks with the SAME file names, so skip "masks" folders
    by_name = {p.name: p for p in dataset.rglob("*.png")
               if "mask" not in [part.lower().rstrip("s") for part in p.parts]}
    if not by_name:
        sys.exit(f"No PNG images found under {dataset}")

    out = Path(args.out)
    copied = 0
    for cls in CLASSES:
        rows = df[df["label"] == cls]
        rows = rows.sample(frac=1, random_state=args.seed)       # random order
        (out / cls).mkdir(parents=True, exist_ok=True)
        n = 0
        for p in rows["path"]:
            name = Path(str(p).replace("\\", "/")).name
            src = by_name.get(name)
            if src is None:
                continue
            shutil.copy2(src, out / cls / name)
            n += 1
            if n == args.per_class:
                break
        copied += n
        print(f"{cls:16} {n} images")
    print(f"\nCopied {copied} test images to {out}")


if __name__ == "__main__":
    main()
