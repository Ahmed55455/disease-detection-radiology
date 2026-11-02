"""Copy TEST-set images into a demo folder (one sub-folder per TRUE class).

Uses the test predictions CSV from Drive (column "path" = test image path, "label" = true class,
"pred" = model prediction) and the Kaggle "COVID-19 Radiography Database" on your laptop.
Only test images are copied, never training images, so the demo is honest.

Options:
  --per-class N   N random test images per class (default 5)
  --seed S        change it to get DIFFERENT images (default 7)
  --errors N      also add N test images that the model gets WRONG (shows that the
                  radiologist must check; their file names start with "WRONG_")
  --clean         empty the output folder first (otherwise new images are added)

Run from the repository root, for example:
    python app/pick_test_images.py --csv data\\predictions\\efficientnet_b0_test_probs.csv ^
        --dataset "%USERPROFILE%\\Downloads\\archive\\COVID-19_Radiography_Dataset" ^
        --out "%USERPROFILE%\\OneDrive\\Desktop\\demo_images" --per-class 5 --errors 3 --seed 11 --clean
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Test predictions CSV (path, label, pred)")
    ap.add_argument("--dataset", required=True, help="Unzipped Kaggle dataset folder")
    ap.add_argument("--out", required=True, help="Output folder for demo images")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--errors", type=int, default=0, help="Extra images the model gets wrong")
    ap.add_argument("--seed", type=int, default=7, help="Change it to get different images")
    ap.add_argument("--clean", action="store_true", help="Empty the output folder first")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not {"path", "label"} <= set(df.columns):
        sys.exit(f"CSV must have 'path' and 'label' columns. Found: {list(df.columns)}")
    if args.errors and "pred" not in df.columns:
        sys.exit("--errors needs a 'pred' column in the CSV")
    df["name"] = [Path(str(p).replace("\\", "/")).name for p in df["path"]]

    # Index every image in the dataset by file name (e.g. COVID-123.png).
    # The dataset also has lung masks with the SAME file names, so skip "masks" folders.
    dataset = Path(args.dataset)
    by_name = {p.name: p for p in dataset.rglob("*.png")
               if "mask" not in [part.lower().rstrip("s") for part in p.parts]}
    if not by_name:
        sys.exit(f"No PNG images found under {dataset}")

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    for cls in CLASSES:
        (out / cls).mkdir(parents=True, exist_ok=True)

    df = df[df["name"].isin(by_name)].sample(frac=1, random_state=args.seed)   # random order
    copied = 0
    for cls in CLASSES:
        rows = df[df["label"] == cls]
        if "pred" in df.columns:
            rows = rows[rows["pred"] == rows["label"]]      # errors are added separately below
        for name in rows["name"][: args.per_class]:
            shutil.copy2(by_name[name], out / cls / name)
            copied += 1
        print(f"{cls:16} {min(len(rows), args.per_class)} images")

    if args.errors:
        wrong = df[df["pred"] != df["label"]][: args.errors]
        for _, r in wrong.iterrows():
            shutil.copy2(by_name[r["name"]], out / r["label"] / f"WRONG_{r['name']}")
            copied += 1
            print(f"WRONG            true {r['label']:16} model said {r['pred']}")

    print(f"\nCopied {copied} test images to {out}")


if __name__ == "__main__":
    main()
