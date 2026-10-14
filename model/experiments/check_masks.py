from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

df = pd.read_csv("data/all_images_clean.csv")


def mask_path(p):
    p = Path(p)
    return p.parent.parent / "masks" / p.name


has_mask = df["path"].map(lambda p: mask_path(p).exists())
print("Images:", len(df), "| with a mask file:", int(has_mask.sum()))
print(df[~has_mask]["label"].value_counts().to_string() if (~has_mask).any() else "No missing masks")

sample = df[has_mask].sample(300, random_state=0)
img_sizes, mask_sizes, vals, lung_frac = set(), set(), set(), []
for p in sample["path"]:
    im = Image.open(p)
    mk = Image.open(mask_path(p))
    img_sizes.add(im.size)
    mask_sizes.add(mk.size)
    a = np.asarray(mk.convert("L"))
    vals.update(np.unique(a).tolist()[:5])
    lung_frac.append((a > 127).mean())

print("Image sizes :", img_sizes)
print("Mask sizes  :", mask_sizes)
print("Some mask pixel values:", sorted(vals)[:6])
print(f"Lung area share of the image: mean {np.mean(lung_frac):.2f}, "
      f"min {np.min(lung_frac):.2f}, max {np.max(lung_frac):.2f}")

# Visual check: image | mask | image with outside-lung area blacked out
pick = df[has_mask].groupby("label").sample(1, random_state=1)
fig, ax = plt.subplots(len(pick), 3, figsize=(7, 2.4 * len(pick)))
for r, (_, row) in enumerate(pick.iterrows()):
    im = Image.open(row["path"]).convert("L")
    mk = Image.open(mask_path(row["path"])).convert("L").resize(im.size)
    masked = np.asarray(im) * (np.asarray(mk) > 127)
    for c, (arr, t) in enumerate([(im, row["label"]), (mk, "mask"), (masked, "masked")]):
        ax[r, c].imshow(arr, cmap="gray"); ax[r, c].set_title(t, fontsize=8); ax[r, c].axis("off")
plt.tight_layout()
plt.savefig("analysis/mask_examples.png", dpi=110)
plt.show()