from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

EXT = {".jpeg", ".jpg", ".png"}


def vec(path):
    im = Image.open(path).convert("L").resize((32, 32), Image.BILINEAR)
    v = np.asarray(im, dtype=np.float32).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


ext_paths = sorted(str(p) for p in Path("data/external/chest_xray").rglob("*")
                   if p.suffix.lower() in EXT)
ours = pd.read_csv("data/all_images_clean.csv")

A = np.stack([vec(p) for p in tqdm(ext_paths, desc="external")])
B = np.stack([vec(p) for p in tqdm(ours["path"], desc="our data")])

best_idx = np.zeros(len(A), dtype=int)
best_corr = np.zeros(len(A), dtype=np.float32)
for i in range(0, len(A), 1000):
    s = A[i:i + 1000] @ B.T
    best_idx[i:i + 1000] = s.argmax(1)
    best_corr[i:i + 1000] = s.max(1)

res = pd.DataFrame({
    "external_path": ext_paths,
    "match_path": ours["path"].values[best_idx],
    "match_label": ours["label"].values[best_idx],
    "corr": best_corr,
})
res.to_csv("data/external_overlap.csv", index=False)

print("\nExternal images:", len(res))
for t in (0.99, 0.98, 0.95, 0.90):
    print(f"best correlation >= {t}: {(res['corr'] >= t).sum()}")
print("\nMatch labels among corr >= 0.98:")
print(res[res["corr"] >= 0.98]["match_label"].value_counts().to_string())
print("\nCorrelation quantiles:")
print(res["corr"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_string())

# Visual check: 6 high-correlation pairs and 3 low-correlation pairs
high = res[res["corr"] >= 0.98].sample(min(6, (res["corr"] >= 0.98).sum()), random_state=0)
low = res.nsmallest(3, "corr")
pick = pd.concat([high, low])
fig, ax = plt.subplots(len(pick), 2, figsize=(5, 2.4 * len(pick)))
ax = np.atleast_2d(ax)
for r, (_, row) in enumerate(pick.iterrows()):
    ax[r, 0].imshow(Image.open(row["external_path"]).convert("L"), cmap="gray")
    ax[r, 0].set_title("external", fontsize=7)
    ax[r, 1].imshow(Image.open(row["match_path"]).convert("L"), cmap="gray")
    ax[r, 1].set_title(f"our data ({row['match_label']}) corr={row['corr']:.3f}", fontsize=7)
    ax[r, 0].axis("off"); ax[r, 1].axis("off")
plt.tight_layout()
plt.savefig("analysis/overlap_examples.png", dpi=110)
plt.show()