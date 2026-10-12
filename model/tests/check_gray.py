import numpy as np
import pandas as pd
from PIL import Image

df = pd.read_csv("data/splits/test.csv").sample(500, random_state=0)
diffs = []
for p in df["path"]:
    arr = np.asarray(Image.open(p).convert("RGB").resize((64, 64)), dtype=np.float32)
    diffs.append(np.abs(arr[..., 0] - arr[..., 1]).mean() + np.abs(arr[..., 1] - arr[..., 2]).mean())

diffs = np.array(diffs)
print("Max color difference :", diffs.max().round(3))
print("Mean color difference:", diffs.mean().round(3))
print("Images above threshold (10):", int((diffs > 10).sum()), "of", len(diffs))