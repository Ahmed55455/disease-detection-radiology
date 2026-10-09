from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/raw/COVID-19_Radiography_Dataset")
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

rows = []
for cls in CLASSES:
    for p in (ROOT / cls / "images").glob("*.png"):
        rows.append({"path": str(p), "label": cls})

df = pd.DataFrame(rows)
print("Total images:", len(df))
print(df["label"].value_counts())

# Plot class distribution
counts = df["label"].value_counts()
ax = counts.plot(kind="bar", color="steelblue")
ax.set_title("Class distribution")
ax.set_ylabel("Number of images")
for i, v in enumerate(counts):
    ax.text(i, v + 100, str(v), ha="center")
plt.tight_layout()
Path("analysis").mkdir(exist_ok=True)
plt.savefig("analysis/class_distribution.png", dpi=150)
plt.show()

# Image sizes and modes (sample of 200 images)
sample = df.sample(200, random_state=42)
sizes = {Image.open(p).size for p in sample["path"]}
modes = {Image.open(p).mode for p in sample["path"]}
print("Image sizes found:", sizes)
print("Image modes found:", modes)

# Show 4 random images per class
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for r, cls in enumerate(CLASSES):
    paths = df[df["label"] == cls].sample(4, random_state=1)["path"]
    for c, p in enumerate(paths):
        axes[r, c].imshow(Image.open(p).convert("L"), cmap="gray")
        axes[r, c].axis("off")
        if c == 0:
            axes[r, c].set_title(cls, loc="left")
plt.tight_layout()
plt.savefig("analysis/samples.png", dpi=120)
plt.show()

df.to_csv("data/all_images.csv", index=False)