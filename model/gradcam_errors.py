from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data import CLASSES, IMG_SIZE, get_transforms
from predict import DEVICE, MODEL_NAME, _load

THRESH = 0.98  # same threshold used for the overlap check
N_METRIC = 100  # images per group for the numeric comparison

model, cam = _load()
print("Model:", MODEL_NAME)

r = pd.read_csv(f"data/predictions/external_{MODEL_NAME}_probs.csv")
overlap = pd.read_csv("data/external_overlap.csv")
seen = {Path(p).as_posix() for p in overlap.loc[overlap["corr"] >= THRESH, "external_path"]}
r = r[~r["path"].map(lambda p: Path(p).as_posix() in seen)]  # clean images only

errors = r[(r.label == "NORMAL") & (r.pred4 != "Normal")]
ok_normal = r[(r.label == "NORMAL") & (r.pred4 == "Normal")]
ok_pneu = r[(r.label == "PNEUMONIA") & (r.pred4 != "Normal")]
print(f"clean images: errors={len(errors)}, correct normal={len(ok_normal)}, "
      f"pneumonia predicted pneumonia={len(ok_pneu)}")

tf = get_transforms(train=False)


def get_cam(path, cls_name):
    img = Image.open(path).convert("L").convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    g = cam(input_tensor=x, targets=[ClassifierOutputTarget(CLASSES.index(cls_name))])[0]
    return img.resize((IMG_SIZE, IMG_SIZE)), g


def border_share(g):
    """Share of Grad-CAM mass in the outer 20% band of the image.
    A uniform map would give 0.64. This is crude: no lung mask is used."""
    h, w = g.shape
    inner = np.zeros_like(g, dtype=bool)
    inner[int(.2 * h):int(.8 * h), int(.2 * w):int(.8 * w)] = True
    return float(g[~inner].sum() / g.sum())


# Numeric comparison
print(f"\nShare of heat in the outer border band (uniform = 0.64), {N_METRIC} images per group:")
for name, df in [("errors (normal -> predicted pneumonia)", errors),
                 ("normal predicted normal", ok_normal),
                 ("pneumonia predicted pneumonia", ok_pneu)]:
    sub = df.sample(min(N_METRIC, len(df)), random_state=0)
    vals = [border_share(get_cam(p, c)[1]) for p, c in zip(sub["path"], sub["pred4"])]
    print(f"  {name:42} mean {np.mean(vals):.3f} | median {np.median(vals):.3f} | n={len(vals)}")

# Figure: 8 errors + 4 correct normals
pick = pd.concat([errors.sample(8, random_state=1).assign(kind="ERROR"),
                  ok_normal.sample(4, random_state=1).assign(kind="correct")])
fig, ax = plt.subplots(3, 8, figsize=(18, 7.2))
for a in ax.ravel():
    a.axis("off")
for i, (_, row) in enumerate(pick.iterrows()):
    img, g = get_cam(row["path"], row["pred4"])
    overlay = show_cam_on_image(np.asarray(img, dtype=np.float32) / 255.0, g, use_rgb=True)
    r_i, c_i = divmod(i, 8)
    ax[r_i, c_i].imshow(overlay)
    ax[r_i, c_i].set_title(f"{row['kind']}: true NORMAL\npred {row['pred4']}", fontsize=8)
plt.tight_layout()
plt.savefig("analysis/gradcam_external_errors.png", dpi=110)
plt.show()