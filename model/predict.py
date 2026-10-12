import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data import CLASSES, IMG_SIZE, get_transforms
from train import build_model

# Paths are relative to this file, so predict() works from any folder
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "efficientnet_b0"
CKPT = REPO_ROOT / "model" / "checkpoints" / "efficientnet_b0_8ep_best.pt"

# Rough heuristic: chest X-rays in the training data are grayscale
COLOR_DIFF_THRESHOLD = 10.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_cam = None


def _target_layer(model, name):
    if name == "resnet50":
        return [model.layer4[-1]]
    return [model.features[-1]]  # densenet121 and efficientnet_b0


def _load():
    global _model, _cam
    if _model is None:
        m = build_model(MODEL_NAME)
        m.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
        _model = m.to(DEVICE).eval()
        _cam = GradCAM(model=_model, target_layers=_target_layer(_model, MODEL_NAME))
    return _model, _cam


def _check_image(image):
    """Return a list of warning strings (empty list = no warning)."""
    warnings = []
    arr = np.asarray(image.resize((64, 64)), dtype=np.float32)
    color_diff = (np.abs(arr[..., 0] - arr[..., 1]).mean()
                  + np.abs(arr[..., 1] - arr[..., 2]).mean())
    if color_diff > COLOR_DIFF_THRESHOLD:
        warnings.append("Image is colored. Chest X-rays are normally grayscale; "
                        "this may not be a chest X-ray.")
    return warnings


def predict(image):
    """image: file path (str/Path) or PIL.Image.
    Returns {
      "class": str,
      "confidence": float,              # probability of the predicted class
      "probabilities": {class: float},
      "heatmap": np.ndarray (224, 224, 3) uint8 RGB,
      "model_name": str,
      "warnings": [str],                # empty if nothing suspicious
    }
    """
    model, cam = _load()
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    warnings = _check_image(image)
    image = image.convert("L").convert("RGB")  # remove color, as in the training data

    x = get_transforms(train=False)(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x).float(), dim=1)[0].cpu().numpy()
    top = int(probs.argmax())
    
    grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(top)])[0]
    rgb = np.asarray(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    heatmap = show_cam_on_image(rgb, grayscale, use_rgb=True)

    return {
        "class": CLASSES[top],
        "confidence": float(probs[top]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
        "heatmap": heatmap,
        "model_name": MODEL_NAME,
        "warnings": warnings,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--save", default=None,
                    help="Optional output PNG path. Nothing is saved unless this is given.")
    args = ap.parse_args()

    if args.image is None:
        test_csv = REPO_ROOT / "data" / "splits" / "test.csv"
        args.image = pd.read_csv(test_csv).sample(1, random_state=3).iloc[0]["path"]

    res = predict(args.image)
    print("Image      :", args.image)
    print("Class      :", res["class"], f"(confidence {res['confidence']:.3f})")
    print("Model      :", res["model_name"])
    for c, p in res["probabilities"].items():
        print(f"  {c:16} {p:.3f}")
    for w in res["warnings"]:
        print("WARNING    :", w)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(Image.open(args.image).convert("RGB")); ax[0].set_title("Original")
    ax[1].imshow(res["heatmap"]); ax[1].set_title(f"Grad-CAM: {res['class']}")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=130)
    plt.show()