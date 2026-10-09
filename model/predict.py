from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data import CLASSES, IMG_SIZE, get_transforms
from train import build_model

MODEL_NAME = "efficientnet_b0"
CKPT = "model/checkpoints/efficientnet_b0_8ep_best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_cam = None


def _target_layer(model, name):
    if name == "resnet50":
        return [model.layer4[-1]]
    if name == "densenet121":
        return [model.features[-1]]
    return [model.features[-1]]  # efficientnet_b0


def _load():
    global _model, _cam
    if _model is None:
        m = build_model(MODEL_NAME)
        m.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
        _model = m.to(DEVICE).eval()
        _cam = GradCAM(model=_model, target_layers=_target_layer(_model, MODEL_NAME))
    return _model, _cam


def predict(image):
    """image: file path (str/Path) or PIL.Image.
    Returns {"class": str, "probabilities": {class: float}, "heatmap": np.ndarray HxWx3 uint8}
    """
    model, cam = _load()
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    x = get_transforms(train=False)(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x).float(), dim=1)[0].cpu().numpy()
    top = int(probs.argmax())

    # Grad-CAM for the predicted class
    grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(top)])[0]
    rgb = np.asarray(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    heatmap = show_cam_on_image(rgb, grayscale, use_rgb=True)

    return {
        "class": CLASSES[top],
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
        "heatmap": heatmap,
    }


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    import pandas as pd

    path = sys.argv[1] if len(sys.argv) > 1 else pd.read_csv("data/splits/test.csv").sample(1, random_state=3).iloc[0]["path"]
    res = predict(path)
    print("Image :", path)
    print("Class :", res["class"])
    for c, p in res["probabilities"].items():
        print(f"  {c:16} {p:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(Image.open(path).convert("RGB")); ax[0].set_title("Original")
    ax[1].imshow(res["heatmap"]); ax[1].set_title(f"Grad-CAM: {res['class']}")
    for a in ax: a.axis("off")
    plt.tight_layout()
    Path("analysis").mkdir(exist_ok=True)
    plt.savefig("analysis/gradcam_example.png", dpi=130)
    plt.show()