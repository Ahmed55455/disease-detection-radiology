"""Inference layer: checks the uploaded file, then calls Member 1's predict().

No Streamlit here, so a FastAPI API can reuse it later.
"""
import io
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "model"))   # works from any folder

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_BYTES = 10 * 1024 * 1024               # 10 MB
MIN_SIDE = 64                               # pixels; smaller images are not usable


class InvalidImageError(ValueError):
    """Raised with a message that can be shown to the user as is."""


def load_image(filename, data):
    """filename: original file name. data: file bytes. Returns an RGB PIL.Image."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidImageError(f"File type '{ext or 'none'}' is not allowed. "
                                "Please upload a PNG or JPG image.")
    if len(data) == 0:
        raise InvalidImageError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise InvalidImageError(f"The file is {len(data) / 1024 / 1024:.1f} MB. "
                                "The maximum size is 10 MB.")
    try:
        Image.open(io.BytesIO(data)).verify()       # detects broken files
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise InvalidImageError("The file could not be read as an image. "
                                "It may be damaged or not a real PNG/JPG.")
    if min(image.size) < MIN_SIDE:
        raise InvalidImageError(f"The image is too small ({image.size[0]}x{image.size[1]}). "
                                f"Minimum is {MIN_SIDE}x{MIN_SIDE} pixels.")
    return image.convert("RGB")


def analyze(image):
    """Run the model. Always returns the same keys, even with an older predict()."""
    from predict import predict          # imported here: loads PyTorch only when needed
    res = predict(image)
    probs = res["probabilities"]
    res.setdefault("confidence", probs[res["class"]])
    res.setdefault("model_name", "unknown")
    res.setdefault("warnings", [])
    return res
