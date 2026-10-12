from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

src = Path("data/raw/COVID-19_Radiography_Dataset/COVID/images/COVID-100.png")
out = Path("data/test_images")
out.mkdir(parents=True, exist_ok=True)

# Test 1: a real X-ray tinted blue (colored image)
gray = Image.open(src).convert("L")
ImageOps.colorize(gray, black="#001030", white="#b8d8ff").save(out / "xray_blue_tint.png")

# Test 2: random colored noise (clearly not an X-ray)
rng = np.random.default_rng(0)
noise = rng.integers(0, 256, size=(299, 299, 3), dtype=np.uint8)
Image.fromarray(noise).save(out / "random_color_noise.png")

print("Saved to", out)