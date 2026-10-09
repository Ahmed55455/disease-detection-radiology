import importlib

packages = [
    "torch", "torchvision", "numpy", "pandas", "sklearn",
    "matplotlib", "seaborn", "PIL", "tqdm", "cv2",
    "jupyter_core", "ipykernel", "pytorch_grad_cam",
]

for name in packages:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "installed")
        print(f"[OK]      {name:18} {version}")
    except ImportError:
        print(f"[MISSING] {name}")

try:
    import torch
    print("\nCUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except ImportError:
    pass