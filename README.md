# Disease Detection from Radiology Images Using Deep Learning

AI-assisted chest X-ray classification (COVID, Lung_Opacity, Normal, Viral Pneumonia)
with Grad-CAM explanations and radiologist worklist prioritization.
The system assists radiologists and does not replace them. It is limited by its training data.

## Setup

Requirements: Python 3.11. A GPU is optional; `predict.py` also runs on CPU, only slower.

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install PyTorch first (GPU, CUDA 12.1), then the other libraries of the model part:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r model/requirements.txt
```

For CPU only, use `pip install torch==2.5.1 torchvision==0.20.1` as the first command.

## Data (not in GitHub)
Download "COVID-19 Radiography Database" from Kaggle (tawsifurrahman) and extract to
`data/raw/COVID-19_Radiography_Dataset/`.

## Pipeline (run from the repo root)
```bash
python model/prep/eda.py
python model/prep/check_duplicates.py
python model/prep/split_data.py
python model/train.py --model efficientnet_b0 --epochs 8 --batch 16
python model/evaluate.py
python model/predict.py
```

## Using the trained model

The model file is not in GitHub. Download it from the shared Google Drive folder
`radiology-checkpoints` and place it here:

```
model/checkpoints/efficientnet_b0_8ep_best.pt
```

Then, from Python:

```python
from predict import predict
result = predict("path/to/xray.png")
print(result["class"], result["confidence"], result["warnings"])
```

The format of the result is described in `docs/interfaces.md`.

## Folders
- `/model`: data pipeline, training, evaluation, predict()
- `/app`: web application
- `/analysis`: statistics, optimization, simulation
- `/docs`: interfaces, model findings and report material

## Documentation
- `docs/interfaces.md`: agreed interfaces between the three parts (model, app, analysis).
- `docs/model_findings.md`: results, experiments and limitations of the model part.