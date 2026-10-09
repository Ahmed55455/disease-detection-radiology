# Disease Detection from Radiology Images Using Deep Learning

AI-assisted chest X-ray classification (COVID, Lung_Opacity, Normal, Viral Pneumonia)
with Grad-CAM explanations and radiologist worklist prioritization.
The system assists radiologists and does not replace them. It is limited by its training data.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn matplotlib seaborn pillow tqdm opencv-python jupyter ipykernel grad-cam
```

## Data (not in GitHub)
Download "COVID-19 Radiography Database" from Kaggle (tawsifurrahman) and extract to
`data/raw/COVID-19_Radiography_Dataset/`.

## Pipeline (run from the repo root)
```bash
python model/eda.py
python model/check_duplicates.py
python model/split_data.py
python model/train.py --model efficientnet_b0 --epochs 8 --batch 16
python model/evaluate.py
python model/predict.py
```

## Folders
- `/model`: data pipeline, training, evaluation, predict()
- `/app`: web application
- `/analysis`: statistics, optimization, simulation
- `/docs`: interfaces and report material

See `docs/interfaces.md` for the agreed interfaces.