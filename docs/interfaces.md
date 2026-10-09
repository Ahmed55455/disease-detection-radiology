# Interfaces Agreement (docs/interfaces.md)

Project: Disease Detection from Radiology Images Using Deep Learning
Status: DRAFT v0.1, written by Member 1 (AI Model). Members 2 and 3 should review and comment.

The system assists radiologists. It does not replace them, and it is limited by its training data.

---

## 1. Class order (fixed, do not change)

| Index | Class name (exact string) |
|---|---|
| 0 | COVID |
| 1 | Lung_Opacity |
| 2 | Normal |
| 3 | Viral Pneumonia |

Everyone must use these exact names and this exact order.

---

## 2. Model output: `predict(image)` (for Member 2, the app)

File: `model/predict.py`

```python
from predict import predict
result = predict(image)   # image: file path (str/Path) or PIL.Image
```

Returns a dict:

```python
{
  "class": "COVID",                      # str, one of the 4 class names
  "probabilities": {                     # dict[str, float], sums to 1.0
      "COVID": 0.97,
      "Lung_Opacity": 0.01,
      "Normal": 0.01,
      "Viral Pneumonia": 0.01,
  },
  "heatmap": <numpy.ndarray>             # shape (224, 224, 3), dtype uint8, RGB,
}                                        # Grad-CAM overlay for the predicted class
```

Notes:
- Confidence = `probabilities[result["class"]]`.
- The model is loaded once, on the first call (lazy loading). The first call is slower.
- Current model: EfficientNet-B0 (checkpoint `model/checkpoints/efficientnet_b0_8ep_best.pt`). It can be switched in `predict.py` (`MODEL_NAME`, `CKPT`) without changing the output format.
- Checkpoints (`*.pt`) are NOT in GitHub. They are shared through Google Drive (folder `radiology-checkpoints`).
- To show the heatmap in Streamlit: `st.image(result["heatmap"])`.
- Grad-CAM is approximate (low-resolution feature map, 7x7, upsampled). It is an aid to explanation, not an exact localization.

---

## 3. Worklist fields (proposal for Member 2)

SQLite table `cases` (suggested):

| Field | Type | Description |
|---|---|---|
| case_id | INTEGER PK | Auto id |
| image_path | TEXT | Stored image file |
| uploaded_at | DATETIME | Arrival time (needed by the simulation) |
| predicted_class | TEXT | One of the 4 class names |
| p_covid, p_lung_opacity, p_normal, p_viral_pneumonia | REAL | Class probabilities |
| confidence | REAL | Probability of the predicted class |
| urgency_score | REAL | 0 to 1, see section 5 |
| heatmap_path | TEXT | Saved Grad-CAM image |
| status | TEXT | `pending`, `in_review`, `done` |
| read_at | DATETIME | When the radiologist read it (needed for time-to-diagnosis) |
| radiologist_label | TEXT | Final label entered by the radiologist (optional) |

Worklist order: `urgency_score` descending, ties broken by `uploaded_at` ascending.

---

## 4. Test predictions file (for Member 3, evaluation and simulation)

Location (local, not in GitHub): `data/predictions/<model>_test_probs.csv`
Models available: `resnet50`, `efficientnet_b0`, `densenet121`.
Shared with the team through Google Drive.

Columns:

| Column | Description |
|---|---|
| path | Image path |
| label | True class |
| p_COVID, p_Lung_Opacity, p_Normal, p_Viral Pneumonia | Predicted probabilities |
| pred | Predicted class (argmax) |

Test set: 3167 images (Normal 1529, Lung_Opacity 902, COVID 535, Viral Pneumonia 201).
The Viral Pneumonia class has only 201 test images, so its metrics have wider confidence intervals.
Summary table: `analysis/model_comparison_test.csv`.

Current test results (single run per model, not statistically compared yet):

| Model | Accuracy | macro-F1 | macro ROC-AUC |
|---|---|---|---|
| ResNet50 | 0.959 | 0.969 | 0.9950 |
| EfficientNet-B0 | 0.955 | 0.964 | 0.9954 |
| DenseNet121 | 0.962 | 0.971 | 0.9950 |

The differences are small. The statistical comparison is Member 3's task.

---

## 5. Urgency score (initial proposal, Member 3 to finalize)

Placeholder rule so Member 2 can build the worklist now:

```
urgency_score = sum over classes of ( probability[class] * severity[class] )
severity = {COVID: 1.0, Viral Pneumonia: 0.7, Lung_Opacity: 0.5, Normal: 0.1}
```

These severity values are assumptions, not clinical facts. Member 3 should replace them using the cost-sensitive threshold analysis and radiology workflow research. Only the field name `urgency_score` and its range (0 to 1) are fixed for now.

---

## 6. Data split (for reproducibility)

- Dataset: COVID-19 Radiography Database, 21,165 images; 54 exact duplicates removed (MD5), leaving 21,111.
- Split: stratified 70/15/15, seed 42 (Train 14,777 / Val 3,167 / Test 3,167).
- Files: `data/splits/train.csv`, `val.csv`, `test.csv` (local; regenerate with `model/split_data.py`).
- Input: 224x224, RGB, ImageNet normalization.
- Test set is used once for final reporting. No model selection is done on it.

---

## 7. Known limitations (to be stated in the report)

- Trained on one public dataset from mixed sources; no patient ID, so near-duplicate images of the same patient may exist across splits and results may be optimistic.
- Some Grad-CAM examples showed attention outside the lung fields (image markers, borders). Lung-mask experiment planned.
- Generalization to the pediatric Chest X-Ray (Pneumonia) dataset not yet tested; a drop is expected.
- The system supports radiologists and does not replace them. It is not clinically validated.

---

## 8. Open questions

1. Final model for the app: EfficientNet-B0 (current, lightest) or DenseNet121 (highest test score)?
2. Final urgency rule (Member 3).
3. Image upload format and size limits in the app (Member 2).