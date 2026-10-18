# Interfaces Agreement (docs/interfaces.md)

Project: Disease Detection from Radiology Images Using Deep Learning
Status: DRAFT v0.2 (2026-10-18), written by Member 1 (AI Model). Members 2 and 3 should review and comment.

The system assists radiologists. It does not replace them, and it is limited by its training data.

Changes from v0.1 (all additive, nothing was removed or renamed):
- `predict()` now also returns `confidence`, `model_name` and `warnings` (section 2).
- Time format fixed to ISO 8601 (section 3).
- Class display names are separated from the internal strings (section 1).
- Test results updated with the seed-variability finding (section 4).
- Limitations updated with what was measured (section 7).

---

## 1. Class order (fixed, do not change)

| Index | Class name (exact string, used in code, files and the database) | Display name (UI only) |
|---|---|---|
| 0 | COVID | COVID-19 |
| 1 | Lung_Opacity | Lung Opacity |
| 2 | Normal | Normal |
| 3 | Viral Pneumonia | Viral Pneumonia |

Everyone must use these exact strings and this exact order in code, CSV files and the database. Display names are only for what the user sees on screen.

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
  "confidence": 0.97,                    # float, probability of the predicted class
  "probabilities": {                     # dict[str, float], sums to 1.0
      "COVID": 0.97,
      "Lung_Opacity": 0.01,
      "Normal": 0.01,
      "Viral Pneumonia": 0.01,
  },
  "heatmap": <numpy.ndarray>,            # shape (224, 224, 3), dtype uint8, RGB,
                                         # Grad-CAM overlay for the predicted class
  "model_name": "efficientnet_b0",       # str, which model produced this result
  "warnings": [],                        # list[str], empty if nothing to flag
}
```

Notes:
- `confidence` equals `probabilities[result["class"]]`. It is a model score, not a calibrated clinical probability, and it is not a reliable sign that the input is a valid chest X-ray.
- `warnings` is a list of short messages. Currently it flags colored inputs (the model expects grayscale X-rays; every input is converted to grayscale before the model). It can be extended later. The app should show the warnings to the user next to the result. A warning does not mean the prediction is wrong, and it should not lower the case priority automatically.
- `model_name` should be stored with each case, so results stay traceable if the model file changes.
- The model is loaded once, on the first call (lazy loading). The first call is slower.
- Current model: EfficientNet-B0 (checkpoint `model/checkpoints/efficientnet_b0_8ep_best.pt`). This is provisional. It can be switched in `predict.py` (`MODEL_NAME`, `CKPT`) without changing the output format.
- Checkpoints (`*.pt`) are NOT in GitHub. They are shared through Google Drive (folder `radiology-checkpoints`).
- To show the heatmap in Streamlit: `st.image(result["heatmap"])`.
- Grad-CAM is approximate (low-resolution feature map, 7x7, upsampled). It is an aid to explanation, not an exact localization.

---

## 3. Worklist fields (for Member 2)

The reference implementation is `app/db.py` (Member 2). The table below is the agreed meaning of the fields; the exact column list is whatever `app/db.py` defines.

SQLite table `cases`:

| Field | Type | Description |
|---|---|---|
| case_id | INTEGER PK | Auto id |
| image_path | TEXT | Stored image file |
| uploaded_at | TEXT | Arrival time, ISO 8601 (for example `2026-10-18T19:14:00+03:00`), needed by the simulation |
| predicted_class | TEXT | One of the 4 class names |
| p_covid, p_lung_opacity, p_normal, p_viral_pneumonia | REAL | Class probabilities |
| confidence | REAL | Probability of the predicted class |
| model_name | TEXT | Model that produced the prediction (suggested) |
| warnings | TEXT | Warnings from `predict()`, for example joined by `;` (suggested) |
| urgency_score | REAL | 0 to 1, see section 5 |
| heatmap_path | TEXT | Saved Grad-CAM image |
| status | TEXT | `pending`, `in_review`, `done` |
| read_at | TEXT | When the radiologist finished reading it, ISO 8601 (needed for time-to-diagnosis) |
| radiologist_label | TEXT | Final label entered by the radiologist (optional) |

Worklist order: `urgency_score` descending, ties broken by `uploaded_at` ascending. A FIFO order (`uploaded_at` ascending) is also kept, for comparison with Member 3's simulation.

CSV export for Member 3: `case_id, uploaded_at, read_at, predicted_class, urgency_score, status`. If the time when the radiologist started a case (`review_started_at`) is useful for the simulation, it should be added to the export (Member 3 and Member 2 to decide).

---

## 4. Test predictions file (for Member 3, evaluation and simulation)

Location (local, not in GitHub): `data/predictions/<model>_test_probs.csv`
Models available: `resnet50`, `efficientnet_b0`, `densenet121` (seed 42 runs, the original training).
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

Test results, original runs (seed 42):

| Model | Accuracy | macro-F1 | macro ROC-AUC |
|---|---|---|---|
| ResNet50 | 0.959 | 0.969 | 0.9950 |
| EfficientNet-B0 | 0.955 | 0.964 | 0.9954 |
| DenseNet121 | 0.962 | 0.971 | 0.9950 |

Variability across 3 training runs per model (seeds 42, 1, 2), mean +- standard deviation (details in `docs/model_findings.md`, section 3.2, and `analysis/seed_runs.csv`):

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| EfficientNet-B0 | 0.958 +- 0.003 | 0.966 +- 0.002 |
| ResNet50 | 0.960 +- 0.001 | 0.970 +- 0.001 |
| DenseNet121 | 0.961 +- 0.001 | 0.970 +- 0.001 |

The differences between models are small, about the size of the variation between runs of one model. The statistical comparison is Member 3's task. It describes the seed-42 checkpoints only. Do not replace the checkpoint after a threshold or urgency rule has been tuned, because the tuning does not transfer exactly to another checkpoint.

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
- Files: `data/splits/train.csv`, `val.csv`, `test.csv` (local; regenerate with `model/prep/split_data.py`).
- Input: 224x224, RGB, ImageNet normalization.
- Test set is used once for final reporting. No model selection is done on it.

---

## 7. Known limitations (to be stated in the report)

Full details and numbers are in `docs/model_findings.md`.

- The system supports radiologists and does not replace them. It is not clinically validated.
- Trained on public data from limited sources; no patient ID in the main dataset, so images of the same patient may be in different splits. This was confirmed for part of two classes; in those groups no effect on test recall was found.
- The Normal and Viral Pneumonia classes partly come from a pediatric collection. On new pediatric images from that collection, sensitivity stays high (about 0.98 to 1.0) but specificity is low and varies between training runs (about 0.43 to 0.67). The cause is not explained.
- The model is sensitive to input changes. A color tint was found to flip a prediction; it is mitigated by grayscale conversion and a warning. There is no detection of "this is not an X-ray".
- Grad-CAM is coarse and was inspected on few images.
- Differences of about one point between the three architectures are within or near run-to-run variation.

---

## 8. Open questions

1. Final model for the app: EfficientNet-B0 (current, lightest) or another model, after Member 3's statistical comparison.
2. Final urgency rule and decision threshold (Member 3).
3. Image upload format and size limits in the app (Member 2).
4. Store `model_name` and `warnings` with each case, and export `review_started_at` (Member 2 and Member 3).