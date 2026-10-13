# app/ — Streamlit app, backend and worklist database (Member 2)

The system assists radiologists and does not replace them. It is limited by its training data.

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit user interface only (pages: Upload & Analyze, Worklist, Export) |
| `inference.py` | Checks the uploaded file, then calls `predict()` from `model/predict.py` |
| `db.py` | SQLite database: save cases, worklist order, status flow, urgency rule, CSV export |
| `test_db.py` | Tests for `db.py` and the upload checks (no PyTorch needed) |

`app.py` never talks to SQLite or PyTorch directly, so a FastAPI API can reuse
`inference.py` and `db.py` later without rewriting them.

## Setup and run (Windows, from the repository root)

```powershell
.venv\Scripts\activate
# model checkpoint from Google Drive -> model\checkpoints\efficientnet_b0_8ep_best.pt
streamlit run app/app.py      # opens http://localhost:8501
python app/test_db.py         # runs the tests
```

## Upload rules

- PNG or JPG/JPEG only, maximum 10 MB, minimum 64x64 pixels.
- Empty, damaged or fake image files show a clear error message instead of crashing.
- `predict()` warnings (for example a colored image, which may not be an X-ray) are shown to the user.

## Worklist

- Order: `urgency_score` descending, ties by `uploaded_at` ascending (oldest first).
- Status flow: `pending -> in_review -> done` (a case in review can go back to `pending`).
- `done` requires the radiologist label (one of the 4 classes) and sets `read_at`.
- Times are ISO 8601 with timezone, for example `2026-10-13T21:30:05+03:00`.

## Urgency rule

`urgency_score = sum(probability[class] * SEVERITY[class])`, range 0 to 1.
The weights are placeholders (COVID 1.0, Viral Pneumonia 0.7, Lung_Opacity 0.5, Normal 0.1)
and live only in `SEVERITY` in `db.py`. When Member 3 sends the final rule: edit `SEVERITY`,
then run `python -c "import sys; sys.path.insert(0,'app'); import db; print(db.recompute_urgency())"`
to update cases already stored.

## CSV export for the simulation (Member 3)

Page **Export** -> **Download worklist.csv**, or in Python `db.export_csv("worklist.csv")`.
Columns: `case_id, uploaded_at, read_at, predicted_class, urgency_score, status`.
All cases are included (also `done`, which have `read_at`).

## Database columns

`case_id, image_path, heatmap_path, uploaded_at, read_at, predicted_class, p_covid,
p_lung_opacity, p_normal, p_viral_pneumonia, confidence, urgency_score, status,
radiologist_label, model_name, review_started_at`.
`model_name` and `review_started_at` were added in week 3; older `cases.db` files are
upgraded automatically.

## Privacy

`app/cases.db` and `app/storage/` (patient images) are in `.gitignore`. Never commit them.
