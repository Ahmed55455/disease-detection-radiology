# app/ — Streamlit app, backend and worklist database (Member 2)

The system assists radiologists and does not replace them. It is limited by its training data.

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit user interface only (pages: Upload & Analyze, Worklist, Export) |
| `inference.py` | Checks the uploaded file, then calls `predict()` from `model/predict.py` |
| `db.py` | SQLite database: save cases, worklist order, status flow, urgency rule, CSV export |
| `test_db.py` | Tests for `db.py` and the upload checks (no PyTorch needed) |
| `api.py` | REST API (FastAPI) using the same `inference.py` and `db.py` |
| `test_api.py` | Tests for the API with a fake model (no checkpoint needed) |

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

## REST API (FastAPI)

```powershell
pip install fastapi uvicorn python-multipart httpx
uvicorn app.api:app --reload        # from the repository root
python app/test_api.py              # API tests
```

Open http://127.0.0.1:8000/docs to try every endpoint in the browser.

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Check that the API is running |
| POST | `/analyze?save=true` | Upload an image (form field `file`), get the prediction, add it to the worklist |
| GET | `/cases?include_done=false` | Worklist, highest urgency first |
| GET | `/cases/{id}` | One case |
| GET | `/cases/{id}/image`, `/cases/{id}/heatmap` | Stored images (PNG) |
| POST | `/cases/{id}/start_review` | `pending -> in_review` |
| POST | `/cases/{id}/release` | `in_review -> pending` |
| POST | `/cases/{id}/done` | `in_review -> done`, body `{"radiologist_label": "COVID"}` |
| GET | `/export.csv` | CSV for the simulation |

Errors: `400` bad upload, `404` unknown case, `409` status move not allowed, `500` model error.
The API and the Streamlit app share the same `cases.db`.

## Privacy

`app/cases.db` and `app/storage/` (patient images) are in `.gitignore`. Never commit them.
