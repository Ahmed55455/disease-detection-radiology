"""REST API (FastAPI) on top of the same backend as the Streamlit app.

Run from the repository root:   uvicorn app.api:app --reload
Interactive docs:               http://127.0.0.1:8000/docs

It reuses inference.py (upload checks + predict()) and db.py (SQLite), so the
Streamlit app and the API always share the same rules and the same database.
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so "import db" works

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

import db
from inference import InvalidImageError, analyze, load_image

app = FastAPI(
    title="Chest X-ray Worklist API",
    description="Assists radiologists and does not replace them. "
                "Limited by its training data; not a medical diagnosis.",
    version="0.1",
)
db.init_db()
_model_lock = threading.Lock()   # the model / Grad-CAM is not safe for parallel calls


class DoneRequest(BaseModel):
    radiologist_label: str | None = None


def _case_or_404(case_id):
    case = db.get_case(case_id)
    if case is None:
        raise HTTPException(404, f"Case {case_id} not found")
    return case


def _public(case):
    """Hide local file paths; give API links instead."""
    case = dict(case)
    case.pop("image_path", None)
    case.pop("heatmap_path", None)
    case["image_url"] = f"/cases/{case['case_id']}/image"
    case["heatmap_url"] = f"/cases/{case['case_id']}/heatmap"
    return case


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...),
                        save: bool = Query(True, description="Add the case to the worklist")):
    """Upload a chest X-ray (PNG/JPG, max 10 MB). Returns the prediction."""
    data = await file.read()
    try:
        image = load_image(file.filename or "", data)
    except InvalidImageError as e:
        raise HTTPException(400, str(e))
    try:
        with _model_lock:
            res = analyze(image)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

    out = {
        "class": res["class"],
        "confidence": res["confidence"],
        "probabilities": res["probabilities"],
        "urgency_score": db.urgency_score(res["probabilities"]),
        "model_name": res["model_name"],
        "warnings": res["warnings"],
        "case_id": None,
    }
    if save:
        out["case_id"] = db.save_case(image, res)
    return out


@app.get("/cases")
def list_cases(include_done: bool = False,
               order: str = Query("ai", pattern="^(ai|fifo)$",
                                  description="ai = urgency first, fifo = arrival order")):
    """Worklist in AI priority order (default) or FIFO order."""
    return [_public(c) for c in db.get_worklist(include_done=include_done, order=order)]


@app.get("/compare")
def compare(include_done: bool = False):
    """Position of each case in FIFO vs AI order (moved_up > 0 = read earlier with AI)."""
    return db.compare_orders(include_done=include_done)


@app.get("/cases/{case_id}")
def get_case(case_id: int):
    return _public(_case_or_404(case_id))


@app.get("/cases/{case_id}/image")
def get_image(case_id: int):
    return FileResponse(_case_or_404(case_id)["image_path"], media_type="image/png")


@app.get("/cases/{case_id}/heatmap")
def get_heatmap(case_id: int):
    return FileResponse(_case_or_404(case_id)["heatmap_path"], media_type="image/png")


def _move(case_id, status, label=None):
    _case_or_404(case_id)
    try:
        db.set_status(case_id, status, label)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return _public(db.get_case(case_id))


@app.post("/cases/{case_id}/start_review")
def start_review(case_id: int):
    return _move(case_id, "in_review")


@app.post("/cases/{case_id}/release")
def release(case_id: int):
    """Put a case in review back to pending."""
    return _move(case_id, "pending")


@app.post("/cases/{case_id}/done")
def mark_done(case_id: int, body: DoneRequest):
    return _move(case_id, "done", body.radiologist_label)


@app.get("/export.csv", response_class=PlainTextResponse)
def export_csv():
    """CSV for the simulation (Member 3)."""
    return PlainTextResponse(db.export_csv(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=worklist.csv"})
