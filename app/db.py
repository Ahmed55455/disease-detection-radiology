"""Database layer (SQLite) for the radiologist worklist.

No Streamlit and no PyTorch here, so this file can be reused later by a FastAPI API
and tested on its own (see app/test_db.py).
"""
import csv
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "cases.db"
STORAGE = APP_DIR / "storage"

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
PROB_COLUMNS = {           # class name -> database column
    "COVID": "p_covid",
    "Lung_Opacity": "p_lung_opacity",
    "Normal": "p_normal",
    "Viral Pneumonia": "p_viral_pneumonia",
}

# ---- Urgency rule: the ONLY place to change it --------------------------------
# Placeholder weights from docs/interfaces.md. Member 3 (Rawan) will send the final rule.
# After changing them, call recompute_urgency() so stored cases use the new rule.
SEVERITY = {"COVID": 1.0, "Viral Pneumonia": 0.7, "Lung_Opacity": 0.5, "Normal": 0.1}

STATUSES = ("pending", "in_review", "done")
ALLOWED_MOVES = {               # status flow: pending -> in_review -> done
    "pending": {"in_review"},
    "in_review": {"done", "pending"},   # "pending" = radiologist releases the case
    "done": set(),
}

EXPORT_COLUMNS = ["case_id", "uploaded_at", "read_at", "predicted_class",
                  "urgency_score", "status"]


def configure(db_path=None, storage=None):
    """Change where data is stored (used by the tests)."""
    global DB_PATH, STORAGE
    if db_path is not None:
        DB_PATH = Path(db_path)
    if storage is not None:
        STORAGE = Path(storage)


def now_iso():
    """Current local time in ISO 8601 with timezone, e.g. 2026-10-13T21:30:05+03:00."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    STORAGE.mkdir(parents=True, exist_ok=True)
    with closing(_conn()) as c, c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT, heatmap_path TEXT,
                uploaded_at TEXT, read_at TEXT,
                predicted_class TEXT,
                p_covid REAL, p_lung_opacity REAL, p_normal REAL, p_viral_pneumonia REAL,
                confidence REAL, urgency_score REAL,
                status TEXT DEFAULT 'pending', radiologist_label TEXT,
                model_name TEXT, review_started_at TEXT
            )""")
        # Older cases.db files (week 1-2) do not have the new columns: add them.
        existing = {r["name"] for r in c.execute("PRAGMA table_info(cases)")}
        for col in ("model_name", "review_started_at"):
            if col not in existing:
                c.execute(f"ALTER TABLE cases ADD COLUMN {col} TEXT")


def urgency_score(probs):
    """Weighted sum of class probabilities (range 0 to 1)."""
    return round(float(sum(p * SEVERITY.get(c, 0.0) for c, p in probs.items())), 4)


def save_case(image, res):
    """image: PIL.Image. res: output of inference.analyze(). Returns the new case_id."""
    uploaded_at = now_iso()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = STORAGE / f"{stamp}_image.png"
    hm_path = STORAGE / f"{stamp}_heatmap.png"
    image.save(img_path)
    Image.fromarray(np.asarray(res["heatmap"], dtype=np.uint8)).save(hm_path)

    p = res["probabilities"]
    with closing(_conn()) as c, c:
        cur = c.execute(
            """INSERT INTO cases (image_path, heatmap_path, uploaded_at, predicted_class,
               p_covid, p_lung_opacity, p_normal, p_viral_pneumonia,
               confidence, urgency_score, status, model_name)
               VALUES (?,?,?,?,?,?,?,?,?,?,'pending',?)""",
            (str(img_path), str(hm_path), uploaded_at, res["class"],
             p["COVID"], p["Lung_Opacity"], p["Normal"], p["Viral Pneumonia"],
             res.get("confidence", p[res["class"]]), urgency_score(p),
             res.get("model_name")),
        )
        return cur.lastrowid


def get_case(case_id):
    with closing(_conn()) as c:
        r = c.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        return dict(r) if r else None


def get_worklist(include_done=False):
    """Cases ordered by urgency (highest first), ties by upload time (oldest first)."""
    where = "" if include_done else "WHERE status != 'done'"
    with closing(_conn()) as c:
        return [dict(r) for r in c.execute(
            f"SELECT * FROM cases {where} ORDER BY urgency_score DESC, uploaded_at ASC")]


def set_status(case_id, new_status, radiologist_label=None):
    """Move a case along pending -> in_review -> done. Raises ValueError if not allowed."""
    if new_status not in STATUSES:
        raise ValueError(f"Unknown status: {new_status}")
    if radiologist_label is not None and radiologist_label not in CLASSES:
        raise ValueError(f"Unknown label: {radiologist_label}")
    case = get_case(case_id)
    if case is None:
        raise ValueError(f"Case {case_id} not found")
    old = case["status"]
    if new_status not in ALLOWED_MOVES[old]:
        raise ValueError(f"Cannot move case {case_id} from '{old}' to '{new_status}'")

    with closing(_conn()) as c, c:
        if new_status == "in_review":
            c.execute("UPDATE cases SET status='in_review', review_started_at=? "
                      "WHERE case_id=?", (now_iso(), case_id))
        elif new_status == "pending":
            c.execute("UPDATE cases SET status='pending', review_started_at=NULL "
                      "WHERE case_id=?", (case_id,))
        else:  # done
            c.execute("UPDATE cases SET status='done', read_at=?, radiologist_label=? "
                      "WHERE case_id=?", (now_iso(), radiologist_label, case_id))


def start_review(case_id):
    set_status(case_id, "in_review")


def mark_done(case_id, radiologist_label=None):
    set_status(case_id, "done", radiologist_label)


def recompute_urgency():
    """Recalculate urgency_score for every stored case with the current SEVERITY."""
    with closing(_conn()) as c, c:
        rows = c.execute("SELECT * FROM cases").fetchall()
        for r in rows:
            probs = {cls: r[col] for cls, col in PROB_COLUMNS.items()}
            c.execute("UPDATE cases SET urgency_score=? WHERE case_id=?",
                      (urgency_score(probs), r["case_id"]))
        return len(rows)


def export_csv(path=None):
    """Write ALL cases (including done) to CSV for Member 3. Returns the CSV text."""
    with closing(_conn()) as c:
        rows = c.execute(f"SELECT {', '.join(EXPORT_COLUMNS)} FROM cases "
                         "ORDER BY uploaded_at ASC, case_id ASC").fetchall()
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(EXPORT_COLUMNS)
    for r in rows:
        w.writerow([r[col] if r[col] is not None else "" for col in EXPORT_COLUMNS])
    text = buf.getvalue()
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
