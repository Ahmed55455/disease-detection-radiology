import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from PIL import Image

DB_PATH = Path("app/cases.db")
STORAGE = Path("app/storage")

# Placeholder weights from docs/interfaces.md (Member 3 will replace them)
SEVERITY = {"COVID": 1.0, "Viral Pneumonia": 0.7, "Lung_Opacity": 0.5, "Normal": 0.1}


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
                status TEXT DEFAULT 'pending', radiologist_label TEXT
            )""")


def urgency_score(probs):
    return sum(p * SEVERITY.get(c, 0.0) for c, p in probs.items())


def save_case(image, res):
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    img_path = STORAGE / f"{stamp}_image.png"
    hm_path = STORAGE / f"{stamp}_heatmap.png"
    image.save(img_path)
    Image.fromarray(res["heatmap"]).save(hm_path)
    p = res["probabilities"]
    with closing(_conn()) as c, c:
        cur = c.execute(
            """INSERT INTO cases (image_path, heatmap_path, uploaded_at, predicted_class,
               p_covid, p_lung_opacity, p_normal, p_viral_pneumonia, confidence, urgency_score)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (str(img_path), str(hm_path), now.isoformat(timespec="seconds"), res["class"],
             p["COVID"], p["Lung_Opacity"], p["Normal"], p["Viral Pneumonia"],
             p[res["class"]], urgency_score(p)),
        )
        return cur.lastrowid


def get_worklist():
    with closing(_conn()) as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM cases WHERE status != 'done' "
            "ORDER BY urgency_score DESC, uploaded_at ASC")]


def mark_done(case_id):
    with closing(_conn()) as c, c:
        c.execute("UPDATE cases SET status='done', read_at=? WHERE case_id=?",
                  (datetime.now().isoformat(timespec="seconds"), case_id))