"""Small tests for app/db.py and the upload checks in app/inference.py.

Run from the repository root:  python app/test_db.py
Uses a temporary folder, so your real app/cases.db is NOT touched. No PyTorch needed.
"""
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db
from inference import InvalidImageError, load_image


def fake_result(cls, probs):
    return {"class": cls, "confidence": probs[cls], "probabilities": probs,
            "heatmap": np.zeros((224, 224, 3), dtype=np.uint8), "model_name": "test"}


def png_bytes(size=(256, 256)):
    buf = io.BytesIO()
    Image.new("L", size, 128).save(buf, format="PNG")
    return buf.getvalue()


def expect_error(func, *args):
    try:
        func(*args)
    except (ValueError, InvalidImageError):
        return
    raise AssertionError(f"{func.__name__}{args} should have raised an error")


def main():
    tmp = Path(tempfile.mkdtemp())
    db.configure(tmp / "test.db", tmp / "storage")
    db.init_db()
    img = Image.new("RGB", (256, 256))

    normal = {"COVID": 0.05, "Lung_Opacity": 0.05, "Normal": 0.85, "Viral Pneumonia": 0.05}
    covid = {"COVID": 0.90, "Lung_Opacity": 0.05, "Normal": 0.03, "Viral Pneumonia": 0.02}

    # 1) Urgency rule
    assert abs(db.urgency_score(covid) - (0.90 + 0.05*0.5 + 0.03*0.1 + 0.02*0.7)) < 1e-4
    print("OK  urgency_score")

    # 2) Order: COVID case first even if uploaded later
    a = db.save_case(img, fake_result("Normal", normal))
    b = db.save_case(img, fake_result("COVID", covid))
    assert [r["case_id"] for r in db.get_worklist()] == [b, a]
    print("OK  worklist order (urgency desc)")

    # 3) Tie: same urgency -> older first
    c = db.save_case(img, fake_result("COVID", covid))
    ids = [r["case_id"] for r in db.get_worklist()]
    assert ids.index(b) < ids.index(c)
    print("OK  tie broken by uploaded_at")

    # 3b) FIFO order and FIFO vs AI comparison
    fifo = [r["case_id"] for r in db.get_worklist(order="fifo")]
    assert fifo == [a, b, c]
    cmp = {r["case_id"]: r for r in db.compare_orders()}
    assert cmp[b]["fifo_position"] == 2 and cmp[b]["ai_position"] == 1 and cmp[b]["moved_up"] == 1
    assert cmp[a]["moved_up"] == -2
    expect_error(db.get_worklist, False, "random")
    print("OK  FIFO order and compare_orders")

    # 4) ISO 8601 time with timezone
    up = db.get_case(a)["uploaded_at"]
    assert "T" in up and ("+" in up or up.endswith("Z") or "-" in up[19:])
    print("OK  uploaded_at is ISO 8601:", up)

    # 5) Status flow pending -> in_review -> done
    assert db.get_case(b)["status"] == "pending"
    expect_error(db.mark_done, b)               # cannot skip in_review
    db.start_review(b)
    assert db.get_case(b)["status"] == "in_review"
    db.mark_done(b, "COVID")
    case = db.get_case(b)
    assert case["status"] == "done" and case["read_at"] and case["radiologist_label"] == "COVID"
    expect_error(db.start_review, b)            # done is final
    expect_error(db.set_status, a, "unknown")
    assert b not in [r["case_id"] for r in db.get_worklist()]
    print("OK  status flow and radiologist_label")

    # 6) CSV export for Member 3
    lines = db.export_csv(tmp / "worklist.csv").strip().splitlines()
    assert lines[0] == ",".join(db.EXPORT_COLUMNS)
    assert len(lines) == 1 + 3                  # header + all 3 cases (done included)
    print("OK  export_csv:", lines[0])

    # 7) Changing the urgency rule updates stored cases
    old = dict(db.SEVERITY)
    db.SEVERITY["Normal"] = 1.0
    db.recompute_urgency()
    assert abs(db.get_case(a)["urgency_score"] - db.urgency_score(normal)) < 1e-4
    db.SEVERITY.update(old)
    print("OK  recompute_urgency")

    # 8) Upload checks
    assert load_image("x.png", png_bytes()).mode == "RGB"
    expect_error(load_image, "x.gif", png_bytes())                      # wrong type
    expect_error(load_image, "x.png", b"")                               # empty
    expect_error(load_image, "x.png", b"not an image at all")            # unreadable
    expect_error(load_image, "x.png", png_bytes((20, 20)))               # too small
    expect_error(load_image, "x.png", b"0" * (10 * 1024 * 1024 + 1))     # > 10 MB
    print("OK  upload validation")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
