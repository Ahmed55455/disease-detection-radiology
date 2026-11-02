"""Tests for app/api.py. Uses a fake model and a temporary database.

Run from the repository root:  python app/test_api.py
"""
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db

tmp = Path(tempfile.mkdtemp())
db.configure(tmp / "test.db", tmp / "storage")   # before importing api: real cases.db untouched

import api
from fastapi.testclient import TestClient

PROBS = {"COVID": 0.9, "Lung_Opacity": 0.05, "Normal": 0.03, "Viral Pneumonia": 0.02}
api.analyze = lambda image: {      # fake model, so the test is fast and needs no checkpoint
    "class": "COVID", "confidence": 0.9, "probabilities": PROBS,
    "heatmap": np.zeros((224, 224, 3), dtype=np.uint8),
    "model_name": "fake", "warnings": []}

client = TestClient(api.app)


def png():
    buf = io.BytesIO()
    Image.new("L", (256, 256), 100).save(buf, format="PNG")
    return buf.getvalue()


def main():
    assert client.get("/health").json() == {"status": "ok"}
    print("OK  /health")

    r = client.post("/analyze", files={"file": ("x.png", png(), "image/png")})
    assert r.status_code == 200, r.text
    case_id = r.json()["case_id"]
    assert r.json()["class"] == "COVID" and case_id
    print("OK  /analyze saved case", case_id)

    r = client.post("/analyze", files={"file": ("fake.png", b"hello", "image/png")})
    assert r.status_code == 400
    print("OK  /analyze rejects a fake image:", r.json()["detail"])

    r = client.post("/analyze", files={"file": ("x.gif", png(), "image/gif")})
    assert r.status_code == 400
    print("OK  /analyze rejects a wrong file type")

    r = client.get("/mobile")
    assert r.status_code == 200 and 'capture="environment"' in r.text
    r = client.post("/analyze", params={"source": "phone"},
                    files={"file": ("phone_photo.jpg", png(), "image/jpeg")})
    phone_id = r.json()["case_id"]
    assert db.get_case(phone_id)["source"] == "phone"
    assert client.post("/analyze", params={"source": "bad"},
                       files={"file": ("x.png", png(), "image/png")}).status_code == 422
    with __import__("contextlib").closing(db._conn()) as con, con:
        con.execute("DELETE FROM cases WHERE case_id=?", (phone_id,))
    print("OK  /mobile page and phone upload (source = phone)")

    # A result with warnings is not added when skip_if_warnings=true (maybe not an X-ray)
    real = api.analyze
    api.analyze = lambda image: {**real(image), "warnings": ["Image is colored."]}
    r = client.post("/analyze", params={"skip_if_warnings": True, "source": "phone"},
                    files={"file": ("lamp.jpg", png(), "image/jpeg")}).json()
    assert r["case_id"] is None and r["not_saved_reason"] == "warnings" and r["heatmap_png"]
    r = client.post("/analyze", params={"skip_if_warnings": False, "source": "phone"},
                    files={"file": ("lamp.jpg", png(), "image/jpeg")}).json()
    assert r["case_id"]                                   # "add anyway"
    assert client.delete(f"/cases/{r['case_id']}").json() == {"deleted": r["case_id"]}
    assert client.get(f"/cases/{r['case_id']}").status_code == 404
    api.analyze = real
    print("OK  suspicious image not added automatically; add anyway; delete case")

    cases = client.get("/cases").json()
    assert cases[0]["warnings"] == [] and cases[0]["model_name"] == "fake"
    assert cases[0]["case_id"] == case_id and "image_path" not in cases[0]
    assert client.get(f"/cases/{case_id}/image").status_code == 200
    assert client.get(f"/cases/{case_id}/heatmap").status_code == 200
    assert client.get("/cases/9999").status_code == 404
    assert client.get("/cases", params={"order": "fifo"}).status_code == 200
    assert client.get("/cases", params={"order": "bad"}).status_code == 422
    assert client.get("/compare").json()[0]["ai_position"] == 1
    print("OK  /cases, order, /compare, images, 404")

    r = client.get(f"/cases/{case_id}/explain")
    assert r.status_code == 200 and "COVID" in r.json()["explanation"]
    r = client.post(f"/cases/{case_id}/ask", json={"question": "How sure is the model?"})
    assert r.status_code == 200 and r.json()["source"] == "template" and r.json()["answer"]
    r = client.post(f"/cases/{case_id}/ask", json={"question": "Should I take antibiotics?"})
    assert r.json()["source"] == "refused"
    assert client.post(f"/cases/{case_id}/ask", json={"question": " "}).status_code == 400
    print("OK  /explain and /ask (template answer, diagnosis question refused)")

    r = client.get(f"/cases/{case_id}/report")
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
    r = client.get(f"/cases/{case_id}/report", params={"fmt": "png"})
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"
    print("OK  /report (PDF and PNG)")

    assert client.post(f"/cases/{case_id}/done", json={}).status_code == 409   # must review first
    assert client.post(f"/cases/{case_id}/start_review").json()["status"] == "in_review"
    r = client.post(f"/cases/{case_id}/done", json={"radiologist_label": "COVID"})
    assert r.json()["status"] == "done" and r.json()["read_at"]
    assert client.get("/cases").json() == []
    assert len(client.get("/cases", params={"include_done": True}).json()) == 1
    print("OK  status flow through the API")

    csv = client.get("/export.csv").text.splitlines()
    assert csv[0] == ",".join(db.EXPORT_COLUMNS) and len(csv) == 2
    print("OK  /export.csv")

    print("\nAll API tests passed.")


if __name__ == "__main__":
    main()
