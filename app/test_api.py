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

    cases = client.get("/cases").json()
    assert cases[0]["case_id"] == case_id and "image_path" not in cases[0]
    assert client.get(f"/cases/{case_id}/image").status_code == 200
    assert client.get(f"/cases/{case_id}/heatmap").status_code == 200
    assert client.get("/cases/9999").status_code == 404
    assert client.get("/cases", params={"order": "fifo"}).status_code == 200
    assert client.get("/cases", params={"order": "bad"}).status_code == 422
    assert client.get("/compare").json()[0]["ai_position"] == 1
    print("OK  /cases, order, /compare, images, 404")

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
