"""FastAPI 엔드포인트 테스트 — 합성 이미지로 /analyze, /topview 검증."""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import app
from tools.make_synthetic import make_photo

client = TestClient(app)


def _png_bytes(bgr) -> bytes:
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_good_frame():
    photo, _, _ = make_photo(seed=1, tilt=0.18)
    r = client.post("/analyze", files={"file": ("f.png", _png_bytes(photo), "image/png")})
    data = r.json()
    assert data["ok"] and data["table_found"]
    assert data["ready"], data
    assert data["diamond_long"] >= 3 and data["diamond_short"] >= 2


def test_analyze_no_table():
    junk = np.random.default_rng(3).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    r = client.post("/analyze", files={"file": ("f.png", _png_bytes(junk), "image/png")})
    data = r.json()
    assert data["ok"] and not data["ready"] and not data["table_found"]


def test_topview():
    photo, _, _ = make_photo(seed=2, tilt=0.22)
    r = client.post("/topview", files={"file": ("f.png", _png_bytes(photo), "image/png")})
    data = r.json()
    assert data["ok"]
    assert data["diamond_count"] >= 16
    png = base64.b64decode(data["image_base64"])
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[1] > img.shape[0]  # 가로가 긴 탑뷰
    # 공 4개: 색 구성 흰1/노1/빨2, 좌표는 경기면 범위 안
    balls = data["balls"]
    assert sorted(b["color"] for b in balls) == ["red", "red", "white", "yellow"]
    for b in balls:
        assert 0 <= b["x_mm"] <= 2540 and 0 <= b["y_mm"] <= 1270
