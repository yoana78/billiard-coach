"""BILLIARD COACH 서버 — 촬영 프레임 분석 + 탑뷰 변환 API.

실행:
  .venv\\Scripts\\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
"""
import base64
import sys
import time
from pathlib import Path

# Windows 콘솔(cp949)에서 한글/특수문자 출력이 서버를 죽이지 않도록
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from shots.guide import compute_guides, guide_to_dict
from detection.table_detector import cloth_mask
from detection.cushion_lines import detect_cushion_lines
from detection.partial import solve_full_from_lines
from detection.diamond_detector import detect_diamonds
from detection.homography import homography_from_corners, warp_topview
from detection.pipeline import image_to_topview, draw_debug, _downscale

app = FastAPI(title="Billiard Coach API")

# 촬영 허용 최소 조건 (기획서 3.2): 장쿠션 3점 이상 + 단쿠션 2점 이상
MIN_LONG_DIAMONDS = 3
MIN_SHORT_DIAMONDS = 2


def _decode(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """상태 확인. UptimeRobot 등 모니터링 도구는 HEAD로 요청하므로 둘 다 허용."""
    return {"status": "ok"}


class BallIn(BaseModel):
    color: str
    x_mm: float
    y_mm: float


class GuidesRequest(BaseModel):
    balls: list[BallIn]
    cue: str = "white"
    game: str = "four"      # 'four' | 'three'
    table: str = "medium"   # 'medium'(중대) | 'large'(대대)


@app.post("/guides")
def guides(req: GuidesRequest):
    """검출된 공 배치에서 샷 가이드 목록 계산 (4구/3구, 중대/대대)."""
    try:
        result = compute_guides([b.model_dump() for b in req.balls], req.cue,
                                game=req.game, table=req.table)
    except ValueError as e:
        return {"ok": False, "reason": str(e)}
    return {"ok": True, "guides": [guide_to_dict(g) for g in result]}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """프리뷰 프레임 1장을 빠르게 분석해 촬영 가능 여부와 안내 문구 반환."""
    bgr = _decode(await file.read())
    if bgr is None:
        return {"ok": False, "ready": False, "message": "이미지 해석 실패"}
    bgr = _downscale(bgr)

    mask, _raw, color = cloth_mask(bgr)
    img_area = bgr.shape[0] * bgr.shape[1]
    if cv2.countNonZero(mask) < img_area * 0.05:
        return {
            "ok": True, "ready": False, "table_found": False,
            "diamond_long": 0, "diamond_short": 0,
            "message": "당구대가 잘 안 보여요",
        }

    lines = detect_cushion_lines(mask)
    n_lines = len(lines)

    n_long = n_short = 0
    ready = False
    if n_lines >= 4:
        for fr in solve_full_from_lines(lines):
            H = homography_from_corners(fr.corners)
            if H is None:
                continue
            top = warp_topview(bgr, H)
            hits = detect_diamonds(top)
            long_c = sum(1 for h in hits if h.side in ("top", "bottom"))
            short_c = sum(1 for h in hits if h.side in ("left", "right"))
            if long_c + short_c > n_long + n_short:
                n_long, n_short = long_c, short_c
            if n_long >= MIN_LONG_DIAMONDS and n_short >= MIN_SHORT_DIAMONDS:
                ready = True
                break
        message = "좋아요! 이대로 촬영하세요" if ready else "다이아몬드가 잘 안 보여요 — 조금 더 위에서"
    elif n_lines == 3:
        # 부분 촬영 모드: 정밀 역산은 촬영 후 수행 (실시간은 라인 수로만 판단)
        ready = True
        message = "일부 촬영 모드 — 이대로 촬영 가능해요"
    elif n_lines == 2:
        # 코너 촬영 모드: 직교 쿠션 2개면 격자 역산 시도 가능
        ready = True
        message = "귀퉁이 촬영 모드 — 이대로 촬영 가능해요"
    else:
        message = "쿠션이 더 보이게 조금 더 뒤로 가주세요"

    return {
        "ok": True, "ready": ready, "table_found": True,
        "cloth_color": color,
        "visible_lines": n_lines,
        "diamond_long": n_long, "diamond_short": n_short,
        "message": message,
    }


# 실사진 디버깅용 업로드 저장 — 개발 PC에서만 켠다 (개인정보/디스크 보호).
# 켜려면 환경변수 SAVE_UPLOADS=1
import os
SAVE_UPLOADS = os.environ.get("SAVE_UPLOADS") == "1"
UPLOAD_DIR = Path(__file__).resolve().parent / "testdata" / "uploads"
if SAVE_UPLOADS:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/topview")
async def topview(file: UploadFile = File(...), debug: bool = False,
                  table: str = "medium"):
    """고해상도 촬영본 → 탑뷰 변환. PNG를 base64로 반환.

    table='large'(대대)면 공 좌표를 대대 mm 스케일로 환산해 반환.
    (중대/대대는 비율·다이아몬드 분할이 같아 인식 자체는 공용이고,
    검출 좌표는 중대 기준 정규화 값이므로 배율만 곱하면 실측이 된다.)
    """
    bgr = _decode(await file.read())
    if bgr is None:
        return {"ok": False, "reason": "이미지 해석 실패"}

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if SAVE_UPLOADS:
        cv2.imwrite(str(UPLOAD_DIR / f"{stamp}_in.jpg"), bgr)

    result = image_to_topview(bgr)
    if result.ok:
        if SAVE_UPLOADS:
            dbg = draw_debug(result)
            cv2.imwrite(str(UPLOAD_DIR / f"{stamp}_out.png"), dbg)
        print(f"[topview] {stamp}: diamonds={len(result.diamonds)}/28 "
              f"err={result.diamond_err_mm:.1f}mm balls={len(result.balls)} "
              f"cloth={result.cloth_color} partial={result.partial} "
              f"lines={result.visible_lines}", flush=True)
    else:
        print(f"[topview] {stamp}: FAIL {result.reason}", flush=True)
    if not result.ok:
        return {"ok": False, "reason": result.reason}

    img = draw_debug(result) if debug else result.topview
    _, png = cv2.imencode(".png", img)
    # 대대: 검출 좌표(중대 기준 정규화)를 실측 스케일로 환산
    from shots.guide import TABLES
    scale = TABLES.get(table, TABLES["medium"]).w / TABLES["medium"].w
    return {
        "ok": True,
        "diamond_count": len(result.diamonds),
        "diamond_err_mm": round(result.diamond_err_mm, 1),
        "refined": result.refined,
        "cloth_color": result.cloth_color,
        "partial": result.partial,
        "table": table,
        "balls": [
            {
                "color": b.color,
                "x_mm": round(b.pos_mm[0] * scale, 1),
                "y_mm": round(b.pos_mm[1] * scale, 1),
                "score": round(b.score, 2),
            }
            for b in result.balls
        ],
        "image_base64": base64.b64encode(png.tobytes()).decode(),
    }
