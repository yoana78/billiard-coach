"""공 4개(흰1/노1/빨2) 위치 검출.

전략: 왜곡이 없는 '원본 사진'에서 색 분리로 공 블롭을 찾고,
중심 좌표만 호모그래피로 탑뷰(mm)에 투영한다.
(탑뷰 워프에서 찾으면 가까운 공이 원근 때문에 수 배로 커져
크기 필터가 불안정하다 — 원본에서는 공이 항상 깨끗한 원/타원.)

거짓 양성 억제: 투영된 위치가 경기면 밖이면 제외하고,
투영된 크기(공 footprint 지름)가 물리적으로 말이 안 되면 제외.

알려진 한계(추후 보정): 공은 높이가 있어 비스듬한 촬영에서는
투영 중심이 실제 접점보다 카메라 반대쪽으로 밀린다(시차).
"""
from dataclasses import dataclass

import cv2
import numpy as np

from . import spec
from .homography import mm_to_px, px_to_mm, PX_PER_MM, project

# 공 색상 HSV 범위 (H: 0~179). 빨강은 색상환 양끝 두 구간.
_BALL_COLORS: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "white": [((0, 0, 170), (179, 70, 255))],
    "yellow": [((18, 80, 120), (40, 255, 255))],
    "red": [((0, 100, 90), (10, 255, 255)), ((165, 100, 90), (179, 255, 255))],
}
# 색별 기대 개수
_EXPECTED = {"white": 1, "yellow": 1, "red": 2}

_BALL_RADIUS_PX = spec.BALL_DIAMETER_MM / 2 * PX_PER_MM
_BALL_AREA_PX = np.pi * _BALL_RADIUS_PX**2


@dataclass
class Ball:
    color: str                  # 'white' | 'yellow' | 'red'
    pos_mm: tuple[float, float]  # 경기면 좌표 (mm)
    score: float                # 검출 신뢰도 (0~1)


def _blob_candidates(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """원본 이미지 마스크에서 공 후보 (cx, cy, 지름px, 원형도) 목록."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    img_area = mask.shape[0] * mask.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area < 60 or area > img_area * 0.05:  # 너무 작거나(잡음) 너무 큰 블롭
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.45:  # 원근으로 타원이 되어도 이 이상은 유지됨
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        diameter = 2.0 * float(np.sqrt(area / np.pi))
        out.append((cx, cy, diameter, float(circularity)))
    return out


def _classify_blob_color(hsv: np.ndarray, mask: np.ndarray) -> str | None:
    """블롭 픽셀들의 중앙값 HSV로 공 색 분류. 공이 아니면 None."""
    px = hsv[mask > 0]
    if len(px) < 20:
        return None
    med_h = float(np.median(px[:, 0]))
    med_s = float(np.median(px[:, 1]))
    med_v = float(np.median(px[:, 2]))
    if med_s <= 85 and med_v >= 130:
        return "white"
    if med_s > 80 and (med_h <= 12 or med_h >= 160):
        return "red"
    if med_s > 80 and 15 <= med_h <= 45:
        return "yellow"
    return None  # 손/큐/그림자 등


def _project_candidate(H, cx, cy, dia_px):
    """이미지 블롭 → (x_mm, y_mm, 투영 지름 mm). 범위 밖이면 None."""
    margin = spec.BALL_DIAMETER_MM
    pts = np.array([
        [cx, cy], [cx - dia_px / 2, cy], [cx + dia_px / 2, cy],
    ], dtype=np.float32)
    top = project(H, pts)
    x_mm, y_mm = px_to_mm((top[0][0], top[0][1]))
    if not (-margin <= x_mm <= spec.TABLE_W_MM + margin
            and -margin <= y_mm <= spec.TABLE_H_MM + margin):
        return None
    dia_mm = float(np.linalg.norm(top[2] - top[1])) / PX_PER_MM
    # 공 지름 65.5mm — 시차 확대를 감안해 40~360mm 허용
    if not (40.0 <= dia_mm <= 360.0):
        return None
    return x_mm, y_mm, dia_mm


def detect_balls(bgr: np.ndarray, H: np.ndarray,
                 cloth_raw: np.ndarray | None = None) -> list[Ball]:
    """원본 사진에서 공을 검출하고 좌표를 탑뷰(mm)로 투영해 반환.

    두 가지 후보를 합친다:
      A) 색 범위 블롭 (흰/노/빨 고정 범위)
      B) 천 구멍 블롭 — 경기면 안에서 천 색이 아닌 원형 영역
         (조명 때문에 색 범위를 벗어난 공을 잡는 안전망, 색은 중앙값으로 분류)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 후보: color → [(score, x_mm, y_mm)]
    cands: dict[str, list[tuple[float, float, float]]] = {
        c: [] for c in _BALL_COLORS}

    # A) 색 범위 기반
    for color, ranges in _BALL_COLORS.items():
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        for cx, cy, dia_px, circ in _blob_candidates(mask):
            proj = _project_candidate(H, cx, cy, dia_px)
            if proj is None:
                continue
            x_mm, y_mm, dia_mm = proj
            size_fit = 1.0 - min(abs(dia_mm - spec.BALL_DIAMETER_MM) / 300.0, 1.0)
            cands[color].append((circ * 0.6 + size_fit * 0.4, x_mm, y_mm))

    # B) 천 구멍 기반 (색상 무관 안전망)
    if cloth_raw is not None:
        from .homography import playfield_corners_px
        pf_img = project(np.linalg.inv(H), playfield_corners_px())
        poly = np.zeros(bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(poly, [pf_img.astype(np.int32)], 255)
        holes = cv2.bitwise_and(poly, cv2.bitwise_not(cloth_raw))
        num, labels, stats, _cent = cv2.connectedComponentsWithStats(holes)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 60 or area > holes.size * 0.02:
                continue
            blob = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = contours[0]
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circ = 4 * np.pi * cv2.contourArea(c) / (peri * peri)
            if circ < 0.4:  # 손/큐는 원형이 아님
                continue
            color = _classify_blob_color(hsv, blob)
            if color is None:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            dia_px = 2.0 * float(np.sqrt(area / np.pi))
            proj = _project_candidate(H, cx, cy, dia_px)
            if proj is None:
                continue
            x_mm, y_mm, dia_mm = proj
            size_fit = 1.0 - min(abs(dia_mm - spec.BALL_DIAMETER_MM) / 300.0, 1.0)
            cands[color].append((circ * 0.55 + size_fit * 0.35, x_mm, y_mm))

    balls: list[Ball] = []
    for color, scored in cands.items():
        scored.sort(key=lambda t: -t[0])
        picked: list[tuple[float, float, float]] = []
        for score, x_mm, y_mm in scored:
            # 같은 공을 두 방식이 중복 검출한 경우 병합 (60mm 이내)
            if any(np.hypot(x_mm - px, y_mm - py) < 60.0
                   for _s, px, py in picked):
                continue
            picked.append((score, x_mm, y_mm))
            if len(picked) >= _EXPECTED[color]:
                break
        for score, x_mm, y_mm in picked:
            # 경기면 살짝 밖(쿠션 붙은 공)은 안쪽으로 클램프
            x_mm = float(np.clip(x_mm, spec.BALL_DIAMETER_MM / 2,
                                 spec.TABLE_W_MM - spec.BALL_DIAMETER_MM / 2))
            y_mm = float(np.clip(y_mm, spec.BALL_DIAMETER_MM / 2,
                                 spec.TABLE_H_MM - spec.BALL_DIAMETER_MM / 2))
            balls.append(Ball(color=color, pos_mm=(x_mm, y_mm), score=float(score)))
    return balls


def draw_balls(img: np.ndarray, balls: list[Ball]) -> None:
    """디버그용: 검출된 공 위치에 외곽 원 + 색 라벨 표시 (제자리 수정)."""
    r = int(_BALL_RADIUS_PX)
    for b in balls:
        px, py = mm_to_px(b.pos_mm)
        cv2.circle(img, (int(px), int(py)), r + 4, (255, 0, 255), 2)
        cv2.putText(img, b.color, (int(px) - r, int(py) - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
