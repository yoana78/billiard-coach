"""탑뷰로 변환된 이미지의 레일 영역에서 다이아몬드 포인트 검출.

전략: 초기 호모그래피로 탑뷰를 만들면 다이아몬드는 규격상 위치 근처에
와 있어야 한다. 각 기대 위치 주변의 작은 창(window)에서 밝은 원형 블롭을
찾아 실제 중심을 정밀 측정한다. (기대 위치 기반 탐색이라 오검출이 적고,
찾은 점들은 호모그래피 재정밀화(refine)에 사용된다.)
"""
from dataclasses import dataclass

import cv2
import numpy as np

from . import spec
from .homography import mm_to_px, PX_PER_MM

# 기대 위치 주변 탐색 반경 (mm) — 초기 호모그래피 오차 허용치
SEARCH_RADIUS_MM = 60.0


@dataclass
class DiamondHit:
    expected_mm: tuple[float, float]   # 규격상 위치 (mm)
    found_px: tuple[float, float]      # 탑뷰 캔버스에서 실측 중심 (px)
    side: str
    score: float                       # 블롭 품질 (0~1, 원형도 기반)


def _find_bright_blob(win_gray: np.ndarray,
                      ppm: float = PX_PER_MM) -> tuple[float, float, float] | None:
    """창 안에서 가장 다이아몬드다운 밝은 블롭의 (cx, cy, score) 반환.

    ppm: 창 이미지의 mm당 픽셀 수 (축소 탑뷰에서 쓸 때 반드시 지정).
    """
    # 창 내 상대적으로 밝은 픽셀 추출 (레일 목재 대비 흰/아이보리 점)
    block = max(9, int(31 * ppm / 0.5)) | 1  # 배율에 맞춘 블록 크기 (홀수)
    block = min(block, (min(win_gray.shape) - 1) | 1)
    if block < 3:
        return None
    thr = cv2.adaptiveThreshold(
        win_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, -20
    )
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    # 다이아몬드 실물 지름 대략 10~25mm → px 면적 범위
    min_area = (8 * ppm) ** 2 * 0.5
    max_area = (40 * ppm) ** 2 * 3.5
    cx0, cy0 = win_gray.shape[1] / 2, win_gray.shape[0] / 2
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.4:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        # 창 중앙(기대 위치)에 가까울수록 가점
        dist = np.hypot(cx - cx0, cy - cy0)
        score = circularity * (1.0 - min(dist / (max(cx0, cy0) * 1.5), 1.0))
        if best is None or score > best[2]:
            best = (cx, cy, score)
    return best


def detect_diamonds(topview_bgr: np.ndarray) -> list[DiamondHit]:
    """탑뷰 이미지에서 20개 다이아몬드 기대 위치를 탐색해 검출 결과 반환."""
    gray = cv2.cvtColor(topview_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    r = int(SEARCH_RADIUS_MM * PX_PER_MM)
    hits: list[DiamondHit] = []
    for (x_mm, y_mm, side) in spec.diamond_points_mm():
        px, py = mm_to_px((x_mm, y_mm))
        x0, x1 = int(px - r), int(px + r)
        y0, y1 = int(py - r), int(py + r)
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue  # 부분 촬영 등으로 창이 캔버스를 벗어나면 스킵
        win = gray[y0:y1, x0:x1]
        if win.size == 0 or win.std() < 3:  # 빈 영역(검은 배경)이면 스킵
            continue
        blob = _find_bright_blob(win)
        if blob is None:
            continue
        cx, cy, score = blob
        hits.append(
            DiamondHit(
                expected_mm=(x_mm, y_mm),
                found_px=(x0 + cx, y0 + cy),
                side=side,
                score=float(score),
            )
        )
    return hits
