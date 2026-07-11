"""정지 이미지에서 당구대 천(경기면) 사각형 검출.

전략: HSV 색공간에서 파란/초록 계열 천 마스크를 만들고,
가장 큰 컨투어를 4각형으로 근사해 경기면 모서리 4점을 얻는다.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# 천 색상 후보 범위 (H: 0~179)
_CLOTH_RANGES = {
    "blue": ((90, 60, 40), (130, 255, 255)),
    "green": ((35, 60, 40), (85, 255, 255)),
}


@dataclass
class TableQuad:
    corners: np.ndarray  # (4,2) float32, 순서: tl, tr, br, bl (장쿠션이 위/아래)
    cloth_color: str     # 'blue' | 'green'
    mask_area_ratio: float  # 이미지 대비 천 면적 비율


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """마스크에서 가장 큰 연결 영역만 남긴다 (옆 당구대/옷 등 분리 배제)."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def _adaptive_range(hsv: np.ndarray) -> tuple[tuple, tuple, str] | None:
    """화면 중앙 패치(대부분 천)를 샘플링해 적응형 색 범위 생성.

    물 빠진 천, 조명 변색 등 고정 범위가 놓치는 경우를 커버한다.
    """
    h, w = hsv.shape[:2]
    patch = hsv[int(h * 0.35):int(h * 0.65), int(w * 0.35):int(w * 0.65)]
    med_h = float(np.median(patch[:, :, 0]))
    med_s = float(np.median(patch[:, :, 1]))
    med_v = float(np.median(patch[:, :, 2]))
    # 중앙이 무채색(벽/바닥)이거나 너무 어두우면 적응형 포기
    if med_s < 25 or med_v < 30:
        return None
    lo = (max(0, int(med_h - 14)), max(20, int(med_s * 0.35)), max(25, int(med_v * 0.3)))
    hi = (min(179, int(med_h + 14)), 255, 255)
    name = "green" if 35 <= med_h <= 85 else "blue"
    return lo, hi, name


def cloth_mask(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """(정제된 최대 연결영역 마스크, 원본 색상 마스크, 색상 이름) 반환."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    candidates: list[tuple[np.ndarray, str]] = []
    adaptive = _adaptive_range(hsv)
    if adaptive is not None:
        lo, hi, name = adaptive
        candidates.append((cv2.inRange(hsv, np.array(lo), np.array(hi)), name))
    for color, (lo, hi) in _CLOTH_RANGES.items():
        candidates.append((cv2.inRange(hsv, np.array(lo), np.array(hi)), color))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    best = None  # (최대 연결영역 크기, 정제 마스크, 원본 마스크, 색이름)
    for raw, color in candidates:
        clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
        comp = _largest_component(clean)
        area = int(cv2.countNonZero(comp))
        if best is None or area > best[0]:
            best = (area, comp, raw, color)

    _area, comp, raw, color = best
    return comp, raw, color


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """좌상→우상→우하→좌하 순으로 정렬."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _approx_quad(contour: np.ndarray) -> np.ndarray | None:
    """컨투어를 4점 다각형으로 근사. epsilon을 점점 키우며 시도."""
    peri = cv2.arcLength(contour, True)
    for eps_ratio in (0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(contour, eps_ratio * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    # 실패 시 최소 외접 회전사각형으로 대체
    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect)


def detect_table(bgr: np.ndarray) -> TableQuad | None:
    """이미지에서 경기면 4각형을 찾는다. 실패 시 None."""
    mask, raw_mask, color = cloth_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = bgr.shape[0] * bgr.shape[1]
    if area < img_area * 0.05:  # 천이 화면의 5% 미만이면 당구대로 보지 않음
        return None
    hull = cv2.convexHull(largest)
    quad = _approx_quad(hull)
    if quad is None:
        return None
    corners = _order_corners(quad)

    # 검증: 사각형 내부가 실제로 균일한 천 색인가 (노이즈/오검출 방지).
    # 공·조명 반사를 감안해 내부의 60% 이상이 원본 색상 마스크에 들어야 함.
    poly = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(poly, [corners.astype(np.int32)], 255)
    inside = int(cv2.countNonZero(poly))
    if inside == 0:
        return None
    cloth_inside = int(cv2.countNonZero(cv2.bitwise_and(raw_mask, poly)))
    if cloth_inside / inside < 0.6:
        return None

    return TableQuad(corners=corners, cloth_color=color, mask_area_ratio=area / img_area)
