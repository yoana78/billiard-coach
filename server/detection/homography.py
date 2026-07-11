"""호모그래피(원근 변환) 유틸 — 원본 사진 좌표 ↔ 탑뷰 mm 좌표."""
import cv2
import numpy as np

from . import spec

# 탑뷰 렌더링 해상도: 1mm 당 픽셀 수
PX_PER_MM = 0.5
# 탑뷰 캔버스에 포함할 레일 여백 (다이아몬드가 레일 위에 있으므로 필요)
MARGIN_MM = spec.RAIL_WIDTH_MM + 30.0


def mm_to_px(pt_mm) -> tuple[float, float]:
    """탑뷰 mm 좌표 → 탑뷰 캔버스 px 좌표."""
    x, y = pt_mm
    return ((x + MARGIN_MM) * PX_PER_MM, (y + MARGIN_MM) * PX_PER_MM)


def px_to_mm(pt_px) -> tuple[float, float]:
    x, y = pt_px
    return (x / PX_PER_MM - MARGIN_MM, y / PX_PER_MM - MARGIN_MM)


def canvas_size() -> tuple[int, int]:
    """탑뷰 캔버스 (width, height) px."""
    w = int(round((spec.TABLE_W_MM + 2 * MARGIN_MM) * PX_PER_MM))
    h = int(round((spec.TABLE_H_MM + 2 * MARGIN_MM) * PX_PER_MM))
    return w, h


def playfield_corners_px() -> np.ndarray:
    """탑뷰 캔버스에서 경기면 4모서리 px (tl, tr, br, bl)."""
    pts = [(0, 0), (spec.TABLE_W_MM, 0), (spec.TABLE_W_MM, spec.TABLE_H_MM), (0, spec.TABLE_H_MM)]
    return np.array([mm_to_px(p) for p in pts], dtype=np.float32)


def cloth_boundary_corners_px() -> np.ndarray:
    """탑뷰 캔버스에서 천 경계(쿠션 바깥) 4모서리 px (tl, tr, br, bl)."""
    return np.array([mm_to_px(p) for p in spec.cloth_boundary_corners_mm()],
                    dtype=np.float32)


def homography_from_corners(image_corners: np.ndarray) -> np.ndarray:
    """사진의 천 경계 4모서리(tl,tr,br,bl) → 탑뷰 px 호모그래피.

    사진에서 검출되는 사각형은 천 경계(쿠션 바깥)이므로 경기면보다
    CUSHION_OVERHANG 만큼 큰 사각형으로 매핑한다.
    장쿠션(긴 변)이 탑뷰의 가로가 되도록 방향을 맞춘다:
    입력 모서리에서 위/아래 변의 평균 길이가 좌/우 변보다 짧으면 90도 돌려 매핑.
    """
    c = image_corners.astype(np.float32)
    top = np.linalg.norm(c[1] - c[0])
    bottom = np.linalg.norm(c[2] - c[3])
    left = np.linalg.norm(c[3] - c[0])
    right = np.linalg.norm(c[2] - c[1])
    if (top + bottom) < (left + right):
        # 세로로 길게 찍힌 경우: tl→(우상)이 되도록 한 칸 회전
        c = np.array([c[3], c[0], c[1], c[2]], dtype=np.float32)
    dst = cloth_boundary_corners_px()
    H, _ = cv2.findHomography(c, dst)
    return H


def refine_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray | None:
    """대응점(원본px → 탑뷰px) 다수로 호모그래피 재추정.

    점이 많으면 RANSAC으로 이상점을 걸러내고, 적으면(부분 촬영 등)
    최소자승으로 전부 사용한다. 대응점은 이미 기대 위치 창 안에서만
    수집되므로 큰 이상점은 드물다.
    """
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    if len(src) < 4:
        return None
    if len(src) >= 6:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 12.0)
        if H is not None:
            return H
    H, _ = cv2.findHomography(src, dst, 0)
    return H


def warp_topview(bgr: np.ndarray, H: np.ndarray) -> np.ndarray:
    w, h = canvas_size()
    return cv2.warpPerspective(bgr, H, (w, h))


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N,2) 점들을 H로 투영."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
