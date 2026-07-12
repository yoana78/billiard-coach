"""코너(귀퉁이) 촬영 → 다이아몬드 격자 정합 역산 검증."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.pipeline import image_to_topview
from detection.homography import project, mm_to_px
from tools.make_synthetic import make_corner_photo
from detection import spec


@pytest.mark.parametrize("seed,tilt", [(0, 0.12), (1, 0.16), (2, 0.20)])
def test_corner_topview(seed, tilt):
    crop, H_truth, _balls = make_corner_photo(seed=seed, tilt=tilt)
    result = image_to_topview(crop)
    assert result.ok, f"변환 실패: {result.reason}"
    assert result.partial

    # 프레임 안에 보이는 좌상단 다이아몬드들의 재투영 오차.
    # 코너 촬영은 상하/좌우 대칭이 원리적으로 모호 → 대칭 변형 중 최소로 판정.
    W, Hh = spec.TABLE_W_MM, spec.TABLE_H_MM
    world_pts = [(0.0, 0.0)]
    for (x, y, _s) in spec.diamond_points_mm():
        if x <= W * 0.4 and y <= Hh * 0.6:
            world_pts.append((x, y))

    variants = [
        lambda p: p,
        lambda p: (W - p[0], p[1]),
        lambda p: (p[0], Hh - p[1]),
        lambda p: (W - p[0], Hh - p[1]),
    ]
    canvas_pts = np.array([mm_to_px(p) for p in world_pts], dtype=np.float32)
    truth_img = project(np.linalg.inv(H_truth), canvas_pts)
    h, w = crop.shape[:2]
    inside = [i for i, p in enumerate(truth_img)
              if 0 <= p[0] < w and 0 <= p[1] < h]
    assert len(inside) >= 4

    best_err = np.inf
    for f in variants:
        cpts = np.array([mm_to_px(f(p)) for p in world_pts], dtype=np.float32)
        est_img = project(np.linalg.inv(result.H), cpts)
        errs = np.linalg.norm(truth_img[inside] - est_img[inside], axis=1)
        best_err = min(best_err, float(errs.mean()))
    assert best_err < 25, f"재투영 평균 오차 {best_err:.1f}px"


def test_corner_too_little():
    """쿠션이 거의 안 보이면 실패 안내를 반환해야 한다."""
    crop, _, _ = make_corner_photo(seed=3, tilt=0.15, keep=0.28)
    result = image_to_topview(crop)
    if not result.ok:
        assert result.reason
    else:
        assert result.partial
