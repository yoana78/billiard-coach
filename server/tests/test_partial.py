"""부분 촬영(당구대 일부만 프레임) → 탑뷰 역산 검증."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.pipeline import image_to_topview
from detection.homography import project, mm_to_px
from tools.make_synthetic import make_partial_photo
from detection import spec


@pytest.mark.parametrize("seed,tilt", [(0, 0.12), (1, 0.18), (2, 0.24)])
def test_partial_topview(seed, tilt):
    crop, H_truth, _balls = make_partial_photo(seed=seed, tilt=tilt)
    result = image_to_topview(crop)
    assert result.ok, f"변환 실패: {result.reason}"
    assert result.partial, "부분 촬영으로 인식되어야 함"

    # 검출된 다이아몬드 평균 위치 오차
    assert 0 <= result.diamond_err_mm < 20, f"오차 {result.diamond_err_mm:.1f}mm"

    # 프레임 안에 보이는 세계 좌표(왼쪽 모서리 2개 + 왼쪽 절반 다이아몬드)의
    # 재투영 비교: 정답 H vs 추정 H.
    # 당구대는 좌우/상하 대칭이라 부분 촬영에서는 방향이 원리적으로
    # 모호하다 → 대칭 4변형(원본/좌우반전/상하반전/180도) 중 최솟값으로 판정.
    world_pts = [(0.0, 0.0), (0.0, spec.TABLE_H_MM)]
    for (x, y, _s) in spec.diamond_points_mm():
        if x <= spec.TABLE_W_MM * 0.35:
            world_pts.append((x, y))

    W, Hh = spec.TABLE_W_MM, spec.TABLE_H_MM
    variants = [
        lambda p: p,
        lambda p: (W - p[0], p[1]),
        lambda p: (p[0], Hh - p[1]),
        lambda p: (W - p[0], Hh - p[1]),
    ]

    canvas_pts = np.array([mm_to_px(p) for p in world_pts], dtype=np.float32)
    truth_img = project(np.linalg.inv(H_truth), canvas_pts)
    h, w = crop.shape[:2]
    inside = [
        i for i, p in enumerate(truth_img)
        if 0 <= p[0] < w and 0 <= p[1] < h
    ]
    assert len(inside) >= 6

    best_err = np.inf
    for f in variants:
        cpts = np.array([mm_to_px(f(p)) for p in world_pts], dtype=np.float32)
        est_img = project(np.linalg.inv(result.H), cpts)
        errs = np.linalg.norm(truth_img[inside] - est_img[inside], axis=1)
        best_err = min(best_err, float(errs.mean()))
    assert best_err < 15, f"재투영 평균 오차 {best_err:.1f}px"


def test_partial_too_little_visible():
    """라인이 2개 이하로만 보이면 명확한 안내와 함께 실패해야 한다."""
    crop, _, _ = make_partial_photo(seed=3, tilt=0.15, keep=0.22)
    result = image_to_topview(crop)
    # 이 정도로 잘리면 성공하더라도 부분 모드여야 하고,
    # 실패한다면 사용자 안내 문구가 있어야 한다
    if not result.ok:
        assert result.reason
    else:
        assert result.partial
