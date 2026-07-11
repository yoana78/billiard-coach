"""합성 이미지 기반 파이프라인 정량 검증.

정답 호모그래피를 아는 합성 사진으로:
  - 경기면 모서리 복원 오차
  - 다이아몬드 검출 개수 / 위치 오차
를 검증한다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.pipeline import image_to_topview
from detection.homography import project, playfield_corners_px, PX_PER_MM
from tools.make_synthetic import make_photo


SEEDS_TILTS = [(0, 0.10), (1, 0.18), (2, 0.25), (3, 0.30), (4, 0.15)]


@pytest.mark.parametrize("seed,tilt", SEEDS_TILTS)
def test_topview_accuracy(seed, tilt):
    photo, H_truth, _balls = make_photo(seed=seed, tilt=tilt)
    result = image_to_topview(photo)
    assert result.ok, f"변환 실패: {result.reason}"

    # 1) 모서리 정확도: 정답 H와 추정 H로 경기면 모서리를 원본 사진에
    #    역투영했을 때 서로 얼마나 가까운가 (mm 단위)
    pf = playfield_corners_px()
    truth_in_photo = project(np.linalg.inv(H_truth), pf)
    est_in_photo = project(np.linalg.inv(result.H), pf)
    corner_err_px = np.linalg.norm(truth_in_photo - est_in_photo, axis=1).mean()
    # 사진 픽셀 기준 10px 이내 (1600px 사진에서 충분히 정밀)
    assert corner_err_px < 10, f"모서리 오차 {corner_err_px:.1f}px"

    # 2) 다이아몬드: 20개 중 16개 이상 검출
    assert len(result.diamonds) >= 16, f"다이아몬드 {len(result.diamonds)}/20"

    # 3) 다이아몬드 평균 위치 오차 15mm 이내 (공 지름 65.5mm의 1/4 수준)
    assert 0 <= result.diamond_err_mm < 15, f"오차 {result.diamond_err_mm:.1f}mm"


@pytest.mark.parametrize("seed,tilt", SEEDS_TILTS)
def test_ball_detection(seed, tilt):
    """공 4개(흰1/노1/빨2)가 정답 위치 근처에서 검출되는가."""
    photo, _H, truth_balls = make_photo(seed=seed, tilt=tilt)
    result = image_to_topview(photo)
    assert result.ok

    found = result.balls
    colors = sorted(b.color for b in found)
    assert colors == ["red", "red", "white", "yellow"], f"검출 색: {colors}"

    # 각 정답 공에 대해 같은 색 검출 공이 20mm 이내에 있어야 함
    for (t_color, tx, ty) in truth_balls:
        dists = [
            np.hypot(b.pos_mm[0] - tx, b.pos_mm[1] - ty)
            for b in found if b.color == t_color
        ]
        assert dists and min(dists) < 20, \
            f"{t_color} 공 위치 오차 {min(dists) if dists else 'N/A'}mm"


def test_no_table_image():
    """당구대가 없는 이미지는 실패를 명확히 보고해야 한다."""
    rng = np.random.default_rng(7)
    junk = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
    result = image_to_topview(junk)
    assert not result.ok
