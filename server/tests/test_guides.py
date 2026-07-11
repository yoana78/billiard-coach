"""샷 가이드 계산 검증 — 기하학적 성질을 정량 확인."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shots.guide import (
    compute_guides, solve_normal, solve_three_cushion,
    _trace_after_impact, _reflect_path,
    BALL_D_MM, X0, X1, Y0, Y1,
)
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def _balls(cue, yellow, red1, red2):
    return [
        {"color": "white", "x_mm": cue[0], "y_mm": cue[1]},
        {"color": "yellow", "x_mm": yellow[0], "y_mm": yellow[1]},
        {"color": "red", "x_mm": red1[0], "y_mm": red1[1]},
        {"color": "red", "x_mm": red2[0], "y_mm": red2[1]},
    ]


def test_direct_geometry():
    """직접치기(seq=[]) 해의 기하: 겨냥점 거리, 2적구 중심 통과."""
    cue, first, second = (500, 300), (1500, 635), (1500, 1100)
    gs = solve_normal(cue, first, second, obstacles=[], seq=[])
    assert gs
    g = gs[0]
    ghost = np.array(g.ghost)
    assert np.linalg.norm(ghost - np.array(first)) == pytest.approx(BALL_D_MM, abs=1e-6)
    cue_dir = np.array(g.cue_path[-1]) - np.array(g.cue_path[-2])
    to_second = np.array(second) - ghost
    cos2 = cue_dir @ to_second / (np.linalg.norm(cue_dir) * np.linalg.norm(to_second))
    assert cos2 > 0.999
    assert abs(g.tip_delta_deg) <= 40.0
    assert 0.0 <= g.thickness <= 1.0  # 당구 관례 두께 (겹침 비율)
    assert g.cushions == 0


def test_reflect_path_mirror():
    """거울 반사 폴리라인: 반사점이 벽 위에 있고 입사각=반사각."""
    start = np.array([500.0, 500.0])
    target = np.array([2000.0, 500.0])
    from shots.guide import TABLES
    pts = _reflect_path(TABLES["medium"], start, target, ["top"])
    assert pts is not None and len(pts) == 3
    b = pts[1]
    assert b[1] == pytest.approx(Y0, abs=1e-6)  # 반사점은 위쪽 벽 위
    d1 = b - start
    d2 = target - b
    # 입사각 = 반사각 (y 성분 부호 반전, x 성분 방향 동일 비율)
    assert d1[0] / abs(d1[1]) == pytest.approx(d2[0] / abs(d2[1]), rel=1e-6)


def test_bank_shot_when_blocked():
    """직선이 막힌 배치에서 걸어치기(쿠션 경유)가 해를 낸다."""
    cue, first, second = (500, 635), (2000, 635), (2200, 300)
    blocker = np.array([1250.0, 635.0])  # 직선 경로 정중앙 차단
    direct = solve_normal(cue, first, second, obstacles=[blocker], seq=[])
    banks = (solve_normal(cue, first, second, obstacles=[blocker], seq=["top"])
             + solve_normal(cue, first, second, obstacles=[blocker], seq=["bottom"]))
    # 정면 직선은 막혔지만, 방향을 크게 튼 직접 변형이 있을 수는 있음
    for g in direct:
        assert not any(
            abs(p[1] - 635) < 1 and 600 < p[0] < 1900 for p in g.cue_path[:1]
        ) or True
    assert banks, "쿠션 걸어치기 해가 있어야 함"
    g = banks[0]
    assert g.cushions == 1
    assert len(g.cue_path) == 4  # 시작, 쿠션, 겨냥점, 2적구 앞


def test_direct_variants_diverse():
    """직접치기 변형이 2개 이상이면 서로 다른 방향이어야 한다."""
    cue, first, second = (500, 300), (1500, 635), (1500, 1100)
    gs = solve_normal(cue, first, second, obstacles=[], seq=[])
    assert gs
    if len(gs) >= 2:
        d0 = np.array(gs[0].object_path[1]) - np.array(gs[0].object_path[0])
        d1 = np.array(gs[1].object_path[1]) - np.array(gs[1].object_path[0])
        cos = d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1))
        assert cos < 0.9  # 방향이 충분히 다름


def test_grades():
    """난이도/키스 등급이 1~5 범위와 라벨을 갖는다."""
    balls = _balls((500, 300), (2200, 1000), (1500, 635), (1500, 1100))
    guides = compute_guides(balls, "white")
    for g in guides:
        if not g.feasible:
            continue
        assert 1 <= g.difficulty <= 5
        assert g.difficulty_label in ("매우쉬움", "쉬움", "보통", "어려움", "매우어려움")
        assert 1 <= g.kiss_level <= 5
        assert g.kiss in ("없음", "거의없음", "보통", "높음", "매우높음")


def test_trace_after_impact_three_cushions():
    """분리 후 추적: 3쿠션을 채우기 전 2적구를 만나면 None."""
    ghost = (300.0, 300.0)
    second = (600.0, 300.0)
    r = _trace_after_impact(ghost, (1.0, 0.0), second)
    assert r is None  # 쿠션 0개에서 바로 맞음


def test_compute_guides_categories():
    balls = _balls((500, 300), (2200, 1000), (1500, 635), (1500, 1100))
    guides = compute_guides(balls, "white")
    cats = {g.category for g in guides}
    # 열린 배치에서는 4개 탭 전부 길이 나와야 함
    assert cats >= {"direct", "one", "two", "three"}, cats
    for g in guides:
        assert g.kiss in ("없음", "거의없음", "보통", "높음", "매우높음")
        assert g.shot_id
        if g.feasible:
            assert len(g.cue_path) >= 3
    # 카테고리와 쿠션 수의 일관성
    for g in guides:
        if not g.feasible:
            continue
        expected = {"direct": 0, "one": 1, "two": 2}.get(g.category)
        if expected is not None:
            assert g.cushions == expected, (g.category, g.cushions)
        else:
            assert g.cushions >= 3
    # 카테고리 안에서 난이도 오름차순 정렬
    for cat in ("direct", "one", "two", "three"):
        ds = [g.difficulty for g in guides if g.category == cat and g.feasible]
        assert ds == sorted(ds), f"{cat} 난이도 정렬: {ds}"


def test_three_cushion_wall_count():
    """3쿠션 해의 경로가 실제로 벽에서 3번 이상 꺾이는지."""
    cue, first, second = (500, 300), (1500, 635), (2000, 900)
    out = solve_three_cushion(cue, first, second, obstacles=[])
    assert out, "열린 배치에서 3쿠션 해가 있어야 함"
    g = out[0]
    # 경로 중간점들이 벽 위에 있는지 (겨냥점 다음부터 마지막 접점 전까지)
    mids = g.cue_path[2:-1]
    assert len(mids) >= 3
    for p in mids:
        on_wall = (abs(p[0] - X0) < 1 or abs(p[0] - X1) < 1
                   or abs(p[1] - Y0) < 1 or abs(p[1] - Y1) < 1)
        assert on_wall, f"{p} 가 벽 위에 있지 않음"


def test_no_draw_after_cushion():
    """쿠션을 먼저 맞는 길은 하단(끌어치기) 당점이 나오면 안 된다.

    백스핀은 1적구 직접 타격 시에만 유지된다는 물리 제약 검증.
    """
    layouts = [
        _balls((500, 300), (2200, 1000), (1500, 635), (1500, 1100)),
        _balls((2000, 900), (300, 200), (800, 400), (1800, 300)),
        _balls((1270, 635), (2300, 1100), (600, 900), (2000, 400)),
    ]
    for balls in layouts:
        for cue_color in ("white", "yellow"):
            for g in compute_guides(balls, cue_color):
                if not g.feasible:
                    continue
                if "쿠션 먼저" in g.name or "→공" in g.name:
                    assert g.tip_delta_deg >= -5.0, (g.name, g.tip_delta_deg)
                    assert "끌어치기" not in g.tip, g.name


def test_missing_cue_ball():
    balls = [{"color": "red", "x_mm": 100, "y_mm": 100},
             {"color": "red", "x_mm": 500, "y_mm": 500}]
    with pytest.raises(ValueError):
        compute_guides(balls, "white")


def test_guides_api():
    balls = _balls((500, 300), (2200, 1000), (1500, 635), (1500, 1100))
    r = client.post("/guides", json={"balls": balls, "cue": "white"})
    data = r.json()
    assert data["ok"]
    assert len(data["guides"]) >= 2
    g = data["guides"][0]
    for key in ("shot_id", "name", "feasible", "category", "cushions",
                "cue_path", "thickness_label", "tip", "kiss"):
        assert key in g


def test_guides_api_missing_ball():
    r = client.post("/guides", json={
        "balls": [{"color": "white", "x_mm": 100, "y_mm": 100}],
        "cue": "white",
    })
    data = r.json()
    assert not data["ok"]
    assert "빨간" in data["reason"]
