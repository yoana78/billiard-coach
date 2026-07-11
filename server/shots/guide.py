"""4구 샷 패턴 라이브러리 — 룰 바탕 가이드 계산.

물리 모델 (이상화):
  - 공-공: 1적구는 중심선(n) 방향 출발, 수구는 당점에 따라
    접선(t0)에서 ±TIP_MAX_DEG 안에서 분리.
  - 공-쿠션: 거울 반사 (스핀/속도에 따른 변화는 아직 미반영).
  - 공 중심의 이동 가능 영역은 쿠션에서 공 반지름만큼 안쪽 사각형.

카테고리:
  - normal: 직접치기 + 1/2쿠션 걸어치기 (수구가 1적구를 맞기 전 쿠션 경유)
  - three: 3쿠션 — 1적구 분리 후 쿠션 3개 이상 맞고 2적구
키스 위험은 등급(낮음/중간/높음), 두께/당점은 상대 표기.
"""
from dataclasses import dataclass, field

import numpy as np

BALL_D_MM = 65.5
BALL_R_MM = BALL_D_MM / 2
TABLE_W_MM = 2540.0
TABLE_H_MM = 1270.0

# 공 '중심'이 움직일 수 있는 사각형 (쿠션에서 반지름만큼 안쪽)
X0, X1 = BALL_R_MM, TABLE_W_MM - BALL_R_MM
Y0, Y1 = BALL_R_MM, TABLE_H_MM - BALL_R_MM

MIN_FULLNESS = 0.15
MAX_FULLNESS = 0.97   # 정면 가까운 두께도 허용 (밀어/끌어치기 모드가 사용)
TIP_MAX_DEG = 40.0
# 두껍게 맞추고 밀어치기(전진)/끌어치기(후진)로 수구를 거의 직선으로
# 보내는 모드의 허용 원뿔 각도와 최소 두께(정면비율)
FOLLOW_CONE_DEG = 25.0
FOLLOW_MIN_FULLNESS = 0.6

# 키스 위험 5등급 거리 기준 (경로와 방해공 중심 거리, mm)
KISS_LEVELS = [(400.0, 1, "없음"), (250.0, 2, "거의없음"),
               (150.0, 3, "보통"), (80.0, 4, "높음"), (0.0, 5, "매우높음")]
DIFFICULTY_LABELS = {1: "매우쉬움", 2: "쉬움", 3: "보통", 4: "어려움", 5: "매우어려움"}

_WALLS = ("top", "bottom", "left", "right")


@dataclass
class Guide:
    shot_id: str
    name: str
    feasible: bool
    category: str = "normal"             # 'normal' | 'three'
    cushions: int = 0                    # 경유 쿠션 수 (전체)
    reason: str = ""
    cue_path: list = field(default_factory=list)     # 수구 경로 꼭짓점들 (mm)
    object_path: list = field(default_factory=list)  # 1적구 예상 진행 (mm)
    ghost: tuple | None = None           # 겨냥점(고스트볼 중심)
    thickness: float = 0.0
    thickness_label: str = ""
    # 타점(두께) 시각화용: 1적구 중심 대비 겨냥 가로 오프셋 (공 지름 단위,
    # 진행 방향 기준 +면 오른쪽). |aim_offset| = sin(컷각), 0=정면 겨냥.
    aim_offset: float = 0.0
    tip: str = "중단"
    tip_delta_deg: float = 0.0
    kiss: str = "없음"                   # 없음/거의없음/보통/높음/매우높음
    kiss_level: int = 1                  # 1(없음)~5(매우높음)
    difficulty: int = 3                  # 1(매우쉬움)~5(매우어려움)
    difficulty_label: str = "보통"
    score: float = 0.0                   # 추천 정렬용 (치기 쉬움)


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _seg_dist(p, a, b) -> float:
    p, a, b = map(lambda x: np.asarray(x, dtype=float), (p, a, b))
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-9 else float(np.clip((p - a) @ ab / denom, 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def _thickness_label(t: float) -> str:
    """당구 관례 두께 t = 1 - |겨냥 오프셋/공지름| (겹쳐 보이는 비율)."""
    if t >= 0.9:
        return "정면 (온두께)"
    if t >= 0.7:
        return "3/4 두께"
    if t >= 0.45:
        return "1/2 두께"
    if t >= 0.28:
        return "1/3 두께"
    if t >= 0.15:
        return "1/4 두께"
    return "아주 얇게"


def _tip_label(delta_deg: float) -> str:
    if delta_deg > 8:
        return "상단 (밀어치기)"
    if delta_deg < -8:
        return "하단 (끌어치기)"
    return "중단"


def _inside_table(p) -> bool:
    return X0 <= p[0] <= X1 and Y0 <= p[1] <= Y1


def _mirror(p, wall):
    x, y = float(p[0]), float(p[1])
    if wall == "left":
        return np.array([2 * X0 - x, y])
    if wall == "right":
        return np.array([2 * X1 - x, y])
    if wall == "top":
        return np.array([x, 2 * Y0 - y])
    return np.array([x, 2 * Y1 - y])


def _wall_cross(p, q, wall):
    """선분 p→q가 벽 직선과 만나는 점 (벽 범위 안일 때만)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    if wall in ("left", "right"):
        wx = X0 if wall == "left" else X1
        if abs(d[0]) < 1e-9:
            return None
        t = (wx - p[0]) / d[0]
        if not (1e-6 < t < 1 - 1e-6):
            return None
        pt = p + t * d
        if Y0 - 1 <= pt[1] <= Y1 + 1:
            return pt
        return None
    wy = Y0 if wall == "top" else Y1
    if abs(d[1]) < 1e-9:
        return None
    t = (wy - p[1]) / d[1]
    if not (1e-6 < t < 1 - 1e-6):
        return None
    pt = p + t * d
    if X0 - 1 <= pt[0] <= X1 + 1:
        return pt
    return None


def _reflect_path(start, target, seq):
    """start → (seq 순서 쿠션 반사) → target 폴리라인. 불가면 None."""
    virt = np.asarray(target, float)
    for w in reversed(seq):
        virt = _mirror(virt, w)
    pts = [np.asarray(start, float)]
    cur_virt = virt
    for w in seq:
        b = _wall_cross(pts[-1], cur_virt, w)
        if b is None:
            return None
        pts.append(b)
        cur_virt = _mirror(cur_virt, w)
    pts.append(np.asarray(target, float))
    return pts


def _path_blocked(pts, obstacles, skip_last_mm=0.0) -> bool:
    """폴리라인이 장애공에 막히는가."""
    for a, b in zip(pts, pts[1:]):
        b_eff = b
        if skip_last_mm > 0 and np.array_equal(b, pts[-1]):
            d = _unit(b - a)
            b_eff = b - d * skip_last_mm
        for o in obstacles:
            if _seg_dist(o, a, b_eff) < BALL_D_MM * 0.95:
                return True
    return False


def _path_kiss(pts, opponent) -> tuple[int, str]:
    """(등급 1~5, 라벨). 1적구도 사실상 방해가 되지만 MVP는 방해공만 본다."""
    if opponent is None:
        return 1, "없음"
    dmin = min(_seg_dist(opponent, a, b) for a, b in zip(pts, pts[1:]))
    for thr, level, label in KISS_LEVELS:
        if dmin >= thr:
            return level, label
    return 5, "매우높음"


def _path_length(pts) -> float:
    return float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a))
                     for a, b in zip(pts, pts[1:])))


def _set_difficulty(g: "Guide") -> None:
    """난이도 1~5: 당점 요구량 + 얇은 두께 + 쿠션 수 + 경로 길이."""
    d = 0.0
    d += abs(g.tip_delta_deg) / TIP_MAX_DEG * 1.4      # 당점을 많이 써야 함
    d += max(0.0, 0.30 - g.thickness) * 4.0            # 얇을수록 어려움
    d += g.cushions * 0.55
    d += max(0.0, _path_length(g.cue_path) - 1800.0) / 3000.0
    level = 1 + min(4, int(d / 0.8))
    g.difficulty = level
    g.difficulty_label = DIFFICULTY_LABELS[level]


def _aim_at_first(arrive_dir_from, first, second_dir_hint=None):
    """도착 방향이 주어졌을 때 필요한 겨냥/분리 계산은 호출부에서."""


def _category_of(total_cushions: int) -> str:
    return {0: "direct", 1: "one", 2: "two"}.get(total_cushions, "three")


def _wall_seqs(n: int) -> list[list[str]]:
    """길이 n의 쿠션 시퀀스 (같은 쿠션 연속 제외)."""
    if n == 0:
        return [[]]
    out = [[w] for w in _WALLS]
    for _ in range(n - 1):
        out = [s + [w] for s in out for w in _WALLS if w != s[-1]]
    return out


def solve_combo(cue, first, second, obstacles, pre_seq, post_seq,
                max_variants: int = 1) -> list[Guide]:
    """일반화된 샷 해법:
      수구 → (pre_seq 쿠션들) → 1적구 → 분리 → (post_seq 쿠션들) → 2적구

    거울 반사 원리를 앞뒤 양쪽에 적용해 닫힌 해로 계산한다.
    같은 배치라도 1적구를 보내는 방향(θ)에 따라 여러 길이 있으므로,
    방향이 충분히 다른(35° 이상) 상위 해를 최대 max_variants개 반환.
    """
    cue = np.asarray(cue, float)
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    if float(np.linalg.norm(second - first)) < BALL_D_MM * 1.5:
        return []
    others = [o for o in obstacles if not np.allclose(o, second)]

    cands = []  # (score, theta, n, ghost, entry, post, delta, fullness)
    for theta in np.arange(0.0, 360.0, 2.0):
        rad = np.deg2rad(theta)
        n = np.array([np.cos(rad), np.sin(rad)])   # 1적구 출발 방향
        ghost = first - n * BALL_D_MM
        if not _inside_table(ghost):
            continue
        entry = _reflect_path(cue, ghost, pre_seq)
        if entry is None:
            continue
        u = _unit(entry[-1] - entry[-2])           # 도착(충돌) 방향
        fullness = float(u @ n)
        if not (MIN_FULLNESS <= fullness <= MAX_FULLNESS):
            continue
        t0 = _unit(u - fullness * n)
        # 분리 후 경로: ghost → (post_seq) → 2적구
        post = _reflect_path(ghost, second, post_seq)
        if post is None:
            continue
        v0 = _unit(post[1] - post[0])              # 분리 직후 방향
        # 분리 방향 달성 방법 판정:
        #  1) tip: 접선(t0)에서 당점으로 ±40° 보정
        #  2) follow: 두껍게 + 밀어치기 → 진행 방향(u) 근처로 관통
        #  3) draw: 두껍게 + 끌어치기 → 진행 반대(-u) 근처로 후진
        mode = None
        delta = 0.0
        along = float(v0 @ t0)
        if along > 0:
            d_tip = float(np.rad2deg(np.arctan2(float(v0 @ n), along)))
            if abs(d_tip) <= TIP_MAX_DEG:
                mode, delta = "tip", d_tip
        if mode is None and fullness >= FOLLOW_MIN_FULLNESS:
            ang_fwd = float(np.rad2deg(np.arccos(np.clip(float(v0 @ u), -1, 1))))
            ang_back = float(np.rad2deg(np.arccos(np.clip(float(-(v0 @ u)), -1, 1))))
            if ang_fwd <= FOLLOW_CONE_DEG:
                mode, delta = "follow", 32.0
            elif ang_back <= FOLLOW_CONE_DEG:
                mode, delta = "draw", -32.0
        if mode is None:
            continue
        # 물리 제약: 백스핀(하단/끌기)은 1적구를 '직접' 칠 때만 유지된다.
        # 쿠션을 먼저 맞으면 구름 회전으로 바뀌어 하단 효과가 사라지고,
        # 직접 타격이라도 거리가 멀면 백스핀이 닳아 끌기가 안 된다.
        if pre_seq and (mode == "draw" or (mode == "tip" and delta < -5.0)):
            continue
        if mode == "draw" and float(np.linalg.norm(ghost - cue)) > 1500.0:
            continue
        if _path_blocked(entry, obstacles, skip_last_mm=BALL_R_MM):
            continue
        # 수구가 1적구를 '미리' 관통하면 안 됨 — 마지막 접근(겨냥점 직전
        # 1.2공지름)만 접촉 허용하고 그 앞 구간 전체를 검사
        trim = entry[-1] - u * BALL_D_MM * 1.2
        entry_body = entry[:-1] + [trim]
        if any(_seg_dist(first, a, b) < BALL_D_MM * 0.95
               for a, b in zip(entry_body, entry_body[1:])):
            continue
        # 분리 후 경로 검증: 마지막 접점 전에 2적구를 스치거나 다른 공에 막히면 안 됨
        v_last = _unit(post[-1] - post[-2])
        contact = second - v_last * BALL_D_MM
        post_pts = post[:-1] + [contact]
        if any(_seg_dist(second, a, b) < BALL_D_MM
               for a, b in zip(post_pts[:-2], post_pts[1:-1])):
            continue
        if _path_blocked(post_pts, others, skip_last_mm=BALL_R_MM):
            continue
        # 당구 관례 두께 t = 1 - |겨냥 오프셋| (겹쳐 보이는 비율)
        aim_offset = -float(u[0] * n[1] - u[1] * n[0])  # -cross(u,n) = ∓sinθ
        player_t = 1.0 - abs(aim_offset)
        # 두께 적정성을 주 기준으로, 당점 요구량은 약한 가중치만.
        # 밀어/끌어치기(두껍게 정면)는 실전에서 가장 쉬운 축에 속함.
        if mode == "tip":
            score = (-abs(player_t - 0.55)
                     - 0.1 * (len(pre_seq) + len(post_seq))
                     - 0.05 * abs(delta) / TIP_MAX_DEG)
        else:
            score = 0.1 - 0.1 * (len(pre_seq) + len(post_seq))
        cands.append((score, theta, n, ghost, entry, post_pts,
                      delta, player_t, aim_offset))

    cands.sort(key=lambda c: -c[0])
    out: list[Guide] = []
    used_thetas: list[float] = []
    total_cushions = len(pre_seq) + len(post_seq)
    for (score, theta, n, ghost, entry, post_pts,
         delta, player_t, aim_offset) in cands:
        # 이미 채택한 길과 1적구 방향이 비슷하면 같은 길로 간주
        if any(min(abs(theta - t), 360 - abs(theta - t)) < 35.0
               for t in used_thetas):
            continue
        used_thetas.append(theta)
        cue_path = ([p.tolist() for p in entry]
                    + [p.tolist() for p in post_pts[1:]])
        out.append(Guide(
            shot_id="", name="", feasible=True,
            category=_category_of(total_cushions), cushions=total_cushions,
            cue_path=cue_path,
            object_path=[first.tolist(), (first + n * 350.0).tolist()],
            ghost=(float(ghost[0]), float(ghost[1])),
            thickness=round(player_t, 2),
            thickness_label=_thickness_label(player_t),
            aim_offset=round(aim_offset, 3),
            tip=_tip_label(delta), tip_delta_deg=round(delta, 1),
            score=score,
        ))
        if len(out) >= max_variants:
            break
    return out


def solve_normal(cue, first, second, obstacles, seq,
                 max_variants: int = 3) -> list[Guide]:
    """하위호환 별칭: 수구가 1적구 전에 seq 쿠션을 경유, 이후 직행."""
    return solve_combo(cue, first, second, obstacles, seq, [],
                       max_variants=max_variants)


def _trace_after_impact(ghost, direction, second, min_cushions=3,
                        max_segments=6):
    """분리 후 수구를 거울 반사로 추적. 쿠션 min_cushions개 이상 맞은 뒤
    2적구에 닿으면 (경로점들, 쿠션벽 목록) 반환. 그 전에 닿으면 None."""
    pos = np.asarray(ghost, float)
    d = _unit(direction)
    pts = [pos]
    walls_hit: list[str] = []
    for _ in range(max_segments):
        # 다음 벽 교차
        cand = []
        if d[0] > 1e-9:
            cand.append((("right"), (X1 - pos[0]) / d[0]))
        elif d[0] < -1e-9:
            cand.append((("left"), (X0 - pos[0]) / d[0]))
        if d[1] > 1e-9:
            cand.append((("bottom"), (Y1 - pos[1]) / d[1]))
        elif d[1] < -1e-9:
            cand.append((("top"), (Y0 - pos[1]) / d[1]))
        if not cand:
            return None
        wall, t_wall = min(cand, key=lambda c: c[1])
        seg_end = pos + d * t_wall

        # 이 구간에서 2적구와 접촉하는가 (중심 거리 = 공 지름)
        rel = np.asarray(second, float) - pos
        proj = float(rel @ d)
        if 0 < proj < t_wall:
            perp2 = float(rel @ rel) - proj * proj
            if perp2 < BALL_D_MM * BALL_D_MM:
                back = float(np.sqrt(max(BALL_D_MM**2 - perp2, 0.0)))
                t_hit = proj - back
                if 0 < t_hit < t_wall:
                    if len(walls_hit) >= min_cushions:
                        pts.append(pos + d * t_hit)
                        return pts, walls_hit
                    return None  # 쿠션 3개 전에 맞음 → 3쿠션 아님

        pts.append(seg_end)
        walls_hit.append(wall)
        pos = seg_end
        if wall in ("left", "right"):
            d = np.array([-d[0], d[1]])
        else:
            d = np.array([d[0], -d[1]])
    return None


def _three_name(walls: list[str], first_wall_side: str) -> str:
    long_walls = ("top", "bottom")
    if walls[0] in long_walls and len(walls) >= 2 and walls[1] not in long_walls:
        return "뒤돌려치기" if first_wall_side == "far" else "앞돌려치기"
    if walls[0] not in long_walls:
        return "옆돌려치기"
    if walls[0] in long_walls and len(walls) >= 2 and walls[1] in long_walls:
        return "더블 쿠션"
    return "3쿠션"


def solve_three_cushion(cue, first, second, obstacles) -> list[Guide]:
    """3쿠션: 수구 → 1적구 → 분리 → 쿠션 3개 이상 → 2적구."""
    cue = np.asarray(cue, float)
    first = np.asarray(first, float)
    second = np.asarray(second, float)

    results: dict[tuple, Guide] = {}
    for theta in np.arange(0.0, 360.0, 2.0):
        rad = np.deg2rad(theta)
        n = np.array([np.cos(rad), np.sin(rad)])
        ghost = first - n * BALL_D_MM
        if not _inside_table(ghost):
            continue
        u = _unit(ghost - cue)
        fullness = float(u @ n)
        if not (MIN_FULLNESS <= fullness <= MAX_FULLNESS):
            continue
        if _path_blocked([cue, ghost], obstacles, skip_last_mm=BALL_R_MM):
            continue
        # 겨냥점 도달 전에 1적구를 스치면 무효 (마지막 접근만 접촉 허용)
        if _seg_dist(first, cue, ghost - u * BALL_D_MM * 1.2) < BALL_D_MM * 0.95:
            continue
        t0 = _unit(u - fullness * n)
        for delta in np.arange(-TIP_MAX_DEG, TIP_MAX_DEG + 0.1, 2.0):
            rad_d = np.deg2rad(delta)
            t_dir = np.cos(rad_d) * t0 + np.sin(rad_d) * n
            traced = _trace_after_impact(ghost, t_dir, second)
            if traced is None:
                continue
            pts, walls = traced
            key = tuple(walls[:3])
            # solve_normal과 동일한 이유로 delta 패널티를 약하게 둔다.
            score = (-abs(fullness - 0.55)
                     - 0.15 * (len(walls) - 3)
                     - 0.05 * abs(delta) / TIP_MAX_DEG)
            if key in results and results[key].score >= score:
                continue
            # 이름: 첫 쿠션이 1적구 기준 2적구 반대편이면 '뒤', 같은 편 '앞'
            first_wall = walls[0]
            if first_wall in ("top", "bottom"):
                side = "far" if (
                    (first_wall == "top") == (second[1] > first[1])
                ) else "near"
            else:
                side = "far"
            cue_path = [cue.tolist()] + [p.tolist() for p in pts]
            aim_offset = -float(u[0] * n[1] - u[1] * n[0])
            player_t = 1.0 - abs(aim_offset)
            results[key] = Guide(
                shot_id="", name=_three_name(walls, side), feasible=True,
                category="three", cushions=len(walls),
                cue_path=cue_path,
                object_path=[first.tolist(), (first + n * 350.0).tolist()],
                ghost=(float(ghost[0]), float(ghost[1])),
                thickness=round(player_t, 2),
                thickness_label=_thickness_label(player_t),
                aim_offset=round(aim_offset, 3),
                tip=_tip_label(float(delta)), tip_delta_deg=round(float(delta), 1),
                score=score,
            )
    out = sorted(results.values(), key=lambda g: -g.score)
    return out[:4]


def _finalize(g: Guide, opponent) -> None:
    """키스 등급 + 난이도 산출."""
    if g.feasible:
        pts = [np.asarray(p, float) for p in g.cue_path]
        g.kiss_level, g.kiss = _path_kiss(pts, opponent)
        _set_difficulty(g)


# 탭(카테고리)별 (공 앞 쿠션 수, 공 뒤 쿠션 수) 조합 — 사용자 정의 분류
_COMBO_PLAN: dict[str, list[tuple[int, int]]] = {
    "direct": [(0, 0)],
    "one": [(1, 0), (0, 1)],
    "two": [(2, 0), (1, 1), (0, 2)],
    "three": [(3, 0), (2, 1), (1, 2)],  # (0,3+)는 추적 방식이 별도 커버
}
_MAX_PER_CATEGORY = 8


def _combo_name(pre_n: int, post_n: int) -> str:
    total = pre_n + post_n
    if total == 0:
        return "직접치기"
    if pre_n == 0:
        return f"{total}쿠션 · 공 먼저"
    if post_n == 0:
        return f"{total}쿠션 · 쿠션 먼저"
    return f"{total}쿠션 · 쿠션{pre_n}→공→쿠션{post_n}"


def compute_guides(balls: list[dict], cue_color: str) -> list[Guide]:
    """검출된 공 배치에서 직접치기/1쿠션/2쿠션/3쿠션 가이드 계산.

    모든 카테고리에서 빨강①/② 순서 양쪽, 쿠션 4개 전부를 조합 탐색하고
    난이도가 쉬운 것부터 정렬해 반환한다.
    """
    by_color: dict[str, list] = {}
    for b in balls:
        by_color.setdefault(b["color"], []).append(
            np.array([float(b["x_mm"]), float(b["y_mm"])]))

    cue_list = by_color.get(cue_color, [])
    reds = by_color.get("red", [])
    opp_color = "yellow" if cue_color == "white" else "white"
    opponent = (by_color.get(opp_color) or [None])[0]

    if not cue_list:
        raise ValueError(f"수구({cue_color})가 검출되지 않음")
    if len(reds) < 2:
        raise ValueError(f"빨간 공이 부족함 ({len(reds)}/2)")
    cue = cue_list[0]

    labels = ["빨강①", "빨강②"]
    by_category: dict[str, list[Guide]] = {c: [] for c in _COMBO_PLAN}

    for i, (first, second) in enumerate([(reds[0], reds[1]), (reds[1], reds[0])]):
        obstacles = [second] + ([opponent] if opponent is not None else [])

        for cat, plans in _COMBO_PLAN.items():
            for pre_n, post_n in plans:
                variants = 3 if cat == "direct" else 1
                for pre_seq in _wall_seqs(pre_n):
                    for post_seq in _wall_seqs(post_n):
                        for g in solve_combo(cue, first, second, obstacles,
                                             pre_seq, post_seq,
                                             max_variants=variants):
                            g.name = (f"{_combo_name(pre_n, post_n)}"
                                      f" ({labels[i]} 먼저)")
                            _finalize(g, opponent)
                            by_category[cat].append(g)

        # 3쿠션 (0+3 이상): 반사 추적 방식 — 4쿠션 이상까지 자연 커버,
        # 뒤돌려치기/옆돌려치기 등 전통 명칭 부여
        for g in solve_three_cushion(cue, first, second, obstacles):
            g.name = f"{g.name} ({labels[i]} 먼저)"
            _finalize(g, opponent)
            by_category["three"].append(g)

    guides: list[Guide] = []
    for cat in ("direct", "one", "two", "three"):
        lst = by_category[cat]
        # 난이도 쉬운 순 → 키스 낮은 순 → 점수 높은 순
        lst.sort(key=lambda g: (g.difficulty, g.kiss_level, -g.score))
        # 사실상 같은 길(겨냥점·경로 끝이 거의 동일) 중복 제거
        kept: list[Guide] = []
        for g in lst:
            dup = False
            for k in kept:
                if (g.ghost and k.ghost
                        and abs(g.ghost[0] - k.ghost[0]) < 40
                        and abs(g.ghost[1] - k.ghost[1]) < 40
                        and g.cushions == k.cushions
                        and len(g.cue_path) == len(k.cue_path)):
                    dup = True
                    break
            if not dup:
                kept.append(g)
            if len(kept) >= _MAX_PER_CATEGORY:
                break
        for j, g in enumerate(kept):
            g.shot_id = f"{cat}_{j}"
        guides.extend(kept)

    # 아무것도 없으면 사유와 함께 불가 항목 반환
    if not guides:
        guides.append(Guide(
            shot_id="none", name="가능한 길 없음", feasible=False,
            reason="현재 배치에서는 계산 가능한 길이 없어요",
        ))
    return guides


def guide_to_dict(g: Guide) -> dict:
    return {
        "shot_id": g.shot_id,
        "name": g.name,
        "feasible": g.feasible,
        "category": g.category,
        "cushions": g.cushions,
        "reason": g.reason,
        "cue_path": g.cue_path,
        "object_path": g.object_path,
        "ghost": list(g.ghost) if g.ghost else None,
        "thickness": g.thickness,
        "thickness_label": g.thickness_label,
        "aim_offset": g.aim_offset,
        "tip": g.tip,
        "tip_delta_deg": g.tip_delta_deg,
        "kiss": g.kiss,
        "kiss_level": g.kiss_level,
        "difficulty": g.difficulty,
        "difficulty_label": g.difficulty_label,
    }
