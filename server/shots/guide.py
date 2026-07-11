"""4구/3구 샷 패턴 라이브러리 — 룰 바탕 가이드 계산.

물리 모델 (이상화):
  - 공-공: 1적구는 중심선(n) 방향 출발, 수구는 당점에 따라
    접선(t0)에서 ±TIP_MAX_DEG 안에서 분리.
  - 두껍게(정면비율 0.6+) 맞추면 밀어치기(전진)/끌어치기(후진) 모드 가능.
    끌기는 1적구 '직접' 타격 + 근거리일 때만 (백스핀 유지 조건).
  - 공-쿠션: 거울 반사 근사 (평균적인 순방향 회전 기준). 회전 방향은
    경로의 꺾임 방향에서 역산해 당점 좌/우로 표시한다.
  - 공 중심의 이동 가능 영역은 쿠션에서 공 반지름만큼 안쪽 사각형.

테이블: 중대(medium)/대대(large) 규격을 TableDims 로 매개변수화.
게임: 4구(four) — 빨강 2개를 순서대로 / 3구(three) — 총 쿠션 3회 이상.
키스·난이도·강도는 5등급, 두께/당점은 상대 표기 (평균 회전력/힘 기준).
"""
from dataclasses import dataclass, field

import numpy as np

MIN_FULLNESS = 0.15
MAX_FULLNESS = 0.97   # 정면 가까운 두께도 허용 (밀어/끌어치기 모드가 사용)
TIP_MAX_DEG = 40.0
# 두껍게 맞추고 밀어치기(전진)/끌어치기(후진) 모드의 허용 원뿔/최소 두께
FOLLOW_CONE_DEG = 25.0
FOLLOW_MIN_FULLNESS = 0.6
DRAW_MAX_DIST_MM = 1500.0  # 끌어치기가 유지되는 최대 직접타격 거리

# 키스 위험 5등급 거리 기준 (경로와 방해공 중심 거리, mm)
KISS_LEVELS = [(400.0, 1, "없음"), (250.0, 2, "거의없음"),
               (150.0, 3, "보통"), (80.0, 4, "높음"), (0.0, 5, "매우높음")]
DIFFICULTY_LABELS = {1: "매우쉬움", 2: "쉬움", 3: "보통", 4: "어려움", 5: "매우어려움"}
POWER_LABELS = {1: "매우 약하게", 2: "약하게", 3: "보통", 4: "강하게", 5: "매우 강하게"}

_WALLS = ("top", "bottom", "left", "right")


@dataclass(frozen=True)
class TableDims:
    """테이블 규격 (mm). 좌표계: x=장쿠션 방향 0~w, y=단쿠션 방향 0~h."""
    w: float
    h: float
    ball_d: float

    @property
    def ball_r(self) -> float:
        return self.ball_d / 2

    # 공 '중심'이 움직일 수 있는 사각형
    @property
    def x0(self) -> float:
        return self.ball_r

    @property
    def x1(self) -> float:
        return self.w - self.ball_r

    @property
    def y0(self) -> float:
        return self.ball_r

    @property
    def y1(self) -> float:
        return self.h - self.ball_r


TABLES: dict[str, TableDims] = {
    "medium": TableDims(w=2540.0, h=1270.0, ball_d=65.5),
    "large": TableDims(w=2844.0, h=1422.0, ball_d=61.5),
}

# 하위호환 (중대 기준 상수 — 기존 테스트/도구용)
_M = TABLES["medium"]
BALL_D_MM = _M.ball_d
BALL_R_MM = _M.ball_r
TABLE_W_MM = _M.w
TABLE_H_MM = _M.h
X0, X1, Y0, Y1 = _M.x0, _M.x1, _M.y0, _M.y1


@dataclass
class Guide:
    shot_id: str
    name: str
    feasible: bool
    category: str = "direct"             # 'direct' | 'one' | 'two' | 'three'
    cushions: int = 0                    # 경유 쿠션 수 (전체)
    reason: str = ""
    cue_path: list = field(default_factory=list)     # 수구 경로 꼭짓점들 (mm)
    object_path: list = field(default_factory=list)  # 1적구 예상 진행 (mm)
    ghost: tuple | None = None           # 겨냥점(고스트볼 중심)
    thickness: float = 0.0               # 당구 관례 두께 (겹침 비율 0~1)
    thickness_label: str = ""
    # 타점(두께) 시각화용: 1적구 중심 대비 겨냥 가로 오프셋 (공 지름 단위,
    # 진행 방향 기준 +면 오른쪽). |aim_offset| = sin(컷각), 0=정면 겨냥.
    aim_offset: float = 0.0
    tip: str = "중단"                    # 당점 요약 라벨
    tip_delta_deg: float = 0.0           # 상하 보정각 (+위/-아래)
    tip_x: int = 0                       # 당점 가로 단계 (-3 좌 ~ +3 우)
    tip_y: int = 0                       # 당점 세로 단계 (-3 하 ~ +3 상)
    power: int = 3                       # 강도 1(매우약)~5(매우강), 평균 힘 기준
    power_label: str = "보통"
    kiss: str = "없음"                   # 없음/거의없음/보통/높음/매우높음
    kiss_level: int = 1
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


def _tip_steps_y(delta_deg: float, mode: str) -> int:
    """상하 당점 단계 (-3~+3). 밀기/끌기 모드는 ±3 (최대 당점)."""
    if mode == "follow":
        return 3
    if mode == "draw":
        return -3
    return int(np.clip(round(delta_deg / 13.4), -3, 3))


def _tip_label_2d(tip_x: int, tip_y: int) -> str:
    v = "상단" if tip_y > 0 else ("하단" if tip_y < 0 else "중단")
    h = "" if tip_x == 0 else (f" 우{abs(tip_x)}" if tip_x > 0 else f" 좌{abs(tip_x)}")
    lv = f"{abs(tip_y)}" if tip_y != 0 else ""
    return f"{v}{lv}{h}"


def _inside_table(dims: TableDims, p) -> bool:
    return dims.x0 <= p[0] <= dims.x1 and dims.y0 <= p[1] <= dims.y1


def _mirror(dims: TableDims, p, wall):
    x, y = float(p[0]), float(p[1])
    if wall == "left":
        return np.array([2 * dims.x0 - x, y])
    if wall == "right":
        return np.array([2 * dims.x1 - x, y])
    if wall == "top":
        return np.array([x, 2 * dims.y0 - y])
    return np.array([x, 2 * dims.y1 - y])


def _wall_cross(dims: TableDims, p, q, wall):
    """선분 p→q가 벽 직선과 만나는 점 (벽 범위 안일 때만)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    if wall in ("left", "right"):
        wx = dims.x0 if wall == "left" else dims.x1
        if abs(d[0]) < 1e-9:
            return None
        t = (wx - p[0]) / d[0]
        if not (1e-6 < t < 1 - 1e-6):
            return None
        pt = p + t * d
        if dims.y0 - 1 <= pt[1] <= dims.y1 + 1:
            return pt
        return None
    wy = dims.y0 if wall == "top" else dims.y1
    if abs(d[1]) < 1e-9:
        return None
    t = (wy - p[1]) / d[1]
    if not (1e-6 < t < 1 - 1e-6):
        return None
    pt = p + t * d
    if dims.x0 - 1 <= pt[0] <= dims.x1 + 1:
        return pt
    return None


def _reflect_path(dims: TableDims, start, target, seq):
    """start → (seq 순서 쿠션 반사) → target 폴리라인. 불가면 None."""
    virt = np.asarray(target, float)
    for w in reversed(seq):
        virt = _mirror(dims, virt, w)
    pts = [np.asarray(start, float)]
    cur_virt = virt
    for w in seq:
        b = _wall_cross(dims, pts[-1], cur_virt, w)
        if b is None:
            return None
        pts.append(b)
        cur_virt = _mirror(dims, cur_virt, w)
    pts.append(np.asarray(target, float))
    return pts


def _path_blocked(dims: TableDims, pts, obstacles, skip_last_mm=0.0) -> bool:
    """폴리라인이 장애공에 막히는가."""
    for a, b in zip(pts, pts[1:]):
        b_eff = b
        if skip_last_mm > 0 and np.array_equal(b, pts[-1]):
            d = _unit(b - a)
            b_eff = b - d * skip_last_mm
        for o in obstacles:
            if _seg_dist(o, a, b_eff) < dims.ball_d * 0.95:
                return True
    return False


def _path_kiss(dims: TableDims, pts, opponent) -> tuple[int, str]:
    """(등급 1~5, 라벨)."""
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


def _side_spin(pts: list, dims: TableDims, total_cushions: int) -> int:
    """당점 좌/우 단계 (-3~+3) — 첫 쿠션에서의 꺾임 방향을 순방향
    회전으로 만들기 위한 평균적인 사이드 스핀 근사."""
    if total_cushions == 0 or len(pts) < 3:
        return 0
    # 첫 번째 '벽 위' 꼭짓점 찾기 (양끝 제외)
    for i in range(1, len(pts) - 1):
        p = pts[i]
        on_wall = (abs(p[0] - dims.x0) < 1 or abs(p[0] - dims.x1) < 1
                   or abs(p[1] - dims.y0) < 1 or abs(p[1] - dims.y1) < 1)
        if not on_wall:
            continue
        d_in = _unit(np.asarray(pts[i]) - np.asarray(pts[i - 1]))
        d_out = _unit(np.asarray(pts[i + 1]) - np.asarray(pts[i]))
        cross = float(d_in[0] * d_out[1] - d_in[1] * d_out[0])
        if abs(cross) < 0.05:
            return 0
        sign = 1 if cross > 0 else -1  # 화면 좌표(y아래) 기준 우회전=+
        mag = 1 if total_cushions == 1 else 2
        return sign * mag
    return 0


def _power_of(g: "Guide") -> int:
    """강도 1~5 — 평균적인 힘 기준: 경로 길이 + 쿠션 + 끌기 가산."""
    length = _path_length(g.cue_path) + 350.0 * g.cushions
    if g.tip_y <= -2:
        length += 600.0  # 끌기는 여분의 힘 필요
    if length < 900:
        return 1
    if length < 1600:
        return 2
    if length < 2600:
        return 3
    if length < 3800:
        return 4
    return 5


def _set_difficulty(g: "Guide") -> None:
    """난이도 1~5: 당점 요구량 + 얇은 두께 + 쿠션 수 + 경로 길이."""
    d = 0.0
    d += (abs(g.tip_y) + abs(g.tip_x)) / 6.0 * 1.4     # 당점을 많이 써야 함
    d += max(0.0, 0.30 - g.thickness) * 4.0            # 얇을수록 어려움
    d += g.cushions * 0.55
    d += max(0.0, _path_length(g.cue_path) - 1800.0) / 3000.0
    level = 1 + min(4, int(d / 0.8))
    g.difficulty = level
    g.difficulty_label = DIFFICULTY_LABELS[level]


def _finalize(dims: TableDims, g: Guide, opponent) -> None:
    """사이드스핀·당점 라벨·키스·난이도·강도 산출."""
    if not g.feasible:
        return
    pts = [np.asarray(p, float) for p in g.cue_path]
    g.tip_x = _side_spin(pts, dims, g.cushions)
    g.tip = _tip_label_2d(g.tip_x, g.tip_y)
    g.kiss_level, g.kiss = _path_kiss(dims, pts, opponent)
    _set_difficulty(g)
    g.power = _power_of(g)
    g.power_label = POWER_LABELS[g.power]


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
                max_variants: int = 1,
                dims: TableDims = _M) -> list[Guide]:
    """일반화된 샷 해법:
      수구 → (pre_seq 쿠션들) → 1적구 → 분리 → (post_seq 쿠션들) → 2적구

    거울 반사 원리를 앞뒤 양쪽에 적용해 닫힌 해로 계산한다.
    방향이 충분히 다른(35° 이상) 상위 해를 최대 max_variants개 반환.
    """
    cue = np.asarray(cue, float)
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    D = dims.ball_d
    if float(np.linalg.norm(second - first)) < D * 1.5:
        return []
    others = [o for o in obstacles if not np.allclose(o, second)]

    cands = []
    for theta in np.arange(0.0, 360.0, 2.0):
        rad = np.deg2rad(theta)
        n = np.array([np.cos(rad), np.sin(rad)])   # 1적구 출발 방향
        ghost = first - n * D
        if not _inside_table(dims, ghost):
            continue
        entry = _reflect_path(dims, cue, ghost, pre_seq)
        if entry is None:
            continue
        u = _unit(entry[-1] - entry[-2])           # 도착(충돌) 방향
        fullness = float(u @ n)
        if not (MIN_FULLNESS <= fullness <= MAX_FULLNESS):
            continue
        t0 = _unit(u - fullness * n)
        post = _reflect_path(dims, ghost, second, post_seq)
        if post is None:
            continue
        v0 = _unit(post[1] - post[0])              # 분리 직후 방향
        # 분리 방향 달성 방법: tip(접선±40°) / follow(밀기) / draw(끌기)
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
        if pre_seq and (mode == "draw" or (mode == "tip" and delta < -5.0)):
            continue
        if mode == "draw" and float(np.linalg.norm(ghost - cue)) > DRAW_MAX_DIST_MM:
            continue
        if _path_blocked(dims, entry, obstacles, skip_last_mm=dims.ball_r):
            continue
        # 수구가 1적구를 '미리' 관통하면 안 됨
        trim = entry[-1] - u * D * 1.2
        entry_body = entry[:-1] + [trim]
        if any(_seg_dist(first, a, b) < D * 0.95
               for a, b in zip(entry_body, entry_body[1:])):
            continue
        # 분리 후 경로 검증
        v_last = _unit(post[-1] - post[-2])
        contact = second - v_last * D
        post_pts = post[:-1] + [contact]
        if any(_seg_dist(second, a, b) < D
               for a, b in zip(post_pts[:-2], post_pts[1:-1])):
            continue
        if _path_blocked(dims, post_pts, others, skip_last_mm=dims.ball_r):
            continue
        aim_offset = -float(u[0] * n[1] - u[1] * n[0])  # -cross(u,n) = ∓sinθ
        player_t = 1.0 - abs(aim_offset)
        if mode == "tip":
            score = (-abs(player_t - 0.55)
                     - 0.1 * (len(pre_seq) + len(post_seq))
                     - 0.05 * abs(delta) / TIP_MAX_DEG)
        else:
            score = 0.1 - 0.1 * (len(pre_seq) + len(post_seq))
        cands.append((score, theta, n, ghost, entry, post_pts,
                      delta, mode, player_t, aim_offset))

    cands.sort(key=lambda c: -c[0])
    out: list[Guide] = []
    used_thetas: list[float] = []
    total_cushions = len(pre_seq) + len(post_seq)
    for (score, theta, n, ghost, entry, post_pts,
         delta, mode, player_t, aim_offset) in cands:
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
            tip_delta_deg=round(delta, 1),
            tip_y=_tip_steps_y(delta, mode),
            score=score,
        ))
        if len(out) >= max_variants:
            break
    return out


def solve_normal(cue, first, second, obstacles, seq,
                 max_variants: int = 3) -> list[Guide]:
    """하위호환 별칭 (중대): 수구가 1적구 전에 seq 쿠션 경유, 이후 직행."""
    return solve_combo(cue, first, second, obstacles, seq, [],
                       max_variants=max_variants, dims=_M)


def _trace_after_impact(ghost, direction, second, min_cushions=3,
                        max_segments=6, dims: TableDims = _M):
    """분리 후 수구를 거울 반사로 추적. 쿠션 min_cushions개 이상 맞은 뒤
    2적구에 닿으면 (경로점들, 쿠션벽 목록) 반환. 그 전에 닿으면 None."""
    pos = np.asarray(ghost, float)
    d = _unit(direction)
    D = dims.ball_d
    pts = [pos]
    walls_hit: list[str] = []
    for _ in range(max_segments):
        cand = []
        if d[0] > 1e-9:
            cand.append((("right"), (dims.x1 - pos[0]) / d[0]))
        elif d[0] < -1e-9:
            cand.append((("left"), (dims.x0 - pos[0]) / d[0]))
        if d[1] > 1e-9:
            cand.append((("bottom"), (dims.y1 - pos[1]) / d[1]))
        elif d[1] < -1e-9:
            cand.append((("top"), (dims.y0 - pos[1]) / d[1]))
        if not cand:
            return None
        wall, t_wall = min(cand, key=lambda c: c[1])
        seg_end = pos + d * t_wall

        rel = np.asarray(second, float) - pos
        proj = float(rel @ d)
        if 0 < proj < t_wall:
            perp2 = float(rel @ rel) - proj * proj
            if perp2 < D * D:
                back = float(np.sqrt(max(D**2 - perp2, 0.0)))
                t_hit = proj - back
                if 0 < t_hit < t_wall:
                    if len(walls_hit) >= min_cushions:
                        pts.append(pos + d * t_hit)
                        return pts, walls_hit
                    return None  # 규정 쿠션 수 전에 맞음

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


def solve_three_cushion(cue, first, second, obstacles,
                        dims: TableDims = _M) -> list[Guide]:
    """공 먼저 → 쿠션 3개 이상 → 2적구 (추적 방식, 4쿠션+ 자연 커버)."""
    cue = np.asarray(cue, float)
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    D = dims.ball_d

    results: dict[tuple, Guide] = {}
    for theta in np.arange(0.0, 360.0, 2.0):
        rad = np.deg2rad(theta)
        n = np.array([np.cos(rad), np.sin(rad)])
        ghost = first - n * D
        if not _inside_table(dims, ghost):
            continue
        u = _unit(ghost - cue)
        fullness = float(u @ n)
        if not (MIN_FULLNESS <= fullness <= MAX_FULLNESS):
            continue
        if _path_blocked(dims, [cue, ghost], obstacles, skip_last_mm=dims.ball_r):
            continue
        if _seg_dist(first, cue, ghost - u * D * 1.2) < D * 0.95:
            continue
        t0 = _unit(u - fullness * n)
        for delta in np.arange(-TIP_MAX_DEG, TIP_MAX_DEG + 0.1, 2.0):
            rad_d = np.deg2rad(delta)
            t_dir = np.cos(rad_d) * t0 + np.sin(rad_d) * n
            traced = _trace_after_impact(ghost, t_dir, second, dims=dims)
            if traced is None:
                continue
            pts, walls = traced
            key = tuple(walls[:3])
            score = (-abs(delta) / TIP_MAX_DEG
                     - abs(fullness - 0.55)
                     - 0.15 * (len(walls) - 3))
            if key in results and results[key].score >= score:
                continue
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
                tip_delta_deg=round(float(delta), 1),
                tip_y=_tip_steps_y(float(delta), "tip"),
                score=score,
            )
    out = sorted(results.values(), key=lambda g: -g.score)
    return out[:4]


# ── 게임별 탭 구성 ──────────────────────────────────────────────
# 4구: 탭 = 수구가 2적구 전까지 맞는 쿠션 '총수'
_PLAN_FOUR: dict[str, list[tuple[int, int]]] = {
    "direct": [(0, 0)],
    "one": [(1, 0), (0, 1)],
    "two": [(2, 0), (1, 1), (0, 2)],
    "three": [(3, 0), (2, 1), (1, 2)],  # (0,3+)는 추적 방식이 별도 커버
}
# 3구: 총 쿠션 3회 이상 필수. 탭 = 1적구 '이전' 쿠션 수
_PLAN_THREE: dict[str, list[tuple[int, int]]] = {
    "direct": [],            # (0,3+) — 추적 방식
    "one": [(1, 2)],
    "two": [(2, 1)],
    "three": [(3, 0)],
}
_MAX_PER_CATEGORY = 8


def _combo_name(game: str, pre_n: int, post_n: int) -> str:
    total = pre_n + post_n
    if game == "three":
        if pre_n == 0:
            return "직접타격 (공 먼저)"
        return f"선({pre_n})쿠션 · 공 · 후({post_n})쿠션"
    if total == 0:
        return "직접치기"
    if pre_n == 0:
        return f"{total}쿠션 · 공 먼저"
    if post_n == 0:
        return f"{total}쿠션 · 쿠션 먼저"
    return f"{total}쿠션 · 쿠션{pre_n}→공→쿠션{post_n}"


def compute_guides(balls: list[dict], cue_color: str,
                   game: str = "four", table: str = "medium") -> list[Guide]:
    """검출된 공 배치에서 가이드 계산.

    game='four': 빨강 2개 (순서 양쪽) / game='three': 상대공+빨강 (순서 양쪽),
    총 쿠션 3회 이상. 각 탭 안에서 난이도 쉬운 순으로 정렬.
    """
    dims = TABLES.get(table, _M)
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
    cue = cue_list[0]

    # 목표공 쌍 구성
    if game == "three":
        if not reds:
            raise ValueError("빨간 공이 검출되지 않음")
        if opponent is None:
            raise ValueError(f"상대공({opp_color})이 검출되지 않음")
        pairs = [(reds[0], opponent, "빨강 먼저"),
                 (opponent, reds[0], "상대공 먼저")]
        plan = _PLAN_THREE
    else:
        if len(reds) < 2:
            raise ValueError(f"빨간 공이 부족함 ({len(reds)}/2)")
        pairs = [(reds[0], reds[1], "빨강① 먼저"),
                 (reds[1], reds[0], "빨강② 먼저")]
        plan = _PLAN_FOUR

    by_category: dict[str, list[Guide]] = {c: [] for c in
                                           ("direct", "one", "two", "three")}

    for first, second, label in pairs:
        obstacles = [second] + (
            [opponent] if game == "four" and opponent is not None else [])

        for cat, plans in plan.items():
            for pre_n, post_n in plans:
                variants = 3 if (game == "four" and cat == "direct") else 1
                for pre_seq in _wall_seqs(pre_n):
                    for post_seq in _wall_seqs(post_n):
                        for g in solve_combo(cue, first, second, obstacles,
                                             pre_seq, post_seq,
                                             max_variants=variants, dims=dims):
                            if game == "three":
                                g.category = cat  # 3구 탭은 선쿠션 수 기준
                            g.name = (f"{_combo_name(game, pre_n, post_n)}"
                                      f" ({label})")
                            _finalize(dims, g, opponent if game == "four" else None)
                            by_category[cat].append(g)

        # (0, 3+) — 추적 방식 (4구는 '3쿠션' 탭, 3구는 '직접타격' 탭)
        trace_cat = "direct" if game == "three" else "three"
        for g in solve_three_cushion(cue, first, second, obstacles, dims=dims):
            g.category = trace_cat
            g.name = f"{g.name} ({label})"
            _finalize(dims, g, opponent if game == "four" else None)
            by_category[trace_cat].append(g)

    guides: list[Guide] = []
    for cat in ("direct", "one", "two", "three"):
        lst = by_category[cat]
        lst.sort(key=lambda g: (g.difficulty, g.kiss_level, -g.score))
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
        "tip_x": g.tip_x,
        "tip_y": g.tip_y,
        "power": g.power,
        "power_label": g.power_label,
        "kiss": g.kiss,
        "kiss_level": g.kiss_level,
        "difficulty": g.difficulty,
        "difficulty_label": g.difficulty_label,
    }
