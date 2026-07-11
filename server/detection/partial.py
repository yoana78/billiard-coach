"""쿠션 라인 기반 호모그래피 추정 — 전체(4라인)/부분(3라인) 촬영 지원.

부분 촬영(3라인) 역산 원리:
  마주보는 쿠션 라인 2개 + 가로지르는 쿠션 라인 1개가 보이면,
  보이는 모서리 2개는 교점으로 확정된다. 안 보이는 모서리 2개는
  각 라인 위 어딘가에 있으므로, 후보 위치를 탐색하며
  "그 위치로 변환했을 때 레일 다이아몬드들이 규격 간격(317.5mm)에
  맞아떨어지는 정도"를 점수화해 최적 위치를 찾는다.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from . import spec
from .cushion_lines import CushionLine, intersect, angle_between, support_extent
from .diamond_detector import _find_bright_blob
from .homography import mm_to_px, canvas_size, PX_PER_MM

# 부분 촬영 판단: 천 컨투어가 이미지 가장자리에 이만큼 닿아 있으면 부분
BORDER_TOUCH_MARGIN = 6

# 다이아몬드 점수 탐색 창 반경 (mm)
SCORE_WIN_MM = 55.0


@dataclass
class LineSolveResult:
    corners: np.ndarray       # (4,2) tl,tr,br,bl — 이미지 좌표 (프레임 밖 가능)
    partial: bool
    visible_lines: int
    score: float = 0.0        # 부분 해의 다이아몬드 점수


def cloth_touches_border(mask: np.ndarray) -> bool:
    m = BORDER_TOUCH_MARGIN
    border = np.concatenate([
        mask[:m, :].ravel(), mask[-m:, :].ravel(),
        mask[:, :m].ravel(), mask[:, -m:].ravel(),
    ])
    return bool(np.count_nonzero(border) > border.size * 0.02)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def solve_full_from_lines(lines: list[CushionLine]) -> list[LineSolveResult]:
    """4개 쿠션 라인 → 가능한 2+2 분할 3가지 전부의 교점 사각형 후보.

    어느 분할이 맞는지는 호출부(pipeline)가 다이아몬드 정합으로 판정한다.
    각도 차가 작은(마주보는 쌍일 가능성이 높은) 분할부터 정렬해 반환.
    """
    if len(lines) < 4:
        return []
    idx_pairs = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    scored = []
    for (a, b), (c, d) in idx_pairs:
        cost = angle_between(lines[a], lines[b]) + angle_between(lines[c], lines[d])
        corners = []
        ok = True
        for i in (a, b):
            for j in (c, d):
                pt = intersect(lines[i], lines[j])
                if pt is None:
                    ok = False
                    break
                corners.append(pt)
            if not ok:
                break
        if not ok:
            continue
        scored.append((cost, LineSolveResult(
            corners=_order_corners(np.array(corners)),
            partial=False, visible_lines=4,
        )))
    scored.sort(key=lambda t: t[0])
    return [r for _c, r in scored]


def _homography_to_canvas(corners: np.ndarray, cross_is_short: bool,
                          flip: bool) -> np.ndarray | None:
    """모서리 4점(c1, c2, f1, f2 순) → 탑뷰 호모그래피.

    c1,c2: 가로지르는 라인 위의 보이는 모서리 / f1,f2: 추정된 먼 모서리.
    검출 라인은 천 경계(쿠션 바깥)이므로 오버행만큼 확장된 사각형에 대응.
    cross_is_short=True 면 가로 라인이 단쿠션 쪽, 아니면 장쿠션 쪽.
    """
    ov = spec.CUSHION_OVERHANG_MM
    x0, y0 = -ov, -ov
    x1, y1 = spec.TABLE_W_MM + ov, spec.TABLE_H_MM + ov
    if cross_is_short:
        dst_mm = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
    else:
        dst_mm = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    if flip:
        dst_mm = [dst_mm[1], dst_mm[0], dst_mm[3], dst_mm[2]]
    dst = np.array([mm_to_px(p) for p in dst_mm], dtype=np.float32)
    Hm, _ = cv2.findHomography(corners.astype(np.float32), dst)
    return Hm


def _diamond_score(gray: np.ndarray, Hm: np.ndarray, scale: float = 0.5):
    """H로 축소 탑뷰를 만들어 다이아몬드 기대 위치 정합 점수 계산.

    반환: (점수, 매칭 개수). 점수는 매칭 수 + 블롭 품질 - 위치 오차 패널티.
    """
    S = np.diag([scale, scale, 1.0])
    Hs = S @ Hm
    w, h = canvas_size()
    sw, sh = int(w * scale), int(h * scale)
    top = cv2.warpPerspective(gray, Hs, (sw, sh))
    valid = cv2.warpPerspective(np.full_like(gray, 255), Hs, (sw, sh))

    ppm = PX_PER_MM * scale
    r = max(4, int(SCORE_WIN_MM * ppm))
    score, matched = 0.0, 0
    for (x_mm, y_mm, _side) in spec.diamond_points_mm():
        px, py = mm_to_px((x_mm, y_mm))
        cx, cy = px * scale, py * scale
        x0, x1 = int(cx - r), int(cx + r)
        y0, y1 = int(cy - r), int(cy + r)
        if x0 < 0 or y0 < 0 or x1 > sw or y1 > sh:
            continue
        if cv2.countNonZero(valid[y0:y1, x0:x1]) < (x1 - x0) * (y1 - y0) * 0.9:
            continue  # 시야 밖(검은 영역 포함) 창은 채점 제외
        win = top[y0:y1, x0:x1]
        if win.std() < 3:
            continue
        blob = _find_bright_blob(win, ppm=ppm)
        if blob is None:
            continue
        bx, by, q = blob
        err_mm = float(np.hypot(bx - (cx - x0), by - (cy - y0))) / ppm
        matched += 1
        score += 1.0 + q * 0.5 - err_mm * 0.01
    return score, matched


def _search_decomposition(gray, mask, la, lb, lc):
    """한 가지 분해(마주보는 la·lb + 가로지르는 lc)에 대해
    먼 모서리 2개를 다이아몬드 정합으로 탐색.

    반환: (score, matched, corners, cross_is_short, flip) 또는 None.
    """
    c1 = intersect(lc, la)
    c2 = intersect(lc, lb)
    if c1 is None or c2 is None:
        return None
    if np.linalg.norm(c1 - c2) < 30:  # 모서리 2개가 사실상 한 점이면 무효
        return None
    v = intersect(la, lb)  # 마주보는 쌍의 소실점 (화면상 평행하면 None)

    ext_a = support_extent(la, mask)
    ext_b = support_extent(lb, mask)
    if ext_a is None or ext_b is None:
        return None
    # 각 라인에서 보이는 모서리 반대쪽 끝(=시야에서 잘린 곳)이 탐색 시작점
    e1 = max(ext_a, key=lambda p: np.linalg.norm(p - c1))
    e2 = max(ext_b, key=lambda p: np.linalg.norm(p - c2))

    diag = float(np.hypot(*gray.shape))

    def far_candidates(e, c, n_samples=12):
        """모서리 c 반대쪽(시야 밖) 먼 모서리 후보들.

        먼 모서리는 e(시야 경계)와 소실점 v 사이에 있다. v가 아주 멀 수
        있으므로(거의 평행) 거리를 로그 간격으로 샘플링한다.
        """
        d_away = e - c
        n = np.linalg.norm(d_away)
        if n < 1e-6:
            return []
        d_away = d_away / n
        s_max = 6.0 * diag
        if v is not None and float((v - e) @ d_away) > 0:
            s_max = min(s_max, 0.97 * float(np.linalg.norm(v - e)))
        s_min = 0.03 * diag
        if s_max <= s_min:
            s_max = s_min * 2
        return [e + d_away * s for s in np.geomspace(s_min, s_max, n_samples)]

    best = None  # (score, matched, corners, cross_is_short, flip)
    cands1 = far_candidates(e1, c1)
    cands2 = far_candidates(e2, c2)
    for cross_is_short in (True, False):
        for flip in (False, True):
            for f1 in cands1:
                for f2 in cands2:
                    corners = np.array([c1, c2, f1, f2])
                    Hm = _homography_to_canvas(corners, cross_is_short, flip)
                    if Hm is None:
                        continue
                    score, matched = _diamond_score(gray, Hm, scale=0.5)
                    if matched < 4:
                        continue
                    if best is None or score > best[0]:
                        best = (score, matched, corners, cross_is_short, flip)

    if best is None:
        return None

    # 최적 근방 미세 탐색
    score0, matched0, corners0, cs, fl = best
    f1_0, f2_0 = corners0[2], corners0[3]
    d1 = la.d if float(la.d @ (f1_0 - c1)) > 0 else -la.d
    d2 = lb.d if float(lb.d @ (f2_0 - c2)) > 0 else -lb.d
    step = diag * 0.02
    for _ in range(2):
        improved = False
        for o1 in (-step, 0, step):
            for o2 in (-step, 0, step):
                f1 = f1_0 + d1 * o1
                f2 = f2_0 + d2 * o2
                corners = np.array([c1, c2, f1, f2])
                Hm = _homography_to_canvas(corners, cs, fl)
                if Hm is None:
                    continue
                score, matched = _diamond_score(gray, Hm, scale=0.5)
                if matched >= 4 and score > score0:
                    score0, matched0 = score, matched
                    f1_0, f2_0 = f1, f2
                    improved = True
        if not improved:
            step *= 0.4

    return (score0, matched0, np.array([c1, c2, f1_0, f2_0]), cs, fl)


def solve_partial_from_lines(gray: np.ndarray, mask: np.ndarray,
                             lines: list[CushionLine]) -> tuple | None:
    """3개 쿠션 라인 → 먼 모서리 역산.

    어느 쌍이 마주보는지 휴리스틱으로 정하지 않고, 가능한 분해 3가지를
    전부 다이아몬드 정합 점수로 평가해 최고를 채택한다.
    """
    if len(lines) < 3:
        return None
    lines = lines[:3]

    best = None  # (score, matched, corners, cs, flip)
    for a, b, cross_i in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
        r = _search_decomposition(gray, mask, lines[a], lines[b], lines[cross_i])
        if r is None:
            continue
        if best is None or r[0] > best[0]:
            best = r
        if best[1] >= 10:  # 충분히 강한 해면 조기 종료
            break

    if best is None:
        return None
    score0, _matched, corners, cs, fl = best
    Hm = _homography_to_canvas(corners, cs, fl)
    return LineSolveResult(
        corners=corners.astype(np.float32),
        partial=True, visible_lines=3, score=score0,
    ), Hm
