"""정지 이미지 → 탑뷰 자동 변환 파이프라인.

흐름:
  1) 천 색상 마스크 → 쿠션 라인 검출 (허프 변환, 가림/부분촬영에 강함)
  2) 라인 4개: 교점 4개로 호모그래피 (전체 촬영)
     라인 3개: 다이아몬드 간격 정합 탐색으로 안 보이는 모서리 역산 (부분 촬영)
     그 외: legacy 컨투어 방식 시도 후 실패 사유 반환
  3) 탑뷰 워프 → 다이아몬드 검출 → 호모그래피 재정밀화
  4) 공 검출 + 품질 지표 리포트
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import spec
from .table_detector import detect_table, cloth_mask, TableQuad
from .cushion_lines import detect_cushion_lines
from .partial import (
    solve_full_from_lines, solve_partial_from_lines,
    solve_corner_from_lines, cloth_touches_border,
)
from .diamond_detector import detect_diamonds, DiamondHit
from .ball_detector import detect_balls, draw_balls, Ball
from .homography import (
    homography_from_corners,
    refine_homography,
    warp_topview,
    playfield_corners_px,
    mm_to_px,
    project,
    PX_PER_MM,
)

# 처리 속도/허프 파라미터 안정화를 위한 입력 크기 상한
MAX_INPUT_DIM = 1600


@dataclass
class TopViewResult:
    ok: bool
    reason: str = ""
    topview: np.ndarray | None = None        # 최종 탑뷰 BGR 이미지
    H: np.ndarray | None = None              # 원본(축소) → 탑뷰 최종 호모그래피
    cloth_color: str = ""
    partial: bool = False                    # 부분 촬영에서 역산했는가
    visible_lines: int = 0
    diamonds: list[DiamondHit] = field(default_factory=list)
    diamond_err_mm: float = -1.0             # 검출 다이아몬드의 기대 위치 대비 평균 오차
    refined: bool = False
    balls: list[Ball] = field(default_factory=list)


def _downscale(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= MAX_INPUT_DIM:
        return bgr
    s = MAX_INPUT_DIM / m
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _quad_cloth_coverage(corners: np.ndarray, raw_mask: np.ndarray) -> float:
    """모서리 4점 내부에서 천 색 픽셀 비율 (이미지 내 영역 기준)."""
    poly = np.zeros(raw_mask.shape, dtype=np.uint8)
    cv2.fillPoly(poly, [corners.astype(np.int32)], 255)
    inside = int(cv2.countNonZero(poly))
    if inside == 0:
        return 0.0
    return int(cv2.countNonZero(cv2.bitwise_and(raw_mask, poly))) / inside


def image_to_topview(bgr: np.ndarray) -> TopViewResult:
    bgr = _downscale(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    mask, raw_mask, color = cloth_mask(bgr)
    img_area = bgr.shape[0] * bgr.shape[1]
    if cv2.countNonZero(mask) < img_area * 0.05:
        return TopViewResult(ok=False, reason="당구대(천)를 찾지 못함")

    lines = detect_cushion_lines(mask)
    n_lines = len(lines)
    touches = cloth_touches_border(mask)

    # 후보 호모그래피를 다이아몬드 검출 수로 검증해 최선을 채택
    def diamonds_of(Hc):
        return detect_diamonds(warp_topview(bgr, Hc))

    best = None  # (다이아몬드 수, H, hits, partial_flag)

    # 1) 전체 촬영 후보: 쿠션 라인 4개 → 분할 3가지 전부 다이아몬드로 평가
    if n_lines >= 4:
        for fr in solve_full_from_lines(lines):
            if _quad_cloth_coverage(fr.corners, raw_mask) < 0.55:
                continue
            Hc = homography_from_corners(fr.corners)
            if Hc is None:
                continue
            hits = diamonds_of(Hc)
            if best is None or len(hits) > best[0]:
                best = (len(hits), Hc, hits, False)
            if best[0] >= 10:
                break

    # 2) 부분 촬영 후보: 전체 해가 없거나 약하면 (라인 3개 이상) 역산 시도
    if n_lines >= 3 and (best is None or best[0] < 6):
        pr = solve_partial_from_lines(gray, mask, lines)
        if pr is not None:
            _res, Hc = pr
            hits = diamonds_of(Hc)
            if len(hits) >= 4 and (best is None or len(hits) > best[0]):
                best = (len(hits), Hc, hits, True)

    # 2.5) 코너 촬영 후보: 직교 쿠션 2개만 보일 때 다이아몬드 격자로 역산
    if n_lines >= 2 and (best is None or best[0] < 6):
        cr = solve_corner_from_lines(gray, mask, lines)
        if cr is not None:
            _res, Hc = cr
            hits = diamonds_of(Hc)
            if len(hits) >= 6 and (best is None or len(hits) > best[0]):
                best = (len(hits), Hc, hits, True)

    # 3) legacy 컨투어 방식 (프레임에 안 잘린 전체 촬영 한정)
    if best is None and not touches:
        quad = detect_table(bgr)
        if quad is not None:
            Hc = homography_from_corners(quad.corners)
            if Hc is not None:
                hits = diamonds_of(Hc)
                best = (len(hits), Hc, hits, False)

    if best is None:
        if n_lines <= 1:
            reason = ("당구대 쿠션이 거의 안 보여요 — "
                      "쿠션이 더 보이게 조금 더 뒤에서 찍어주세요")
        else:
            reason = ("당구대 다이아몬드가 잘 안 보여요 — "
                      "빛반사가 적은 각도에서 다시 찍어주세요")
        return TopViewResult(ok=False, reason=reason, visible_lines=n_lines)

    _n, H0, hits, partial_flag = best
    # 최종 검증: 다이아몬드가 최소한으로도 안 맞으면 엉터리 변환이므로 거부
    if len(hits) < 4:
        return TopViewResult(
            ok=False, visible_lines=n_lines,
            reason="다이아몬드 인식 부족 — 레일의 흰 점들이 잘 보이게 찍어주세요",
        )

    # 다이아몬드 대응점으로 재정밀화 — 매 반복 결과를 평가해
    # (검출 수 증가 또는 오차 감소) 좋아질 때만 채택한다.
    def _mean_err(hs):
        if not hs:
            return -1.0
        errs = [
            np.hypot(h.found_px[0] - mm_to_px(h.expected_mm)[0],
                     h.found_px[1] - mm_to_px(h.expected_mm)[1]) / PX_PER_MM
            for h in hs
        ]
        return float(np.mean(errs))

    H = H0
    hits = detect_diamonds(warp_topview(bgr, H))
    err = _mean_err(hits)
    refined = False
    for _pass in range(3):
        if len(hits) < 4:
            break
        H_inv = np.linalg.inv(H)
        found_px = np.array([h.found_px for h in hits], dtype=np.float32)
        src_in_image = project(H_inv, found_px)
        dst_expected = np.array(
            [mm_to_px(h.expected_mm) for h in hits], dtype=np.float32
        )
        H_ref = refine_homography(src_in_image, dst_expected)
        if H_ref is None:
            break
        hits_ref = detect_diamonds(warp_topview(bgr, H_ref))
        err_ref = _mean_err(hits_ref)
        better = (len(hits_ref) > len(hits)
                  or (len(hits_ref) == len(hits) and 0 <= err_ref < err))
        if not better:
            break
        H, hits, err = H_ref, hits_ref, err_ref
        refined = True

    topview = warp_topview(bgr, H)
    final_hits = hits
    err_mm = err

    balls = detect_balls(bgr, H, cloth_raw=raw_mask)

    return TopViewResult(
        ok=True,
        topview=topview,
        H=H,
        cloth_color=color,
        partial=partial_flag,
        visible_lines=n_lines,
        diamonds=final_hits,
        diamond_err_mm=err_mm,
        refined=refined,
        balls=balls,
    )


def draw_debug(result: TopViewResult) -> np.ndarray | None:
    """탑뷰 위에 규격 그리드/다이아몬드/공 검출 결과를 그린 디버그 이미지."""
    if not result.ok or result.topview is None:
        return None
    img = result.topview.copy()
    pf = playfield_corners_px().astype(np.int32)
    cv2.polylines(img, [pf], True, (0, 255, 255), 2)
    for (x, y, _side) in spec.diamond_points_mm():
        px, py = mm_to_px((x, y))
        cv2.drawMarker(img, (int(px), int(py)), (0, 255, 255),
                       cv2.MARKER_CROSS, 12, 1)
    for hit in result.diamonds:
        fx, fy = hit.found_px
        cv2.circle(img, (int(fx), int(fy)), 8, (0, 255, 0), 2)
    draw_balls(img, result.balls)
    txt = (f"diamonds: {len(result.diamonds)}/{len(spec.diamond_points_mm())}  "
           f"err: {result.diamond_err_mm:.1f}mm  refined: {result.refined}  "
           f"balls: {len(result.balls)}/4  "
           f"{'PARTIAL' if result.partial else 'FULL'}({result.visible_lines})")
    cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img
