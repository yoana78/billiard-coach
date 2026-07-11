"""천(경기면) 마스크 경계에서 쿠션 라인(직선) 검출.

컨투어 4각형 근사 대신 허프 변환으로 직선을 찾는다:
  - 손/큐/공이 경계를 가려도 남은 구간으로 직선을 복원할 수 있음
  - 이미지 프레임에 잘린 경계(부분 촬영)는 쿠션 라인에서 제외됨
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CushionLine:
    p: np.ndarray        # 라인 위의 한 점
    d: np.ndarray        # 단위 방향 벡터
    support: float       # 지지 세그먼트 총 길이 (px)

    def homog(self) -> np.ndarray:
        """동차 표현 (a, b, c): ax + by + c = 0."""
        n = np.array([-self.d[1], self.d[0]])
        return np.array([n[0], n[1], -float(n @ self.p)])


def intersect(l1: CushionLine, l2: CushionLine) -> np.ndarray | None:
    """두 라인의 교점. 화면상 거의 평행하면 None."""
    h = np.cross(l1.homog(), l2.homog())
    if abs(h[2]) < 1e-9:
        return None
    return h[:2] / h[2]


def angle_between(l1: CushionLine, l2: CushionLine) -> float:
    """두 라인 방향 사이 예각 (도)."""
    c = abs(float(l1.d @ l2.d))
    return float(np.rad2deg(np.arccos(np.clip(c, 0, 1))))


def _fit_line(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """가중 PCA 직선 피팅 → (중심점, 단위방향)."""
    wsum = weights.sum()
    mean = (points * weights[:, None]).sum(axis=0) / wsum
    centered = points - mean
    cov = (centered * weights[:, None]).T @ centered / wsum
    eigvals, eigvecs = np.linalg.eigh(cov)
    d = eigvecs[:, np.argmax(eigvals)]
    return mean, d / np.linalg.norm(d)


def detect_cushion_lines(mask: np.ndarray, max_lines: int = 4,
                         border_margin: int = 8) -> list[CushionLine]:
    """천 마스크의 경계에서 쿠션 라인들을 검출 (지지 길이 내림차순)."""
    h, w = mask.shape
    diag = float(np.hypot(h, w))

    grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    # 이미지 프레임에 잘린 경계는 쿠션이 아니므로 제거
    grad[:border_margin, :] = 0
    grad[-border_margin:, :] = 0
    grad[:, :border_margin] = 0
    grad[:, -border_margin:] = 0

    segs = cv2.HoughLinesP(
        grad, 1, np.pi / 360,
        threshold=60,
        minLineLength=int(diag * 0.08),
        maxLineGap=int(diag * 0.02),
    )
    if segs is None:
        return []

    # (theta, rho) 기준으로 세그먼트를 클러스터링
    clusters: list[dict] = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        p1 = np.array([x1, y1], dtype=float)
        p2 = np.array([x2, y2], dtype=float)
        length = float(np.linalg.norm(p2 - p1))
        if length < 1:
            continue
        d = (p2 - p1) / length
        theta = np.arctan2(d[1], d[0]) % np.pi          # 방향 (0~π)
        n = np.array([-d[1], d[0]])
        rho = float(n @ p1)                              # 원점 거리 (부호 포함)
        # n 부호 정규화: 지배 성분이 양수가 되도록 (수평/수직 근처에서 안정)
        dom = 0 if abs(n[0]) >= abs(n[1]) else 1
        if n[dom] < 0:
            n, rho = -n, -rho

        placed = False
        for c in clusters:
            dtheta = abs(theta - c["theta"])
            dtheta = min(dtheta, np.pi - dtheta)
            if dtheta < np.deg2rad(4) and abs(rho - c["rho"]) < diag * 0.02:
                c["points"].append((p1, length))
                c["points"].append((p2, length))
                c["support"] += length
                # 지지 가중 평균으로 대표값 갱신
                t = length / c["support"]
                c["rho"] = c["rho"] * (1 - t) + rho * t
                placed = True
                break
        if not placed:
            clusters.append({
                "theta": theta, "rho": rho, "support": length,
                "points": [(p1, length), (p2, length)],
            })

    lines = []
    for c in clusters:
        if c["support"] < diag * 0.08:
            continue
        pts = np.array([p for p, _l in c["points"]])
        wts = np.array([l for _p, l in c["points"]])
        p, d = _fit_line(pts, wts)
        lines.append(CushionLine(p=p, d=d, support=c["support"]))

    lines.sort(key=lambda l: -l.support)

    # 최종 안전망: 같은 쿠션에서 나온 이중 에지(안/바깥, 그림자 등) 병합.
    # 실제 당구대는 쿠션이 천으로 덮여 있어 거의 평행한 라인이 짝으로 잡힌다.
    merged: list[CushionLine] = []
    for ln in lines:  # support 내림차순이므로 강한 라인이 대표가 됨
        dup = None
        for m in merged:
            n_m = np.array([-m.d[1], m.d[0]])
            if (angle_between(ln, m) < 8.0
                    and abs(float((ln.p - m.p) @ n_m)) < diag * 0.12):
                dup = m
                break
        if dup is not None:
            dup.support += ln.support
        else:
            merged.append(ln)

    # 진짜 쿠션 경계 필터: 한쪽에만 천이 있는 라인만 통과
    merged = [l for l in merged if side_consistency(l, mask) >= 0.75]

    # 최강 라인 대비 지지도가 너무 약한 라인은 잡음으로 간주
    if merged:
        top = merged[0].support
        merged = [l for l in merged if l.support >= top * 0.15]

    return merged[:max_lines]


def side_consistency(line: CushionLine, mask: np.ndarray,
                     delta: int = 7, n_samples: int = 120) -> float:
    """진짜 쿠션 경계인지 판정: 라인을 따라 '한쪽은 천, 반대쪽은 천 아님'
    패턴이 얼마나 일관적인가 (0~1). 사람/큐 윤곽 라인은 낮게 나온다."""
    h, w = mask.shape
    n = np.array([-line.d[1], line.d[0]])
    # 이미지 안에서 라인 위 샘플점 생성
    ts = np.linspace(-float(np.hypot(h, w)), float(np.hypot(h, w)), n_samples * 4)
    pts = line.p[None, :] + ts[:, None] * line.d[None, :]
    ok = ((pts[:, 0] >= delta) & (pts[:, 0] < w - delta)
          & (pts[:, 1] >= delta) & (pts[:, 1] < h - delta))
    pts = pts[ok]
    if len(pts) < 10:
        return 0.0
    a_pts = (pts + n * delta).astype(int)
    b_pts = (pts - n * delta).astype(int)
    a = mask[a_pts[:, 1], a_pts[:, 0]] > 0
    b = mask[b_pts[:, 1], b_pts[:, 0]] > 0
    edge = a != b  # 경계에 걸친 샘플만
    n_edge = int(edge.sum())
    if n_edge < 10:
        return 0.0
    pos = int((a & edge).sum())
    return max(pos, n_edge - pos) / n_edge


def support_extent(line: CushionLine, mask: np.ndarray,
                   band_px: int = 6) -> tuple[np.ndarray, np.ndarray] | None:
    """마스크 경계가 라인을 지지하는 구간의 양 끝점."""
    grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(grad)
    if len(xs) == 0:
        return None
    pts = np.stack([xs, ys], axis=1).astype(float)
    n = np.array([-line.d[1], line.d[0]])
    dist = np.abs((pts - line.p) @ n)
    on_line = pts[dist < band_px]
    if len(on_line) < 2:
        return None
    t = (on_line - line.p) @ line.d
    return (line.p + line.d * t.min(), line.p + line.d * t.max())
