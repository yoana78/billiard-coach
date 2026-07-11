"""시각 확인용: 합성 사진 → 탑뷰 → 샷 가이드 오버레이 저장."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.pipeline import image_to_topview
from detection.homography import mm_to_px
from shots.guide import compute_guides
from tools.make_synthetic import make_photo


def draw_guide(img, g, cue_color):
    color = (255, 255, 255) if cue_color == "white" else (60, 220, 240)
    pts = [tuple(map(int, mm_to_px(p))) for p in g.cue_path]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, 3, cv2.LINE_AA)
    if g.ghost:
        gp = tuple(map(int, mm_to_px(g.ghost)))
        cv2.circle(img, gp, 16, (200, 200, 200), 2, cv2.LINE_AA)
    op = [tuple(map(int, mm_to_px(p))) for p in g.object_path]
    if len(op) >= 2:
        cv2.arrowedLine(img, op[0], op[1], (80, 80, 255), 3, cv2.LINE_AA,
                        tipLength=0.15)
    cv2.putText(img, f"{g.name} | {g.thickness_label} | kiss:{g.kiss}",
                (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    photo, _, _ = make_photo(seed=seed, tilt=0.2)
    result = image_to_topview(photo)
    assert result.ok
    balls = [
        {"color": b.color, "x_mm": b.pos_mm[0], "y_mm": b.pos_mm[1]}
        for b in result.balls
    ]
    guides = compute_guides(balls, "white")
    out_dir = Path(__file__).resolve().parents[1] / "testdata"
    for g in guides:
        img = result.topview.copy()
        if g.feasible:
            draw_guide(img, g, "white")
        else:
            cv2.putText(img, f"{g.name}: infeasible ({g.reason})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
        p = out_dir / f"guide_{seed}_{g.shot_id}.png"
        cv2.imwrite(str(p), img)
        print("saved", p, "| feasible:", g.feasible)
