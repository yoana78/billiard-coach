"""업로드 사진 일괄 진단: 원본+마스크+라인 패널 이미지 생성."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.pipeline import _downscale
from detection.table_detector import cloth_mask
from detection.cushion_lines import detect_cushion_lines

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]


def panel(path: Path) -> Path:
    bgr = _downscale(cv2.imread(str(path)))
    mask, raw, color = cloth_mask(bgr)
    lines = detect_cushion_lines(mask)

    vis_mask = bgr.copy()
    vis_mask[mask > 0] = (vis_mask[mask > 0] * 0.4 + np.array([0, 200, 0]) * 0.6)
    vis_lines = bgr.copy()
    for i, l in enumerate(lines):
        p1 = (l.p - l.d * 3000).astype(int)
        p2 = (l.p + l.d * 3000).astype(int)
        cv2.line(vis_lines, tuple(p1), tuple(p2), COLORS[i % 4], 3)

    h = max(bgr.shape[0], 1)
    row = np.hstack([bgr, vis_mask, vis_lines])
    txt = f"{path.name} cloth={color} ratio={cv2.countNonZero(mask)/mask.size:.2f} lines={len(lines)}"
    cv2.putText(row, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2, cv2.LINE_AA)
    out = path.with_name(path.stem + "_diag.png")
    cv2.imwrite(str(out), row)
    return out


if __name__ == "__main__":
    up = Path(__file__).resolve().parents[1] / "testdata" / "uploads"
    targets = sys.argv[1:] or [p.name for p in sorted(up.glob("*_in.jpg"))[-3:]]
    for name in targets:
        p = up / name if not Path(name).exists() else Path(name)
        print("->", panel(p))
