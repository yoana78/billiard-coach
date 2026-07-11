"""업로드 사진들의 공 검출 회귀 확인."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.pipeline import image_to_topview

up = Path(__file__).resolve().parents[1] / "testdata" / "uploads"
for name in sys.argv[1:]:
    r = image_to_topview(cv2.imread(str(up / f"{name}_in.jpg")))
    if r.ok:
        print(name, "| balls:", sorted(b.color for b in r.balls),
              "| diamonds", len(r.diamonds), "| err", round(r.diamond_err_mm, 1))
    else:
        print(name, "| FAIL:", r.reason)
