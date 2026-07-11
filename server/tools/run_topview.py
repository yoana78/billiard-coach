"""CLI: 사진 파일 → 탑뷰 변환 결과 저장.

사용법:
  python tools\\run_topview.py <입력사진> [출력파일]

출력: 탑뷰 이미지 + 디버그 오버레이(_debug 접미사) 저장, 품질 지표 출력.
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.pipeline import image_to_topview, draw_debug


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    bgr = cv2.imread(str(src))
    if bgr is None:
        print(f"이미지를 열 수 없음: {src}")
        return 1
    result = image_to_topview(bgr)
    if not result.ok:
        print(f"변환 실패: {result.reason}")
        return 2
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_topview.png")
    cv2.imwrite(str(out), result.topview)
    dbg = draw_debug(result)
    dbg_path = out.with_name(out.stem + "_debug.png")
    cv2.imwrite(str(dbg_path), dbg)
    print(f"천 색상: {result.cloth_color}")
    print(f"촬영 유형: {'부분' if result.partial else '전체'} (라인 {result.visible_lines}개)")
    print(f"다이아몬드 검출: {len(result.diamonds)}/28")
    print(f"평균 위치 오차: {result.diamond_err_mm:.1f} mm")
    print(f"재정밀화(refine): {result.refined}")
    print(f"저장: {out}")
    print(f"디버그: {dbg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
