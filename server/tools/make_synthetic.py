"""검증용 합성 당구대 사진 생성기.

탑뷰 기준으로 중대(파란 천 + 목재 레일 + 다이아몬드 20개 + 공 4개)를
그린 뒤, 임의의 원근 변환을 적용해 '비스듬히 찍은 사진'을 만든다.
정답 호모그래피를 알고 있으므로 파이프라인 정확도를 정량 검증할 수 있다.
"""
import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection import spec
from detection.homography import mm_to_px, canvas_size, PX_PER_MM


def render_topview_truth(
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """규격대로 그린 탑뷰 원본(정답) 이미지와 공 정답 위치 목록.

    반환 balls: [(color, x_mm, y_mm), ...] — color: white/yellow/red
    """
    w, h = canvas_size()
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # 바닥(어두운 회색)

    # 레일 (목재 갈색): 경기면보다 RAIL_WIDTH 만큼 큰 사각형
    r0 = mm_to_px((-spec.RAIL_WIDTH_MM, -spec.RAIL_WIDTH_MM))
    r1 = mm_to_px((spec.TABLE_W_MM + spec.RAIL_WIDTH_MM,
                   spec.TABLE_H_MM + spec.RAIL_WIDTH_MM))
    cv2.rectangle(img, (int(r0[0]), int(r0[1])), (int(r1[0]), int(r1[1])),
                  (40, 70, 120), -1)  # BGR 갈색

    # 천 (파랑): 실제 당구대처럼 쿠션(오버행)까지 천으로 덮인 영역
    ov = spec.CUSHION_OVERHANG_MM
    c0 = mm_to_px((-ov, -ov))
    c1 = mm_to_px((spec.TABLE_W_MM + ov, spec.TABLE_H_MM + ov))
    cv2.rectangle(img, (int(c0[0]), int(c0[1])), (int(c1[0]), int(c1[1])),
                  (170, 110, 40), -1)  # BGR 파란 천 (쿠션부: 살짝 어둡게)
    p0 = mm_to_px((0, 0))
    p1 = mm_to_px((spec.TABLE_W_MM, spec.TABLE_H_MM))
    cv2.rectangle(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                  (180, 120, 30), -1)  # 경기면

    # 다이아몬드 (흰 점, 지름 약 18mm)
    rad = max(2, int(9 * PX_PER_MM))
    for (x, y, _side) in spec.diamond_points_mm():
        px, py = mm_to_px((x, y))
        cv2.circle(img, (int(px), int(py)), rad, (235, 235, 235), -1)

    # 공 4개 (흰/노랑/빨강2) — 서로 겹치지 않게 배치, 정답 위치 기록
    if rng is None:
        rng = np.random.default_rng(0)
    colors = [("white", (255, 255, 255)), ("yellow", (60, 220, 240)),
              ("red", (50, 50, 220)), ("red", (50, 50, 220))]
    ball_r = int(spec.BALL_DIAMETER_MM / 2 * PX_PER_MM)
    balls: list[tuple[str, float, float]] = []
    placed: list[tuple[float, float]] = []
    for name, col in colors:
        for _ in range(100):
            bx = rng.uniform(200, spec.TABLE_W_MM - 200)
            by = rng.uniform(200, spec.TABLE_H_MM - 200)
            if all(np.hypot(bx - x, by - y) > spec.BALL_DIAMETER_MM * 2
                   for x, y in placed):
                break
        placed.append((bx, by))
        balls.append((name, bx, by))
        px, py = mm_to_px((bx, by))
        cv2.circle(img, (int(px), int(py)), ball_r, col, -1)

    return img, balls


def make_photo(seed: int = 0, out_size=(1600, 1200),
               tilt: float = 0.25) -> tuple[np.ndarray, np.ndarray, list]:
    """합성 '사진', 정답 호모그래피(사진 → 탑뷰 캔버스 px), 공 정답 위치 반환.

    tilt: 원근 왜곡 정도 (0=정면 탑뷰, 0.3=꽤 비스듬)
    """
    rng = np.random.default_rng(seed)
    truth, balls = render_topview_truth(rng)
    th, tw = truth.shape[:2]
    ow, oh = out_size

    # 탑뷰 4모서리를 사진 안의 사다리꼴로 매핑 (위쪽이 좁아지는 원근 흉내)
    m = 0.08  # 사진 가장자리 여백 비율
    top_inset = tilt * ow * rng.uniform(0.7, 1.3) * 0.5
    jitter = lambda s: rng.uniform(-s, s)
    j = ow * 0.02
    dst = np.array([
        [ow * m + top_inset + jitter(j),  oh * m + jitter(j)],
        [ow * (1 - m) - top_inset + jitter(j), oh * m + jitter(j)],
        [ow * (1 - m) + jitter(j), oh * (1 - m) + jitter(j)],
        [ow * m + jitter(j), oh * (1 - m) + jitter(j)],
    ], dtype=np.float32)
    src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)

    H_truth_to_photo = cv2.getPerspectiveTransform(src, dst)
    photo = np.zeros((oh, ow, 3), dtype=np.uint8)
    photo[:] = (25, 28, 32)
    cv2.warpPerspective(truth, H_truth_to_photo, (ow, oh), dst=photo,
                        borderMode=cv2.BORDER_TRANSPARENT)

    # 조명 변화 + 노이즈 (현실감)
    gamma = rng.uniform(0.85, 1.15)
    lut = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    photo = lut[photo]
    noise = rng.normal(0, 4, photo.shape).astype(np.int16)
    photo = np.clip(photo.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    H_photo_to_truth = np.linalg.inv(H_truth_to_photo)
    return photo, H_photo_to_truth, balls


def make_partial_photo(seed: int = 0, tilt: float = 0.2,
                       keep: float = 0.6) -> tuple[np.ndarray, np.ndarray, list]:
    """당구대 일부만 프레임에 담긴 합성 사진 (오른쪽 keep 비율 이후 잘림).

    잘라도 x0=0 이므로 정답 호모그래피는 그대로 유효하다.
    """
    photo, H, balls = make_photo(seed=seed, out_size=(2200, 1300), tilt=tilt)
    x1 = int(photo.shape[1] * keep)
    return photo[:, :x1].copy(), H, balls


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "testdata"
    out_dir.mkdir(exist_ok=True)
    for seed in range(3):
        photo, _, _ = make_photo(seed=seed, tilt=0.15 + 0.08 * seed)
        p = out_dir / f"synthetic_{seed}.png"
        cv2.imwrite(str(p), photo)
        print("saved", p)
