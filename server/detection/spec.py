"""중대(中臺) 당구대 규격 상수 — 기획서 2장 기준.

모든 실측 단위는 mm. 탑뷰 좌표계는 (0,0)=경기면(쿠션 안쪽) 좌상단,
x=가로(장쿠션 방향, 0~2540), y=세로(단쿠션 방향, 0~1270).
"""

# 경기면 (쿠션 안쪽)
TABLE_W_MM = 2540.0   # 장쿠션 안쪽 길이
TABLE_H_MM = 1270.0   # 단쿠션 안쪽 길이

# 쿠션/레일
CUSHION_HEIGHT_MM = 38.0   # 펠트면과 쿠션 상면(다이아몬드가 붙은 면)의 높이 차 — 시차 보정 상수
RAIL_WIDTH_MM = 150.0      # 쿠션 코 ~ 레일 바깥까지 목재 폭 (합성 렌더링용 근사값)
DIAMOND_OFFSET_MM = 95.0   # 쿠션 날(고무 끝)에서 다이아몬드 중심까지 거리 (사용자 실측, 중대 기준)
# 실제 당구대는 쿠션도 천으로 덮여 있어, 사진에서 보이는 '천 경계'는
# 경기면이 아니라 쿠션 바깥(목재 레일 시작) 라인이다. 그 수평 오프셋.
CUSHION_OVERHANG_MM = 40.0


def cloth_boundary_corners_mm():
    """천 경계(쿠션 바깥) 사각형의 4모서리 (tl, tr, br, bl) mm."""
    ov = CUSHION_OVERHANG_MM
    return [(-ov, -ov), (TABLE_W_MM + ov, -ov),
            (TABLE_W_MM + ov, TABLE_H_MM + ov), (-ov, TABLE_H_MM + ov)]

# 공
BALL_DIAMETER_MM = 65.5

# 다이아몬드 포인트 간격
DIAMOND_SPACING_MM = 317.5  # = TABLE_W_MM / 8 = TABLE_H_MM / 4


def diamond_points_mm():
    """탑뷰 좌표계(mm)의 다이아몬드 28점 목록을 (x, y, side) 로 반환.

    side: 'top' | 'bottom' (장쿠션 9점씩) | 'left' | 'right' (단쿠션 5점씩)
    실제 당구대에는 모서리 시작점 포인트까지 있는 경우가 많다
    (사용자 확인: 장쿠션 9개 = 0/8~8/8, 단쿠션 5개 = 0/4~4/4).
    다이아몬드는 경기면 바깥 레일 위에 있으므로 좌표가 0보다 작거나
    TABLE_W/H 보다 클 수 있다.
    """
    pts = []
    for i in range(0, 9):  # 장쿠션 8등분 + 양 끝 시작점 → 9점
        x = i * DIAMOND_SPACING_MM
        pts.append((x, -DIAMOND_OFFSET_MM, "top"))
        pts.append((x, TABLE_H_MM + DIAMOND_OFFSET_MM, "bottom"))
    for j in range(0, 5):  # 단쿠션 4등분 + 양 끝 시작점 → 5점
        y = j * DIAMOND_SPACING_MM
        pts.append((-DIAMOND_OFFSET_MM, y, "left"))
        pts.append((TABLE_W_MM + DIAMOND_OFFSET_MM, y, "right"))
    return pts
