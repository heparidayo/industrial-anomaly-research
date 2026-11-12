"""면취 고리 펼치기(극좌표) 테스트.

면취 폭과 "일부만 넓은 구간"의 각도를 알고 있는 가짜 스프링 끝면을 그려서,
펼친 띠에서 폭이 일정하게 나오는지, 넓은 구간이 정확한 위치에 나오는지 확인한다.
"""
import unittest

import cv2
import numpy as np

from src.preprocessing.company.polar import (
    square_columns, stack_rows, unwrap_band, unwrap_image,
)

CENTER, RADIUS, SIZE = (205, 195), 150, 420     # 가짜 부품의 중심, 외곽 반지름, 이미지 크기
CHAMFER = 0.92                                  # 정상 면취 안쪽 경계 = 반지름 0.92배 (폭 12px)
WIDE = 0.85                                     # 넓은 구간의 안쪽 경계 = 0.85배 (폭 22.5px)
WIDE_RANGE = (120.0, 140.0)                     # 넓은 구간 각도 (시작점 기준)


def make_chamfer_face(start_deg=40.0, wide=True):
    """가짜 스프링 끝면 (흑백 → BGR).

    - 배경 90, 가운데 구멍 25 (반지름 0.4배)
    - 하얀 연마면 240 (0.4배 ~ 면취 경계)
    - 면취 띠 45 (면취 경계 ~ 1.0배, 실제 사진처럼 어두움). wide=True면 WIDE_RANGE 각도만 띠가 안쪽으로 넓음
      (띠를 배경보다 밝게 그리면 "연마면→띠"와 "띠→배경" 두 경계가 똑같이 뚜렷해져 중심 찾기가 헷갈린다)
    - 연마 안 된 구간: 시작점에서 각도가 작아지는 쪽 90°를 110으로, 바로 앞 2°는 40 (선재 끝 절벽)
    """
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]                                  # 모든 픽셀 좌표
    distance = np.hypot(xs - CENTER[0], ys - CENTER[1]) / RADIUS       # 중심 거리 (반지름 배수)
    angle = np.degrees(np.arctan2(ys - CENTER[1], xs - CENTER[0]))     # 중심에서 본 각도
    relative = (angle - start_deg) % 360.0                             # 시작점 기준 각도 (0~360)
    edge = np.full(distance.shape, CHAMFER)                            # 위치마다 면취 안쪽 경계
    if wide:
        in_wide = (relative >= WIDE_RANGE[0]) & (relative <= WIDE_RANGE[1])
        edge[in_wide] = WIDE                                           # 넓은 구간은 경계가 더 안쪽
    image = np.full((SIZE, SIZE), 90, np.uint8)                        # 배경
    image[distance <= 1.0] = 45                                        # 면취 띠 (부품 전체를 먼저 칠함)
    face = (distance > 0.4) & (distance < edge)                        # 하얀 연마면 영역
    image[face] = 240
    signed = (relative + 180.0) % 360.0 - 180.0                        # -180~180 기준 각도
    image[face & (signed < 0) & (signed >= -90)] = 110                 # 연마 안 된 구간
    image[face & (signed < 0) & (signed >= -2)] = 40                   # 선재 끝 어두운 선
    image[distance <= 0.4] = 25                                        # 구멍
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def boundary_rows(strip):
    """띠의 열마다 위(바깥)에서부터 처음으로 하얀 연마면(>200)이 나오는 행 = 면취 안쪽 경계."""
    gray = strip[..., 0].astype(int)                                   # 흑백이라 한 채널만 사용
    return (gray > 200).argmax(axis=0)                                 # 열마다 첫 True 위치


class PolarUnwrapTests(unittest.TestCase):
    def test_stack_rows_order_and_square(self):
        strip = np.arange(2 * 8).reshape(2, 8)                         # 2행 8열짜리 띠
        stacked = stack_rows(strip, 2)                                 # 두 토막으로 쌓기
        np.testing.assert_array_equal(stacked[:2], strip[:, :4])       # 위 = 앞쪽 각도
        np.testing.assert_array_equal(stacked[2:], strip[:, 4:])       # 아래 = 뒤쪽 각도
        self.assertEqual(square_columns(45, 4), 720)                   # 정사각형 조건 = 4×4×45
        with self.assertRaises(ValueError):
            stack_rows(np.zeros((2, 7)), 2)                            # 나눠떨어지지 않으면 오류

    def test_constant_width_is_flat_and_wide_part_is_found(self):
        image = make_chamfer_face(start_deg=40.0)
        columns = 720
        strip = unwrap_band(image, CENTER, 40.0, RADIUS, radial_px=45, columns=columns)
        rows = boundary_rows(strip)
        # 각도 → 열 번호 (펼친 범위 5°~260°)
        column_of = lambda deg: int((deg - 5.0) / 255.0 * columns)
        wide_cols = slice(column_of(WIDE_RANGE[0]) + 3, column_of(WIDE_RANGE[1]) - 3)
        normal = np.r_[rows[:column_of(WIDE_RANGE[0]) - 3], rows[column_of(WIDE_RANGE[1]) + 3:]]
        # 정상 구간은 경계가 한 줄로 평평해야 한다 (펼치기가 맞으면 원이 직선이 됨)
        self.assertLessEqual(normal.max() - normal.min(), 1)
        # 넓은 구간은 경계가 더 아래(안쪽)에 있어야 한다: 0.07배 × 150px ≈ 10px ≈ 10행
        self.assertGreater(np.median(rows[wide_cols]) - np.median(normal), 7)

    def test_same_part_with_different_start_gives_same_result(self):
        # 시작점만 다른 같은 부품 → 펼친 결과가 거의 같아야 한다 (시작점 기준으로 펼치므로)
        a, info_a = unwrap_image(make_chamfer_face(start_deg=30.0), CENTER, RADIUS, rows=4, radial_px=45)
        b, info_b = unwrap_image(make_chamfer_face(start_deg=200.0), CENTER, RADIUS, rows=4, radial_px=45)
        self.assertEqual(a.shape, (180, 180, 3))                       # 4 × 45 = 180 정사각형
        self.assertLess(np.abs(a.astype(int) - b.astype(int)).mean(), 3.0)
        self.assertLess(abs(info_a["start_deg"] - 30.0), 2.0)          # 시작점도 맞게 찾았는지


if __name__ == "__main__":
    unittest.main()
