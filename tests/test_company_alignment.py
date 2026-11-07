"""스프링 시작점 정렬 테스트.

정답(중심, 시작점 각도)을 알고 있는 가짜 스프링 끝면 이미지를 그려서,
코드가 그 값을 찾아내는지 확인한다.
"""
import unittest

import cv2
import numpy as np

from src.preprocessing.company.alignment import (
    align_and_mask, align_image, find_part_center, find_spring_start,
)


def make_spring_face(center=(205, 195), radius=150, start_deg=40.0, size=420):
    """가짜 스프링 끝면 (흑백 → BGR).

    - 배경: 어두운 회색(60)
    - 부품 원: 바깥 테두리 어두운 링(40) 안에 하얀 연마면(240)
    - 가운데 구멍: 어두움(25), 반지름 = 0.4배
    - 연마 안 된 구간: 시작점에서 각도가 작아지는 쪽 90°를 어둡게(110)
      + 시작점 바로 앞 2°는 아주 어두운 선(40) → 실제 사진의 "절벽" 경계를 흉내
    """
    image = np.full((size, size), 60, np.uint8)
    cx, cy = center
    ys, xs = np.mgrid[0:size, 0:size]
    distance = np.hypot(xs - cx, ys - cy)
    angle = np.degrees(np.arctan2(ys - cy, xs - cx))
    relative = (angle - start_deg + 180) % 360 - 180  # 시작점 기준 각도
    image[distance <= radius] = 40                     # 바깥 어두운 링
    face = (distance <= 0.9 * radius) & (distance > 0.4 * radius)
    image[face] = 240                                  # 하얀 연마면
    image[face & (relative < 0) & (relative >= -90)] = 110   # 연마 안 된 구간
    image[face & (relative < 0) & (relative >= -2)] = 40     # 선재 끝 어두운 선
    image[distance <= 0.4 * radius] = 25               # 구멍
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class SpringAlignmentTests(unittest.TestCase):
    def test_center_found_from_rough_guess(self):
        image = make_spring_face(center=(205, 195))
        # 대략값을 일부러 12px 정도 틀리게 줘도 실제 중심을 찾아야 한다
        center = find_part_center(image, (215, 188), 150)
        self.assertLess(np.hypot(center[0] - 205, center[1] - 195), 3.0)

    def test_spring_start_angle(self):
        for true_start in (40.0, -120.0, 175.0):
            image = make_spring_face(start_deg=true_start)
            start, rise = find_spring_start(image, (205, 195), 150)
            # 359°와 1°처럼 한 바퀴 경계를 넘어도 차이가 2°로 나오게 계산
            error = abs((start - true_start + 180) % 360 - 180)
            self.assertLess(error, 2.0, msg=f"정답 {true_start}°, 검출 {start}°")
            self.assertGreater(rise, 80)

    def test_aligned_image_puts_start_at_target_and_masks(self):
        image = make_spring_face(start_deg=75.0)
        aligned = align_and_mask(image, (205, 195), 75.0, 150, target_deg=-90.0,
                                 exclusion_deg=(-100.0, 5.0), fill_value=0)
        size = aligned.shape[0]
        c = (size - 1) / 2.0

        def pixel(relative_deg, radius_ratio):
            # 정렬 후 "시작점 기준 각도"의 연마면 위치 픽셀 밝기
            # (각도, 반지름) → (행, 열) 픽셀 위치로 바꿔서 읽는다
            a = np.deg2rad(-90.0 + relative_deg)
            r = radius_ratio * 150
            return int(aligned[int(round(c + r * np.sin(a))), int(round(c + r * np.cos(a))), 0])

        self.assertGreater(pixel(+30, 0.65), 200)   # 시작점 다음 = 하얀 연마면이 보여야 함
        self.assertEqual(pixel(-50, 0.65), 0)       # 제외 영역 = 가려짐
        self.assertEqual(int(aligned[0, 0, 0]), 0)  # 부품 바깥(모서리) = 가려짐

    def test_rotated_copies_look_the_same_after_alignment(self):
        # 시작점만 다른 두 부품이 정렬 후에는 거의 같은 이미지가 되어야 한다 (이 단계의 목적)
        a, _ = align_image(make_spring_face(start_deg=10.0), (205, 195), 150)
        b, _ = align_image(make_spring_face(start_deg=200.0), (205, 195), 150)
        difference = np.abs(a.astype(int) - b.astype(int)).mean()
        self.assertLess(difference, 3.0)


if __name__ == "__main__":
    unittest.main()
