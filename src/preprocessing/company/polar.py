"""스프링 끝면의 면취 고리를 직사각형 띠로 펼친다 (극좌표 펼치기).

[왜 필요한가]
기업 불량 정의는 면취 폭 w'(반지름 방향 두께)와 각도 범위로 되어 있다.
  과다(over) = 폭이 넓음, 미흡 = 폭이 좁음, 다단(step) = 일부 각도 구간만 폭이 다름
고리를 펼치면 "가로 = 각도, 세로 = 반지름"이 되어 폭이 띠의 세로 두께로 바로 보인다.
또 구멍·배경을 버리고 면취 띠만 남기므로, 224px로 줄일 때 폭 정보가 덜 사라진다.

[하는 일] 이미지마다
  1) 부품 중심과 스프링 시작점을 찾는다 (alignment.py 함수 재사용)
  2) 시작점 기준 검사 구간(기본 5°~260°)의 고리를 띠로 펼친다   unwrap_band
  3) 긴 띠를 몇 토막으로 잘라 위아래로 쌓아 정사각형으로 만든다  stack_rows
     → PatchCore 입력 변환(짧은 변 232로 줄이고 가운데 224 자르기)에서 잘리는 부분이 없게

[각도 기준] alignment.py와 같다 (이미지 좌표 atan2, 각도가 커지면 화면에서 시계 방향).
시작점에서 각도가 커지는 쪽이 연마면, 작아지는 쪽이 연마되지 않은 구간(검사 제외)이다.
"""

import cv2
import numpy as np

from .alignment import find_part_center, find_spring_start


def unwrap_band(image, center, start_deg, radius, *, inner=0.75, outer=1.05,
                angle_range=(5.0, 260.0), radial_px=108, columns=1728):
    """고리(반지름 inner~outer 배, 시작점 기준 angle_range 각도)를 직사각형 띠로 펼친다.

    image      : 원본 이미지 (BGR)
    center     : 부품 중심 (x, y)
    start_deg  : 스프링 시작점 각도 (도)
    radius     : 부품 외곽선 반지름 (카메라별 대략값)
    inner/outer: 펼칠 반지름 범위 (외곽선 반지름의 배수)
    angle_range: 펼칠 각도 범위 (시작점 기준, 도). 기업 검사 제외 -5°~100°를 뺀 나머지
    radial_px  : 띠의 세로 칸 수 (반지름 방향)
    columns    : 띠의 가로 칸 수 (각도 방향)

    반환: [radial_px, columns, 채널] 띠. 맨 윗줄 = 바깥(outer), 맨 아랫줄 = 안쪽(inner),
          맨 왼쪽 열 = 시작점 + angle_range[0], 맨 오른쪽 열 = 시작점 + angle_range[1]
    """
    # 세로 칸마다의 반지름: 위(바깥) → 아래(안쪽) 순서
    radii = np.linspace(outer * radius, inner * radius, radial_px)
    # 가로 칸마다의 각도: 칸의 가운데 위치를 쓰려고 반 칸씩 안쪽에서 시작/끝
    low, high = angle_range                                      # 펼칠 각도 범위 (시작점 기준)
    step = (high - low) / columns                                # 한 칸이 차지하는 각도
    angles = np.deg2rad(start_deg + low + step * (np.arange(columns) + 0.5))  # 칸 중심 각도(라디안)
    # 칸마다 원본에서 가져올 위치 = 중심 + 반지름 × (cos, sin)  (np.outer = 모든 조합 곱셈표)
    map_x = (center[0] + np.outer(radii, np.cos(angles))).astype(np.float32)  # 가져올 x 좌표 표
    map_y = (center[1] + np.outer(radii, np.sin(angles))).astype(np.float32)  # 가져올 y 좌표 표
    # remap: 표에 적힌 위치의 색을 가져와 새 이미지를 채운다 (소수점 위치는 주변 4픽셀 보간)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # 사진 밖은 검정


def stack_rows(strip, rows):
    """가로로 긴 띠를 rows 토막으로 잘라 위에서 아래로 쌓는다.

    예: rows=4 → [0~25%][25~50%][50~75%][75~100%] 각도 구간이 위에서부터 차례로 놓인다.
    """
    if strip.shape[1] % rows:                                    # 토막 길이가 딱 나눠떨어져야 함
        raise ValueError("띠의 가로 칸 수가 토막 수로 나눠떨어지지 않습니다.")
    width = strip.shape[1] // rows                               # 토막 하나의 가로 칸 수
    pieces = [strip[:, i * width:(i + 1) * width] for i in range(rows)]  # 왼쪽부터 차례로 자르기
    return np.vstack(pieces)                                     # 위에서 아래로 쌓기


def square_columns(radial_px, rows):
    """쌓은 결과가 정사각형이 되는 띠의 가로 칸 수.

    쌓은 이미지: 세로 = rows × radial_px, 가로 = columns / rows
    → 둘이 같으려면 columns = rows × rows × radial_px
    """
    return rows * rows * radial_px                               # 정사각형 조건


def unwrap_image(image, approx_center, approx_radius, *, rows=4, radial_px=108, columns=None,
                 **band_options):
    """중심·시작점 찾기 → 띠 펼치기 → 토막 내어 쌓기를 한 번에. 검출 정보도 반환한다.

    columns=None : 쌓은 결과가 정사각형이 되도록 가로 칸 수를 자동 계산 (PatchCore 기본 입력용)
    columns=숫자 : 그 가로 칸 수로 펼친다. rows=1과 함께 쓰면 긴 띠를 쌓지 않고 그대로 저장
                   (PatchCore를 input_mode="full"로 실행해 띠 전체를 자르지 않고 볼 때)
    """
    center = find_part_center(image, approx_center, approx_radius)            # 1) 부품 중심
    start_deg, rise = find_spring_start(image, center, approx_radius)         # 1) 시작점 각도
    if columns is None:                                                       # 가로 칸 수를 안 정했으면
        columns = square_columns(radial_px, rows)                             # 정사각형이 되는 값 사용
    strip = unwrap_band(image, center, start_deg, approx_radius,
                        radial_px=radial_px, columns=columns, **band_options)  # 2) 띠로 펼치기
    stacked = stack_rows(strip, rows)                                         # 3) 토막 내어 쌓기
    info = {"center_x": float(center[0]), "center_y": float(center[1]),      # 검출 결과 기록용
            "start_deg": start_deg, "start_rise": rise}
    return stacked, info
