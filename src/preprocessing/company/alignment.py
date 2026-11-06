"""스프링 시작점 기준 회전 정렬과 검사 제외 영역 마스킹.

[왜 필요한가]
스프링 끝면 사진은 선재가 시작되는 지점(연마되지 않은 어두운 부채꼴)의 위치가
부품마다 제각각이다. PatchCore는 학습한 정상과 "모습"을 비교하므로,
시작점 위치가 다르면 정상인데도 처음 보는 모습으로 판단해 과검이 생긴다.
기업도 시작점(0°) 기준 -5° ~ 100° 구간을 검사에서 제외한다.

[하는 일] 이미지마다
  1) 부품 원의 중심을 사진에서 찾는다          find_part_center
  2) 고리를 따라 밝기를 재서 시작점 각도를 찾는다  find_spring_start
  3) 시작점이 항상 같은 방향에 오도록 돌리고,
     부품 바깥과 제외 영역을 같은 색으로 가린다     align_and_mask

[라벨 사용 규칙]
라벨은 카메라별 "대략적인 중심·반지름"(학습 정상 이미지 평균)을 정할 때만 쓰고,
각 이미지의 중심과 시작점은 사진만으로 찾는다. (실제 검사에는 라벨이 없기 때문)

[각도 기준]
이미지 좌표(x 오른쪽, y 아래)에서 atan2(y, x) 각도를 쓴다.
→ 0° = 오른쪽(3시), 90° = 아래(6시), -90° = 위(12시), 각도가 커지면 화면에서 시계 방향.
연마되지 않은 어두운 구간은 시작점에서 각도가 "작아지는" 쪽에 있다.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------- 공통 도우미
def _to_gray(image):
    """컬러(BGR)면 흑백으로, 계산 정밀도를 위해 float32로.

    BGR = OpenCV가 쓰는 색 순서 (파랑, 초록, 빨강).
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def _polar_samples(gray, center, radii, angles_rad):
    """중심에서 (반지름 × 각도) 격자 위치의 밝기를 뽑는다.

    반환 모양: [반지름 개수, 각도 개수]
    cv2.remap은 소수점 좌표도 주변 픽셀로 보간해서 읽어 준다.
    """
    # 원 위의 점 = 중심 + (반지름 × cos, 반지름 × sin)
    # np.outer: 모든 반지름과 모든 각도의 조합을 한 번에 계산 (곱셈표 만들기)
    xs = center[0] + np.outer(radii, np.cos(angles_rad))
    ys = center[1] + np.outer(radii, np.sin(angles_rad))
    return cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)


def _fit_circle(points):
    """점들에 가장 잘 맞는 원 (대수적 최소제곱).

    원의 식 x² + y² = 2·cx·x + 2·cy·y + k 를 선형 방정식으로 풀어 cx, cy, k를 구한다.
    → 점들이 원 위에 딱 맞지 않아도 "전체 오차가 가장 작은" 원을 한 번에 구한다.
    """
    x, y = points[:, 0], points[:, 1]
    a = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    # lstsq: 점마다 생기는 오차의 제곱 합이 가장 작아지는 cx, cy, k를 구한다
    cx, cy, k = np.linalg.lstsq(a, b, rcond=None)[0]
    # 식을 정리하면 k = 반지름² − cx² − cy² 이라서 반지름 = √(k + cx² + cy²)
    return np.array([cx, cy]), float(np.sqrt(k + cx ** 2 + cy ** 2))


def _robust_fit_circle(points, iterations=5, keep=0.8):
    """원 맞춤을 반복하며 원에서 많이 벗어난 점(불량·그림자 등) 20%를 버린다."""
    if len(points) < 3:
        raise ValueError("원을 맞출 경계점이 부족합니다.")
    center, radius = _fit_circle(points)
    for _ in range(iterations):
        # 점마다 원에서 얼마나 떨어져 있나 (중심까지 거리 − 반지름)
        residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        # 원에 가까운 80%만 남기고 다시 맞춘다
        selected = residual <= np.quantile(residual, keep)
        center, radius = _fit_circle(points[selected])
    return center, radius


def _edge_points(gray, center, r_min, r_max, direction, n_angles=180):
    """각도마다 반지름 방향으로 밝기가 가장 크게 바뀌는 지점(경계)을 찾는다.

    direction=+1: 안쪽이 어둡고 바깥이 밝은 경계 (가운데 구멍 → 하얀 연마면)
    direction=-1: 안쪽이 밝고 바깥이 어두운 경계 (하얀 연마면 → 바깥 테두리)
    경계가 약한 각도(연마 안 된 구간 등) 30%는 버린다.
    """
    angles = np.deg2rad(np.linspace(0, 360, n_angles, endpoint=False))
    radii = np.arange(r_min, r_max, 1.0)
    samples = _polar_samples(gray, center, radii, angles)
    # 반지름 방향으로 살짝 흐리게 해서 표면 긁힘 같은 잡음 경계를 줄인다
    samples = cv2.GaussianBlur(samples, (1, 7), 0)
    change = np.diff(samples, axis=0) * direction  # 반지름 1px마다 밝기 변화
    index = change.argmax(axis=0)     # 각도마다 밝기가 가장 크게 바뀐 위치 = 경계
    strength = change.max(axis=0)     # 그때 바뀐 양 (클수록 뚜렷한 경계)
    # 변화량은 두 칸 "사이"의 값이라 반 칸(0.5px)을 더한다
    edge_radius = radii[index] + 0.5
    # (반지름, 각도) → 이미지의 (x, y) 좌표
    points = np.c_[center[0] + edge_radius * np.cos(angles),
                   center[1] + edge_radius * np.sin(angles)]
    return points[strength > np.quantile(strength, 0.3)]


# ---------------------------------------------------------------- 1) 원 중심 찾기
def find_part_center(image, approx_center, approx_radius):
    """사진에서 부품 원의 중심을 찾는다 (두 단계).

    approx_center, approx_radius: 카메라별 대략적인 값 (학습 정상 라벨 평균)
      1단계: 가운데 구멍 경계(어두움→밝음)로 원을 맞춰 대략 위치를 보정
             → 구멍은 불량과 상관없이 모양이 일정해서 크게 틀리지 않음
      2단계: 그 중심에서 하얀 연마면의 바깥 경계(밝음→어두움)로 다시 맞춤
             → 원이 커서 중심이 더 정밀해짐
    회사 데이터 97장 기준, 라벨 외곽선 중심과의 차이: 중앙값 약 4px, 최대 약 26px
    (고정 중심만 쓸 때: 중앙값 약 10px, 최대 약 50px)
    """
    gray = _to_gray(image)
    approx_center = np.asarray(approx_center, dtype=float)
    hole_center, _ = _robust_fit_circle(
        _edge_points(gray, approx_center, 0.25 * approx_radius, 0.6 * approx_radius, +1))
    face_center, _ = _robust_fit_circle(
        _edge_points(gray, hole_center, 0.8 * approx_radius, 1.05 * approx_radius, -1))
    return face_center


# ---------------------------------------------------------------- 2) 시작점 찾기
def ring_brightness(image, center, radius, inner=0.55, outer=0.8, step_deg=0.5):
    """연마면 띠(반지름 inner~outer 배) 위의 각도별 평균 밝기.

    반환: (각도 배열[도], 밝기 배열). 기본 0.5° 간격 → 720개.
    안쪽 구멍과 바깥 테두리를 피해서 하얀 연마면만 지나가는 위치를 본다.
    """
    angles_deg = np.arange(0, 360, step_deg)
    radii = np.linspace(inner * radius, outer * radius, 24)
    samples = _polar_samples(_to_gray(image), center, radii, np.deg2rad(angles_deg))
    return angles_deg, samples.mean(axis=0)


def find_spring_start(image, center, radius, window_deg=3.0, **ring_options):
    """스프링 선재 시작점 각도를 찾는다.

    연마되지 않은 구간은 어둡고, 선재 끝에서는 "아주 어두운 선 → 바로 하얀 연마면"으로
    밝기가 절벽처럼 뛴다. 각 각도에서 (window_deg 앞 밝기 − window_deg 뒤 밝기)가
    가장 큰 곳 = 시작점.

    반환: (시작점 각도[도, 0~360], 밝기 상승폭) — 상승폭이 작으면 검출을 의심해야 한다.
    """
    angles, brightness = ring_brightness(image, center, radius, **ring_options)
    step = angles[1] - angles[0]                    # 한 칸의 각도 (기본 0.5°)
    shift = max(1, int(round(window_deg / step)))   # window_deg가 몇 칸인지 (3° → 6칸)
    count = len(angles)
    # 1) 넓게 보기: ±window_deg 차이가 가장 큰 곳 = 절벽이 있는 대략적인 위치
    #    np.roll로 한 바퀴를 이어 붙여 계산 (0°와 360° 경계도 자연스럽게 처리)
    rise = np.roll(brightness, -shift) - np.roll(brightness, shift)
    index = int(np.argmax(rise))
    # 2) 좁게 보기: 그 주변 ±window_deg 안에서 바로 옆 칸과의 밝기 차이가 가장 큰 곳
    #    = 절벽의 정확한 위치. (1단계만 쓰면 절벽 앞뒤 몇 칸이 같은 값이라 1~2° 밀린다)
    # % count: 마지막 칸 다음은 다시 0번 칸 (시계처럼 한 바퀴 돌아옴)
    neighbors = [(index + offset) % count for offset in range(-shift, shift)]
    step_rise = [brightness[(i + 1) % count] - brightness[i] for i in neighbors]
    edge = neighbors[int(np.argmax(step_rise))]
    # 절벽은 edge 칸과 그다음 칸 사이에 있으므로 반 칸을 더한다
    start = (angles[edge] + step / 2.0) % 360.0
    return float(start), float(rise[index])


# ---------------------------------------------------------------- 3) 정렬 + 가리기
def align_and_mask(image, center, start_deg, radius, *, target_deg=-90.0,
                   exclusion_deg=(-100.0, 5.0), outer_margin=1.03,
                   crop_scale=1.08, fill_value=0):
    """시작점이 target_deg 방향에 오도록 돌리고, 부품만 남기고 나머지를 가린다.

    image        : 원본 이미지 (BGR)
    center       : 부품 중심 (find_part_center 결과)
    start_deg    : 시작점 각도 (find_spring_start 결과)
    radius       : 부품 외곽선 반지름 (카메라별 대략값)
    target_deg   : 정렬 후 시작점이 올 방향 (-90 = 12시 방향)
    exclusion_deg: 시작점 기준 가릴 각도 범위 (기업 기준 -5°~100°, 어두운 쪽이 음수)
    outer_margin : 반지름의 몇 배 바깥부터 가릴지 (배경·받침대 제거)
    crop_scale   : 출력 정사각형 한 변 = 2 × radius × crop_scale
    fill_value   : 가린 부분을 채울 값 (모든 이미지에서 같아야 PatchCore가 차이로 보지 않음)

    반환: 정렬·마스킹된 정사각형 이미지 (부품 중심이 이미지 한가운데)
    """
    size = int(round(2 * radius * crop_scale))
    out_center = (size - 1) / 2.0

    # 회전량: cv2는 양수 각도 = 화면 반시계 방향. 우리 각도는 시계 방향으로 커지므로
    # 반시계로 (start - target)만큼 돌리면 start가 target 위치로 온다.
    rotation = start_deg - target_deg
    # matrix: "어디로 옮길지"를 담은 2x3 숫자 표. 회전과 이동을 한 번에 적용할 수 있다
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), rotation, 1.0)
    # 회전 후 부품 중심이 출력 이미지 한가운데 오도록 평행 이동을 더한다
    matrix[0, 2] += out_center - center[0]
    matrix[1, 2] += out_center - center[1]
    aligned = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fill_value)

    # 가릴 영역 계산 (출력 이미지 좌표, 각도는 같은 규칙)
    ys, xs = np.mgrid[0:size, 0:size]          # 모든 픽셀의 (y, x) 좌표
    dx, dy = xs - out_center, ys - out_center
    distance = np.hypot(dx, dy)                 # 픽셀마다 중심까지 거리
    angle = np.degrees(np.arctan2(dy, dx))      # 픽셀마다 중심에서 본 각도
    # 정렬 후 시작점 기준 각도. +180 → %360 → -180 은 값을 -180°~180° 범위로 맞추는 방법
    relative = (angle - target_deg + 180.0) % 360.0 - 180.0
    low, high = exclusion_deg
    # 가릴 픽셀 = 부품보다 바깥이거나(|) 제외 각도 범위 안에 있는 픽셀
    mask = (distance > outer_margin * radius) | ((relative >= low) & (relative <= high))
    aligned[mask] = fill_value
    return aligned


def align_image(image, approx_center, approx_radius, **align_options):
    """1) 중심 → 2) 시작점 → 3) 정렬·가리기를 한 번에. 검출 정보도 함께 반환한다."""
    center = find_part_center(image, approx_center, approx_radius)
    start_deg, rise = find_spring_start(image, center, approx_radius)
    aligned = align_and_mask(image, center, start_deg, approx_radius, **align_options)
    info = {"center_x": float(center[0]), "center_y": float(center[1]),
            "start_deg": start_deg, "start_rise": rise}
    return aligned, info
