"""회사 PLC 데이터용 전처리 (학습과 추론에 똑같이 적용).

1. 조명 정규화 (CLAHE)
   LAB 색공간의 밝기(L) 채널에만 CLAHE를 적용해, 사진마다 다른 반사 · 밝기를 고르게 맞춥니다.

2. 스프링 시작점 기준 정렬
   edge_mask.spring_start_angle로 시작점(0°)과 방향을 찾고,
   제외 구간(-5°~100°)의 가운데가 항상 이미지 아래쪽(90°)에 오도록 이미지를 돌립니다.
   방향 판정이 가끔 틀려도 구간의 '가운데'를 맞추므로 어두운 부채꼴은 늘 같은 자리에 옵니다.

추론 때는 돌린 이미지로 결함 맵을 만든 뒤, 맵을 반대로 돌려 원래 방향으로 되돌립니다.
그래서 테두리 마스크 · 사례 그림 등 나머지 단계는 그대로 씁니다.
"""

import cv2
import numpy as np

from .edge_mask import spring_start_angle

TARGET_DEG = 90.0  # 정렬 후 제외 구간 가운데가 올 각도 (이미지 아래쪽)


def apply_clahe(image_bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """uint8 BGR → uint8 BGR (밝기 채널만 CLAHE)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def align_angle(image_bgr: np.ndarray, deg_from: float = -5, deg_to: float = 100) -> float:
    """이미지를 몇 도 돌려야 하는지 (cv2.getRotationMatrix2D에 넣는 각도)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    start, direction, _ = spring_start_angle(gray)
    center = start + direction * (deg_from + deg_to) / 2  # 제외 구간 가운데 (이미지 좌표, 시계 방향 +)
    # cv2 각도는 반시계 방향 +. 각도 center인 점을 TARGET_DEG로 옮기려면 center - TARGET_DEG 만큼 돌림
    return float(center - TARGET_DEG)


def rotate(image: np.ndarray, angle: float, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=interpolation, borderMode=cv2.BORDER_REPLICATE)


def preprocess(image_bgr: np.ndarray, size: int, clahe: bool, align: bool):
    """uint8 BGR → (uint8 BGR (size, size), 돌린 각도). 정렬 각도는 CLAHE 전 원본으로 계산."""
    image = cv2.resize(image_bgr, (size, size))
    angle = align_angle(image) if align else 0.0
    if clahe:
        image = apply_clahe(image)
    if angle:
        image = rotate(image, angle)
    return image, angle


# ── 3. 테두리 띠 펴기 (극좌표 변환) ─────────────────────────────
# 부품 원 중심(이미지 가운데)을 기준으로 바깥 원 근처 링을 각도 × 반지름의 가로 띠로 폅니다.
#   가로(열) = 각도: 스프링 시작점 제외 구간(-5°~100°)이 끝난 곳부터 시계 방향으로 나머지 255°
#   세로(행) = 반지름: 위쪽 = 바깥(r_out), 아래쪽 = 안쪽(r_in). 반지름 단위는 바깥 원 반지름 R
# 제외 구간은 띠에 아예 들어가지 않고, 띠의 왼쪽 · 오른쪽 끝(이음매)도 그 구간 양쪽에 놓입니다.
# 이미지는 prepare_company_plc.py에서 원 중심에 맞춰 반폭 = R × 1.1로 잘려 있다고 가정합니다.

POLAR_H, POLAR_W = 128, 512      # 띠 크기 (32의 배수) · 1024는 GPU 메모리 부족으로 512로
POLAR_R = (0.80, 1.03)           # 펼 반지름 범위 (R 단위)
EXCLUDE = (-5.0, 100.0)          # 스프링 시작점 기준 제외 구간 (°)


def exclude_center(image_bgr: np.ndarray, deg_from: float = EXCLUDE[0], deg_to: float = EXCLUDE[1]) -> float:
    """제외 구간 가운데 각도 (이미지 좌표, x 오른쪽 0°, 시계 방향 +). 방향 판정이 틀려도 가운데는 어두운 쪽."""
    small = cv2.resize(image_bgr, (256, 256))
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    start, direction, _ = spring_start_angle(gray)
    return float(start + direction * (deg_from + deg_to) / 2)


def _polar_span(deg_from: float = EXCLUDE[0], deg_to: float = EXCLUDE[1]):
    half = (deg_to - deg_from) / 2
    return half, 360.0 - 2 * half  # (가운데에서 띠 시작까지 °, 띠가 덮는 각도 °)


def unwrap(image: np.ndarray, center_deg: float, offset: float = 0.0,
           h: int = POLAR_H, w: int = POLAR_W, r_range=POLAR_R, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """정사각형 이미지 → (h, w) 띠. offset: 각도 흔들기(°, 학습 증강용)."""
    S = image.shape[0]
    c, R = S / 2, S / 2 / 1.1
    half, span = _polar_span()
    theta = np.deg2rad(center_deg + half + offset + span * (np.arange(w) + 0.5) / w)
    radius = R * (r_range[1] - (r_range[1] - r_range[0]) * (np.arange(h) + 0.5) / h)
    xs = (c + np.outer(radius, np.cos(theta))).astype(np.float32)
    ys = (c + np.outer(radius, np.sin(theta))).astype(np.float32)
    return cv2.remap(image, xs, ys, interpolation, borderMode=cv2.BORDER_REPLICATE)


def rewrap(strip: np.ndarray, center_deg: float, size: int, r_range=POLAR_R, fill=None) -> np.ndarray:
    """띠 → 원래 정사각형 이미지 좌표 (size, size). 띠에 없는 곳은 fill(없으면 0)."""
    h, w = strip.shape[:2]
    c, R = size / 2, size / 2 / 1.1
    half, span = _polar_span()
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    r = np.hypot(xx - c, yy - c) / R
    rel = (np.rad2deg(np.arctan2(yy - c, xx - c)) - (center_deg + half)) % 360
    col = (rel / span * w - 0.5).astype(np.float32)
    row = ((r_range[1] - r) / (r_range[1] - r_range[0]) * h - 0.5).astype(np.float32)
    out = cv2.remap(strip.astype(np.float32), col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = (rel <= span) & (r >= r_range[0]) & (r <= r_range[1])
    if fill is not None:
        fill = fill.astype(np.float32)
        out = np.where(inside[..., None] if out.ndim == 3 else inside, out, fill)
    elif out.ndim == 3:
        out = out * inside[..., None]
    else:
        out = out * inside
    return out


def preprocess_polar(image_bgr: np.ndarray, clahe: bool, offset: float = 0.0, center_deg: float | None = None):
    """uint8 BGR (원본 해상도) → (uint8 띠 (POLAR_H, POLAR_W, 3), 제외 구간 가운데 각도)."""
    if center_deg is None:
        center_deg = exclude_center(image_bgr)
    strip = unwrap(image_bgr, center_deg, offset)
    if clahe:
        strip = apply_clahe(strip)
    return strip, center_deg
