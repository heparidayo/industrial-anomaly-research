"""테두리 영역 마스크 — 결함 맵에서 부품 테두리 띠만 남기고 나머지는 0으로.

회사 PLC 데이터의 불량(over · step · 미흡)은 부품 상면(가공면) 바깥 경계와
부품 바깥 원 사이의 '테두리 띠'에서 생긴다고 가정합니다.
그래서 상면 안쪽(흰 면, 가운데 구멍, 어두운 부채꼴의 반사)과 부품 밖 배경에 나온 반응은
점수에서 빼고, 테두리 띠(바깥 원 ~ 흰 면 끝 + 흰 면 안쪽 12px)만 봅니다.

마스크는 scripts/draem/make_edge_masks.py가 라벨 txt로 만들어
    data/processed/company_plc/<category>/edge_masks/<이미지 이름>.png
에 저장합니다 (흰색 = 보는 영역).

모델은 그대로 두고 점수 계산만 바꾸므로 재학습이 필요 없습니다.
"""

from pathlib import Path

import cv2
import numpy as np

MASK_DIR = "edge_masks"

# 마스크 종류: 이름 → (폴더, 표시 이름). threshold JSON에는 threshold_<이름>으로 저장
MASK_VARIANTS = {
    "edge": ("edge_masks", "테두리"),
    "edge_ex": ("edge_masks_ex", "테두리 · 스프링 시작점 제외"),
    # 라벨 없이 만드는 공정한 마스크: 바깥 원 기준 고정 폭 링(띠 펴기와 같은 0.80~1.03R) − 시작점 구간
    "ring_ex": ("ring_masks_ex", "고정 폭 링 · 시작점 제외 (라벨 없음)"),
}


def spring_start_angle(gray: np.ndarray, r_in: float = 0.62, r_out: float = 0.85):
    """스프링 시작점 각도(우리 정의)를 찾습니다. gray: 부품 원 중심에 맞춰 자른 정사각형 흑백 이미지.

    연마면 안쪽 링(바깥 원 반지름의 r_in~r_out)을 각도마다 밝기를 재면,
    스프링 시작점 부근은 연마되지 않아 어둡습니다(어두운 부채꼴).
    가장 길게 이어지는 어두운 구간을 찾고, 그 양 끝 중 밝기가 '급하게' 바뀌는 쪽을 시작점(0°)으로,
    어두운 구간이 이어지는 방향을 + 방향으로 정합니다.

    반환: (시작 각도°, 방향 +1/-1, 어두운 구간 폭°)  — 각도는 이미지 좌표(x 오른쪽 0°, 시계 방향 +)
    """
    import cv2

    S = gray.shape[0]
    c, R = S / 2, S / 2 / 1.1  # 자를 때 반폭 = 반지름 × 1.1
    ang = np.deg2rad(np.arange(360))
    rs = np.linspace(r_in * R, r_out * R, 12)
    xs = (c + np.outer(rs, np.cos(ang))).astype(np.float32)
    ys = (c + np.outer(rs, np.sin(ang))).astype(np.float32)
    raw = np.median(cv2.remap(gray.astype(np.float32), xs, ys, cv2.INTER_LINEAR), axis=0)
    smooth = np.convolve(np.r_[raw[-7:], raw, raw[:7]], np.ones(15) / 15, "valid")
    hi, lo = np.percentile(smooth, 90), smooth.min()
    dark = smooth < lo + 0.4 * (hi - lo)

    d2, best, i = np.r_[dark, dark], (0, 0), 0
    while i < 720:  # 원형으로 이어진 가장 긴 어두운 구간
        if d2[i]:
            j = i
            while j < 720 and d2[j] and j - i < 360:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    a, b = best[0] % 360, (best[1] - 1) % 360
    q = np.convolve(np.r_[raw[-2:], raw, raw[:2]], np.ones(5) / 5, "valid")
    jump_a = q[(a - 6) % 360] - q[(a + 6) % 360]  # 밝음 → 어두움 (a에서)
    jump_b = q[(b + 6) % 360] - q[(b - 6) % 360]  # 밝음 → 어두움 (b에서)
    return (a, +1, best[1] - best[0]) if jump_a >= jump_b else (b, -1, best[1] - best[0])


def sector_mask(size: int, start: float, direction: int, deg_from: float = -5, deg_to: float = 100) -> np.ndarray:
    """시작점 기준 deg_from~deg_to 부채꼴 = 1, 나머지 0. (size, size) uint8 0/255"""
    import cv2

    t0, t1 = (start + deg_from, start + deg_to) if direction > 0 else (start - deg_to, start - deg_from)
    m = np.zeros((size, size), np.uint8)
    cv2.ellipse(m, (size // 2, size // 2), (size, size), 0, t0, t1, 255, -1)
    return m


def mask_path(data_root: Path, category: str, image_path: Path) -> Path:
    return Path(data_root) / category / MASK_DIR / f"{Path(image_path).stem}.png"


def load_edge_mask(path: Path, size: int) -> np.ndarray:  # noqa: D401
    """(size, size) float32 0/1. 파일이 없으면 FileNotFoundError."""
    buf = np.fromfile(str(path), dtype=np.uint8) if Path(path).is_file() else None
    mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf is not None else None
    if mask is None:
        raise FileNotFoundError(f"테두리 마스크가 없습니다: {path}  (make_edge_masks.py를 먼저 실행하세요)")
    if mask.shape != (size, size):
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


def masked_score(anomaly_map: np.ndarray, mask: np.ndarray, kernel: int = 21) -> tuple[float, np.ndarray]:
    """마스크 밖을 0으로 만든 결함 맵과, 원본과 같은 방식(평균 풀링 후 최댓값)의 점수."""
    amap = np.asarray(anomaly_map, dtype=np.float32) * mask
    # torch avg_pool2d(padding=k//2, count_include_pad=True)와 같게: 바깥은 0으로 채우고 평균
    pooled = cv2.boxFilter(amap, -1, (kernel, kernel), normalize=True, borderType=cv2.BORDER_CONSTANT)
    return float(pooled.max()), amap
