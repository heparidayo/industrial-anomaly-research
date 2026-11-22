"""DRAEM 데이터셋.

- DRAEMTrainDataset: 정상 이미지에 DTD 텍스처 + Perlin 마스크로 가짜 결함을 합성
- load_image:        추론용 이미지 읽기 (학습과 같은 전처리)

이미지는 원본 구현과 같이 OpenCV(BGR)로 읽고 0~1 범위로 맞춥니다.
PatchCore처럼 ImageNet 정규화는 하지 않습니다.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .imgaug_compat import Autocontrast, apply_chain, ia, iaa
from .perlin import rand_perlin_2d_np
from .preprocess import POLAR_H, POLAR_W, exclude_center, preprocess, preprocess_polar, rotate

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in Path(folder).glob(pattern) if p.suffix.lower() in IMAGE_EXTS)


def imread(path: Path) -> np.ndarray | None:
    """cv2.imread 와 같지만 윈도우 한글 경로도 읽음."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def load_image(path: Path, image_size) -> np.ndarray:
    """(H, W, 3) float32, 0~1, BGR. image_size: 정사각형 한 변(int) 또는 (높이, 너비)."""
    image = imread(path)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    h, w = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    image = cv2.resize(image, dsize=(w, h))
    return image.astype(np.float32) / 255.0


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """(H, W, C) numpy → (C, H, W) tensor."""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 워커마다 numpy / imgaug 난수를 다르게, 그러나 재현 가능하게 설정."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    ia.seed(seed)


class DRAEMTrainDataset(Dataset):
    """정상 학습 이미지 → (원본, 결함 합성 이미지, 결함 마스크, 결함 여부)."""

    # 원본 DRAEM data_loader.py와 같은 augmenter 목록 (DTD 텍스처 변형용)
    AUGMENTERS = [
        iaa.GammaContrast((0.5, 2.0), per_channel=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
        iaa.pillike.EnhanceSharpness(),
        iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
        iaa.Solarize(0.5, threshold=(32, 128)),
        iaa.Posterize(),
        iaa.Invert(),
        Autocontrast(),  # NumPy 2 호환 대체 (원본: iaa.pillike.Autocontrast)
        iaa.pillike.Equalize(),
        iaa.Affine(rotate=(-45, 45)),
    ]

    def __init__(
        self,
        image_paths: list[Path],
        anomaly_source_paths: list[Path],
        image_size: int = 256,
        rotate_normal_prob: float = 0.3,
        rotate_degrees: float = 90,
        perlin_scale: int = 6,
        min_perlin_scale: int = 0,
        threshold: float = 0.5,
        clahe: bool = False,
        align: bool = False,
        band_dir: Path | None = None,
        band_dilate: int = 6,
        polar: bool = False,
        rect: tuple[int, int] | None = None,
    ):
        """clahe: 조명 정규화 · align: 스프링 시작점 정렬 (회전 증강은 ±rotate_degrees 미세 흔들기로 바뀜)
        band_dir: 테두리 마스크 폴더(edge_masks_ex 등). 주면 가짜 결함을 그 안(+band_dilate px)에만 합성.
        polar: 테두리 띠 펴기. 스프링 시작점 제외 구간을 잘라 낸 (POLAR_H, POLAR_W) 띠로 학습.
               회전 증강은 띠를 펼 때 각도를 ±rotate_degrees 흔드는 것으로 바뀜 (정렬은 항상 켜짐)."""
        if not image_paths:
            raise ValueError("정상 학습 이미지가 없습니다.")
        if not anomaly_source_paths:
            raise ValueError("이상 소스(DTD) 이미지가 없습니다.")

        self.image_paths = list(image_paths)
        self.anomaly_source_paths = list(anomaly_source_paths)
        self.image_size = image_size
        self.rotate_normal_prob = rotate_normal_prob
        self.perlin_scale = perlin_scale
        self.min_perlin_scale = min_perlin_scale
        self.threshold = threshold
        # 정상 이미지 회전 증강과 Perlin 마스크 회전에 함께 사용 (원본: ±90°)
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-rotate_degrees, rotate_degrees))])
        self.rotate_degrees = rotate_degrees
        self.clahe, self.align = clahe, align
        self.band_dir = Path(band_dir) if band_dir else None
        self.band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_dilate + 1,) * 2)
        self.polar = polar
        if polar and band_dir is not None:
            raise ValueError("polar와 band_dir는 함께 쓸 수 없습니다.")
        self.rect = tuple(rect) if rect else None   # 이미 펴진 직사각형 띠를 그대로 (높이, 너비)로 학습
        if self.rect and (polar or align or band_dir is not None):
            raise ValueError("rect 는 polar · align · band_dir 와 함께 쓸 수 없습니다.")
        self.h, self.w = (POLAR_H, POLAR_W) if polar else (self.rect or (image_size, image_size))
        self._cache = {}  # 경로 → (전처리된 이미지, 띠 마스크). 정렬 각도 계산을 매번 하지 않도록

    def __len__(self):
        return len(self.image_paths)

    def _rand_augmenters(self):
        """원본과 같이 10개 중 3개를 무작위로 골라 순서대로 적용."""
        idx = np.random.choice(len(self.AUGMENTERS), 3, replace=False)
        return [self.AUGMENTERS[i] for i in idx]

    def _load(self, path: Path):
        if path not in self._cache and self.polar:
            raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if raw is None:
                raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
            self._cache[path] = (raw, exclude_center(raw))  # 원본 해상도 + 제외 구간 가운데 각도
        if path not in self._cache and self.rect:
            raw = imread(path)
            if raw is None:
                raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
            self._cache[path] = (cv2.resize(raw, (self.w, self.h)), None)
        if path not in self._cache:
            raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
            image, angle = preprocess(raw, self.image_size, self.clahe, self.align)
            band = None
            if self.band_dir is not None:
                m = cv2.imread(str(self.band_dir / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    raise FileNotFoundError(f"테두리 마스크가 없습니다: {self.band_dir / (path.stem + '.png')}")
                m = cv2.resize(m, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                if angle:
                    m = rotate(m, angle, cv2.INTER_NEAREST)
                band = (cv2.dilate(m, self.band_kernel) > 127).astype(np.float32)
            self._cache[path] = (image, band)
        return self._cache[path]

    def _synthesize(self, image: np.ndarray, source_path: Path, band: np.ndarray | None = None):
        """정상 이미지(0~1) 위에 텍스처 결함을 합성. 50% 확률로 결함 없음. band가 있으면 그 안에만."""
        h, w = self.h, self.w

        # 50%는 결함 없는 샘플 (원본과 동일)
        if np.random.rand() > 0.5:
            return image, np.zeros((h, w, 1), np.float32), np.array([0.0], np.float32)

        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        source = cv2.resize(source, dsize=(h, h))
        if w != h:  # 가로로 긴 띠: 정사각형 텍스처를 가로로 이어 붙여 늘어나지 않게
            source = np.tile(source, (1, -(-w // h), 1))[:, :w]
        source = apply_chain(self._rand_augmenters(), source)  # NumPy 2 읽기 전용 배열 대응

        for _ in range(10):  # 띠 안에 결함이 생길 때까지 최대 10번
            # 띠(높이 128)는 원본 256보다 낮으므로 격자 수 상한을 한 단계 낮추고(칸 최소 8px),
            # 가로 격자 수는 가로/세로 비율만큼 늘려 결함 모양이 옆으로 늘어나지 않게
            top = self.perlin_scale - (1 if w != h else 0)
            sx = 2 ** np.random.randint(self.min_perlin_scale, top)
            sy = 2 ** np.random.randint(self.min_perlin_scale, top) * (w // h)
            noise = rand_perlin_2d_np((h, w), (sx, sy), rng=np.random.default_rng(np.random.randint(2 ** 31)))
            noise = self.rot(image=noise)
            mask = (noise > self.threshold).astype(np.float32)
            if band is not None:
                mask = mask * band
            if mask.sum() > 0:
                break
        mask = mask[..., None]
        texture = source.astype(np.float32) * mask / 255.0

        beta = np.random.rand() * 0.8
        augmented = image * (1 - mask) + (1 - beta) * texture + beta * image * mask
        augmented = (mask * augmented + (1 - mask) * image).astype(np.float32)

        has_anomaly = 1.0 if mask.sum() > 0 else 0.0
        return augmented, mask, np.array([has_anomaly], np.float32)

    def __getitem__(self, index):
        if self.polar:
            raw, center = self._load(self.image_paths[index])
            offset = (np.random.uniform(-self.rotate_degrees, self.rotate_degrees)
                      if np.random.rand() < self.rotate_normal_prob else 0.0)
            image, _ = preprocess_polar(raw, self.clahe, offset, center)
            band = None
        else:
            image, band = self._load(self.image_paths[index])

        # 정상 이미지 자체도 일정 확률로 회전 (원본: 30%). 띠 마스크도 같은 각도로
        if not self.polar and not self.rect and np.random.rand() < self.rotate_normal_prob:
            angle = np.random.uniform(-self.rotate_degrees, self.rotate_degrees)
            image = rotate(image, angle)
            if band is not None:
                band = rotate(band, angle, cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        source_path = self.anomaly_source_paths[np.random.randint(len(self.anomaly_source_paths))]
        augmented, mask, has_anomaly = self._synthesize(image, source_path, band)

        return {
            "image": to_tensor(image),
            "augmented_image": to_tensor(augmented),
            "anomaly_mask": to_tensor(mask),
            "has_anomaly": torch.from_numpy(has_anomaly),
            "idx": index,
        }
