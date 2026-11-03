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

from .imgaug_compat import Autocontrast, ia, iaa
from .perlin import rand_perlin_2d_np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in Path(folder).glob(pattern) if p.suffix.lower() in IMAGE_EXTS)


def load_image(path: Path, image_size: int) -> np.ndarray:
    """(H, W, 3) float32, 0~1, BGR."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    image = cv2.resize(image, dsize=(image_size, image_size))
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
    ):
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

    def __len__(self):
        return len(self.image_paths)

    def _rand_augmenter(self):
        idx = np.random.choice(len(self.AUGMENTERS), 3, replace=False)
        return iaa.Sequential([self.AUGMENTERS[i] for i in idx])

    def _synthesize(self, image: np.ndarray, source_path: Path):
        """정상 이미지(0~1) 위에 텍스처 결함을 합성. 50% 확률로 결함 없음."""
        size = self.image_size

        # 50%는 결함 없는 샘플 (원본과 동일)
        if np.random.rand() > 0.5:
            return image, np.zeros((size, size, 1), np.float32), np.array([0.0], np.float32)

        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        source = cv2.resize(source, dsize=(size, size))
        source = self._rand_augmenter()(image=source)

        sx = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale)
        sy = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale)
        noise = rand_perlin_2d_np((size, size), (sx, sy), rng=np.random.default_rng(np.random.randint(2 ** 31)))
        noise = self.rot(image=noise)

        mask = (noise > self.threshold).astype(np.float32)[..., None]
        texture = source.astype(np.float32) * mask / 255.0

        beta = np.random.rand() * 0.8
        augmented = image * (1 - mask) + (1 - beta) * texture + beta * image * mask
        augmented = (mask * augmented + (1 - mask) * image).astype(np.float32)

        has_anomaly = 1.0 if mask.sum() > 0 else 0.0
        return augmented, mask, np.array([has_anomaly], np.float32)

    def __getitem__(self, index):
        image = cv2.imread(str(self.image_paths[index]), cv2.IMREAD_COLOR)
        image = cv2.resize(image, dsize=(self.image_size, self.image_size))

        # 정상 이미지 자체도 일정 확률로 회전 (원본: 30%)
        if np.random.rand() < self.rotate_normal_prob:
            image = self.rot(image=image)

        image = image.astype(np.float32) / 255.0
        source_path = self.anomaly_source_paths[np.random.randint(len(self.anomaly_source_paths))]
        augmented, mask, has_anomaly = self._synthesize(image, source_path)

        return {
            "image": to_tensor(image),
            "augmented_image": to_tensor(augmented),
            "anomaly_mask": to_tensor(mask),
            "has_anomaly": torch.from_numpy(has_anomaly),
            "idx": index,
        }
