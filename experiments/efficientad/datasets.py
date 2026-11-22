"""
EfficientAD 시나리오 학습용 데이터셋.

MVTec AD 형식의 폴더 구조를 가정합니다.
    <root>/<category>/train/good/*.png
    <root>/<category>/test/<good|불량유형>/*.png
    <root>/<category>/ground_truth/<불량유형>/*_mask.png
VisA나 MVTec LOCO도 이 구조로 정리하면 그대로 쓸 수 있습니다
(MVTec LOCO의 공식 평가는 별도 스크립트를 사용해야 합니다).

- 검증 세트: 맵 정규화용 "학습에 쓰지 않은 정상 이미지"가 필요하므로
  train/good에서 일정 비율을 떼어 냅니다. 비율(기본 10%)은 논문에 명시되어
  있지 않아 임의로 정한 값입니다.
- 모든 이미지는 256×256으로 리사이즈하고, 증강 이후에 ImageNet 평균·표준편차로
  정규화합니다(논문 Appendix A.1).
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from preprocess import Preprocess

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_normalize = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _images_in(folder: Path) -> list[Path]:
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"폴더가 없습니다: {folder}")
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def _load_resized(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((size, size), Image.BILINEAR)


def split_train_val(category_dir: str, val_ratio: float = 0.1, seed: int = 0):
    """train/good을 학습용과 검증용으로 나눔 (재현 가능하도록 시드 고정)."""
    paths = _images_in(Path(category_dir) / "train" / "good")
    if not paths:
        raise FileNotFoundError(f"{category_dir}/train/good 에 이미지가 없습니다")
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(paths) * val_ratio)))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def ae_augment(img: Image.Image) -> Image.Image:
    """오토인코더 학습용 증강 (Algorithm 1의 23~28행).
    밝기·대비·채도 중 하나를 골라 U(0.8, 1.2) 배로 조정."""
    factor = random.uniform(0.8, 1.2)
    choice = random.randint(1, 3)
    if choice == 1:
        return TF.adjust_brightness(img, factor)
    if choice == 2:
        return TF.adjust_contrast(img, factor)
    return TF.adjust_saturation(img, factor)


class GeoAug:
    """정상 편차 증강: 회전 + 작은 평행 이동 (논문에는 없는 선택 기능, 기본 꺼짐).

    rot_deg  : 최대 회전 각도. 각 이미지마다 [-rot_deg, +rot_deg]에서 무작위.
               원형 부품이 어떤 각도로든 놓일 수 있다면 180 (= 전체 회전).
    shift_px : 최대 이동 픽셀 (256×256 기준). 가로·세로 각각 [-shift_px, +shift_px].
               원본 이미지 기준 픽셀 수 p를 쓰려면 p × 256 / 원본 크기로 환산.
    prob     : 증강을 적용할 확률 (나머지는 원본 그대로).

    회전·이동 후 비는 가장자리는 검은색 대신 "반사"로 채웁니다(BORDER_REFLECT_101).
    검은 모서리는 테스트 이미지에 없는 패턴이라, 모델이 그것을 정상으로 배우면 안 되기 때문.
    확대·축소는 넣지 않습니다. 면취 폭을 바꿔 과다·미흡을 흉내 낼 수 있기 때문입니다.
    """

    def __init__(self, rot_deg: float = 0.0, shift_px: float = 0.0, prob: float = 1.0):
        self.rot_deg, self.shift_px, self.prob = rot_deg, shift_px, prob

    @property
    def enabled(self) -> bool:
        return self.rot_deg > 0 or self.shift_px > 0

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.enabled or random.random() >= self.prob:
            return img
        import cv2
        import numpy as np
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        angle = random.uniform(-self.rot_deg, self.rot_deg)
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)   # 배율 1.0 고정
        mat[0, 2] += random.uniform(-self.shift_px, self.shift_px)
        mat[1, 2] += random.uniform(-self.shift_px, self.shift_px)
        out = cv2.warpAffine(arr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(out)

    def __repr__(self):
        return f"GeoAug(rot_deg={self.rot_deg}, shift_px={self.shift_px}, prob={self.prob})"


class NormalImages(Dataset):
    """정상 이미지 목록. with_ae_aug=True이면 (Student–Teacher용, 오토인코더용)을 반환.

    논문 기본: Student–Teacher용은 증강 없음, 오토인코더용은 밝기·대비·채도 증강.
    geo_aug를 주면 회전·이동을 추가로 적용합니다 (geo_target으로 적용 대상 선택):
      "both" : 같은 회전·이동을 두 입력 모두에 적용 (오토인코더용은 그 위에 색 증강)
      "st"   : Student–Teacher용에만
      "ae"   : 오토인코더용에만
    """

    def __init__(self, paths, image_size: int = 256, with_ae_aug: bool = False,
                 geo_aug: GeoAug | None = None, geo_target: str = "both",
                 preprocess: Preprocess | None = None):
        self.paths = list(paths)
        self.image_size = image_size
        self.pre = preprocess if preprocess is not None else Preprocess(size=image_size)
        self.with_ae_aug = with_ae_aug
        self.geo_aug = geo_aug if geo_aug is not None and geo_aug.enabled else None
        if geo_target not in ("both", "st", "ae"):
            raise ValueError(f"geo_target은 both/st/ae 중 하나: {geo_target}")
        self.geo_target = geo_target

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img, _ = self.pre(Image.open(self.paths[idx]))        # 자르기 → 256 축소 → 밝기 정규화
        if self.geo_aug is None:
            st_img = ae_img = img
        elif self.geo_target == "both":
            st_img = ae_img = self.geo_aug(img)
        elif self.geo_target == "st":
            st_img, ae_img = self.geo_aug(img), img
        else:
            st_img, ae_img = img, self.geo_aug(img)
        if self.with_ae_aug:
            return _normalize(st_img), _normalize(ae_augment(ae_img))
        return _normalize(st_img)


class ImageNetPenalty(Dataset):
    """Pretraining penalty용 ImageNet 이미지 (Appendix A.1 마지막 항목).
    512×512로 리사이즈 → 확률 0.3으로 흑백 → 중앙 256×256 크롭."""

    def __init__(self, root: str, image_size: int = 256):
        self.paths = _images_in(Path(root))
        if not self.paths:
            raise FileNotFoundError(f"{root} 아래에서 이미지를 찾지 못했습니다")
        self.tf = transforms.Compose([                    # 논문: 512 리사이즈 → 256 크롭 (입력 크기의 2배 → 입력 크기)
            transforms.Resize((2 * image_size, 2 * image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomGrayscale(p=0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.tf(Image.open(self.paths[idx]).convert("RGB"))


class TestImages(Dataset):
    """테스트 이미지. (이미지, 라벨 0/1, 원본 크기 마스크, 원본 (H, W), 경로)를 반환."""

    def __init__(self, category_dir: str, image_size: int = 256, preprocess: Preprocess | None = None,
                 split: str = "test"):
        self.pre = preprocess if preprocess is not None else Preprocess(size=image_size)
        root = Path(category_dir)
        if not (root / split).is_dir():
            raise FileNotFoundError(f"평가 폴더가 없습니다: {root / split}")
        self.items = []
        for defect_dir in sorted((root / split).iterdir()):
            if not defect_dir.is_dir():
                continue
            label = 0 if defect_dir.name == "good" else 1
            for p in _images_in(defect_dir):
                mask = None
                if label == 1:
                    cand = root / "ground_truth" / defect_dir.name / f"{p.stem}_mask.png"
                    mask = cand if cand.exists() else None
                self.items.append((p, label, mask))
        if not self.items:
            raise FileNotFoundError(f"{category_dir}/{split} 에 이미지가 없습니다")
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, mask_path = self.items[idx]
        img, box = self.pre(Image.open(path))
        x = _normalize(img)
        h, w = box[3] - box[1], box[2] - box[0]                 # 자른 영역 크기 (자르지 않으면 원본 크기)
        if mask_path is not None:
            m = Image.open(mask_path).convert("L").crop(box)
            mask = (TF.pil_to_tensor(m)[0] > 0).to(torch.uint8)
        else:
            mask = torch.zeros(h, w, dtype=torch.uint8)
        return x, label, mask, (h, w), str(path)


if __name__ == "__main__":
    # 증강 미리보기: 원본 1장 + 증강 결과 N장을 한 이미지로 저장 (내 컴퓨터에만 저장됨)
    #   python datasets.py --images <train/good 폴더> --rot_deg 180 --shift_px 6 --out aug_preview.png
    import argparse
    ap = argparse.ArgumentParser(description="회전·이동 증강 미리보기")
    ap.add_argument("--images", required=True)
    ap.add_argument("--rot_deg", type=float, default=0.0)
    ap.add_argument("--shift_px", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--out", default="aug_preview.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    src = _load_resized(_images_in(Path(args.images))[0], 256)
    aug = GeoAug(args.rot_deg, args.shift_px)
    tiles = [src] + [aug(src) for _ in range(args.n)]
    sheet = Image.new("RGB", (256 * 4, 256 * ((len(tiles) + 3) // 4)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 4) * 256, (i // 4) * 256))
    sheet.save(args.out)
    print(f"{aug} → 미리보기 저장: {Path(args.out).resolve()} (왼쪽 위가 원본)")
