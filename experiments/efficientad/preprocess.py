"""
입력 전처리: 면취면 주변 자르기 + 이미지별 밝기 정규화 (논문에는 없는 선택 기능).

순서:  원본 → (1) 자르기 → (2) 256×256으로 축소 → (3) 밝기 정규화 → (이후 증강·ImageNet 정규화)

(1) 자르기 --crop
    none                 자르지 않음 (기본값, 이전과 동일)
    center:0.7           가운데 정사각형을 짧은 변의 70%만큼 자름
    box:x0,y0,x1,y1      원본 픽셀 좌표로 직접 지정 (예: box:200,180,1020,1000)
    auto:0.15            가장 큰 밝은 물체(스프링)를 찾아 정사각형으로 자르고 15% 여백.
                         찾지 못하면 자르지 않음. 반드시 미리보기로 확인하고 쓰세요.
    목적: 원본(예: 1216×1216)을 그대로 256으로 줄이면 면취 폭의 세부가 사라지므로,
          부품 주변만 남겨 축소 배율을 줄입니다.

(2) 밝기 정규화 --bright_norm
    none      하지 않음 (기본값)
    mean      이미지마다 평균 밝기를 127.5로 맞춤 (전체를 밝게/어둡게 이동만)
    meanstd   평균 127.5, 표준편차 64로 맞춤 (밝기와 대비를 모두 통일)
    목적: 촬영 날짜마다 다른 전체 밝기(예: 132 대 147)를 없애, 모델이 날짜가 아니라
          불량의 모양에 반응하게 합니다. 밝기 통계는 흑백 기준으로 구하고 세 채널에 똑같이 적용해
          색 비율은 유지합니다.

설정은 train.py가 모델 파일(meta)에 저장하고, predict.py가 자동으로 같은 설정을 씁니다.

미리보기 (내 컴퓨터에만 저장, 원본에 자를 영역을 빨간 상자로 표시하고 결과를 옆에 보여 줌):
  python preprocess.py --images <폴더> [<폴더> ...] --crop center:0.7 --bright_norm meanstd
"""

import numpy as np
from PIL import Image, ImageDraw

TARGET_MEAN, TARGET_STD = 127.5, 64.0


def _parse_crop(spec: str):
    spec = (spec or "none").strip().lower()
    if spec == "none":
        return ("none", None)
    kind, _, val = spec.partition(":")
    if kind == "center":
        f = float(val or 0.7)
        if not 0 < f <= 1:
            raise ValueError("center 비율은 0보다 크고 1 이하여야 합니다 (예: center:0.7)")
        return ("center", f)
    if kind == "box":
        nums = [int(float(v)) for v in val.split(",")]
        if len(nums) != 4 or nums[2] <= nums[0] or nums[3] <= nums[1]:
            raise ValueError("box는 x0,y0,x1,y1 (x1>x0, y1>y0) 형식이어야 합니다")
        return ("box", tuple(nums))
    if kind == "auto":
        return ("auto", float(val or 0.15))
    raise ValueError(f"알 수 없는 --crop 형식: {spec} (none / center:0.7 / box:x0,y0,x1,y1 / auto:0.15)")


def _auto_box(img: Image.Image, margin: float):
    """가장 큰 밝은 덩어리를 찾아 정사각형 상자로. 실패하면 None."""
    import cv2
    w, h = img.size
    scale = 512 / max(w, h)
    small = np.asarray(img.convert("L").resize((max(1, int(w * scale)), max(1, int(h * scale)))))
    blur = cv2.GaussianBlur(small, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = small.shape[0] * small.shape[1]
    cands = [c for c in contours if 0.02 * area_img < cv2.contourArea(c) < 0.95 * area_img]
    if not cands:
        return None
    x, y, bw, bh = cv2.boundingRect(max(cands, key=cv2.contourArea))
    cx, cy = (x + bw / 2) / scale, (y + bh / 2) / scale
    half = max(bw, bh) / scale / 2 * (1 + margin)
    return _clip_square(cx, cy, half, w, h)


def _clip_square(cx, cy, half, w, h):
    half = min(half, min(w, h) / 2)
    x0 = int(round(min(max(cx - half, 0), w - 2 * half)))
    y0 = int(round(min(max(cy - half, 0), h - 2 * half)))
    s = int(round(2 * half))
    return (x0, y0, x0 + s, y0 + s)


class Preprocess:
    def __init__(self, crop: str = "none", bright_norm: str = "none", size: int = 256):
        self.crop = crop or "none"
        self.bright_norm = bright_norm or "none"
        self.size = size
        self._crop = _parse_crop(self.crop)
        if self.bright_norm not in ("none", "mean", "meanstd"):
            raise ValueError("--bright_norm은 none / mean / meanstd 중 하나")

    @property
    def enabled(self) -> bool:
        return self._crop[0] != "none" or self.bright_norm != "none"

    def crop_box(self, img: Image.Image):
        w, h = img.size
        kind, val = self._crop
        if kind == "center":
            return _clip_square(w / 2, h / 2, min(w, h) * val / 2, w, h)
        if kind == "box":
            x0, y0, x1, y1 = val
            return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
        if kind == "auto":
            return _auto_box(img, val) or (0, 0, w, h)
        return (0, 0, w, h)

    def normalize_brightness(self, img: Image.Image) -> Image.Image:
        if self.bright_norm == "none":
            return img
        a = np.asarray(img, dtype=np.float32)
        gray = a.mean(axis=2)
        m = gray.mean()
        if self.bright_norm == "mean":
            out = a - m + TARGET_MEAN
        else:
            s = gray.std() + 1e-6
            out = (a - m) / s * TARGET_STD + TARGET_MEAN
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def __call__(self, img: Image.Image):
        """반환: (전처리된 size×size RGB 이미지, 원본에서 자른 상자 (x0, y0, x1, y1))."""
        img = img.convert("RGB")
        box = self.crop_box(img)
        if box != (0, 0, img.size[0], img.size[1]):
            img = img.crop(box)
        img = img.resize((self.size, self.size), Image.BILINEAR)
        return self.normalize_brightness(img), box

    def to_dict(self):
        return {"crop": self.crop, "bright_norm": self.bright_norm, "size": self.size}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(d.get("crop", "none"), d.get("bright_norm", "none"), d.get("size", 256))

    def __repr__(self):
        return f"Preprocess(crop={self.crop}, bright_norm={self.bright_norm})"


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="전처리 미리보기")
    ap.add_argument("--images", nargs="+", required=True, help="이미지 폴더 (여러 개 가능, 예: 정상 폴더와 불량 폴더)")
    ap.add_argument("--crop", default="none")
    ap.add_argument("--bright_norm", default="none")
    ap.add_argument("--n", type=int, default=6, help="보여 줄 이미지 수")
    ap.add_argument("--out", default="preprocess_preview.png")
    args = ap.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = []
    for d in args.images:
        files = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in exts)
        k = max(1, args.n // len(args.images))
        step = max(1, len(files) // k)
        paths += files[::step][:k]
    pre = Preprocess(args.crop, args.bright_norm)
    tile = 256
    sheet = Image.new("RGB", (tile * 2, tile * len(paths)), "white")
    print(f"{pre}")
    print(f"{'파일':<45} {'자른 상자':<24} {'밝기 전':>7} {'밝기 후':>7}")
    for i, p in enumerate(paths):
        orig = Image.open(p).convert("RGB")
        out, box = pre(orig)
        thumb = orig.resize((tile, tile), Image.BILINEAR)
        sx, sy = tile / orig.size[0], tile / orig.size[1]
        ImageDraw.Draw(thumb).rectangle([box[0] * sx, box[1] * sy, box[2] * sx - 1, box[3] * sy - 1],
                                        outline=(255, 0, 0), width=3)
        sheet.paste(thumb, (0, i * tile))
        sheet.paste(out, (tile, i * tile))
        before = np.asarray(orig.convert("L"), dtype=np.float32).mean()
        after = np.asarray(out.convert("L"), dtype=np.float32).mean()
        print(f"{p.name[:45]:<45} {str(box):<24} {before:7.1f} {after:7.1f}")
    sheet.save(args.out)
    print(f"미리보기 저장: {Path(args.out).resolve()}  (왼쪽: 원본+자를 영역, 오른쪽: 모델 입력)")
