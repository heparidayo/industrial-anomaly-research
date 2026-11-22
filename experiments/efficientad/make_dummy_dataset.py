"""
파이프라인 동작 확인용 가짜 데이터셋 생성기.

실제 데이터 없이 train_student.py 등을 끝까지 돌려 보기 위한 것입니다.
정상 이미지: 배경 위에 원형 부품 하나 (위치·밝기에 약간의 편차)
불량 이미지: 정상 이미지에 긁힘(선) 또는 얼룩(점)을 추가하고 마스크 저장

생성 구조 (train_student.py가 읽는 형식):
  <out>/<category>/train/good/*.png
  <out>/<category>/test/good/*.png
  <out>/<category>/test/scratch/*.png,  test/stain/*.png
  <out>/<category>/ground_truth/scratch/*_mask.png, ground_truth/stain/*_mask.png

사용 예 (프로젝트 루트에서):
  python make_dummy_dataset.py --out data/raw/dummy --category widget
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def normal_image(rng, size):
    h = w = size
    base = np.tile(np.linspace(90, 160, w)[None, :, None], (h, 1, 3))
    base = base + rng.normal(0, 3, base.shape)
    img = Image.fromarray(np.clip(base, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    cx, cy = size // 2 + rng.integers(-6, 7, 2)
    r = size // 4
    tone = int(rng.integers(190, 215))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(tone, tone, tone - 20), outline=(60, 60, 60), width=3)
    d.ellipse([cx - r // 3, cy - r // 3, cx + r // 3, cy + r // 3], fill=(70, 70, 80))
    return img


def add_defect(rng, img, kind, size):
    mask = Image.new("L", img.size, 0)
    di, dm = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    c = size // 2
    if kind == "scratch":
        x0, y0 = c + rng.integers(-size // 5, size // 5, 2)
        x1, y1 = x0 + rng.integers(20, 50), y0 + rng.integers(-30, 30)
        di.line([x0, y0, x1, y1], fill=(30, 30, 30), width=3)
        dm.line([x0, y0, x1, y1], fill=255, width=5)
    else:
        x, y = c + rng.integers(-size // 5, size // 5, 2)
        r = int(rng.integers(6, 14))
        di.ellipse([x - r, y - r, x + r, y + r], fill=(140, 90, 40))
        dm.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return img, mask


def main():
    ap = argparse.ArgumentParser(description="가짜 이상 탐지 데이터셋 생성")
    ap.add_argument("--out", default="data/raw/dummy")
    ap.add_argument("--category", default="widget")
    ap.add_argument("--n_train", type=int, default=60)
    ap.add_argument("--n_test_good", type=int, default=15)
    ap.add_argument("--n_test_defect", type=int, default=10, help="불량 유형별 장수")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out) / args.category

    def save(img, sub, name):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        img.save(d / name)

    for i in range(args.n_train):
        save(normal_image(rng, args.size), "train/good", f"{i:03d}.png")
    for i in range(args.n_test_good):
        save(normal_image(rng, args.size), "test/good", f"{i:03d}.png")
    for kind in ("scratch", "stain"):
        for i in range(args.n_test_defect):
            img, mask = add_defect(rng, normal_image(rng, args.size), kind, args.size)
            save(img, f"test/{kind}", f"{i:03d}.png")
            save(mask, f"ground_truth/{kind}", f"{i:03d}_mask.png")

    print(f"생성 완료: {root.resolve()}")
    print(f"  train/good {args.n_train}장, test/good {args.n_test_good}장, "
          f"불량 2종 × {args.n_test_defect}장 (+마스크)")


if __name__ == "__main__":
    main()
