"""DTD(Describable Textures Dataset)를 Hugging Face 미러에서 받아 DRAEM용 폴더로 저장.

DTD 공식 사이트(robots.ox.ac.uk)에 접속이 안 될 때 사용합니다.
결과: data/raw/dtd/images/<텍스처 이름>/*.jpg  (47개 폴더, 총 5,640장)

공용 .venv를 건드리지 않도록, 시스템 Python에 datasets만 따로 설치해서 실행하는 것을 권장합니다.
    py -3.11 -m pip install --user datasets pillow
    py -3.11 scripts/draem/download_dtd_hf.py

다른 미러를 쓰려면:
    py -3.11 scripts/draem/download_dtd_hf.py --repo Voxel51/Describable-Textures-Dataset
"""

from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="DTD를 Hugging Face 미러에서 받아 폴더로 저장")
    parser.add_argument("--repo", default="cansa/Describable-Textures-Dataset-DTD")
    parser.add_argument("--out", type=Path, default=Path("data/raw/dtd/images"))
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"이미 파일이 있습니다: {out}  (지우거나 --out을 바꾸세요)")

    from datasets import ClassLabel, Image, load_dataset

    print(f"다운로드: {args.repo} (약 600MB, 처음 한 번만 받음)")
    ds = load_dataset(args.repo, split="train")

    # 이미지 / 라벨 컬럼 자동으로 찾기 (미러마다 이름이 다를 수 있음)
    image_col = next(k for k, f in ds.features.items() if isinstance(f, Image))
    label_col = next((k for k, f in ds.features.items() if isinstance(f, ClassLabel)), None)
    names = ds.features[label_col].names if label_col else None
    print(f"컬럼: 이미지 '{image_col}', 라벨 '{label_col}' · 이미지 {len(ds)}장 · 클래스 {len(names) if names else '?'}개")

    counts = {}
    for i, ex in enumerate(ds):
        cls = names[ex[label_col]] if names else "texture"
        folder = out / cls
        folder.mkdir(parents=True, exist_ok=True)
        ex[image_col].convert("RGB").save(folder / f"{cls}_{i:04d}.jpg", quality=95)
        counts[cls] = counts.get(cls, 0) + 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(ds)}")

    print(f"\n완료: {out}")
    print(f"폴더 {len(counts)}개 · 이미지 {sum(counts.values())}장 (공식 DTD: 47개 · 5,640장)")
    if len(counts) != 47 or sum(counts.values()) != 5640:
        print("※ 공식 구성과 개수가 다릅니다. DRAEM 학습에는 문제없지만 발표에는 실제 장수를 적으세요.")


if __name__ == "__main__":
    main()
