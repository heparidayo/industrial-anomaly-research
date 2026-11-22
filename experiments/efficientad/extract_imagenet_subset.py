"""
Kaggle ImageNet zip에서 "클래스마다 같은 수"의 학습 이미지를 골라 바로 추출합니다.
전체 압축 해제(약 160GB) 없이, 1,000개 클래스의 다양성을 유지한 부분 집합을 만듭니다.

입력: imagenet-object-localization-challenge.zip (Kaggle에서 받은 파일)
출력: <out>/train/<클래스ID>/*.JPEG  (클래스당 --per_class 장)
      <out>/val/*.JPEG              (--with_val을 주면 검증 세트 5만 장 전체)

용량 기준 (대략, JPEG 평균 약 110KB):
  --per_class 100  → 10만 장, 약 11GB
  --per_class 300  → 30만 장, 약 33GB
  --per_class 600  → 60만 장, 약 66GB

사용 예:
  python extract_imagenet_subset.py --zip D:\\...\\imagenet-object-localization-challenge.zip ^
      --out D:\\...\\data\\raw\\imagenet --per_class 300 --with_val
"""

import argparse
import random
import zipfile
from collections import defaultdict
from pathlib import Path

TRAIN_PREFIX = "ILSVRC/Data/CLS-LOC/train/"
VAL_PREFIX = "ILSVRC/Data/CLS-LOC/val/"


def main():
    ap = argparse.ArgumentParser(description="ImageNet 클래스 균형 부분 집합 추출")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_class", type=int, default=300, help="클래스당 학습 이미지 수")
    ap.add_argument("--with_val", action="store_true", help="검증 세트 5만 장도 함께 추출")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    rng = random.Random(args.seed)
    with zipfile.ZipFile(args.zip) as zf:
        print("압축 파일 목록 읽는 중...")
        by_class, val = defaultdict(list), []
        for info in zf.infolist():
            name = info.filename
            if not name.lower().endswith((".jpeg", ".jpg")):
                continue
            if name.startswith(TRAIN_PREFIX):
                wnid = name[len(TRAIN_PREFIX):].split("/")[0]
                by_class[wnid].append(info)
            elif name.startswith(VAL_PREFIX) and args.with_val:
                val.append(info)
        if not by_class:
            raise SystemExit(f"'{TRAIN_PREFIX}' 아래 이미지를 찾지 못했습니다. Kaggle ImageNet zip이 맞는지 확인하세요.")

        chosen = []
        for wnid in sorted(by_class):
            files = sorted(by_class[wnid], key=lambda i: i.filename)
            chosen += rng.sample(files, min(args.per_class, len(files)))
        jobs = [(i, out / "train" / i.filename[len(TRAIN_PREFIX):]) for i in chosen]
        jobs += [(i, out / "val" / Path(i.filename).name) for i in val]
        total_gb = sum(i.file_size for i, _ in jobs) / 1e9
        print(f"클래스 {len(by_class)}개 × 최대 {args.per_class}장 = 학습 {len(chosen)}장"
              + (f", 검증 {len(val)}장" if val else "") + f" → 약 {total_gb:.1f}GB")

        for n, (info, dst) in enumerate(jobs, 1):
            if not dst.resolve().is_relative_to(out):
                raise ValueError(f"출력 폴더 밖을 가리키는 항목: {info.filename}")
            if dst.exists() and dst.stat().st_size == info.file_size:
                continue                                  # 이어서 실행할 때 이미 받은 파일은 건너뜀
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dst, "wb") as f:
                f.write(src.read())
            if n % 5000 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}")
    print(f"완료: {out.resolve()}")


if __name__ == "__main__":
    main()
