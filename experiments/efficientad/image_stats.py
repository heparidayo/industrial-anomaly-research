"""
날짜·폴더·카메라별 이미지 통계 — 촬영 환경 차이를 숫자로 확인합니다.

prepare_company_data.py가 만든 manifest.csv를 읽어, 묶음마다
  - 장수, 원본 해상도
  - 평균 밝기(0~255)와 밝기 표준편차(대비)
  - 묶음 안에서 이미지마다의 평균 밝기가 얼마나 흔들리는지
를 출력합니다. 이미지 내용은 출력하지 않습니다.

해석:
  - 날짜 사이 평균 밝기 차이가 같은 날짜 안의 흔들림보다 확실히 크면 → 촬영 환경 차이가 있음
  - 날짜 사이 차이가 같은 날짜 안의 흔들림과 비슷하면 → 환경 차이는 작음

사용 예:
  python image_stats.py --manifest ..\\..\\data\\processed\\company_ad_d0929\\manifest.csv
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description="날짜·폴더·카메라별 이미지 통계")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max_per_group", type=int, default=60, help="묶음마다 최대 몇 장까지 읽을지")
    args = ap.parse_args()

    groups = defaultdict(list)
    with open(args.manifest, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            date = r["product_id"][4:12]
            groups[(r["camera"], r["original_folder"], date)].append(r["src"])

    print(f"{'카메라':>4} {'폴더':<8} {'날짜':<9} {'장수':>4} {'해상도':<11} "
          f"{'평균밝기':>7} {'대비':>6} {'이미지간흔들림':>10}")
    for key in sorted(groups):
        cam, folder, date = key
        paths = groups[key][: args.max_per_group]
        means, stds, sizes = [], [], set()
        for p in paths:
            with Image.open(p) as im:
                sizes.add(f"{im.size[0]}×{im.size[1]}")
                a = np.asarray(im.convert("L"), dtype=np.float32)
            means.append(a.mean())
            stds.append(a.std())
        size = next(iter(sizes)) if len(sizes) == 1 else f"{len(sizes)}종"
        print(f"{cam:>4} {folder:<8} {date:<9} {len(groups[key]):>4} {size:<11} "
              f"{np.mean(means):7.1f} {np.mean(stds):6.1f} {np.std(means):10.2f}")


if __name__ == "__main__":
    main()
