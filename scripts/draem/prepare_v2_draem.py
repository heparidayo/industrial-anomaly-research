"""팀 키트(company_v2) 분할을 그대로 DRAEM 학습 폴더로 옮긴다 (부품 중심으로 잘라서).

키트 결과(company_v2_cam12)는 1216×1216 원본이라 부품이 작고 날짜마다 위치·배율이 다르다.
그래서 키트가 이미지마다 찾은 중심·반지름(…_polarstrip/polar.csv)으로 기존 DRAEM 데이터와 똑같이
반폭 R × 1.1 정사각형으로 자른다 (make_ring_masks.py / threshold_from_val.py 가 이 크기를 가정).

분할은 키트 split.csv 그대로:
  train/good        ← 키트 train 정상
  val/good          ← 키트 validation 정상 (기준값용, threshold_from_val.py)
  test/good         ← 키트 test 정상
  test/<불량>        ← 키트 validation + test 불량 (DRAEM은 불량을 학습·기준값에 쓰지 않으므로
                        띠 두께 판정과 같은 불량 52장으로 평가)
  unused            ← 건너뜀

저장소 전처리(scripts/preprocessing/company, configs/datasets/company_v2*.yaml)를 먼저 실행해
data/processed/company 에 결과를 만든 뒤:

    python scripts/draem/prepare_v2_draem.py                                   # 고리 (잘라서)
    python scripts/draem/prepare_v2_draem.py --category company_v2_cam12_polarstrip --no-crop --out data/processed/company_v2_draem_strip
    python scripts/draem/prepare_v2_draem.py --category company_v2_cam12_polar --no-crop --out data/processed/company_v2_draem_stack

결과: data/processed/company_v2_draem/camera_1, camera_2  (+ split_v2.csv, _check/crops_<camera>.png)
"""

from pathlib import Path
import argparse
import csv
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "draem"))

from prepare_company_plc import crop_square, imread, imwrite  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="키트 분할 → DRAEM 폴더")
    parser.add_argument("--kit-root", type=Path, default=Path("data/processed/company"),
                        help="저장소 전처리 결과 폴더 (기본: data/processed/company)")
    parser.add_argument("--category", default="company_v2_cam12")
    parser.add_argument("--polar-category", default="company_v2_cam12_polarstrip",
                        help="중심·반지름이 든 polar.csv 가 있는 카테고리")
    parser.add_argument("--out", type=Path, default=Path("data/processed/company_v2_draem"))
    parser.add_argument("--margin", type=float, default=1.1)
    parser.add_argument("--no-crop", action="store_true",
                        help="자르지 않고 그대로 복사 (키트 띠·적층 카테고리용: --category company_v2_cam12_polarstrip 등)")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if not args.kit_root.is_absolute():
        args.kit_root = ROOT / args.kit_root

    src = args.kit_root / args.category
    with (src / "split.csv").open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["prepared_image"]]
    with (args.kit_root / args.polar_category / "polar.csv").open(encoding="utf-8-sig") as f:
        polar = {r["prepared_image"]: r for r in csv.DictReader(f)}
    if any(r["prepared_image"] not in polar for r in rows):
        raise SystemExit("polar.csv 에 없는 이미지가 있습니다. 키트 unwrap_dataset.py 를 먼저 실행하세요.")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"출력 폴더가 이미 있습니다 (덮어쓰지 않음): {out}")

    records, previews = [], {}
    for i, r in enumerate(rows, 1):
        p = polar[r["prepared_image"]]
        cx, cy, R = float(p["center_x"]), float(p["center_y"]), float(p["radius"])
        image = imread(src / r["prepared_image"])
        crop = image if args.no_crop else crop_square(image, cx, cy, int(round(R * args.margin)))
        if r["class"] == "good":
            part = {"train": "train", "validation": "val", "test": "test"}[r["split"]]
        else:
            part = "test"
        folder = {"미흡": "insufficient"}.get(r["class"], r["class"])   # cv2.imread 가 윈도우 한글 경로를 못 읽음
        dst = out / r["camera"] / part / folder / r["file"]
        imwrite(dst, crop)
        records.append({"camera": r["camera"], "kit_split": r["split"], "draem_split": part, "class": r["class"],
                        "group_id": r["group_id"], "file": r["file"],
                        "center_x": cx, "center_y": cy, "radius": R, "path": dst.relative_to(out).as_posix()})
        previews.setdefault((r["camera"], r["class"]), [])
        if len(previews[(r["camera"], r["class"])]) < 6:
            ph = 160 if crop.shape[0] >= crop.shape[1] // 2 else max(24, 640 * crop.shape[0] // crop.shape[1])
            pw = 160 if ph == 160 else 640
            previews[(r["camera"], r["class"])].append(cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_AREA))
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}")

    with (out / "split_v2.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    chk = out / "_check"
    chk.mkdir(exist_ok=True)
    for cam in sorted({r["camera"] for r in records}):
        lines = []
        for cls in ("good", "over", "step", "미흡"):
            tiles = previews.get((cam, cls), [])
            if tiles:
                tiles += [np.full_like(tiles[0], 255)] * (6 - len(tiles))
                lines.append(np.hstack(tiles) if tiles[0].shape[1] == 160 else np.vstack(tiles))
        cv2.imencode(".png", np.vstack(lines))[1].tofile(str(chk / f"crops_{cam}.png"))

    for cam in sorted({r["camera"] for r in records}):
        c = lambda part, cls=None: sum(1 for r in records if r["camera"] == cam and r["draem_split"] == part  # noqa: E731
                                       and (cls is None or (r["class"] == "good") == (cls == "good")))
        print(f"{cam}: train {c('train')} · val {c('val')} · test 정상 {c('test', 'good')} · test 불량 {c('test', 'bad')}")
    print(f"→ {out}   (확인: _check/crops_<camera>.png)")


if __name__ == "__main__":
    main()
