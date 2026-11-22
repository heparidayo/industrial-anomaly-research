"""회사 PLC 데이터용 테두리 마스크 만들기 (라벨 txt 기준).

라벨 txt (YOLO 세그멘테이션):
    class 1 = 부품 바깥 원
    class 0 = 부품 상면(가공면)의 바깥 경계

마스크 = (바깥 원 + outer-pad px) 안쪽  AND  (상면 경계에서 band px 안쪽까지 깎은 상면) 바깥  (기본 band 3px)
       = 부품 테두리 띠(바깥 원 ~ 흰 면 끝) + 경계선 오차 여유

prepare_company_plc.py와 같은 방식(바깥 원 중심, 반폭 = 반지름 × margin)으로 잘라
256×256 결함 맵과 같은 좌표로 저장합니다.

예시:
    .\\.venv\\Scripts\\python.exe scripts/draem/make_edge_masks.py --src "C:\\대세 라벨링"

결과:
    data/processed/company_plc/<category>/edge_masks/<이미지 이름>.png   (흰색 = 보는 영역)
    data/processed/company_plc/_check_edge/*.png                        확인용 (초록 = 보는 영역)
"""

from pathlib import Path
import argparse
import csv
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_company_plc import imread, imwrite  # noqa: E402  (한글 경로 대응 입출력)

sys.path.insert(0, str(ROOT))
from src.models.draem.edge_mask import sector_mask, spring_start_angle  # noqa: E402


def read_polygons(txt: Path, width: int, height: int) -> dict:
    polys = {}
    for line in txt.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) >= 7:
            polys[values[0]] = np.array(values[1:], dtype=float).reshape(-1, 2) * [width, height]
    return polys


def main():
    parser = argparse.ArgumentParser(description="라벨 txt로 테두리 마스크 만들기")
    parser.add_argument("--src", type=Path, required=True, help="'대세 라벨링' 폴더 (라벨 txt 위치)")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/company_plc"))
    parser.add_argument("--categories", nargs="+", default=["camera_1", "camera_2", "cam12"])
    parser.add_argument("--size", type=int, default=256, help="결함 맵 크기 (학습 image_size)")
    parser.add_argument("--margin", type=float, default=1.1, help="prepare_company_plc.py의 --margin과 같게")
    parser.add_argument("--band", type=int, default=3, help="상면(흰 면) 경계에서 안쪽으로 포함할 폭 (256 기준 px, 경계선 라벨 오차 정도)")
    parser.add_argument("--outer-pad", type=int, default=3, help="바깥 원에서 바깥쪽으로 포함할 폭 (256 기준 px)")
    parser.add_argument("--check", type=int, default=8, help="데이터셋마다 확인용 그림 장수")
    parser.add_argument("--spring-range", type=float, nargs=2, default=[-5, 100], metavar=("FROM", "TO"),
                        help="스프링 시작점 기준 제외 각도 (edge_masks_ex). 기본 -5 100")
    args = parser.parse_args()

    src = args.src.resolve()
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    labels = {t.stem: t for t in src.rglob("labels/**/*.txt")}
    print(f"라벨 txt {len(labels)}개 · band {args.band}px · outer-pad {args.outer_pad}px (결함 맵 {args.size}px 기준)")

    S = args.size
    kernel_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.band + 1, 2 * args.band + 1))
    kernel_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.outer_pad + 1, 2 * args.outer_pad + 1))
    check_dir = data_root / "_check_edge"

    for cat in args.categories:
        manifest = data_root / cat / "manifest.csv"
        if not manifest.is_file():
            print(f"[건너뜀] {cat}: manifest.csv가 없습니다.")
            continue
        rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
        out_dir = data_root / cat / "edge_masks"
        out_ex = data_root / cat / "edge_masks_ex"
        coverage, coverage_ex, missing, checked, angles = [], [], [], 0, []

        for row in rows:
            stem = Path(row["image_id"]).stem
            txt = labels.get(stem)
            if txt is None:
                missing.append(stem)
                continue
            src_img = src / row["source_image"]
            h, w = imread(src_img).shape[:2] if src_img.is_file() else (1216, 1216)
            polys = read_polygons(txt, w, h)
            if "0" not in polys or "1" not in polys:
                missing.append(stem)
                continue

            # prepare_company_plc.crop_square와 같은 좌표 변환
            cx, cy, r = float(row["roi_cx"]), float(row["roi_cy"]), float(row["roi_r"])
            half = int(round(r * args.margin))
            x0, y0 = int(round(cx)) - half, int(round(cy)) - half
            scale = S / (2 * half)
            to_mask = lambda p: np.round((p - [x0, y0]) * scale).astype(np.int32)  # noqa: E731

            outer = np.zeros((S, S), np.uint8)
            cv2.fillPoly(outer, [to_mask(polys["1"])], 255)
            outer = cv2.dilate(outer, kernel_out)
            face = np.zeros((S, S), np.uint8)
            cv2.fillPoly(face, [to_mask(polys["0"])], 255)
            face_inner = cv2.erode(face, kernel_in)
            mask = cv2.bitwise_and(outer, cv2.bitwise_not(face_inner))

            imwrite(out_dir / f"{stem}.png", mask)
            coverage.append(mask.mean() / 255)

            # 스프링 시작점 제외: 자른 이미지에서 시작점 각도를 찾아 -5°~100° 부채꼴을 마스크에서 뺌
            img_path = ROOT / row["image_id"]
            image = cv2.resize(imread(img_path), (S, S)) if img_path.is_file() else None
            if image is not None:
                gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                start, direction, dark_w = spring_start_angle(gray)
                sector = sector_mask(S, start, direction, *args.spring_range)
                mask_ex = cv2.bitwise_and(mask, cv2.bitwise_not(sector))
                angles.append({"image": Path(row["image_id"]).name, "split": row["split"],
                               "defect_type": row["defect_type"], "start_deg": start,
                               "direction": direction, "dark_width_deg": dark_w})
            else:
                mask_ex = mask
            imwrite(out_ex / f"{stem}.png", mask_ex)
            coverage_ex.append(mask_ex.mean() / 255)

            if checked < args.check and row["split"] == "test" and image is not None:
                vis = image.copy()
                vis[mask > 0] = (0.5 * vis[mask > 0] + [0, 127, 0]).astype(np.uint8)
                vis_ex = image.copy()
                vis_ex[mask_ex > 0] = (0.5 * vis_ex[mask_ex > 0] + [0, 127, 0]).astype(np.uint8)
                ov = vis_ex.copy()
                ov[sector > 0] = (0.6 * ov[sector > 0] + [0, 80, 102]).astype(np.uint8)
                imwrite(check_dir / f"{cat}_{row['defect_type']}_{stem}.png", np.hstack([image, vis, ov]))
                checked += 1

        print(f"  {cat}: 마스크 {len(coverage)}장 · 평균 면적 {np.mean(coverage) * 100:.1f}%"
              f" → 스프링 시작점 제외 {np.mean(coverage_ex) * 100:.1f}%"
              + (f" · 라벨 없음 {len(missing)}장: {missing[:3]}" if missing else ""))
        if angles:
            with open(data_root / cat / "spring_start.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(angles[0]))
                w.writeheader()
                w.writerows(angles)

    print(f"\n확인용 그림: {check_dir}  (원본 | 테두리 | 테두리 - 스프링 시작점 구간(노랑))")
    print("결과 폴더: edge_masks (테두리) · edge_masks_ex (스프링 시작점 제외) · spring_start.csv (찾은 각도)")
    print("다음: kfold_threshold.py (마스크 종류마다 threshold 계산) → make_report.py --mask edge / edge_ex")


if __name__ == "__main__":
    main()
