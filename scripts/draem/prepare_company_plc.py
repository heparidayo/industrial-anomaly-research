"""회사 PLC 데이터 → MVTec 형식 변환 (+ 부품 원 기준 ROI 자르기).

입력 폴더 구조 (회사 제공 '데세 라벨링'):
    데세 라벨링/
    ├── 정상/*.png
    ├── over/images/train/*.png      over/labels/train/*.txt
    ├── step/images/train/*.png      step/labels/train/*.txt
    └── 미흡/images/train/*.png      미흡/labels/train/*.txt

    - 파일 이름 끝의 _camera_1 / _camera_2 로 카메라를 구분합니다.
    - txt는 YOLO 세그멘테이션 형식입니다. class 1 = 부품 바깥 원(정원), class 0 = 가공면 경계.
      결함 위치가 아니므로 정답은 폴더 이름으로 정합니다.

출력 구조:
    data/processed/company_plc/
    ├── camera_1/  train/good · test/good · test/over · test/step · test/insufficient · manifest.csv
    ├── camera_2/  (같은 구조)
    ├── cam12/     (camera_1 + camera_2 합침)
    └── _check/    원 검출 확인용 그림

ROI: 정상 이미지의 txt도 같은 부품의 결함 폴더 labels/에 들어 있으므로(카메라 2대 모두 라벨링),
모든 이미지를 **같은 방법(txt의 바깥 원)**으로 자릅니다. txt가 없으면 Hough 원 검출로 대신합니다.
(정상과 불량을 다른 방법으로 자르면 자르는 방식 차이를 모델이 학습할 수 있음)
--roi hough 로 모든 이미지를 원 검출로 자를 수도 있습니다.

예시 (Windows PowerShell, 프로젝트 루트):
    # 1) 원 검출만 확인 (파일을 만들지 않음)
    .\\.venv\\Scripts\\python.exe scripts/draem/prepare_company_plc.py --src "D:\\data\\데세 라벨링" --check-only

    # 2) 변환 (정상 카메라당 학습 10장, 나머지 테스트)
    .\\.venv\\Scripts\\python.exe scripts/draem/prepare_company_plc.py --src "D:\\data\\데세 라벨링"

    # PatchCore와 같은 학습 정상 10장을 쓰려면 파일 이름 목록(한 줄에 하나)을 지정
    .\\.venv\\Scripts\\python.exe scripts/draem/prepare_company_plc.py --src "..." --train-list train_normals.txt
"""

from pathlib import Path
import argparse
import csv
import random
import re
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 폴더 이름 → 출력 결함 종류 (한글 폴더명은 영문으로)
DEFECT_DIRS = {"over": "over", "step": "step", "미흡": "insufficient"}
NORMAL_DIR = "정상"
CAMERA_RE = re.compile(r"_camera_(\d+)$")

MANIFEST_FIELDS = [
    "image_id", "source_image", "split", "label", "defect_type", "camera", "part_id", "part_folder",
    "roi_cx", "roi_cy", "roi_r", "roi_method", "txt_cx", "txt_cy", "txt_r",
]


# ── 파일 입출력 (Windows 한글 경로 대응) ─────────────────────────
# cv2.imread / imwrite는 Windows에서 한글이 들어간 경로를 열지 못하므로 바이트로 읽고 씁니다.

def imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"PNG 인코딩 실패: {path}")
    buf.tofile(str(path))


# ── 라벨 / 원 ────────────────────────────────────────────────

def read_label_circle(txt_path: Path, width: int, height: int):
    """txt의 class 1 다각형(바깥 원)에서 중심과 반지름(px)을 구합니다. 없으면 None."""
    if not txt_path.is_file():
        return None
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7 or values[0] != "1":
            continue
        pts = np.array(values[1:], dtype=float).reshape(-1, 2) * [width, height]
        center = pts.mean(axis=0)
        radius = np.hypot(*(pts - center).T).mean()
        return float(center[0]), float(center[1]), float(radius)
    return None


def detect_circle(image: np.ndarray, r_min: int, r_max: int):
    """Hough 원 검출. 가장 강한 후보 1개 (cx, cy, r) 또는 None."""
    gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20,
        param1=100, param2=40, minRadius=r_min, maxRadius=r_max,
    )
    if circles is None:
        return None
    cx, cy, r = circles[0][0]
    return float(cx), float(cy), float(r)


def crop_square(image: np.ndarray, cx: float, cy: float, half: int) -> np.ndarray:
    """(cx, cy) 중심의 2*half 정사각형. 이미지 밖은 가장자리 픽셀로 채움."""
    h, w = image.shape[:2]
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return image[y0:y1, x0:x1]


# ── 수집 ────────────────────────────────────────────────────

def camera_of(path: Path) -> str:
    m = CAMERA_RE.search(path.stem)
    return f"camera_{m.group(1)}" if m else "camera_unknown"


PART_RE = re.compile(r"-(\d+)_camera_\d+$")


def part_id_of(path: Path) -> str:
    m = PART_RE.search(path.stem)
    return m.group(1) if m else ""


def collect(src: Path):
    """[{path, label, defect_type, camera, txt, part_folder}] 목록.

    회사 데이터는 부품 1개를 camera_1 · camera_2로 찍고, 라벨 txt는 부품의 결함 폴더에
    두 카메라 것이 모두 들어 있습니다. (예: 정상/…000051_camera_2.png 의 txt는
    step/labels/train/…000051_camera_2.txt) 그래서 모든 labels 폴더에서 파일 이름으로 찾습니다.
    """
    label_index = {}
    for folder in DEFECT_DIRS:
        for t in (src / folder / "labels").rglob("*.txt"):
            label_index[t.stem] = (t, folder)

    items = []
    normal_dir = src / NORMAL_DIR
    for p in sorted(normal_dir.rglob("*.png")):
        txt, part_folder = label_index.get(p.stem, (None, ""))
        items.append({"path": p, "label": 0, "defect_type": "good",
                      "camera": camera_of(p), "txt": txt, "part_folder": part_folder})

    for folder, defect_type in DEFECT_DIRS.items():
        img_dir = src / folder / "images" / "train"
        for p in sorted(img_dir.glob("*.png")):
            txt, _ = label_index.get(p.stem, (None, ""))
            items.append({"path": p, "label": 1, "defect_type": defect_type,
                          "camera": camera_of(p), "txt": txt, "part_folder": folder})
    return items


def split_normals(normals, n_train: int, seed: int, train_names: set | None):
    """카메라별로 학습 n_train장, 나머지 테스트. train_names가 있으면 그 목록을 따름."""
    by_cam = {}
    for it in normals:
        by_cam.setdefault(it["camera"], []).append(it)

    for cam, group in sorted(by_cam.items()):
        group.sort(key=lambda it: it["path"].name)
        if train_names is not None:
            train = [it for it in group if it["path"].name in train_names or it["path"].stem in train_names]
        else:
            order = group[:]
            random.Random(f"{seed}-{cam}").shuffle(order)
            train = order[:n_train]
        train_ids = {id(it) for it in train}
        for it in group:
            it["split"] = "train" if id(it) in train_ids else "test"
        print(f"  {cam}: 정상 {len(group)}장 → train {len(train)} / test {len(group) - len(train)}")


# ── 메인 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="회사 PLC 데이터 → MVTec 형식 변환")
    parser.add_argument("--src", type=Path, required=True, help="'데세 라벨링' 폴더")
    parser.add_argument("--dst", type=Path, default=Path("data/processed/company_plc"))
    parser.add_argument("--n-train", type=int, default=10, help="카메라당 학습 정상 장수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-list", type=Path, default=None,
                        help="학습 정상 파일 이름 목록 (PatchCore와 맞출 때)")
    parser.add_argument("--margin", type=float, default=1.1, help="자를 정사각형 반폭 = 반지름 × margin")
    parser.add_argument("--r-min", type=int, default=None, help="원 반지름 하한(px). 기본: txt 원에서 자동")
    parser.add_argument("--r-max", type=int, default=None, help="원 반지름 상한(px). 기본: txt 원에서 자동")
    parser.add_argument("--roi", choices=["txt", "hough"], default="txt",
                        help="ROI 원: txt(라벨 원, 기본) 또는 hough(원 검출)")
    parser.add_argument("--no-crop", action="store_true", help="ROI 자르기 없이 원본 크기로 저장")
    parser.add_argument("--check-only", action="store_true", help="원 검출 확인만 하고 파일은 만들지 않음")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst if args.dst.is_absolute() else ROOT / args.dst
    if not src.is_dir():
        parser.error(f"입력 폴더가 없습니다: {src}")

    items = collect(src)
    if not items:
        parser.error("이미지를 찾지 못했습니다. 폴더 구조를 확인하세요.")

    # 1. 이미지 크기와 txt 원 읽기
    for it in items:
        h, w = imread(it["path"]).shape[:2]
        it["size"] = (w, h)
        it["txt_circle"] = read_label_circle(it["txt"], w, h) if it["txt"] else None

    # 2. 검출 반지름 범위: txt 원 반지름 범위 ± 20px
    radii = [it["txt_circle"][2] for it in items if it["txt_circle"]]
    r_min = args.r_min or (int(min(radii)) - 20 if radii else 300)
    r_max = args.r_max or (int(max(radii)) + 20 if radii else 420)

    print("=== 회사 PLC 데이터 ===")
    for label_name in ["good", *DEFECT_DIRS.values()]:
        for cam in sorted({it["camera"] for it in items}):
            n = sum(1 for it in items if it["defect_type"] == label_name and it["camera"] == cam)
            if n:
                print(f"  {label_name:13s} {cam}: {n}장")
    print(f"원 검출 반지름 범위: {r_min}~{r_max}px (txt 원 {len(radii)}개 기준)")

    n_txt = sum(1 for it in items if it["txt_circle"])
    print(f"txt 원이 있는 이미지: {n_txt}/{len(items)}장 (정상 포함)")

    # 3. ROI 원 정하기
    #    --roi txt  : txt 원 사용 (사람이 라벨링한 값), 없으면 원 검출
    #    --roi hough: 모든 이미지를 원 검출, 실패하면 txt 원
    #    어느 쪽이든 원 검출 결과를 txt 원과 비교해 오차를 보고합니다.
    errors = []
    for it in items:
        image = imread(it["path"])
        det = detect_circle(image, r_min, r_max)
        txt = it["txt_circle"]

        if args.roi == "txt" and txt is not None:
            it["roi"], it["roi_method"] = txt, "txt"
        elif det is not None:
            it["roi"], it["roi_method"] = det, "hough"
        elif txt is not None:
            it["roi"], it["roi_method"] = txt, "txt_fallback"
        else:
            w, h = it["size"]
            it["roi"], it["roi_method"] = (w / 2, h / 2, (r_min + r_max) / 2), "center_fallback"

        if det is not None and txt is not None:
            errors.append((np.hypot(det[0] - txt[0], det[1] - txt[1]), abs(det[2] - txt[2]), it))
        elif det is None:
            print(f"  [원 검출 실패] {it['path'].name}")

    methods = {}
    for it in items:
        methods[it["roi_method"]] = methods.get(it["roi_method"], 0) + 1
    print(f"\nROI 방식: " + " · ".join(f"{k} {v}장" for k, v in methods.items()))
    fallbacks = [it for it in items if it["roi_method"] not in ("txt", "hough")]

    if errors:
        center_err = np.array([e[0] for e in errors])
        radius_err = np.array([e[1] for e in errors])
        print(f"원 검출의 txt 원 대비 오차 ({len(errors)}장): 중심 중앙값 {np.median(center_err):.1f}px · 최대 {center_err.max():.1f}px, "
              f"반지름 중앙값 {np.median(radius_err):.1f}px · 최대 {radius_err.max():.1f}px")
        bad = [e for e in errors if e[0] > 15 or e[1] > 15]
        for c_err, r_err, it in bad:
            print(f"  [오차 큼] {it['path'].name}: 중심 {c_err:.1f}px, 반지름 {r_err:.1f}px")

    # 확인용 그림: 정상 전부 + 불량 중 오차 큰 것
    check_dir = dst / "_check"
    check_targets = [it for it in items if it["label"] == 0] + [e[2] for e in errors if e[0] > 15 or e[1] > 15] + fallbacks
    for it in check_targets:
        vis = imread(it["path"])
        cx, cy, r = it["roi"]
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), 3)
        if it["txt_circle"]:
            tx, ty, tr = it["txt_circle"]
            cv2.circle(vis, (int(tx), int(ty)), int(tr), (0, 255, 0), 2)
        imwrite(check_dir / f"{it['defect_type']}_{it['path'].stem}.png", cv2.resize(vis, None, fx=0.5, fy=0.5))
    print(f"확인용 그림 {len(check_targets)}장: {check_dir}  (빨강 = 검출 원, 초록 = txt 원)")

    if args.check_only:
        print("\n--check-only: 변환은 하지 않았습니다.")
        return

    # 4. 분할
    print("\n=== 분할 ===")
    train_names = None
    if args.train_list is not None:
        train_names = {ln.strip() for ln in args.train_list.read_text(encoding="utf-8").splitlines() if ln.strip()}
    normals = [it for it in items if it["label"] == 0]
    split_normals(normals, args.n_train, args.seed, train_names)
    for it in items:
        if it["label"] == 1:
            it["split"] = "test"

    # 5. 저장: 카메라별 + cam12
    datasets = {cam: [it for it in items if it["camera"] == cam] for cam in sorted({it["camera"] for it in items})}
    if len(datasets) > 1:
        datasets["cam12"] = items

    for name in datasets:
        if (dst / name).exists():
            parser.error(f"이미 출력 폴더가 있습니다: {dst / name}  (지우거나 --dst를 바꾸세요)")

    print("\n=== 저장 ===")
    for name, group in datasets.items():
        out_root = dst / name
        rows = []
        for it in group:
            image = imread(it["path"])
            cx, cy, r = it["roi"]
            if not args.no_crop:
                image = crop_square(image, cx, cy, int(round(r * args.margin)))
            sub = "train/good" if it["split"] == "train" else f"test/{it['defect_type']}"
            out_path = out_root / sub / it["path"].name
            imwrite(out_path, image)

            txt = it["txt_circle"] or ("", "", "")
            rows.append({
                "image_id": out_path.relative_to(ROOT).as_posix() if out_path.is_relative_to(ROOT) else out_path.as_posix(),
                "source_image": it["path"].relative_to(src).as_posix(),
                "split": it["split"], "label": it["label"], "defect_type": it["defect_type"],
                "camera": it["camera"],
                "part_id": part_id_of(it["path"]), "part_folder": it["part_folder"],
                "roi_cx": round(cx, 1), "roi_cy": round(cy, 1), "roi_r": round(r, 1),
                "roi_method": it["roi_method"] if not args.no_crop else "none",
                "txt_cx": round(txt[0], 1) if txt[0] != "" else "",
                "txt_cy": round(txt[1], 1) if txt[1] != "" else "",
                "txt_r": round(txt[2], 1) if txt[2] != "" else "",
            })

        with open(out_root / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        counts = {}
        for row in rows:
            key = "train/good" if row["split"] == "train" else f"test/{row['defect_type']}"
            counts[key] = counts.get(key, 0) + 1
        print(f"  {name}: " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    print(f"\n완료: {dst}")


if __name__ == "__main__":
    main()
