"""
산업체 스프링 데이터 → 카메라별 EfficientAD 폴더 구조 변환.

입력 (받은 데이터):
  <root>/정상/*.bmp
  <root>/over/images/train/*.bmp,  <root>/over/label/train/*
  <root>/step/...,  <root>/미흡/...   (하위 폴더 이름과 관계없이 모두 수집)

파일 이름 규칙: PLC-YYYYMMDD-HHMMSS-ffffff-NNNNNN_camera_1
  제품 ID = "_camera_" 앞부분,  카메라 번호 = 뒤 숫자
  제품 ID 안의 날짜·시각으로 촬영 순서를 알 수 있음

출력 (카메라마다 하나의 "카테고리"):
  <out>/cam1/train/good/      정상 중 학습용 (train.py가 여기서 다시 10%를 검증용으로 분리)
  <out>/cam1/test/good/       정상 중 테스트용
  <out>/cam1/test/over/ step/ insufficient/   불량 (불량 이미지는 전부 테스트용)
  <out>/cam2/...              같은 구조
  <out>/manifest.csv          원본 경로, 제품 ID, 카메라, 클래스, 분할
  <out>/split_config.json     실행 설정 기록

원칙:
  - 원본은 수정하지 않고 복사만 합니다.
  - 정상은 "제품 단위"로 나눕니다. 같은 제품의 camera_1과 camera_2는 항상 같은 쪽
    (학습 또는 테스트)에 들어가, 두 카메라 결합 평가에서 섞이지 않습니다.
  - 불량 폴더의 이미지는 YOLO 학습용 train 폴더에 있더라도 EfficientAD에서는
    전부 테스트용입니다 (EfficientAD는 정상만으로 학습).

분할 방식 (--split):
  time   : 촬영 시각 순으로 앞쪽을 학습, 뒤쪽을 테스트 (실제 운용처럼 "미래" 제품으로 평가)
  random : 제품 ID를 시드 고정으로 무작위 분할

사용 예:
  python prepare_company_data.py --root "D:\\data\\대세 라벨링" ^
      --out ..\\..\\data\\processed\\company_ad --test_ratio 0.2 --split time
"""

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NAME_RE = re.compile(r"^(?P<pid>.+)_camera_(?P<cam>\d+)$")
TIME_RE = re.compile(r"(\d{8})-(\d{6})-(\d{6})")
DEFAULT_CLASS_MAP = {"정상": "good", "over": "over", "step": "step", "미흡": "insufficient",
                     "누락": "missing", "과다": "over", "다단": "step"}


def parse_name(stem: str):
    m = NAME_RE.match(stem)
    return (m.group("pid"), int(m.group("cam"))) if m else (None, None)


def product_time(pid: str):
    m = TIME_RE.search(pid)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2) + m.group(3), "%Y%m%d%H%M%S%f")
    except ValueError:
        return None


def collect(root: Path, class_map: dict):
    """(원본 경로, 제품 ID, 카메라, 출력 클래스 이름, 원래 폴더 이름) 목록."""
    items, skipped = [], Counter()
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        out_cls = class_map.get(cls_dir.name)
        if out_cls is None and cls_dir.name.strip().startswith(("정상", "good", "normal")):
            out_cls = "good"                         # 정상1, 정상2처럼 나뉜 정상 폴더
            print(f"  [알림] '{cls_dir.name}' 폴더를 정상(good)으로 처리합니다")
        if out_cls is None:
            out_cls = re.sub(r"[^A-Za-z0-9_]+", "_", cls_dir.name).strip("_").lower() or "unknown"
            print(f"  [알림] '{cls_dir.name}' 폴더는 매핑이 없어 '{out_cls}'로 저장합니다")
        label_dirs = [cls_dir / n for n in ("label", "labels") if (cls_dir / n).is_dir()]
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() not in IMG_EXTS or any(d in p.parents for d in label_dirs):
                continue
            pid, cam = parse_name(p.stem)
            if pid is None:
                skipped["이름 규칙 불일치"] += 1
                continue
            items.append((p, pid, cam, out_cls, cls_dir.name))
    return items, skipped


def main():
    ap = argparse.ArgumentParser(description="산업체 데이터를 카메라별 EfficientAD 구조로 변환")
    ap.add_argument("--root", required=True, help="'대세 라벨링' 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test_ratio", type=float, default=0.2, help="정상 제품 중 테스트 비율")
    ap.add_argument("--split", choices=["time", "random"], default="time")
    ap.add_argument("--test_dates", default="",
                    help="이 날짜(YYYYMMDD, 쉼표로 여러 개)의 정상을 전부 테스트로 보내고 나머지 정상은 학습에 사용. "
                         "지정하면 --split·--test_ratio는 무시됨")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true", help="복사하지 않고 통계만 출력")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    if not root.is_dir():
        raise SystemExit(f"폴더가 없습니다: {root.resolve()}")
    if out.exists() and any(out.iterdir()) and not args.dry_run:
        raise SystemExit(f"출력 폴더가 비어 있지 않습니다: {out.resolve()} (다른 이름을 쓰거나 비워 주세요)")

    items, skipped = collect(root, DEFAULT_CLASS_MAP)
    if skipped:
        print(f"  [알림] 제외: {dict(skipped)}")

    # ------------------------------------------------------------ 정상 제품 분할
    normal_pids = sorted({pid for _, pid, _, c, _ in items if c == "good"})
    test_dates = {d.strip() for d in args.test_dates.split(",") if d.strip()}
    if test_dates:
        def pdate(pid):
            m = TIME_RE.search(pid)
            return m.group(1) if m else None
        test_pids = {pid for pid in normal_pids if pdate(pid) in test_dates}
        if not test_pids:
            raise SystemExit(f"--test_dates {sorted(test_dates)}에 해당하는 정상 제품이 없습니다")
        args.split = f"date({','.join(sorted(test_dates))})"
    elif args.split == "time":
        times = {pid: product_time(pid) for pid in normal_pids}
        if any(t is None for t in times.values()):
            raise SystemExit("제품 ID에서 시각을 읽지 못한 항목이 있어 --split random을 사용하세요")
        normal_pids.sort(key=lambda pid: times[pid])
        n_test = int(round(len(normal_pids) * args.test_ratio))
        test_pids = set(normal_pids[len(normal_pids) - n_test:])
    else:
        rng = random.Random(args.seed)
        shuffled = normal_pids[:]
        rng.shuffle(shuffled)
        test_pids = set(shuffled[:int(round(len(shuffled) * args.test_ratio))])

    # ------------------------------------------------------------ 배치 결정
    rows, counts = [], defaultdict(Counter)
    cams_per_product = defaultdict(set)
    seen = set()
    for src, pid, cam, cls, orig in items:
        key = (pid, cam, cls)
        if key in seen:
            counts["(중복)"][f"cam{cam}"] += 1
            continue
        seen.add(key)
        cams_per_product[(pid, cls)].add(cam)
        if cls == "good":
            split = "test" if pid in test_pids else "train"
        else:
            split = "test"
        dst = out / f"cam{cam}" / split / cls / src.name
        rows.append({"product_id": pid, "camera": cam, "class": cls, "original_folder": orig,
                     "split": split, "src": str(src), "dst": str(dst)})
        counts[f"cam{cam}"][f"{split}/{cls}"] += 1

    # ------------------------------------------------------------ 요약
    print(f"\n정상 제품 {len(normal_pids)}개 → 학습 {len(normal_pids) - len(test_pids)}, "
          f"테스트 {len(test_pids)} (분할: {args.split})")
    for cam_key in sorted(k for k in counts if k.startswith("cam")):
        print(f"\n■ {cam_key}")
        for k, v in sorted(counts[cam_key].items()):
            print(f"  {k:28s} {v:6d}장")
    if "(중복)" in counts:
        print(f"\n  [주의] 같은 제품·카메라·클래스 중복: {dict(counts['(중복)'])}")
    one_cam = Counter(cls for (pid, cls), cams in cams_per_product.items() if len(cams) < 2)
    if one_cam:
        print(f"  [주의] 카메라 한쪽만 있는 제품 (클래스별): {dict(one_cam)}")
    dup_cls = Counter()
    by_pid = defaultdict(set)
    for (pid, cls) in cams_per_product:
        by_pid[pid].add(cls)
    for pid, clss in by_pid.items():
        if len(clss) > 1:
            dup_cls["+".join(sorted(clss))] += 1
    if dup_cls:
        print(f"  [주의] 여러 클래스에 동시에 들어 있는 제품: {dict(dup_cls)}")

    if args.dry_run:
        print("\n(dry run: 복사하지 않음)")
        return

    # ------------------------------------------------------------ 복사·기록
    for r in rows:
        dst = Path(r["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["src"], dst)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cfg = dict(vars(args), n_normal_products=len(normal_pids), n_test_products=len(test_pids),
               created=datetime.now().isoformat(timespec="seconds"))
    (out / "split_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n복사 완료: {out.resolve()}  (manifest.csv, split_config.json 포함)")


if __name__ == "__main__":
    main()
