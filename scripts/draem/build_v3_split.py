"""기존 정상 + 추가 정상을 합쳐 학습 / 기준값(val) / 테스트로 다시 나누기.

기존: 정상 학습 10 · 테스트 6 (모두 9/29), 불량은 테스트에만
추가: prepare_extra_normals.py로 자른 정상 (9/15 ~ 10/29)
→ 카메라마다 정상을 촬영 날짜별로 섞어서 학습 50% · val 20% · 테스트 30%로 나눔 (날짜가 고루 들어가게)
→ 불량 테스트 이미지는 기존 그대로 복사 (이전 결과와 비교 가능)
→ 기준값은 fold 모델 대신 '학습에 안 쓴 val 정상'을 최종 모델로 채점해서 정함 (threshold_from_val.py)

    python scripts/draem/build_v3_split.py

결과: data/processed/company_plc_v3/<camera>/{train/good, val/good, test/good, test/<불량유형>}
     data/processed/company_plc_v3/split.csv (어느 이미지가 어디로 갔는지)
"""

from pathlib import Path
import argparse
import csv
import random
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]


def date_of(p: Path) -> str:
    m = re.search(r"PLC-(\d{8})", p.stem)
    return m.group(1) if m else "unknown"


def main():
    parser = argparse.ArgumentParser(description="v3 데이터 분할")
    parser.add_argument("--old", type=Path, default=Path("data/processed/company_plc"))
    parser.add_argument("--extra", type=Path, default=Path("data/processed/company_plc_extra"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed/company_plc_v3"))
    parser.add_argument("--cameras", nargs="+", default=["camera_1", "camera_2"])
    parser.add_argument("--train", type=float, default=0.5)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    res = lambda p: p if p.is_absolute() else ROOT / p  # noqa: E731
    old, extra, dst = res(args.old), res(args.extra), res(args.dst)
    if dst.exists():
        raise SystemExit(f"이미 있습니다: {dst} (지우거나 --dst를 바꾸세요)")

    rows = []
    for cam in args.cameras:
        normals = sorted((old / cam / "train" / "good").glob("*.png")) + sorted((old / cam / "test" / "good").glob("*.png")) \
            + sorted((extra / cam / "good").glob("*.png"))
        by_date = {}
        for p in normals:
            by_date.setdefault(date_of(p), []).append(p)
        rng = random.Random(f"{args.seed}-{cam}")
        for d, ps in sorted(by_date.items()):
            rng.shuffle(ps)
            n = len(ps)
            n_tr = max(1, round(n * args.train)) if n >= 3 else n
            n_va = round(n * args.val) if n >= 3 else 0
            for i, p in enumerate(ps):
                split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
                out = dst / cam / split / "good" / p.name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)
                rows.append({"camera": cam, "split": split, "label": "good", "date": d, "file": p.name})
        for ddir in sorted((old / cam / "test").iterdir()):
            if ddir.is_dir() and ddir.name != "good":
                for p in sorted(ddir.glob("*.png")):
                    out = dst / cam / "test" / ddir.name / p.name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, out)
                    rows.append({"camera": cam, "split": "test", "label": ddir.name, "date": date_of(p), "file": p.name})

    with open(dst / "split.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for cam in args.cameras:
        c = lambda s, lab: sum(r["camera"] == cam and r["split"] == s and (r["label"] == lab) == (lab == "good")  # noqa: E731
                               for r in rows)
        print(f"{cam}: 학습 정상 {c('train', 'good')} · val 정상 {c('val', 'good')} · "
              f"테스트 정상 {c('test', 'good')} · 테스트 불량 {c('test', 'bad')}")
        dates = sorted({r['date'] for r in rows if r['camera'] == cam and r['label'] == 'good'})
        for d in dates:
            n = {s: sum(r['camera'] == cam and r['label'] == 'good' and r['date'] == d and r['split'] == s for r in rows)
                 for s in ('train', 'val', 'test')}
            print(f"   {d}: 학습 {n['train']} · val {n['val']} · 테스트 {n['test']}")
    print(f"→ {dst}")


if __name__ == "__main__":
    main()
