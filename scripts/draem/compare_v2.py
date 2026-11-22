"""재학습한 DRAEM(키트 분할) 결과와 띠 두께 판정 결과를 같은 표로 비교한다.

같은 test 정상 · 같은 불량(키트 validation+test 의 9/29 불량)으로:
  - AUROC (모든 날짜 정상 / 9/29 정상만)
  - 이미지 단위 과검률 · 종류별 검출
  - 부품 단위 (같은 촬영번호의 두 카메라 중 하나라도 걸리면 불량)
  - 날짜별 과검률, 판정 시간

    python scripts/draem/compare_v2.py ^
        --draem 고리 outputs/report/v2_ring <cam1 json> <cam2 json> threshold_ring_ex ^
        --draem 띠 outputs/report/v2_strip <cam1 json> <cam2 json> threshold ^
        --band-scores outputs/band_best/best_scores.csv --out outputs/report/compare_v2.md

결과: --out (기본 outputs/report/compare_v2.md)
"""

from pathlib import Path
import argparse
import collections
import csv
import json
import re

import numpy as np

CAMS = ("camera_1", "camera_2")
TYPES = ("over", "미흡", "step")


def auroc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(neg) * len(pos)))


def date_of(name):
    m = re.search(r"PLC-(\d{8})-", name)
    return f"{int(m[1][4:6])}/{int(m[1][6:8])}" if m else "?"


def date_key(d):
    return tuple(int(x) for x in d.split("/")) if "/" in d else (99, 99)


def load_draem(report: Path, thr_jsons, split_csv: Path, key: str = "threshold_ring_ex"):
    with split_csv.open(encoding="utf-8-sig") as f:
        meta = {(r["camera"], r["file"]): r for r in csv.DictReader(f)}
    recs, ms = [], []
    for cam, tj in zip(CAMS, thr_jsons):
        thr = float(json.loads(Path(tj).read_text(encoding="utf-8"))[key])
        with (report / cam / "predictions.csv").open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                name = Path(r["image_id"]).name
                m = meta[(cam, name)]
                s = float(r["anomaly_score"])
                ms.append(float(r["inference_ms"]))
                cls = {"insufficient": "미흡"}.get(r["defect_type"], r["defect_type"])
                recs.append({"camera": cam, "file": name, "class": cls, "group": m["group_id"],
                             "date": date_of(name), "score": s / thr, "pred": int(s > thr)})
    return recs, float(np.median(ms))


def load_band(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    recs = []
    for r in rows:   # DRAEM 과 같은 이미지만: test 정상 + validation·test 불량
        if r["class"] == "good" and r["split"] != "test":
            continue
        if r["class"] != "good" and r["split"] == "train":
            continue
        score = float(r["score"]) if r.get("score") else max(float(r["over_z"]), float(r["thin_z"]), float(r["step_z"]))
        recs.append({"camera": r["camera"], "file": r["file"], "class": r["class"], "group": r["group_id"],
                     "date": r["date"], "score": score, "pred": int(r["pred_defect"])})
    return recs


def summarize(name, recs, ms=None):
    L = [f"### {name}", "", "| 카메라 | AUROC 모든 날짜 | AUROC 9/29 정상만 | test 정상 과검 | over | 미흡 | step | 불량 전체 |",
         "|---|---|---|---|---|---|---|---|"]
    for cam in list(CAMS) + ["all"]:
        s = [r for r in recs if cam == "all" or r["camera"] == cam]
        g = [r for r in s if r["class"] == "good"]
        g9 = [r for r in g if r["date"] == "9/29"]
        d = [r for r in s if r["class"] != "good"]
        fa = sum(r["pred"] for r in g)
        det = [f"{sum(r['pred'] for r in d if r['class'] == t)}/{sum(1 for r in d if r['class'] == t)}" for t in TYPES]
        a_all = auroc([r["score"] for r in g], [r["score"] for r in d]) if cam != "all" else float("nan")
        a_929 = auroc([r["score"] for r in g9], [r["score"] for r in d]) if cam != "all" else float("nan")
        hit = sum(r["pred"] for r in d)
        L.append(f"| {cam} | {a_all:.3f} | {a_929:.3f} | {fa}/{len(g)} ({fa / max(len(g), 1):.1%}) | "
                 + " | ".join(det) + f" | {hit}/{len(d)} ({hit / max(len(d), 1):.0%}) |")
    groups = collections.defaultdict(list)
    for r in recs:
        groups[r["group"]].append(r)
    pg = [g for g in groups.values() if all(r["class"] == "good" for r in g)]
    pd = [g for g in groups.values() if any(r["class"] != "good" for r in g)]
    pfa = sum(any(r["pred"] for r in g) for g in pg)
    phit = sum(any(r["pred"] for r in g) for g in pd)
    by = {t: (sum(any(r["pred"] for r in g) for g in pd if any(r["class"] == t for r in g)),
              sum(1 for g in pd if any(r["class"] == t for r in g))) for t in TYPES}
    L += ["", f"- 부품 단위: 정상 부품 과검 {pfa}/{len(pg)} ({pfa / max(len(pg), 1):.1%}) · 불량 부품 검출 "
          f"{phit}/{len(pd)} ({phit / max(len(pd), 1):.0%}) · " + " · ".join(f"{t} {a}/{b}" for t, (a, b) in by.items())]
    if ms is not None:
        L.append(f"- 판정 시간(중앙값): {ms:.1f} ms/장")
    L += ["", "| 날짜 | camera_1 과검 | camera_2 과검 |", "|---|---|---|"]
    for dt in sorted({r["date"] for r in recs if r["class"] == "good"}, key=date_key):
        cells = []
        for cam in CAMS:
            g = [r for r in recs if r["class"] == "good" and r["camera"] == cam and r["date"] == dt]
            cells.append(f"{sum(r['pred'] for r in g)}/{len(g)}" if g else "—")
        L.append(f"| {dt} | " + " | ".join(cells) + " |")
    return L + [""]


def main():
    ap = argparse.ArgumentParser(description="DRAEM(키트 분할) vs 띠 두께 비교")
    ap.add_argument("--draem", nargs=5, action="append", required=True,
                    metavar=("이름", "REPORT", "TH_CAM1", "TH_CAM2", "KEY"),
                    help="DRAEM 결과 하나: 이름, make_report --out 폴더, camera_1·camera_2 기준값 json, "
                         "json 키(threshold_ring_ex=링 마스크, threshold=마스크 없음). 여러 번 줄 수 있음")
    ap.add_argument("--out", type=Path, default=Path("outputs/report/compare_v2.md"))
    ap.add_argument("--split", type=Path, default=Path("data/processed/company_v2_draem/split_v2.csv"))
    ap.add_argument("--band-scores", type=Path, default=None, help="band_best_eval.py 의 best_scores.csv")
    args = ap.parse_args()

    L = ["# DRAEM 재학습(키트 분할) vs 띠 두께 판정", "",
         "- DRAEM: 키트 train 정상으로 학습, 기준값 = val 정상(최종 모델로 채점) 점수 최댓값",
         "- 평가: 키트 test 정상 + validation·test 의 9/29 불량", ""]
    for name, report, th1, th2, key in args.draem:
        recs, ms = load_draem(Path(report), [Path(th1), Path(th2)], args.split, key)
        L += summarize(f"DRAEM · {name}", recs, ms)
    if args.band_scores:
        L += summarize("띠 두께 판정", load_band(args.band_scores))
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
