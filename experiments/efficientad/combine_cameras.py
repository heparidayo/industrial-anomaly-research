"""
두 카메라 결과 결합 평가 — "둘 다 정상일 때만 정상, 한쪽이라도 불량이면 불량".

입력: predict.py가 카메라별로 만든 예측 CSV 두 개
  python predict.py --checkpoint runs/cam1/efficientad_s.pth --data_root <out> --category cam1 --out_csv pred_cam1.csv
  python predict.py --checkpoint runs/cam2/efficientad_s.pth --data_root <out> --category cam2 --out_csv pred_cam2.csv

결합 규칙: 제품 점수 = max(카메라1 점수, 카메라2 점수)
  두 모델 모두 분위수 정규화를 거쳐 점수가 "정상 노이즈 대비 몇 배인가"라는
  같은 눈금이므로, 최댓값이 곧 OR 판정이 됩니다.

출력:
  - 제품 단위 AU-ROC (전체, 불량 유형별), 카메라별 유형별 AU-ROC
  - 참고용 임계값 표: 정상 제품 오탐률 1%·5%가 되는 임계값에서 유형별 검출률
    (테스트 세트로 정한 임계값이므로 실제 운용 성능의 추정치가 아니라 참고값)
  - combined.csv: 제품별 두 카메라 점수와 최종 점수

사용 예:
  python combine_cameras.py --cam1 pred_cam1.csv --cam2 pred_cam2.csv --out combined
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics import auroc

NAME_RE = re.compile(r"^(?P<pid>.+)_camera_(?P<cam>\d+)$")


def read_pred(path, cam_expected):
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Path(r["image_path"])
            m = NAME_RE.match(p.stem)
            if not m:
                continue
            if int(m.group("cam")) != cam_expected:
                print(f"  [주의] {path}에 camera_{m.group('cam')} 이미지가 섞여 있음: {p.name}")
            dtype = p.parent.name                         # good / over / step / insufficient ...
            out[(m.group("pid"), dtype)] = float(r["score"])
    return out


def per_type_auroc(rows, score_key):
    good = [r[score_key] for r in rows if r["type"] == "good" and r[score_key] is not None]
    res = {}
    for t in sorted({r["type"] for r in rows} - {"good"}):
        bad = [r[score_key] for r in rows if r["type"] == t and r[score_key] is not None]
        res[t] = {"n": len(bad), "auroc": auroc([0] * len(good) + [1] * len(bad), good + bad)}
    return res


def main():
    ap = argparse.ArgumentParser(description="두 카메라 결합 평가")
    ap.add_argument("--cam1", required=True)
    ap.add_argument("--cam2", required=True)
    ap.add_argument("--out", default="combined")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    p1, p2 = read_pred(args.cam1, 1), read_pred(args.cam2, 2)
    rows = []
    for key in sorted(set(p1) | set(p2)):
        pid, dtype = key
        s1, s2 = p1.get(key), p2.get(key)
        rows.append({"product_id": pid, "type": dtype, "score_cam1": s1, "score_cam2": s2,
                     "score": max(s for s in (s1, s2) if s is not None),
                     "cameras": int(s1 is not None) + int(s2 is not None)})
    missing = sum(r["cameras"] < 2 for r in rows)

    labels = [0 if r["type"] == "good" else 1 for r in rows]
    result = {
        "n_products": len(rows), "products_missing_a_camera": missing,
        "product_auroc": auroc(labels, [r["score"] for r in rows]),
        "product_auroc_by_type": per_type_auroc(rows, "score"),
        "cam1_auroc_by_type": per_type_auroc(rows, "score_cam1"),
        "cam2_auroc_by_type": per_type_auroc(rows, "score_cam2"),
    }

    good_scores = np.array([r["score"] for r in rows if r["type"] == "good"])
    thresholds = {}
    for fpr in (0.01, 0.05):
        thr = float(np.quantile(good_scores, 1 - fpr))
        recall = {}
        for t in sorted({r["type"] for r in rows} - {"good"}):
            s = np.array([r["score"] for r in rows if r["type"] == t])
            recall[t] = float((s > thr).mean())
        thresholds[f"fpr_{int(fpr * 100)}pct"] = {"threshold": thr, "recall_by_type": recall}
    result["reference_thresholds_test_tuned"] = thresholds

    # ------------------------------------------------------------ 출력
    print(f"제품 {len(rows)}개 (카메라 한쪽 누락 {missing}개)")
    print(f"제품 단위 AU-ROC (전체): {result['product_auroc'] * 100:.2f}%")
    print("\n불량 유형별 AU-ROC (정상 대 해당 유형)")
    print(f"  {'유형':14s} {'장수':>5s} {'결합':>8s} {'cam1':>8s} {'cam2':>8s}")
    for t, v in result["product_auroc_by_type"].items():
        c1 = result["cam1_auroc_by_type"].get(t, {}).get("auroc", float("nan"))
        c2 = result["cam2_auroc_by_type"].get(t, {}).get("auroc", float("nan"))
        print(f"  {t:14s} {v['n']:5d} {v['auroc'] * 100:7.2f}% {c1 * 100:7.2f}% {c2 * 100:7.2f}%")
    print("\n참고: 정상 오탐률 기준 유형별 검출률 (테스트 세트로 정한 임계값)")
    for k, v in thresholds.items():
        rec = ", ".join(f"{t} {r * 100:.1f}%" for t, r in v["recall_by_type"].items())
        print(f"  {k}: 임계값 {v['threshold']:.4f} → {rec}")

    with open(out / "combined.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "combined_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out / 'combined.csv'}, {out / 'combined_result.json'}")


if __name__ == "__main__":
    main()
