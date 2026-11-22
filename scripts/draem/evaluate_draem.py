"""예측 CSV 평가: 이미지 AUROC · 과검률 · 미검률 · 추론시간.

run_draem_batch.py가 만든 CSV를 그대로 읽습니다. (PatchCore CSV도 컬럼이 같아서 같이 쓸 수 있음)
    필요한 컬럼: label (0 정상 / 1 불량), defect_type, anomaly_score, (선택) inference_ms

    - 이미지 AUROC: threshold 없이 계산. 유형별(정상 vs over / step / insufficient)도 계산
    - 과검률 · 미검률: threshold가 있을 때만. 장수와 함께 출력 (예: 1/6)
    - 추론시간: ms/장 평균

예시:
    # AUROC · 추론시간만
    .\\.venv\\Scripts\\python.exe scripts/draem/evaluate_draem.py outputs/predictions/draem/company_plc_camera_1_original.csv

    # kfold_threshold.py 결과로 과검률 · 미검률까지
    .\\.venv\\Scripts\\python.exe scripts/draem/evaluate_draem.py <CSV> --threshold-json outputs/draem/camera_1/threshold/<결과>.json

    # 여러 CSV를 한 표로 저장
    .\\.venv\\Scripts\\python.exe scripts/draem/evaluate_draem.py <cam1.csv> <cam2.csv> <cam12.csv> --out outputs/eval/draem_company.csv
"""

from pathlib import Path
import argparse
import csv
import json

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    labels = np.array([int(r["label"]) for r in rows])
    scores = np.array([float(r["anomaly_score"]) for r in rows])
    types = np.array([r["defect_type"] for r in rows])
    ms = np.array([float(r["inference_ms"]) for r in rows if r.get("inference_ms")])
    return labels, scores, types, ms


def frac(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b * 100:.1f}%)" if b else "-"


def evaluate(path: Path, threshold: float | None) -> dict:
    labels, scores, types, ms = load(path)
    n_good, n_bad = int((labels == 0).sum()), int((labels == 1).sum())
    if n_good == 0 or n_bad == 0:
        raise SystemExit(f"{path.name}: 정상과 불량이 모두 있어야 AUROC를 계산할 수 있습니다.")

    defect_types = sorted(set(types[labels == 1]))
    result = {
        "dataset": path.stem,
        "n_good": n_good,
        "n_bad": n_bad,
        "image_auroc": roc_auc_score(labels, scores),
    }
    # 유형별 AUROC: 정상 전체 vs 그 유형만
    for t in defect_types:
        mask = (labels == 0) | (types == t)
        result[f"auroc_{t}"] = roc_auc_score(labels[mask], scores[mask])

    if threshold is not None:
        pred = scores > threshold
        result["threshold"] = threshold
        result["false_alarm"] = frac(int((pred & (labels == 0)).sum()), n_good)   # 과검률
        result["miss"] = frac(int((~pred & (labels == 1)).sum()), n_bad)         # 미검률
        for t in defect_types:
            m = types == t
            result[f"miss_{t}"] = frac(int((~pred & m).sum()), int(m.sum()))

    result["inference_ms"] = float(ms.mean()) if len(ms) else float("nan")
    return result


def main():
    parser = argparse.ArgumentParser(description="예측 CSV 평가 (AUROC · 과검률 · 미검률 · 추론시간)")
    parser.add_argument("csv", type=Path, nargs="+", help="run_draem_batch.py 결과 CSV (여러 개 가능)")
    parser.add_argument("--threshold", type=float, default=None, help="점수 > threshold 이면 불량")
    parser.add_argument("--threshold-json", type=Path, default=None,
                        help="kfold_threshold.py 결과 JSON에서 threshold를 읽음")
    parser.add_argument("--out", type=Path, default=None, help="결과를 CSV로 저장")
    args = parser.parse_args()

    if args.threshold_json is not None:
        if args.threshold is not None:
            parser.error("--threshold와 --threshold-json은 하나만 쓰세요.")
        tj = args.threshold_json if args.threshold_json.is_absolute() else ROOT / args.threshold_json
        args.threshold = float(json.loads(tj.read_text(encoding="utf-8"))["threshold"])
        print(f"threshold {args.threshold:.4f} ({tj.name})")

    results = []
    for p in args.csv:
        p = p if p.is_absolute() else ROOT / p
        if not p.is_file():
            parser.error(f"CSV가 없습니다: {p}")
        results.append(evaluate(p, args.threshold))

    for r in results:
        print(f"\n=== {r['dataset']} (정상 {r['n_good']} · 불량 {r['n_bad']}) ===")
        print(f"이미지 AUROC   {r['image_auroc']:.3f}")
        for k, v in r.items():
            if k.startswith("auroc_"):
                print(f"  {k[6:]:13s} {v:.3f}")
        if "threshold" in r:
            print(f"과검률         {r['false_alarm']}")
            print(f"미검률         {r['miss']}")
            for k, v in r.items():
                if k.startswith("miss_"):
                    print(f"  {k[5:]:13s} {v}")
        else:
            print("과검률 · 미검률: threshold가 없어 계산하지 않음 (--threshold-json 지정)")
        print(f"추론시간       {r['inference_ms']:.1f} ms/장")

    if args.out:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = []
        for r in results:
            keys += [k for k in r if k not in keys]
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
