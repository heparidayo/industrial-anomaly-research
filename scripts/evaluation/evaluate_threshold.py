"""고정된 임계값 JSON을 예측 CSV에 적용하는 공통 평가 진입점."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.threshold_metrics import threshold_metrics, validate_rows


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "label", "defect_type", "anomaly_score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("예측 CSV 필수 컬럼이 없습니다.")
        rows = list(reader)
    for row in rows:
        row["label"] = int(row["label"])
        row["anomaly_score"] = float(row["anomaly_score"])
        row["variation"] = row.get("variation") or "original"
        row["level"] = float(row.get("level") or 0)
        if not math.isfinite(row["level"]):
            raise ValueError("편차 강도는 유한한 수여야 합니다.")
        if row.get("inference_ms") is not None:
            row["inference_ms"] = float(row["inference_ms"])
            if not math.isfinite(row["inference_ms"]) or row["inference_ms"] < 0:
                raise ValueError("처리시간이 잘못되었습니다.")
    validate_rows(rows)
    return rows


def evaluate_groups(rows, threshold, extra_defects=None):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variation"], row["level"])].append(row)
    results = []
    for (variation, level), group in sorted(
            groups.items(), key=lambda item: (item[0][0] != "original", item[0])):
        normals = [row for row in group if row["label"] == 0]
        appended = not any(row["label"] == 1 for row in group) and bool(extra_defects)
        evaluated = group + extra_defects if appended else group
        results.append({
            "variation": variation, "level": level,
            "normal_mean_score": fmean(row["anomaly_score"] for row in normals) if normals else None,
            **threshold_metrics(evaluated, threshold),
            "original_defects_reused": appended,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="고정 임계값 기반 공통 과검·미검·F1 평가")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--threshold", type=Path, required=True)
    parser.add_argument("--defects-csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--category", help="임계값 파일의 카테고리를 확인할 때 지정")
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    try:
        source, threshold_path = resolve(args.csv), resolve(args.threshold)
        info = json.loads(threshold_path.read_text(encoding="utf-8"))
        if args.category and info.get("category") != args.category:
            raise ValueError("임계값 카테고리가 일치하지 않습니다.")
        metadata = None
        sidecar = source.with_suffix(source.suffix + ".meta.json")
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if info.get("category") and metadata.get("category") != info["category"]:
                raise ValueError("예측과 임계값의 카테고리가 다릅니다.")
            if info.get("percentage") is not None and metadata.get("coreset") != {
                    key: info[key] for key in ("percentage", "projection_dim", "seed")}:
                raise ValueError("예측과 임계값의 Coreset 설정이 다릅니다.")
            if info.get("raw_memory_sha256") and metadata.get("raw_memory_sha256") != info["raw_memory_sha256"]:
                raise ValueError("예측과 임계값이 다른 학습 특징에서 생성되었습니다.")
        rows = load_csv(source)
        if args.defects_csv and metadata:
            defects_sidecar = resolve(args.defects_csv).with_suffix(args.defects_csv.suffix + ".meta.json")
            if defects_sidecar.is_file():
                defect_metadata = json.loads(defects_sidecar.read_text(encoding="utf-8"))
                for key in ("category", "memory_sha256"):
                    if metadata.get(key) != defect_metadata.get(key):
                        raise ValueError("원본 불량 CSV가 정상편차와 다른 모델·카테고리에서 생성되었습니다.")
        defects = [row for row in load_csv(resolve(args.defects_csv))
                   if row["label"] == 1] if args.defects_csv else []
        groups = evaluate_groups(rows, float(info["threshold"]), defects)
        output = resolve(args.output) if args.output else source.with_name(source.stem + "_threshold_metrics.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "csv": str(source), "threshold_file": str(threshold_path),
            "threshold": info["threshold"], "rule": "anomaly_score > threshold",
            "groups": groups,
            "note": "원본 불량 재사용 그룹의 미검률은 변형된 불량에 대한 결과가 아닙니다.",
        }
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for group in groups:
        print(group["variation"], group["level"],
              "FPR:", group["fpr"], "FNR:", group["fnr"], "F1:", group["f1"])
    print("결과:", output)


if __name__ == "__main__":
    main()
