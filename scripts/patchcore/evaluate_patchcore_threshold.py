
# [이 스크립트의 역할] threshold를 적용해 과검률 / 미검률 / F1 계산
# compute_patchcore_threshold.py 로 정한 threshold를 추론 결과 CSV에 그대로 적용한다.
# 판정 규칙: anomaly_score > threshold 이면 불량(1), 아니면 정상(0)
#
# 지표 (불량을 양성으로 본다):
#   과검률(FPR) = 불량으로 판정된 정상 이미지 수 / 정상 이미지 수
#   미검률(FNR) = 정상으로 판정된 불량 이미지 수 / 불량 이미지 수
#   F1          = 정밀도(불량 판정 중 진짜 불량 비율)와 재현율(1 - 미검률)의 조화평균
#
# 실행 예 (원본 테스트셋):
#   python scripts/patchcore/evaluate_patchcore_threshold.py
#       --csv outputs/predictions/patchcore/bottle_original.csv
#       --threshold outputs/patchcore/bottle/threshold_1pct_d128_s0_k10.json
#
# 실행 예 (정상편차 CSV, 정상 이미지만 있음):
#   python scripts/patchcore/evaluate_patchcore_threshold.py
#       --csv outputs/predictions/patchcore/bottle_bottle_variation_v2.csv
#       --threshold outputs/patchcore/bottle/threshold_1pct_d128_s0_k10.json
#       --defects-csv outputs/predictions/patchcore/bottle_original.csv
#   --defects-csv를 주면 원본 불량 이미지를 각 편차 그룹에 붙여서
#   "편차가 생긴 정상 vs 원본 불량" 기준의 미검률/F1/AUROC도 계산한다.
#
# 결과: <CSV 이름>_threshold_metrics.json (CSV와 같은 폴더)
#
# 참고 지표 (테스트 정답을 보고 정한 값이라 메인 결과로 쓰면 안 됨):
#   zero_miss : 불량을 하나도 놓치지 않는 가장 높은 threshold에서의 과검률
#   best_f1   : 테스트셋에서 F1이 가장 높게 나오는 threshold (성능 상한선)
#
# 주의: 불량이 정상보다 훨씬 많은 데이터(회사 데이터)에서는
#       "전부 불량"으로 판정해도 F1이 높게 나온다 → F1만 보고 판단하면 안 된다.

from pathlib import Path
import argparse
import csv
import json
import sys
# defaultdict: 없는 키를 처음 쓸 때 자동으로 빈 값(여기선 빈 리스트)을 만들어 주는 딕셔너리
from collections import defaultdict
from statistics import fmean

from sklearn.metrics import roc_auc_score

# 레포 최상위 폴더를 찾아 import 경로에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def resolve(path):
    # 상대경로면 레포 최상위 기준 절대경로로 바꾼다
    return path if path.is_absolute() else ROOT / path


def load_csv(path):
    # 추론 결과 CSV를 읽고 label, anomaly_score를 숫자로 바꾼다
    # utf-8-sig: 엑셀용으로 저장된 CSV 앞의 BOM 문자를 자동으로 제거
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    # CSV는 모든 값이 글자로 읽히므로 계산에 쓸 값은 숫자로 변환
    for row in rows:
        row["label"] = int(row["label"])
        row["anomaly_score"] = float(row["anomaly_score"])
    return rows


def threshold_metrics(rows, threshold):
    """정해진 threshold로 판정했을 때의 지표"""
    normals = [r for r in rows if r["label"] == 0]   # 정답이 정상인 이미지
    defects = [r for r in rows if r["label"] == 1]   # 정답이 불량인 이미지

    # True는 1, False는 0으로 더해지므로 sum = 조건을 만족한 개수
    # 과검: 정상인데 점수가 threshold보다 높아서 불량으로 판정된 수
    false_alarms = sum(r["anomaly_score"] > threshold for r in normals)
    # 검출: 불량이고 점수가 threshold보다 높아서 제대로 잡은 수
    caught = sum(r["anomaly_score"] > threshold for r in defects)
    # 미검: 불량인데 놓친 수
    missed = len(defects) - caught

    result = {
        "normal_count": len(normals),
        "false_alarm": false_alarms,
        # 정상이 없으면 비율 계산 불가 → None
        "fpr": false_alarms / len(normals) if normals else None,
        "defect_count": len(defects),
        "missed": missed,
        # 불량이 없으면 비율 계산 불가 → None
        "fnr": missed / len(defects) if defects else None,
    }

    # 아래 지표들은 불량 이미지가 있어야 계산할 수 있다
    if defects:
        # 정밀도: 불량이라고 판정한 것 중 진짜 불량의 비율 (불량 판정이 0개면 0)
        precision = caught / (caught + false_alarms) if caught + false_alarms else 0.0
        # 재현율: 진짜 불량 중 잡아낸 비율 (= 1 - 미검률)
        recall = caught / len(defects)
        result["precision"] = precision
        result["recall"] = recall
        # F1: 정밀도와 재현율의 조화평균 (둘 다 0이면 0)
        result["f1"] = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        # 불량 종류별 미검
        # by_type = {"over": [True, False, ...], "step": [...]}  (True = 검출 성공)
        by_type = defaultdict(list)
        for r in defects:
            by_type[r["defect_type"]].append(r["anomaly_score"] > threshold)
        result["missed_by_type"] = {
            name: {"count": len(hits), "missed": len(hits) - sum(hits)}
            for name, hits in sorted(by_type.items())
        }

    # AUROC는 threshold와 무관하지만, 정상과 불량이 모두 있으면 같이 기록
    if normals and defects:
        result["auroc"] = roc_auc_score(
            [r["label"] for r in rows],
            [r["anomaly_score"] for r in rows],
        )

    return result


def reference_metrics(rows):
    """테스트 정답을 보고 정한 참고용 threshold 2개"""
    normals = [r["anomaly_score"] for r in rows if r["label"] == 0]
    defects = [r["anomaly_score"] for r in rows if r["label"] == 1]
    # 정상과 불량이 둘 다 있어야 계산 가능
    if not normals or not defects:
        return None

    # 미검 0: 가장 낮은 불량 점수 바로 아래를 threshold로 → 모든 불량 검출
    # 이때 그 점수 이상인 정상은 전부 과검이 된다
    zero_miss = min(defects)
    zero_miss_fpr = sum(s >= zero_miss for s in normals) / len(normals)

    # F1 최대: 모든 점수를 후보 threshold로 시험 (score >= t 를 불량으로)
    best = None
    # set: 중복 점수 제거, sorted: 작은 값부터 시험
    for t in sorted(set(normals + defects)):
        tp = sum(s >= t for s in defects)   # 잡은 불량 수
        fp = sum(s >= t for s in normals)   # 과검된 정상 수
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / len(defects)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # 지금까지 중 F1이 가장 높으면 기록
        if best is None or f1 > best["f1"]:
            best = {
                "threshold": t,
                "f1": f1,
                "fpr": fp / len(normals),
                "fnr": 1 - recall,
            }

    return {
        "zero_miss": {"threshold": zero_miss, "fpr": zero_miss_fpr},
        "best_f1": best,
    }


def fmt(value, count=None, total=None):
    # 비율을 화면 출력용 글자로 (예: 0.333, 2, 6 → ' 33.3% (2/6)')
    if value is None:
        return "-"
    text = f"{value * 100:5.1f}%"
    if count is not None:
        text += f" ({count}/{total})"
    return text


def main():
    parser = argparse.ArgumentParser(
        description="PatchCore threshold 기반 과검률/미검률/F1 평가"
    )
    # --csv: 평가할 추론 결과 CSV (원본 테스트셋 또는 정상편차)
    parser.add_argument("--csv", type=Path, required=True)
    # --threshold: compute_patchcore_threshold.py 결과 JSON
    parser.add_argument("--threshold", type=Path, required=True)
    # --defects-csv: 정상편차 CSV를 평가할 때 붙일 원본 불량 이미지 CSV (선택)
    parser.add_argument("--defects-csv", type=Path, default=None)
    # --output: 결과 JSON 경로 (안 주면 CSV 옆에 자동 이름)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    csv_path = resolve(args.csv)
    threshold_path = resolve(args.threshold)

    for path in (csv_path, threshold_path):
        if not path.is_file():
            parser.error(f"파일이 없습니다: {path}")

    # threshold 값 읽기
    with open(threshold_path, "r", encoding="utf-8") as file:
        threshold_info = json.load(file)
    threshold = threshold_info["threshold"]

    rows = load_csv(csv_path)

    # 원본 불량 이미지 (편차 CSV 평가용)
    # 편차 CSV에는 정상만 있어서 미검률/AUROC를 계산하려면 불량을 따로 가져와야 한다
    extra_defects = []
    if args.defects_csv:
        defects_path = resolve(args.defects_csv)
        if not defects_path.is_file():
            parser.error(f"파일이 없습니다: {defects_path}")
        extra_defects = [r for r in load_csv(defects_path) if r["label"] == 1]

    # (편차 종류, 강도)별로 묶기. 원본 CSV는 ("original", 0) 그룹 하나
    # 예) ("brightness", -20.0), ("rotation", 5.0), ...
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variation"], float(r["level"]))].append(r)

    results = []
    # 정렬: "original"을 맨 앞에, 나머지는 편차 이름 → 강도 순
    # (item[0][0] != "original" 은 original이면 False(0)라서 먼저 온다)
    for (variation, level), group_rows in sorted(
        groups.items(), key=lambda item: (item[0][0] != "original", item[0])
    ):
        # 그룹 안에 불량이 있으면 그대로, 없으면(편차 CSV) 원본 불량을 붙여서 평가
        has_defects = any(r["label"] == 1 for r in group_rows)
        eval_rows = group_rows if has_defects else group_rows + extra_defects

        entry = {
            "variation": variation,
            "level": level,
            # 이 그룹 정상 이미지들의 평균 점수 (편차가 점수를 얼마나 올리는지 확인용)
            "normal_mean_score": fmean(
                r["anomaly_score"] for r in group_rows if r["label"] == 0
            ),
            # ** : threshold_metrics 결과 딕셔너리를 여기에 펼쳐서 합친다
            **threshold_metrics(eval_rows, threshold),
            # 불량을 원본 CSV에서 가져왔으면 그 경로를 기록
            "defects_from": None if has_defects or not extra_defects
            else str(args.defects_csv.as_posix()),
        }
        # 참고 지표는 그룹 자체에 불량이 있을 때(원본 테스트셋)만 계산
        if has_defects:
            entry["reference"] = reference_metrics(group_rows)
        results.append(entry)

    # 결과 JSON 경로: 안 주면 <CSV 이름>_threshold_metrics.json
    output_path = args.output or csv_path.with_name(
        f"{csv_path.stem}_threshold_metrics.json"
    )
    output_path = resolve(output_path)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({
            "csv": csv_path.relative_to(ROOT).as_posix(),
            "threshold_file": threshold_path.relative_to(ROOT).as_posix(),
            "threshold": threshold,
            "rule": "anomaly_score > threshold → 불량",
            "groups": results,
        }, file, ensure_ascii=False, indent=2)

    # 화면에 표 형태로 출력
    # :<16 = 왼쪽 정렬 16칸, :>6 = 오른쪽 정렬 6칸
    print("=== threshold 기반 평가 ===")
    print(f"threshold: {threshold:.4f} ({threshold_path.name})")
    print(f"{'편차':<16}{'강도':>6}  {'정상평균':>8}  {'과검률':<16}{'미검률':<18}{'F1':>6}{'AUROC':>8}")
    for e in results:
        f1 = f"{e['f1']:.3f}" if "f1" in e else "-"
        auroc = f"{e['auroc']:.3f}" if "auroc" in e else "-"
        # :g = 불필요한 소수점 0 제거 (-20.0 → -20)
        print(
            f"{e['variation']:<16}{e['level']:>6g}  {e['normal_mean_score']:8.3f}  "
            f"{fmt(e['fpr'], e['false_alarm'], e['normal_count']):<16}"
            f"{fmt(e['fnr'], e['missed'], e['defect_count']):<18}{f1:>6}{auroc:>8}"
        )
        # 불량 종류별 미검은 original 그룹만 출력 (편차 그룹은 같은 불량이라 반복됨)
        if e.get("missed_by_type") and e["variation"] == "original":
            for name, m in e["missed_by_type"].items():
                print(f"    미검 {name}: {m['missed']}/{m['count']}")
        # 참고 지표 출력
        if e.get("reference"):
            ref = e["reference"]
            print(f"    [참고] 미검 0 기준 과검률: {fmt(ref['zero_miss']['fpr'])}"
                  f" / F1 최대 {ref['best_f1']['f1']:.3f}"
                  f" (과검 {fmt(ref['best_f1']['fpr'])}, 미검 {fmt(ref['best_f1']['fnr'])})")

    print("\n저장 위치:", output_path)


# 직접 실행했을 때만 main() 실행
if __name__ == "__main__":
    main()
