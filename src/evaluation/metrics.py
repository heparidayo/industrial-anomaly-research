# 지표 계산


import math
from collections import defaultdict
from statistics import fmean, median

from sklearn.metrics import roc_auc_score


def evaluate_image_predictions(records):
    """모델별 추론 결과에서 이미지 단위 성능을 평가한다."""

    if not records:
        raise ValueError("평가할 데이터가 없습니다.")

    image_ids = [row["image_id"] for row in records]

    if len(image_ids) != len(set(image_ids)):
        raise ValueError("중복된 image_id가 존재합니다.")

    labels = [int(row["label"]) for row in records]
    scores = [float(row["anomaly_score"]) for row in records]
    times = [float(row["inference_ms"]) for row in records]

    if set(labels) != {0, 1}:
        raise ValueError("AUROC 계산에는 정상(0), 불량(1)이 모두 필요합니다.")

    if not all(math.isfinite(v) for v in scores + times):
        raise ValueError("점수 또는 시간에 유효하지 않은 값이 있습니다.")

    if any(t < 0 for t in times):
        raise ValueError("추론 시간이 음수입니다.")

    # 1. 전체 Image-level AUROC
    # 정상/불량 점수로 ROC 곡선 아래 면적 계산
    auc = roc_auc_score(labels, scores)

    # 2. 결함 종류별 통계
    groups = defaultdict(list)

    for record in records:
        groups[record["defect_type"]].append(
            float(record["anomaly_score"])
        )

    group_results = {}

    for name, values in sorted(groups.items()):
        group_results[name] = {
            "count": len(values),
            "mean": fmean(values),
            "min": min(values),
            "max": max(values),
        }

    # 3. 평가 결과 반환
    return {
        "image_count": len(records),
        "normal_count": labels.count(0),
        "anomaly_count": labels.count(1),
        "image_auroc": float(auc),
        "mean_inference_ms": fmean(times),
        "median_inference_ms": median(times),
        "groups": group_results,
    }
