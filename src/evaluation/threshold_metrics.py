"""모델과 무관한 고정 임계값 평가. 테스트 정답으로 임계값을 탐색하지 않는다."""

from collections import defaultdict
import math

from sklearn.metrics import roc_auc_score


def validate_rows(rows):
    if not rows:
        raise ValueError("평가할 예측이 없습니다.")
    ids = []
    for row in rows:
        if row["label"] not in (0, 1):
            raise ValueError("label은 정상 0 또는 불량 1이어야 합니다.")
        if not math.isfinite(float(row["anomaly_score"])):
            raise ValueError("이상 점수는 유한한 수여야 합니다.")
        if "image_id" in row:
            ids.append(row["image_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("중복 image_id가 있습니다.")


def threshold_metrics(rows, threshold):
    """공통 판정 규칙: score > threshold이면 불량. 같으면 정상이다."""
    validate_rows(rows)
    if not math.isfinite(float(threshold)):
        raise ValueError("임계값은 유한한 수여야 합니다.")
    normals = [row for row in rows if row["label"] == 0]
    defects = [row for row in rows if row["label"] == 1]
    fp = sum(row["anomaly_score"] > threshold for row in normals)
    tp = sum(row["anomaly_score"] > threshold for row in defects)
    fn = len(defects) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(defects) if defects else None
    result = {
        "normal_count": len(normals), "false_alarm": fp,
        "fpr": fp / len(normals) if normals else None,
        "defect_count": len(defects), "missed": fn,
        "fnr": fn / len(defects) if defects else None,
        "precision": precision if defects else None, "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)
               if precision + recall else 0.0) if defects else None,
    }
    by_type = defaultdict(list)
    for row in defects:
        by_type[row["defect_type"]].append(row["anomaly_score"] > threshold)
    result["missed_by_type"] = {
        name: {"count": len(hits), "missed": len(hits) - sum(hits)}
        for name, hits in sorted(by_type.items())
    }
    if normals and defects:
        result["auroc"] = float(roc_auc_score(
            [row["label"] for row in rows], [row["anomaly_score"] for row in rows]))
    return result
