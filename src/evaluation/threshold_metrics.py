# threshold 기반 판정 지표 (과검률 / 미검률 / F1)
#
# 모델과 무관하게 "정답(label)과 이상 점수(anomaly_score)"만 있으면 계산할 수 있다.
# → PatchCore, DRAEM, EfficientAD 모두 팀 공통 예측 CSV 형식이면 그대로 사용 가능
#
# 판정 규칙: anomaly_score > threshold 이면 불량(1), 아니면 정상(0)
# 입력 rows: {"label": int, "anomaly_score": float, "defect_type": str, ...} 리스트

from collections import defaultdict

from sklearn.metrics import roc_auc_score

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
