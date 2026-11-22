"""
이상 탐지 평가 지표.

- auroc: 이미지 단위 AU-ROC (순위 기반, 정확한 값)
- pixel_metrics: 픽셀 단위 AU-ROC와 AU-PRO (오탐률 30%까지, 논문 기본 지표)

AU-PRO는 불량 "영역"마다 같은 가중치를 줍니다. 큰 불량 하나보다 작은 불량
여러 개를 놓치는 것을 더 엄하게 평가하는 지표입니다(MVTec AD 논문 [7]).
메모리를 아끼기 위해 임계값을 n_bins개 구간으로 나눈 히스토그램으로 곡선을
계산합니다. 공식 평가 스크립트(모든 고유 점수를 임계값으로 사용)와는 아주
작은 차이가 날 수 있으니, 최종 보고 수치는 공식 스크립트로 확인하세요.
"""

import numpy as np
from scipy.ndimage import label

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x / 1.x


def auroc(labels, scores) -> float:
    """순위 기반 AU-ROC (동점은 평균 순위). sklearn roc_auc_score와 같은 값."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _rev_cumsum(h):
    return np.cumsum(h[::-1])[::-1]


def _area_up_to(fpr, y, limit):
    """fpr 오름차순 곡선 y를 fpr=limit까지 적분 (경계는 선형 보간)."""
    fpr = np.concatenate([[0.0], fpr])
    y = np.concatenate([[0.0], y])
    if limit < fpr[-1]:
        i = np.searchsorted(fpr, limit, side="right")
        y_lim = np.interp(limit, fpr[i - 1:i + 1], y[i - 1:i + 1])
        fpr = np.concatenate([fpr[:i], [limit]])
        y = np.concatenate([y[:i], [y_lim]])
    return float(_trapz(y, fpr))


def pixel_metrics(maps, masks, fpr_limit: float = 0.3, n_bins: int = 2000) -> dict:
    """maps, masks: 같은 크기의 2D 배열 목록 (맵은 float, 마스크는 bool).
    정상 이미지는 마스크가 전부 False. 반환: {'pixel_auroc', 'au_pro'}"""
    lo = min(float(m.min()) for m in maps)
    hi = max(float(m.max()) for m in maps)
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    ok_hist = np.zeros(n_bins)
    bad_hist = np.zeros(n_bins)
    pro_hist = np.zeros(n_bins)
    n_regions = 0
    for m, gt in zip(maps, masks):
        m = np.asarray(m, dtype=np.float32)
        gt = np.asarray(gt, dtype=bool)
        ok_hist += np.histogram(m[~gt], edges)[0]
        if gt.any():
            bad_hist += np.histogram(m[gt], edges)[0]
            lab, n = label(gt, structure=np.ones((3, 3)))
            for k in range(1, n + 1):
                region = lab == k
                pro_hist += np.histogram(m[region], edges)[0] / region.sum()
            n_regions += n
    if ok_hist.sum() == 0 or n_regions == 0:
        return {"pixel_auroc": float("nan"), "au_pro": float("nan")}

    # 임계값을 높은 쪽에서 낮추며: 각 구간 이상인 픽셀 비율
    fpr = _rev_cumsum(ok_hist)[::-1] / ok_hist.sum()
    tpr = _rev_cumsum(bad_hist)[::-1] / bad_hist.sum()
    pro = _rev_cumsum(pro_hist)[::-1] / n_regions
    return {"pixel_auroc": _area_up_to(fpr, tpr, 1.0),
            "au_pro": _area_up_to(fpr, pro, fpr_limit) / fpr_limit}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 완벽한 맵이면 1, 무작위 맵이면 AU-ROC ≈ 0.5, AU-PRO ≈ 0.15 (= 0.3/2)
    masks = [np.zeros((64, 64), bool) for _ in range(4)]
    masks[0][10:20, 10:20] = True
    masks[1][30:34, 40:60] = True
    perfect = [m.astype(np.float32) for m in masks]
    noise = [rng.random((64, 64)).astype(np.float32) for _ in masks]
    print("완벽한 맵:", pixel_metrics(perfect, masks))
    print("무작위 맵:", pixel_metrics(noise, masks))
