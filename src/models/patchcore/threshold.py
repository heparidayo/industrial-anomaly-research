"""PatchCore 학습 정상 특징의 촬영그룹 held-out 점수로 임계값을 보정한다."""
import random
import torch
from .coreset import approximate_greedy_coreset_indices
from .scoring import compute_patch_scores


# 임계값 규칙 (2026-11-19 팀 결정: 95% 분위수, PATCHCORE_PROGRESS.md I-8)
#   max      : 번갈아 뺀 학습 정상 점수의 최댓값 (예전 규칙, 학습 장수가 늘면 가장 이상한 1장이 기준이 됨)
#   quantile : 그 점수들의 q 분위수 (0.95 → 정상의 약 5%를 불량으로 판정, 과검률을 미리 정할 수 있음)
THRESHOLD_RULES = ("quantile", "max")
DEFAULT_THRESHOLD_RULE = "quantile"
DEFAULT_THRESHOLD_QUANTILE = 0.95


def check_threshold_rule(rule, quantile):
    """규칙 이름과 분위수 값이 올바른지 확인한다."""
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"임계값 규칙은 {THRESHOLD_RULES} 중 하나여야 합니다: {rule}")
    if rule == "quantile" and not 0 < quantile < 1:
        raise ValueError("분위수는 0과 1 사이여야 합니다 (예: 0.95).")


def threshold_tag(rule, quantile):
    """결과 파일 이름 꼬리. max는 빈 문자열이라 예전 파일 이름과 같다 (기존 결과 그대로 사용)."""
    check_threshold_rule(rule, quantile)
    return "" if rule == "max" else "_q" + f"{quantile * 100:g}".replace(".", "p")  # 0.95 → _q95


def threshold_from_scores(scores, rule=DEFAULT_THRESHOLD_RULE, quantile=DEFAULT_THRESHOLD_QUANTILE):
    """번갈아 뺀 학습 정상 점수들 → 임계값 하나.

    분위수는 작은 것부터 줄 세운 뒤 위치 (장수−1)×q 의 값 (사이에 떨어지면 양옆 값을 비율대로 섞음,
    numpy·torch 기본 방식과 같음). 예: 100장, q=0.95 → 위치 94.05 → 95번째와 96번째 사이
    """
    check_threshold_rule(rule, quantile)
    values = torch.as_tensor(list(scores), dtype=torch.float64)          # 정밀도 위해 float64
    if values.numel() == 0:
        raise ValueError("임계값을 정할 학습 정상 점수가 없습니다.")
    if rule == "max":
        return values.max().item()                                       # 예전 규칙
    return values.quantile(quantile).item()                              # 분위수 (선형 보간)


def effective_folds(groups, requested):
    if not isinstance(requested, int) or requested < 2:
        raise ValueError("folds는 2 이상이어야 합니다.")
    count = min(requested, len(set(groups)))
    if count < 2:
        raise ValueError("서로 다른 학습 촬영그룹이 최소 2개 필요합니다.")
    return count


def held_out_scores(per_image, folds, percentage, projection_dim, seed, device,
                    log=print, *, groups=None):
    if per_image.ndim != 3 or per_image.shape[0] < 2:
        raise ValueError("정상 특징은 이미지별 [N, patches, dim]이고 N >= 2여야 합니다.")
    groups = list(range(len(per_image))) if groups is None else list(groups)
    if len(groups) != len(per_image):
        raise ValueError("이미지 수와 촬영그룹 수가 일치하지 않습니다.")
    folds = effective_folds(groups, folds)
    unique = list(dict.fromkeys(groups))
    random.Random(seed).shuffle(unique)
    fold_by_group = {group: i % folds for i, group in enumerate(unique)}
    fold_of = torch.tensor([fold_by_group[group] for group in groups])
    scores = [None] * len(groups)
    for fold in range(folds):
        held_out = (fold_of == fold).nonzero().flatten()
        kept = (fold_of != fold).nonzero().flatten()
        features = per_image[kept].reshape(-1, per_image.shape[-1])
        indices = approximate_greedy_coreset_indices(
            features=features, percentage=percentage, projection_dim=projection_dim,
            seed=seed, device=device, log_every=10**9)
        bank = features.index_select(0, indices.cpu()).to(device)
        for index in held_out.tolist():
            scores[index] = compute_patch_scores(
                per_image[index].unsqueeze(0).to(device), bank).max().item()
        if log:
            log(f"[fold {fold + 1}/{folds}] held-out {len(held_out)} images")
    return scores
