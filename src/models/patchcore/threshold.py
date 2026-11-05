"""PatchCore 학습 정상 특징의 촬영그룹 held-out 점수로 임계값을 보정한다."""
import random
import torch
from .coreset import approximate_greedy_coreset_indices
from .scoring import compute_patch_scores


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
