# PatchCore threshold(불량 판정 기준 점수) 계산
#
# 학습 정상 이미지를 K개 그룹으로 나눠 돌아가며 한 그룹씩 빼고
#   1) 나머지 이미지로 coreset 메모리 뱅크를 만들고
#   2) 뺀 그룹의 이미지를 채점한다 → "처음 보는 정상 이미지의 점수"
# 모든 그룹을 한 번씩 빼면 학습 정상 이미지 전부에 대해 점수가 하나씩 생긴다.
#
# 학습 이미지를 그대로 채점하면 안 되는 이유:
#   그 이미지는 메모리 뱅크 안에 들어 있어서 자기 자신과의 거리가 0에 가까워진다.

import torch

from .coreset import approximate_greedy_coreset_indices
from .scoring import compute_patch_scores


def held_out_scores(
    per_image,          # [이미지 수, 784, 1024] 학습 정상 patch 벡터 (이미지별로 묶은 것)
    folds,              # 그룹 수 (이미지 수와 같으면 한 장씩 빼는 leave-one-out)
    percentage,         # coreset 비율 (실제 추론용 메모리 뱅크와 같아야 함)
    projection_dim,     # coreset 투영 차원
    seed,               # coreset 시드
    device,             # 거리 계산 장치
    log=print,          # 그룹별 진행 출력 함수 (None이면 출력 안 함)
):
    """학습 정상 이미지마다 held-out 이미지 점수를 계산해 리스트로 반환한다."""
    image_count = per_image.shape[0]

    # 이미지 i는 (i % folds)번 그룹 → 그룹마다 이미지가 골고루 섞인다
    # 예) 이미지 25장, 10그룹 → 0번 그룹 = 0, 10, 20번 이미지 / 1번 그룹 = 1, 11, 21번 ...
    fold_of = torch.arange(image_count) % folds
    # 이미지별 held-out 점수를 채울 자리 (처음엔 비어 있음)
    scores = [None] * image_count

    for fold in range(folds):
        # 이번에 뺄 이미지 번호들 / 남길 이미지 번호들
        # nonzero(): 조건이 True인 위치 번호, flatten(): 1차원 목록으로
        held_out = (fold_of == fold).nonzero().flatten()
        kept = (fold_of != fold).nonzero().flatten()

        # 1. 뺀 그룹을 제외한 정상 이미지로 coreset 메모리 뱅크 생성
        # 남긴 이미지들의 patch 벡터를 다시 한 줄로 이어 붙인다 → [남긴 이미지 수 * 784, 1024]
        kept_features = per_image[kept].reshape(-1, per_image.shape[-1])
        # 실제 추론용 메모리 뱅크와 같은 방식(같은 비율, 투영 차원, 시드)으로 대표 벡터 선택
        indices = approximate_greedy_coreset_indices(
            features=kept_features,
            percentage=percentage,
            projection_dim=projection_dim,
            seed=seed,
            log_every=10**9,  # 진행 로그 생략
        )
        # 선택된 번호의 벡터만 뽑아서 계산 장치로 (indices가 GPU에 있을 수 있어 CPU로 맞춤)
        memory_bank = kept_features.index_select(0, indices.cpu()).to(device)

        # 2. 뺀 그룹의 이미지 채점 (이미지 점수 = patch 점수 최댓값)
        # 실제 추론(model.py)과 똑같이 계산한다
        for i in held_out.tolist():
            # per_image[i]: [784, 1024] → unsqueeze(0): [1, 784, 1024] (배치 차원 추가)
            patch_scores = compute_patch_scores(
                per_image[i].unsqueeze(0).to(device),
                memory_bank,
            )
            # 784개 patch 점수 중 가장 큰 값 = 이 이미지의 이상 점수
            scores[i] = patch_scores.max().item()

        if log:
            log(f"[그룹 {fold + 1}/{folds}] 제외 {len(held_out)}장, "
                f"메모리 뱅크 {tuple(memory_bank.shape)}")

    return scores
