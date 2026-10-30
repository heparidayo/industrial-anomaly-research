
# [이 파일의 역할]
# PatchCore가 "이상하다"를 판단하는 핵심 계산.
#
# 테스트 이미지의 patch 벡터 하나하나에 대해
# 메모리 뱅크(정상 대표 벡터들) 중 가장 가까운 것까지의 거리를 잰다.
#   - 거리가 작다 → 비슷한 정상 patch가 있다 → 정상
#   - 거리가 크다 → 어떤 정상 patch와도 닮지 않았다 → 이상
#
# 이 거리가 바로 patch별 Anomaly Score다.
# (이미지 전체 점수는 이 중 최댓값을 쓴다 → model.py, run_patchcore.py 참고)

import torch


# inference_mode: 기울기 계산을 끈다 (학습하지 않으므로 불필요)
@torch.inference_mode()
def compute_patch_scores(
    patch_features,  # 테스트 이미지의 patch 벡터 [B, N, D] (예: [1, 784, 1024])
    memory_bank,     # 정상 대표 벡터 [M, D] (예: [1638, 1024])
    chunk_size=256,  # 한 번에 거리를 계산할 patch 수
):
    """
    테스트 Patch와 Memory Bank의 최근접 거리를 계산한다.

    patch_features: [B, N, D]
    memory_bank:    [M, D]

    반환: [B, N] Patch별 Anomaly Score
    """

    # ----- 입력값 검사 -----
    # patch_features는 3차원, memory_bank는 2차원이어야 한다
    if patch_features.ndim != 3 or memory_bank.ndim != 2:
        raise ValueError("입력 텐서 차원이 올바르지 않습니다.")

    # B=이미지 수, N=이미지당 patch 수(784), D=벡터 차원(1024)
    batch, num_patches, dim = patch_features.shape

    # 메모리 뱅크가 비어 있거나, 벡터 차원이 다르면 거리를 잴 수 없다
    if memory_bank.shape[0] == 0 or memory_bank.shape[1] != dim:
        raise ValueError("Memory Bank의 크기가 올바르지 않습니다.")

    # 하나는 GPU, 하나는 CPU에 있으면 계산이 안 된다
    if patch_features.device != memory_bank.device:
        raise ValueError("두 텐서는 같은 장치에 있어야 합니다.")

    if chunk_size < 1:
        raise ValueError("chunk_size는 양수여야 합니다.")

    # [B, N, D] -> [B*N, D]
    # 이미지 구분 없이 모든 patch를 한 줄로 늘어놓는다
    queries = patch_features.reshape(-1, dim)

    # 조각별 결과를 모을 목록
    scores = []

    # 메모리 사용량을 줄이기 위해 나누어 계산
    # 784개 patch 전부와 1638개 메모리를 한 번에 계산하면 거리표가 커지므로
    # 256개씩 나눠 계산한다 (결과는 한 번에 한 것과 똑같다)
    for start in range(0, len(queries), chunk_size):
        end = start + chunk_size
        query_chunk = queries[start:end]

        # [chunk, M] 유클리드 거리
        # cdist: 이번 조각의 patch 각각과 메모리 뱅크 벡터 각각 사이의 거리표
        # p=2: 유클리드 거리 (각 차원 차이의 제곱합에 루트, 일반적인 "직선 거리")
        distances = torch.cdist(
            query_chunk,
            memory_bank,
            p=2,
        )

        # 가장 가까운 정상 특징까지의 거리
        # min(dim=1): patch마다 M개 거리 중 가장 작은 값 → [chunk]
        # (.values: 최솟값 자체. 몇 번째 메모리였는지(.indices)는 쓰지 않는다)
        nearest = distances.min(dim=1).values
        scores.append(nearest)

    # 조각들을 이어붙이고 이미지별로 다시 나눈다 → [B, N]
    # 예) [1, 784]: 이미지 1장의 784개 위치 각각의 이상 점수
    return torch.cat(scores).reshape(batch, num_patches)
