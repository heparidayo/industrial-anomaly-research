
import torch


@torch.inference_mode()
def compute_patch_scores(
    patch_features,
    memory_bank,
    chunk_size=256,
):
    """
    테스트 Patch와 Memory Bank의 최근접 거리를 계산한다.

    patch_features: [B, N, D]
    memory_bank:    [M, D]

    반환: [B, N] Patch별 Anomaly Score
    """

    if patch_features.ndim != 3 or memory_bank.ndim != 2:
        raise ValueError("입력 텐서 차원이 올바르지 않습니다.")

    batch, num_patches, dim = patch_features.shape

    if memory_bank.shape[0] == 0 or memory_bank.shape[1] != dim:
        raise ValueError("Memory Bank의 크기가 올바르지 않습니다.")

    if patch_features.device != memory_bank.device:
        raise ValueError("두 텐서는 같은 장치에 있어야 합니다.")

    if chunk_size < 1:
        raise ValueError("chunk_size는 양수여야 합니다.")

    # [B, N, D] -> [B*N, D]
    queries = patch_features.reshape(-1, dim)

    scores = []

    # 메모리 사용량을 줄이기 위해 나누어 계산
    for start in range(0, len(queries), chunk_size):
        end = start + chunk_size
        query_chunk = queries[start:end]

        # [chunk, M] 유클리드 거리
        distances = torch.cdist(
            query_chunk,
            memory_bank,
            p=2,
        )

        # 가장 가까운 정상 특징까지의 거리
        nearest = distances.min(dim=1).values
        scores.append(nearest)

    return torch.cat(scores).reshape(batch, num_patches)
