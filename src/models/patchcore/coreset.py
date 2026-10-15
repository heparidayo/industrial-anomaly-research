
import math
import torch

# 224x244 기준 10% Coreset 설정
# https://github.com/amazon-science/patchcore-inspection/blob/main/sample_training.sh

@torch.no_grad()
def approximate_greedy_coreset_indices(
    features,
    percentage=0.01,
    projection_dim=128,
    seed=0,
    device=None,
    batch_size=4096,
    log_every=100,
):
    """
    Approximate Greedy Coreset으로 대표 특징의 인덱스를 선택한다.

    Args:
        features: [N, D] 원본 정상 특징
        percentage: 선택 비율 (0.01 = 1%)
        projection_dim: 거리 계산용 투영 차원
        seed: 랜덤 시드

    Returns:
        indices: 선택된 원본 특징의 인덱스 [K]
    """

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features는 [N, D] 텐서여야 합니다.")

    if not 0 < percentage <= 1:
        raise ValueError("percentage는 (0, 1] 범위여야 합니다.")

    if projection_dim <= 0 or batch_size <= 0:
        raise ValueError("투영 차원과 배치 크기는 양수여야 합니다.")

    n, original_dim = features.shape
    num_select = max(1, int(n * percentage))

    if num_select == n:
        return torch.arange(n, dtype=torch.long)

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    # 1. 랜덤 시드 고정
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    # 2. 가우시안 무작위 투영 행렬 생성
    # [1024, 128]
    projection = torch.randn(
        original_dim,
        projection_dim,
        generator=rng,
        dtype=torch.float32,
    ).div_(math.sqrt(projection_dim)).to(device)

    # GPU에는 축소된 특징만 저장
    projected = torch.empty(
        (n, projection_dim),
        dtype=torch.float32,
        device=device,
    )

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        chunk = features[start:end].to(
            device=device,
            dtype=torch.float32,
        )

        projected[start:end] = chunk @ projection

    print("투영 후 특징 크기:", tuple(projected.shape))

    # 3. 시작 기준 특징 10개 무작위 선택
    num_starts = min(10, n)

    start_indices = torch.randperm(
        n, generator=rng
    )[:num_starts].to(device)

    # 각 특징에서 초기 기준점까지의 평균 거리
    nearest_distances = torch.cdist(
        projected,
        projected[start_indices],
    ).mean(dim=1)

    selected_indices = []

    # 4. Greedy Coreset 선택
    for step in range(num_select):

        # 현재 대표 집합에서 가장 먼 특징 선택
        index = int(torch.argmax(nearest_distances).item())

        selected_indices.append(index)

        # 새 대표 특징까지의 거리 계산
        distances = torch.cdist(
            projected,
            projected[index:index + 1],
        ).squeeze(1)

        # 기존 최근접 거리와 비교해 갱신
        nearest_distances = torch.minimum(
            nearest_distances,
            distances,
        )

        # 동일한 인덱스 중복 선택 방지
        nearest_distances[index] = -1.0

        if (
            (step + 1) % log_every == 0
            or step + 1 == num_select
        ):
            print(f"[Coreset] {step + 1}/{num_select}")

    return torch.tensor(
        selected_indices,
        dtype=torch.long,
    )
