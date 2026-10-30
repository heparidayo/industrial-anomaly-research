
# [이 파일의 역할]
# memory_bank.py가 모은 정상 patch 벡터(bottle 기준 163,856개)는 너무 많다.
# 테스트할 때 매번 이 전부와 거리를 재면 느리고 메모리도 많이 든다.
# 그래서 전체를 "골고루 대표하는" 일부만 골라낸다. 이것이 Coreset이다.
#
# 1%면 163,856개 → 1,638개 (약 100배 작아짐)
#
# [고르는 방법: Greedy(탐욕) Coreset]
# "지금까지 뽑은 대표들로부터 가장 멀리 떨어진 벡터"를 하나씩 계속 뽑는다.
# → 비슷한 벡터가 몰려 있는 곳에서는 조금만, 드문 벡터도 빠짐없이 뽑히게 되어
#   적은 수로도 정상 데이터의 전체 범위를 넓게 덮을 수 있다.
#   (그냥 무작위로 뽑으면 흔한 패턴만 뽑히고 드문 정상 패턴을 놓치기 쉽다)
#
# [Approximate(근사)인 이유]
# 1024차원 그대로 거리를 재면 느리므로, 무작위 투영으로 128차원으로 줄인 뒤
# 그 공간에서 거리를 재서 고른다. 고른 "번호(인덱스)"만 돌려주고,
# 실제 메모리 뱅크에는 원래 1024차원 벡터를 넣는다 (build_patchcore_coreset.py).

import math
import torch

# 224x244 기준 10% Coreset 설정
# https://github.com/amazon-science/patchcore-inspection/blob/main/sample_training.sh
# (참고: 위 주석은 공식 예제 설정을 가리킨다. 이 함수의 기본값은 percentage=0.01(1%)이고,
#  실제 비율은 build_patchcore_coreset.py의 --percentage 인자로 정한다)

# no_grad: 기울기 계산을 끈다 (학습하지 않으므로 불필요, 메모리 절약)
@torch.no_grad()
def approximate_greedy_coreset_indices(
    features,           # [N, D] 정상 patch 벡터 전체 (예: [163856, 1024])
    percentage=0.01,    # 남길 비율 (0.01 = 1%)
    projection_dim=128, # 거리 계산을 위해 줄일 차원
    seed=0,             # 랜덤 시드 (같으면 매번 같은 결과)
    device=None,        # 계산 장치 (None이면 GPU가 있으면 GPU)
    batch_size=4096,    # 투영할 때 한 번에 GPU로 올릴 벡터 수
    log_every=100,      # 몇 개 뽑을 때마다 진행 상황을 출력할지
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

    # ----- 입력값 검사 -----
    # features는 2차원 표 [개수, 차원]이어야 하고 비어 있으면 안 된다
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features는 [N, D] 텐서여야 합니다.")

    # 비율은 0보다 크고 1 이하
    if not 0 < percentage <= 1:
        raise ValueError("percentage는 (0, 1] 범위여야 합니다.")

    if projection_dim <= 0 or batch_size <= 0:
        raise ValueError("투영 차원과 배치 크기는 양수여야 합니다.")

    # n = 전체 벡터 수, original_dim = 원래 차원 (1024)
    n, original_dim = features.shape
    # 뽑을 개수 (최소 1개). 예) 163856 * 0.01 = 1638
    num_select = max(1, int(n * percentage))

    # 100%를 고르면 전부 고르는 것이므로 계산 없이 0~n-1 전체 번호를 돌려준다
    if num_select == n:
        return torch.arange(n, dtype=torch.long)

    # 계산 장치 결정
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    # 1. 랜덤 시드 고정
    # 이 함수 전용 난수 생성기를 만든다.
    # seed가 같으면 투영 행렬과 시작점이 매번 똑같이 나와서 결과가 재현된다.
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    # 2. 가우시안 무작위 투영 행렬 생성
    # [1024, 128]
    # 정규분포 난수로 채운 행렬을 곱하면 1024차원 → 128차원으로 줄어든다.
    # (Johnson-Lindenstrauss 원리: 무작위 투영을 해도 점들 사이의 "거리 관계"는
    #  대략 유지된다. 그래서 줄인 공간에서 멀고 가까움을 판단해도 괜찮다)
    # sqrt(128)로 나누는 것은 투영 후 거리의 크기를 원래와 비슷하게 맞추기 위함
    projection = torch.randn(
        original_dim,
        projection_dim,
        generator=rng,
        dtype=torch.float32,
    ).div_(math.sqrt(projection_dim)).to(device)

    # GPU에는 축소된 특징만 저장
    # 투영 결과를 담을 빈 표 [N, 128] (163856 x 128 ≈ 84MB)
    # 원본 [N, 1024] 전체를 GPU에 올리지 않아도 된다
    projected = torch.empty(
        (n, projection_dim),
        dtype=torch.float32,
        device=device,
    )

    # 원본을 4096개씩 잘라서 GPU로 올리고 → 투영 → 결과 표에 채운다
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        chunk = features[start:end].to(
            device=device,
            dtype=torch.float32,
        )

        # @ = 행렬곱. [chunk, 1024] @ [1024, 128] = [chunk, 128]
        projected[start:end] = chunk @ projection

    print("투영 후 특징 크기:", tuple(projected.shape))

    # 3. 시작 기준 특징 10개 무작위 선택
    # Greedy 방식은 "이미 뽑은 대표들로부터 먼 것"을 고르는데,
    # 맨 처음에는 뽑은 대표가 없으므로 임시 기준점이 필요하다.
    num_starts = min(10, n)

    # 0~n-1을 무작위로 섞은 뒤 앞의 10개를 임시 기준점으로 사용
    start_indices = torch.randperm(
        n, generator=rng
    )[:num_starts].to(device)

    # 각 특징에서 초기 기준점까지의 평균 거리
    # cdist: 모든 벡터 [N]와 기준점 10개 사이의 거리표 [N, 10]
    # mean(dim=1): 벡터마다 10개 거리의 평균 → [N]
    # nearest_distances[i] = "벡터 i가 현재 대표 집합에서 얼마나 떨어져 있는가"
    nearest_distances = torch.cdist(
        projected,
        projected[start_indices],
    ).mean(dim=1)

    # 뽑힌 벡터의 번호를 모을 목록
    selected_indices = []

    # 4. Greedy Coreset 선택
    # num_select번(예: 1638번) 반복하며 하나씩 뽑는다
    for step in range(num_select):

        # 현재 대표 집합에서 가장 먼 특징 선택
        # argmax: 거리가 가장 큰 벡터의 번호
        # = 지금까지의 대표들로 가장 설명이 안 되는(가장 동떨어진) 벡터
        index = int(torch.argmax(nearest_distances).item())

        selected_indices.append(index)

        # 새 대표 특징까지의 거리 계산
        # 모든 벡터 [N]와 방금 뽑은 벡터 1개 사이의 거리 → [N]
        distances = torch.cdist(
            projected,
            projected[index:index + 1],
        ).squeeze(1)

        # 기존 최근접 거리와 비교해 갱신
        # 새 대표가 추가됐으니, 각 벡터의 "가장 가까운 대표까지의 거리"는
        # 기존 값과 새 대표까지의 거리 중 더 작은 쪽이 된다.
        # → 새 대표 근처에 있던 벡터들은 거리가 작아져서 다음에 뽑힐 가능성이 줄어든다
        nearest_distances = torch.minimum(
            nearest_distances,
            distances,
        )

        # 동일한 인덱스 중복 선택 방지
        # 방금 뽑은 벡터는 거리를 -1로 만들어 다시는 argmax에 걸리지 않게 한다
        nearest_distances[index] = -1.0

        # 진행 상황 출력 (100개마다, 그리고 마지막에)
        if (
            (step + 1) % log_every == 0
            or step + 1 == num_select
        ):
            print(f"[Coreset] {step + 1}/{num_select}")

    # 뽑힌 번호들을 텐서로 반환 [num_select]
    # (벡터 자체가 아니라 "몇 번째 벡터인지"만 돌려준다)
    return torch.tensor(
        selected_indices,
        dtype=torch.long,
    )
