# Memory bank 후보 생성

# [이 파일의 역할]
# PatchCore의 "학습" 단계에 해당한다. (실제로는 신경망 학습이 아니라 "기억"이다)
# 정상 학습 이미지(train/good) 전부를 하나씩 넣어서
# 각 이미지의 784개 patch 벡터를 모두 모아 하나의 큰 표(memory)에 저장한다.
#
# 이 표가 "정상이란 이렇게 생긴 것"이라는 기준이 된다.
# 나중에 테스트 이미지의 patch가 이 표의 어떤 벡터와도 멀다면 → 이상(anomaly)
#
# bottle 기준: 209장 x 784 patch = 163,856개 벡터 (각 1024차원, 약 640MB)
# 이 원본은 너무 크기 때문에 다음 단계(coreset.py)에서 대표만 골라 줄인다.


import torch
from PIL import Image


def collect_normal_features(
    image_paths,      # 정상 학습 이미지 경로 목록
    backbone,         # backbone.py의 PatchCoreBackbone
    patch_extractor,  # features.py의 PatchFeatureExtractor
    transform,        # 이미지 → 텐서 전처리 (크기 조정, 정규화 등)
    device,           # 연산 장치 (cuda/cpu)
):
    """정상 이미지에서 Patch Feature를 추출하여 CPU에 수집한다."""
    # 결과를 GPU가 아닌 CPU(일반 RAM)에 모으는 이유:
    # 640MB 이상이 될 수 있어서 GPU 메모리를 아끼기 위함

    if not image_paths:
        raise ValueError("정상 이미지가 없습니다.")

    # 아직 첫 이미지를 처리하기 전이라 크기를 모르므로 None으로 시작
    memory = None
    patches_per_image = None

    # enumerate: index(0, 1, 2, ...)와 경로를 함께 꺼낸다
    for index, image_path in enumerate(image_paths):

        # 1. 이미지 읽기 및 텐서 변환
        # with 구문: 블록이 끝나면 파일을 자동으로 닫는다
        with Image.open(image_path) as image:
            # 흑백/투명도(RGBA) 이미지가 섞여 있어도 항상 3채널 RGB로 통일
            image = image.convert("RGB")
            # transform: 크기 조정 + 가운데 자르기(224x224) + ImageNet 기준 정규화
            # unsqueeze(0): [3, 224, 224] → [1, 3, 224, 224] (배치 차원 추가, 1장짜리 묶음)
            tensor = transform(image).unsqueeze(0).to(device)

        # 2. CNN 및 Patch Feature 추출
        with torch.inference_mode(): # 여기서도 미분 생략 (학습x)
            feature_maps = backbone.extract(tensor)   # layer2, layer3 특징 맵
            embeddings = patch_extractor(feature_maps) # [1, 784, 1024]

            # (1, 784, 1024) -> (784, 1024)
            # 배치 차원을 떼고, GPU → CPU로 옮긴다
            patches = embeddings.squeeze(0).cpu()

        # 3. 첫 이미지에서 저장 공간 확보
        # 첫 이미지를 처리해봐야 이미지당 patch 수(784)와 차원(1024)을 알 수 있다.
        # 그때 "이미지 수 x 784" 행짜리 빈 표를 한 번에 만들어 둔다.
        # (매번 이어붙이는 것보다 미리 만들어 채우는 쪽이 훨씬 빠르고 메모리도 덜 쓴다)
        if memory is None:
            patches_per_image, dim = patches.shape

            memory = torch.empty(
                (len(image_paths) * patches_per_image, dim),
                dtype=torch.float32,
                device="cpu",
            )

        # 모든 이미지는 같은 크기로 전처리되므로 patch 모양도 같아야 한다.
        # 다르면 데이터나 전처리에 문제가 있는 것이므로 즉시 멈춘다.
        if patches.shape != (patches_per_image, memory.shape[1]):
            raise ValueError(
                f"특징 크기가 일치하지 않습니다: {image_path}"
            )

        # 4. 미리 확보한 메모리에 특징 저장
        # 이미지 index번째가 들어갈 행 범위를 계산
        # 예) index=0 → 0~783행, index=1 → 784~1567행, ...
        start = index * patches_per_image
        end = start + patches_per_image

        # 해당 행 범위에 이번 이미지의 patch 벡터를 복사
        memory[start:end].copy_(patches)

        # 진행 상황 출력 (10장마다, 그리고 마지막 장에서)
        if (index + 1) % 10 == 0 or index + 1 == len(image_paths):
            print(f"[특징 추출] {index + 1}/{len(image_paths)}")

    # memory: [이미지 수 * 784, 1024] 전체 정상 patch 벡터
    # patches_per_image: 이미지당 patch 수 (784)
    return memory, patches_per_image
