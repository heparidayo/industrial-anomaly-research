# Memory bank 후보 생성


import torch
from PIL import Image


def collect_normal_features(
    image_paths,
    backbone,
    patch_extractor,
    transform,
    device,
):
    """정상 이미지에서 Patch Feature를 추출하여 CPU에 수집한다."""

    if not image_paths:
        raise ValueError("정상 이미지가 없습니다.")

    memory = None
    patches_per_image = None

    for index, image_path in enumerate(image_paths):

        # 1. 이미지 읽기 및 텐서 변환
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

        # 2. CNN 및 Patch Feature 추출
        with torch.inference_mode(): # 여기서도 미분 생략 (학습x)
            feature_maps = backbone.extract(tensor)
            embeddings = patch_extractor(feature_maps)

            # (1, 784, 1024) -> (784, 1024)
            patches = embeddings.squeeze(0).cpu()

        # 3. 첫 이미지에서 저장 공간 확보
        if memory is None:
            patches_per_image, dim = patches.shape

            memory = torch.empty(
                (len(image_paths) * patches_per_image, dim),
                dtype=torch.float32,
                device="cpu",
            )

        if patches.shape != (patches_per_image, memory.shape[1]):
            raise ValueError(
                f"특징 크기가 일치하지 않습니다: {image_path}"
            )

        # 4. 미리 확보한 메모리에 특징 저장
        start = index * patches_per_image
        end = start + patches_per_image

        memory[start:end].copy_(patches)

        if (index + 1) % 10 == 0 or index + 1 == len(image_paths):
            print(f"[특징 추출] {index + 1}/{len(image_paths)}")

    return memory, patches_per_image
