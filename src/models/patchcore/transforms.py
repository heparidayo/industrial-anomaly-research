"""PatchCore 입력 변환 (이미지 → 모델에 넣을 숫자 텐서).

[입력 방식 두 가지]
center_crop (기본, 기존과 동일)
    WideResNet50 기본 변환: 짧은 변을 232로 맞춤 → 가운데 224×224만 자름 → 정규화.
    정사각형 사진(bottle, 회사 원본·정렬·4토막 펼치기)에 쓴다.
full
    자르지 않는다. 사진 전체를 (input_scale 배율로 크기만 바꿔서) 그대로 넣는다 → 정규화.
    가로로 긴 띠(펼친 면취 고리)처럼 정사각형이 아닌 사진에 쓴다.
    → center_crop으로 긴 띠를 넣으면 가운데 약 6%만 남고 나머지가 잘려 나간다.

[주의] 학습(메모리 뱅크 만들기)과 평가(점수 매기기)는 반드시 같은 방식을 써야 한다.
그래서 학습 때 쓴 방식을 메모리 파일에 저장하고, 평가 코드는 그 값을 읽어서 같은 변환을 만든다.
"""

import torch
from torchvision.models import Wide_ResNet50_2_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

INPUT_MODES = ("center_crop", "full")   # 쓸 수 있는 입력 방식 이름
DEFAULT_INPUT_MODE = "center_crop"      # 옵션을 안 주면 기존과 똑같이 동작
DEFAULT_INPUT_SCALE = 1.0               # full 방식의 크기 배율 (1.0 = 원본 크기 그대로)


class FullImageTransform:
    """자르지 않는 변환: (크기 조절) → 0~1 실수 텐서 → ImageNet 평균·표준편차로 정규화."""

    def __init__(self, scale, mean, std):
        self.scale = float(scale)           # 크기 배율
        self.mean = list(mean)              # 정규화 평균 (기본 변환과 같은 값)
        self.std = list(std)                # 정규화 표준편차 (기본 변환과 같은 값)

    def __call__(self, image):
        tensor = F.pil_to_tensor(image)                         # PIL 이미지 → [채널, 세로, 가로] 정수 텐서
        tensor = F.convert_image_dtype(tensor, torch.float32)   # 0~255 정수 → 0~1 실수
        if self.scale != 1.0:                                   # 배율이 1이 아니면 크기 조절
            height, width = tensor.shape[-2:]                   # 원래 세로·가로
            size = [max(1, round(height * self.scale)), max(1, round(width * self.scale))]
            tensor = F.resize(tensor, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        return F.normalize(tensor, self.mean, self.std)         # 기본 변환과 같은 정규화

    def __repr__(self):
        return f"FullImageTransform(scale={self.scale}, mean={self.mean}, std={self.std})"


def check_input_options(input_mode, input_scale):
    """입력 방식 이름과 배율이 올바른지 확인한다. 잘못되면 ValueError."""
    if input_mode not in INPUT_MODES:                                       # 모르는 이름이면 오류
        raise ValueError(f"input_mode는 {INPUT_MODES} 중 하나여야 합니다: {input_mode}")
    if not float(input_scale) > 0:                                          # 배율은 양수만
        raise ValueError(f"input_scale은 양수여야 합니다: {input_scale}")
    if input_mode == "center_crop" and float(input_scale) != 1.0:           # 기본 방식은 배율을 안 씀
        raise ValueError("input_scale은 input_mode='full'에서만 바꿀 수 있습니다.")


def make_transform(input_mode=DEFAULT_INPUT_MODE, input_scale=DEFAULT_INPUT_SCALE):
    """입력 방식에 맞는 변환을 만든다. 학습·평가·단독 실행이 모두 이 함수를 쓴다."""
    check_input_options(input_mode, input_scale)                            # 옵션 검사
    default = Wide_ResNet50_2_Weights.DEFAULT.transforms()                  # 기존 기본 변환
    if input_mode == "center_crop":
        return default                                                      # 기존과 완전히 같음
    return FullImageTransform(input_scale, default.mean, default.std)       # 자르지 않는 변환


def input_options_of(saved):
    """저장된 메모리 파일(dict)에서 입력 방식을 읽는다. 옛 파일(값 없음)은 기본값으로 본다."""
    mode = saved.get("input_mode", DEFAULT_INPUT_MODE)                      # 없으면 center_crop
    scale = float(saved.get("input_scale", DEFAULT_INPUT_SCALE))            # 없으면 1.0
    return mode, scale
