"""PatchCore 입력 방식(center_crop / full) 테스트.

- 기본(center_crop)은 기존 weights.transforms()와 완전히 같아야 한다 → 기존 실험이 안 바뀜
- full은 긴 띠를 자르지 않고 전체를 넣어야 한다
- 학습 때 쓴 입력 방식이 메모리 파일을 통해 평가(추론기)까지 그대로 이어져야 한다
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision.models import Wide_ResNet50_2_Weights

from scripts.patchcore.run_patchcore_experiment import check_input_mode, input_options_changed
from src.models.patchcore.transforms import (
    FullImageTransform, check_input_options, input_options_of, make_transform,
)


def random_image(height, width, seed=0):
    """테스트용 무작위 컬러 이미지 (PIL)."""
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


class InputTransformTests(unittest.TestCase):
    def test_default_is_exactly_the_old_transform(self):
        image = random_image(300, 300)
        old = Wide_ResNet50_2_Weights.DEFAULT.transforms()(image)          # 기존 변환 결과
        new = make_transform()(image)                                       # 옵션 없이 만든 변환 결과
        self.assertTrue(torch.equal(old, new))                              # 한 값도 다르면 안 됨
        self.assertEqual(tuple(new.shape), (3, 224, 224))

    def test_full_keeps_the_whole_strip(self):
        strip = random_image(108, 1728)                                     # 펼친 긴 띠 크기
        self.assertEqual(tuple(make_transform("full")(strip).shape), (3, 108, 1728))        # 그대로
        self.assertEqual(tuple(make_transform("full", 0.5)(strip).shape), (3, 54, 864))     # 절반
        # 비교: 기본 변환은 긴 띠의 가운데 224×224만 남긴다 (그래서 full이 필요)
        self.assertEqual(tuple(make_transform()(strip).shape), (3, 224, 224))

    def test_full_uses_same_normalization_as_default(self):
        # full도 기본 변환과 같은 정규화를 써야 한다: (픽셀/255 − 평균) / 표준편차
        # 전부 흰색(255) 이미지 → 채널마다 (1 − 평균) / 표준편차 값이 나와야 함
        image = Image.fromarray(np.full((4, 4, 3), 255, np.uint8))         # 전부 흰색
        tensor = make_transform("full")(image)
        default = Wide_ResNet50_2_Weights.DEFAULT.transforms()
        expected = [(1.0 - m) / s for m, s in zip(default.mean, default.std)]
        for channel, value in enumerate(expected):
            self.assertAlmostEqual(tensor[channel, 0, 0].item(), value, places=5)

    def test_invalid_options_rejected(self):
        with self.assertRaises(ValueError):
            check_input_options("resize", 1.0)                              # 없는 방식
        with self.assertRaises(ValueError):
            check_input_options("full", 0)                                  # 배율 0
        with self.assertRaises(ValueError):
            check_input_options("center_crop", 2.0)                         # 기본 방식에 배율 지정

    def test_saved_memory_options_and_old_files(self):
        self.assertEqual(input_options_of({}), ("center_crop", 1.0))        # 옛 메모리 파일 = 기본
        self.assertEqual(input_options_of({"input_mode": "full", "input_scale": 0.5}), ("full", 0.5))
        self.assertIsInstance(make_transform(*input_options_of({"input_mode": "full"})), FullImageTransform)

    def test_runner_rejects_memory_made_with_other_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw_memory_full.pt"
            meta = raw.with_suffix(".pt.meta.json")
            full = SimpleNamespace(input_mode="full", input_scale=1.0)       # 이번 요청: full
            old = SimpleNamespace()                                          # 옵션 없는 옛 호출 = 기본
            meta.write_text(json.dumps({"category": "x"}), encoding="utf-8")  # 옛 기록 (값 없음 = 기본)
            check_input_mode(raw, old)                                       # 기본끼리는 통과
            with self.assertRaises(ValueError):
                check_input_mode(raw, full)                                  # 기본 메모리 + full 요청 → 중단
            meta.write_text(json.dumps({"input_mode": "full", "input_scale": 1.0}), encoding="utf-8")
            check_input_mode(raw, full)                                      # 같은 방식은 통과
            self.assertFalse(input_options_changed(old))                     # 기본이면 기록에 안 남김
            self.assertTrue(input_options_changed(full))


if __name__ == "__main__":
    unittest.main()
