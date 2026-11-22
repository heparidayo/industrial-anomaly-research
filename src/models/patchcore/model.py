
from pathlib import Path

import torch
from PIL import Image

from .backbone import PatchCoreBackbone
from .transforms import input_options_of, make_transform  # 입력 변환 (학습과 같은 방식)
from .features import PatchFeatureExtractor
from .scoring import compute_patch_scores
from .anomaly_map import create_anomaly_map


class PatchCorePredictor:
    """저장된 Memory Bank를 이용한 PatchCore 추론기"""

    def __init__(self, memory_path, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        saved = torch.load(
            Path(memory_path),
            map_location="cpu",
            weights_only=True,
        )

        self.category = saved.get("category")

        self.memory_bank = saved["embeddings"].to(self.device)

        # 메모리를 만들 때(학습) 쓴 입력 방식을 읽어서 같은 변환을 만든다
        # (옛 메모리 파일은 값이 없어서 기본 = 기존 weights.transforms()와 같음)
        self.input_mode, self.input_scale = input_options_of(saved)
        self.transform = make_transform(self.input_mode, self.input_scale)

        self.backbone = PatchCoreBackbone(self.device)

        self.patch_extractor = PatchFeatureExtractor(
            patch_size=3,
            pretrain_embed_dim=1024,
            target_embed_dim=1024,
        ).to(self.device)

        self.patch_extractor.eval()

    @torch.inference_mode()
    def predict(self, image_path, with_map=False):
        """한 장의 이미지를 추론한다."""

        with Image.open(image_path) as image:
            tensor = self.transform(
                image.convert("RGB")
            ).unsqueeze(0).to(self.device)

        feature_maps = self.backbone.extract(tensor)
        patch_features = self.patch_extractor(feature_maps)

        patch_scores = compute_patch_scores(
            patch_features,
            self.memory_bank,
        )

        result = {
            "image_score": patch_scores.max().item(),
            "patch_count": patch_scores.shape[1],
        }

        if with_map:
            anomaly_map = create_anomaly_map(
                patch_scores,
                grid_size=feature_maps["layer2"].shape[-2:],
                image_size=tensor.shape[-2:],
                sigma=4.0,
            )

            result["anomaly_map"] = anomaly_map[0].cpu()
            result["image_tensor"] = tensor[0].cpu()

        return result
