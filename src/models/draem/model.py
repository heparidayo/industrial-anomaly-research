"""DRAEMPredictor — 학습된 체크포인트로 이미지 1장씩 추론.

PatchCorePredictor와 같은 모양으로 맞춰 두었습니다.
    model = DRAEMPredictor(checkpoint_path)
    model.device, model.category
    result = model.predict(image_path)
    result["image_score"]   # float, 클수록 이상
    result["anomaly_map"]   # (H, W) tensor, 0~1, 픽셀별 결함 확률
    result["image"]         # (H, W, 3) numpy, 0~1, BGR (시각화용)
    result["reconstruction"]# (H, W, 3) numpy, 0~1, BGR (시각화용)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import load_image, to_tensor
from .networks import build_networks


class DRAEMPredictor:
    def __init__(self, checkpoint_path: Path, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        self.category = saved["category"]
        self.image_size = saved["image_size"]
        self.score_kernel = saved.get("score_kernel", 21)
        self.meta = {k: v for k, v in saved.items() if not k.endswith("_state")}

        self.reconstructive, self.discriminative = build_networks(
            saved["rec_base_width"], saved["seg_base_width"]
        )
        self.reconstructive.load_state_dict(saved["reconstructive_state"])
        self.discriminative.load_state_dict(saved["discriminative_state"])
        self.reconstructive.to(self.device).eval()
        self.discriminative.to(self.device).eval()

    @torch.inference_mode()
    def predict_tensor(self, batch: torch.Tensor):
        """batch: (N, 3, H, W), 0~1, BGR → (image_scores (N,), anomaly_maps (N, H, W), reconstructions)."""
        batch = batch.to(self.device)

        reconstruction = self.reconstructive(batch)
        joined = torch.cat((reconstruction, batch), dim=1)
        mask_prob = torch.softmax(self.discriminative(joined), dim=1)[:, 1:, :, :]

        # 이미지 점수: 결함 확률 맵을 평균 풀링한 뒤 최댓값 (원본과 동일)
        k = self.score_kernel
        pooled = F.avg_pool2d(mask_prob, k, stride=1, padding=k // 2)
        image_scores = pooled.flatten(1).max(dim=1).values

        return image_scores, mask_prob[:, 0], reconstruction

    def predict(self, image_path: Path) -> dict:
        image = load_image(image_path, self.image_size)
        scores, maps, recs = self.predict_tensor(to_tensor(image).unsqueeze(0))

        return {
            "image_score": scores[0].item(),
            "anomaly_map": maps[0].cpu(),
            "image": image,
            "reconstruction": np.clip(recs[0].cpu().numpy().transpose(1, 2, 0), 0, 1),
        }
