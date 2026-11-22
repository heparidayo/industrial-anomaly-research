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

import cv2

from .dataset import load_image, to_tensor
from .networks import build_networks
from .preprocess import preprocess, preprocess_polar, rewrap, rotate


class DRAEMPredictor:
    def __init__(self, checkpoint_path: Path, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        self.category = saved["category"]
        self.image_size = saved["image_size"]
        self.score_kernel = saved.get("score_kernel", 21)
        self.meta = {k: v for k, v in saved.items() if not k.endswith("_state")}
        self.clahe = bool(saved.get("clahe", False))
        self.align = bool(saved.get("align_spring", False))
        self.polar = bool(saved.get("polar", False))
        self.rect = tuple(saved["rect"]) if saved.get("rect") else None   # (높이, 너비) 직사각형 띠 모델

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

    def predict(self, image_path: Path, offset: float = 0.0) -> dict:
        """offset: 채점용 각도 증강(°). 이미지를 그만큼 돌려 추론하고, 결함 맵은 원래 방향으로 되돌려 반환."""
        if self.rect:
            # 이미 펴진 띠(또는 적층) 이미지를 (높이, 너비)로 줄여 그대로 추론
            image = load_image(image_path, self.rect)
            scores, maps, recs = self.predict_tensor(to_tensor(image).unsqueeze(0))
            return {
                "image_score": scores[0].item(),
                "anomaly_map": maps[0].float().cpu(),
                "image": image,
                "reconstruction": np.clip(recs[0].float().cpu().numpy().transpose(1, 2, 0), 0, 1),
            }
        image = load_image(image_path, self.image_size)  # 시각화용 원본 (원래 방향)
        if self.polar:
            # 띠 펴기 → 띠에서 점수 계산 → 결함 맵은 원래 이미지 좌표로 되돌림 (띠 밖 = 0)
            raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            strip, center = preprocess_polar(raw, self.clahe, offset)
            center = center + offset  # 띠 첫 열이 실제로 놓인 각도
            scores, maps, recs = self.predict_tensor(to_tensor(strip.astype(np.float32) / 255.0).unsqueeze(0))
            smap = maps[0].float().cpu().numpy()
            srec = np.clip(recs[0].float().cpu().numpy().transpose(1, 2, 0), 0, 1)
            return {
                "image_score": scores[0].item(),
                "anomaly_map": torch.from_numpy(np.ascontiguousarray(rewrap(smap, center, self.image_size))),
                "image": image,
                "reconstruction": rewrap(srec, center, self.image_size, fill=image),
                "strip": strip.astype(np.float32) / 255.0,
                "strip_map": smap,
            }
        if offset:
            return self._predict_rotated(image_path, image, offset)
        if not (self.clahe or self.align):
            scores, maps, recs = self.predict_tensor(to_tensor(image).unsqueeze(0))
            return {
                "image_score": scores[0].item(),
                "anomaly_map": maps[0].cpu(),
                "image": image,
                "reconstruction": np.clip(recs[0].cpu().numpy().transpose(1, 2, 0), 0, 1),
            }

        # 학습과 같은 전처리(조명 정규화 · 스프링 시작점 정렬) → 결함 맵을 원래 방향으로 되돌림
        raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        model_in, angle = preprocess(raw, self.image_size, self.clahe, self.align)
        scores, maps, recs = self.predict_tensor(to_tensor(model_in.astype(np.float32) / 255.0).unsqueeze(0))
        amap = maps[0].float().cpu().numpy()
        rec = np.clip(recs[0].float().cpu().numpy().transpose(1, 2, 0), 0, 1)
        if angle:
            amap = rotate(amap, -angle)
            rec = rotate(rec, -angle)
        return {
            "image_score": scores[0].item(),  # 점수는 돌린 상태의 맵에서 (회전과 무관)
            "anomaly_map": torch.from_numpy(np.ascontiguousarray(amap)),
            "image": image,
            "reconstruction": rec,
        }

    def _predict_rotated(self, image_path: Path, image: np.ndarray, offset: float) -> dict:
        """띠 펴기가 아닌 모델의 각도 증강 채점: 원본을 offset만큼 돌려 추론 → 맵을 되돌림."""
        raw = rotate(cv2.imread(str(image_path), cv2.IMREAD_COLOR), offset)
        model_in, angle = preprocess(raw, self.image_size, self.clahe, self.align)
        scores, maps, _ = self.predict_tensor(to_tensor(model_in.astype(np.float32) / 255.0).unsqueeze(0))
        amap = rotate(maps[0].float().cpu().numpy(), -angle - offset) if (angle or offset) else maps[0].float().cpu().numpy()
        return {
            "image_score": scores[0].item(),
            "anomaly_map": torch.from_numpy(np.ascontiguousarray(amap)),
            "image": image,
            "reconstruction": image,
        }
