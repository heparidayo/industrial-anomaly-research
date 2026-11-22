"""
EfficientAD 전체 모델 묶음 — 논문 Algorithm 2 (추론).

Teacher, Student, 오토인코더와 함께 추론에 필요한 값
(Teacher 채널 평균·표준편차, 맵 정규화 분위수 4개)을 하나의 nn.Module에 담아
체크포인트 파일 하나로 저장·불러오기 할 수 있게 합니다.

추론 순서 (Algorithm 2):
  1) T, S, A 순전파
  2) Teacher 출력 채널 정규화
  3) 로컬 맵  = 채널 평균 (T̂ − S[:384])²
     글로벌 맵 = 채널 평균 (A − S[384:])²
  4) 256×256으로 bilinear 리사이즈
  5) 분위수 정규화: 0.1 · (M − q_a) / (q_b − q_a)
  6) 결합 맵 = 0.5 · 로컬 + 0.5 · 글로벌, 이미지 점수 = 결합 맵 최댓값
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import get_autoencoder
from pdn import get_pdn
from student import TEACHER_CH, get_student, split_student_output


def _channel_mse_map(a: torch.Tensor, b: torch.Tensor, out_size: int) -> torch.Tensor:
    m = ((a - b) ** 2).mean(dim=1, keepdim=True)
    if m.shape[-1] != out_size:
        m = F.interpolate(m, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return m


class EfficientAD(nn.Module):
    def __init__(self, size: str = "s", padding: bool = True, image_size: int = 256):
        super().__init__()
        self.size, self.padding, self.image_size = size, padding, image_size
        self.teacher = get_pdn(size, TEACHER_CH, padding)
        self.student = get_student(size, padding)
        with torch.no_grad():
            feat = self.teacher(torch.zeros(1, 3, image_size, image_size)).shape[-1]
        self.feat_size = feat                       # 64 (패딩) 또는 56 (무패딩)
        self.autoencoder = get_autoencoder(out_size=feat)
        self.register_buffer("t_mean", torch.zeros(1, TEACHER_CH, 1, 1))
        self.register_buffer("t_std", torch.ones(1, TEACHER_CH, 1, 1))
        # q_st_a, q_st_b, q_ae_a, q_ae_b
        self.register_buffer("quantiles", torch.tensor([0.0, 1.0, 0.0, 1.0]))

    # -------------------------------------------------------------- 학습 보조
    def freeze_teacher(self):
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def normalize_teacher(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.t_mean) / self.t_std

    # -------------------------------------------------------------- 추론
    @torch.no_grad()
    def raw_maps(self, x: torch.Tensor):
        """정규화 전 로컬·글로벌 맵 (B, 1, image_size, image_size)."""
        t = self.normalize_teacher(self.teacher(x))
        s_st, s_ae = split_student_output(self.student(x))
        a = self.autoencoder(x)
        return (_channel_mse_map(t, s_st, self.image_size),
                _channel_mse_map(a, s_ae, self.image_size))

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """반환: (결합 맵, 정규화된 로컬 맵, 정규화된 글로벌 맵), 각각 (B, 1, H, W)."""
        m_st, m_ae = self.raw_maps(x)
        q = self.quantiles.to(m_st.dtype)
        n_st = 0.1 * (m_st - q[0]) / (q[1] - q[0])
        n_ae = 0.1 * (m_ae - q[2]) / (q[3] - q[2])
        return 0.5 * n_st + 0.5 * n_ae, n_st, n_ae

    @staticmethod
    def image_score(m: torch.Tensor) -> torch.Tensor:
        return m.flatten(1).amax(dim=1)

    # -------------------------------------------------------------- 저장·불러오기
    def save(self, path: str, meta: dict | None = None):
        torch.save({"state_dict": self.state_dict(), "size": self.size, "padding": self.padding,
                    "image_size": self.image_size, "meta": meta or {}}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "EfficientAD":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(ckpt["size"], ckpt["padding"], ckpt.get("image_size", 256))
        state = dict(ckpt["state_dict"])
        for k in ("weight", "bias"):                 # 이전 버전 호환: 오토인코더 압축층 이름 변경
            old = f"autoencoder.encoder.10.{k}"
            if old in state:
                state[f"autoencoder.to_latent.{k}"] = state.pop(old)
        model.load_state_dict(state)
        model.meta = ckpt.get("meta", {})
        return model.eval()
