"""DRAEM 실행 환경 점검.

공용 .venv에 requirements-draem.txt를 설치한 뒤 실행합니다.
    .\\.venv\\Scripts\\python.exe scripts/draem/check_draem_env.py

1) 공용 패키지 버전 출력
2) imgaug 호환 패치 import
3) DRAEM 원본 data_loader가 쓰는 augmenter 전부를 실제 이미지에 한 번씩 적용
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
import cv2  # noqa: E402

print(f"python      {sys.version.split()[0]}")
print(f"torch       {torch.__version__}  (cuda: {torch.cuda.is_available()})")
print(f"torchvision {torchvision.__version__}")
print(f"numpy       {np.__version__}")
print(f"opencv      {cv2.__version__}")

from src.models.draem.imgaug_compat import Autocontrast, ia, iaa  # noqa: E402

print(f"imgaug      {ia.__version__}")

# DRAEM 원본 data_loader.py의 augmenter 목록
augmenters = [
    iaa.GammaContrast((0.5, 2.0), per_channel=True),
    iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
    iaa.pillike.EnhanceSharpness(),
    iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
    iaa.Solarize(0.5, threshold=(32, 128)),
    iaa.Posterize(),
    iaa.Invert(),
    Autocontrast(),
    iaa.pillike.Equalize(),
    iaa.Affine(rotate=(-45, 45)),
]

img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
failed = []
for aug in augmenters:
    try:
        out = aug(image=img)
        assert out.shape == img.shape
    except Exception as e:  # noqa: BLE001
        failed.append((aug.__class__.__name__, repr(e)))

if failed:
    for name, err in failed:
        print(f"[FAIL] {name}: {err}")
    sys.exit(1)

print(f"augmenter {len(augmenters)}개 동작 확인 — DRAEM 환경 OK")
