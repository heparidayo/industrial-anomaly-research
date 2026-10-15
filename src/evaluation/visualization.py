
from pathlib import Path

import cv2
import numpy as np


def save_anomaly_visualization(
    image_tensor,
    anomaly_map,
    output_path: Path,
    vmax=25.0,
):
    """
    정규화된 RGB 이미지와 Anomaly Map을 시각화한다.

    저장 순서:
    [입력 이미지 | Heatmap | Overlay]
    """

    if vmax <= 0:
        raise ValueError("vmax는 0보다 커야 합니다.")

    # 1. ImageNet 정규화 해제
    image = image_tensor.detach().cpu().float()
    image = image.permute(1, 2, 0).numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    # RGB -> BGR (OpenCV 저장 형식)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 2. Anomaly Map을 색상 이미지로 변환
    scores = anomaly_map.detach().cpu().numpy()

    if scores.shape != image.shape[:2]:
        raise ValueError("이미지와 Anomaly Map 크기가 다릅니다.")

    normalized = np.clip(scores / vmax, 0, 1)
    heatmap_gray = (normalized * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heatmap_gray,
        cv2.COLORMAP_JET,
    )

    # 3. 원본과 Heatmap 합성
    overlay = cv2.addWeighted(
        image, 0.55,
        heatmap, 0.45,
        0,
    )

    # 4. 3개 이미지를 가로로 연결
    result = np.concatenate(
        [image, heatmap, overlay],
        axis=1,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError("시각화 이미지 저장 실패")

    return output_path
