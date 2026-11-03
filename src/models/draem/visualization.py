"""DRAEM 결과 시각화: 원본 | 복원 이미지 | 결함 확률 맵 | 오버레이.

공용 src/evaluation/visualization.py는 PatchCore의 ImageNet 정규화 텐서를 기준으로
만들어졌을 수 있어서, DRAEM(0~1, BGR 입력)은 별도 함수로 둡니다.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def save_draem_visualization(
    image: np.ndarray,
    reconstruction: np.ndarray,
    anomaly_map: np.ndarray,
    output_path: Path,
    image_score: float | None = None,
    vmax: float = 1.0,
) -> None:
    """image / reconstruction: (H, W, 3) BGR 0~1, anomaly_map: (H, W) 0~1."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = image[..., ::-1]
    rec_rgb = reconstruction[..., ::-1]
    amap = np.asarray(anomaly_map)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    panels = [("Input", rgb), ("Reconstruction", rec_rgb)]
    for ax, (title, img) in zip(axes[:2], panels):
        ax.imshow(img)
        ax.set_title(title)

    im = axes[2].imshow(amap, cmap="jet", vmin=0, vmax=vmax)
    axes[2].set_title("Anomaly map")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(rgb)
    axes[3].imshow(amap, cmap="jet", vmin=0, vmax=vmax, alpha=0.45)
    axes[3].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    if image_score is not None:
        fig.suptitle(f"Image anomaly score: {image_score:.4f}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
