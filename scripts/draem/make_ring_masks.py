"""라벨 없이 만드는 공정한 평가 마스크 — 바깥 원 기준 고정 폭 링 − 스프링 시작점 구간.

edge_masks_ex는 테스트 이미지 각자의 라벨(연마면 경계)로 만들어서, 불량 모양 정보가 마스크에 섞여 들어갑니다.
이 스크립트는 라벨을 쓰지 않습니다. 모든 이미지에 똑같이
    - 바깥 원 반지름 R의 0.80~1.03배 링 (띠 펴기와 같은 범위)
    - 에서 스프링 시작점 −5°~100° 구간 (띠 펴기와 같은 방법으로 찾음) 을 뺀 영역
을 마스크로 저장합니다. 현장 이미지에도 그대로 만들 수 있습니다.

    python scripts/draem/make_ring_masks.py --category camera_2

결과: data/processed/company_plc/<category>/ring_masks_ex/<이미지 이름>.png
     outputs/report/ring_preview/<category>.png (확인용, 몇 장)
"""

from pathlib import Path
import argparse
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.draem.preprocess import EXCLUDE, POLAR_R, exclude_center  # noqa: E402


def ring_mask(image_bgr: np.ndarray) -> np.ndarray:
    S = image_bgr.shape[0]
    c, R = S / 2, S / 2 / 1.1
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) + 0.5
    r = np.hypot(xx - c, yy - c) / R
    ring = (r >= POLAR_R[0]) & (r <= POLAR_R[1])
    center = exclude_center(image_bgr)
    half = (EXCLUDE[1] - EXCLUDE[0]) / 2
    diff = (np.rad2deg(np.arctan2(yy - c, xx - c)) - center + 180) % 360 - 180
    return ((ring & (np.abs(diff) > half)) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="라벨 없는 고정 폭 링 마스크")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/company_plc"))
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    root = (args.data_root if args.data_root.is_absolute() else ROOT / args.data_root) / args.category
    out = root / "ring_masks_ex"
    out.mkdir(exist_ok=True)
    paths = sorted(root.glob("train/good/*.png")) + sorted(root.glob("test/*/*.png"))
    if not paths:
        raise SystemExit(f"이미지가 없습니다: {root}")
    previews = []
    for i, p in enumerate(paths):
        image = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        m = ring_mask(image)
        cv2.imencode(".png", m)[1].tofile(str(out / f"{p.stem}.png"))
        if i % max(1, len(paths) // 8) == 0 and len(previews) < 8:
            small = cv2.resize(image, (256, 256))
            ov = small.copy()
            ov[cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST) > 0] = (0, 200, 0)
            previews.append(cv2.addWeighted(small, 0.6, ov, 0.4, 0))
    prev_path = ROOT / "outputs/report/ring_preview" / f"{args.category}.png"
    prev_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.hstack(previews))[1].tofile(str(prev_path))
    print(f"{len(paths)}장 → {out}")
    print(f"확인용 그림: {prev_path}")


if __name__ == "__main__":
    main()
