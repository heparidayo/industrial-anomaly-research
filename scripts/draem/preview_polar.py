"""띠 펴기 미리보기 — 학습 전에 스프링 시작점 · 제외 구간 · 펼친 띠가 맞는지 눈으로 확인.

    python scripts/draem/preview_polar.py --category camera_1

이미지마다 한 줄:  원본(초록 원 = 펼 반지름 범위, 빨간 선 = 잘라 낼 제외 구간)  |  펼친 띠
결과: outputs/report/polar_preview/<category>.png
빨간 선 사이에 어두운 부채꼴이 들어와 있지 않은 이미지가 있으면 알려 주세요.
"""

from pathlib import Path
import argparse
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.draem.preprocess import EXCLUDE, POLAR_R, preprocess_polar  # noqa: E402


def row(path: Path, clahe: bool) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    strip, center = preprocess_polar(raw, clahe)
    S, c = 256, 128
    R = S / 2 / 1.1
    vis = cv2.resize(raw, (S, S))
    for rr in POLAR_R:
        cv2.circle(vis, (c, c), int(R * rr), (0, 255, 0), 1)
    half = (EXCLUDE[1] - EXCLUDE[0]) / 2
    for d in (center - half, center + half):
        t = np.deg2rad(d)
        cv2.line(vis, (c, c), (int(c + R * 1.05 * np.cos(t)), int(c + R * 1.05 * np.sin(t))), (0, 0, 255), 2)
    strip = cv2.resize(strip, (1024, 128))
    right = np.vstack([strip, np.full((128, 1024, 3), 255, np.uint8)])
    cv2.putText(right, f"{path.parent.name}  {path.stem[-26:]}", (8, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return np.hstack([vis, np.full((S, 8, 3), 255, np.uint8), right])


def main():
    parser = argparse.ArgumentParser(description="띠 펴기 미리보기")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/company_plc"))
    parser.add_argument("--category", default="camera_1")
    parser.add_argument("--clahe", action="store_true")
    args = parser.parse_args()

    root = (args.data_root if args.data_root.is_absolute() else ROOT / args.data_root) / args.category
    paths = sorted(root.glob("train/good/*.png")) + sorted(root.glob("test/*/*.png"))
    if not paths:
        raise SystemExit(f"이미지가 없습니다: {root}")
    rows = [row(p, args.clahe) for p in paths]
    out = ROOT / "outputs/report/polar_preview" / f"{args.category}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    space = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    cv2.imencode(".png", np.vstack([x for r in rows for x in (r, space)]))[1].tofile(str(out))
    print(f"{len(paths)}장 → {out}")


if __name__ == "__main__":
    main()
