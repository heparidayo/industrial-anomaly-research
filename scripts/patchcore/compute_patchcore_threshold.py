"""학습 정상 특징만으로 PatchCore의 실험용 임계값을 보정한다."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.patchcore.threshold import effective_folds, held_out_scores
from src.models.patchcore.artifacts import file_sha256, image_group_id


def main():
    parser = argparse.ArgumentParser(description="PatchCore 학습 촬영그룹 held-out 임계값")
    parser.add_argument("--category", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--percentage", type=float, default=0.01)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    try:
        source = resolve(args.input or Path("outputs/patchcore") / args.category / "raw_memory_full.pt")
        data = torch.load(source, map_location="cpu", weights_only=True)
        if data["category"] != args.category or data.get("stage") != "raw_candidates":
            raise ValueError("정상 학습 특징 파일의 카테고리·단계를 확인하세요.")
        names = data["image_names"]
        groups = data.get("image_group_ids", [image_group_id(name) for name in names])
        folds = effective_folds(groups, args.folds)
        ratio = f"{args.percentage * 100:g}pct".replace(".", "p")
        output = resolve(args.output) if args.output else source.parent / (
            f"threshold_{ratio}_d{args.projection_dim}_s{args.seed}_k{folds}.json")
        if output.exists():
            raise FileExistsError(f"임계값 결과가 이미 있습니다: {output}")
        features = data["embeddings"].view(len(names), data["patches_per_image"], -1)
        scores = held_out_scores(
            features, folds, args.percentage, args.projection_dim, args.seed,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"), groups=groups)
        values = torch.tensor(scores)
        result = {
            "category": args.category, "model": "patchcore_prototype",
            "method": "held_out_normal_max", "calibration_split": "train_normal_group_oof",
            "threshold": values.max().item(), "folds": folds,
            "percentage": args.percentage, "projection_dim": args.projection_dim,
            "seed": args.seed, "source": source.name, "raw_memory_sha256": file_sha256(source),
            "normal_score_stats": {
                "count": len(names), "mean": values.mean().item(),
                "std": values.std().item(), "min": values.min().item(),
                "p95": values.quantile(0.95).item(), "max": values.max().item(),
            },
            "held_out_scores": dict(zip(names, scores)),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    print("임계값:", result["threshold"], "결과:", output)


if __name__ == "__main__":
    main()
