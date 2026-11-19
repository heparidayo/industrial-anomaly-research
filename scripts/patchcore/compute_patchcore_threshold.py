"""학습 정상 특징만으로 PatchCore의 실험용 임계값을 보정한다.

규칙: 기본은 95% 분위수 (--rule max 로 예전 최댓값 규칙 재현). 파일 이름에 규칙 꼬리(_q95)가 붙어
      예전 최댓값 결과(threshold_..._k10.json)와 섞이지 않는다.
--from-scores: 같은 메모리·설정으로 이미 계산한 임계값 파일이 있으면, 그 안의 번갈아 빼기 점수를
               다시 쓰고 규칙만 바꾼다 (fold마다 coreset을 다시 만드는 비싼 계산 생략).
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.patchcore.threshold import (
    DEFAULT_THRESHOLD_QUANTILE, DEFAULT_THRESHOLD_RULE, THRESHOLD_RULES,
    effective_folds, held_out_scores, threshold_from_scores, threshold_tag,
)
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
    parser.add_argument("--rule", choices=THRESHOLD_RULES, default=DEFAULT_THRESHOLD_RULE)       # 임계값 규칙
    parser.add_argument("--quantile", type=float, default=DEFAULT_THRESHOLD_QUANTILE)           # 분위수 (rule=quantile)
    parser.add_argument("--from-scores", type=Path)    # 번갈아 빼기 점수를 재사용할 기존 임계값 파일
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
        tag = threshold_tag(args.rule, args.quantile)                                # 규칙 꼬리 (max면 "")
        ratio = f"{args.percentage * 100:g}pct".replace(".", "p")
        output = resolve(args.output) if args.output else source.parent / (
            f"threshold_{ratio}_d{args.projection_dim}_s{args.seed}_k{folds}{tag}.json")
        if output.exists():
            raise FileExistsError(f"임계값 결과가 이미 있습니다: {output}")
        raw_sha = file_sha256(source)                                                # 이 메모리의 지문
        if args.from_scores:                                                         # 기존 점수 재사용
            previous = json.loads(resolve(args.from_scores).read_text(encoding="utf-8"))
            same = {"category": args.category, "percentage": args.percentage, "folds": folds,
                    "projection_dim": args.projection_dim, "seed": args.seed, "raw_memory_sha256": raw_sha}
            if any(previous.get(key) != value for key, value in same.items()):     # 다른 메모리·설정이면 거부
                raise ValueError(f"재사용할 점수의 메모리·설정이 다릅니다: {args.from_scores}")
            if list(previous["held_out_scores"]) != list(names):                     # 같은 이미지·같은 순서인지
                raise ValueError("재사용할 점수의 이미지 목록이 학습 특징과 다릅니다.")
            scores = [previous["held_out_scores"][name] for name in names]
        else:                                                                        # 처음부터 계산 (fold마다 coreset)
            features = data["embeddings"].view(len(names), data["patches_per_image"], -1)
            scores = held_out_scores(
                features, folds, args.percentage, args.projection_dim, args.seed,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"), groups=groups)
        values = torch.tensor(scores)
        result = {
            "category": args.category, "model": "patchcore_prototype",
            # 예전 max 결과와 같은 method 이름을 유지하고, 분위수는 새 이름
            "method": "held_out_normal_max" if args.rule == "max" else "held_out_normal_quantile",
            "rule": args.rule, "quantile": args.quantile if args.rule == "quantile" else None,
            "calibration_split": "train_normal_group_oof",
            "threshold": threshold_from_scores(scores, args.rule, args.quantile), "folds": folds,
            "percentage": args.percentage, "projection_dim": args.projection_dim,
            "seed": args.seed, "source": source.name, "raw_memory_sha256": raw_sha,
            "scores_reused_from": str(args.from_scores) if args.from_scores else None,  # 재사용 여부 기록
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
