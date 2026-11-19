


from pathlib import Path
import argparse
import json
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.patchcore.coreset import (
    approximate_greedy_coreset_indices,
)
from src.models.patchcore.artifacts import file_sha256
from src.models.patchcore.transforms import input_options_of  # 저장된 입력 방식 읽기


def main():
    parser = argparse.ArgumentParser(
        description="PatchCore Coreset 생성"
    )


    parser.add_argument("--input", type=Path, required=True)

    parser.add_argument("--percentage", type=float, default=0.01)

    parser.add_argument("--projection-dim", type=int, default=128)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    input_path = args.input


    if not input_path.is_absolute():
        input_path = ROOT / input_path


    input_path = input_path.resolve()

    if not input_path.is_file():
        parser.error(f"입력 파일이 없습니다: {input_path}")

    # 비율과 설정을 파일명에 기록


    ratio_name = f"{args.percentage * 100:g}pct".replace(".", "p")

    output_path = input_path.parent / (
        f"coreset_{input_path.stem}_{ratio_name}"
        f"_d{args.projection_dim}_s{args.seed}.pt"
    )


    if output_path.exists():
        raise SystemExit(f"이미 결과가 존재합니다: {output_path}")

    # 1. 이전 단계에서 저장한 정상 특징 읽기

    data = torch.load(
        input_path,
        map_location="cpu",
        weights_only=True,
    )


    features = data["embeddings"]
    # 원본 메모리를 만들 때 쓴 입력 방식 (옛 파일은 값이 없어서 기본값 = 기존 방식)
    input_mode, input_scale = input_options_of(data)

    print("=== PatchCore Coreset 생성 ===")
    print("원본 특징:", tuple(features.shape))
    print("선택 비율:", args.percentage)
    print("투영 차원:", args.projection_dim)
    print("Random Seed:", args.seed)

    # 2. 대표 특징 인덱스 선택


    start_time = time.perf_counter()


    indices = approximate_greedy_coreset_indices(
        features=features,
        percentage=args.percentage,
        projection_dim=args.projection_dim,
        seed=args.seed,
    )

    elapsed = time.perf_counter() - start_time


    memory_bank = features.index_select(0, indices)

    # 선택 인덱스 중복 여부 확인

    if torch.unique(indices).numel() != len(indices):
        raise RuntimeError("Coreset 인덱스가 중복되었습니다.")

    # 4. 결과 저장

    torch.save({
        "embeddings": memory_bank,
        "selected_indices": indices,
        "percentage": args.percentage,
        "projection_dim": args.projection_dim,
        "seed": args.seed,
        "source": input_path.name,
        "method": "approx_greedy_gaussian_projection",
        "category": data["category"],
        "backbone": data["backbone"],
        "image_count": data["image_count"],
        "raw_memory_sha256": file_sha256(input_path),
        # 학습 입력 방식을 그대로 넘겨서 추론기(PatchCorePredictor)가 같은 변환을 쓰게 한다
        # (옛 원본 메모리에는 값이 없으므로 기본값 = 기존 방식)
        "input_mode": input_mode,
        "input_scale": input_scale,
    }, output_path)
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(json.dumps({
        "category": data["category"], "percentage": args.percentage,
        "projection_dim": args.projection_dim, "seed": args.seed,
        "raw_memory_sha256": file_sha256(input_path),
        "input_mode": input_mode, "input_scale": input_scale,  # 기록용
    }, indent=2), encoding="utf-8")

    print("\n=== Coreset 생성 완료 ===")
    print("원본 개수:", len(features))
    print("선택 개수:", len(memory_bank))
    print("최종 Shape:", tuple(memory_bank.shape))
    print("소요 시간:", f"{elapsed:.2f}초")
    print("저장 위치:", output_path)


if __name__ == "__main__":
    main()
