
# [이 스크립트의 역할] PatchCore 2단계 - Coreset으로 메모리 뱅크 줄이기
# 1단계에서 만든 raw_memory_*.pt (정상 patch 벡터 전체)를 읽어서
# 전체를 골고루 대표하는 일부만 골라 실제 추론용 메모리 뱅크를 만든다.
#
# 실행 예:
#   python scripts/patchcore/build_patchcore_coreset.py --input outputs/patchcore/bottle/raw_memory_full.pt --percentage 0.01
#
# 결과: 입력 파일과 같은 폴더에
#   coreset_raw_memory_full_1pct_d128_s0.pt  (bottle 1% 기준 [1638, 1024])
#   파일명 = coreset_<입력파일명>_<비율>pct_d<투영차원>_s<시드>.pt
# 다음 단계: run_patchcore_batch.py (테스트셋 전체) 또는 run_patchcore.py (이미지 몇 장)
#
# coreset 비율을 바꿔가며(예: 0.01, 0.1, 0.25) 실행하면 파일명이 달라서
# 서로 덮어쓰지 않고 여러 메모리 뱅크를 만들어 비교할 수 있다.

from pathlib import Path
import argparse
import sys
import time

import torch

# 레포 최상위 폴더를 찾아 import 경로에 추가 (build_patchcore_memory.py와 같은 방식)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.patchcore.coreset import (
    approximate_greedy_coreset_indices,
)


def main():
    parser = argparse.ArgumentParser(
        description="PatchCore Coreset 생성"
    )

    # --input: 1단계 결과 파일 경로 (필수)
    parser.add_argument("--input", type=Path, required=True)
    # --percentage: 남길 비율 (0.01 = 1%, 0.1 = 10%)
    parser.add_argument("--percentage", type=float, default=0.01)
    # --projection-dim: 거리 계산용으로 줄일 차원 (선택에만 쓰고, 저장은 원래 1024차원)
    parser.add_argument("--projection-dim", type=int, default=128)
    # --seed: 랜덤 시드 (같으면 매번 같은 coreset이 나온다)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    input_path = args.input

    # 상대경로로 줬다면 레포 최상위 기준으로 해석
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    # 경로를 정리된 절대경로로 변환 (".." 같은 표현 제거)
    input_path = input_path.resolve()

    if not input_path.is_file():
        parser.error(f"입력 파일이 없습니다: {input_path}")

    # 비율과 설정을 파일명에 기록
    # 0.01 → "1pct", 0.1 → "10pct", 0.005 → "0.5pct" → "0p5pct"
    # (:g 는 불필요한 0을 빼고 표시, 파일명에 점(.)이 들어가지 않게 p로 바꿈)
    ratio_name = f"{args.percentage * 100:g}pct".replace(".", "p")

    output_path = input_path.parent / (
        f"coreset_{input_path.stem}_{ratio_name}"
        f"_d{args.projection_dim}_s{args.seed}.pt"
    )

    # 같은 설정의 결과가 이미 있으면 덮어쓰지 않고 멈춤
    if output_path.exists():
        raise SystemExit(f"이미 결과가 존재합니다: {output_path}")

    # 1. 이전 단계에서 저장한 정상 특징 읽기
    # weights_only=True: 텐서와 기본 값만 읽음 (보안)
    data = torch.load(
        input_path,
        map_location="cpu",
        weights_only=True,
    )

    # 정상 patch 벡터 전체 [N, 1024]
    features = data["embeddings"]

    print("=== PatchCore Coreset 생성 ===")
    print("원본 특징:", tuple(features.shape))
    print("선택 비율:", args.percentage)
    print("투영 차원:", args.projection_dim)
    print("Random Seed:", args.seed)

    # 2. 대표 특징 인덱스 선택
    # coreset 선택에 걸리는 시간을 재기 위해 시작 시각 기록
    # (perf_counter: 짧은 시간 측정에 정확한 시계)
    start_time = time.perf_counter()

    # 대표로 뽑힌 벡터들의 "번호" 목록을 받는다 (예: 1638개 번호)
    indices = approximate_greedy_coreset_indices(
        features=features,
        percentage=args.percentage,
        projection_dim=args.projection_dim,
        seed=args.seed,
    )

    elapsed = time.perf_counter() - start_time

    # 3. 원본 1024차원 특징을 선택
    # 고를 때는 128차원으로 줄인 값을 썼지만,
    # 실제 메모리 뱅크에는 정보 손실이 없는 원래 1024차원 벡터를 넣는다
    # index_select(0, indices): 0번 축(행)에서 indices 번호의 행만 뽑기
    memory_bank = features.index_select(0, indices)

    # 선택 인덱스 중복 여부 확인
    # 같은 벡터가 두 번 뽑혔다면 coreset 코드에 문제가 있는 것이므로 멈춘다
    if torch.unique(indices).numel() != len(indices):
        raise RuntimeError("Coreset 인덱스가 중복되었습니다.")

    # 4. 결과 저장
    # 메모리 뱅크와 함께 만든 조건도 같이 저장 (나중에 확인·재현용)
    torch.save({
        "embeddings": memory_bank,          # 실제 추론에 쓰는 정상 대표 벡터 [K, 1024]
        "selected_indices": indices,        # 원본에서 몇 번째 벡터들을 골랐는지
        "percentage": args.percentage,      # 선택 비율
        "projection_dim": args.projection_dim,
        "seed": args.seed,
        "source": input_path.name,          # 어떤 원본 파일에서 만들었는지
        "method": "approx_greedy_gaussian_projection",
        "category": data["category"],       # 원본 정보 그대로 이어받기
        "backbone": data["backbone"],
        "image_count": data["image_count"],
    }, output_path)

    print("\n=== Coreset 생성 완료 ===")
    print("원본 개수:", len(features))
    print("선택 개수:", len(memory_bank))
    print("최종 Shape:", tuple(memory_bank.shape))
    print("소요 시간:", f"{elapsed:.2f}초")
    print("저장 위치:", output_path)


# 직접 실행했을 때만 main() 실행
if __name__ == "__main__":
    main()
