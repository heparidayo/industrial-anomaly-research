
# [이 스크립트의 역할] PatchCore threshold(불량 판정 기준 점수) 정하기
# 테스트 데이터를 보지 않고, 학습 정상 이미지만으로 threshold를 정한다.
#
# 방법: 학습 정상 이미지를 K개 그룹으로 나눠 돌아가며 한 그룹씩 빼고
#   1) 나머지 이미지로 메모리 뱅크(coreset)를 만들고
#   2) 뺀 그룹의 이미지를 채점한다 → "처음 보는 정상 이미지의 점수"
# 모든 그룹을 한 번씩 빼면 학습 정상 이미지 전부에 대해 점수가 하나씩 생긴다.
# threshold = 그 점수들의 최댓값 (이보다 높으면 불량으로 판정)
#
# 학습 이미지를 그대로 채점하면 안 되는 이유:
#   그 이미지는 메모리 뱅크 안에 들어 있어서 자기 자신과의 거리가 0에 가까워진다.
#
# 테스트 이미지로 threshold를 정하면 안 되는 이유:
#   정답을 보고 기준을 정하는 셈이라 결과가 실제보다 좋게 나온다.
#
# 실행 예:
#   python scripts/patchcore/compute_patchcore_threshold.py --category bottle
#   python scripts/patchcore/compute_patchcore_threshold.py --category company_cam1
#
# 입력: outputs/patchcore/<category>/raw_memory_full.pt (build_patchcore_memory.py 결과)
#       → 특징을 다시 뽑지 않고 이미 저장된 정상 patch 벡터를 재사용한다 (빠름)
# 결과: outputs/patchcore/<category>/threshold_<비율>pct_d<투영차원>_s<시드>_k<그룹수>.json
#       (예: threshold_1pct_d128_s0_k10.json)
# 다음 단계: evaluate_patchcore_threshold.py 로 과검률/미검률/F1 계산

from pathlib import Path
import argparse
import json
import sys
import time

import torch

# 레포 최상위 폴더를 찾아 import 경로에 추가 (다른 PatchCore 스크립트와 같은 방식)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# coreset 선택 함수 (build_patchcore_coreset.py와 같은 함수)
from src.models.patchcore.coreset import approximate_greedy_coreset_indices
# 최근접 거리 계산 함수 (실제 추론과 같은 함수)
from src.models.patchcore.scoring import compute_patch_scores


def main():
    parser = argparse.ArgumentParser(
        description="PatchCore threshold 계산 (학습 정상 이미지 held-out)"
    )

    # --category: threshold를 정할 카테고리 (필수)
    parser.add_argument("--category", required=True)
    # --input: 정상 특징 파일 (안 주면 outputs/patchcore/<category>/raw_memory_full.pt)
    parser.add_argument("--input", type=Path, default=None)
    # --folds: 그룹 수 (이미지 수 이상이면 한 장씩 빼는 leave-one-out)
    #   회사 데이터(10장) → 10그룹 = 한 장씩 빼기
    #   bottle(209장)     → 10그룹 = 약 21장씩 빼기 (209번 반복하면 너무 오래 걸려서)
    parser.add_argument("--folds", type=int, default=10)
    # 아래 3개는 실제 추론용 coreset과 같은 값을 써야 한다
    # (다른 설정으로 threshold를 정하면 실제 추론 점수와 기준이 어긋난다)
    parser.add_argument("--percentage", type=float, default=0.01)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # 입력 경로를 안 줬으면 기본 경로 사용 (or: 왼쪽이 None이면 오른쪽 값)
    input_path = args.input or (
        Path("outputs/patchcore") / args.category / "raw_memory_full.pt"
    )
    # 상대경로면 레포 최상위 기준으로 해석
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.is_file():
        parser.error(f"정상 특징 파일이 없습니다: {input_path}")

    # 1단계 결과 파일 읽기 (weights_only=True: 텐서와 기본 값만 읽음, 보안)
    data = torch.load(input_path, map_location="cpu", weights_only=True)

    # 다른 카테고리의 특징 파일을 실수로 넣는 것을 막는다
    if data["category"] != args.category:
        parser.error("정상 특징 파일의 카테고리가 일치하지 않습니다.")

    # raw_memory는 이미지 순서대로 784개씩 patch 벡터가 이어 붙어 있다
    # → [이미지 수, 784, 1024] 로 다시 나눈다
    # (이미지 단위로 "빼고 / 남기기"를 하려면 이미지별로 묶여 있어야 한다)
    features = data["embeddings"]                    # [이미지 수 * 784, 1024]
    image_names = data["image_names"]                # 이미지 파일명 목록 (저장 순서와 같음)
    patches_per_image = data["patches_per_image"]    # 784
    image_count = len(image_names)
    # view: 데이터 복사 없이 모양만 바꿔서 보기 (-1: 나머지 차원은 자동 계산 = 1024)
    per_image = features.view(image_count, patches_per_image, -1)

    # 그룹 수는 이미지 수보다 많을 수 없다
    folds = min(args.folds, image_count)
    # 최소 2그룹은 있어야 "빼고 남기기"가 된다
    if folds < 2:
        parser.error("그룹 수는 2 이상이어야 합니다.")

    # 출력 파일명에 설정을 기록 (build_patchcore_coreset.py와 같은 방식)
    # 0.01 → "1pct", 0.005 → "0p5pct"
    ratio_name = f"{args.percentage * 100:g}pct".replace(".", "p")
    output_path = input_path.parent / (
        f"threshold_{ratio_name}_d{args.projection_dim}"
        f"_s{args.seed}_k{folds}.json"
    )

    # 같은 설정의 결과가 이미 있으면 덮어쓰지 않고 멈춤
    if output_path.exists():
        raise SystemExit(f"이미 결과가 존재합니다: {output_path}")

    # GPU가 있으면 GPU, 없으면 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== PatchCore threshold 계산 ===")
    print("카테고리:", args.category)
    print("학습 정상 이미지:", image_count)
    print("그룹 수:", folds, "(leave-one-out)" if folds == image_count else "")

    # 이미지 i는 (i % folds)번 그룹 → 그룹마다 이미지가 골고루 섞인다
    # 예) 이미지 25장, 10그룹 → 0번 그룹 = 0, 10, 20번 이미지 / 1번 그룹 = 1, 11, 21번 ...
    fold_of = torch.arange(image_count) % folds
    # 이미지별 held-out 점수를 채울 자리 (처음엔 비어 있음)
    scores = [None] * image_count
    start_time = time.perf_counter()

    for fold in range(folds):
        # 이번에 뺄 이미지 번호들 / 남길 이미지 번호들
        # nonzero(): 조건이 True인 위치 번호, flatten(): 1차원 목록으로
        held_out = (fold_of == fold).nonzero().flatten()
        kept = (fold_of != fold).nonzero().flatten()

        # 1. 뺀 그룹을 제외한 정상 이미지로 coreset 메모리 뱅크 생성
        # 남긴 이미지들의 patch 벡터를 다시 한 줄로 이어 붙인다 → [남긴 이미지 수 * 784, 1024]
        kept_features = per_image[kept].reshape(-1, per_image.shape[-1])
        # 실제 추론용 메모리 뱅크와 같은 방식(같은 비율, 투영 차원, 시드)으로 대표 벡터 선택
        indices = approximate_greedy_coreset_indices(
            features=kept_features,
            percentage=args.percentage,
            projection_dim=args.projection_dim,
            seed=args.seed,
            log_every=10**9,  # 진행 로그 생략
        )
        # 선택된 번호의 벡터만 뽑아서 GPU로 (indices가 GPU에 있을 수 있어 CPU로 맞춤)
        memory_bank = kept_features.index_select(0, indices.cpu()).to(device)

        # 2. 뺀 그룹의 이미지 채점 (이미지 점수 = patch 점수 최댓값)
        # 실제 추론(model.py)과 똑같이 계산한다
        for i in held_out.tolist():
            # per_image[i]: [784, 1024] → unsqueeze(0): [1, 784, 1024] (배치 차원 추가)
            patch_scores = compute_patch_scores(
                per_image[i].unsqueeze(0).to(device),
                memory_bank,
            )
            # 784개 patch 점수 중 가장 큰 값 = 이 이미지의 이상 점수
            scores[i] = patch_scores.max().item()

        print(f"[그룹 {fold + 1}/{folds}] 제외 {len(held_out)}장, "
              f"메모리 뱅크 {tuple(memory_bank.shape)}")

    elapsed = time.perf_counter() - start_time
    # 통계 계산을 위해 텐서로 변환
    values = torch.tensor(scores)

    # 결과 정리
    result = {
        "category": args.category,
        "method": "held_out_normal_max",     # 규칙 이름: held-out 정상 점수의 최댓값
        "threshold": values.max().item(),    # 실제로 쓰는 threshold
        "folds": folds,
        "percentage": args.percentage,
        "projection_dim": args.projection_dim,
        "seed": args.seed,
        "source": input_path.name,           # 어떤 특징 파일에서 계산했는지
        # held-out 정상 점수 분포 (threshold가 적절한지 확인용)
        "normal_score_stats": {
            "count": image_count,
            "mean": values.mean().item(),
            # 표준편차 (이미지가 1장이면 계산 불가 → 0)
            "std": values.std().item() if image_count > 1 else 0.0,
            "min": values.min().item(),
            # 상위 5% 지점 (최댓값이 혼자 튀는지 비교할 때 사용)
            "p95": values.quantile(0.95).item(),
            "max": values.max().item(),
        },
        # 이미지별 점수 (어떤 정상 이미지가 threshold를 끌어올렸는지 찾을 때 사용)
        # zip: 파일명과 점수를 짝지어 {파일명: 점수} 딕셔너리로
        "held_out_scores": dict(zip(image_names, scores)),
    }

    # JSON 저장 (ensure_ascii=False: 한글 그대로, indent=2: 사람이 읽기 좋게 줄바꿈)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    stats = result["normal_score_stats"]
    print("\n=== threshold 계산 완료 ===")
    print(f"정상 held-out 점수: 평균 {stats['mean']:.4f}, "
          f"범위 [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"threshold: {result['threshold']:.4f}")
    print(f"소요 시간: {elapsed:.1f}초")
    print("저장 위치:", output_path)


# 직접 실행했을 때만 main() 실행
if __name__ == "__main__":
    main()
