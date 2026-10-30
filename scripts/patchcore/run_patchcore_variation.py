
# [이 스크립트의 역할] 정상편차 이미지 추론 → 결과 CSV 저장
# scripts/preprocessing/preprocess.py 가 만든 정상편차 이미지(manifest.csv)를 읽어서
# 이미지별 이상 점수와 추론시간을 run_patchcore_batch.py 와 같은 형식의 CSV로 저장한다.
#
# 목적: 불량이 아닌데 밝기/회전/위치가 조금 달라졌다는 이유만으로
#       점수가 올라가서 불량으로 잘못 판정(과검)되는지 확인한다.
#
# 실행 예:
#   python scripts/patchcore/run_patchcore_variation.py
#       --manifest data/processed/bottle/bottle_variation_v2/manifest.csv
#       --memory outputs/patchcore/bottle/coreset_raw_memory_full_1pct_d128_s0.pt
#
# 결과: outputs/predictions/patchcore/<category>_<실험이름>.csv
#   (예: bottle_bottle_variation_v2.csv)
# 다음 단계: evaluate_patchcore_threshold.py 로 편차별 과검률 계산
#
# CSV 컬럼은 run_patchcore_batch.py 와 같다.
#   label       : 편차 이미지는 모두 정상이므로 0
#   defect_type : "good"
#   variation   : brightness / rotation / translation_x / original
#   level       : 편차 강도 (밝기 %, 회전 각도, 이동 픽셀)

from pathlib import Path
import argparse
import csv
import sys
import time

import torch

# 레포 최상위 폴더를 찾아 import 경로에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 메모리 뱅크 로딩 + 추론을 묶어둔 클래스 (run_patchcore_batch.py와 같은 것)
from src.models.patchcore.model import PatchCorePredictor


def main():
    parser = argparse.ArgumentParser(
        description="PatchCore 정상편차 이미지 추론"
    )
    # --manifest: preprocess.py가 만든 manifest.csv 경로 (필수)
    parser.add_argument("--manifest", type=Path, required=True)
    # --memory: coreset 메모리 뱅크 파일 (필수, 원본 추론과 같은 파일을 써야 비교 가능)
    parser.add_argument("--memory", type=Path, required=True)
    # --output: CSV 저장 경로 (안 주면 기본 경로 사용)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # 상대경로면 레포 최상위 기준으로 해석
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    memory_path = args.memory
    if not memory_path.is_absolute():
        memory_path = ROOT / memory_path

    if not manifest_path.is_file():
        parser.error(f"manifest 파일이 없습니다: {manifest_path}")

    if not memory_path.is_file():
        parser.error("Memory Bank 파일이 없습니다.")

    # manifest 위치: data/processed/<category>/<실험이름>/manifest.csv
    # → 폴더 이름에서 실험 이름과 카테고리를 알아낸다
    experiment_name = manifest_path.parent.name          # 예) bottle_variation_v2
    category = manifest_path.parent.parent.name          # 예) bottle

    # 출력 경로: 안 주면 outputs/predictions/patchcore/<category>_<실험이름>.csv
    output_path = args.output or (
        Path("outputs/predictions/patchcore")
        / f"{category}_{experiment_name}.csv"
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    # 기존 결과를 덮어쓰지 않도록 보호
    if output_path.exists():
        parser.error("이미 CSV 파일이 존재합니다.")

    # manifest 읽기
    # 한 줄 = 편차 이미지 1장 (image_id, source_image, label, variation, level)
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as file:
        entries = list(csv.DictReader(file))

    if not entries:
        parser.error("manifest에 이미지가 없습니다.")

    # 편차 이미지는 정상만 있어야 한다 (preprocess.py는 test/good만 변형)
    if any(entry["label"] != "normal" for entry in entries):
        parser.error("manifest에 정상이 아닌 이미지가 있습니다.")

    # PatchCore 모델을 한 번만 로드 (CNN + 메모리 뱅크)
    model = PatchCorePredictor(memory_path)

    # bottle 메모리 뱅크로 다른 카테고리 편차 이미지를 테스트하는 실수를 막는다
    if model.category != category:
        parser.error("Memory Bank의 카테고리가 일치하지 않습니다.")

    print("=== PatchCore 정상편차 추론 ===")
    print("사용 장치:", model.device)
    print("실험:", experiment_name)
    print("이미지:", len(entries))

    # GPU 워밍업 (run_patchcore_batch.py와 같은 이유)
    # 첫 실행의 준비 시간이 추론시간 측정에 섞이지 않도록 한 번 미리 돌리고 결과는 버린다
    model.predict(ROOT / entries[0]["image_id"])

    # 이미지별 결과(딕셔너리)를 모을 목록
    records = []

    for index, entry in enumerate(entries):
        # manifest의 image_id는 레포 기준 상대경로
        image_path = ROOT / entry["image_id"]

        # 앞서 쌓인 GPU 작업이 끝날 때까지 기다린 뒤 시간 측정 시작
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        # 이미지 1장 추론 (파일 읽기 + 전처리 + 특징 추출 + 거리 계산 포함)
        result = model.predict(image_path)

        # 이번 이미지의 GPU 작업이 실제로 끝난 뒤 측정 종료
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        # 초 → 밀리초
        inference_ms = (time.perf_counter() - start) * 1000

        records.append({
            "image_id": entry["image_id"],
            "label": 0,                       # 편차 이미지는 모두 정상
            "defect_type": "good",
            "variation": entry["variation"],  # 편차 종류 (manifest 값 그대로)
            "level": entry["level"],          # 편차 강도 (manifest 값 그대로)
            "anomaly_score": result["image_score"],
            "inference_ms": round(inference_ms, 3),
        })

        # 20장마다 진행 상황 출력
        if (index + 1) % 20 == 0 or index + 1 == len(entries):
            print(f"[{index + 1}/{len(entries)}]")

    # 저장 폴더가 없으면 만든다
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV 저장 (run_patchcore_batch.py와 같은 컬럼 순서, 엑셀 호환 인코딩)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "label",
                "defect_type",
                "variation",
                "level",
                "anomaly_score",
                "inference_ms",
            ],
        )
        writer.writeheader()       # 첫 줄: 컬럼 이름
        writer.writerows(records)  # 나머지 줄: 이미지별 결과

    print("\n=== 추론 완료 ===")
    print("총 이미지:", len(records))
    print("CSV 저장:", output_path)


# 직접 실행했을 때만 main() 실행
if __name__ == "__main__":
    main()
