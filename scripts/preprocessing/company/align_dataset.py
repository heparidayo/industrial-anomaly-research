"""준비된 회사 카테고리를 '스프링 시작점 정렬 + 검사 제외 영역 마스킹' 버전으로 만든다.

실행 예 (먼저 prepare_dataset.py로 company_cam1/2/12를 만들어 둔다):
  python scripts/preprocessing/company/align_dataset.py --config configs/datasets/company_align_v1.yaml

결과: data/processed/company/company_cam1_aligned 등 (분할은 입력과 동일)
이후 PatchCore 실행은 기존과 같고 카테고리 이름만 바꾼다:
  python scripts/patchcore/run_patchcore_experiment.py --name company_aligned_validation_v1
      --categories company_cam1_aligned company_cam2_aligned company_cam12_aligned
      --data-root data/processed/company --split validation --pcts 0.01
"""
import argparse
from pathlib import Path
import sys

import yaml

# 레포 최상위 폴더를 import 경로에 추가 (scripts/preprocessing/company → 세 단계 위)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.preprocessing.company.prepare_aligned import prepare_aligned_dataset


def main():
    parser = argparse.ArgumentParser(description="회사 데이터 시작점 정렬·제외 영역 마스킹")
    parser.add_argument("--config", type=Path, default=Path("configs/datasets/company_align_v1.yaml"))
    # 설정 파일 값을 명령줄에서 덮어쓸 때 사용
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--suffix")
    args = parser.parse_args()

    # 상대 경로("configs/...")는 레포 최상위 폴더 기준으로 바꾼다 (어디서 실행해도 같은 파일)
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    try:
        config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
        settings = config["alignment"]
        # YAML 값을 정렬 함수가 받는 형태(숫자, 튜플)로 바꾼다
        options = {
            "target_deg": float(settings["target_deg"]),
            "exclusion_deg": tuple(float(v) for v in settings["exclusion_deg"]),
            "outer_margin": float(settings["outer_margin"]),
            "crop_scale": float(settings["crop_scale"]),
            "fill_value": int(settings["fill_value"]),
        }
        created = prepare_aligned_dataset(
            resolve(Path(config["source"]["root"])),
            args.categories or config["source"]["categories"],
            suffix=args.suffix or config["output"]["suffix"],
            options=options,
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))

    for category, path in created.items():
        print(f"{category}: {path}")
    print("각 카테고리의 alignment.csv에서 low_confidence(시작점 검출이 애매한 이미지)를 확인하세요.")


if __name__ == "__main__":
    main()
