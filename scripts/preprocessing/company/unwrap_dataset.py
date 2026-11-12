"""준비된 회사 카테고리를 '면취 고리 펼치기(극좌표)' 버전으로 만든다.

실행 예 (먼저 prepare_dataset.py로 company_cam1/2/12를 만들어 둔다):
  python scripts/preprocessing/company/unwrap_dataset.py --config configs/datasets/company_polar_v1.yaml

결과: data/processed/company/company_cam1_polar 등 (분할은 입력과 동일)
이후 PatchCore 실행은 기존과 같고 카테고리 이름만 바꾼다:
  python scripts/patchcore/run_patchcore_experiment.py --name company_polar_validation_v1
      --categories company_cam1_polar company_cam2_polar company_cam12_polar
      --data-root data/processed/company --split validation --pcts 0.01
"""
import argparse
from pathlib import Path
import sys

import yaml

# 레포 최상위 폴더를 import 경로에 추가 (scripts/preprocessing/company → 세 단계 위)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.preprocessing.company.prepare_polar import prepare_polar_dataset


def main():
    parser = argparse.ArgumentParser(description="회사 데이터 면취 고리 펼치기")
    parser.add_argument("--config", type=Path, default=Path("configs/datasets/company_polar_v1.yaml"))
    parser.add_argument("--categories", nargs="+")      # 설정 파일의 카테고리 대신 쓸 목록
    parser.add_argument("--suffix")                     # 설정 파일의 출력 이름 꼬리 대신 쓸 값
    args = parser.parse_args()

    # 상대 경로("configs/...")는 레포 최상위 폴더 기준으로 바꾼다 (어디서 실행해도 같은 파일)
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    try:
        config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))  # 설정 읽기
        settings = config["polar"]                                                 # 펼치기 설정 부분
        options = {                                                                # 함수가 받는 형태로 변환
            "inner": float(settings["inner"]),
            "outer": float(settings["outer"]),
            "angle_range": tuple(float(v) for v in settings["angle_range"]),
            "radial_px": int(settings["radial_px"]),
            "rows": int(settings["rows"]),
        }
        if settings.get("columns") is not None:                                   # 가로 칸 수를 직접 정한 경우만
            options["columns"] = int(settings["columns"])                         # (없으면 정사각형 자동 계산)
        created = prepare_polar_dataset(
            resolve(Path(config["source"]["root"])),
            args.categories or config["source"]["categories"],
            suffix=args.suffix or config["output"]["suffix"],
            options=options,
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))                                                   # 오류를 읽기 쉽게 출력

    for category, path in created.items():
        print(f"{category}: {path}")                                               # 만든 카테고리 목록
    print("각 카테고리의 polar.csv에서 low_confidence(시작점 검출이 애매한 이미지)를 확인하세요.")


if __name__ == "__main__":
    main()
