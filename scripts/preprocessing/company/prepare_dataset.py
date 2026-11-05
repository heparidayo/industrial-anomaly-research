"""산업체 원본을 촬영번호 단위의 공통 데이터 입력으로 준비한다."""
import argparse
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.preprocessing.company.prepare import prepare_company_dataset


def main():
    parser = argparse.ArgumentParser(description="산업체 데이터 공통 분할·ROI 준비")
    parser.add_argument("--config", type=Path, default=Path("configs/datasets/company_v1.yaml"))
    parser.add_argument("--src", type=Path)
    parser.add_argument("--dst", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-validation", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--roi", action="store_true")
    parser.add_argument("--roi-half", type=int)
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    try:
        config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
        settings = config["split"]
        categories = prepare_company_dataset(
            resolve(args.src or Path(config["dataset"]["root"])),
            resolve(args.dst or Path(config["output"]["root"])),
            prefix=args.prefix or config["output"]["prefix"],
            seed=args.seed if args.seed is not None else settings["seed"],
            n_train=args.n_train if args.n_train is not None else settings["n_train_per_camera"],
            n_validation=args.n_validation if args.n_validation is not None else settings["n_validation_per_camera"],
            roi=args.roi,
            roi_half=args.roi_half if args.roi_half is not None else config["roi"]["half_size"],
            reference_image_size=config["roi"]["reference_image_size"],
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    for category, path in categories.items():
        print(f"{category}: {path}")
    print("제안 분할입니다. split.csv의 train/validation/test/unused를 담당자와 확인하세요.")


if __name__ == "__main__":
    main()
