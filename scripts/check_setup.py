from pathlib import Path
import sys

# 1. 프로젝트 root 경로
ROOT = Path(__file__).resolve().parents[1]

# 2. 프로젝트 필수 폴더
DIRECTORIES = [
    "configs",
    "data",
    "outputs",
    "experiments",
    "scripts",
    "src/preprocessing",
    "src/models",
    "src/evaluation",
]


def main():
    print("프로젝트 경로:", ROOT)
    print("Python 버전:", sys.version.split()[0])

    print("\n[필수 폴더 확인]")

    for directory in DIRECTORIES:
        path = ROOT / directory
        status = "정상" if path.is_dir() else "누락"
        print(f"[{status}] {directory}")

    missing = [
        name for name in DIRECTORIES
        if not (ROOT / name).is_dir()
    ]

    if missing:
        raise SystemExit("\n필수 폴더가 누락되었습니다. 폴더 구조를 확인해주세요.")

    print("\n모든 필수 폴더가 확인되었습니다!")


if __name__ == "__main__":
    main()