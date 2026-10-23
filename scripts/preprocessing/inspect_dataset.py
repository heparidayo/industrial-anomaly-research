# MVTecAD 데이터셋이 정상적으로 있는지 검사합니다


from pathlib import Path

# 프로젝트 루트
ROOT = Path(__file__).resolve().parents[2]


# 현재 Bottle 데이터셋만 검토합니다
# Bottle 데이터셋 경로
DATASET = ROOT / "data" / "raw" / "mvtec_ad" / "bottle"


def count_images(directory):
    """폴더에 포함된 이미지 개수를 반환한다."""
    if not directory.is_dir():
        return 0

    extensions = {".png", ".jpg", ".jpeg"}

    # *rglob("*") 
    # 모든 하위 폴더에서 탐색합니다
    return sum(
        1 for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def main():
    print("=== MVTec AD 데이터셋 검사 ===\n")

    print("데이터 경로:", DATASET)

    if not DATASET.is_dir():
        raise SystemExit(
            "\n데이터셋 폴더가 없습니다. 다운로드 경로를 확인하세요."
        )

    train = DATASET / "train" / "good"
    test = DATASET / "test"
    ground_truth = DATASET / "ground_truth"

    if not train.is_dir() or not test.is_dir():
        raise SystemExit(
            "\ntrain/good 또는 test 폴더가 없습니다."
        )

    print("\n[학습 데이터]")
    print("정상 이미지:", count_images(train), "장")

    print("\n[테스트 데이터]")

    for folder in sorted(test.iterdir()):
        if not folder.is_dir():
            continue

        label = "정상" if folder.name == "good" else "불량"

        print(
            f"{folder.name} ({label}): "
            f"{count_images(folder)}장"
        )

    print("\n[정답 마스크]")
    print("마스크 이미지:", count_images(ground_truth), "장")

    print("\n데이터셋 검사 완료!")


if __name__ == "__main__":
    main()
