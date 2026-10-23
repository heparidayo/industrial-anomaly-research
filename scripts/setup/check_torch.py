
"""PyTorch와 CUDA GPU 실행 환경을 점검한다."""

import torch
import torchvision


def main():
    print("=== PyTorch 환경 확인 ===\n")

    print("PyTorch 버전:", torch.__version__)
    print("Torchvision 버전:", torchvision.__version__)
    print("PyTorch CUDA 버전:", torch.version.cuda)

    if not torch.cuda.is_available():
        raise SystemExit(
            "\n[오류] CUDA GPU를 사용할 수 없습니다."
        )

    # 사용할 GPU 선택 (여기서는 첫번째)
    device = torch.device("cuda:0")

    print("\n[GPU 정보]")
    print("GPU 이름:", torch.cuda.get_device_name(0))

    # GPU에서 실제 텐서 연산 수행
    a = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        device=device
    )

    b = torch.tensor(
        [[5.0, 6.0], [7.0, 8.0]],
        device=device
    )

    result = a @ b # 행렬곱

    print("\n[GPU 행렬 곱셈]")
    print(result)
    print("연산 장치:", result.device)

    print("\nPyTorch GPU 테스트 성공!")


if __name__ == "__main__":
    main()
