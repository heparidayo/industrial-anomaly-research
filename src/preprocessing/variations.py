from pathlib import Path
import cv2
import numpy as np


def adjust_brightness(image, percent):
    """이미지의 밝기를 지정한 비율만큼 변경한다."""

    factor = 1.0 + percent / 100.0 # 배율 뽑고 (ex: 1.25배)

    # 정수 연산 중 오버플로 방지를 위해 실수형으로 변환
    result = image.astype(np.float32) * factor

    # 픽셀 범위를 0~255로 제한
    result = np.clip(result, 0, 255)

    # 이미지 저장에 적합한 uint8 자료형으로 복원
    return result.astype(np.uint8)



def load_image(path: Path):
    """이미지 파일을 읽어온다."""
    image = cv2.imread(str(path)) # im 읽어서 numpy array로 변환 

    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")

    return image


def save_image(path: Path, image):
    """이미지를 지정한 경로에 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True) # 상위 폴더 자동 생성

    if not cv2.imwrite(str(path), image): # arr 다시 im로 저장
        raise RuntimeError(f"이미지 저장에 실패했습니다: {path}")
    

