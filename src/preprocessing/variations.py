from pathlib import Path
import cv2
import numpy as np


# 3. 위치 이동
def translate_image(image, dx, dy):
    """이미지를 x축 dx, y축 dy 픽셀만큼 이동한다."""

    height, width = image.shape[:2]

    # 2x3 이동 변환 행렬
    matrix = np.float32([
        [1, 0, dx], # x + dx
        [0, 1, dy]  # y + dy
    ])

    result = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return result

# 2. 회전 
def rotate_image(image, angle):
    """이미지를 중심 기준으로 angle도 회전한다."""

    height, width = image.shape[:2] # im 구조가 (Height, Width, Channels)

    # 회전 중심 (이미지 중심점)
    center = ((width - 1) / 2, (height - 1) / 2)

    # 2x3 회전 변환 행렬 생성
    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    ) # 1.0은 배율 (회전만 시키기위해)

    # 실제 이미지 회전
    result = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR, # 쌍선형 보간 (Bilinear Interpolation)
        borderMode=cv2.BORDER_REFLECT_101 # 임시 경계 처리
    )

    return result


# 1. 밝기
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
    

