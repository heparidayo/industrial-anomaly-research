# 부품만 회전시키는 정상편차 변환
#
# variations.py의 rotate_image는 이미지 전체(배경, 받침대 포함)를 돌리고
# 빈 모서리를 반사(reflect)로 채운다. 이 모서리가 실제 공정에 없는 인공물이 되어
# 과검을 과장할 수 있다.
# 실제 공정에서는 받침대는 고정이고 부품만 돌아가므로,
# 부품 원 안쪽만 회전하고 바깥(배경)은 원본을 그대로 둔다.

import cv2
import numpy as np


def rotate_part(image, center, radius, angle):
    """
    원(center, radius) 안쪽만 angle도 회전한다.

    image : BGR 이미지 (H, W, 3)
    center: 회전 중심 (x, y) 픽셀
    radius: 회전할 원의 반지름 픽셀
    angle : 회전 각도 (양수 = 반시계 방향, cv2 기준)
    """
    height, width = image.shape[:2]

    # 원 마스크 (경계를 7x7 가우시안으로 약 3px 부드럽게 → 이음새 완화)
    # 원 안 = 1.0, 바깥 = 0.0 인 흑백 마스크 (두께 -1 = 속이 꽉 찬 원)
    mask = np.zeros((height, width), np.float32)
    cv2.circle(mask, (round(center[0]), round(center[1])), round(radius), 1.0, -1)
    # [..., None]: (H, W) → (H, W, 1) 로 바꿔 컬러 3채널 이미지에 곱할 수 있게 함
    mask = cv2.GaussianBlur(mask, (7, 7), 0)[..., None]

    # 이미지 전체를 회전한 뒤 (variations.py의 rotate_image와 같은 보간·경계 설정, 중심만 부품 원 중심)
    # getRotationMatrix2D(중심, 각도, 배율 1.0) → 회전 변환 행렬
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # 원 안쪽은 회전본, 바깥은 원본
    # 결과 = 마스크 × 회전본 + (1 - 마스크) × 원본  (경계는 둘이 부드럽게 섞임)
    # round → uint8: 0~255 정수 이미지로 되돌림
    return (mask * rotated + (1 - mask) * image).round().astype(np.uint8)
