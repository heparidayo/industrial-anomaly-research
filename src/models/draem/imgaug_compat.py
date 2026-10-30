"""imgaug 0.4.0을 공용 환경(NumPy 2.x)에서 사용하기 위한 호환 패치.

imgaug 0.4.0은 NumPy 2.0에서 제거된 ``np.sctypes`` 등을 import 시점에 참조합니다.
공용 numpy 버전을 낮추지 않기 위해, imgaug를 불러오기 전에 빠진 속성만 채워 둡니다.

또한 ``iaa.pillike.Autocontrast``는 NumPy 2.x에서 TypeError가 나므로,
같은 동작(PIL ImageOps.autocontrast, cutoff 0~20%)을 하는 ``Autocontrast()``를 제공합니다.

DRAEM 코드에서는 imgaug를 직접 import하지 말고 아래처럼 사용하세요.

    from src.models.draem.imgaug_compat import iaa, Autocontrast
"""

import warnings

import numpy as np
from PIL import Image, ImageOps

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }
# np.sctypes는 Numpy 2.x에서 제거되었는데 imgaug는 import될 때 이 값을 읽어 어떤 자료형이 정수/실수인지 목록을 만듬.
# if not hasattr(...): NumPy에 이미 있으면 건드리지 않습니다. NumPy 1.x 환경에서도 안전
# np.object, np.int 같은 예전 이름을 예방 차원에서 채웁니다. NumPy 2에서는 이 이름을 확인하는 것만으로도 경고가 떠서, 확인하는 구간만 경고를 숨김.

# NumPy 1.24+에서 제거된 내장형 별칭 (imgaug 내부 일부 경로에서 참조)
# hasattr 검사 자체가 NumPy 2.x에서 FutureWarning을 내므로 경고를 숨깁니다.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    for _name, _alias in (("bool", bool), ("int", int), ("float", float),
                          ("complex", complex), ("object", object)):
        if not hasattr(np, _name):
            setattr(np, _name, _alias)

import imgaug as ia  # noqa: E402  (패치 이후에 import 해야 함)
import imgaug.augmenters as iaa  # noqa: E402


def _autocontrast_images(images, random_state, parents, hooks):
    """uint8 이미지마다 cutoff를 0~20%에서 뽑아 PIL autocontrast 적용 (채널별)."""
    cutoffs = random_state.integers(0, 21, size=(len(images),))
    result = []
    for image, cutoff in zip(images, cutoffs):
        pil = Image.fromarray(np.ascontiguousarray(image))
        result.append(np.asarray(ImageOps.autocontrast(pil, cutoff=int(cutoff))))
    return result
# PIL 원래 함수를 직접 호출

def Autocontrast():  # noqa: N802  (imgaug augmenter처럼 쓰도록 대문자 이름)
    """iaa.pillike.Autocontrast(cutoff=(0, 20)) 대체."""
    return iaa.Lambda(func_images=_autocontrast_images, name="Autocontrast")
# 위 함수를 ingaug 변환 형태로 감싸서 원래 쓰던 자리에 그대로 넣을 수 있게 함

__all__ = ["ia", "iaa", "Autocontrast"]
