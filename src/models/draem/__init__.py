"""DRAEM (Zavrtanik et al., ICCV 2021) 구현.

- networks.py      재구성 / 판별 U-Net
- perlin.py        Perlin 노이즈 (결함 모양 마스크)
- dataset.py       정상 이미지 + DTD 텍스처로 가짜 결함 합성
- loss.py          SSIM, Focal Loss
- model.py         DRAEMPredictor (추론용 래퍼)
- visualization.py DRAEM 결과 시각화
"""
