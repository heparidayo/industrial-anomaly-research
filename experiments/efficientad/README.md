# EfficientAD 구현 (로즈마리팀 · EfficientAD 담당)

논문 *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies*
(Batzner et al., arXiv:2303.14535v3)의 Appendix A(Table 6~8, Algorithm 1~3)를 PyTorch로 구현한 코드입니다.
정상 이미지만으로 학습해, 이미지 단위 이상 점수와 픽셀 단위 이상 맵을 출력합니다.

- [1. 환경](#1-환경)
- [2. 파일 구성](#2-파일-구성)
- [3. 전체 흐름](#3-전체-흐름)
- [4. 실행 방법](#4-실행-방법)
- [5. 출력물](#5-출력물)
- [6. 논문과 다르거나 논문에 없어서 정한 것](#6-논문과-다르거나-논문에-없어서-정한-것)
- [7. 주의사항](#7-주의사항)

## 1. 환경

팀 공용 환경(`Python 3.11`, `torch 2.9.0`, `torchvision 0.24.0`)과 공용 `requirements.txt`만으로 동작하며,
**추가 패키지가 필요 없습니다.** AU-PRO 계산의 SciPy는 scikit-learn과 함께, 히트맵의 OpenCV는 공용 패키지로 설치됩니다.

팀 공통 분할로 실험할 때는 [v2 실행 안내](README_v2.md)를 따릅니다. 아래 개별 명령은 저장소 루트에서 다음 두 줄을 실행한 뒤 사용합니다.

```powershell
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
Set-Location .\experiments\efficientad
```

새 PowerShell 창을 열 때마다 위 두 줄을 먼저 실행하세요.

## 2. 파일 구성

| 구분 | 파일 | 역할 | 논문 근거 |
|---|---|---|---|
| 모델 | `pdn.py` | Teacher·Student 공용 PDN (S/M), 패딩 스위치 | Table 6, 7 |
| | `feature_extractor.py` | 증류 정답 Ψ (WideResNet-101 + PatchCore 후처리) | Appendix A.2 |
| | `student.py` | Student(768채널), hard feature loss, pretraining penalty | Section 3.2 |
| | `autoencoder.py` | 오토인코더 (64차원 병목) | Table 8 |
| | `efficientad.py` | 세 네트워크 + 정규화 값을 묶은 모델, 추론 | Algorithm 2 |
| 학습 | `distill_teacher.py` | Teacher 증류 (ImageNet, 한 번만) | Algorithm 3 |
| | `train.py` | **전체 학습** + 분위수 계산 + 테스트 평가 | Algorithm 1 |
| | `train_student.py` | Student–Teacher만 학습하는 점검용 (선택) | Algorithm 1 일부 |
| 추론·평가 | `predict.py` | 예측 CSV, 히트맵, 지연 시간 측정 | Algorithm 2 |
| | `metrics.py` | 이미지 AU-ROC, 픽셀 AU-ROC, AU-PRO | Section 4 |
| | `combine_cameras.py` | 두 카메라 결합 판정·유형별 평가 | 자체 추가 |
| 데이터 | `datasets.py` | 데이터 로더, 검증 분할, 증강(회전·이동 포함) | Appendix A.1 |
| | `preprocess.py` | 입력 전처리: 면취면 주변 자르기, 이미지별 밝기 정규화 (+미리보기) | 자체 추가 |
| | `image_stats.py` | 날짜·폴더·카메라별 밝기·해상도 통계 | 자체 추가 |
| | `prepare_company_data.py` | 산업체 데이터 → 카메라별 폴더 구조 변환 | 자체 추가 |
| | `inspect_company_data.py` | 데이터 구조 점검 (내용 없이 구조만 요약) | 자체 추가 |
| | `make_dummy_dataset.py` | 테스트용 가짜 데이터 생성 | 자체 추가 |
| | `download_imagenet_hf.py` | Hugging Face ImageNet(parquet) → JPEG 추출 | 자체 추가 |
| | `extract_imagenet_subset.py` | Kaggle ImageNet zip에서 클래스 균형 부분 추출 | 자체 추가 |

## 3. 전체 흐름

```text
[1단계, 한 번만]  ImageNet ──▶ distill_teacher.py ──▶ teacher_s.pth
                                   (pdn.py, feature_extractor.py)

[2단계, 부품·카메라마다]
  원본 데이터 ──▶ prepare_company_data.py ──▶ company_ad\cam1, cam2
                                                   │
  teacher_s.pth + ImageNet(penalty용) ──▶ train.py ┘──▶ efficientad_s.pth, result.json
                                   (student.py, autoencoder.py, efficientad.py)

[3단계, 판정]
  efficientad_s.pth ──▶ predict.py ──▶ pred_cam1.csv, pred_cam2.csv ──▶ combine_cameras.py
```

데이터 폴더 규칙 (`train.py`, `predict.py`가 읽는 구조):

```text
<data_root>\<category>\
├── train\good\              정상 이미지 (학습용, 이 중 10%는 자동으로 검증용)
├── test\good\               정상 이미지 (테스트용)
├── test\<불량유형>\          불량 이미지 (폴더 이름 = 유형, good이 아니면 불량)
└── ground_truth\<불량유형>\  <이름>_mask.png (선택, 픽셀 평가용)
```

## 4. 실행 방법

### 4-1. 구조 검증 (계산량이 작아 CPU로 실행, 학습·추론은 GPU 자동 사용)

```powershell
& $py pdn.py           # 출력 크기, 파라미터 수, 수용 영역 33×33 확인
& $py student.py       # hard feature loss가 0.1% 원소를 고르는지 확인
& $py autoencoder.py   # 디코더 각 층 크기 확인
& $py metrics.py       # 평가 지표 확인
```

### 4-2. ImageNet 준비

Hugging Face(`ILSVRC/imagenet-1k`)에서 이용 조건에 동의하고 로그인한 뒤 받습니다.

```powershell
& $py -m pip install huggingface_hub pyarrow
& $py -c "from huggingface_hub import login; login()"
& $py download_imagenet_hf.py --out ..\..\data\raw\imagenet --val               # 검증 세트 5만 장
& $py download_imagenet_hf.py --out ..\..\data\raw\imagenet --train_shards 20   # (직접 증류 시) 학습 세트 일부
```

검증 세트는 pretraining penalty에 충분하고, 직접 증류에는 학습 세트 조각을 추가하는 것을 권합니다.
Kaggle zip을 받은 경우에는 `extract_imagenet_subset.py --per_class 300`으로 부분 추출합니다.

### 4-3. Teacher 증류 (1단계)

```powershell
# 짧은 점검
& $py distill_teacher.py --imagenet_dir ..\..\data\raw\imagenet --size s `
    --iterations 200 --stat_images 160 --amp --workers 4 --out teacher_check.pth

# 본 실행 (논문 설정: 60,000회, 배치 16). 중단 시 같은 명령에 --resume
& $py distill_teacher.py --imagenet_dir ..\..\data\raw\imagenet --size s `
    --amp --workers 4 --ckpt_every 5000 --log_every 500 --out teacher_s.pth `
    2>&1 | Tee-Object -FilePath distill_s.log
```

RTX 4050 Laptop(6GB) 측정 기준 초당 0.6~0.9회 → **약 19~28시간**. 절전 모드를 끄고 전원을 연결하세요.

**증류를 건너뛰려면** anomalib이 공개한 사전 증류 Teacher 가중치([efficientad_pretrained_weights.zip](https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip), 배포처 [anomalib GitHub](https://github.com/open-edge-platform/anomalib))를 받아 변환합니다.
층 수와 가중치 모양이 우리 `pdn.py`와 맞는지 확인한 뒤 저장합니다.

```powershell
curl.exe -L -o efficientad_pretrained_weights.zip https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip
tar -xf efficientad_pretrained_weights.zip
& $py convert_teacher.py --src pretrained_teacher_small.pth --size s --out teacher_s_anomalib.pth
```

### 4-4. 데이터 준비

```powershell
# (선택) 동작 확인용 가짜 데이터
& $py make_dummy_dataset.py --out ..\..\data\raw\dummy --category widget

# 산업체 데이터: 구조 점검 → 변환 미리보기 → 변환
& $py inspect_company_data.py --root "D:\경로\대세 라벨링"
& $py prepare_company_data.py --root "D:\경로\대세 라벨링" --out ..\..\data\processed\company_ad --dry_run
& $py prepare_company_data.py --root "D:\경로\대세 라벨링" --out ..\..\data\processed\company_ad
```

`prepare_company_data.py`는 파일 이름(`..._camera_1`)으로 카메라를 나누고, 정상은 **제품 단위·시간순**으로
학습 80% / 테스트 20%로 나눕니다(`--test_ratio`, `--split random` 변경 가능). 불량 이미지는 전부 테스트용입니다.

### 4-5. 학습 (2단계, 카메라마다)

```powershell
& $py train.py --data_root ..\..\data\processed\company_ad --category cam1 `
    --teacher teacher_s.pth --imagenet_dir ..\..\data\raw\imagenet --size s `
    --no_pixel_eval --out_dir ..\..\outputs\efficientad
```

| 주요 옵션 | 의미 | 기본값 |
|---|---|---|
| `--iterations` | 학습 반복 수 (논문 70,000) | 70000 |
| `--size` | `s` 또는 `m` (M은 Teacher도 M으로 증류 필요) | s |
| `--image_size` | 모델 입력 크기 (논문 256, 4단 적층은 432 가능) | 256 |
| `--no_padding` | Teacher·Student 무패딩 (논문 속도 측정 설정) | 꺼짐 |
| `--no_pixel_eval` | 마스크가 없을 때 픽셀 평가 생략 | 꺼짐 |
| `--skip_eval` | 테스트 세트 없이 학습만 | 꺼짐 |
| `--eval_split` | 학습 후 평가할 폴더: `test` / `validation` | test |
| `--rot_deg`, `--shift_px` | 정상 편차 증강 (회전·이동, 논문에 없음) | 0 (꺼짐) |
| `--geo_target` | 증강 대상: `both` / `st` / `ae` | both |
| `--crop` | 면취면 주변 자르기: `none` / `center:0.7` / `box:x0,y0,x1,y1` / `auto:0.15` | none |
| `--bright_norm` | 이미지별 밝기 정규화: `none` / `mean` / `meanstd` | none |
| `--crop` | 면취면 주변 자르기: `none` / `center:0.7` / `box:x0,y0,x1,y1` / `auto:0.15` | none |
| `--bright_norm` | 이미지별 밝기 정규화: `none` / `mean` / `meanstd` | none |

증강을 켜기 전에 미리보기로 확인하세요.

```powershell
& $py datasets.py --images ..\..\data\processed\company_ad\cam1\train\good --rot_deg 180 --shift_px 2
```

### 4-6. 추론과 두 카메라 결합 (3단계)

```powershell
& $py predict.py --checkpoint ..\..\outputs\efficientad\cam1\efficientad_s.pth `
    --data_root ..\..\data\processed\company_ad --category cam1 --out_csv pred_cam1.csv --save_maps maps_cam1
& $py predict.py --checkpoint ..\..\outputs\efficientad\cam2\efficientad_s.pth `
    --data_root ..\..\data\processed\company_ad --category cam2 --out_csv pred_cam2.csv
& $py combine_cameras.py --cam1 pred_cam1.csv --cam2 pred_cam2.csv --out combined
```

라벨 없는 이미지 폴더를 검사할 때는 `--data_root`, `--category` 대신 `--input <폴더>`를 씁니다.

### 4-7. 지연 시간 측정

```powershell
& $py predict.py --checkpoint ..\..\outputs\efficientad\cam1\efficientad_s.pth --benchmark 1000 --fp16
```

CPU→GPU 전송부터 결과가 CPU에 도착할 때까지, 1,000회 워밍업 후 1,000회 평균 (논문 Appendix E 방식).

### 4-8. 팀 공통 v2 데이터 (company_v2_cam12, 카메라 1·2 통합)

팀 키트로 만든 `data/processed/company/` 아래 폴더를 그대로 씁니다. 담당자 구현의 기본 실행 조건은 **고리 원본**과 **4단 적층**입니다. 긴 띠를 명시적으로 선택하면 정사각형으로 리사이즈합니다. 입력 크기와 실행 예시는 [v2 실행 안내](README_v2.md)에 있습니다.

```powershell
$v2 = "..\..\data\processed\company"
# 학습 + 검증 59장 평가 (형태마다)
& $py -u train.py --data_root $v2 --category company_v2_cam12_polar --teacher teacher_s_anomalib.pth `
    --imagenet_dir ..\..\data\raw\imagenet --no_pixel_eval --eval_split validation --out_dir ..\..\outputs\efficientad_v2
# 검증 59장 팀 공통 형식 결과
& $py predict.py --checkpoint ..\..\outputs\efficientad_v2\company_v2_cam12_polar\efficientad_s.pth `
    --data_root $v2 --category company_v2_cam12_polar --split validation `
    --out_csv pred_polar_val.csv --template_csv efficientad_s_polar_validation.csv
```

`--template_csv`는 `image_id, label, defect_type, anomaly_score, inference_ms, camera, group_id` 형식이며,
`camera`, `group_id`, `defect_type`은 각 폴더의 `split.csv`에서 가져옵니다. `inference_ms`는 GPU 전송부터 점수가
CPU에 도착할 때까지의 시간(이미지 읽기·전처리 제외, 첫 이미지 전 워밍업 10회)입니다.
설정과 임계값을 확정한 뒤에는 `--split test`로 테스트 231장을 평가합니다.

## 5. 출력물

| 파일 | 내용 |
|---|---|
| `efficientad_<size>.pth` | Teacher 포함 전체 모델 + 정규화 통계 + 분위수 4개 + 학습 설정(meta). 이 파일 하나로 추론 |
| `result.json` | 분위수, 이미지 AU-ROC(결합·로컬만·글로벌만, 불량 유형별), 픽셀 AU-ROC, AU-PRO |
| `pred_*.csv` | `image_path, label, score, score_local, score_global` (팀 공통 형식 확정 시 `predict.py`의 `write_csv`만 수정) |
| `maps_*\*.npy, *.png` | 원본 크기 결합 이상 맵과 히트맵 (0~1 컬러 스케일, 0.1 ≈ 정상 픽셀 상위 0.5% 수준) |
| `combined\combined_result.json` | 제품 단위 AU-ROC, 유형별·카메라별 AU-ROC, 참고용 임계값별 검출률 |

## 6. 논문과 다르거나 논문에 없어서 정한 것

| 항목 | 논문 | 구현 | 이유 |
|---|---|---|---|
| 검증 세트 | 보지 않은 정상 학습 이미지 | `train/good`의 10%, 시드 고정 | 비율 미명시. 팀 기준 확정 시 교체 |
| 경계값 계산 | p-분위수 | `kthvalue` | `torch.quantile` 원소 수 제한 |
| Ψ 연산 순서 | 보간 → 채널 풀링 | 채널 풀링 → 보간 | 메모리 절약, 선형이라 결과 동일 |
| 증류 시 패딩 | 언급 없음 | 항상 켬 | Ψ와 출력 크기 64×64 일치. 가중치는 무패딩 모델에도 그대로 사용 가능 |
| 혼합 정밀도 | float32 학습 | `--amp`로 Ψ만 float16 (선택) | 6GB GPU 대응 |
| AU-PRO | 모든 점수를 임계값으로 | 2,000구간 히스토그램 | 메모리 절약. 정확한 계산과 소수점 다섯째 자리까지 일치 |
| 정상 편차 증강 | 없음 | 회전·이동 선택 옵션 | 팀 연구 질문(정상 편차) 실험용 |
| 입력 전처리 | 전체 이미지를 256으로 축소 | 자르기·밝기 정규화 선택 옵션 (모델 파일에 저장, 예측 시 자동 적용) | 날짜별 밝기 차이, 1216→256 축소로 인한 폭 정보 손실 대응 |
| 입력 전처리 | 전체 이미지를 256으로 축소 | 자르기·밝기 정규화 선택 옵션 (모델 파일에 저장, 예측 시 자동 적용) | 날짜별 밝기 차이, 1216→256 축소로 인한 폭 정보 손실 대응 |

그 밖의 하이퍼파라미터(반복 횟수, 학습률, p_hard 0.999, 분위수 0.9·0.995 등)는 모두 Appendix A 그대로입니다.

## 7. 주의사항

- **산업체 데이터**는 외부 AI 서비스에 올리지 마세요. 모든 스크립트는 로컬에서만 동작합니다.
- **Teacher 품질이 결과를 좌우합니다.** 동작 확인용 Teacher(배경 화면·MVTec으로 짧게 증류)는 실험에 쓰지 마세요.
- **ImageNet**은 Teacher 증류와 pretraining penalty에 필요하며 산업체 데이터로 대체할 수 없습니다(일부만 있어도 동작).
- **미흡·과다 같은 미세 치수 불량**은 논문이 한계로 명시한 영역입니다. 불량 유형별 결과를 함께 보고하세요.
- 최종 보고 수치는 팀 평가 스크립트나 공식 평가 스크립트로 한 번 더 확인하는 것을 권합니다.
