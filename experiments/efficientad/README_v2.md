# EfficientAD 팀 공통 v2 실험 실행 방법

팀 키트(`company_v2_team_kit`)로 만든 공통 데이터로 EfficientAD를 학습하고, 팀 공통 형식의 결과 CSV를 만드는 방법입니다.
카메라 1·2 통합 조건이며, `run_v2.ps1` 하나로 학습부터 결과 저장까지 실행합니다.

## 1. 실행 전 준비 (처음 한 번)

| 준비물 | 위치 (기본값) | 확인 |
|---|---|---|
| EfficientAD 가상환경 | `.venv` | 저장소 루트의 공용 환경 |
| 코드 (이 폴더) | `experiments/efficientad` | `train.py`, `run_v2.ps1` |
| 팀 공통 데이터 | `data/processed/company` | 각 형태 폴더의 `split.csv` |
| Teacher | 코드 폴더의 `teacher_s_anomalib.pth` | [anomalib 사전 증류 가중치](https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip)를 `convert_teacher.py`로 변환한 파일 (아래) |
| ImageNet | `data/raw/imagenet` | 학습 시 penalty에 사용하는 이미지 |

아래 명령은 저장소 루트에서 다음을 실행한 뒤 사용합니다. 데이터 위치가 다르면 실행 옵션으로 지정할 수 있습니다.

```powershell
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
Set-Location .\experiments\efficientad
```

### Teacher 가중치 받기 (`teacher_s_anomalib.pth`)

ImageNet으로 사전 증류된 EfficientAD Teacher는 anomalib이 공개한 가중치를 사용합니다.

- 다운로드: [efficientad_pretrained_weights.zip](https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip)
- 배포처: [anomalib GitHub](https://github.com/open-edge-platform/anomalib) (릴리스 `efficientad_pretrained_weights`)

압축 안의 `pretrained_teacher_small.pth`(S용)를 우리 코드 형식으로 변환해 코드 폴더에 둡니다.

```powershell
curl.exe -L -o efficientad_pretrained_weights.zip https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip
tar -xf efficientad_pretrained_weights.zip
$src = (Get-ChildItem -Recurse -Filter *small*.pth | Where-Object { $_.Name -notlike "teacher_s*" } | Select-Object -First 1).FullName
& $py convert_teacher.py --src $src --size s --out teacher_s_anomalib.pth
```

층마다 `OK`가 나오고 `저장: teacher_s_anomalib.pth`가 출력되면 완료입니다. M 모델은 `pretrained_teacher_medium.pth`를 `--size m`으로 변환합니다.

## 2. 실행 (검증 59장)

코드 폴더에서 PowerShell을 열고 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_v2.ps1
```

스크립트가 형태마다 다음을 순서대로 합니다.

1. **사전 점검**: Python, Teacher, ImageNet, 팀 데이터 위치와 GPU 인식 여부
2. **학습**: `train/good`으로 70,000회 학습 (`unused`는 쓰지 않음)
3. **검증 평가**: 검증 59장 AU-ROC 출력
4. **예측과 저장**: 검증 59장의 이미지별 점수를 팀 공통 형식으로 저장

기본값은 **고리 원본**(`company_v2_cam12`)과 **4단 적층**(`company_v2_cam12_polar`)입니다. 긴 띠의 정사각형 리사이즈 조건만 실행하려면 `-Forms company_v2_cam12_polarstrip`을 지정합니다.
학습 시간은 형태당 RTX 4050 노트북 약 2.3시간이며, RTX 5080에서는 훨씬 짧습니다.
밤새 돌린다면 Windows 업데이트의 "업데이트 일시 중지"를 켜 두세요(자동 재시작 방지).

## 3. 테스트 231장 평가 (설정 확정 후)

팀에서 임계값과 설정을 확정한 뒤, **이미 학습한 모델로** 테스트를 평가합니다. 다시 학습하지 않습니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_v2.ps1 -Split test -SkipTrain
```

## 4. 옵션

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `-Python` | 저장소의 `.venv/Scripts/python.exe` | 사용할 Python |
| `-DataRoot` | 저장소의 `data/processed/company` | 팀 공통 데이터 폴더 |
| `-ImageNet` | `..\..\data\raw\imagenet` | ImageNet 폴더 (penalty용) |
| `-Teacher` | `teacher_s_anomalib.pth` | Teacher 가중치 |
| `-OutDir` | `..\..\outputs\efficientad_v2` | 모델·결과·로그 저장 위치 |
| `-Split` | `validation` | 평가할 분할: `validation` 또는 `test` |
| `-Forms` | `company_v2_cam12,company_v2_cam12_polar` | 실행할 형태 |
| `-Iterations` | `70000` | 학습 반복 수 (논문 설정) |
| `-ImageSize` | `256` | 모델 입력 크기. 4단 적층은 `432`로 원본 해상도 그대로 사용 가능 (아래 4-1) |
| `-SkipTrain` | 꺼짐 | 학습을 건너뛰고 저장된 모델로 예측만 |

예시:

```powershell
# 4단 적층만 실행
powershell -ExecutionPolicy Bypass -File .\run_v2.ps1 -Forms company_v2_cam12_polar

# 동작 확인용으로 짧게 (결과는 의미 없음)
powershell -ExecutionPolicy Bypass -File .\run_v2.ps1 -Iterations 500 -OutDir ..\..\outputs\efficientad_v2_check

# 학교 PC처럼 경로가 다를 때
powershell -ExecutionPolicy Bypass -File .\run_v2.ps1 `
    -Python $py `
    -DataRoot E:\company_v2_team_kit\data\processed\company
```

### 4-1. 4단 적층을 원본 해상도(432)로 학습

기본 설정은 논문대로 모든 입력을 256×256으로 줄입니다. 4단 적층(432×432)은 이때 해상도의 약 40%를 잃으므로,
`-ImageSize 432`로 원본 그대로 넣을 수 있습니다. 출력 폴더를 따로 지정해 256 결과와 섞이지 않게 합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_v2.ps1 -Forms company_v2_cam12_polar -ImageSize 432 -OutDir ..\..\outputs\efficientad_v2_432
```

- 입력 크기는 모델 파일에 저장되어, 예측·테스트 때 자동으로 같은 크기가 적용됩니다.
- 계산량이 약 2.85배라 학습 시간도 그만큼 늘어납니다 (RTX 4050 노트북 기준 약 6~7시간).
- Teacher는 256 입력으로 증류된 가중치를 그대로 씁니다. 같은 33×33 픽셀 범위가 실제로는 더 좁은 면적을 보게 되어,
  더 세밀한 무늬에 반응합니다.
- 256 결과와 검증 AU-ROC를 비교해 더 나은 쪽을 팀 설정으로 정합니다.

## 5. 출력물

```text
outputs\efficientad_v2\
├── company_v2_cam12\         efficientad_s.pth (모델), result.json (학습 설정·검증 결과)
├── company_v2_cam12_polar\   같은 구성
├── results\
│   ├── efficientad_s_ring_validation.csv          ← 팀에 공유
│   ├── efficientad_s_polar_validation.csv         ← 팀에 공유
│   └── efficientad_s_*_validation_detail.csv      로컬·글로벌 점수 포함 (분석용)
└── logs\                     학습·예측 로그 (실행 시각별)
```

팀 공유 파일은 `predictions_template.csv`와 같은 열입니다.

| 열 | 내용 |
|---|---|
| `image_id` | 이미지 파일 이름 |
| `label` | 0 = 정상, 1 = 불량 |
| `defect_type` | good / over / step / 미흡 |
| `anomaly_score` | 이상 점수 (클수록 불량, 결합 맵 최댓값) |
| `inference_ms` | GPU 전송부터 점수가 CPU에 도착할 때까지 (이미지 읽기 제외, 워밍업 10회 후) |
| `camera`, `group_id` | 각 형태 폴더의 `split.csv` 값 |

## 6. 입력 처리 방식 (팀 보고용)

| 형태 | 원본 크기 | 모델 입력 | 처리 |
|---|---|---|---|
| 고리 원본 | 1216×1216 | 256×256 | bilinear 축소, 자르기 없음 |
| 4단 적층 | 432×432 | 256×256 (기본) 또는 432×432 (`-ImageSize 432`) | 256은 bilinear 축소, 432는 축소 없음. 자르기 없음. 긴 띠 전체를 4등분해 쌓은 영상이라 띠의 양 끝이 모두 포함됨 |
| 긴 띠 | 1728×108 | 명시적으로 선택하면 기본 256×256 | 가로는 6.75배 축소되고 세로는 약 2.37배 확대됨. 원래 종횡비를 유지하는 실험은 별도 입력 설정이 필요함 |

이후 이미지넷 평균·표준편차로 정규화합니다. 모델은 EfficientAD-S, Teacher는 anomalib 공개 사전 증류 가중치([다운로드](https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip))입니다.

## 7. 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| "이 시스템에서 스크립트를 실행할 수 없으므로…" | `powershell -ExecutionPolicy Bypass -File .\run_v2.ps1`처럼 실행 (시스템 설정은 바뀌지 않음) |
| `[중단] Python이 없습니다` | `-Python`으로 가상환경의 python.exe 경로 지정 |
| `[중단] 팀 공통 데이터가 없습니다` | 키트 전처리를 먼저 실행하거나 `-DataRoot` 지정 |
| `[중단] Teacher 파일이 없습니다` | `convert_teacher.py`로 만든 파일을 코드 폴더에 두거나 `-Teacher` 지정 |
| 사전 점검에 `cpu (GPU를 찾지 못함)` | GPU용 PyTorch(cu128)가 아님. 가상환경 확인 |
| `[중단] 학습된 모델이 없습니다` | `-SkipTrain` 없이 먼저 학습 |
| 실행 중 컴퓨터 재시작 | 그 형태는 처음부터 다시 학습해야 함. 끝난 형태는 `-Forms`로 빼고 실행 |
| 한글 경로·파일 오류 | 실행 전 `$env:PYTHONUTF8 = "1"` |
