<#
EfficientAD 팀 공통 v2 실험 실행 스크립트 (카메라 1·2 통합)

하는 일 (형태마다):
  1) train.py   : train/good으로 학습 → 지정한 분할(validation 또는 test) 평가
  2) predict.py : 같은 분할을 예측 → 팀 공통 형식 CSV 저장

형태: 고리 원본(company_v2_cam12), 4단 적층(company_v2_cam12_polar)
      긴 띠(polarstrip)는 256×256 입력에서 크게 왜곡되어 4단 적층으로 대신함

사용 예 (이 파일이 있는 코드 폴더에서):
  powershell -ExecutionPolicy Bypass -File .\run_v2.ps1                       # 학습 + 검증 59장
  powershell -ExecutionPolicy Bypass -File .\run_v2.ps1 -Split test -SkipTrain # 학습된 모델로 테스트 231장
  powershell -ExecutionPolicy Bypass -File .\run_v2.ps1 -Forms company_v2_cam12_polar -ImageSize 432 -OutDir ..\..\outputs\efficientad_v2_432
                                                                              # 4단 적층을 원본 해상도(432)로
#>
param(
    [string]$Python   = (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"),
    [string]$DataRoot = (Join-Path $PSScriptRoot "..\..\data\processed\company"),
    [string]$ImageNet = "..\..\data\raw\imagenet",
    [string]$Teacher  = "teacher_s_anomalib.pth",
    [string]$OutDir   = "..\..\outputs\efficientad_v2",
    [ValidateSet("validation", "test")][string]$Split = "validation",
    [ValidateSet("company_v2_cam12", "company_v2_cam12_polarstrip", "company_v2_cam12_polar")]
    [string[]]$Forms  = @("company_v2_cam12", "company_v2_cam12_polar"),
    [int]$Iterations  = 70000,
    [int]$ImageSize   = 256,
    [switch]$SkipTrain
)

Set-Location -LiteralPath $PSScriptRoot
$short = @{ "company_v2_cam12" = "ring"; "company_v2_cam12_polar" = "polar"; "company_v2_cam12_polarstrip" = "strip" }

function Fail([string]$msg) { Write-Host "[중단] $msg" -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------ 사전 점검
Write-Host "== 사전 점검 ==" -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $Python))  { Fail "Python이 없습니다: $Python (-Python 으로 경로 지정)" }
if (-not (Test-Path -LiteralPath "train.py")) { Fail "코드 폴더에서 실행하세요 (train.py 없음): $PSScriptRoot" }
if (-not $SkipTrain) {
    if (-not (Test-Path -LiteralPath $Teacher))  { Fail "Teacher 파일이 없습니다: $Teacher (-Teacher 로 지정)" }
    if (-not (Test-Path -LiteralPath $ImageNet)) { Fail "ImageNet 폴더가 없습니다: $ImageNet (-ImageNet 으로 지정)" }
}
foreach ($f in $Forms) {
    if (-not (Test-Path -LiteralPath (Join-Path $DataRoot "$f\split.csv"))) { Fail "팀 공통 데이터가 없습니다: $DataRoot\$f" }
    if ($f -eq "company_v2_cam12_polarstrip") { Write-Host "  긴 띠를 정사각형으로 리사이즈하므로 가로세로 비율이 달라집니다." -ForegroundColor Yellow }
}
& $Python -c "import torch; print('  장치:', 'cuda ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu (GPU를 찾지 못함)')"
$resultDir = Join-Path $OutDir "results"
$logDir = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force -Path $resultDir, $logDir | Out-Null
Write-Host "  데이터: $DataRoot"
Write-Host "  분할: $Split, 형태: $($Forms -join ', '), 학습: $(if ($SkipTrain) { '건너뜀' } else { "$Iterations 회, 입력 ${ImageSize}×${ImageSize}" })"

# ------------------------------------------------------------ 형태별 실행
$summary = @()
foreach ($f in $Forms) {
    $name = if ($short.ContainsKey($f)) { $short[$f] } else { $f }
    $ckpt = Join-Path $OutDir "$f\efficientad_s.pth"
    $stamp = Get-Date -Format "yyyyMMdd_HHmm"

    if (-not $SkipTrain) {
        Write-Host "`n== [$name] 학습 + $Split 평가 ==" -ForegroundColor Cyan
        $log = Join-Path $logDir "train_${name}_$stamp.txt"
        & $Python -u train.py --data_root $DataRoot --category $f --teacher $Teacher --imagenet_dir $ImageNet `
            --no_pixel_eval --eval_split $Split --iterations $Iterations --image_size $ImageSize --out_dir $OutDir 2>&1 |
            ForEach-Object { "$_" } | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { Fail "[$name] 학습 실패. 로그: $log" }
    }
    if (-not (Test-Path -LiteralPath $ckpt)) { Fail "[$name] 학습된 모델이 없습니다: $ckpt (-SkipTrain 없이 먼저 학습)" }

    Write-Host "`n== [$name] $Split 예측 → 팀 공통 형식 ==" -ForegroundColor Cyan
    $team = Join-Path $resultDir "efficientad_s_${name}_$Split.csv"
    $raw = Join-Path $resultDir "efficientad_s_${name}_${Split}_detail.csv"
    $log = Join-Path $logDir "predict_${name}_${Split}_$stamp.txt"
    & $Python predict.py --checkpoint $ckpt --data_root $DataRoot --category $f --split $Split `
        --out_csv $raw --template_csv $team 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Fail "[$name] 예측 실패. 로그: $log" }
    $out = Get-Content -LiteralPath $log -Encoding UTF8
    $auc = ($out | Select-String -Pattern "^AU-ROC: (.+)$" | Select-Object -First 1)
    $ms = ($out | Select-String -Pattern "^평균 추론 시간: (.+)$" | Select-Object -First 1)
    $summary += [pscustomobject]@{
        형태 = $name; 분할 = $Split
        AUROC = if ($auc) { $auc.Matches[0].Groups[1].Value } else { "-" }
        추론시간 = if ($ms) { $ms.Matches[0].Groups[1].Value } else { "-" }
        결과파일 = $team
    }
}

Write-Host "`n== 요약 ==" -ForegroundColor Green
foreach ($r in $summary) {
    Write-Host ("  [{0}] {1}  AU-ROC {2}  추론 {3}" -f $r.형태, $r.분할, $r.AUROC, $r.추론시간)
    Write-Host ("        → {0}" -f $r.결과파일)
}
Write-Host ("팀에 공유할 파일: " + (Join-Path $resultDir "efficientad_s_*_$Split.csv")) -ForegroundColor Green
