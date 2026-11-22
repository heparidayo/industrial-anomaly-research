# DRAEM 회사 데이터 밤샘 실행: 최종 모델 + 5-fold threshold + 리포트
# 실행: powershell -ExecutionPolicy Bypass -File scripts\draem\run_overnight.ps1
# 로그: outputs\draem\overnight_<날짜>.log
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = ".\.venv\Scripts\python.exe"
$steps = 2000
$common = @("--data-root", "data\processed\company_plc", "--steps", $steps,
            "--rotate-normal-prob", "0.5", "--rotate-degrees", "180", "--amp")
New-Item -ItemType Directory -Force outputs\draem | Out-Null
$log = "outputs\draem\overnight_$(Get-Date -Format yyyyMMdd_HHmm).log"

foreach ($cat in @("camera_1", "camera_2", "cam12")) {
    $done = Get-ChildItem "outputs\draem\$cat\draem_st${steps}_*amp*.pt" -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notmatch "_last|timing|fold" }
    if (-not $done) {
        "=== $cat 최종 모델 $(Get-Date) ===" | Tee-Object -Append $log
        & $py scripts\draem\train_draem.py --category $cat @common 2>&1 | Tee-Object -Append $log
    } else { "=== $cat 최종 모델 이미 있음: $($done[0].Name) ===" | Tee-Object -Append $log }

    "=== $cat k-fold threshold $(Get-Date) ===" | Tee-Object -Append $log
    & $py scripts\draem\kfold_threshold.py --category $cat --folds 5 @common 2>&1 | Tee-Object -Append $log
}

"=== 리포트 $(Get-Date) ===" | Tee-Object -Append $log
& $py scripts\draem\make_report.py --categories camera_1 camera_2 cam12 2>&1 | Tee-Object -Append $log
"=== 끝 $(Get-Date) ===" | Tee-Object -Append $log
