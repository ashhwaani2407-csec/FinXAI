# FIIntell — start backend + Streamlit dashboard (Windows PowerShell)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$env:PYTHONPATH = $Root
$env:FIINTELL_ENABLE_FINBERT = if ($env:FIINTELL_ENABLE_FINBERT) { $env:FIINTELL_ENABLE_FINBERT } else { "false" }
$env:FIINTELL_BACKEND_URL = "http://127.0.0.1:8000"

Write-Host "FIIntell root: $Root"
Write-Host "Installing dependencies (if needed)..."
python -m pip install -r requirements.txt -q
pip install -e . -q 2>$null

Write-Host "Starting FastAPI on http://127.0.0.1:8000 ..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$Root'; `$env:PYTHONPATH='$Root'; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
)

Start-Sleep -Seconds 4

Write-Host "Starting Streamlit on http://127.0.0.1:8501 ..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$Root'; `$env:PYTHONPATH='$Root'; `$env:FIINTELL_BACKEND_URL='http://127.0.0.1:8000'; `$env:FIINTELL_ENABLE_FINBERT='$($env:FIINTELL_ENABLE_FINBERT)'; streamlit run frontend/app.py --server.port 8501 --server.headless true"
)

Write-Host ""
Write-Host "Open dashboard: http://127.0.0.1:8501"
Write-Host "API health:     http://127.0.0.1:8000/healthz"
Write-Host "Stop: close the two PowerShell windows that opened."
