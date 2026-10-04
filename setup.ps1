# One-time setup (Windows PowerShell). Run from the project folder:  .\setup.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m data.seed
.\.venv\Scripts\python -m pytest -q
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "`nCreated .env. Open it and set LLM_PROVIDER and your API key before starting." -ForegroundColor Yellow
}
Write-Host "`nSetup finished. Next: .\start.ps1" -ForegroundColor Green
