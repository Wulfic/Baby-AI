# Baby-AI first-time setup.
#
# Builds the Fabric mod (baby_ai_mod) and creates the Python venv (.venv)
# with all dependencies for the baby_ai package. Re-run any time after
# pulling changes to pick up new dependencies or rebuild the mod.
#
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# ── 1. Fabric mod (Java) ──────────────────────────────────────────────
Write-Host "==> Building Fabric mod" -ForegroundColor Cyan

$jdk21 = Get-ChildItem "C:\Program Files\Microsoft" -Directory -Filter "jdk-21*" -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $jdk21) {
    $jdk21 = Get-ChildItem "C:\Program Files\Eclipse Adoptium" -Directory -Filter "jdk-21*" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $jdk21) {
    Write-Host "    JDK 21 not found — installing via winget..." -ForegroundColor Yellow
    winget install --id Microsoft.OpenJDK.21 -e --accept-source-agreements --accept-package-agreements
    $jdk21 = Get-ChildItem "C:\Program Files\Microsoft" -Directory -Filter "jdk-21*" |
        Select-Object -First 1 -ExpandProperty FullName
}
Write-Host "    Using JDK: $jdk21"

Push-Location "$root\baby_ai_mod"
try {
    $env:JAVA_HOME = $jdk21
    & .\gradlew.bat build --console=plain
    if ($LASTEXITCODE -ne 0) { throw "Gradle build failed" }
}
finally {
    Pop-Location
}

$jar = Get-ChildItem "$root\baby_ai_mod\build\libs" -Filter "*.jar" | Select-Object -First 1
Write-Host "    Built: $($jar.FullName)" -ForegroundColor Green

# ── 2. Install mod into .minecraft/mods ────────────────────────────────
$envFile = "$root\.env"
if (Test-Path $envFile) {
    $mcDirLine = Get-Content $envFile | Where-Object { $_ -match '^MC_DIR=' }
    if ($mcDirLine) {
        $mcDir = ($mcDirLine -split '=', 2)[1].Trim()
        $modsDir = Join-Path $mcDir "mods"
        if (Test-Path $modsDir) {
            Copy-Item $jar.FullName -Destination $modsDir -Force
            Write-Host "    Installed to $modsDir" -ForegroundColor Green
        }
        else {
            Write-Host "    Skipped install: $modsDir not found" -ForegroundColor Yellow
        }
    }
}

# ── 3. Python venv ──────────────────────────────────────────────────────
Write-Host "==> Setting up Python venv" -ForegroundColor Cyan

$venvPython = "$root\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    python -m venv "$root\.venv"
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
& $venvPython -m pip install -r "$root\requirements.txt"

Write-Host "==> Verifying imports" -ForegroundColor Cyan
& $venvPython -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
& $venvPython -c "import baby_ai.config" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "baby_ai package failed to import" }

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Mod jar:  baby_ai_mod\build\libs\$($jar.Name)"
Write-Host "  Activate venv: .venv\Scripts\Activate.ps1"
Write-Host "  Run:           .venv\Scripts\python.exe main.py"
