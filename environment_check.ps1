# =============================================================================
# environment_check.ps1
# AI Evaluation Platform — Project 0: Environment Validation
# Run with: .\environment_check.ps1
# =============================================================================

#Requires -Version 5.1

$ErrorActionPreference = "Continue"

# ── Colours ───────────────────────────────────────────────────────────────────
function Write-Pass  { param($msg) Write-Host "[PASS] $msg" -ForegroundColor Green }
function Write-Fail  { param($msg) Write-Host "[FAIL] $msg" -ForegroundColor Red;   $script:Errors++ }
function Write-Warn  { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow; $script:Warnings++ }
function Write-Info  { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Header {
    param($title)
    Write-Host ""
    Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
}

$script:Errors   = 0
$script:Warnings = 0

function Get-CommandVersion {
    param([string]$Cmd, [string]$Args = "--version")
    try {
        $out = & $Cmd $Args 2>&1 | Select-Object -First 1
        return $out.ToString().Trim()
    } catch {
        return $null
    }
}

function Compare-Version {
    param([string]$Actual, [string]$Minimum)
    $a = [version]($Actual -replace '[^\d.]','')
    $m = [version]$Minimum
    return $a -ge $m
}

# =============================================================================
Write-Header "1. OPERATING SYSTEM"
# =============================================================================

$os = [System.Environment]::OSVersion
Write-Info "OS       : $($os.VersionString)"
Write-Info "Platform : $([System.Runtime.InteropServices.RuntimeInformation]::OSDescription)"
Write-Info "Arch     : $([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture)"

$winVer = (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion" -ErrorAction SilentlyContinue)
if ($winVer) {
    Write-Pass "OS: Windows $($winVer.DisplayVersion) (Build $($winVer.CurrentBuild))"
} else {
    Write-Pass "OS: $($os.VersionString)"
}

# =============================================================================
Write-Header "2. PYTHON VALIDATION"
# =============================================================================

# Python
$pyCmd = $null
foreach ($cmd in @("python", "python3")) {
    $v = Get-CommandVersion $cmd "--version"
    if ($v -and $v -match "Python (\d+\.\d+\.\d+)") {
        $pyVer = $Matches[1]
        $pyCmd = $cmd
        if (Compare-Version $pyVer "3.12.0") {
            Write-Pass "Python: $pyVer (>= 3.12.0 required)"
        } else {
            Write-Fail "Python: $pyVer is below minimum 3.12.0 — upgrade from https://python.org"
        }
        break
    }
}
if (-not $pyCmd) {
    Write-Fail "Python: NOT FOUND — install from https://python.org"
}

# pip
$pipV = Get-CommandVersion "pip" "--version"
if ($pipV) {
    $pipVer = ($pipV -split " ")[1]
    Write-Pass "pip: $pipVer"
} else {
    Write-Fail "pip: NOT FOUND — run: python -m ensurepip --upgrade"
}

# venv
try {
    & python -m venv --help 2>&1 | Out-Null
    Write-Pass "venv: available"
} catch {
    Write-Fail "venv: NOT AVAILABLE — run: pip install virtualenv"
}

# poetry
$poetryV = Get-CommandVersion "poetry" "--version"
if ($poetryV) {
    Write-Pass "poetry: $poetryV"
} else {
    Write-Warn "poetry: NOT INSTALLED"
    Write-Info "  Install: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -"
    Write-Info "  Or: pip install poetry"
}

# =============================================================================
Write-Header "3. OLLAMA VALIDATION"
# =============================================================================

$ollamaV = Get-CommandVersion "ollama" "--version"
if ($ollamaV) {
    Write-Pass "Ollama binary: installed ($ollamaV)"
} else {
    Write-Fail "Ollama: NOT INSTALLED — download from https://ollama.com"
}

# Check if server is running
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
    Write-Pass "Ollama server: RUNNING"

    $models = $response.models
    if ($models) {
        Write-Info "Installed models:"
        foreach ($m in $models) {
            Write-Info "  → $($m.name)  [$([math]::Round($m.size/1GB,1)) GB]"
        }
    }

    # Required models check
    $required = @("llama3", "qwen2.5", "mistral")
    foreach ($req in $required) {
        $found = $models | Where-Object { $_.name -like "*$req*" }
        if ($found) {
            Write-Pass "Model: $req found ($($found.name))"
        } else {
            Write-Warn "Model: $req NOT found — run: ollama pull $req"
        }
    }
} catch {
    Write-Warn "Ollama server: NOT RUNNING"
    Write-Info "  Start with: Start-Process ollama -ArgumentList 'serve'"
    Write-Info "  Or open the Ollama desktop app"
}

# =============================================================================
Write-Header "4. GIT VALIDATION"
# =============================================================================

$gitV = Get-CommandVersion "git" "--version"
if ($gitV) {
    Write-Pass "Git: $gitV"
} else {
    Write-Fail "Git: NOT INSTALLED — download from https://git-scm.com"
}

# Git identity
$gitUser  = git config --global user.name  2>$null
$gitEmail = git config --global user.email 2>$null

if ($gitUser)  { Write-Pass "Git user.name : $gitUser" }
else           { Write-Warn "Git user.name  not set — run: git config --global user.name 'Your Name'" }

if ($gitEmail) { Write-Pass "Git user.email: $gitEmail" }
else           { Write-Warn "Git user.email not set — run: git config --global user.email 'you@example.com'" }

# GitHub CLI
$ghV = Get-CommandVersion "gh" "--version"
if ($ghV) {
    Write-Pass "GitHub CLI: $($ghV.Split("`n")[0])"
    $ghAuth = gh auth status 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Pass "GitHub CLI: authenticated"
    } else {
        Write-Warn "GitHub CLI: not authenticated — run: gh auth login"
    }
} else {
    Write-Warn "GitHub CLI: NOT INSTALLED (optional) — https://cli.github.com"
}

# SSH key
$sshKeys = @("$env:USERPROFILE\.ssh\id_ed25519.pub", "$env:USERPROFILE\.ssh\id_rsa.pub")
$sshFound = $sshKeys | Where-Object { Test-Path $_ }
if ($sshFound) {
    Write-Pass "SSH key: found ($($sshFound[0]))"
} else {
    Write-Warn "SSH key: not found — run: ssh-keygen -t ed25519 -C '$gitEmail'"
}

# =============================================================================
Write-Header "5. PROJECT DEPENDENCIES PREVIEW"
# =============================================================================

Write-Info "Checking key Python packages (project venvs will install these):"

$deps = @("langchain", "langgraph", "mlflow", "fastapi", "uvicorn", "pytest", "pydantic", "httpx")
foreach ($dep in $deps) {
    $check = python -c "import $dep; print(getattr($dep, '__version__', 'installed'))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $check) {
        Write-Pass "  $dep`: $check"
    } else {
        Write-Info "  $dep`: not yet installed (will be installed per project)"
    }
}

# =============================================================================
Write-Header "SUMMARY"
# =============================================================================

Write-Host ""
if ($script:Errors -eq 0 -and $script:Warnings -eq 0) {
    Write-Host "  ✅ All checks passed — environment is ready!" -ForegroundColor Green
} elseif ($script:Errors -eq 0) {
    Write-Host "  ⚠️  $($script:Warnings) warning(s) — review above, then proceed." -ForegroundColor Yellow
} else {
    Write-Host "  ❌ $($script:Errors) error(s), $($script:Warnings) warning(s) — fix errors before proceeding." -ForegroundColor Red
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start Ollama: open Ollama desktop app or run: ollama serve" -ForegroundColor Cyan
Write-Host "  2. Pull missing models: ollama pull llama3" -ForegroundColor Cyan
Write-Host "  3. Install poetry: pip install poetry" -ForegroundColor Cyan
Write-Host "  4. Then run Project 1: HuggingFace Safety Dataset" -ForegroundColor Cyan
Write-Host ""
