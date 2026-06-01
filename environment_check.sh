#!/usr/bin/env bash
# =============================================================================
# environment_check.sh
# AI Evaluation Platform — Project 0: Environment Validation
# Run with: bash environment_check.sh
# =============================================================================

# strict mode intentionally omitted for cross-platform Git Bash compatibility

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

PASS="${GREEN}[PASS]${NC}"
FAIL="${RED}[FAIL]${NC}"
WARN="${YELLOW}[WARN]${NC}"
INFO="${CYAN}[INFO]${NC}"

ERRORS=0
WARNINGS=0

# ── Helpers ───────────────────────────────────────────────────────────────────
pass()  { echo -e "${PASS} $1"; }
fail()  { echo -e "${FAIL} $1"; ((ERRORS++)) || true; }
warn()  { echo -e "${WARN} $1"; ((WARNINGS++)) || true; }
info()  { echo -e "${INFO} $1"; }
header(){ echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${NC}"; \
          echo -e "${BOLD}${CYAN}  $1${NC}"; \
          echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}"; }

require_version() {
    local label="$1" actual="$2" required="$3"
    local a_major a_minor a_patch r_major r_minor r_patch
    IFS='.' read -r a_major a_minor a_patch <<< "$actual"
    IFS='.' read -r r_major r_minor r_patch <<< "$required"
    a_major=${a_major:-0}; a_minor=${a_minor:-0}; a_patch=${a_patch:-0}
    r_major=${r_major:-0}; r_minor=${r_minor:-0}; r_patch=${r_patch:-0}
    if (( a_major > r_major )) || \
       (( a_major == r_major && a_minor > r_minor )) || \
       (( a_major == r_major && a_minor == r_minor && a_patch >= r_patch )); then
        pass "$label: $actual (>= $required required)"
    else
        fail "$label: $actual is below minimum $required"
    fi
}

# =============================================================================
header "1. OPERATING SYSTEM"
# =============================================================================

OS_NAME=$(uname -s 2>/dev/null || echo "Unknown")
OS_ARCH=$(uname -m 2>/dev/null || echo "Unknown")
OS_FULL=$(uname -a 2>/dev/null || echo "Unknown")

info "Kernel : $OS_NAME"
info "Arch   : $OS_ARCH"
info "Full   : $OS_FULL"

if [[ "$OS_NAME" == MINGW* || "$OS_NAME" == MSYS* || "$OS_NAME" == CYGWIN* ]]; then
    pass "OS: Windows (Git Bash / MSYS2 environment detected)"
elif [[ "$OS_NAME" == "Linux" ]]; then
    pass "OS: Linux"
elif [[ "$OS_NAME" == "Darwin" ]]; then
    pass "OS: macOS"
else
    warn "OS: Unrecognised — $OS_NAME (may still work)"
fi

# =============================================================================
header "2. PYTHON VALIDATION"
# =============================================================================

# Python binary — on Windows Git Bash, 'python3' may resolve to a stub
PY_VERSION=""
for _cmd in python3 python; do
    _raw=$(${_cmd} --version 2>&1 || true)
    if echo "$_raw" | grep -qE "^Python [0-9]+\.[0-9]+"; then
        PY_VERSION=$(echo "$_raw" | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")
        PY_CMD="$_cmd"
        require_version "Python ($_cmd)" "$PY_VERSION" "3.12.0"
        break
    fi
done
if [[ -z "$PY_VERSION" ]]; then
    fail "Python: NOT FOUND — install from https://python.org"
    PY_VERSION="0.0.0"
fi

# pip
if command -v pip &>/dev/null; then
    PIP_VERSION=$(pip --version 2>&1 | awk '{print $2}')
    pass "pip: $PIP_VERSION"
else
    fail "pip: NOT FOUND — run: python -m ensurepip --upgrade"
fi

# venv (test by actually running it, not --help)
if python3 -m venv --help &>/dev/null 2>&1; then
    pass "venv: available"
else
    # Windows fallback check
    if python -m venv --help &>/dev/null 2>&1; then
        pass "venv: available (via python)"
    else
        fail "venv: NOT AVAILABLE — run: pip install virtualenv"
    fi
fi

# poetry
if command -v poetry &>/dev/null; then
    POETRY_VERSION=$(poetry --version 2>&1 | awk '{print $3}')
    pass "poetry: $POETRY_VERSION"
else
    warn "poetry: NOT INSTALLED"
    info "  Install: curl -sSL https://install.python-poetry.org | python3 -"
    info "  Or: pip install poetry"
fi

# =============================================================================
header "3. OLLAMA VALIDATION"
# =============================================================================

if command -v ollama &>/dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
    pass "Ollama binary: installed (version $OLLAMA_VERSION)"
else
    fail "Ollama: NOT INSTALLED — download from https://ollama.com"
fi

# Check if Ollama server is running
if ollama list &>/dev/null 2>&1; then
    pass "Ollama server: RUNNING"
    echo ""
    info "Installed models:"
    ollama list | tail -n +2 | while IFS= read -r line; do
        info "  → $line"
    done

    # Check for required models
    MODELS_RAW=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}')
    REQUIRED_MODELS=("llama3" "qwen2.5" "mistral")
    for model in "${REQUIRED_MODELS[@]}"; do
        if echo "$MODELS_RAW" | grep -qi "$model"; then
            pass "Model: $model found"
        else
            warn "Model: $model NOT found — run: ollama pull $model"
        fi
    done
else
    warn "Ollama server: NOT RUNNING"
    info "  Start with: ollama serve"
    info "  Or on Windows: start the Ollama desktop app"
fi

# =============================================================================
header "4. GIT VALIDATION"
# =============================================================================

if command -v git &>/dev/null; then
    GIT_VERSION=$(git --version | awk '{print $3}')
    pass "Git: $GIT_VERSION"
else
    fail "Git: NOT INSTALLED — download from https://git-scm.com"
fi

# Git config
GIT_USER=$(git config --global user.name 2>/dev/null || echo "")
GIT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

if [[ -n "$GIT_USER" ]]; then
    pass "Git user.name: $GIT_USER"
else
    warn "Git user.name: not set — run: git config --global user.name 'Your Name'"
fi

if [[ -n "$GIT_EMAIL" ]]; then
    pass "Git user.email: $GIT_EMAIL"
else
    warn "Git user.email: not set — run: git config --global user.email 'you@example.com'"
fi

# GitHub CLI (optional)
if command -v gh &>/dev/null; then
    GH_VERSION=$(gh --version | head -1 | awk '{print $3}')
    pass "GitHub CLI (gh): $GH_VERSION"
    if gh auth status &>/dev/null 2>&1; then
        pass "GitHub CLI: authenticated"
    else
        warn "GitHub CLI: installed but NOT authenticated — run: gh auth login"
    fi
else
    warn "GitHub CLI: NOT INSTALLED (optional)"
    info "  Install: https://cli.github.com"
fi

# SSH key check
if [[ -f "$HOME/.ssh/id_ed25519.pub" || -f "$HOME/.ssh/id_rsa.pub" ]]; then
    pass "SSH key: found"
else
    warn "SSH key: not found — run: ssh-keygen -t ed25519 -C '$GIT_EMAIL'"
fi

# =============================================================================
header "5. PROJECT DEPENDENCIES PREVIEW"
# =============================================================================

DEPS=(
    "langchain"
    "langgraph"
    "mlflow"
    "fastapi"
    "uvicorn"
    "pytest"
    "pydantic"
    "httpx"
)

info "Checking key Python packages (project venvs will install these):"
for dep in "${DEPS[@]}"; do
    if python3 -c "import $dep" &>/dev/null 2>&1; then
        VERSION=$(python3 -c "import $dep; print(getattr($dep, '__version__', 'installed'))" 2>/dev/null)
        pass "  $dep: $VERSION"
    else
        info "  $dep: not yet installed (will be installed per project)"
    fi
done

# =============================================================================
header "SUMMARY"
# =============================================================================

echo ""
if [[ $ERRORS -eq 0 && $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}  ✅ All checks passed — environment is ready!${NC}"
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}${BOLD}  ⚠️  $WARNINGS warning(s) — review above, then proceed.${NC}"
else
    echo -e "${RED}${BOLD}  ❌ $ERRORS error(s), $WARNINGS warning(s) — fix errors before proceeding.${NC}"
fi

echo ""
echo -e "${CYAN}Next step: Start Ollama → ollama serve${NC}"
echo -e "${CYAN}Then run Project 1: HuggingFace Safety Dataset${NC}"
echo ""
