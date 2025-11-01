#!/usr/bin/env bash
# ============================================================
# Wisdom of Sheep Python environment bootstrapper
# ------------------------------------------------------------
# Creates (or reuses) a virtual environment and installs the
# locked dependency set that predates the FastAPI middleware
# regression. This restores the stable runtime used before the
# broken middleware refactor.
#
# Usage:
#   ./setup_python_env.sh [VENV_PATH]
#
# Environment variables:
#   PYTHON - Python interpreter to use (default: python3.11)
# ============================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3.11}"
VENV_DIR="${1:-${ROOT_DIR}/.venv}"
REQ_FILE="${ROOT_DIR}/requirements.lock"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[setup] Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[setup] Missing dependency lock file: ${REQ_FILE}" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] Creating virtual environment at ${VENV_DIR}" >&2
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "[setup] Reusing existing virtual environment at ${VENV_DIR}" >&2
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --requirement "${REQ_FILE}"

echo "[setup] Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
