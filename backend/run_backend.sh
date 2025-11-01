#!/bin/bash
# ============================================================
# Wisdom of Sheep Backend Launcher
# ------------------------------------------------------------
# Sets Oracle credentials and launches the FastAPI backend
# ============================================================

# --- Oracle credentials ---
export WOS_ORACLE_USER="carlhudson83"
export WOS_ORACLE_PASS="Briandavidson68-"

# --- Optional: show what’s being set (for debug only) ---
echo "Starting WOS backend..."
echo "Oracle user: $WOS_ORACLE_USER"
echo "Backend port: 8000"
echo

# --- Launch backend ---
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
