#!/usr/bin/env bash
# Video2Docs Web UI starter (Unix/macOS)
# - Activates ./venv if present
# - Starts the Flask web app
set -euo pipefail

# Move to repo root
cd "$(dirname "$0")"

if [ -f ".env" ]; then
  echo "Using .env in $(pwd)"
else
  echo "No .env found. You can copy .env.example to .env and adjust settings."
fi

if [ -f "venv/bin/activate" ]; then
  echo "Activating virtual environment..."
  # shellcheck disable=SC1091
  source "venv/bin/activate"
else
  echo "No virtual environment found at venv/bin/activate. Continuing with system Python."
fi

# Start the web application
exec python -m src.webapp "$@"
