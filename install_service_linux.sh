#!/usr/bin/env bash
# Install Video2Docs as a systemd service on Linux
#
# This script creates and enables a systemd unit that runs the web UI
# as a simple service. It prefers the project's ./venv Python if present.
#
# Usage:
#   sudo ./install_service_linux.sh [--user USER] [--name SERVICE_NAME]
#
# Notes:
# - The service will run in this repository directory.
# - The app loads environment variables from .env automatically if present.
# - Logs can be viewed with: journalctl -u <service-name> -f
set -euo pipefail

SERVICE_NAME="video2docs"
RUN_AS_USER="${SUDO_USER:-${USER}}"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      RUN_AS_USER="$2"; shift 2 ;;
    --name)
      SERVICE_NAME="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Resolve repository root (directory where this script resides)
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Ensure run_web.sh is executable if present (it activates venv and starts the app)
if [[ -f "$REPO_DIR/run_web.sh" ]]; then
  chmod +x "$REPO_DIR/run_web.sh"
fi

# Prefer venv Python if available
if [[ -x "$REPO_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_DIR/venv/bin/python"
else
  # Fallback to system python
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Python not found. Please install Python 3." >&2
    exit 1
  fi
fi

UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

echo "Installing systemd service '${SERVICE_NAME}'..."

echo "- Working directory: $REPO_DIR"
echo "- Python: $PYTHON_BIN"
echo "- Run as user: $RUN_AS_USER"

# Create unit file content
read -r -d '' UNIT_CONTENT <<EOF
[Unit]
Description=Video2Docs Web Service
After=network.target

[Service]
Type=simple
WorkingDirectory=${REPO_DIR}
# Load environment variables from .env if present (optional)
EnvironmentFile=-${REPO_DIR}/.env
# Use the repo's launcher which activates the virtual environment if present
ExecStart=${REPO_DIR}/run_web.sh
Restart=on-failure
RestartSec=5
User=${RUN_AS_USER}
# Ensure we have a sane PATH (include venv bin dir if present)
Environment=PATH=${REPO_DIR}/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin

[Install]
WantedBy=multi-user.target
EOF

# Write the unit (requires root)
if [[ $EUID -ne 0 ]]; then
  echo "This script must be run with sudo/root to write ${UNIT_PATH}." >&2
  exit 1
fi

echo "$UNIT_CONTENT" > "$UNIT_PATH"
chmod 644 "$UNIT_PATH"

# Reload and enable
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"
systemctl restart "${SERVICE_NAME}.service"

echo "Service '${SERVICE_NAME}' installed and started."
echo "View logs with: sudo journalctl -u ${SERVICE_NAME} -f"