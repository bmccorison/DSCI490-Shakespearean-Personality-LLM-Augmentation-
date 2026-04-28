#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/interface"
BACKEND_DIR="$SCRIPT_DIR/uv_config"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' is not installed." >&2
    exit 1
  fi
}

require_command npm
require_command uv

if [[ ! -d "$FRONTEND_DIR" ]]; then
  echo "Error: frontend directory not found at $FRONTEND_DIR" >&2
  exit 1
fi

if [[ ! -d "$BACKEND_DIR" ]]; then
  echo "Error: backend directory not found at $BACKEND_DIR" >&2
  exit 1
fi

echo "Installing frontend dependencies in interface/..."
(
  cd "$FRONTEND_DIR"
  npm install
)

echo "Installing backend dependencies in uv_config/..."
(
  cd "$BACKEND_DIR"
  uv sync
)

echo "Setup complete."
