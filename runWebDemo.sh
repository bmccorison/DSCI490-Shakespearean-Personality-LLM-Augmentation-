#!/usr/bin/env bash
# A script to run both the backend and frontend for the web demo, with graceful shutdown on Ctrl+C.

set -euo pipefail

# Declare the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Override if needed, e.g.:
# BACKEND_CMD="uvicorn app:app --host 0.0.0.0 --port 8000" ./runWebDemo.sh
# FRONTEND_CMD="npm run dev -- --host 0.0.0.0 --port 5173" ./runWebDemo.sh
if [[ -n "${BACKEND_CMD:-}" ]]; then
  BACKEND_CMD="${BACKEND_CMD}"
elif [[ -x "$ROOT_DIR/.venv/bin/python3" ]]; then
  BACKEND_CMD="$ROOT_DIR/.venv/bin/python3 app.py"
else
  BACKEND_CMD="python3 app.py"
fi
FRONTEND_CMD="${FRONTEND_CMD:-npm run dev}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
BACKEND_READY_ATTEMPTS="${BACKEND_READY_ATTEMPTS:-20}"
BACKEND_READY_DELAY_SECS="${BACKEND_READY_DELAY_SECS:-1}"

# Set placeholders pids for clean shutdown
backend_pid=""
frontend_pid=""

# Function to gracefully shutdown a service given its PID and name
shutdown_service() {
  local pid="$1"
  local name="$2"

  if [[ -z "$pid" ]]; then
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping ${name}..."
    kill -TERM "$pid" 2>/dev/null || true
    pkill -TERM -P "$pid" 2>/dev/null || true
  fi

  wait "$pid" 2>/dev/null || true
}

wait_for_backend() {
  local attempt

  for ((attempt=1; attempt<=BACKEND_READY_ATTEMPTS; attempt++)); do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
      return 1
    fi

    if BACKEND_HOST="$BACKEND_HOST" BACKEND_PORT="$BACKEND_PORT" python3 - <<'PY'
import os
import socket
import sys

host = os.environ["BACKEND_HOST"]
port = int(os.environ["BACKEND_PORT"])

sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()

sys.exit(0)
PY
    then
      return 0
    fi

    sleep "$BACKEND_READY_DELAY_SECS"
  done

  return 1
}

# Cleanup function to stop both services on exit
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  shutdown_service "$frontend_pid" "frontend"
  shutdown_service "$backend_pid" "backend"

  exit "$exit_code"
}

trap cleanup EXIT INT TERM

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but was not found."
  exit 1
fi

if [[ ! -f "$ROOT_DIR/interface/package.json" ]]; then
  echo "Could not find frontend project at $ROOT_DIR/interface."
  exit 1
fi

# Find the backend and frontend paths, then run startup commands
echo "Starting backend with: $BACKEND_CMD"
(
  cd "$ROOT_DIR"
  bash -lc "$BACKEND_CMD"
) &
backend_pid=$!

echo "Waiting for backend on ${BACKEND_HOST}:${BACKEND_PORT}..."
if wait_for_backend; then
  echo "Backend is accepting connections."
else
  echo "Backend was not ready in time; starting frontend anyway."
fi

echo "Starting frontend with: $FRONTEND_CMD"
(
  cd "$ROOT_DIR/interface"
  bash -lc "$FRONTEND_CMD"
) &
frontend_pid=$!

echo "Backend PID: $backend_pid"
echo "Frontend PID: $frontend_pid"
echo "Press Ctrl+C to stop both."

if wait -n "$backend_pid" "$frontend_pid"; then
  echo "One service exited cleanly. Shutting down the other..."
else
  echo "One service exited with an error. Shutting down the other..."
fi
