#!/usr/bin/env bash
# Run the test suite via the project's uv environment.
# Any extra arguments are forwarded to pytest (e.g. -k, -v, --tb=short).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec uv run --project uv_config pytest tests/ "$@"
