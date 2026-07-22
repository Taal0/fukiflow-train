#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Free port 8080 if something is already listening
if lsof -t -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1; then
  kill "$(lsof -t -iTCP:8080 -sTCP:LISTEN)" 2>/dev/null || true
  sleep 1
fi

# shellcheck disable=SC1091
source "$ROOT/.venv-ls/bin/activate"

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$ROOT/data"
export LABEL_STUDIO_BASE_DATA_DIR="$ROOT/data/label-studio-data"
mkdir -p "$LABEL_STUDIO_BASE_DATA_DIR"

echo "Label Studio → http://localhost:8080"
echo "Local files root: $LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"
echo "Import: data/labeling/imports/preannotations.json"
echo "Config: label-studio/config.xml"
echo "Log: data/label-studio-data/server.log"

exec label-studio start --host 127.0.0.1 --port 8080
