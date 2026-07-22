#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-8091}"
DIR="$ROOT/data/labeling/to_label"

if [[ ! -d "$DIR" ]]; then
  echo "Missing $DIR — run: python scripts/select_to_label.py --copy" >&2
  exit 1
fi

if lsof -t -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  kill "$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN)" 2>/dev/null || true
  sleep 1
fi

exec python3 "$ROOT/scripts/cors_static_server.py" --dir "$DIR" --host 127.0.0.1 --port "$PORT"
