#!/usr/bin/env bash
# Eval / predict on a folder of pages → runs/eval/<name>/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

SOURCE="${1:-$ROOT/data/eval/test_manga}"
MODEL="${MODEL:-$ROOT/runs/train/v1/weights/best.pt}"
DEVICE="${DEVICE:-mps}"
NAME="${NAME:-test_manga}"
PROJECT="$ROOT/runs/eval"
CONF="${CONF:-0.5}"

if [[ ! -f "$MODEL" ]]; then
  echo "Missing model: $MODEL" >&2
  exit 1
fi
if [[ ! -e "$SOURCE" ]]; then
  echo "Missing source: $SOURCE" >&2
  echo "Put pages in data/eval/test_manga/ or pass a path:" >&2
  echo "  ./scripts/eval.sh /path/to/pages" >&2
  exit 1
fi

echo "Eval: model=$MODEL source=$SOURCE conf=$CONF → $PROJECT/$NAME"
if ! yolo predict \
  model="$MODEL" \
  source="$SOURCE" \
  conf="$CONF" \
  device="$DEVICE" \
  project="$PROJECT" \
  name="$NAME" \
  exist_ok=True; then
  if [[ "$DEVICE" == "mps" ]]; then
    echo "MPS failed; retrying with device=cpu" >&2
    yolo predict \
      model="$MODEL" \
      source="$SOURCE" \
      conf="$CONF" \
      device=cpu \
      project="$PROJECT" \
      name="$NAME" \
      exist_ok=True
  else
    exit 1
  fi
fi

echo "Done. Overlays under runs/eval/$NAME/"
