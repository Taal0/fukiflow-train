#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

SOURCE="${1:-samples/page_smoke.png}"
DEVICE="${DEVICE:-mps}"
MODEL="${MODEL:-$ROOT/runs/train/v1/weights/best.pt}"
PROJECT="$ROOT/runs/smoke"
NAME="${NAME:-v1}"

if [[ ! -f "$MODEL" ]]; then
  echo "Missing model: $MODEL" >&2
  echo "Train first, or set MODEL=/path/to/weights.pt" >&2
  exit 1
fi

if [[ ! -f "$SOURCE" ]]; then
  echo "Missing source image: $SOURCE" >&2
  exit 1
fi

echo "Smoke predict: model=$MODEL source=$SOURCE device=$DEVICE project=$PROJECT name=$NAME"
if ! yolo predict model="$MODEL" source="$SOURCE" device="$DEVICE" project="$PROJECT" name="$NAME" exist_ok=True; then
  if [[ "$DEVICE" == "mps" ]]; then
    echo "MPS failed; retrying with device=cpu" >&2
    yolo predict model="$MODEL" source="$SOURCE" device=cpu project="$PROJECT" name="$NAME" exist_ok=True
  else
    exit 1
  fi
fi

echo "Done. Overlay under runs/smoke/$NAME/"
