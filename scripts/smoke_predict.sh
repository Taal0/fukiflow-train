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
PROJECT="$ROOT/runs/smoke"

if [[ ! -f "$SOURCE" ]]; then
  echo "Missing source image: $SOURCE" >&2
  exit 1
fi

echo "Smoke predict: model=yolov8n-seg.pt source=$SOURCE device=$DEVICE project=$PROJECT"
if ! yolo predict model=yolov8n-seg.pt source="$SOURCE" device="$DEVICE" project="$PROJECT" name=pretrained exist_ok=True; then
  if [[ "$DEVICE" == "mps" ]]; then
    echo "MPS failed; retrying with device=cpu" >&2
    yolo predict model=yolov8n-seg.pt source="$SOURCE" device=cpu project="$PROJECT" name=pretrained exist_ok=True
  else
    exit 1
  fi
fi

echo "Done. Overlay under runs/smoke/pretrained/"
