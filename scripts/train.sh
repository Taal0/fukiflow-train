#!/usr/bin/env bash
# Train bubble-seg from this repo (outputs → runs/train/<name>/).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

NAME="${NAME:-v1}"
# Default: datasets/bubble/<name>/dataset.yaml (v1, bubble-500, …)
DATA="${DATA:-$ROOT/datasets/bubble/$NAME/dataset.yaml}"
# v1 from COCO-seg pretrained; later: MODEL=runs/train/v1/weights/best.pt
MODEL="${MODEL:-$ROOT/weights/yolov8n-seg.pt}"
DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
PROJECT="$ROOT/runs/train"

if [[ ! -f "$DATA" ]]; then
  echo "Missing dataset yaml: $DATA" >&2
  echo "Expected datasets/bubble/<name>/dataset.yaml (e.g. datasets/bubble/v1)." >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Missing model: $MODEL" >&2
  exit 1
fi

echo "Train: name=$NAME data=$DATA model=$MODEL device=$DEVICE epochs=$EPOCHS → $PROJECT/$NAME"
if ! yolo segment train \
  model="$MODEL" \
  data="$DATA" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  device="$DEVICE" \
  project="$PROJECT" \
  name="$NAME"; then
  if [[ "$DEVICE" == "mps" ]]; then
    echo "MPS failed; retrying with device=cpu" >&2
    yolo segment train \
      model="$MODEL" \
      data="$DATA" \
      epochs="$EPOCHS" \
      imgsz="$IMGSZ" \
      device=cpu \
      project="$PROJECT" \
      name="$NAME"
  else
    exit 1
  fi
fi

echo "Done. Weights: runs/train/$NAME/weights/best.pt"
