# fukiflow-bubble-yolo

Speech-bubble **YOLO-seg** training and evaluation for FukiFlow.

## Purpose

- Train / eval a segmentation model for manga speech bubbles
- Measure whether polygons are reliable enough for a later hybrid pipeline

## Not this repo

- No `fukiflow-vision` integration
- No API, OCR, or inpaint
- Hybrid (balloon fill + CTD fallback) is decided **after** the eval gate passes

## Setup

Requires Python 3.11 (Apple Silicon / MPS supported).

```bash
cd fukiflow-bubble-yolo
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Smoke (pretrained, untrained)

Confirms the Ultralytics pipeline works on a manga page. COCO pretrained weights will **not** detect bubbles well — that is expected.

```bash
./scripts/smoke_predict.sh
# or:
yolo predict model=yolov8n-seg.pt source=samples/page_smoke.png device=mps project=runs/smoke name=pretrained
```

If MPS fails, use `device=cpu`. Overlay lands in `runs/smoke/pretrained/`.

## Next steps

1. Dataset (Roboflow + local pages) → `datasets/bubble-seg/`
2. Label completion + class schema
3. Fine-tune on M4 (`device=mps`) → `best.pt`
4. Eval gate on `test manga` before any Vision hybrid talk
