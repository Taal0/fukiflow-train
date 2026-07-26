# fukiflow-train

Bubble + panel YOLO training / eval / datasets for FukiFlow.  
Vision runtime stays in `../fukiflow-vision` (copy `best.pt` only).

## Layout

```
datasets/bubble/v1/       # frozen bubble set (181) — keep
datasets/bubble/bubble-500/  # v1 + clean review (superset)
datasets/panel/           # panel labels; images → symlink bubble-500
datasets/clean/           # clean/inpaint pages (page_NNN.jpg + json)
runs/train/               # bubble checkpoints
runs/panel/               # panel_v1…v3 + eval overlays
runs/eval/ · runs/smoke/
labeling-wip/             # AnyLabeling export scratch
scripts/train.sh · eval.sh · smoke_predict.sh
```

Labeling UI: `../labeling/` (X-AnyLabeling + `custom-models/`).

## Setup

```bash
cd fukiflow-train
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
chmod +x scripts/*.sh
```

## Train (bubble)

```bash
./scripts/train.sh
# NAME=bubble-500 DATA=datasets/bubble/bubble-500/dataset.yaml ./scripts/train.sh
NAME=v2 MODEL=runs/train/v1/weights/best.pt DATA=datasets/bubble/v2/dataset.yaml ./scripts/train.sh
```

## Smoke / eval

```bash
./scripts/smoke_predict.sh
./scripts/eval.sh
./scripts/eval.sh /path/to/pages
```

## Panel

```bash
# example — adjust name/epochs as needed
yolo detect train \
  model=weights/yolov8n.pt \
  data=datasets/panel/data.yaml \
  project=runs/panel name=panel_v4
```

## Vision handoff

```
runs/train/v1/weights/best.pt      → fukiflow-vision/models/bubble/…
runs/panel/panel_v3/weights/best.pt → fukiflow-vision/models/panel/…
```

## Notes

- `datasets/bubble/v1` (181) is **not** the same as `datasets/clean` (422 `page_*`).
- All 181 of v1 are inside `datasets/bubble/bubble-500`.
- Delete candidates from the 2026-07 reorg live in `../_pending_delete/` until approved.
