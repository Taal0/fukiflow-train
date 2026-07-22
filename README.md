# fukiflow-bubble-yolo

Speech-bubble **YOLO-seg** training and evaluation for FukiFlow.  
Self-contained: train / eval / smoke only need this repo (+ `pip install -r requirements.txt`).

## Layout

```
datasets/v1/          # frozen train set (images+labels+dataset.yaml)
datasets/v2/          # later (dialog + sfx) — do not overwrite v1
runs/train/v1/        # checkpoints, curves, results.csv
runs/eval/            # page overlays
runs/smoke/           # quick predict
weights/              # COCO pretrained yolov8n-seg.pt
data/labeling/        # Label Studio pool (to_label, imports, exports)
data/eval/            # put holdout pages here (e.g. test_manga/)
data/raw/             # optional Roboflow export archive
scripts/              # train.sh · eval.sh · smoke_predict.sh · LS helpers
```

## Setup

```bash
cd fukiflow-bubble-yolo
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
chmod +x scripts/*.sh
```

## Train

```bash
./scripts/train.sh                    # → runs/train/v1/ (NAME=v1 default)
NAME=v2 MODEL=runs/train/v1/weights/best.pt DATA=datasets/v2/dataset.yaml ./scripts/train.sh
```

Always run from this repo so outputs stay under `runs/train/`.

## Smoke / eval

```bash
./scripts/smoke_predict.sh
./scripts/smoke_predict.sh path/to/page.png

# Holdout folder (default: data/eval/test_manga)
mkdir -p data/eval/test_manga   # copy pages in
./scripts/eval.sh
./scripts/eval.sh /path/to/pages
```

Overlays: `runs/smoke/v1/`, `runs/eval/test_manga/`.  
Override: `MODEL=...` `NAME=...` `CONF=0.5` `DEVICE=cpu`.

## Label Studio

```bash
source .venv/bin/activate
# Roboflow export: either data/raw/... or --source ~/Downloads/YOLO/...
python scripts/select_to_label.py --copy --source ~/Downloads/YOLO/manga-2-segmentation.v1i.yolov8
python scripts/yolo_to_labelstudio.py --image-root http://127.0.0.1:8091
./scripts/serve_to_label.sh    # → :8091
./scripts/start_label_studio.sh  # → :8080  (needs .venv-ls)
```

Import: `data/labeling/imports/preannotations.json` · config: `label-studio/config.xml`.

Docker: `image-root /data/labeling/to_label` with compose volume `../data:/data`.

## Dataset v1

| Split | Images |
|-------|--------|
| train | 145 |
| val   | 36 |
| **total** | **181** |

See `datasets/v1/manifest.json`. Class: `bubble` (sfx → v2).

## Vision handoff

Copy inference weight only (not the full train folder):

`runs/train/v1/weights/best.pt` → `fukiflow-vision/models/bubble/yolov8n-seg-bubble-v1.pt`

## Next

1. Eval / inset-fill smoke before Vision wiring  
2. SFX class → `datasets/v2` + `NAME=v2` train (keep v1)
