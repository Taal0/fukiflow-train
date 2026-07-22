#!/usr/bin/env python3
"""Convert selected YOLO-seg labels to Label Studio import JSON (class → bubble)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEEP_CLASSES = {1, 2, 3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--selection",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "labeling"
        / "selection.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "labeling"
        / "imports"
        / "preannotations.json",
    )
    p.add_argument(
        "--image-root",
        default="http://localhost:8091",
        help="Image URL prefix (CORS static server for data/labeling/to_label)",
    )
    return p.parse_args()


def yolo_polys_to_ls(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Return Label Studio result items for kept classes."""
    text = label_path.read_text().strip()
    if not text:
        return []

    results: list[dict] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        if cls not in KEEP_CLASSES:
            continue
        coords = [float(x) for x in parts[1:]]
        if len(coords) % 2 != 0:
            continue
        points = []
        for i in range(0, len(coords), 2):
            # YOLO normalized xy → LS percent of image
            points.append([coords[i] * 100.0, coords[i + 1] * 100.0])
        if len(points) < 3:
            continue
        results.append(
            {
                "original_width": img_w,
                "original_height": img_h,
                "image_rotation": 0,
                "value": {
                    "points": points,
                    "polygonlabels": ["bubble"],
                },
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels",
                "origin": "manual",
            }
        )
    return results


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # w, h


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    items = selection["items"]
    tasks: list[dict] = []

    missing = 0
    for row in items:
        dest_name = row.get("dest")
        if not dest_name:
            missing += 1
            continue
        local_img = args.selection.parent / "to_label" / dest_name
        if not local_img.exists():
            # fallback to source path for size
            local_img = Path(row["image"])
        w, h = image_size(local_img)
        results = yolo_polys_to_ls(Path(row["label"]), w, h)
        task = {
            "data": {
                "image": f"{args.image_root.rstrip('/')}/{dest_name}",
            },
            "meta": {
                "group": row["group"],
                "base": row["base"],
                "source_label": row["label"],
                "kept_objects": row["kept_objects"],
            },
        }
        if results:
            task["annotations"] = [
                {
                    "result": results,
                    "ground_truth": False,
                }
            ]
        tasks.append(task)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tasks, indent=2) + "\n")
    with_ann = sum(1 for t in tasks if t.get("annotations"))
    print(
        f"Wrote {args.out}: {len(tasks)} tasks, {with_ann} with polygons"
        + (f", {missing} missing dest" if missing else "")
    )


if __name__ == "__main__":
    main()
