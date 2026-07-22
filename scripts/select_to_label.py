#!/usr/bin/env python3
"""Select 200 manga + 50 manhwa pages from Roboflow YOLO-seg export into data/labeling/to_label."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

MANHWA_RE = re.compile(
    r"(bilibili|webtoon|manhwa|thunderscans|asura|flame|reaper|leviatan|"
    r"topmanhua|mangabuddy|toon)",
    re.I,
)
# Keep jagged shout boxes (1) + location (2) + speech (3); drop noisy class 0.
KEEP_CLASSES = {1, 2, 3}
MAX_ASPECT = 3.0  # exclude ultra-tall webtoon strips


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "manga-2-segmentation.v1i.yolov8",
        help="Roboflow YOLO-seg export root (default: data/raw/...; else pass --source)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "labeling"
        / "to_label",
    )
    p.add_argument(
        "--selection",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "labeling"
        / "selection.json",
    )
    p.add_argument("--manga", type=int, default=200)
    p.add_argument("--manhwa", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="Copy images into --out")
    return p.parse_args()


def image_for_label(label: Path, images_dir: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        cand = images_dir / f"{label.stem}{ext}"
        if cand.exists():
            return cand
    return None


def label_stats(label: Path) -> tuple[set[int], int]:
    text = label.read_text().strip()
    if not text:
        return set(), 0
    classes: set[int] = set()
    kept = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        classes.add(cls)
        if cls in KEEP_CLASSES:
            kept += 1
    return classes, kept


def aspect_ratio(img: Path) -> float:
    # Avoid heavy deps: read via Pillow if available, else skip filter.
    try:
        from PIL import Image

        with Image.open(img) as im:
            w, h = im.size
        if w <= 0 or h <= 0:
            return 1.0
        return max(w, h) / min(w, h)
    except Exception:
        return 1.0


def stem_base(stem: str) -> str:
    return stem.split(".rf.")[0]


def collect_candidates(source: Path) -> tuple[list[dict], list[dict]]:
    manga: list[dict] = []
    manhwa: list[dict] = []
    for split in ("train", "valid", "test"):
        labels_dir = source / split / "labels"
        images_dir = source / split / "images"
        if not labels_dir.is_dir():
            continue
        for label in labels_dir.glob("*.txt"):
            classes, kept = label_stats(label)
            if kept == 0:
                continue
            img = image_for_label(label, images_dir)
            if img is None:
                continue
            ar = aspect_ratio(img)
            if ar > MAX_ASPECT:
                continue
            base = stem_base(label.stem)
            is_manhwa = bool(MANHWA_RE.search(base))
            row = {
                "split": split,
                "stem": label.stem,
                "base": base,
                "image": str(img),
                "label": str(label),
                "classes": sorted(classes),
                "kept_objects": kept,
                "aspect": round(ar, 3),
                "group": "manhwa" if is_manhwa else "manga",
            }
            (manhwa if is_manhwa else manga).append(row)
    return manga, manhwa


def pick(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    # Prefer more kept objects, then shuffle within buckets for diversity.
    pool = sorted(pool, key=lambda r: (-r["kept_objects"], r["base"]))
    # Dedupe by base name (same page across exports / hashes).
    seen: set[str] = set()
    unique: list[dict] = []
    for row in pool:
        if row["base"] in seen:
            continue
        seen.add(row["base"])
        unique.append(row)
    if len(unique) <= n:
        return unique
    # Stratify: take top half by object count, sample rest for variety.
    head = unique[: max(n // 2, 1)]
    tail = unique[max(n // 2, 1) :]
    need = n - len(head)
    rng.shuffle(tail)
    return head + tail[:need]


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Source dataset not found: {args.source}")

    rng = random.Random(args.seed)
    manga_pool, manhwa_pool = collect_candidates(args.source)
    selected_manga = pick(manga_pool, args.manga, rng)
    selected_manhwa = pick(manhwa_pool, args.manhwa, rng)
    selected = selected_manga + selected_manhwa

    if len(selected_manga) < args.manga:
        print(f"WARN: only {len(selected_manga)}/{args.manga} manga candidates")
    if len(selected_manhwa) < args.manhwa:
        print(f"WARN: only {len(selected_manhwa)}/{args.manhwa} manhwa candidates")

    payload = {
        "source": str(args.source),
        "seed": args.seed,
        "keep_classes": sorted(KEEP_CLASSES),
        "max_aspect": MAX_ASPECT,
        "counts": {
            "manga": len(selected_manga),
            "manhwa": len(selected_manhwa),
            "total": len(selected),
            "manga_pool": len(manga_pool),
            "manhwa_pool": len(manhwa_pool),
        },
        "items": selected,
    }

    args.selection.parent.mkdir(parents=True, exist_ok=True)
    args.selection.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.selection} ({len(selected)} items)")

    if args.copy:
        if args.out.exists():
            shutil.rmtree(args.out)
        args.out.mkdir(parents=True, exist_ok=True)
        for row in selected:
            src = Path(row["image"])
            # Stable short name: group_base + original extension
            safe = re.sub(r"[^\w.\-]+", "_", row["base"])[:120]
            dest = args.out / f"{row['group']}__{safe}{src.suffix.lower()}"
            shutil.copy2(src, dest)
            row["dest"] = dest.name
        args.selection.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Copied {len(selected)} images → {args.out}")


if __name__ == "__main__":
    main()
