#!/usr/bin/env python3
"""Select Japanese manga pages via Tesseract OCR and move them into clean_dataset."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Paths after 2026-07 reorg (raw Roboflow may live under ../_pending_delete/raw-roboflow/)
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
OUT = ROOT / "datasets" / "clean"
RAW = BASE / "_pending_delete" / "raw-roboflow"
SMALL = RAW / "manga-speech-bubble-detection.v3i.yolov8"
LARGE = RAW / "manga-2-segmentation.v1i.yolov8"
LOG = OUT / "clean_dataset_scan_log.jsonl"
STATE = OUT / "clean_dataset_state.json"
TARGET = 320  # was 500; resume later if needed
WORKERS = 8
BATCH = 24

# Known non-Japanese sources / English scanlations
CN_NAME = (
    "bilibili",
    "topmanhua",
    "manhua",
    "webtoon",
    "thunderscans",
    "mangabuddy",
    "manhuaren",
)
EN_NAME = ("en-", "-en-", "_en_")

CN_MARKERS = set("你们您这那吗吧呢啊啦啥咱喽嘞喔哟嘿嘛")
CN_MARKERS |= set("們您這那嗎")
CN_MARKERS |= set("脸现样还没从对让给开关问东车见买听读写")
CN_MARKERS |= set("臉現樣還沒從對讓給開關問東車見買聽讀寫")

HIRA = re.compile(r"[\u3041-\u3096]+")
KATA = re.compile(r"[\u30A1-\u30FAー]+")
HANGUL = re.compile(r"[\uAC00-\uD7AF]")
LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for split in ("train", "valid", "test"):
        d = root / split / "images"
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT and p.is_file():
                files.append(p)
    return files


def ocr(path: str, langs: str, psm: str = "11") -> str:
    r = subprocess.run(
        ["tesseract", path, "stdout", "-l", langs, "--psm", psm],
        capture_output=True,
        text=True,
        timeout=180,
    )
    return r.stdout or ""


def classify_one(path_str: str) -> dict:
    path = Path(path_str)
    name = path.name.lower()

    if any(x in name for x in CN_NAME) or name.startswith("en-") or any(x in name for x in EN_NAME):
        return {
            "path": path_str,
            "is_jp": False,
            "reason": "filename_non_jp",
            "hira": 0,
            "kata": 0,
            "cn_hits": 0,
            "latin": 0,
        }

    try:
        jp_text = ocr(path_str, "jpn+jpn_vert", "11")
        cn_text = ocr(path_str, "chi_sim", "11")
        eng_text = ocr(path_str, "eng", "11")
    except Exception as e:  # noqa: BLE001
        return {
            "path": path_str,
            "is_jp": False,
            "reason": f"ocr_error:{e}",
            "hira": 0,
            "kata": 0,
            "cn_hits": 0,
            "latin": 0,
        }

    hira = sum(len(r) for r in HIRA.findall(jp_text))
    kata = sum(len(r) for r in KATA.findall(jp_text))
    kana = hira + kata
    hira_runs_ge2 = sum(1 for r in HIRA.findall(jp_text) if len(r) >= 2)
    hangul = len(HANGUL.findall(jp_text + cn_text))
    cn_hits = sum(1 for ch in cn_text if ch in CN_MARKERS)
    latin = len(LATIN_WORD.findall(eng_text))

    # Reject non-Japanese languages first
    if hangul >= 3:
        reason, is_jp = "hangul", False
    elif cn_hits >= 2:
        reason, is_jp = "cn_markers", False
    elif latin >= 10 and latin >= hira:
        # English-heavy page (scanlation / EN source). Hiragana must dominate.
        reason, is_jp = "english", False
    # Japanese manga dialogue almost always has hiragana; require it strongly.
    elif hira >= 12 and hira_runs_ge2 >= 4 and kana >= 18 and cn_hits <= 1 and latin < hira * 2:
        reason, is_jp = "jp", True
    else:
        reason, is_jp = "weak", False

    return {
        "path": path_str,
        "is_jp": is_jp,
        "reason": reason,
        "hira": hira,
        "kata": kata,
        "cn_hits": cn_hits,
        "latin": latin,
        "hira_runs": hira_runs_ge2,
    }


def unique_dest_name(src: Path, dest_dir: Path) -> Path:
    candidate = dest_dir / src.name
    if not candidate.exists():
        return candidate
    stem, suf = src.stem, src.suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}__{i}{suf}"
        if not candidate.exists():
            return candidate
        i += 1


def out_count() -> int:
    return sum(1 for p in OUT.iterdir() if p.is_file())


def load_log_rows() -> list[dict]:
    if not LOG.exists():
        return []
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return rows


def restore_false_positives() -> int:
    """Move known-bad files in clean_dataset back to original paths from log."""
    rows = load_log_rows()
    by_name: dict[str, str] = {}
    for r in rows:
        if not r.get("is_jp"):
            continue
        src = Path(r["path"])
        by_name[src.name] = str(src)

    restored = 0
    for p in list(OUT.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        bad = (
            name.startswith("en-")
            or any(x in name for x in CN_NAME)
            or any(x in name for x in EN_NAME)
        )
        if not bad:
            continue
        orig = by_name.get(p.name)
        if orig:
            dest = Path(orig)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest = dest.with_name(dest.stem + "__restored" + dest.suffix)
            shutil.move(str(p), str(dest))
        else:
            # Fallback: put into small train/images if origin unknown
            fallback = SMALL / "train" / "images" / p.name
            fallback.parent.mkdir(parents=True, exist_ok=True)
            if fallback.exists():
                fallback = fallback.with_name(fallback.stem + "__restored" + fallback.suffix)
            shutil.move(str(p), str(fallback))
        restored += 1
        print(f"  restored non-JP: {p.name}", flush=True)
    return restored


def revalidate_existing(log_fh) -> int:
    """Re-OCR files already in clean_dataset; restore rejects."""
    files = [p for p in OUT.iterdir() if p.is_file()]
    if not files:
        return 0

    rows = load_log_rows()
    by_name = {Path(r["path"]).name: r["path"] for r in rows if r.get("is_jp")}
    removed = 0

    print(f"Revalidating {len(files)} existing clean_dataset files...", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(classify_one, str(p)): p for p in files}
        for fut in as_completed(futs):
            p = futs[fut]
            res = fut.result()
            res["revalidate"] = True
            res["clean_path"] = str(p)
            log_fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            log_fh.flush()
            if res["is_jp"]:
                continue
            # restore
            orig = by_name.get(p.name)
            if orig:
                dest = Path(orig)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    dest = dest.with_name(dest.stem + "__restored" + dest.suffix)
                shutil.move(str(p), str(dest))
            else:
                dest = SMALL / "train" / "images" / p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    dest = dest.with_name(dest.stem + "__restored" + dest.suffix)
                shutil.move(str(p), str(dest))
            removed += 1
            print(f"  revalidate REJECT ({res['reason']}): {p.name}", flush=True)
    return removed


def load_done_paths() -> set[str]:
    done: set[str] = set()
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            # Skip revalidate rows for "done" of source paths
            if row.get("revalidate"):
                continue
            done.add(row["path"])
    return done


def scan_and_move(images: list[Path], log_fh) -> tuple[int, dict, int]:
    moved = 0
    scanned = 0
    stats: dict[str, int] = {}
    done = load_done_paths()
    # Also skip files that no longer exist (already moved)
    pending = [p for p in images if str(p) not in done and p.exists()]
    print(f"Pending scan: {len(pending)} (skip done/missing {len(images) - len(pending)})", flush=True)

    i = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        while i < len(pending):
            if out_count() >= TARGET:
                break
            batch = pending[i : i + BATCH]
            i += BATCH
            futs = [ex.submit(classify_one, str(p)) for p in batch]
            for fut in as_completed(futs):
                scanned += 1
                res = fut.result()
                log_fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                log_fh.flush()

                reason = res.get("reason", "other")
                if str(reason).startswith("ocr_error"):
                    reason = "ocr_error"
                stats[reason] = stats.get(reason, 0) + 1

                if res["is_jp"] and out_count() < TARGET:
                    src = Path(res["path"])
                    if not src.exists():
                        continue
                    dst = unique_dest_name(src, OUT)
                    shutil.move(str(src), str(dst))
                    moved += 1
                    print(
                        f"  MOVE [{out_count()}/{TARGET}] hira={res['hira']} latin={res['latin']} {src.name}",
                        flush=True,
                    )

                if scanned % 20 == 0:
                    print(
                        f"  scanned={scanned}/{len(pending)} moved_total={out_count()} stats={stats}",
                        flush=True,
                    )

    return moved, stats, scanned


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"OUT={OUT} existing={out_count()} target={TARGET}", flush=True)

    t0 = time.time()
    with LOG.open("a", encoding="utf-8") as log_fh:
        print("=== CLEANUP: restore filename-based non-JP ===", flush=True)
        n_rest = restore_false_positives()
        print(f"Restored by filename: {n_rest}; remaining={out_count()}", flush=True)

        print("=== CLEANUP: revalidate remaining with stricter OCR ===", flush=True)
        n_rej = revalidate_existing(log_fh)
        print(f"Revalidate rejected: {n_rej}; remaining={out_count()}", flush=True)

        if out_count() >= TARGET:
            print("Already have enough after cleanup.", flush=True)
        else:
            small_imgs = list_images(SMALL)
            large_imgs = list_images(LARGE)
            print(f"Small pool={len(small_imgs)} Large pool={len(large_imgs)}", flush=True)

            print("=== PHASE 1: small dataset ===", flush=True)
            moved1, stats1, n1 = scan_and_move(small_imgs, log_fh)
            print(
                f"Phase1 done: moved={moved1} total={out_count()} scanned={n1} stats={stats1}",
                flush=True,
            )

            if out_count() < TARGET:
                print("=== PHASE 2: large dataset ===", flush=True)
                moved2, stats2, n2 = scan_and_move(large_imgs, log_fh)
                print(
                    f"Phase2 done: moved={moved2} total={out_count()} scanned={n2} stats={stats2}",
                    flush=True,
                )

    state = {
        "total_moved": out_count(),
        "target": TARGET,
        "elapsed_sec": round(time.time() - t0, 1),
        "out": str(OUT),
    }
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("DONE", state, flush=True)


if __name__ == "__main__":
    main()
