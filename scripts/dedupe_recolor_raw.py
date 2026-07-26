#!/usr/bin/env python3
"""Thorough near-duplicate cull + color/grayscale re-split for datasets/raw.

Roboflow-style datasets often ship the same page with tiny augmentations
(brightness, flip, mild crop/zoom). Exact SHA alone misses those.

Pipeline:
  1) Exact SHA-256 duplicates
  2) Multi-hash near-dup clustering (aHash / dHash / pHash), incl. hflip
  3) Pixel verify (downscale MSE + histogram corr), keep best per cluster
  4) Strict color vs grayscale classification (LAB chroma + HSV sat)
  5) Rename survivors to manga_page_NNNN / colored_page_NNNN

Usage:
  python scripts/dedupe_recolor_raw.py            # scan only → report
  python scripts/dedupe_recolor_raw.py --apply    # cull + recolor + rename
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
ANALYSIS = RAW / "_analysis"
REJECTED = RAW / "_rejected_near_dup"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Near-dup thresholds (tuned for Roboflow light augments + color/gray twins)
HAMMING_AHASH = 10
HAMMING_DHASH = 12
HAMMING_PHASH = 14
VERIFY_MSE_MAX = 22.0  # on 0–255, 128² gray (after optional hflip)
VERIFY_HIST_MIN = 0.90
# Structural similarity — catches brightness / color↔gray twins that fail raw MSE
VERIFY_CORR_MIN = 0.985
VERIFY_CORR_RELAXED = 0.970
VERIFY_MSE_NORM_MAX = 0.045  # after z-score normalize
HASH_SIZE = 8


@dataclass
class ImageRecord:
    path: str
    name: str
    sha256: str
    nbytes: int
    width: int
    height: int
    ahash: str
    dhash: str
    phash: str
    ahash_flip: str
    dhash_flip: str
    phash_flip: str
    ahash_center: str
    dhash_center: str
    phash_center: str
    color_score: float
    is_color: bool
    color_reason: str


@dataclass
class Cluster:
    id: int
    members: list[str] = field(default_factory=list)
    keep: str | None = None
    reason: str = ""
    scores: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMG_EXT
        and not p.name.startswith(".__")
        and not p.name.startswith(".")
    )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def bits_to_hex(bits: np.ndarray) -> str:
    bits = bits.astype(np.uint8).reshape(-1)
    # pad to multiple of 8
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    packed = np.packbits(bits)
    return packed.tobytes().hex()


def hex_to_bits(h: str) -> np.ndarray:
    raw = np.frombuffer(bytes.fromhex(h), dtype=np.uint8)
    return np.unpackbits(raw)


def hamming(a: str, b: str) -> int:
    return int(np.count_nonzero(hex_to_bits(a) != hex_to_bits(b)))


def ahash(gray: np.ndarray, size: int = HASH_SIZE) -> str:
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return bits_to_hex(small > small.mean())


def dhash(gray: np.ndarray, size: int = HASH_SIZE) -> str:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return bits_to_hex(diff)


def phash(gray: np.ndarray, size: int = 32, hash_size: int = HASH_SIZE) -> str:
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:hash_size, :hash_size].copy()
    # exclude DC for median (common practice)
    med = np.median(low[1:, 1:]) if hash_size > 1 else np.median(low)
    return bits_to_hex(low > med)


def load_bgr(path: Path, max_side: int = 1600) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    m = max(h, w)
    if m > max_side:
        scale = max_side / m
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# color classification (strict)
# ---------------------------------------------------------------------------


def classify_color(bgr: np.ndarray) -> tuple[bool, float, str]:
    """Return (is_color, score 0–1, reason).

    Multi-signal: LAB chroma, HSV sat, and per-channel RGB spread on midtones.
    Ignores paper/ink extremes and requires spatially distributed color (not a
    single sticker/JPEG blotch).
    """
    h, w = bgr.shape[:2]
    sample = bgr
    if max(h, w) > 720:
        scale = 720 / max(h, w)
        sample = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).astype(np.float32)
    rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB).astype(np.float32)

    L = lab[:, :, 0] * (100.0 / 255.0)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    sat = hsv[:, :, 1]
    spread = rgb.max(axis=2) - rgb.min(axis=2)

    mid = (L > 10.0) & (L < 94.0)
    if int(mid.sum()) < 80:
        mid = np.ones_like(L, dtype=bool)

    chroma_mid = chroma[mid]
    sat_mid = sat[mid]
    spread_mid = spread[mid]

    mean_chroma = float(chroma_mid.mean())
    p90_chroma = float(np.percentile(chroma_mid, 90))
    p95_chroma = float(np.percentile(chroma_mid, 95))
    p99_chroma = float(np.percentile(chroma_mid, 99))
    frac_chroma = float((chroma_mid > 7.0).mean())
    frac_chroma_strong = float((chroma_mid > 12.0).mean())
    frac_chroma_vstrong = float((chroma_mid > 20.0).mean())
    mean_sat = float(sat_mid.mean())
    frac_sat = float((sat_mid > 16.0).mean())
    frac_sat_strong = float((sat_mid > 35.0).mean())
    mean_spread = float(spread_mid.mean())
    frac_spread = float((spread_mid > 14.0).mean())

    # spatial coverage: color should appear in multiple tiles, not one corner blotch
    tile = 8
    th, tw = sample.shape[0] // tile, sample.shape[1] // tile
    color_tiles = 0
    total_tiles = 0
    for ty in range(tile):
        for tx in range(tile):
            patch = chroma[ty * th : (ty + 1) * th, tx * tw : (tx + 1) * tw]
            if patch.size == 0:
                continue
            total_tiles += 1
            if float((patch > 10.0).mean()) > 0.08:
                color_tiles += 1
    tile_frac = color_tiles / max(total_tiles, 1)

    score = (
        0.28 * min(frac_chroma_strong / 0.10, 1.0)
        + 0.18 * min(frac_sat_strong / 0.08, 1.0)
        + 0.16 * min(p95_chroma / 20.0, 1.0)
        + 0.12 * min(mean_chroma / 9.0, 1.0)
        + 0.10 * min(mean_sat / 26.0, 1.0)
        + 0.08 * min(frac_spread / 0.20, 1.0)
        + 0.08 * min(tile_frac / 0.25, 1.0)
    )

    # strong: clearly printed color / manhwa
    strong = (
        (frac_chroma_strong >= 0.05 and tile_frac >= 0.12)
        or (frac_sat_strong >= 0.04 and tile_frac >= 0.12)
        or (p95_chroma >= 16.0 and frac_chroma >= 0.10 and tile_frac >= 0.15)
        or (frac_chroma_vstrong >= 0.03 and mean_chroma >= 6.0)
        or (mean_chroma >= 8.5 and frac_sat >= 0.12 and tile_frac >= 0.18)
    )
    # weak: light tint / limited palette pages
    weak = (
        score >= 0.40
        and frac_chroma >= 0.08
        and mean_chroma >= 5.0
        and tile_frac >= 0.10
        and p99_chroma >= 12.0
    )
    # reject JPEG noise on B/W: tiny chroma spikes, no spatial coverage
    jpeg_noise = tile_frac < 0.08 and frac_chroma_strong < 0.03 and p99_chroma < 25.0

    is_color = bool((strong or weak) and not jpeg_noise)
    reason = (
        f"score={score:.3f} meanC={mean_chroma:.2f} p90C={p90_chroma:.2f} "
        f"p95C={p95_chroma:.2f} p99C={p99_chroma:.2f} "
        f"fC={frac_chroma:.3f} fCs={frac_chroma_strong:.3f} fCv={frac_chroma_vstrong:.3f} "
        f"meanS={mean_sat:.2f} fS={frac_sat:.3f} fSs={frac_sat_strong:.3f} "
        f"meanSp={mean_spread:.2f} fSp={frac_spread:.3f} tiles={tile_frac:.3f} "
        f"strong={strong} weak={weak} jpeg_noise={jpeg_noise}"
    )
    return is_color, float(score), reason


# ---------------------------------------------------------------------------
# near-dup verify
# ---------------------------------------------------------------------------


def gray128(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)


def hist_corr(bgr_a: np.ndarray, bgr_b: np.ndarray) -> float:
    ha = cv2.calcHist([bgr_a], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    hb = cv2.calcHist([bgr_b], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def _z(x: np.ndarray) -> np.ndarray:
    return (x - float(x.mean())) / (float(x.std()) + 1e-6)


def verify_pair(bgr_a: np.ndarray, bgr_b: np.ndarray) -> tuple[bool, dict]:
    ga = gray128(bgr_a)
    gb = gray128(bgr_b)
    candidates = [("direct", gb), ("hflip", np.fliplr(gb))]

    best = None
    for label, gbx in candidates:
        mse = float(np.mean((ga - gbx) ** 2))
        mse_z = float(np.mean((_z(ga) - _z(gbx)) ** 2))
        corr = float(np.corrcoef(ga.ravel(), gbx.ravel())[0, 1])
        if best is None or corr > best["corr"] or (corr == best["corr"] and mse < best["mse"]):
            best = {"align": label, "mse": mse, "mse_z": mse_z, "corr": corr}

    assert best is not None
    flipped = best["align"] == "hflip"
    bgr_b_aln = cv2.flip(bgr_b, 1) if flipped else bgr_b
    hc = hist_corr(bgr_a, bgr_b_aln)

    # absolute near-identical
    abs_ok = best["mse"] <= VERIFY_MSE_MAX and hc >= VERIFY_HIST_MIN
    # brightness / soft augment
    soft_ok = best["mse_z"] <= 0.035 and hc >= 0.88 and best["mse"] <= 40.0
    # structural twin (color↔gray, exposure, mild crop) — primary for Roboflow leftovers
    struct_ok = best["corr"] >= VERIFY_CORR_MIN or (
        best["corr"] >= VERIFY_CORR_RELAXED and best["mse_z"] <= VERIFY_MSE_NORM_MAX
    )

    ok = bool(abs_ok or soft_ok or struct_ok)
    detail = {
        "mse": round(best["mse"], 3),
        "mse_z": round(best["mse_z"], 4),
        "corr": round(best["corr"], 5),
        "hist_corr": round(hc, 4),
        "used_hflip": flipped,
        "abs_ok": abs_ok,
        "soft_ok": soft_ok,
        "struct_ok": struct_ok,
        "match": ok,
    }
    return ok, detail


def hash_near(a: ImageRecord, b: ImageRecord) -> tuple[bool, dict]:
    """Cheap hash gate before pixel verify. Checks direct + flipped + center crops."""
    da = hamming(a.ahash, b.ahash)
    dd = hamming(a.dhash, b.dhash)
    dp = hamming(a.phash, b.phash)
    da_f = min(hamming(a.ahash, b.ahash_flip), hamming(a.ahash_flip, b.ahash))
    dd_f = min(hamming(a.dhash, b.dhash_flip), hamming(a.dhash_flip, b.dhash))
    dp_f = min(hamming(a.phash, b.phash_flip), hamming(a.phash_flip, b.phash))
    da_c = hamming(a.ahash_center, b.ahash_center)
    dd_c = hamming(a.dhash_center, b.dhash_center)
    dp_c = hamming(a.phash_center, b.phash_center)

    def triple(x, y, z, hx=HAMMING_AHASH, hy=HAMMING_DHASH, hz=HAMMING_PHASH) -> bool:
        return x <= hx and y <= hy and z <= hz

    def strong(x, y, z) -> bool:
        return sum([x <= HAMMING_AHASH - 2, y <= HAMMING_DHASH - 2, z <= HAMMING_PHASH - 2]) >= 2

    # very strong single-signal: near-identical pHash (color/gray twins often land here)
    phash_twin = min(dp, dp_f, dp_c) <= 4 and min(dd, dd_f, dd_c) <= 8

    ok = (
        triple(da, dd, dp)
        or triple(da_f, dd_f, dp_f)
        or triple(da_c, dd_c, dp_c)
        or strong(da, dd, dp)
        or strong(da_f, dd_f, dp_f)
        or strong(da_c, dd_c, dp_c)
        or phash_twin
    )
    return ok, {
        "da": da,
        "dd": dd,
        "dp": dp,
        "da_f": da_f,
        "dd_f": dd_f,
        "dp_f": dp_f,
        "da_c": da_c,
        "dd_c": dd_c,
        "dp_c": dp_c,
        "gate": ok,
    }


# ---------------------------------------------------------------------------
# union-find
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def center_crop(gray: np.ndarray, frac: float = 0.8) -> np.ndarray:
    h, w = gray.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return gray[y0 : y0 + ch, x0 : x0 + cw]


def build_record(path: Path) -> ImageRecord | None:
    bgr = load_bgr(path)
    if bgr is None:
        print(f"  ! unreadable: {path.name}")
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_flip = np.fliplr(gray)
    gray_c = center_crop(gray)
    is_color, score, reason = classify_color(bgr)
    h, w = bgr.shape[:2]
    return ImageRecord(
        path=str(path),
        name=path.name,
        sha256=sha256_file(path),
        nbytes=path.stat().st_size,
        width=w,
        height=h,
        ahash=ahash(gray),
        dhash=dhash(gray),
        phash=phash(gray),
        ahash_flip=ahash(gray_flip),
        dhash_flip=dhash(gray_flip),
        phash_flip=phash(gray_flip),
        ahash_center=ahash(gray_c),
        dhash_center=dhash(gray_c),
        phash_center=phash(gray_c),
        color_score=score,
        is_color=is_color,
        color_reason=reason,
    )


def prefer_keep(recs: list[ImageRecord]) -> ImageRecord:
    """Prefer color over gray twin, then larger / heavier file."""
    return max(
        recs,
        key=lambda r: (
            1 if r.is_color else 0,
            r.color_score,
            r.width * r.height,
            r.nbytes,
            -len(r.name),
        ),
    )


def scan(images: list[Path]) -> tuple[list[ImageRecord], list[Cluster], dict]:
    t0 = time.time()
    print(f"[1/4] indexing {len(images)} images...")
    records: list[ImageRecord] = []
    for i, p in enumerate(images, 1):
        rec = build_record(p)
        if rec:
            records.append(rec)
        if i % 200 == 0 or i == len(images):
            print(f"  indexed {i}/{len(images)}")

    # exact SHA groups
    print("[2/4] exact SHA groups...")
    by_sha: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        by_sha[r.sha256].append(r)
    exact_dup_groups = {k: v for k, v in by_sha.items() if len(v) > 1}

    # near-dup candidates via hash gate + verify
    print("[3/4] near-duplicate search (hash gate + pixel verify)...")
    n = len(records)
    uf = UnionFind(n)
    pair_details: list[dict] = []
    # bucket by rough aspect to cut comparisons slightly
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        ar = round(r.width / max(r.height, 1), 1)
        buckets[(int(ar * 10), 0)].append(i)
        # also put in neighboring aspect buckets
        buckets[(int(ar * 10) - 1, 0)].append(i)
        buckets[(int(ar * 10) + 1, 0)].append(i)

    compared = set()
    verified_pairs = 0
    hash_hits = 0
    for idxs in buckets.values():
        uniq = sorted(set(idxs))
        for ai in range(len(uniq)):
            for bi in range(ai + 1, len(uniq)):
                i, j = uniq[ai], uniq[bi]
                key = (i, j) if i < j else (j, i)
                if key in compared:
                    continue
                compared.add(key)
                a, b = records[i], records[j]
                if a.sha256 == b.sha256:
                    uf.union(i, j)
                    continue
                gate_ok, gate = hash_near(a, b)
                if not gate_ok:
                    continue
                hash_hits += 1
                bgr_a = load_bgr(Path(a.path))
                bgr_b = load_bgr(Path(b.path))
                if bgr_a is None or bgr_b is None:
                    continue
                ok, detail = verify_pair(bgr_a, bgr_b)
                verified_pairs += 1
                if ok:
                    uf.union(i, j)
                    pair_details.append(
                        {
                            "a": a.name,
                            "b": b.name,
                            **gate,
                            **detail,
                        }
                    )

    print(f"  pairs compared~{len(compared)} hash_hits={hash_hits} verified={verified_pairs} matches={len(pair_details)}")

    # build clusters (size>=2)
    print("[4/4] clustering + keep selection...")
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    clusters: list[Cluster] = []
    cid = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        recs = [records[i] for i in members]
        keep = prefer_keep(recs)
        # reason: exact vs near
        shas = {r.sha256 for r in recs}
        reason = "exact_sha" if len(shas) == 1 else "near_dup"
        clusters.append(
            Cluster(
                id=cid,
                members=[r.name for r in recs],
                keep=keep.name,
                reason=reason,
                scores=[
                    {
                        "name": r.name,
                        "px": r.width * r.height,
                        "bytes": r.nbytes,
                        "sha256": r.sha256[:12],
                    }
                    for r in sorted(recs, key=lambda x: (-x.width * x.height, -x.nbytes))
                ],
            )
        )
        cid += 1

    clusters.sort(key=lambda c: (-len(c.members), c.id))

    color_n = sum(1 for r in records if r.is_color)
    gray_n = len(records) - color_n
    drop_n = sum(len(c.members) - 1 for c in clusters)

    summary = {
        "images": len(records),
        "exact_sha_groups": len(exact_dup_groups),
        "exact_sha_extra_files": sum(len(v) - 1 for v in exact_dup_groups.values()),
        "near_dup_clusters": len(clusters),
        "files_to_reject_if_apply": drop_n,
        "survivors_if_apply": len(records) - drop_n,
        "color_now": color_n,
        "gray_now": gray_n,
        "elapsed_sec": round(time.time() - t0, 1),
        "hash_hits": hash_hits,
        "verified_matches": len(pair_details),
    }
    return records, clusters, {"summary": summary, "pair_details": pair_details, "exact_sha_groups": {
        k: [r.name for r in v] for k, v in exact_dup_groups.items()
    }}


def write_report(records: list[ImageRecord], clusters: list[Cluster], meta: dict) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary = meta["summary"]

    (ANALYSIS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ANALYSIS / "records.jsonl").write_text(
        "\n".join(json.dumps(asdict(r), ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    (ANALYSIS / "clusters.jsonl").write_text(
        "\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in clusters) + "\n",
        encoding="utf-8",
    )
    (ANALYSIS / "pair_matches.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in meta["pair_details"]) + "\n",
        encoding="utf-8",
    )

    # human-readable
    lines = [
        "# raw near-duplicate + color scan",
        "",
        f"- images: **{summary['images']}**",
        f"- exact SHA groups: **{summary['exact_sha_groups']}** (extra files: {summary['exact_sha_extra_files']})",
        f"- near-dup clusters: **{summary['near_dup_clusters']}**",
        f"- reject if apply: **{summary['files_to_reject_if_apply']}** → survivors **{summary['survivors_if_apply']}**",
        f"- color / gray (current classifier): **{summary['color_now']}** / **{summary['gray_now']}**",
        f"- elapsed: {summary['elapsed_sec']}s",
        "",
        "## Largest clusters",
        "",
    ]
    for c in clusters[:40]:
        lines.append(f"### cluster {c.id} ({c.reason}) n={len(c.members)} keep=`{c.keep}`")
        for m in c.members:
            mark = "✓" if m == c.keep else "✗"
            lines.append(f"- {mark} `{m}`")
        lines.append("")

    # color borderline review
    borderline = [
        r
        for r in records
        if 0.30 <= r.color_score <= 0.55
    ]
    borderline.sort(key=lambda r: r.color_score)
    lines.append("## Color borderline (score 0.30–0.55)")
    lines.append("")
    for r in borderline[:80]:
        kind = "COLOR" if r.is_color else "GRAY"
        lines.append(f"- `{r.name}` → **{kind}** score={r.color_score:.3f}")
    lines.append("")

    (ANALYSIS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report → {ANALYSIS / 'REPORT.md'}")


def apply_cull_and_rename(records: list[ImageRecord], clusters: list[Cluster]) -> None:
    REJECTED.mkdir(parents=True, exist_ok=True)

    reject_names: set[str] = set()
    keep_names: set[str] = set()
    for c in clusters:
        for m in c.members:
            if m == c.keep:
                keep_names.add(m)
            else:
                reject_names.add(m)

    # move rejects
    moved = 0
    for name in sorted(reject_names):
        src = RAW / name
        if not src.exists():
            continue
        dest = REJECTED / name
        if dest.exists():
            dest = REJECTED / f"{src.stem}__{uuid.uuid4().hex[:8]}{src.suffix}"
        shutil.move(str(src), str(dest))
        moved += 1
    print(f"moved rejects → {REJECTED} ({moved})")

    # survivors: remaining images in RAW (not in analysis/rejected)
    survivors = list_images(RAW)
    print(f"reclassifying + renaming {len(survivors)} survivors...")

    manga: list[Path] = []
    colored: list[Path] = []
    class_rows = []
    for p in survivors:
        bgr = load_bgr(p)
        if bgr is None:
            continue
        is_color, score, reason = classify_color(bgr)
        class_rows.append({"old": p.name, "is_color": is_color, "score": score, "reason": reason})
        (colored if is_color else manga).append(p)

    # two-phase rename
    jobs: list[tuple[Path, str]] = []
    wm = max(4, len(str(len(manga))))
    wc = max(4, len(str(len(colored))))
    for i, p in enumerate(sorted(manga, key=lambda x: x.name.lower()), 1):
        jobs.append((p, f"manga_page_{i:0{wm}d}{p.suffix.lower()}"))
    for i, p in enumerate(sorted(colored, key=lambda x: x.name.lower()), 1):
        jobs.append((p, f"colored_page_{i:0{wc}d}{p.suffix.lower()}"))

    staged: list[tuple[Path, str]] = []
    for src, final in jobs:
        tmp = RAW / f".__rn_{uuid.uuid4().hex}{src.suffix.lower()}"
        src.rename(tmp)
        staged.append((tmp, final))
    for tmp, final in staged:
        tmp.rename(RAW / final)

    final_by_old = {src.name: final for src, final in jobs}
    rename_map = [
        {
            "old": row["old"],
            "new": final_by_old.get(row["old"]),
            "is_color": row["is_color"],
            "score": row["score"],
            "reason": row["reason"],
        }
        for row in class_rows
    ]
    (ANALYSIS / "apply_rename_map.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rename_map) + "\n",
        encoding="utf-8",
    )

    apply_summary = {
        "rejected": moved,
        "survivors": len(survivors),
        "manga_page": len(manga),
        "colored_page": len(colored),
    }
    (ANALYSIS / "apply_summary.json").write_text(json.dumps(apply_summary, indent=2), encoding="utf-8")
    print(json.dumps(apply_summary, indent=2))


def main() -> None:
    global RAW, ANALYSIS, REJECTED

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="reject near-dups, recolor-split, rename")
    ap.add_argument("--dir", type=Path, default=None, help="image folder (default: datasets/raw)")
    args = ap.parse_args()

    RAW = (args.dir or ROOT / "datasets" / "raw").resolve()
    ANALYSIS = RAW / "_analysis"
    REJECTED = RAW / "_rejected_near_dup"

    images = list_images(RAW)
    if not images:
        raise SystemExit(f"no images in {RAW}")

    records, clusters, meta = scan(images)
    write_report(records, clusters, meta)
    print(json.dumps(meta["summary"], indent=2))

    if args.apply:
        print("\n=== APPLY ===")
        apply_cull_and_rename(records, clusters)
        print("done.")
    else:
        print("\nscan-only. re-run with --apply to cull + rename.")


if __name__ == "__main__":
    main()
