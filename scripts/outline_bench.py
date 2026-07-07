"""Outline-extraction benchmark: grade photo-side extractors against GT v2 on CLEAN samples.

Extractors (pluggable, graded with the same scorer — peakle.localize.outline_score):
  color    best colour-candidate skyline only (the current production extraction; no internal lines)
  dexined  DexiNed learned edges (kornia), thresholded — includes internal ridge candidates
  sam3     SAM3 concept-instance silhouette map (the best ridge source on the private-photo set)

Per extractor: threshold sweep, report the best-F1 operating point with PER-FAMILY recall —
R_internal is the number this benchmark exists to move (baseline ~0.03-0.26 from colour skylines).

Usage: python scripts/outline_bench.py [--extractors color,dexined,sam3] [--max-n 20] [--manifest ...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from peakle.localize.extract import extract_candidates
from peakle.localize.outline_score import rows_to_mask, score_outlines

from peakle.localize.paths import BASE, GEOPOSE_DIR as DATA, GTV2_DIR as GTV2


def load_rgb(name: str, w: int, h: int) -> np.ndarray:
    rgb = np.asarray(Image.open(DATA / name / "cyl/photo_crop.jpg").convert("RGB"), np.uint8)
    if rgb.shape[:2] != (h, w):
        rgb = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR), np.uint8)
    return rgb


def extractor_color(rgb: np.ndarray, _cache: dict) -> list[tuple[str, np.ndarray]]:
    h = rgb.shape[0]
    best = None
    for c in extract_candidates(rgb).values():
        if c.coverage >= 0.25 and (best is None or c.coverage > best.coverage):
            best = c
    return [("-", rows_to_mask(best.rows, h))] if best else []


def extractor_dexined(rgb: np.ndarray, cache: dict) -> list[tuple[str, np.ndarray]]:
    if "dexined" not in cache:
        from peakle.edges import load_learned_edges

        cache["dexined"] = load_learned_edges()
    det = cache["dexined"]
    if det is None:
        return []
    emap = det.detect(rgb.astype(np.float64) / 255.0)  # detect() expects RGB in [0, 1]
    return [(f"th={t}", emap >= t) for t in (0.25, 0.4, 0.55)]


def extractor_sam3(rgb: np.ndarray, cache: dict) -> list[tuple[str, np.ndarray]]:
    if "sam3" not in cache:
        from peakle.segmenters import load_segmenter

        cache["sam3"] = load_segmenter("sam3")
    seg = cache["sam3"]
    if seg is None:
        return []
    mountain = seg.terrain_mask(rgb)
    sil = seg.silhouette_map(rgb, mountain)
    return [(f"sil>={t}", sil >= t) for t in (1, 2)]


EXTRACTORS = {"color": extractor_color, "dexined": extractor_dexined, "sam3": extractor_sam3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractors", default="color,dexined,sam3")
    ap.add_argument("--max-n", type=int, default=20)
    ap.add_argument("--manifest", default=str(BASE / "scripts/geopose_manifest_60.txt"))
    args = ap.parse_args()

    index = {r["name"]: r for r in json.load(open(GTV2 / "index.json"))}
    names = [ln.strip() for ln in Path(args.manifest).read_text().splitlines() if ln.strip()]
    clean = [n for n in names if index.get(n, {}).get("quality") == "CLEAN"][: args.max_n]
    print(f"{len(clean)} CLEAN samples; extractors: {args.extractors}\n")

    cache: dict = {}
    results: dict[str, list] = {}
    for ex_name in args.extractors.split(","):
        fn = EXTRACTORS[ex_name]
        per_variant: dict[str, list] = {}
        t0 = time.time()
        for n in clean:
            g = index[n]
            rgb = load_rgb(n, g["width"], g["height"])
            try:
                variants = fn(rgb, cache)
            except Exception as exc:
                print(f"  {ex_name} failed on {n}: {exc}")
                continue
            for tag, mask in variants:
                s = score_outlines(mask, GTV2 / f"{n}.npz")
                per_variant.setdefault(tag, []).append(s)
        for tag, scores in per_variant.items():
            med = lambda k: float(np.median([getattr(s, k) for s in scores]))
            results[f"{ex_name} {tag}"] = [
                med("precision"),
                med("recall_skyline"),
                med("recall_internal"),
                med("f1"),
                len(scores),
            ]
        print(f"[{ex_name}] done in {time.time() - t0:.0f}s")

    print(f"\n{'extractor':22} {'P':>6} {'R_sky':>6} {'R_int':>6} {'F1':>6} {'n':>4}   (medians over CLEAN samples)")
    for tag, (p, rs, ri, f1, n) in sorted(results.items(), key=lambda kv: -kv[1][3]):
        print(f"{tag:22} {p:6.2f} {rs:6.2f} {ri:6.2f} {f1:6.2f} {n:4}")


if __name__ == "__main__":
    main()
