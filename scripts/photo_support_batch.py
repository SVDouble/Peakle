"""Compute photo-edge support (and all GT-Lab layers) for every GT v2 sample.

Resumable: samples with an existing support.json are skipped.  Ends with the corpus support
distribution per family and the worst offenders — GT lines the photograph does not show, which
must be down-weighted in extractor grading and explanation matching.

Usage: python scripts/photo_support_batch.py [--max-n N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from peakle.web.api.gtlab import GTV2, LAYERS, _build_layers

FAMS = ("sky", "occ", "rib", "cou")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=10**9)
    args = ap.parse_args()

    names = [r["name"] for r in json.loads((GTV2 / "index.json").read_text())][: args.max_n]
    done = skipped = failed = 0
    for i, name in enumerate(names):
        if (LAYERS / name / "support.json").exists():
            skipped += 1
            continue
        t0 = time.time()
        try:
            _build_layers(name)
            done += 1
            sup = json.loads((LAYERS / name / "support.json").read_text())
            print(f"[{i+1}/{len(names)}] {name}: " + " ".join(f"{k}={v}" for k, v in sup.items() if k.startswith("gt_"))
                  + f" ({time.time()-t0:.0f}s)", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[{i+1}/{len(names)}] {name}: FAILED {exc}", flush=True)

    rows = []
    for name in names:
        p = LAYERS / name / "support.json"
        if p.exists():
            rows.append((name, json.loads(p.read_text())))
    print(f"\nbuilt {done}, skipped {skipped}, failed {failed}; support available for {len(rows)}")
    for src in ("gt", "dem"):
        stats = []
        for fam in FAMS:
            vals = [s[f"{src}_{fam}"] for _, s in rows if s.get(f"{src}_{fam}") is not None]
            stats.append(f"{fam} med={np.median(vals):.2f} p10={np.percentile(vals, 10):.2f}" if vals else f"{fam} n/a")
        print(f"{src:4}: " + " | ".join(stats))
    worst = sorted((s.get("gt_occ"), n) for n, s in rows if s.get("gt_occ") is not None)[:15]
    print("\nworst GT occlusion support (dataset lines the photo does not show):")
    for v, n in worst:
        print(f"  {v:.2f}  {n}")


if __name__ == "__main__":
    main()
