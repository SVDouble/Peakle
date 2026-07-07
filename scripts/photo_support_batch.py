"""Compute photo-edge support (and all GT-Lab layers) for every GT v2 sample.

Resumable: samples with an existing support.json are skipped. Ends with the corpus support
distribution per family and the worst offenders — GT lines the photograph does not show, which
must be down-weighted in extractor grading and explanation matching.

Usage: python scripts/photo_support_batch.py [--max-n N]
"""

from __future__ import annotations

import argparse
import json
import time

from peakle.localize.gtquality import support_stats
from peakle.web.api.gtlab import GTV2, LAYERS, _build_layers


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
            print(
                f"[{i + 1}/{len(names)}] {name}: "
                + " ".join(f"{k}={v}" for k, v in sup.items() if k.startswith("gt_"))
                + f" ({time.time() - t0:.0f}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — one bad sample must not kill the batch
            failed += 1
            print(f"[{i + 1}/{len(names)}] {name}: FAILED {exc}", flush=True)

    rows = [(name, json.loads((LAYERS / name / "support.json").read_text())) for name in names if (LAYERS / name / "support.json").exists()]
    print(f"\nbuilt {done}, skipped {skipped}, failed {failed}; support available for {len(rows)}")
    for src, stats in support_stats(rows).items():
        print(f"{src:4}: " + " | ".join(stats))
    worst = sorted((s.get("gt_occ"), n) for n, s in rows if s.get("gt_occ") is not None)[:15]
    print("\nworst GT occlusion support (dataset lines the photo does not show):")
    for v, n in worst:
        print(f"  {v:.2f}  {n}")


if __name__ == "__main__":
    main()
