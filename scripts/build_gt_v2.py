"""CLI for the GT v2 builder (the pipeline itself lives in peakle.localize.gtbuild).

Output (resumable — existing samples are skipped unless --force):
  local/derived/gt_v2/<name>.npz / .json   per-sample arrays + record
  local/derived/gt_v2/index.json           rebuilt at the end

Usage:
  python scripts/build_gt_v2.py --manifest scripts/geopose_manifest_60.txt
  python scripts/build_gt_v2.py --manual            # all MANUAL samples in the corpus
  python scripts/build_gt_v2.py --all               # every complete sample
  python scripts/build_gt_v2.py --samples a,b,c
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from peakle.localize.gtbuild import GTV2_DIR, build_index, build_one, discover_samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--manual", action="store_true", help="all MANUAL samples in the corpus")
    ap.add_argument("--all", action="store_true", help="every complete sample in the corpus")
    ap.add_argument("--force", action="store_true", help="rebuild samples that already have records")
    ap.add_argument("--max-n", type=int, default=10**9)
    args = ap.parse_args()

    if args.samples:
        names = args.samples.split(",")
    elif args.manifest:
        names = [ln.strip() for ln in Path(args.manifest).read_text().splitlines() if ln.strip()]
    elif args.manual:
        names = discover_samples("manual")
    elif args.all:
        names = discover_samples("all")
    else:
        raise SystemExit("pass --manifest, --samples, --manual or --all")
    names = names[: args.max_n]

    done = skipped = failed = 0
    for i, name in enumerate(names):
        if not args.force and (GTV2_DIR / f"{name}.json").exists():
            skipped += 1
            continue
        t0 = time.time()
        try:
            rec = build_one(name)
            done += 1
            print(
                f"[{i + 1}/{len(names)}] {name}: {rec.quality} sky={rec.sky_cons_px} "
                f"ct={rec.contour_cons_px} dyaw={rec.dyaw_deg:+.1f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — one bad sample must not kill the batch
            failed += 1
            traceback.print_exc()
            print(f"[{i + 1}/{len(names)}] {name}: FAILED {exc}", flush=True)

    index = build_index()
    clean = [r for r in index if r["quality"] == "CLEAN"]
    print(f"\nbuilt {done}, skipped {skipped}, failed {failed}; index: {len(index)} total, {len(clean)} CLEAN")


if __name__ == "__main__":
    main()
