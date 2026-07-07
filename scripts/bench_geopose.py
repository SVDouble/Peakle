"""CLI for the GeoPose3K orientation benchmark (logic in peakle.localize.bench).

Usage: python scripts/bench_geopose.py [--max-n 60] [--samples name1,name2] [--extent-km 90]
Writes local/output/<dt>-geopose-bench/{results.json, summary.md, overlays/*.jpg}.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from peakle.localize.bench import find_sample_dirs, run_sample, summarize

BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=60)
    ap.add_argument("--samples", default=None, help="comma-separated sample names")
    ap.add_argument("--extent-km", type=float, default=90.0)
    ap.add_argument("--grid", type=int, default=3000)
    ap.add_argument("--extractor", choices=["color", "sam3"], default="color")
    ap.add_argument(
        "--manifest",
        default=str(BASE / "scripts/geopose_manifest_60.txt"),
        help="pinned sample list; keeps runs comparable as the corpus grows",
    )
    args = ap.parse_args()

    dirs = find_sample_dirs()
    if args.samples:
        want = set(args.samples.split(","))
        dirs = [d for d in dirs if d.name in want]
    elif Path(args.manifest).exists():
        want = [ln.strip() for ln in Path(args.manifest).read_text().splitlines() if ln.strip()]
        by_name = {d.name: d for d in dirs}
        dirs = [by_name[n] for n in want if n in by_name]
    dirs = dirs[: args.max_n]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = BASE / f"local/output/{stamp}-geopose-bench"
    (outdir / "overlays").mkdir(parents=True, exist_ok=True)

    rows = []
    for i, d in enumerate(dirs):
        t0 = time.time()
        try:
            rec = run_sample(d, args.extent_km * 1000.0, args.grid, outdir / "overlays", args.extractor)
        except Exception as exc:  # a bad sample must not kill the whole benchmark
            traceback.print_exc()
            rec = {"name": d.name, "manual": False, "gt_yaw": float("nan"), "error": str(exc)}
        rows.append(rec)
        o, e = rec.get("oracle", {}), rec.get("extracted", {})
        print(
            f"[{i + 1}/{len(dirs)}] {d.name}: oracle {o.get('yaw_err', 'ERR'):>6} extr {e.get('yaw_err', 'ERR'):>6} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    (outdir / "results.json").write_text(json.dumps(rows, indent=1))
    summary = summarize([r for r in rows if "error" not in r])
    (outdir / "summary.md").write_text(summary)
    print(f"\n{summary}\n-> {outdir}")


if __name__ == "__main__":
    main()
