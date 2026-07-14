"""CLI for the GeoPose3K orientation benchmark (logic in peakle.localize.bench).

Usage: python -m peakle.scripts.bench_geopose [--max-n 60] [--samples name1,name2] [--extent-km 90]
Writes local/output/<dt>-geopose-bench/{results.json, summary.md, overlays/*.jpg}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from peakle.localize.bench import find_sample_dirs, run_sample, summarize
from peakle.localize.paths import BASE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=60)
    ap.add_argument("--samples", default=None, help="comma-separated sample names")
    ap.add_argument("--extent-km", type=float, default=90.0)
    ap.add_argument("--grid", type=int, default=3000)
    ap.add_argument(
        "--extractor",
        choices=["color", "dexined", "sam3", "mobile_sam", "sam2"],
        default="color",
        help="explicit photo extractor; learned backends may load/download their documented weights",
    )
    ap.add_argument(
        "--manifest",
        default=str(Path(__file__).with_name("geopose_manifest_60.txt")),
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
    run_metadata = _run_metadata(args, outdir.name, len(dirs), status="running")
    (outdir / "run.json").write_text(json.dumps(run_metadata, indent=2))

    rows = []
    for i, d in enumerate(dirs):
        t0 = time.time()
        try:
            rec = run_sample(d, args.extent_km * 1000.0, args.grid, outdir / "overlays", args.extractor)
        except Exception as exc:  # a bad sample must not kill the whole benchmark
            traceback.print_exc()
            rec = {"name": d.name, "manual": False, "gt_yaw": float("nan"), "error": str(exc)}
        rec["runtime_s"] = round(time.time() - t0, 3)
        rec["algorithm"] = "horizon"
        rec["prior_regime"] = "known_position_no_orientation_prior"
        rows.append(rec)
        oracle_value, extracted_value = rec.get("oracle"), rec.get("extracted")
        o = oracle_value if isinstance(oracle_value, dict) else {}
        e = extracted_value if isinstance(extracted_value, dict) else {}
        print(
            f"[{i + 1}/{len(dirs)}] {d.name}: oracle {o.get('yaw_err', 'ERR'):>6} extr {e.get('yaw_err', 'ERR'):>6} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    (outdir / "results.json").write_text(json.dumps(rows, indent=1))
    summary = summarize([r for r in rows if "error" not in r])
    (outdir / "summary.md").write_text(summary)
    run_metadata["status"] = "complete"
    run_metadata["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    run_metadata["completed_samples"] = sum("error" not in row for row in rows)
    run_metadata["failed_samples"] = sum("error" in row for row in rows)
    (outdir / "run.json").write_text(json.dumps(run_metadata, indent=2))
    print(f"\n{summary}\n-> {outdir}")


def _run_metadata(args: argparse.Namespace, run_id: str, sample_count: int, *, status: str) -> dict:
    manifest = Path(args.manifest)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "code": _git_provenance(),
        "dataset": {
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest) if manifest.exists() else None,
            "requested_samples": sample_count,
        },
        "matrix": {
            "algorithms": ["horizon"],
            "evidence_tracks": ["pfm_oracle", "photo_auto"],
            "prior_regimes": ["known_position_no_orientation_prior"],
            "seeds": [],
        },
        "terrain": {
            "source": "Copernicus GLO-30",
            "extent_km": args.extent_km,
            "grid": args.grid,
            "nominal_resolution_m": 30.0,
        },
        "extractor": args.extractor,
        "compatibility_policy": "gt_dem_compat_v1",
    }


def _git_provenance() -> dict[str, str | bool | None]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.run(args, cwd=BASE, check=True, capture_output=True, text=True).stdout.strip()
        except OSError, subprocess.CalledProcessError:
            return None

    sha = command("git", "rev-parse", "HEAD")
    status = command("git", "status", "--porcelain")
    return {"git_sha": sha, "dirty": bool(status) if status is not None else None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
