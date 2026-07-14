"""Provision an audited SWISSIMAGE tile cache for render-matching experiments.

Benchmark execution itself is offline.  This separate command is the only path
that opts into WMTS downloads, and it writes a content-hash manifest for every
tile acquired or reused.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from peakle.localize.bench import find_sample_dirs
from peakle.localize.geopose import load_sample
from peakle.localize.paths import BASE
from peakle.localize.strategy_bench import MatrixConfig, load_benchmark_terrain
from peakle.rendering.orthophoto import MissingOrthophotoTiles, SwissImageProvider


def main() -> None:
    args = _parser().parse_args()
    if not args.dry_run and args.time == "current" and not args.allow_mutable_current:
        raise SystemExit(
            "refusing to provision the mutable SWISSIMAGE 'current' layer without an explicit acknowledgement; "
            "use a dated --time value for reproducibility, or pass --allow-mutable-current"
        )
    samples = _selected_samples(args.samples)
    if not samples:
        raise SystemExit("no complete GeoPose sample matched --samples")
    cache_dir = Path(args.cache).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = BASE / cache_dir
    terrain_config = MatrixConfig(
        extent_m=args.extent_km * 1000.0,
        terrain_grid=args.grid,
        root_seed=args.seed,
        map_center_offset_fraction=args.map_offset_fraction,
    )
    provider = SwissImageProvider(
        cache_dir=cache_dir,
        zoom=args.zoom,
        time=args.time,
        download_missing=not args.dry_run,
        max_tiles=args.max_tiles,
        download_workers=args.workers,
    )
    manifests: list[dict[str, Any]] = []
    for sample_dir in samples:
        sample = load_sample(sample_dir)
        selection = load_benchmark_terrain(sample, terrain_config)
        terrain = selection.terrain
        bounds = {
            "east_min_m": float(terrain.x_m[0]),
            "east_max_m": float(terrain.x_m[-1]),
            "north_min_m": float(terrain.y_m[0]),
            "north_max_m": float(terrain.y_m[-1]),
        }
        try:
            mosaic = provider.mosaic_for_local_bounds(terrain.frame, **bounds)
        except MissingOrthophotoTiles as exc:
            print(f"{sample.name}: {len(exc.missing)}/{exc.requested} tiles missing (dry run)")
            manifests.append(
                {
                    "sample": sample.name,
                    "status": "missing",
                    "requested_tiles": exc.requested,
                    "missing_tiles": len(exc.missing),
                    "bounds_local_m": bounds,
                }
            )
            continue
        provenance = mosaic.provenance()
        print(
            f"{sample.name}: {provenance['tile_count']} tiles, "
            f"manifest {provenance['tile_manifest_sha256'][:12]}…, "
            f"network={'yes' if provenance['network_used_during_mosaic_build'] else 'no'}"
        )
        manifests.append(
            {
                "sample": sample.name,
                "status": "complete",
                "terrain_origin": terrain.frame.origin.model_dump(mode="json"),
                "bounds_local_m": bounds,
                "mosaic": provenance,
                "tiles": list(mosaic.tile_records),
            }
        )

    payload = {
        "schema": "peakle_orthophoto_cache_manifest_v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": "Swiss Federal Office of Topography swisstopo / geo.admin.ch",
        "layer": provider.layer,
        "service_url_template": provider.url_template,
        "time": provider.time,
        "zoom": provider.zoom,
        "cache_dir": str(cache_dir),
        "terrain_config": {
            "extent_m": terrain_config.extent_m,
            "grid": terrain_config.terrain_grid,
            "root_seed": terrain_config.root_seed,
            "map_center_offset_fraction": terrain_config.map_center_offset_fraction,
        },
        "dry_run": args.dry_run,
        "mutable_current_acknowledged": bool(args.allow_mutable_current),
        "samples": manifests,
    }
    if args.dry_run:
        return
    output = Path(args.manifest) if args.manifest else _default_manifest(cache_dir)
    if not output.is_absolute():
        output = BASE / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote immutable cache manifest: {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, help="comma-separated complete GeoPose sample names")
    parser.add_argument("--cache", default="local/data/swissimage")
    parser.add_argument("--manifest", help="new manifest JSON path; must not exist")
    parser.add_argument("--zoom", type=int, default=14)
    parser.add_argument(
        "--time",
        default="current",
        help="WMTS time identifier; prefer a dated value because 'current' can change in place",
    )
    parser.add_argument(
        "--allow-mutable-current",
        action="store_true",
        help="explicitly allow provisioning into a cache path whose upstream 'current' tiles may change",
    )
    parser.add_argument("--extent-km", type=float, default=40.0)
    parser.add_argument("--grid", type=int, default=1335)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--map-offset-fraction", type=float, default=0.16)
    parser.add_argument("--max-tiles", type=int, default=1600)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _selected_samples(value: str) -> list[Path]:
    requested = [name.strip() for name in value.split(",") if name.strip()]
    available = {path.name: path for path in find_sample_dirs()}
    missing = [name for name in requested if name not in available]
    if missing:
        raise SystemExit(f"unknown or incomplete GeoPose sample(s): {', '.join(missing)}")
    return [available[name] for name in requested]


def _default_manifest(cache_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return cache_dir / "manifests" / f"{stamp}-swissimage.json"


if __name__ == "__main__":
    main()
