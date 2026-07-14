"""Run the provenance-rich GeoPose workbench-strategy matrix.

Example (the default is deliberately one sample because the full-pose methods
are expensive)::

    python -m peakle.scripts.bench_pose_matrix --profile core --max-n 1

Use ``--algorithms`` / ``--evidence`` / ``--regimes`` for focused experiments.
Every completed run is committed as a new write-once
``local/output/*-geopose-bench`` artifact that the benchmark API can discover.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy
import scipy

from peakle.localize.bench import find_sample_dirs
from peakle.localize.correspondence import (
    CORRESPONDENCE_CACHE_ENTRY_SCHEMA,
    CORRESPONDENCE_CACHE_KEY_SCHEMA,
)
from peakle.localize.paths import BASE
from peakle.localize.render_match_pnp import CandidateValidationConfig
from peakle.localize.strategy_bench import (
    ALGORITHMS,
    DEFAULT_ALGORITHMS,
    EVIDENCE_TRACKS,
    MATRIX_EXTRACTORS,
    PRIOR_REGIMES,
    MatrixConfig,
    build_matrix_cells,
    commit_artifact,
    compatibility_contract,
    config_record,
    default_terrain_cache_inventory,
    file_sha256,
    input_fingerprint,
    run_sample_matrix,
    source_implementation_fingerprint,
)


def main() -> None:
    args = _parser().parse_args()
    config = MatrixConfig(
        profile=args.profile,
        algorithms=_csv_choice(args.algorithms, ALGORITHMS, "algorithm"),
        evidence_tracks=_csv_choice(args.evidence, EVIDENCE_TRACKS, "evidence track"),
        prior_regimes=_csv_choice(args.regimes, PRIOR_REGIMES, "prior regime"),
        perturbation_bucket=args.perturbation,
        replicates=args.replicates,
        root_seed=args.seed,
        extent_m=args.extent_km * 1000.0,
        terrain_grid=args.grid,
        terrain_stride=args.terrain_stride,
        extractor=args.extractor,
        map_center_offset_fraction=args.map_offset_fraction,
        render_matcher=args.render_matcher,
        render_modality=args.render_modality,
        render_width_px=args.render_width,
        render_height_px=args.render_height,
        render_yaw_step_deg=args.render_yaw_step,
        render_refinement_passes=args.render_refinement_passes,
        native_patch_stride=args.native_patch_stride,
        render_candidate_validation=CandidateValidationConfig(enabled=not args.disable_candidate_validation),
        matcher_command=tuple(shlex.split(args.matcher_command)) if args.matcher_command else (),
        matcher_id=args.matcher_id,
        matcher_manifest_path=args.matcher_manifest,
        matcher_cache_dir=args.matcher_cache,
        orthophoto_cache_dir=args.orthophoto_cache,
        orthophoto_zoom=args.orthophoto_zoom,
        orthophoto_time=args.orthophoto_time,
        orthophoto_max_tiles=args.orthophoto_max_tiles,
    )
    config.validate()
    sample_dirs = _selected_samples(args)
    if not sample_dirs:
        raise SystemExit("no complete GeoPose samples matched the request")
    output_dir = Path(args.output) if args.output else _default_output_dir()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing artifact directory: {output_dir}")

    requested_cells = build_matrix_cells(config)
    applicable_per_sample = sum(cell.applicable for cell in requested_cells)
    print(
        f"GeoPose matrix: {len(sample_dirs)} sample(s), {applicable_per_sample} applicable cells/sample, "
        f"profile={config.profile}; output will be {output_dir}",
        flush=True,
    )
    started_at = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for index, sample_dir in enumerate(sample_dirs, 1):
        sample_started = time.perf_counter()

        def report(case: dict[str, Any]) -> None:
            status = case.get("status")
            if status == "skipped":
                return
            success = case.get("success", {}).get("value")
            outcome = "PASS" if success is True else "FAIL" if success is False else str(status).upper()
            print(
                f"  {case['algorithm']:<10} {case['prior_regime']:<20} "
                f"{case['evidence_track']:<10} {outcome:<5} {float(case.get('runtime_s') or 0.0):7.1f}s",
                flush=True,
            )

        try:
            row, sample_cases = run_sample_matrix(sample_dir, config, progress=report)
        except Exception as exc:  # terrain/input failure must not destroy results from earlier samples
            row = {
                "name": sample_dir.name,
                "manual": False,
                "error": {"type": type(exc).__name__, "message": str(exc)[:500]},
            }
            sample_cases = []
            print(f"[{index}/{len(sample_dirs)}] {sample_dir.name}: ERROR {type(exc).__name__}: {exc}", flush=True)
        else:
            print(
                f"[{index}/{len(sample_dirs)}] {sample_dir.name}: {len(sample_cases)} matrix records "
                f"in {time.perf_counter() - sample_started:.1f}s",
                flush=True,
            )
        rows.append(row)
        cases.extend(sample_cases)

    finished_at = datetime.now(UTC)
    metadata = _run_metadata(
        config,
        args,
        sample_dirs,
        started_at=started_at,
        finished_at=finished_at,
        wall_runtime_s=round((finished_at - started_at).total_seconds(), 3),
    )
    committed = commit_artifact(output_dir, run_metadata=metadata, rows=rows, matrix_cases=cases)
    print(
        f"Committed {len(cases)} matrix records to {output_dir} (results sha256 {committed['results_sha256'][:12]}…).",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="optional sample cap; defaults to one for a manifest/default run and all explicitly named samples",
    )
    parser.add_argument("--samples", help="comma-separated sample names (overrides manifest)")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).with_name("geopose_manifest_60.txt")),
        help="pinned sample list used when --samples is absent",
    )
    parser.add_argument("--profile", choices=("core", "full"), default="core")
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--evidence", default=",".join(EVIDENCE_TRACKS))
    parser.add_argument("--regimes", default=",".join(PRIOR_REGIMES))
    parser.add_argument("--perturbation", choices=("mild", "standard", "hard"), default="standard")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--extent-km", type=float, default=40.0)
    parser.add_argument("--grid", type=int, default=1335, help="40 km default keeps native GLO-30 spacing")
    parser.add_argument("--terrain-stride", type=int, default=6)
    parser.add_argument("--extractor", choices=MATRIX_EXTRACTORS, default="color")
    parser.add_argument("--map-offset-fraction", type=float, default=0.16)
    parser.add_argument(
        "--render-matcher",
        choices=("disabled", "sift", "worker"),
        default="disabled",
        help="SIFT is a hermetic control; learned matchers use the external worker contract",
    )
    parser.add_argument(
        "--render-modality",
        choices=("hillshade", "normal", "relative_depth", "orthophoto"),
        default="hillshade",
    )
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=256)
    parser.add_argument("--render-yaw-step", type=float, default=30.0)
    parser.add_argument("--render-refinement-passes", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--native-patch-stride",
        type=int,
        default=8,
        help="fine elevation mesh decimation (2 m swissALTI3D -> 16 m at the default stride)",
    )
    parser.add_argument(
        "--disable-candidate-validation",
        action="store_true",
        help=(
            "ablation only: allow the selected PnP pose without the default held-out spatial "
            "reprojection and candidate-render visibility checks"
        ),
    )
    parser.add_argument(
        "--matcher-command",
        help="quoted worker command; Peakle appends '--request /absolute/request.json'",
    )
    parser.add_argument("--matcher-id", default="external_matcher")
    parser.add_argument("--matcher-manifest", help="JSON manifest with verified local model artifacts")
    parser.add_argument(
        "--matcher-cache",
        help="optional content-addressed correspondence cache; disabled when omitted",
    )
    parser.add_argument("--orthophoto-cache", default="local/data/swissimage")
    parser.add_argument("--orthophoto-zoom", type=int, default=14)
    parser.add_argument("--orthophoto-time", default="current")
    parser.add_argument("--orthophoto-max-tiles", type=int, default=1600)
    parser.add_argument("--output", help="new output directory; must not already exist")
    return parser


def _selected_samples(args: argparse.Namespace) -> list[Path]:
    dirs = find_sample_dirs()
    by_name = {path.name: path for path in dirs}
    if args.samples:
        names = [name.strip() for name in args.samples.split(",") if name.strip()]
    else:
        manifest = Path(args.manifest)
        names = (
            [line.strip() for line in manifest.read_text().splitlines() if line.strip()] if manifest.exists() else []
        )
    selected = [by_name[name] for name in names if name in by_name] if names else dirs
    limit = args.max_n if args.max_n is not None else (len(selected) if args.samples else 1)
    if limit < 1:
        raise SystemExit("--max-n must be positive")
    return selected[:limit]


def _csv_choice(value: str, allowed: tuple[str, ...], label: str):
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(selected) - set(allowed)
    if unknown:
        raise SystemExit(f"unknown {label}(s): {', '.join(sorted(unknown))}; expected {', '.join(allowed)}")
    if not selected:
        raise SystemExit(f"at least one {label} is required")
    return selected


def _default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return BASE / f"local/output/{stamp}-matrix-geopose-bench"


def _run_metadata(
    config: MatrixConfig,
    args: argparse.Namespace,
    sample_dirs: list[Path],
    *,
    started_at: datetime,
    finished_at: datetime,
    wall_runtime_s: float,
) -> dict[str, Any]:
    manifest = Path(args.manifest)
    implementation_paths = [
        BASE / "src/peakle/localize/strategy_bench.py",
        BASE / "src/peakle/scripts/bench_pose_matrix.py",
        BASE / "src/peakle/localize/compatibility.py",
        BASE / "src/peakle/localize/extract.py",
        BASE / "src/peakle/optimization/solve.py",
        BASE / "src/peakle/optimization/objective.py",
        BASE / "src/peakle/optimization/contour_database.py",
        BASE / "src/peakle/optimization/horizon.py",
        BASE / "src/peakle/rendering/point_skyline.py",
        BASE / "src/peakle/localize/raycast.py",
        BASE / "src/peakle/localize/correspondence.py",
        BASE / "src/peakle/scripts/roma_match_worker.py",
        BASE / "src/peakle/localize/pnp.py",
        BASE / "src/peakle/localize/candidate_validation.py",
        BASE / "src/peakle/localize/render_match_pnp.py",
        BASE / "src/peakle/localize/swissdem.py",
        BASE / "src/peakle/rendering/orthophoto.py",
        BASE / "src/peakle/rendering/terrain_view.py",
        BASE / "src/peakle/rendering/rasterizer.py",
    ]
    cells = build_matrix_cells(config)
    return {
        "created_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "wall_runtime_s": wall_runtime_s,
        "code": _code_provenance(implementation_paths),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "peakle": _package_version("peakle"),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "uv_lock_sha256": file_sha256(BASE / "uv.lock") if (BASE / "uv.lock").exists() else None,
        },
        "dataset": {
            "name": "GeoPose3K",
            "manifest": str(manifest),
            "manifest_sha256": file_sha256(manifest) if manifest.exists() else None,
            "sample_names": [path.name for path in sample_dirs],
            "requested_samples": len(sample_dirs),
            "input_fingerprint": input_fingerprint(sample_dirs),
        },
        "matrix": {
            **config_record(config),
            "requested_cells_per_sample": len(cells),
            "applicable_cells_per_sample": sum(cell.applicable for cell in cells),
            "cell_contract": [
                {
                    "algorithm": cell.algorithm,
                    "prior_regime": cell.prior_regime,
                    "evidence_track": cell.evidence_track,
                    "replicate": cell.replicate,
                    "applicable": cell.applicable,
                    "skip_reason": cell.skip_reason,
                }
                for cell in cells
            ],
        },
        "terrain": {
            "far_source": "Copernicus GLO-30",
            "far_nominal_resolution_m": 30.0,
            "near_source": "cached swissALTI3D 2 m where available",
            "evaluation_reference_patch": (
                "May be centred on the reference solely for fixed-pose GT-to-DEM compatibility; explicitly never "
                "fused into or supplied to an estimator."
            ),
            "estimator_patch_center": (
                "Supplied position prior only; no-position-prior cells receive no native patch."
            ),
            "estimator_patch_network_allowed": False,
            "native_near_patch_used_by_render_pnp_when_prior_centered_and_visible": True,
            "native_near_patch_render_stride": config.native_patch_stride,
            "native_near_patch_used_by_contour_full_pose_objective": False,
            "objective_limitation": (
                "Each prior-assisted cell receives an isolated regular TerrainMap with its prior-centred Swiss "
                "heights fused onto that grid. Contour full-pose objectives do not consume the native 2 m mesh. "
                "Horizon and Render-PnP also receive the prior-centred native patch; render frames record whether "
                "it contributed visible z-buffer pixels and its effective mesh spacing."
            ),
            "cache_inventory": default_terrain_cache_inventory(),
        },
        "extractor": config.extractor,
        "render_matching": {
            "matcher": config.render_matcher,
            "matcher_id": config.matcher_id if config.render_matcher == "worker" else config.render_matcher,
            "model_manifest": config.matcher_manifest_path,
            "correspondence_cache": _matcher_cache_provenance(config),
            "modality": config.render_modality,
            "resolution_px": [config.render_width_px, config.render_height_px],
            "yaw_step_deg": config.render_yaw_step_deg,
            "refinement_passes": config.render_refinement_passes,
            "native_patch_stride": config.native_patch_stride,
            "candidate_validation": config_record(config)["render_candidate_validation"],
            "orthophoto": {
                "cache": config.orthophoto_cache_dir,
                "zoom": config.orthophoto_zoom,
                "time": config.orthophoto_time,
                "network_allowed_during_benchmark": False,
            },
            "projection_policy": (
                "pinhole candidate renders; query correspondences scored through exact cyltan projection"
            ),
        },
        "compatibility": compatibility_contract(),
        "scoring": {
            "success": {
                "horizontal_position_error_m_lte": 100.0,
                "absolute_yaw_error_deg_lte": 5.0,
            },
            "pitch": "not scored: GeoPose crops contain an unknown global vertical offset",
            "primary_ranking": (
                "published MANUAL references in MAP_A/MAP_B; exact-reference retention cells "
                "and disclosed leakage excluded"
            ),
            "solver_errors_count_as_failures": True,
        },
        "limitations": [
            "No-prior means within the supplied regional DEM window, not country-scale image retrieval.",
            "Global currently assumes a neutral eye height and pitch and is excluded from primary ranking.",
            (
                "The raw_metadata prior regime copies the exact published/refined reference and is "
                "retention-only; original noisy Flickr metadata is diagnostic-only."
            ),
            "Render-PnP remains ranking-excluded until a provisioned learned matcher and terrain appearance pass "
            "the declared real-data validation gate.",
        ],
    }


def _matcher_cache_provenance(config: MatrixConfig) -> dict[str, Any]:
    configured = config.matcher_cache_dir
    resolved: Path | None = None
    if configured is not None:
        resolved = Path(configured).expanduser()
        if not resolved.is_absolute():
            resolved = BASE / resolved
        resolved = resolved.resolve()
    return {
        "enabled": configured is not None,
        "configured_path": configured,
        "resolved_path": str(resolved) if resolved is not None else None,
        "entry_schema": CORRESPONDENCE_CACHE_ENTRY_SCHEMA,
        "key_schema": CORRESPONDENCE_CACHE_KEY_SCHEMA,
        "mode": "read_through_atomic_content_addressed" if configured is not None else "disabled",
        "corrupt_entry_policy": "recompute_and_record_reason",
        "network_allowed_during_inference": False,
    }


def _code_provenance(implementation_paths: list[Path]) -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    diff = _git("diff", "--binary", "HEAD")
    return {
        "git_sha": sha,
        "dirty": bool(status) if status is not None else None,
        "git_status_sha256": hashlib.sha256((status or "").encode()).hexdigest(),
        "git_diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
        "implementation": source_implementation_fingerprint(implementation_paths),
    }


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(("git", *args), cwd=BASE, check=True, capture_output=True, text=True).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
