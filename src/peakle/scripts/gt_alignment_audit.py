"""CLI: audit GeoPose GT-v2 alignment against photo and DEM evidence.

The web endpoint is useful for the UI, but corpus trust needs a reproducible artifact:
JSON for tooling, CSV for sorting, and Markdown for human review.

Usage:
  python -m peakle.scripts.gt_alignment_audit
  python -m peakle.scripts.gt_alignment_audit --metric-missing --swiss-check 50
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from peakle.domain.coordinates import GeoPoint, LocalFrame, LocalPoint
from peakle.localize.copdem import load_cop_around
from peakle.localize.geopose import load_sample
from peakle.localize.gtquality import alignment_audit, metric_skyline_errors_for_record, skyline_vertical_error_stats_m
from peakle.localize.gtrefine import crop_az_deg, dem_skyline_with_range
from peakle.localize.paths import BASE, COP_TILES_DIR, GEOPOSE_DIR, GTV2_DIR
from peakle.localize.swissdem import in_switzerland, load_swiss_patch

SWISS_DIR = BASE / "local/data/swissalti"
OUTPUT_DIR = BASE / "local/output"
METER_FIELDS = (
    "sky_error_m",
    "sky_error_median_m",
    "sky_error_p90_m",
    "pfm_error_m",
    "pfm_error_median_m",
    "pfm_error_p90_m",
    "sky_range_median_m",
)
SUPPORT_FIELDS = (
    "pfm_sky",
    "gt_sky",
    "gt_occ",
    "gt_rib",
    "gt_cou",
    "dem_sky",
    "dem_occ",
    "dem_rib",
    "dem_cou",
)
METRIC_CACHE_VERSION = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=GEOPOSE_DIR)
    parser.add_argument("--tiles-dir", type=Path, default=COP_TILES_DIR)
    parser.add_argument("--gt-dir", type=Path, default=GTV2_DIR)
    parser.add_argument("--swiss-dir", type=Path, default=SWISS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--metric-cache",
        type=Path,
        default=None,
        help="JSON cache for expensive meter-space skyline errors.",
    )
    parser.add_argument("--limit", type=int, default=80, help="Rows shown in the Markdown top table.")
    parser.add_argument("--include-clean", action="store_true", help="Include clean rows in audit.json rows.")
    parser.add_argument("--metric-missing", action="store_true", help="Compute meter errors for records missing them.")
    parser.add_argument("--refresh-metric", action="store_true", help="Recompute meter errors for every record.")
    parser.add_argument(
        "--swiss-check",
        type=int,
        default=0,
        help="Run swissALTI3D sensitivity on the N worst Swiss records with cached coverage.",
    )
    args = parser.parse_args()

    out = args.out or OUTPUT_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-gt-alignment-audit"
    out.mkdir(parents=True, exist_ok=True)

    records = _load_records(args.gt_dir)
    records = _enrich_support(records, args.gt_dir)
    if args.metric_missing or args.refresh_metric:
        metric_cache = args.metric_cache or args.gt_dir / "audit" / "metric_errors.json"
        records = _enrich_meter_metrics(
            records,
            args.data_dir,
            args.tiles_dir,
            args.gt_dir,
            metric_cache,
            refresh=args.refresh_metric,
        )

    report = alignment_audit(records, limit=len(records), include_clean=True)
    filtered = alignment_audit(records, limit=len(records), include_clean=args.include_clean)
    swiss_rows = _swiss_sensitivity(records, args) if args.swiss_check > 0 else []

    _write_json(out / "audit.json", filtered)
    _write_csv(out / "audit.csv", report["rows"])
    if swiss_rows:
        _write_csv(out / "swiss_sensitivity.csv", swiss_rows)
    _write_markdown(out / "summary.md", records, report, filtered, swiss_rows, args.limit)
    print(f"wrote {out}")


def _load_records(gt_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(gt_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        records.append(json.loads(path.read_text()))
    if not records:
        raise SystemExit(f"no GT-v2 records under {gt_dir}")
    return records


def _enrich_support(records: list[dict[str, Any]], gt_dir: Path) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        row = dict(rec)
        support_path = gt_dir / "layers" / str(rec["name"]) / "support.json"
        if support_path.exists():
            support = json.loads(support_path.read_text())
            for key in SUPPORT_FIELDS:
                row[f"support_{key}"] = support.get(key)
        out.append(row)
    return out


def _enrich_meter_metrics(
    records: list[dict[str, Any]],
    data_dir: Path,
    tiles_dir: Path,
    gt_dir: Path,
    cache_path: Path,
    *,
    refresh: bool,
) -> list[dict[str, Any]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    enriched = []
    for index, rec in enumerate(records, 1):
        name = str(rec["name"])
        cached = cache.get(name)
        if (
            not refresh
            and isinstance(cached, dict)
            and cached.get("_version") == METRIC_CACHE_VERSION
            and all(cached.get(field) is not None for field in METER_FIELDS)
        ):
            row = dict(rec)
            row.update({field: cached.get(field) for field in METER_FIELDS})
            enriched.append(row)
            continue
        needs = refresh or any(rec.get(field) is None for field in METER_FIELDS)
        if not needs:
            enriched.append(rec)
            continue
        try:
            metrics = metric_skyline_errors_for_record(rec, data_dir=data_dir, tiles_dir=tiles_dir, out_dir=gt_dir)
        except Exception as exc:  # noqa: BLE001 - this is an audit; keep going and surface the failure
            row = dict(rec)
            row["metric_error"] = str(exc)
        else:
            row = dict(rec)
            row.update(metrics)
            cache[name] = metrics | {"_version": METRIC_CACHE_VERSION}
            cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
        enriched.append(row)
        print(f"metric {index}/{len(records)} {name}", flush=True)
    return enriched


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    flat = [_flatten_row(row) for row in rows]
    fields = sorted({key for row in flat for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if key == "metrics" and isinstance(value, dict):
            flat.update(value)
        elif key == "failure_modes" and isinstance(value, list):
            flat["failure_modes"] = ";".join(str(mode.get("code")) for mode in value)
        elif key == "reasons" and isinstance(value, list):
            flat["reasons"] = ";".join(str(reason) for reason in value)
        elif isinstance(value, (str, int, float)) or value is None:
            flat[key] = value
    return flat


def _swiss_sensitivity(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ranked = alignment_audit(records, limit=len(records), include_clean=True)["rows"]
    rows = []
    checked = 0
    for audit_row in ranked:
        if checked >= args.swiss_check:
            break
        rec = next(record for record in records if record["name"] == audit_row["name"])
        sample = load_sample(args.data_dir / str(rec["name"]))
        frame = LocalFrame(origin=GeoPoint(latitude_deg=sample.lat, longitude_deg=sample.lon, elevation_m=0.0))
        cam_geo = frame.local_to_geo(
            LocalPoint(east_m=float(rec.get("de_m") or 0.0), north_m=float(rec.get("dn_m") or 0.0), up_m=0.0)
        )
        if not in_switzerland(cam_geo.latitude_deg, cam_geo.longitude_deg):
            continue
        checked += 1
        rows.append(
            _swiss_sensitivity_one(
                rec,
                sample_lat=sample.lat,
                sample_lon=sample.lon,
                cam_lat=cam_geo.latitude_deg,
                cam_lon=cam_geo.longitude_deg,
                data_dir=args.data_dir,
                tiles_dir=args.tiles_dir,
                gt_dir=args.gt_dir,
                swiss_dir=args.swiss_dir,
            )
        )
        print(f"swiss {checked}/{args.swiss_check} {rec['name']}: {rows[-1]['status']}", flush=True)
    return rows


def _swiss_sensitivity_one(
    rec: dict[str, Any],
    *,
    sample_lat: float,
    sample_lon: float,
    cam_lat: float,
    cam_lon: float,
    data_dir: Path,
    tiles_dir: Path,
    gt_dir: Path,
    swiss_dir: Path,
) -> dict[str, Any]:
    name = str(rec["name"])
    base = {
        "name": name,
        "quality": rec.get("quality"),
        "cop_sky_error_m": rec.get("sky_error_m"),
        "cop_pfm_error_m": rec.get("pfm_error_m"),
        "cam_lat": round(cam_lat, 6),
        "cam_lon": round(cam_lon, 6),
    }
    try:
        patch = load_swiss_patch(swiss_dir, cam_lat, cam_lon, res=5.0, radius_m=12_000.0)
        if patch is None:
            return base | {"status": "no_cached_swissalti_patch"}
        # load_swiss_patch returns coordinates relative to the patch center. The Copernicus terrain
        # and GT arrays are in the sample-local frame, so translate the patch into that frame.
        patch.x_m = patch.x_m + float(rec.get("de_m") or 0.0)
        patch.y_m = patch.y_m + float(rec.get("dn_m") or 0.0)
        terrain = load_cop_around(tiles_dir, sample_lat, sample_lon, extent_m=90_000.0, grid=3000)
        arrays = np.load(gt_dir / f"{name}.npz")
        obs = arrays["gt_skyline"].astype(float)
        pfm = arrays["pfm_skyline"].astype(float) if "pfm_skyline" in arrays.files else obs
        cop = arrays["dem_skyline"].astype(float)
        az = crop_az_deg(int(rec["width"]), float(rec["fov_deg"]), float(rec["yaw_deg"]))
        rows, ranges = dem_skyline_with_range(
            terrain,
            float(rec["cam_z_m"]),
            az,
            int(rec["width"]),
            int(rec["height"]),
            float(rec["fov_deg"]),
            float(rec.get("de_m") or 0.0),
            float(rec.get("dn_m") or 0.0),
            patch=patch,
        )
        swiss = _apply_vertical_terms(rows, rec)
        obs_stats = skyline_vertical_error_stats_m(
            obs, swiss, ranges, int(rec["width"]), int(rec["height"]), float(rec["fov_deg"])
        )
        pfm_stats = skyline_vertical_error_stats_m(
            pfm, swiss, ranges, int(rec["width"]), int(rec["height"]), float(rec["fov_deg"])
        )
        delta = _finite_abs_median(swiss - cop)
    except Exception as exc:  # noqa: BLE001 - audit should produce a row for failure
        return base | {"status": "error", "error": str(exc)[:200]}
    return base | {
        "status": "ok",
        "swiss_sky_error_m": obs_stats["mean_m"],
        "swiss_sky_error_p90_m": obs_stats["p90_m"],
        "swiss_pfm_error_m": pfm_stats["mean_m"],
        "swiss_pfm_error_p90_m": pfm_stats["p90_m"],
        "swiss_range_median_m": obs_stats["range_median_m"],
        "swiss_vs_cop_median_px": delta,
        "sky_error_delta_m": _delta(rec.get("sky_error_m"), obs_stats["mean_m"]),
        "pfm_error_delta_m": _delta(rec.get("pfm_error_m"), pfm_stats["mean_m"]),
    }


def _apply_vertical_terms(rows: np.ndarray, rec: dict[str, Any]) -> np.ndarray:
    cols = np.arange(int(rec["width"]), dtype=float) - (int(rec["width"]) - 1) / 2.0
    return rows + float(rec.get("dv_px") or 0.0) + math.tan(math.radians(float(rec.get("tilt_deg") or 0.0))) * cols


def _finite_abs_median(values: np.ndarray) -> float | None:
    finite = np.abs(values[np.isfinite(values)])
    return round(float(np.median(finite)), 2) if finite.size else None


def _delta(before: Any, after: Any) -> float | None:
    try:
        if before is None or after is None:
            return None
        return round(float(before) - float(after), 1)
    except TypeError, ValueError:
        return None


def _write_markdown(
    path: Path,
    records: list[dict[str, Any]],
    all_report: dict[str, Any],
    filtered_report: dict[str, Any],
    swiss_rows: list[dict[str, Any]],
    limit: int,
) -> None:
    rows = all_report["rows"]
    mode_counts = all_report["mode_counts"]
    obs_counts = Counter(str(row.get("obs_source")) for row in records)
    support = _support_summary(records)
    metric_summary = _metric_summary(records)
    top = [row for row in rows if row["failure_modes"]][:limit]
    lines = [
        "# GT Alignment Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Corpus",
        "",
        f"- Total records: {all_report['total']}",
        f"- CLEAN: {all_report['clean']}",
        f"- SUSPECT: {all_report['suspect']}",
        f"- Observation source: {_counts(obs_counts)}",
        "",
        "## Metric Distributions",
        "",
        _metric_table(metric_summary),
        "",
        "## Failure Modes",
        "",
        _mode_table(mode_counts),
        "",
        "## Photo-Edge Support",
        "",
        _support_table(support),
        "",
        "## Worst Records",
        "",
        _top_table(top),
    ]
    if swiss_rows:
        lines.extend(["", "## swissALTI3D Sensitivity", "", _swiss_table(swiss_rows)])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A low pixel error but high meter error means the mismatch is small on screen but physically large",
            "  because the responsible horizon is far away.",
            "- A high DEM-vs-PFM error with strong PFM photo support points at terrain/source limitations",
            "  or pose labels, not a photo extractor failure.",
            "- Weak GT/PFM photo support means the dataset depth/render line may not correspond to an edge",
            "  visible in the photo.",
            "- If swissALTI3D reduces the error, the sample is probably usable with a better terrain source.",
            "  If it does not, the pose/camera/photo evidence itself is suspect.",
            "",
            f"Rows returned by the filtered JSON: {len(filtered_report['rows'])}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _metric_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    keys = (
        "sky_cons_px",
        "pfm_cons_px",
        "contour_cons_px",
        "sky_error_m",
        "sky_error_p90_m",
        "pfm_error_m",
        "pfm_error_p90_m",
        "pfm_offset_px",
    )
    out = {}
    for key in keys:
        values = np.asarray([float(rec[key]) for rec in records if _is_number(rec.get(key))], dtype=float)
        if values.size:
            out[key] = {
                "n": int(values.size),
                "p50": round(float(np.percentile(values, 50)), 2),
                "p75": round(float(np.percentile(values, 75)), 2),
                "p90": round(float(np.percentile(values, 90)), 2),
                "max": round(float(np.max(values)), 2),
            }
    return out


def _support_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    out = {}
    for key in SUPPORT_FIELDS:
        field = f"support_{key}"
        values = np.asarray([float(rec[field]) for rec in records if _is_number(rec.get(field))], dtype=float)
        if values.size:
            out[key] = {
                "n": int(values.size),
                "p10": round(float(np.percentile(values, 10)), 3),
                "p50": round(float(np.percentile(values, 50)), 3),
                "weak_lt_0_5": int((values < 0.5).sum()),
            }
    return out


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except TypeError, ValueError:
        return False


def _metric_table(summary: dict[str, dict[str, float | int]]) -> str:
    lines = ["| metric | n | p50 | p75 | p90 | max |", "|---|---:|---:|---:|---:|---:|"]
    for key, stats in summary.items():
        lines.append(f"| `{key}` | {stats['n']} | {stats['p50']} | {stats['p75']} | {stats['p90']} | {stats['max']} |")
    return "\n".join(lines)


def _mode_table(counts: dict[str, int]) -> str:
    lines = ["| failure mode | count |", "|---|---:|"]
    for mode, count in counts.items():
        lines.append(f"| `{mode}` | {count} |")
    return "\n".join(lines) if counts else "No failures classified."


def _support_table(summary: dict[str, dict[str, float | int]]) -> str:
    lines = ["| line family | n | p10 | median | weak < 0.5 |", "|---|---:|---:|---:|---:|"]
    for key, stats in summary.items():
        lines.append(f"| `{key}` | {stats['n']} | {stats['p10']} | {stats['p50']} | {stats['weak_lt_0_5']} |")
    return "\n".join(lines)


def _top_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| rank | sample | quality | source | severity | failure modes | sky px | sky m | pfm px | pfm m |",
        "|---:|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        m = row["metrics"]
        modes = ", ".join(mode["code"] for mode in row["failure_modes"])
        lines.append(
            f"| {rank} | `{row['name']}` | {row.get('quality')} | {row.get('obs_source')} | "
            f"{row['severity']} | {modes} | {_fmt(m.get('sky_cons_px'))} | {_fmt(m.get('sky_error_m'))} | "
            f"{_fmt(m.get('pfm_cons_px'))} | {_fmt(m.get('pfm_error_m'))} |"
        )
    return "\n".join(lines) if rows else "No rows exceed failure thresholds."


def _swiss_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| sample | status | Cop sky m | Swiss sky m | delta m | Swiss-vs-Cop px |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['status']} | {_fmt(row.get('cop_sky_error_m'))} | "
            f"{_fmt(row.get('swiss_sky_error_m'))} | {_fmt(row.get('sky_error_delta_m'))} | "
            f"{_fmt(row.get('swiss_vs_cop_median_px'))} |"
        )
    return "\n".join(lines)


def _counts(counts: Counter[str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


if __name__ == "__main__":
    main()
