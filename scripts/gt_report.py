"""Ground-truth dataset & scoring report generator.

Produces the figures + JSON digest that docs/the GT report artifact embeds:
  1. corpus statistics over ALL downloaded GeoPose3K samples (positions, FOV, yaw, pitch, roll);
  2. cleanliness audit of the pinned bench-60 (gt-consistency, camera-below-ground, solver
     consensus vs GT label);
  3. worked scoring examples on real samples: capped curve chamfer at right/wrong yaw, the 360°
     yaw profile with basin + alias rival, the terrain self-similarity (SNR) scan, and the
     multi-hypothesis extraction arbitration.

Usage: python scripts/gt_report.py <bench_results.json> [--out local/output/<dt>-gt-report]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from peakle.localize.copdem import load_cop_around
from peakle.localize.extract import extract_candidates
from peakle.localize.geopose import load_sample, oracle_skyline, read_pfm
from peakle.localize.solve import HorizonProfile, curve_chamfer, solve_orientation

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "local/data/geopose"
TILES = BASE / "local/data/copernicus"
MAX_W = 1152

# validated reference palette (dataviz skill) — light surface
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, SURF, AXIS = "#e1e0d9", "#fcfcfb", "#c3c2b7"
BLUE, AQUA, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
GOOD, CRIT = "#0ca30c", "#d03b3b"

plt.rcParams.update(
    {
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": SEC,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUT,
        "ytick.color": MUT,
        "text.color": INK,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("  ->", path.name)


# ---------------------------------------------------------------- corpus stats
def corpus_stats(out: Path) -> dict:
    samples = []
    for d in sorted(DATA.iterdir()):
        if (d / "info.txt").exists():
            try:
                samples.append(load_sample(d))
            except Exception:
                pass
    man = [s for s in samples if s.manual]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for sel, color, label in [(man, BLUE, f"MANUAL ({len(man)})"), ([s for s in samples if not s.manual], AQUA, f"AUTO ({len(samples)-len(man)})")]:
        ax.scatter([s.lon for s in sel], [s.lat for s in sel], s=14, c=color, alpha=0.75, label=label, edgecolors="none")
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    ax.set_title("Viewpoint positions — all downloaded samples")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out / "corpus_map.png")

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 2.9))
    for ax, vals, title, unit in [
        (axes[0], [s.fov_deg for s in samples], "horizontal FOV", "deg"),
        (axes[1], [s.yaw_gt_deg for s in samples], "GT yaw (azimuth)", "deg"),
        (axes[2], [s.pitch_gt_deg for s in samples], "GT pitch", "deg"),
        (axes[3], [abs(s.roll_gt_deg) for s in samples], "GT |roll|", "deg"),
    ]:
        ax.hist(vals, bins=28, color=BLUE, edgecolor=SURF, linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel(unit)
    axes[0].set_ylabel("samples")
    _save(fig, out / "corpus_hists.png")

    r = np.abs([s.roll_gt_deg for s in samples])
    return {
        "n_total": len(samples),
        "n_manual": len(man),
        "fov_range": [round(float(np.percentile([s.fov_deg for s in samples], p)), 1) for p in (5, 50, 95)],
        "elev_range": [round(float(np.percentile([s.elev_m for s in samples], p))) for p in (5, 50, 95)],
        "roll_med_p90": [round(float(np.median(r)), 1), round(float(np.percentile(r, 90)), 1)],
        "lat_range": [round(min(s.lat for s in samples), 2), round(max(s.lat for s in samples), 2)],
        "lon_range": [round(min(s.lon for s in samples), 2), round(max(s.lon for s in samples), 2)],
    }


# ---------------------------------------------------------- bench-level charts
def bench_charts(rows: list[dict], out: Path) -> dict:
    ok_ch, bad_ch = [], []
    for r in rows:
        for t in ("oracle", "extracted"):
            s = r.get(t)
            if isinstance(s, dict) and "chamfer_px" in s:
                (ok_ch if s["correct"] else bad_ch).append(s["chamfer_px"])

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    rng = np.random.default_rng(0)
    ax.scatter(ok_ch, 1 + rng.uniform(-0.16, 0.16, len(ok_ch)), s=26, c=GOOD, alpha=0.8, edgecolors="none", label=f"correct ({len(ok_ch)})")
    ax.scatter(bad_ch, 0 + rng.uniform(-0.16, 0.16, len(bad_ch)), s=30, c=CRIT, alpha=0.8, marker="x", label=f"wrong ({len(bad_ch)})")
    ax.set_yticks([0, 1], ["wrong\n(|yaw err| > 5°)", "correct"])
    ax.set_xlabel("final chamfer residual (px)")
    ax.set_title("The residual cannot judge correctness — the ranges overlap")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(-0.5, 1.5)
    _save(fig, out / "chamfer_overlap.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 2.9))
    gc = [r["gt_consistency_px"] for r in rows if "gt_consistency_px" in r]
    axes[0].hist(gc, bins=24, color=BLUE, edgecolor=SURF, linewidth=0.6)
    axes[0].set_title("GT consistency: GT-depth skyline vs our DEM @ GT yaw")
    axes[0].set_xlabel("chamfer (px), vertical shift free")
    axes[0].set_ylabel("samples")
    ag = [r["alt_above_ground"] for r in rows if "alt_above_ground" in r]
    axes[1].hist(ag, bins=24, color=BLUE, edgecolor=SURF, linewidth=0.6)
    axes[1].set_title("GT camera altitude minus DEM ground")
    axes[1].set_xlabel("m (negative = camera below terrain)")
    ee = [r["extraction_err_px"] for r in rows if r.get("extraction_err_px") is not None]
    axes[2].hist(ee, bins=24, color=BLUE, edgecolor=SURF, linewidth=0.6)
    axes[2].set_title("Extraction error vs GT skyline (winning candidate)")
    axes[2].set_xlabel("median |Δrow| (px)")
    _save(fig, out / "cleanliness_hists.png")

    # per-feature AUC (from calibrate_verdict logic, inlined for the figure)
    feats = {"alias_ratio": False, "snr": False, "chamfer_px": True, "well_width_deg": True, "coverage": False}
    solves = [s for r in rows for s in (r.get("oracle"), r.get("extracted")) if isinstance(s, dict) and "chamfer_px" in s]
    aucs = {}
    for k, invert in feats.items():
        g = np.array([s[k] for s in solves if s["correct"] and np.isfinite(s.get(k, np.nan))])
        b = np.array([s[k] for s in solves if not s["correct"] and np.isfinite(s.get(k, np.nan))])
        a = float((g[:, None] > b[None, :]).mean() + 0.5 * (g[:, None] == b[None, :]).mean())
        aucs[k] = round(1 - a if invert else a, 2)
    order = sorted(aucs, key=aucs.get)
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    ax.barh(range(len(order)), [aucs[k] for k in order], color=BLUE, height=0.55)
    ax.set_yticks(range(len(order)), order)
    ax.axvline(0.5, color=AXIS, linewidth=1, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC — does the diagnostic separate correct from wrong?  (0.5 = useless)")
    for i, k in enumerate(order):
        ax.text(aucs[k] + 0.01, i, f"{aucs[k]:.2f}", va="center", color=SEC, fontsize=10)
    ax.set_title("Diagnostic power of the no-GT solve signals")
    _save(fig, out / "diagnostic_auc.png")
    return {"aucs": aucs, "gt_consistency_median": round(float(np.median(gc)), 1)}


# -------------------------------------------------------------- audit table
def build_audit(rows: list[dict], results_path: Path, out: Path) -> list[dict]:
    """Flag suspect samples: bad GT↔DEM agreement, buried camera, or a solver consensus that
    contradicts the label WITHOUT the DEM contradicting it (that combination = solver failure,
    not GT dirt — it is excluded here)."""

    def ang(a):
        return (a + 180) % 360 - 180

    audit = []
    for r in rows:
        o, e = r.get("oracle", {}), r.get("extracted", {})
        flags = []
        if r.get("gt_consistency_px", 0) > 25:
            flags.append(f"GT↔DEM mismatch {r['gt_consistency_px']:.0f}px")
        if r.get("alt_above_ground", 0) < -20:
            flags.append(f"camera {-r['alt_above_ground']:.0f}m below DEM ground")
        oy, ey = o.get("yaw_err"), e.get("yaw_err")
        if (
            isinstance(oy, (int, float)) and isinstance(ey, (int, float)) and np.isfinite(ey)
            and abs(oy) > 20 and abs(ang(oy - ey)) < 10 and r.get("gt_consistency_px", 0) > 25
        ):
            flags.append(f"solver consensus at {oy:+.0f}° from GT label")
        if flags:
            audit.append({"name": r["name"], "manual": r["manual"], "flags": flags,
                          "gt_consistency": r.get("gt_consistency_px"), "oracle_err": oy})
    audit.sort(key=lambda a: -(a["gt_consistency"] or 0))
    overlays = results_path.parent / "overlays"
    for a in audit[:8]:
        src = overlays / f"{a['name']}.jpg"
        if src.exists():
            im = Image.open(src)
            im.thumbnail((560, 560))
            im.save(out / f"audit_{a['name']}.jpg", quality=78)
    (out / "audit.json").write_text(json.dumps(audit, indent=1))
    return audit


# ------------------------------------------------------------- worked examples
def _std_rgb(sample) -> np.ndarray:
    rgb = np.asarray(Image.open(sample.photo_path).convert("RGB"), np.uint8)
    if rgb.shape[1] > MAX_W:
        s = MAX_W / rgb.shape[1]
        rgb = np.asarray(Image.fromarray(rgb).resize((MAX_W, round(rgb.shape[0] * s)), Image.BILINEAR), np.uint8)
    return rgb


def _oracle_std(sample, w: int, h: int) -> np.ndarray:
    o = oracle_skyline(sample.depth_path)
    dep_h = read_pfm(sample.depth_path).shape[0]
    x = np.linspace(0, 1, len(o))
    fin = np.isfinite(o)
    out = np.interp(np.linspace(0, 1, w), x[fin], o[fin])
    cov = np.interp(np.linspace(0, 1, w), x, fin.astype(float)) > 0.5
    return np.where(cov, out * (h / dep_h), np.nan)


def _draw_lines(rgb: np.ndarray, layers: list[tuple[np.ndarray, str, int]], header: str = "") -> Image.Image:
    hex2rgb = lambda hx: tuple(int(hx[i : i + 2], 16) for i in (1, 3, 5))
    im = Image.fromarray(np.clip(rgb.astype(float) * 1.15, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    h, w = rgb.shape[:2]
    for rows, color, width in layers:
        pts = [(c, float(rows[c])) for c in range(w) if np.isfinite(rows[c]) and -20 <= rows[c] < h + 20]
        if len(pts) > 1:
            dr.line(pts, fill=hex2rgb(color), width=width)
    if header:
        dr.rectangle([0, 0, w, 20], fill=(11, 11, 11))
        dr.text((6, 4), header, fill=(252, 252, 251))
    return im


def example_anatomy(sample, out: Path) -> None:
    rgb = _std_rgb(sample)
    h, w = rgb.shape[:2]
    Image.fromarray(rgb).save(out / "anatomy_photo.jpg", quality=88)
    depth = read_pfm(sample.depth_path).astype(float)
    depth[depth <= 0] = np.nan
    fig, ax = plt.subplots(figsize=(7.0, 7.0 * h / w))
    ax.imshow(depth / 1000.0, cmap="Blues", origin="upper")
    ax.set_axis_off()
    ax.set_title("GT depth render (km) — sky = empty", fontsize=11)
    _save(fig, out / "anatomy_depth.png")
    orc = _oracle_std(sample, w, h)
    _draw_lines(rgb, [(orc, GOOD, 3)], "oracle skyline = first terrain row of the GT depth").save(
        out / "anatomy_oracle.jpg", quality=88
    )


def example_chamfer(sample, profile, out: Path) -> dict:
    rgb = _std_rgb(sample)
    h, w = rgb.shape[:2]
    obs = _oracle_std(sample, w, h)
    res = {}
    for tag, yaw in [("right", sample.yaw_gt_deg), ("wrong", (sample.yaw_gt_deg + 120) % 360)]:
        best = (np.inf, None)
        for dv in range(-260, 261, 10):
            rows = profile.rows_cyl_tan(w, h, sample.fov_deg, yaw, 0.0) + dv
            c = curve_chamfer(obs, rows, cap=60.0)
            if c < best[0]:
                best = (c, rows)
        cham, rows = best
        img = _draw_lines(rgb, [(obs, GOOD, 3), (rows, YELLOW, 3)],
                          f"DEM @ yaw {yaw:.0f}deg  |  capped symmetric chamfer = {cham:.1f}px")
        img.save(out / f"chamfer_{tag}.jpg", quality=86)
        res[tag] = round(float(cham), 1)
    return res


def example_profile(sample, profile, track_rows, h, tag, out: Path, title: str) -> dict:
    s = solve_orientation(track_rows, h, profile, fov_deg=sample.fov_deg, projection="cyltan")
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    ax.plot(s.yaw_profile_deg, s.yaw_profile_chamfer, color=BLUE, linewidth=1.6)
    cmin = float(np.min(s.yaw_profile_chamfer))
    ax.axhline(cmin * 1.15, color=AXIS, linewidth=1, linestyle="--")
    ax.axvline(sample.yaw_gt_deg, color=GOOD, linewidth=1.4, label=f"GT yaw {sample.yaw_gt_deg:.0f}°")
    ax.axvline(s.yaw_deg, color=VIOLET, linewidth=1.4, linestyle=":", label=f"solved {s.yaw_deg:.0f}°")
    ax.set_xlabel("yaw (deg)")
    ax.set_ylabel("chamfer (px)")
    ax.set_title(f"{title} — well {s.well_width_deg:.0f}°, alias {s.alias_ratio:.2f}, snr {s.snr:.1f} → {s.verdict}")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _save(fig, out / f"profile_{tag}.png")

    fig, ax = plt.subplots(figsize=(8.6, 2.4))
    ax.plot(s.self_profile_deg, s.self_profile_chamfer, color=AQUA, linewidth=1.6)
    ax.axvline(s.yaw_deg, color=VIOLET, linewidth=1.4, linestyle=":")
    ax.set_xlabel("yaw (deg)")
    ax.set_ylabel("chamfer (px)")
    ax.set_title(f"terrain self-similarity: solved DEM window vs the rest of the horizon (min outside basin = {s.terrain_distinct_px:.0f}px)")
    _save(fig, out / f"selfscan_{tag}.png")
    return {
        "yaw": round(s.yaw_deg, 1), "err": round((s.yaw_deg - sample.yaw_gt_deg + 180) % 360 - 180, 1),
        "chamfer": round(s.chamfer_px, 1), "well": s.well_width_deg, "alias": round(s.alias_ratio, 2),
        "snr": round(s.snr, 1), "verdict": s.verdict,
    }


def example_multihyp(sample, profile, out: Path) -> dict:
    rgb = _std_rgb(sample)
    h, w = rgb.shape[:2]
    obs_gt = _oracle_std(sample, w, h)
    cands = extract_candidates(rgb)
    layers = [(obs_gt, GOOD, 2)]
    stats = {}
    for name, color in [("blue", CRIT), ("bright", BLUE)]:
        c = cands[name]
        layers.append((c.rows, color, 3))
        if c.coverage >= 0.25:
            s = solve_orientation(c.rows, h, profile, fov_deg=sample.fov_deg, projection="cyltan")
            stats[name] = {"coverage": round(c.coverage, 2), "chamfer": round(s.chamfer_px, 1),
                           "yaw_err": round((s.yaw_deg - sample.yaw_gt_deg + 180) % 360 - 180, 1), "verdict": s.verdict}
    _draw_lines(rgb, layers, "hypotheses: blue-dominant detector (red) vs bright detector (blue); GT skyline green").save(
        out / "multihyp.jpg", quality=86
    )
    return stats


def load_profile(sample) -> HorizonProfile:
    terrain = load_cop_around(TILES, sample.lat, sample.lon, extent_m=90000.0, grid=3000)
    ground = terrain.elevation_at(0.0, 0.0)
    return HorizonProfile(terrain, max(sample.elev_m, ground + 2.0), step=25.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else BASE / f"local/output/{datetime.now():%Y%m%d-%H%M%S}-gt-report"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.load(open(args.results)) if "error" not in r]
    digest: dict = {"bench_results": args.results, "generated": datetime.now().isoformat(timespec="seconds")}

    print("corpus stats over all downloaded samples...")
    digest["corpus"] = corpus_stats(out)
    print("bench-60 charts...")
    digest["bench"] = bench_charts(rows, out)
    print("audit table...")
    digest["n_flagged"] = len(build_audit(rows, Path(args.results).resolve(), out))

    print("worked examples (re-solving 3 samples)...")
    clean = load_sample(DATA / "eth_ch1_2011-10-04_15:22:47_01024")
    prof_clean = load_profile(clean)
    example_anatomy(clean, out)
    digest["chamfer_demo"] = example_chamfer(clean, prof_clean, out)
    rgbc = _std_rgb(clean)
    digest["profile_clean"] = example_profile(
        clean, prof_clean, _oracle_std(clean, rgbc.shape[1], rgbc.shape[0]), rgbc.shape[0], "clean", out,
        "360° yaw profile — distinctive horizon (oracle skyline)")

    amb = load_sample(DATA / "eth_ch1_24945353_01024")
    prof_amb = load_profile(amb)
    rgba = _std_rgb(amb)
    cand = extract_candidates(rgba)["blue"]
    digest["profile_ambiguous"] = example_profile(
        amb, prof_amb, cand.rows, rgba.shape[0], "ambiguous", out,
        "360° yaw profile — self-similar horizon (extracted skyline)")

    multi = load_sample(DATA / "eth_ch1_60648419_01024")
    digest["multihyp"] = example_multihyp(multi, load_profile(multi), out)

    (out / "digest.json").write_text(json.dumps(digest, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
