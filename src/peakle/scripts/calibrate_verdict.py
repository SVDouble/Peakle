"""CLI: calibrate the solve verdict against benchmark outcomes (logic in peakle.localize.gtquality).

Reads bench_geopose results.json file(s) and reports, with numbers not vibes: which diagnostics
separate correct from wrong solves (per-feature AUC), the current gate's precision/recall, and the
max-recall thresholds that keep ~100% precision. Scored on MANUAL-GT samples by default.

Usage: python -m peakle.scripts.calibrate_verdict local/output/<dt>-geopose-bench/results.json [more.json]
       [--include-auto]
"""

from __future__ import annotations

import argparse

from peakle.localize.gtquality import best_precision_gate, diagnostic_aucs, load_bench_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--include-auto", action="store_true")
    args = ap.parse_args()

    rows = load_bench_rows(args.results, args.include_auto)
    good = [r for r in rows if r["correct"]]
    print(f"{len(rows)} solves ({len(good)} correct, {len(rows) - len(good)} wrong) from {len(args.results)} file(s)\n")

    print(f"{'diagnostic':12} {'med correct':>12} {'med wrong':>12} {'AUC':>6}   (AUC ~0.5 = useless)")
    for key, med_ok, med_bad, a in diagnostic_aucs(rows):
        print(f"{key:12} {med_ok:12.3f} {med_bad:12.3f} {a:6.2f}")

    conf = [r for r in rows if r["verdict"] == "CONFIRMED"]
    n_ok = sum(1 for r in conf if r["correct"])
    print(
        f"\ncurrent verdict: CONFIRMED on {len(conf)} solves, {n_ok} correct -> precision {n_ok / len(conf):.2f}"
        if conf
        else "\ncurrent verdict: nothing CONFIRMED"
    )
    lost = sum(1 for r in good if r["verdict"] != "CONFIRMED")
    print(f"correct solves not confirmed (lost recall): {lost}/{len(good)}")

    best = best_precision_gate(rows)
    if best:
        print(
            f"\nbest 100%-precision gate on this data (n={best['n']} confirmed, recall {best['recall']:.0%}):\n"
            f"  CONFIRMED iff alias_ratio >= {best['alias']} AND well_width <= {best['well']} AND "
            f"chamfer <= {best['chamfer']}px AND coverage >= {best['coverage']} AND snr >= {best['snr']}"
        )
        print(
            "NB: thresholds chosen on this benchmark; re-run when the benchmark grows. "
            "Precision=1.0 here does NOT guarantee 1.0 in the field."
        )
    else:
        print("\nno threshold combination reaches 100% precision — inspect false positives first")


if __name__ == "__main__":
    main()
