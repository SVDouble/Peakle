"""Re-tier GT v2 using photo-edge support: if the ground truth is clean, things should align.

A sample whose GT skyline has weak photo-edge support is photo-inconsistent — the dataset's own
render disagrees with the photograph (label pose error, registration, scenery change) — and must
not grade solvers/extractors even if our DEM reproduces the render perfectly.  Run AFTER
scripts/photo_support_batch.py.

Gates (initial; tighten as the corpus grows):
  gt_sky support < 0.50                    -> SUSPECT "photo-inconsistent skyline"
  gt_occ support < 0.20 with density >=0.3 -> SUSPECT "photo-inconsistent occlusions"

Usage: python scripts/apply_support_gate.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GTV2 = BASE / "local/derived/gt_v2"

SKY_MIN = 0.50
OCC_MIN = 0.20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index = json.loads((GTV2 / "index.json").read_text())
    changed = 0
    for rec in index:
        sup_path = GTV2 / "layers" / rec["name"] / "support.json"
        if not sup_path.exists():
            continue
        sup = json.loads(sup_path.read_text())
        reasons = [r for r in rec.get("reasons", []) if not r.startswith("photo-inconsistent")]
        sky = sup.get("gt_sky")
        occ = sup.get("gt_occ")
        if sky is not None and sky < SKY_MIN:
            reasons.append(f"photo-inconsistent skyline (support {sky:.2f} < {SKY_MIN})")
        if occ is not None and occ < OCC_MIN and rec.get("gt_contour_density", 0) >= 0.3:
            reasons.append(f"photo-inconsistent occlusions (support {occ:.2f} < {OCC_MIN})")
        quality = "CLEAN" if not reasons else "SUSPECT"
        if quality != rec["quality"] or reasons != rec.get("reasons", []):
            changed += 1
            if not args.dry_run:
                rec["quality"], rec["reasons"] = quality, reasons
                (GTV2 / f"{rec['name']}.json").write_text(json.dumps(rec, indent=1))
    if not args.dry_run:
        (GTV2 / "index.json").write_text(json.dumps(index, indent=1))
    clean = sum(1 for r in index if r["quality"] == "CLEAN")
    print(f"{'would change' if args.dry_run else 'changed'} {changed} records; CLEAN {clean}/{len(index)}")


if __name__ == "__main__":
    main()
