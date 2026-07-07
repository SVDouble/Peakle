"""CLI: re-tier GT v2 by photo-edge support (logic in peakle.localize.gtquality).

A sample whose GT skyline/occlusions have weak photo-edge support is photo-inconsistent — the
dataset's own render disagrees with the photograph — and must not grade solvers. Run AFTER
peakle.scripts.photo_support_batch.

Usage: python -m peakle.scripts.apply_support_gate [--dry-run]
"""

from __future__ import annotations

import argparse

from peakle.localize.gtquality import apply_support_gate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    changed, clean, total = apply_support_gate(dry_run=args.dry_run)
    print(f"{'would change' if args.dry_run else 'changed'} {changed} records; CLEAN {clean}/{total}")


if __name__ == "__main__":
    main()
