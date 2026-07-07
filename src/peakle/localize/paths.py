"""Canonical corpus locations — the single source of truth for the on-disk data layout.

Every module and script that touches the GeoPose3K corpus, the DEM tiles, or the derived GT v2
records imports these instead of recomputing ``Path(__file__).parents[...] / "local/..."`` (which
had drifted into a copy in each script and each of gtbuild/bench/gtquality).
"""

from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parents[3]

GEOPOSE_DIR = BASE / "local/data/geopose"  # raw GeoPose3K samples (photo + depth + info)
COP_TILES_DIR = BASE / "local/data/copernicus"  # Copernicus GLO-30 DEM tiles
SWISS_DIR = BASE / "local/data/swissalti"  # swissALTI3D 2 m patches
GTV2_DIR = BASE / "local/derived/gt_v2"  # refined GT records, npz arrays, layer PNGs
OUTPUT_DIR = BASE / "local/output"  # timestamped run outputs

STD_WIDTH = 1152  # standard solve/build image width; keeps yaw resolution well under 0.1 deg
