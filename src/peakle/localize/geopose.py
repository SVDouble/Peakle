"""GeoPose3K ground-truth access.

Dataset: 3000+ Alpine photos with GT pose and GT-rendered depth
(merlin.fit.vutbr.cz/elevation/geoPose3K_final_publish.tar.gz).  Photos under ``cyl/`` are
CYLINDRICAL crops.  ``info.txt`` layout: line1 = MANUAL|AUTO pose-source flag, line2 = ZYZ Euler
(a, b, g) radians, lines 3-6 = refined lat, lon, elevation m, and horizontal FOV radians.  In
newer archives, optional lines 7-10 retain the corresponding original/noisy metadata.  The
refined values remain the sole pose reference; the original values are label-ambiguity
diagnostics only.

Orientation decode: ``R = Rx(-g) @ Rz(-b) @ Ry(-a) @ P`` with ``P = [[0,1,0],[0,0,1],[1,0,0]]``;
look = R @ [1,0,0]; east = -look[2], north = look[0], up = look[1].
Convention chosen EMPIRICALLY (2026-07-05): brute-forced axis orders/signs against 35 solver-
verified poses; this one has max yaw error 2.5° / mean 0.7° incl. the large-|b| samples where the
previous ``Rz(-g)Rx(-b)Ry(-a)`` decode was off by 4-7° (the two are indistinguishable for small b,
which is how the old one originally "verified").  Note large |b| also predicts a large vertical
crop offset (~1.2·b) — solvers must allow pitch/shift bounds of ±50° on this dataset.

The AUTO flag matters: automatic poses in the dataset are not human-verified and some are simply
wrong — benchmark scoring should be restricted to (or at least split by) MANUAL samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GeoPoseOriginalMetadata:
    """Original/noisy location and FOV retained alongside the refined label."""

    lat: float
    lon: float
    elev_m: float
    fov_deg: float


@dataclass
class GeoPoseSample:
    name: str
    root: Path
    manual: bool  # pose flagged MANUAL (human-verified) vs AUTO
    lat: float
    lon: float
    elev_m: float
    fov_deg: float
    yaw_gt_deg: float  # azimuth, 0 = North, clockwise to East
    pitch_gt_deg: float
    roll_gt_deg: float  # image-plane roll; the solver assumes 0, so large |roll| = harder GT
    original_metadata: GeoPoseOriginalMetadata | None = None

    @property
    def photo_path(self) -> Path:
        return self.root / "cyl" / "photo_crop.jpg"

    @property
    def depth_path(self) -> Path:
        return self.root / "cyl" / "distance_crop.pfm"


def _decode_orientation(a: float, b: float, g: float) -> tuple[float, float, float]:
    def rx(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def ry(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rz(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    perm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    rot = rx(-g) @ rz(-b) @ ry(-a) @ perm
    look = rot @ np.array([1.0, 0.0, 0.0])
    east, north, up = -look[2], look[0], look[1]
    yaw = math.degrees(math.atan2(east, north)) % 360.0
    pitch = math.degrees(math.atan2(up, math.hypot(east, north)))
    # roll: angle of the camera's image-up axis (camera z in this convention: at zero Euler it
    # maps to world up [0,1,0]) around the look direction, relative to true vertical
    img_up = rot @ np.array([0.0, 0.0, 1.0])
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, look)
    right = right / max(np.linalg.norm(right), 1e-9)
    true_up = np.cross(look, right)
    roll = math.degrees(math.atan2(float(img_up @ right), float(img_up @ true_up)))
    return yaw, pitch, roll


def load_sample(sample_dir: str | Path) -> GeoPoseSample:
    root = Path(sample_dir)
    lines = [ln.strip() for ln in (root / "info.txt").read_text().splitlines() if ln.strip()]
    a, b, g = (float(x) for x in lines[1].split())
    yaw, pitch, roll = _decode_orientation(a, b, g)
    original_metadata = (
        GeoPoseOriginalMetadata(
            lat=float(lines[6]),
            lon=float(lines[7]),
            elev_m=float(lines[8]),
            fov_deg=math.degrees(float(lines[9])),
        )
        if len(lines) >= 10
        else None
    )
    return GeoPoseSample(
        name=root.name,
        root=root,
        manual=lines[0].upper().startswith("MANUAL"),
        lat=float(lines[2]),
        lon=float(lines[3]),
        elev_m=float(lines[4]),
        fov_deg=math.degrees(float(lines[5])),
        yaw_gt_deg=yaw,
        pitch_gt_deg=pitch,
        roll_gt_deg=roll,
        original_metadata=original_metadata,
    )


def read_pfm(path: str | Path) -> np.ndarray:
    with open(path, "rb") as f:
        typ = f.readline().decode().strip()
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), "<f4" if scale < 0 else ">f4")
    data = data.reshape(h, w, 3 if typ == "PF" else 1)
    return np.flipud(data[..., 0])


def oracle_skyline(depth_pfm: str | Path) -> np.ndarray:
    """Per-column skyline row from the GT-rendered depth map (sky encoded as <= 0)."""

    depth = read_pfm(depth_pfm)
    return depth_skyline(depth)


def depth_skyline(depth: np.ndarray) -> np.ndarray:
    """Per-column skyline row from a GeoPose depth image (sky encoded as <= 0)."""

    terr = depth > 0
    h, w = terr.shape
    rows = np.full(w, np.nan)
    has = terr.any(axis=0)
    rows[has] = np.argmax(terr[:, has], axis=0).astype(float)
    return rows


def resampled_oracle_skyline(depth_pfm: str | Path, width: int, height: int) -> np.ndarray:
    """GT depth skyline resampled into the working image size."""

    depth = read_pfm(depth_pfm)
    rows = depth_skyline(depth)
    src_x = np.linspace(0.0, 1.0, len(rows))
    finite = np.isfinite(rows)
    if not finite.any():
        return np.full(width, np.nan)
    dst_x = np.linspace(0.0, 1.0, width)
    valid = np.interp(dst_x, src_x, finite.astype(float)) > 0.5
    scaled = np.interp(dst_x, src_x[finite], rows[finite]) * (height / depth.shape[0])
    return np.where(valid, scaled, np.nan)
