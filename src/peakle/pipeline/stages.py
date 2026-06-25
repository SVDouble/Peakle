"""Independent, augmenting pipeline stages and the optimization step.

Each stage reads the `Evidence` and fills one more field, recording what it added.
The default pipeline is a small DAG:

    {exif, contour, depth} -> {intrinsics, position} ; depth -> ridges

Then `localize` consumes whatever is present: a GPS prior routes to a precise
local solve; without one it runs the prior-free global search and returns all
plausible candidates.
"""

from __future__ import annotations

import asyncio
import math
from typing import Protocol

import numpy as np

from peakle.depth import DepthEstimator, estimate_depth
from peakle.domain.coordinates import EARTH_RADIUS_M, LocalPoint
from peakle.domain.pose import PosePrior
from peakle.domain.terrain import TerrainMap
from peakle.edges import EdgeDetector, estimate_edges
from peakle.optimization.solve import PoseSolveResult, solve_pose
from peakle.pipeline.evidence import Evidence, ExifData
from peakle.pipeline.exif import horizontal_fov_deg, intrinsics_from_exif, read_exif
from peakle.rendering.skyline import contour_from_profile
from peakle.segmentation import extract_dp, extract_ridges

DEFAULT_FOV_DEG = 55.0


class Stage(Protocol):
    """A pipeline stage that augments the evidence in place.

    `requires` names the stages that must finish first; stages with disjoint
    requirements run concurrently. Each stage writes its own evidence field, so
    concurrent stages never touch the same attribute.
    """

    name: str
    requires: tuple[str, ...]

    def run(self, evidence: Evidence) -> Evidence: ...


class ExifStage:
    """Reads EXIF camera/GPS metadata."""

    name = "exif"
    requires: tuple[str, ...] = ()

    def run(self, evidence: Evidence) -> Evidence:
        evidence.exif = read_exif(evidence.pil)
        evidence.log(f"exif: {'read' if evidence.exif.present else 'none'}")
        return evidence


class IntrinsicsStage:
    """Derives camera intrinsics (FOV) from EXIF focal length or a default."""

    name = "intrinsics"
    requires: tuple[str, ...] = ("exif",)

    def __init__(self, default_fov_deg: float = DEFAULT_FOV_DEG) -> None:
        self.default_fov_deg = default_fov_deg

    def run(self, evidence: Evidence) -> Evidence:
        exif = evidence.exif or ExifData()
        evidence.intrinsics = intrinsics_from_exif(
            exif, evidence.image_width_px, evidence.image_height_px, self.default_fov_deg
        )
        source = "exif focal length" if horizontal_fov_deg(exif) is not None else "default FOV"
        evidence.log(f"intrinsics: {evidence.intrinsics.horizontal_fov_deg():.0f}deg from {source}")
        return evidence


class PositionPriorStage:
    """Position extraction: turns an EXIF GPS fix into a local pose prior."""

    name = "position"
    requires: tuple[str, ...] = ("exif",)

    def __init__(self, terrain: TerrainMap) -> None:
        self.terrain = terrain

    def run(self, evidence: Evidence) -> Evidence:
        exif = evidence.exif
        if exif and exif.gps_lat_deg is not None and exif.gps_lon_deg is not None:
            evidence.prior = _prior_from_gps(exif, self.terrain)
            evidence.log("position: prior from GPS")
        else:
            evidence.log("position: no GPS, prior-free")
        return evidence


class ContourStage:
    """Extracts the skyline outline from the image (robust DP segmentation)."""

    name = "contour"
    requires: tuple[str, ...] = ()

    def run(self, evidence: Evidence) -> Evidence:
        profile = extract_dp(evidence.image)
        evidence.contour = contour_from_profile(profile, evidence.image_height_px, source="photo")
        evidence.log(f"contour: dp skyline, {len(evidence.contour.points)} pts")
        return evidence


class DepthStage:
    """Estimates a dense relative depth map (classical haze, or a learned model)."""

    name = "depth"
    requires: tuple[str, ...] = ()

    def __init__(self, estimator: DepthEstimator | None = None) -> None:
        self.estimator = estimator

    def run(self, evidence: Evidence) -> Evidence:
        evidence.depth = estimate_depth(evidence.image, self.estimator)
        evidence.log(f"depth: {(self.estimator.name if self.estimator else 'haze')} map")
        return evidence


class EdgeStage:
    """Visual edge layer: learned boundary detection (DexiNed) if a detector is given.

    Independent of depth, so it runs concurrently with it. When no detector is
    supplied the edges are None and ridge extraction uses the classical response.
    """

    name = "edges"
    requires: tuple[str, ...] = ()

    def __init__(self, detector: EdgeDetector | None = None) -> None:
        self.detector = detector

    def run(self, evidence: Evidence) -> Evidence:
        evidence.edges = estimate_edges(evidence.image, self.detector)
        evidence.log(f"edges: {self.detector.name if self.detector else 'none (classical)'}")
        return evidence


class RidgeStage:
    """Extracts the skyline + internal ridge lines (with confidence) from depth + edges."""

    name = "ridges"
    requires: tuple[str, ...] = ("depth", "edges")

    def __init__(self, max_internal: int = 80) -> None:
        self.max_internal = max_internal

    def run(self, evidence: Evidence) -> Evidence:
        evidence.ridges = extract_ridges(
            evidence.image, max_internal=self.max_internal, depth=evidence.depth, edges=evidence.edges
        )
        internal = sum(int(np.isfinite(ridge.rows).any()) for ridge in evidence.ridges.ridges)
        evidence.log(f"ridges: skyline + {internal} internal layers")
        return evidence


class Pipeline:
    """Runs stages as a dependency DAG (async), augmenting one evidence object.

    Stages are grouped into layers by their `requires`; every stage in a layer is
    independent, so when `parallel` is set the layer's stages run concurrently via
    `asyncio.to_thread` (the heavy contour/depth work is numpy/scipy and releases
    the GIL). The metadata branch (exif -> intrinsics/position) and the vision
    branch (contour -> depth) thus overlap.
    """

    def __init__(self, stages: list[Stage], *, parallel: bool = True) -> None:
        self.stages = stages
        self.parallel = parallel

    async def run(self, evidence: Evidence) -> Evidence:
        order = {stage.name: index for index, stage in enumerate(self.stages)}
        remaining = list(self.stages)
        completed: set[str] = set()
        while remaining:
            ready = [stage for stage in remaining if set(stage.requires) <= completed]
            if not ready:
                blocked = ", ".join(stage.name for stage in remaining)
                msg = f"unsatisfiable stage dependencies among: {blocked}"
                raise ValueError(msg)
            if self.parallel and len(ready) > 1:
                await asyncio.gather(*(asyncio.to_thread(stage.run, evidence) for stage in ready))
            else:
                for stage in ready:
                    stage.run(evidence)
            completed.update(stage.name for stage in ready)
            remaining = [stage for stage in remaining if stage.name not in completed]
        # Concurrent stages can append provenance out of order; restore it.
        evidence.provenance.sort(key=lambda line: order.get(line.split(":", 1)[0], len(order)))
        return evidence

    def run_sync(self, evidence: Evidence) -> Evidence:
        """Convenience synchronous runner (drives the async pipeline to completion)."""

        return asyncio.run(self.run(evidence))


def default_pipeline(
    terrain: TerrainMap,
    default_fov_deg: float = DEFAULT_FOV_DEG,
    depth_estimator: DepthEstimator | None = None,
    edge_detector: EdgeDetector | None = None,
) -> Pipeline:
    """Standard DAG: {exif, contour, depth, edges} -> {intrinsics, position, ridges}.

    Pass `depth_estimator=load_learned_depth()` for Depth-Anything (vs classical
    haze) and `edge_detector=load_learned_edges()` for DexiNed boundaries (vs the
    classical depth-aware response).
    """

    return Pipeline(
        [
            ExifStage(),
            IntrinsicsStage(default_fov_deg),
            PositionPriorStage(terrain),
            ContourStage(),
            DepthStage(depth_estimator),
            EdgeStage(edge_detector),
            RidgeStage(),
        ]
    )


def localize(evidence: Evidence, terrain: TerrainMap) -> PoseSolveResult:
    """Optimization step: solve a pose from the accumulated evidence.

    With a GPS-derived prior it runs a precise local solve; without one it runs
    the prior-free global search, whose result carries all plausible candidates.
    """

    if evidence.intrinsics is None or evidence.contour is None:
        msg = "evidence needs intrinsics and a contour before localization"
        raise ValueError(msg)
    if evidence.prior is not None:
        return solve_pose(terrain, evidence.contour, evidence.intrinsics, evidence.prior, strategy="powell")
    return solve_pose(terrain, evidence.contour, evidence.intrinsics, _centered_prior(terrain), strategy="global")


def _prior_from_gps(exif: ExifData, terrain: TerrainMap) -> PosePrior:
    origin = terrain.spec.origin
    east_scale = EARTH_RADIUS_M * math.cos(math.radians(origin.latitude_deg))
    north_m = math.radians(exif.gps_lat_deg - origin.latitude_deg) * EARTH_RADIUS_M
    east_m = math.radians(exif.gps_lon_deg - origin.longitude_deg) * east_scale
    ground = terrain.elevation_at(east_m, north_m)
    up_m = exif.gps_alt_m if exif.gps_alt_m is not None else ground + 2.0
    yaw_deg = ((exif.heading_deg + 180.0) % 360.0) - 180.0 if exif.heading_deg is not None else 0.0
    # GPS position is reliable (~tight), but phone *compass* heading is not — on a
    # real photo (jubigrat) it was ~125deg off — so the heading is only a weak hint
    # and yaw is searched broadly.
    return PosePrior(
        position=LocalPoint(east_m=east_m, north_m=north_m, up_m=up_m),
        yaw_deg=yaw_deg,
        pitch_deg=0.0,
        horizontal_sigma_m=30.0,
        vertical_sigma_m=20.0,
        yaw_sigma_deg=90.0,
        pitch_sigma_deg=15.0,
    )


def _centered_prior(terrain: TerrainMap) -> PosePrior:
    return PosePrior(
        position=LocalPoint(
            east_m=float((terrain.x_m[0] + terrain.x_m[-1]) / 2.0),
            north_m=float((terrain.y_m[0] + terrain.y_m[-1]) / 2.0),
            up_m=float(terrain.elevation_m.mean()) + 100.0,
        ),
        yaw_deg=0.0,
        pitch_deg=0.0,
        horizontal_sigma_m=1000.0,
        vertical_sigma_m=50.0,
        yaw_sigma_deg=90.0,
        pitch_sigma_deg=20.0,
    )
