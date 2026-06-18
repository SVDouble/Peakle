"""Synthetic scene metadata."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.peaks import Peak
from peakle.domain.pose import PosePrior
from peakle.domain.terrain import TerrainSpec


class SyntheticScene(BaseModel):
    """Serializable metadata for one synthetic demo scene.

    Attributes:
        terrain_spec: Terrain generation specification.
        intrinsics: Camera intrinsics.
        true_camera: Ground-truth synthetic camera pose.
        pose_prior: Noisy pose prior used for optimization.
        peaks: Detected synthetic peaks.
        artifacts: Relative artifact path map.
    """

    terrain_spec: TerrainSpec
    intrinsics: CameraIntrinsics
    true_camera: CameraExtrinsics
    pose_prior: PosePrior
    peaks: list[Peak]
    artifacts: dict[str, Path]
