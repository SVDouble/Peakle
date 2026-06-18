"""On-the-fly synthetic view computation.

`compute_view` runs the full per-view pipeline (render, contour extraction, pose
estimation, peak annotation) in memory with no file I/O. The live web server
calls it lazily per request; the `peakle.demo` pipeline reuses it and then adds
the artifact writes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from peakle.annotation.labeling import PeakLabeler
from peakle.domain.annotations import PeakAnnotation
from peakle.domain.camera import CameraExtrinsics
from peakle.domain.contours import SkylineContour
from peakle.domain.pose import PoseEstimate, PosePrior
from peakle.optimization.pose_search import PoseOptimizer, add_synthetic_truth_metrics
from peakle.rendering.rasterizer import RenderArrays
from peakle.rendering.skyline import extract_skyline_from_mask
from peakle.scene.state import SceneState, noisy_prior, view_id, view_label

POSE_NOISE_SEED_OFFSET = 10_000


class ComputedView(BaseModel):
    """A fully computed synthetic camera view.

    Attributes:
        view_id: Stable view identifier.
        label: Display label.
        true_camera: Ground-truth camera pose.
        pose_prior: Noisy pose prior used to seed optimization.
        contour: Observed skyline contour extracted from the render.
        pose_estimate: Estimated pose and fit diagnostics.
        estimated_profile: Predicted skyline profile for the estimated pose.
        annotations: Projected peak annotations.
        render_arrays: Raw render outputs (image, mask, profile).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    view_id: str
    label: str
    true_camera: CameraExtrinsics
    pose_prior: PosePrior
    contour: SkylineContour
    pose_estimate: PoseEstimate
    estimated_profile: NDArray[np.float64]
    annotations: list[PeakAnnotation]
    render_arrays: RenderArrays


def compute_view(state: SceneState, index: int) -> ComputedView:
    """Computes one synthetic camera view in memory.

    Args:
        state: Shared scene state (terrain, peaks, intrinsics, cameras).
        index: Camera index into `state.true_cameras`.

    Returns:
        The fully computed view, including render arrays for optional export.
    """

    true_camera = state.true_cameras[index]
    rng = np.random.default_rng(state.seed + POSE_NOISE_SEED_OFFSET + index)
    prior = noisy_prior(
        state.pose_noise,
        true_camera.position,
        true_camera.yaw_deg,
        true_camera.pitch_deg,
        rng,
    )

    render_arrays = state.renderer.render(state.terrain, state.intrinsics, true_camera)
    contour = extract_skyline_from_mask(render_arrays.terrain_mask)
    estimate = PoseOptimizer(
        max_iterations=state.optimization_max_iterations,
        objective_terrain_stride=state.objective_terrain_stride,
    ).estimate(
        terrain=state.terrain,
        contour=contour,
        intrinsics=state.intrinsics,
        prior=prior,
    )
    estimate = add_synthetic_truth_metrics(estimate, true_camera)
    estimated_profile = state.renderer.skyline_profile(state.terrain, state.intrinsics, estimate.extrinsics)
    annotations = PeakLabeler().build_annotations(
        peaks=state.peaks,
        intrinsics=state.intrinsics,
        extrinsics=estimate.extrinsics,
        skyline_profile=estimated_profile,
    )
    return ComputedView(
        view_id=view_id(index),
        label=view_label(index),
        true_camera=true_camera,
        pose_prior=prior,
        contour=contour,
        pose_estimate=estimate,
        estimated_profile=estimated_profile,
        annotations=annotations,
        render_arrays=render_arrays,
    )
