"""Photo-to-render matching, metric lifting, and robust query-pose recovery."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, label

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics, CameraModel
from peakle.domain.coordinates import LocalPoint
from peakle.domain.pose import PosePrior
from peakle.domain.terrain import TerrainMap
from peakle.localize.candidate_validation import (
    CANDIDATE_VALIDATION_SCHEMA,
    HOLDOUT_PARTITION_METHOD,
    HOLDOUT_PARTITION_SCHEMA,
    CandidateValidationConfig,
    validate_candidate_pose,
)
from peakle.localize.candidate_validation import (
    candidate_render_extrinsics as _candidate_render_extrinsics,
)
from peakle.localize.candidate_validation import (
    query_holdout_fold as _query_holdout_fold,
)
from peakle.localize.candidate_validation import (
    query_spatial_holdout_mask as _query_spatial_holdout_mask,
)
from peakle.localize.correspondence import DenseMatcher, MatchSet, match_image_fan
from peakle.localize.pnp import PoseRansacConfig, PoseRansacResult, fit_pose_ransac
from peakle.rendering.orthophoto import AppearanceRaster
from peakle.rendering.rasterizer import HeightfieldLike
from peakle.rendering.terrain_view import (
    RenderModality,
    TerrainRenderBundle,
    TerrainViewRenderer,
    lift_render_pixels,
)

RenderMatchStatus = Literal["solved", "abstained"]
MATCH_SELECTION_SCHEMA = "peakle_spatial_match_selection_v2"
MATCH_SELECTION_METHOD = "deterministic_joint_grid_confidence_cap_after_lift_v2"
QUERY_PADDING_MASK_SCHEMA = "peakle_query_warp_padding_mask_v1"
QUERY_PADDING_MASK_METHOD = "border_connected_near_black_with_constrained_1px_fringe_v1"
_QUERY_PADDING_CORE_MAX_CHANNEL = 8
_QUERY_PADDING_FRINGE_MAX_CHANNEL = 24
_QUERY_PADDING_MIN_COMPONENT_PIXELS = 16
_QUERY_PADDING_MIN_COMPONENT_FRACTION = 0.0005
RENDER_SEED_SCHEMA = "peakle_render_match_seed_v1"


@dataclass(frozen=True)
class _QueryPaddingMask:
    """Truth-free mask for black padding introduced by a query-image warp."""

    mask: NDArray[np.bool_]
    record: dict[str, Any]


@dataclass(frozen=True)
class RenderSeed:
    """Truth-blind render hypothesis, independent of the statistical prior.

    The position and yaw place the terrain-render fan. ``pitch_deg`` initializes
    the query-camera fit; the pinhole render pitch remains the explicit
    ``RenderMatchConfig.render_pitch_deg`` setting. In particular, a cyltan
    atlas crop-shift must not be passed here as a physical pinhole pitch.
    """

    position: LocalPoint
    yaw_deg: float
    pitch_deg: float

    @classmethod
    def from_prior(cls, prior: PosePrior) -> RenderSeed:
        """Preserve the legacy prior-seeded behavior for existing callers."""

        return cls(
            position=prior.position.model_copy(deep=True),
            yaw_deg=prior.yaw_deg,
            pitch_deg=prior.pitch_deg,
        )

    def validate(self) -> None:
        values = (*self.position.as_tuple(), self.yaw_deg, self.pitch_deg)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("render seed coordinates and angles must be finite")
        if not -360.0 <= self.yaw_deg <= 360.0:
            raise ValueError("render seed yaw must be in [-360, 360] degrees")
        if not -89.0 <= self.pitch_deg <= 89.0:
            raise ValueError("render seed pitch must be in [-89, 89] degrees")

    def to_record(self, *, source: str) -> dict[str, Any]:
        return {
            "schema": RENDER_SEED_SCHEMA,
            "source": source,
            "position": self.position.model_dump(mode="json"),
            "yaw_deg": self.yaw_deg,
            "query_pnp_initial_pitch_deg": self.pitch_deg,
            "render_pitch_source": "render_config.render_pitch_deg",
            "uses_reference_truth": False,
        }


@dataclass(frozen=True)
class RenderMatchConfig:
    """Complete fan/matching/lifting configuration for one experiment."""

    render_width_px: int = 320
    render_height_px: int = 256
    render_horizontal_fov_deg: float | None = None
    minimum_render_fov_deg: float = 45.0
    maximum_render_fov_deg: float = 75.0
    yaw_step_deg: float = 30.0
    orientation_prior_half_span_deg: float = 60.0
    render_pitch_deg: float = 0.0
    modality: RenderModality = "hillshade"
    terrain_stride: int = 6
    native_patch_stride: int = 8
    max_matches_per_frame: int = 800
    match_selection_cell_px: int = 16
    min_lifted_matches_per_frame: int = 12
    max_relative_depth_span: float = 0.08
    expand_pnp_search_radii_from_prior_sigma: bool = True
    refinement_passes: int = 1
    candidate_validation: CandidateValidationConfig = field(default_factory=CandidateValidationConfig)
    pnp: PoseRansacConfig = field(default_factory=PoseRansacConfig)

    def validate(self) -> None:
        if self.render_width_px < 64 or self.render_height_px < 64:
            raise ValueError("render matching frames must be at least 64 pixels in each dimension")
        if not 5.0 <= self.yaw_step_deg <= 90.0:
            raise ValueError("render fan yaw step must be in [5, 90] degrees")
        if self.orientation_prior_half_span_deg < 0.0 or (
            self.orientation_prior_half_span_deg != 0.0
            and self.orientation_prior_half_span_deg < self.yaw_step_deg / 2.0
        ):
            raise ValueError("orientation-prior fan span must be zero or cover at least half a yaw step")
        if self.terrain_stride < 1:
            raise ValueError("render matching terrain stride must be positive")
        if self.native_patch_stride < 1:
            raise ValueError("render matching native patch stride must be positive")
        if self.max_matches_per_frame < self.min_lifted_matches_per_frame:
            raise ValueError("max matches per frame cannot be smaller than the minimum lifted count")
        if not 2 <= self.match_selection_cell_px <= 256:
            raise ValueError("match selection cell size must be in [2, 256] pixels")
        if self.refinement_passes not in {0, 1}:
            raise ValueError("the initial render-PnP implementation supports zero or one refinement pass")
        self.candidate_validation.validate()
        self.pnp.validate()


@dataclass(frozen=True)
class RenderMatchPoseResult:
    """Final pose or explicit abstention with per-frame evidence."""

    status: RenderMatchStatus
    extrinsics: CameraExtrinsics | None
    diagnostics: dict[str, Any]
    candidates: tuple[CameraExtrinsics, ...] = ()

    @property
    def solved(self) -> bool:
        return self.status == "solved" and self.extrinsics is not None


@dataclass(frozen=True)
class _FrameAttempt:
    index: int
    render: TerrainRenderBundle
    match_set: MatchSet
    lifted_world: NDArray[np.float64]
    query_xy: NDArray[np.float64]
    confidence: NDArray[np.float64]
    holdout_world: NDArray[np.float64]
    holdout_query_xy: NDArray[np.float64]
    holdout_confidence: NDArray[np.float64]
    pnp_result: PoseRansacResult | None
    record: dict[str, Any]


def solve_render_match_pose(
    terrain: TerrainMap,
    query_rgb: NDArray[np.uint8],
    query_camera: CameraModel,
    prior: PosePrior,
    matcher: DenseMatcher,
    *,
    use_position_prior: bool,
    use_orientation_prior: bool,
    appearance: AppearanceRaster | None = None,
    native_elevation_patch: HeightfieldLike | None = None,
    render_seed: RenderSeed | None = None,
    config: RenderMatchConfig | None = None,
    seed: int = 0,
    renderer: TerrainViewRenderer | None = None,
) -> RenderMatchPoseResult:
    """Render at an explicit hypothesis while keeping prior terms independent.

    Omitting ``render_seed`` preserves the original behavior by deriving it
    from ``prior``. An explicit seed can also provide the required render
    location when position-prior regularization is disabled.
    """

    settings = config or RenderMatchConfig()
    settings.validate()
    query = np.asarray(query_rgb)
    if query.shape != (query_camera.height_px, query_camera.width_px, 3):
        raise ValueError(
            "query RGB shape and query camera disagree: "
            f"{query.shape} versus {(query_camera.height_px, query_camera.width_px, 3)}"
        )
    if query.dtype != np.uint8:
        query = np.clip(np.rint(query), 0.0, 255.0).astype(np.uint8)
    query_sha256 = hashlib.sha256(memoryview(np.ascontiguousarray(query)).cast("B")).hexdigest()
    holdout_fold = _query_holdout_fold(query_sha256, settings.candidate_validation)
    query_padding = _query_warp_padding_mask(query)
    if render_seed is None and not use_position_prior:
        return _abstained_result(
            settings,
            matcher,
            seed,
            reason="render_pnp_requires_position_seed",
            query=query,
            query_camera=query_camera,
            query_padding=query_padding,
        )
    render_seed_source = "explicit_argument" if render_seed is not None else "pose_prior_compatibility_fallback"
    active_render_seed = render_seed or RenderSeed.from_prior(prior)
    active_render_seed.validate()
    view_renderer = renderer or TerrainViewRenderer()
    render_fov = _render_fov(query_camera, settings)
    render_intrinsics = CameraIntrinsics.from_horizontal_fov(
        settings.render_width_px,
        settings.render_height_px,
        render_fov,
    )
    yaws = render_fan_yaws(active_render_seed.yaw_deg, use_orientation_prior, settings)
    bundles: list[TerrainRenderBundle] = []
    for yaw_deg in yaws:
        render_pose = CameraExtrinsics(
            position=active_render_seed.position,
            yaw_deg=yaw_deg,
            pitch_deg=settings.render_pitch_deg,
            roll_deg=0.0,
        )
        bundles.append(
            view_renderer.render(
                terrain,
                render_intrinsics,
                render_pose,
                modality=settings.modality,
                appearance=appearance,
                terrain_stride=settings.terrain_stride,
                native_elevation_patch=native_elevation_patch,
                native_patch_stride=settings.native_patch_stride,
            )
        )
    match_sets = match_image_fan(matcher, query, [bundle.rgb for bundle in bundles])
    attempts: list[_FrameAttempt] = []
    for index, (bundle, matches) in enumerate(zip(bundles, match_sets, strict=True)):
        attempt = _match_and_fit_frame(
            index,
            bundle,
            matches,
            query_camera,
            prior,
            terrain,
            use_position_prior,
            use_orientation_prior,
            settings,
            seed,
            query_padding=query_padding,
            native_elevation_patch=native_elevation_patch,
            holdout_fold=holdout_fold,
            query_sha256=query_sha256,
            render_seed=active_render_seed,
            render_seed_source=render_seed_source,
        )
        attempts.append(attempt)

    solved_attempts = [attempt for attempt in attempts if attempt.pnp_result is not None and attempt.pnp_result.solved]
    base_diagnostics = _base_diagnostics(
        settings,
        matcher,
        seed,
        query,
        query_camera,
        render_fov,
        yaws,
        attempts,
        query_padding,
        prior,
        active_render_seed,
        render_seed_source,
        use_position_prior,
        use_orientation_prior,
    )
    if not solved_attempts:
        return RenderMatchPoseResult(
            status="abstained",
            extrinsics=None,
            candidates=(),
            diagnostics={
                **base_diagnostics,
                "abstain_reason": "no_render_frame_produced_an_accepted_pnp_consensus",
            },
        )
    solved_attempts.sort(key=_frame_rank, reverse=True)
    best = solved_attempts[0]
    validation_source = best
    if best.pnp_result is None or best.pnp_result.extrinsics is None:
        raise RuntimeError("solved frame lost its PnP estimate")
    estimate = best.pnp_result.extrinsics
    refinement_record: dict[str, Any] | None = None
    if settings.refinement_passes:
        refined_pose = CameraExtrinsics(
            position=estimate.position,
            yaw_deg=estimate.yaw_deg,
            pitch_deg=settings.render_pitch_deg,
            roll_deg=0.0,
        )
        refined_bundle = view_renderer.render(
            terrain,
            render_intrinsics,
            refined_pose,
            modality=settings.modality,
            appearance=appearance,
            terrain_stride=settings.terrain_stride,
            native_elevation_patch=native_elevation_patch,
            native_patch_stride=settings.native_patch_stride,
        )
        tight_pnp = replace(
            settings.pnp,
            horizontal_search_radius_m=min(settings.pnp.horizontal_search_radius_m, 300.0),
            vertical_search_radius_m=min(settings.pnp.vertical_search_radius_m, 180.0),
            yaw_search_radius_deg=min(settings.pnp.yaw_search_radius_deg, 20.0),
            seed=_derived_seed(seed, "refinement"),
        )
        refined_settings = replace(settings, pnp=tight_pnp, refinement_passes=0)
        refinement_seed = RenderSeed(
            position=estimate.position,
            yaw_deg=estimate.yaw_deg,
            pitch_deg=estimate.pitch_deg,
        )
        refined = _match_and_fit_frame(
            len(attempts),
            refined_bundle,
            matcher.match(query, refined_bundle.rgb),
            query_camera,
            prior,
            terrain,
            use_position_prior,
            use_orientation_prior,
            refined_settings,
            seed,
            query_padding=query_padding,
            native_elevation_patch=native_elevation_patch,
            holdout_fold=holdout_fold,
            query_sha256=query_sha256,
            render_seed=refinement_seed,
            render_seed_source="first_pass_estimate",
        )
        refinement_record = refined.record
        refinement_batch = _common_matcher_batch([refined])
        if refinement_batch is not None:
            refinement_record = {**refinement_record, "matcher_batch": refinement_batch}
        if refined.pnp_result is not None and refined.pnp_result.solved and refined.pnp_result.extrinsics is not None:
            estimate = refined.pnp_result.extrinsics
            best = refined

    candidate_validation: dict[str, Any]
    if settings.candidate_validation.enabled:
        candidate_render_pose = _candidate_render_extrinsics(query_camera, estimate)
        validation_multiplier = settings.candidate_validation.render_resolution_multiplier
        candidate_validation_intrinsics = CameraIntrinsics.from_horizontal_fov(
            settings.render_width_px * validation_multiplier,
            settings.render_height_px * validation_multiplier,
            render_fov,
        )
        candidate_render = view_renderer.render(
            terrain,
            candidate_validation_intrinsics,
            candidate_render_pose,
            modality=settings.modality,
            appearance=appearance,
            terrain_stride=settings.terrain_stride,
            native_elevation_patch=native_elevation_patch,
            native_patch_stride=settings.native_patch_stride,
        )
        candidate_validation = _validate_candidate_pose(
            validation_source,
            candidate_render,
            query_camera,
            estimate,
            settings,
            holdout_fold=holdout_fold,
        )
    else:
        candidate_validation = {
            "schema": CANDIDATE_VALIDATION_SCHEMA,
            "enabled": False,
            "passed": True,
            "failures": [],
            "uses_reference_truth": False,
            "withheld_from_geometric_pose_fit": False,
            "withheld_from_geometric_frame_ranking": False,
            "matcher_used_full_query_image": True,
            "worker_candidate_selection_precedes_holdout": None,
            "note": (
                "explicit ablation: no lifted correspondences were withheld; the ordinary spatial cap "
                "still limits pose-fitting input"
            ),
        }

    final_pnp = best.pnp_result
    final_diagnostics = {
        **base_diagnostics,
        "selected_frame_index": validation_source.index,
        "selected_frame_yaw_deg": validation_source.render.extrinsics.yaw_deg,
        "final_fit_frame_index": best.index,
        "refinement": refinement_record,
        "final_pnp": final_pnp.diagnostics if final_pnp is not None else None,
        "candidate_validation": candidate_validation,
    }
    if not candidate_validation["passed"]:
        return RenderMatchPoseResult(
            status="abstained",
            extrinsics=None,
            candidates=(),
            diagnostics={
                **final_diagnostics,
                "abstain_reason": "candidate_pose_holdout_validation_failed",
                "rejected_candidate_pose": estimate.model_dump(mode="json"),
                "runner_up_retry_attempted": False,
            },
        )
    return RenderMatchPoseResult(
        status="solved",
        extrinsics=estimate,
        # Only the selected pose is exposed. Training-ranked runner-ups are
        # never validated, including when the gate is disabled for ablation.
        candidates=(estimate,),
        diagnostics=final_diagnostics,
    )


def render_fan_yaws(
    prior_yaw_deg: float,
    use_orientation_prior: bool,
    config: RenderMatchConfig,
) -> tuple[float, ...]:
    """Deterministic seed-centred fan (full 360° without orientation guidance)."""

    if use_orientation_prior:
        if config.orientation_prior_half_span_deg == 0.0:
            return (round(_wrap180(prior_yaw_deg), 10),)
        offsets = np.arange(
            -config.orientation_prior_half_span_deg,
            config.orientation_prior_half_span_deg + config.yaw_step_deg * 0.5,
            config.yaw_step_deg,
        )
        values = [_wrap180(prior_yaw_deg + float(offset)) for offset in offsets]
    else:
        values = [_wrap180(float(value)) for value in np.arange(-180.0, 180.0, config.yaw_step_deg)]
    # Preserve order while removing a possible wrapped duplicate.
    return tuple(dict.fromkeys(round(value, 10) for value in values))


def _query_warp_padding_mask(query_rgb: NDArray[np.uint8]) -> _QueryPaddingMask:
    """Find only meaningful near-black components connected to the image border.

    GeoPose's cylindrical-tangent warp can leave black wedges around the valid
    photo.  A global darkness threshold would also erase dark mountain faces,
    so the core is restricted to connected components that touch an image
    border.  One constrained pixel of slightly lighter near-black fringe is
    included for resampling antialiasing; the dilation cannot grow through
    normal image content.
    """

    query = np.asarray(query_rgb)
    if query.ndim != 3 or query.shape[2] != 3:
        raise ValueError(f"query RGB must have shape (H, W, 3), got {query.shape}")
    if query.dtype != np.uint8:
        query = np.clip(np.rint(query), 0.0, 255.0).astype(np.uint8)
    height, width = query.shape[:2]
    pixel_count = height * width
    minimum_component_pixels = max(
        _QUERY_PADDING_MIN_COMPONENT_PIXELS,
        int(math.ceil(pixel_count * _QUERY_PADDING_MIN_COMPONENT_FRACTION)),
    )
    max_channel = np.max(query, axis=2)
    core_candidate = max_channel <= _QUERY_PADDING_CORE_MAX_CHANNEL
    connectivity = np.ones((3, 3), dtype=np.uint8)
    components, component_count = label(core_candidate, structure=connectivity)
    border_labels = np.unique(
        np.concatenate(
            (
                components[0, :],
                components[-1, :],
                components[:, 0],
                components[:, -1],
            )
        )
    )
    border_labels = border_labels[border_labels != 0]
    counts = np.bincount(components.ravel(), minlength=component_count + 1)
    retained_labels = border_labels[counts[border_labels] >= minimum_component_pixels]
    if retained_labels.size:
        core_mask = np.isin(components, retained_labels)
        fringe_candidate = max_channel <= _QUERY_PADDING_FRINGE_MAX_CHANNEL
        dilated = binary_dilation(core_mask, structure=connectivity, iterations=1)
        fringe_mask = dilated & fringe_candidate & ~core_mask
        mask = core_mask | fringe_mask
        no_op_reason: str | None = None
    else:
        core_mask = np.zeros((height, width), dtype=np.bool_)
        fringe_mask = np.zeros_like(core_mask)
        mask = np.zeros_like(core_mask)
        no_op_reason = "no_meaningful_border_connected_near_black_component"
    mask = np.asarray(mask, dtype=np.bool_)
    mask.flags.writeable = False
    masked_pixels = int(mask.sum())
    record = {
        "schema": QUERY_PADDING_MASK_SCHEMA,
        "method": QUERY_PADDING_MASK_METHOD,
        "truth_free": True,
        "active": masked_pixels > 0,
        "no_op_reason": no_op_reason,
        "application_stage": "after worker selection/cache restoration; before render lifting and spatial cap",
        "changes_matcher_or_cache_inputs": False,
        "coordinate_sampling": "nearest original-resolution pixel centre via floor(x + 0.5)",
        "thresholds": {
            "core_max_rgb_channel_uint8": _QUERY_PADDING_CORE_MAX_CHANNEL,
            "antialias_fringe_max_rgb_channel_uint8": _QUERY_PADDING_FRINGE_MAX_CHANNEL,
            "antialias_fringe_radius_px": 1,
            "component_connectivity": 8,
            "minimum_component_pixels": minimum_component_pixels,
            "minimum_component_fraction": _QUERY_PADDING_MIN_COMPONENT_FRACTION,
        },
        "connected_components": {
            "near_black_total": int(component_count),
            "touching_border": int(border_labels.size),
            "retained_as_meaningful": int(retained_labels.size),
        },
        "core_masked_pixels": int(core_mask.sum()),
        "antialias_fringe_pixels": int(fringe_mask.sum()),
        "masked_pixels": masked_pixels,
        "image_pixels": pixel_count,
        "masked_pixel_fraction": round(masked_pixels / max(pixel_count, 1), 8),
    }
    return _QueryPaddingMask(mask=mask, record=record)


def _matches_in_query_padding(
    padding: _QueryPaddingMask,
    query_xy_px: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Classify subpixel match locations using nearest pixel-centre sampling."""

    points = np.asarray(query_xy_px, dtype=np.float64)
    rejected = np.zeros(points.shape[0], dtype=np.bool_)
    if not padding.record["active"] or points.size == 0:
        return rejected
    columns = np.floor(points[:, 0] + 0.5).astype(np.int64)
    rows = np.floor(points[:, 1] + 0.5).astype(np.int64)
    height, width = padding.mask.shape
    in_bounds = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    rejected[in_bounds] = padding.mask[rows[in_bounds], columns[in_bounds]]
    return rejected


def _balanced_match_cap_indices(
    query_xy: NDArray[np.float64],
    render_xy: NDArray[np.float64],
    confidence: NDArray[np.float64],
    *,
    max_matches: int,
    cell_px: int,
) -> NDArray[np.int64]:
    """Select a deterministic confidence-ranked cap spread across both images.

    The progressive per-cell cap matches the worker's ordering contract.  It can
    therefore re-select a smaller balanced prefix from cached worker candidates
    without rerunning learned inference.
    """

    scores = np.asarray(confidence, dtype=np.float64)
    if scores.size == 0:
        return np.empty(0, dtype=np.int64)
    source = np.asarray(query_xy, dtype=np.float64)
    target = np.asarray(render_xy, dtype=np.float64)
    original_index = np.arange(scores.size, dtype=np.int64)
    order = np.lexsort(
        (
            original_index,
            target[:, 1],
            target[:, 0],
            source[:, 1],
            source[:, 0],
            -scores,
        )
    )
    if scores.size <= max_matches:
        return order.astype(np.int64, copy=False)

    query_cells = np.floor(source / cell_px).astype(np.int64)
    render_cells = np.floor(target / cell_px).astype(np.int64)
    query_counts: dict[tuple[int, int], int] = {}
    render_counts: dict[tuple[int, int], int] = {}
    selected: list[int] = []
    selected_mask = np.zeros(scores.size, dtype=np.bool_)
    cap = 1
    while len(selected) < max_matches:
        for raw_index in order:
            index = int(raw_index)
            if selected_mask[index]:
                continue
            query_key = (int(query_cells[index, 0]), int(query_cells[index, 1]))
            render_key = (int(render_cells[index, 0]), int(render_cells[index, 1]))
            if query_counts.get(query_key, 0) >= cap or render_counts.get(render_key, 0) >= cap:
                continue
            selected.append(index)
            selected_mask[index] = True
            query_counts[query_key] = query_counts.get(query_key, 0) + 1
            render_counts[render_key] = render_counts.get(render_key, 0) + 1
            if len(selected) == max_matches:
                break
        cap += 1
        if cap > scores.size:
            break
    return np.asarray(selected, dtype=np.int64)


def _match_selection_record(
    matches: MatchSet,
    selected: NDArray[np.int64],
    *,
    query_camera: CameraModel,
    render_shape: tuple[int, ...],
    max_matches: int,
    cell_px: int,
    raw_matches: MatchSet | None = None,
    pre_lift_matches: MatchSet | None = None,
) -> dict[str, Any]:
    worker_matches = raw_matches if raw_matches is not None else matches
    lifting_input = pre_lift_matches if pre_lift_matches is not None else worker_matches
    return {
        "schema": MATCH_SELECTION_SCHEMA,
        "method": MATCH_SELECTION_METHOD,
        "cap_stage": "after_render_lifting",
        "cell_px": cell_px,
        "budget": max_matches,
        "worker_selected_matches": worker_matches.count,
        "query_padding_valid_matches": lifting_input.count,
        "rejected_by_query_padding_before_lifting": worker_matches.count - lifting_input.count,
        "lift_valid_matches": matches.count,
        "rejected_by_lifting_before_cap": lifting_input.count - matches.count,
        "raw_worker_exceeds_budget": worker_matches.count > max_matches,
        "input_matches": matches.count,
        "selected_matches": int(selected.size),
        "cap_applied": matches.count > max_matches,
        "raw_worker_distribution": {
            "query": _match_distribution(
                worker_matches.query_xy_px,
                width=query_camera.width_px,
                height=query_camera.height_px,
                cell_px=cell_px,
            ),
            "render": _match_distribution(
                worker_matches.render_xy_px,
                width=render_shape[1],
                height=render_shape[0],
                cell_px=cell_px,
            ),
        },
        "query_padding_valid_distribution": {
            "query": _match_distribution(
                lifting_input.query_xy_px,
                width=query_camera.width_px,
                height=query_camera.height_px,
                cell_px=cell_px,
            ),
            "render": _match_distribution(
                lifting_input.render_xy_px,
                width=render_shape[1],
                height=render_shape[0],
                cell_px=cell_px,
            ),
        },
        "input_distribution": {
            "query": _match_distribution(
                matches.query_xy_px,
                width=query_camera.width_px,
                height=query_camera.height_px,
                cell_px=cell_px,
            ),
            "render": _match_distribution(
                matches.render_xy_px,
                width=render_shape[1],
                height=render_shape[0],
                cell_px=cell_px,
            ),
        },
        "selected_distribution": {
            "query": _match_distribution(
                matches.query_xy_px[selected],
                width=query_camera.width_px,
                height=query_camera.height_px,
                cell_px=cell_px,
            ),
            "render": _match_distribution(
                matches.render_xy_px[selected],
                width=render_shape[1],
                height=render_shape[0],
                cell_px=cell_px,
            ),
        },
    }


def _match_distribution(
    xy: NDArray[np.float64],
    *,
    width: int,
    height: int,
    cell_px: int,
) -> dict[str, Any]:
    points = np.asarray(xy, dtype=np.float64)
    if points.size == 0:
        return {
            "occupied_4x4_cells": 0,
            "occupied_selection_cells": 0,
            "x_span_fraction": 0.0,
            "y_span_fraction": 0.0,
        }
    x_fraction = np.clip(points[:, 0] / max(width - 1, 1), 0.0, 1.0)
    y_fraction = np.clip(points[:, 1] / max(height - 1, 1), 0.0, 1.0)
    coarse_x = np.minimum((x_fraction * 4).astype(np.int64), 3)
    coarse_y = np.minimum((y_fraction * 4).astype(np.int64), 3)
    fine = np.floor(points / cell_px).astype(np.int64)
    return {
        "occupied_4x4_cells": len(set(zip(coarse_x.tolist(), coarse_y.tolist(), strict=True))),
        "occupied_selection_cells": len(set(map(tuple, fine.tolist()))),
        "x_span_fraction": round(float(np.ptp(x_fraction)), 6),
        "y_span_fraction": round(float(np.ptp(y_fraction)), 6),
    }


def _subset_match_set(matches: MatchSet, selected: NDArray[np.bool_]) -> MatchSet:
    mask = np.asarray(selected, dtype=np.bool_)
    if mask.shape != (matches.count,):
        raise ValueError("match subset mask has the wrong shape")
    return MatchSet(
        query_xy_px=matches.query_xy_px[mask],
        render_xy_px=matches.render_xy_px[mask],
        confidence=matches.confidence[mask],
        diagnostics=matches.diagnostics,
        provenance=matches.provenance,
    )


def _match_and_fit_frame(
    index: int,
    render: TerrainRenderBundle,
    match_set: MatchSet,
    query_camera: CameraModel,
    prior: PosePrior,
    terrain: TerrainMap,
    use_position_prior: bool,
    use_orientation_prior: bool,
    settings: RenderMatchConfig,
    seed: int,
    *,
    query_padding: _QueryPaddingMask | None = None,
    native_elevation_patch: HeightfieldLike | None = None,
    holdout_fold: int = 0,
    query_sha256: str | None = None,
    render_seed: RenderSeed | None = None,
    render_seed_source: str | None = None,
) -> _FrameAttempt:
    active_render_seed = render_seed or RenderSeed.from_prior(prior)
    active_render_seed_source = render_seed_source or (
        "explicit_argument" if render_seed is not None else "pose_prior_compatibility_fallback"
    )
    active_render_seed.validate()
    matches = match_set.chosen()
    if query_padding is None:
        query_padding = _query_warp_padding_mask(
            np.full((query_camera.height_px, query_camera.width_px, 3), 255, dtype=np.uint8)
        )
    rejected_by_query_padding = _matches_in_query_padding(query_padding, matches.query_xy_px)
    padding_valid_matches = MatchSet(
        query_xy_px=matches.query_xy_px[~rejected_by_query_padding],
        render_xy_px=matches.render_xy_px[~rejected_by_query_padding],
        confidence=matches.confidence[~rejected_by_query_padding],
        diagnostics=matches.diagnostics,
        provenance=matches.provenance,
    )
    lifted = lift_render_pixels(
        render,
        padding_valid_matches.render_xy_px,
        max_relative_depth_span=settings.max_relative_depth_span,
    )
    lift_valid_matches = MatchSet(
        query_xy_px=padding_valid_matches.query_xy_px[lifted.valid],
        render_xy_px=padding_valid_matches.render_xy_px[lifted.valid],
        confidence=padding_valid_matches.confidence[lifted.valid],
    )
    validation_config = settings.candidate_validation
    raw_holdout = _query_spatial_holdout_mask(
        matches.query_xy_px,
        query_camera,
        validation_config,
        holdout_fold,
    )
    padding_holdout = _query_spatial_holdout_mask(
        padding_valid_matches.query_xy_px,
        query_camera,
        validation_config,
        holdout_fold,
    )
    lift_holdout = _query_spatial_holdout_mask(
        lift_valid_matches.query_xy_px,
        query_camera,
        validation_config,
        holdout_fold,
    )
    raw_training_matches = _subset_match_set(matches, ~raw_holdout)
    raw_holdout_matches = _subset_match_set(matches, raw_holdout)
    padding_training_matches = _subset_match_set(padding_valid_matches, ~padding_holdout)
    padding_holdout_matches = _subset_match_set(padding_valid_matches, padding_holdout)
    lift_training_matches = _subset_match_set(lift_valid_matches, ~lift_holdout)
    lift_holdout_matches = _subset_match_set(lift_valid_matches, lift_holdout)
    training_order = _balanced_match_cap_indices(
        lift_training_matches.query_xy_px,
        lift_training_matches.render_xy_px,
        lift_training_matches.confidence,
        max_matches=settings.max_matches_per_frame,
        cell_px=settings.match_selection_cell_px,
    )
    holdout_order = _balanced_match_cap_indices(
        lift_holdout_matches.query_xy_px,
        lift_holdout_matches.render_xy_px,
        lift_holdout_matches.confidence,
        max_matches=validation_config.max_holdout_matches_per_frame,
        cell_px=settings.match_selection_cell_px,
    )
    training_selection_record = _match_selection_record(
        lift_training_matches,
        training_order,
        query_camera=query_camera,
        render_shape=render.rgb.shape,
        max_matches=settings.max_matches_per_frame,
        cell_px=settings.match_selection_cell_px,
        raw_matches=raw_training_matches,
        pre_lift_matches=padding_training_matches,
    )
    holdout_selection_record = _match_selection_record(
        lift_holdout_matches,
        holdout_order,
        query_camera=query_camera,
        render_shape=render.rgb.shape,
        max_matches=validation_config.max_holdout_matches_per_frame,
        cell_px=settings.match_selection_cell_px,
        raw_matches=raw_holdout_matches,
        pre_lift_matches=padding_holdout_matches,
    )
    lift_world = lifted.world_xyz_m[lifted.valid]
    training_world_all = lift_world[~lift_holdout]
    holdout_world_all = lift_world[lift_holdout]
    query_xy = lift_training_matches.query_xy_px[training_order]
    confidence = lift_training_matches.confidence[training_order]
    world = training_world_all[training_order]
    holdout_query_xy = lift_holdout_matches.query_xy_px[holdout_order]
    holdout_confidence = lift_holdout_matches.confidence[holdout_order]
    holdout_world = holdout_world_all[holdout_order]
    pnp_result: PoseRansacResult | None = None
    initial: CameraExtrinsics | None = None
    if len(world) >= settings.min_lifted_matches_per_frame:
        initial = CameraExtrinsics(
            position=active_render_seed.position,
            yaw_deg=render.extrinsics.yaw_deg,
            pitch_deg=active_render_seed.pitch_deg if use_orientation_prior else 0.0,
            roll_deg=0.0,
        )
        clearance_policy = settings.pnp.clearance_constraint_policy
        if clearance_policy == "auto":
            # GeoPose cyltan pitch is an exact global crop-shift nuisance. With
            # a position prior, independently optimizing camera-up is therefore
            # vertically degenerate; couple it to DEM clearance by default.
            clearance_policy = (
                "prior_ground_coupled" if query_camera.projection == "cyltan" and use_position_prior else "free_up"
            )
        pnp_settings = replace(
            settings.pnp,
            clearance_constraint_policy=clearance_policy,
            horizontal_search_radius_m=(
                max(
                    settings.pnp.horizontal_search_radius_m,
                    min(3.0 * prior.horizontal_sigma_m, 2_000.0),
                )
                if use_position_prior and settings.expand_pnp_search_radii_from_prior_sigma
                else settings.pnp.horizontal_search_radius_m
            ),
            vertical_search_radius_m=(
                max(
                    settings.pnp.vertical_search_radius_m,
                    min(3.0 * prior.vertical_sigma_m, 800.0),
                )
                if use_position_prior and settings.expand_pnp_search_radii_from_prior_sigma
                else settings.pnp.vertical_search_radius_m
            ),
            seed=_derived_seed(seed, index, render.extrinsics.yaw_deg),
        )
        pnp_result = fit_pose_ransac(
            world,
            query_xy,
            confidence,
            query_camera,
            initial,
            prior=prior,
            terrain=terrain,
            native_elevation_patch=native_elevation_patch,
            use_position_prior=use_position_prior,
            use_orientation_prior=use_orientation_prior,
            config=pnp_settings,
        )
    record = {
        "index": index,
        "yaw_deg": render.extrinsics.yaw_deg,
        "pitch_deg": render.extrinsics.pitch_deg,
        "matcher_matches": matches.count,
        "query_padding_valid_matches_before_lifting": padding_valid_matches.count,
        "query_padding_rejected_matches": int(rejected_by_query_padding.sum()),
        "lift_valid_matches_before_cap": lift_valid_matches.count,
        "matches_after_cap": int(len(training_order)),
        "match_stage_counts": {
            "worker_selected": matches.count,
            "after_query_padding_rejection": padding_valid_matches.count,
            "after_render_lifting": lift_valid_matches.count,
            "after_spatial_cap": int(len(training_order)),
            "training_after_spatial_cap": int(len(training_order)),
            "holdout_after_spatial_cap": int(len(holdout_order)),
            "total_after_independent_caps": int(len(training_order) + len(holdout_order)),
        },
        "query_padding_filter": {
            **query_padding.record,
            "input_matches": matches.count,
            "rejected_matches": int(rejected_by_query_padding.sum()),
            "matches_after_filter": padding_valid_matches.count,
        },
        "match_selection": training_selection_record,
        "holdout_match_selection": holdout_selection_record,
        "holdout_partition": {
            "schema": HOLDOUT_PARTITION_SCHEMA,
            "method": HOLDOUT_PARTITION_METHOD,
            "enabled": validation_config.enabled,
            "query_content_sha256": query_sha256,
            "grid": {
                "columns": validation_config.query_grid_columns,
                "rows": validation_config.query_grid_rows,
            },
            "folds": validation_config.folds,
            "heldout_fold": holdout_fold if validation_config.enabled else None,
            "training_folds": (
                [fold for fold in range(validation_config.folds) if fold != holdout_fold]
                if validation_config.enabled
                else list(range(validation_config.folds))
            ),
            "partition_stage": (
                "after matcher worker selection, query-padding rejection, and render lifting; "
                "before independent local spatial caps"
            ),
            "withheld_from_geometric_pose_fit": validation_config.enabled,
            "withheld_from_geometric_frame_ranking": validation_config.enabled,
            "matcher_used_full_query_image": True,
            "worker_candidate_selection_precedes_holdout": True,
            "uses_reference_truth": False,
        },
        "lifted_matches": int(len(world)),
        "heldout_lifted_matches": int(len(holdout_world)),
        "lifting": lifted.rejection_counts,
        "matcher_diagnostics": matches.diagnostics,
        "render": render.provenance,
        "render_seed": active_render_seed.to_record(source=active_render_seed_source),
        "pnp_initial_pose": initial.model_dump(mode="json") if initial is not None else None,
        "pose_prior_usage": {
            "position_regularization_enabled": use_position_prior,
            "orientation_regularization_enabled": use_orientation_prior,
            "clearance_anchor_uses_supplied_prior": (query_camera.projection == "cyltan" and use_position_prior),
        },
        "pnp_status": pnp_result.status if pnp_result is not None else "not_attempted",
        "pnp": pnp_result.diagnostics if pnp_result is not None else None,
    }
    return _FrameAttempt(
        index=index,
        render=render,
        match_set=matches,
        lifted_world=world,
        query_xy=query_xy,
        confidence=confidence,
        holdout_world=holdout_world,
        holdout_query_xy=holdout_query_xy,
        holdout_confidence=holdout_confidence,
        pnp_result=pnp_result,
        record=record,
    )


def _validate_candidate_pose(
    source: _FrameAttempt,
    candidate_render: TerrainRenderBundle,
    query_camera: CameraModel,
    candidate: CameraExtrinsics,
    settings: RenderMatchConfig,
    *,
    holdout_fold: int,
) -> dict[str, Any]:
    """Gate one training-selected candidate on never-fit geometric evidence."""

    return validate_candidate_pose(
        source_render=source.render,
        source_frame_index=source.index,
        holdout_world=source.holdout_world,
        holdout_query_xy=source.holdout_query_xy,
        candidate_render=candidate_render,
        query_camera=query_camera,
        candidate=candidate,
        pnp=settings.pnp,
        config=settings.candidate_validation,
        holdout_fold=holdout_fold,
    )


def _frame_rank(attempt: _FrameAttempt) -> tuple[int, float, float]:
    if attempt.pnp_result is None:
        return (-1, -1.0, -math.inf)
    diagnostics = attempt.pnp_result.diagnostics
    inliers = int(diagnostics.get("inliers") or 0)
    ratio = float(diagnostics.get("inlier_ratio") or 0.0)
    median = diagnostics.get("median_reprojection_error_px")
    return (inliers, ratio, -float(median if median is not None else math.inf))


def _base_diagnostics(
    settings: RenderMatchConfig,
    matcher: DenseMatcher,
    seed: int,
    query: NDArray[np.uint8],
    query_camera: CameraModel,
    render_fov: float,
    yaws: tuple[float, ...],
    attempts: list[_FrameAttempt],
    query_padding: _QueryPaddingMask,
    prior: PosePrior,
    render_seed: RenderSeed,
    render_seed_source: str,
    use_position_prior: bool,
    use_orientation_prior: bool,
) -> dict[str, Any]:
    diagnostics = {
        "kind": "render_match_lift_pnp_v2",
        "matcher": matcher.identity(),
        "query": {
            "shape": list(query.shape),
            "sha256": hashlib.sha256(memoryview(np.ascontiguousarray(query)).cast("B")).hexdigest(),
            "camera": query_camera.model_dump(mode="json"),
            "warp_padding_mask": query_padding.record,
        },
        "render_config": _render_config_record(settings, render_fov),
        "render_seed": render_seed.to_record(source=render_seed_source),
        "pose_prior": {
            **prior.model_dump(mode="json"),
            "position_regularization_enabled": use_position_prior,
            "orientation_regularization_enabled": use_orientation_prior,
            "clearance_anchor_enabled": query_camera.projection == "cyltan" and use_position_prior,
            "unchanged_during_refinement": True,
        },
        "render_fan_yaws_deg": list(yaws),
        "frames": [attempt.record for attempt in attempts],
        "seed": seed,
        "estimator_inputs": {
            "query_rgb": True,
            "terrain": True,
            "pose_prior": True,
            "explicit_render_seed": render_seed_source == "explicit_argument",
            "source_depth_pfm": False,
            "reference_pose": False,
            "gt_v2_refined_pose": False,
            "photo_skyline": False,
        },
    }
    matcher_batch = _common_matcher_batch(attempts)
    if matcher_batch is not None:
        diagnostics["matcher_batch"] = matcher_batch
    return diagnostics


def _common_matcher_batch(attempts: list[_FrameAttempt]) -> dict[str, Any] | None:
    batches = [
        attempt.match_set.provenance.get("worker_batch")
        for attempt in attempts
        if attempt.match_set.provenance.get("worker_batch") is not None
    ]
    if not batches:
        return None
    first = batches[0]
    if not isinstance(first, dict) or any(batch != first for batch in batches[1:]):
        raise RuntimeError("one render fan unexpectedly contains multiple worker batches")
    return first


def _abstained_result(
    settings: RenderMatchConfig,
    matcher: DenseMatcher,
    seed: int,
    *,
    reason: str,
    query: NDArray[np.uint8],
    query_camera: CameraModel,
    query_padding: _QueryPaddingMask,
) -> RenderMatchPoseResult:
    return RenderMatchPoseResult(
        status="abstained",
        extrinsics=None,
        diagnostics={
            "kind": "render_match_lift_pnp_v2",
            "abstain_reason": reason,
            "matcher": matcher.identity(),
            "render_config": _render_config_record(settings, _render_fov(query_camera, settings)),
            "query_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(query)).cast("B")).hexdigest(),
            "query_warp_padding_mask": query_padding.record,
            "seed": seed,
        },
    )


def _render_fov(camera: CameraModel, config: RenderMatchConfig) -> float:
    if config.render_horizontal_fov_deg is not None:
        value = config.render_horizontal_fov_deg
    else:
        value = camera.horizontal_fov_deg
    return float(np.clip(value, config.minimum_render_fov_deg, config.maximum_render_fov_deg))


def _render_config_record(config: RenderMatchConfig, actual_fov: float) -> dict[str, Any]:
    return {
        "render_width_px": config.render_width_px,
        "render_height_px": config.render_height_px,
        "render_horizontal_fov_deg": actual_fov,
        "yaw_step_deg": config.yaw_step_deg,
        "orientation_prior_half_span_deg": config.orientation_prior_half_span_deg,
        "render_pitch_deg": config.render_pitch_deg,
        "modality": config.modality,
        "terrain_stride": config.terrain_stride,
        "native_patch_stride": config.native_patch_stride,
        "max_matches_per_frame": config.max_matches_per_frame,
        "match_selection": {
            "schema": MATCH_SELECTION_SCHEMA,
            "method": MATCH_SELECTION_METHOD,
            "cell_px": config.match_selection_cell_px,
        },
        "min_lifted_matches_per_frame": config.min_lifted_matches_per_frame,
        "max_relative_depth_span": config.max_relative_depth_span,
        "expand_pnp_search_radii_from_prior_sigma": config.expand_pnp_search_radii_from_prior_sigma,
        "refinement_passes": config.refinement_passes,
        "candidate_validation": dict(config.candidate_validation.__dict__),
        "pnp": dict(config.pnp.__dict__),
    }


def _derived_seed(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _wrap180(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0
