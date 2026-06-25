"""Staged, augmenting photo-processing pipeline.

`Evidence` flows through independent stages (EXIF -> intrinsics -> position prior,
contour, depth, ...) that each add one cue, then `localize` optimizes a pose from
whatever was gathered. Independent stages run concurrently.
"""

from peakle.pipeline.evidence import Evidence, ExifData
from peakle.pipeline.stages import (
    ContourStage,
    DepthStage,
    EdgeStage,
    ExifStage,
    IntrinsicsStage,
    Pipeline,
    PositionPriorStage,
    RidgeStage,
    default_pipeline,
    localize,
)

__all__ = [
    "ContourStage",
    "DepthStage",
    "EdgeStage",
    "Evidence",
    "ExifData",
    "ExifStage",
    "IntrinsicsStage",
    "Pipeline",
    "PositionPriorStage",
    "RidgeStage",
    "default_pipeline",
    "localize",
]
