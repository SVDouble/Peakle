"""Image artifact domain models."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class RenderedFrame(BaseModel):
    """Paths for synthetic render artifacts.

    Attributes:
        image_path: Path to the RGB render.
        mask_path: Path to the binary terrain mask.
        contour_debug_path: Optional contour debug image path.
    """

    image_path: Path
    mask_path: Path
    contour_debug_path: Path | None
