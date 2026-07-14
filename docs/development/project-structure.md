# Project Structure

> **Original architecture proposal.** The thin-CLI principle remains valid, but the tree below no
> longer describes the whole research codebase. The consolidation target and current guardrails are
> maintained in the [research and development program](../research-and-development.md).

This structure separates domain models, geometry, rendering, optimization,
annotation, persistence, and command-line orchestration. The CLI should assemble
validated configs and call services; it should not own geometry or optimization
logic.

## Proposed Tree

```text
.
|-- pyproject.toml
|-- uv.lock
|-- README.md
|-- docs/
|   `-- development/
|       |-- README.md
|       |-- project-structure.md
|       |-- synthetic-demo-pipeline.md
|       |-- algorithms.md
|       `-- roadmap-and-risks.md
|-- src/
|   `-- peakle/
|       |-- __init__.py
|       |-- config.py
|       |-- cli.py
|       |-- domain/
|       |   |-- __init__.py
|       |   |-- annotations.py
|       |   |-- camera.py
|       |   |-- contours.py
|       |   |-- coordinates.py
|       |   |-- images.py
|       |   |-- peaks.py
|       |   |-- pose.py
|       |   `-- terrain.py
|       |-- terrain/
|       |   |-- __init__.py
|       |   |-- generator.py
|       |   `-- peak_detection.py
|       |-- rendering/
|       |   |-- __init__.py
|       |   |-- pinhole.py
|       |   |-- rasterizer.py
|       |   `-- skyline.py
|       |-- optimization/
|       |   |-- __init__.py
|       |   |-- objective.py
|       |   |-- pose_search.py
|       |   `-- scoring.py
|       |-- annotation/
|       |   |-- __init__.py
|       |   |-- labeling.py
|       |   `-- overlay.py
|       |-- io/
|       |   |-- __init__.py
|       |   |-- artifacts.py
|       |   `-- images.py
|       |-- web/
|       |   |-- __init__.py
|       |   |-- server.py
|       |   |-- viewer.py
|       |   `-- static/
|       |       |-- app.js
|       |       |-- index.html
|       |       `-- styles.css
|       `-- demo/
|           |-- __init__.py
|           `-- pipeline.py
`-- tests/
    |-- unit/
    `-- integration/
```

## Package Setup

`pyproject.toml` should declare:

```toml
[project]
name = "peakle"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
  "numpy",
  "pillow",
  "pydantic",
  "pydantic-settings",
  "scipy",
]

[project.scripts]
peakle = "peakle.cli:main"
```

Likely development dependencies:

```toml
[dependency-groups]
dev = [
  "pytest",
  "ruff",
  "ty",
]
```

The first implementation can avoid heavier packages such as OpenCV. Synthetic
contours can be extracted from the renderer's terrain mask with NumPy. Real
photo support can later introduce OpenCV or scikit-image behind a separate
adapter.

## Domain Models

Domain models should be Pydantic models when they cross module boundaries,
persist to disk, or represent user-provided configuration. Pure internal numeric
arrays can stay as NumPy arrays wrapped by small validated containers.

Suggested models:

- `GeoPoint`: latitude, longitude, elevation meters.
- `LocalPoint`: east, north, up meters in the local tangent frame.
- `GeoBounds`: coordinate extent for terrain generation.
- `CameraIntrinsics`: image width, height, focal length, principal point.
- `CameraExtrinsics`: camera position and yaw, pitch, roll.
- `PosePrior`: noisy position and uncertainty bounds.
- `TerrainSpec`: grid size, bounds, random seed, elevation controls.
- `TerrainMap`: coordinate grids, elevation grid, origin, and metadata.
- `Peak`: id, generated name, geodetic position, local position, elevation, and
  prominence.
- `SyntheticScene`: terrain id, true camera, intrinsics, and selected peaks.
- `RenderedFrame`: image path, mask path, true depth or z-buffer metadata.
- `SkylineContour`: image-space points ordered by pixel column.
- `PoseEstimate`: optimized extrinsics, score, residual statistics, iterations.
- `PeakAnnotation`: peak id, image position, visibility status, and label box.

## Service Interfaces

Services should accept validated domain objects and return validated result
objects. They should avoid reading environment variables or parsing CLI options.

Example public interface:

```python
class TerrainGenerator:
    """Generates synthetic terrain in an Earth-like local coordinate frame.

    Args:
        spec: Validated terrain generation parameters.

    Attributes:
        spec: The immutable terrain generation specification.
    """

    def generate(self) -> TerrainMap:
        """Builds an elevation grid and coordinate metadata.

        Returns:
            A terrain map containing local coordinates, geodetic coordinates,
            elevation values, and generation metadata.
        """


class PoseOptimizer:
    """Estimates camera extrinsics from a skyline contour and pose prior."""

    def estimate(
        self,
        terrain: TerrainMap,
        contour: SkylineContour,
        intrinsics: CameraIntrinsics,
        prior: PosePrior,
    ) -> PoseEstimate:
        """Optimizes camera extrinsics against the observed skyline contour.

        Args:
            terrain: Terrain model used to render candidate skylines.
            contour: Observed image-space skyline contour.
            intrinsics: Known camera intrinsics.
            prior: Noisy coordinate and uncertainty information.

        Returns:
            The best camera extrinsics and fit diagnostics.
        """
```

## Boundary Rules

- `domain/` contains models and lightweight coordinate math only.
- `terrain/` creates terrain and finds peaks, but does not render images.
- `rendering/` projects terrain into camera space and extracts synthetic
  contours.
- `optimization/` owns candidate pose generation, objective scoring, and solver
  orchestration.
- `annotation/` projects known peaks through a pose estimate and draws overlays.
- `io/` reads and writes files, but does not decide how artifacts are produced.
- `web/` writes and serves the static browser viewer for generated artifacts.
- `demo/` composes services into a reproducible synthetic run.
- `cli.py` is a thin adapter over `demo.pipeline`.

## Settings

`peakle.config.AppSettings` should use `pydantic-settings` and contain only
runtime defaults:

- `artifact_dir`: default output directory.
- `random_seed`: default seed for reproducible demos.
- `log_level`: CLI logging level.
- `image_width` and `image_height`: default render size.
- `optimization_max_iterations`: default solver budget.

Project behavior that belongs to a reproducible scene should live in serialized
Pydantic specs, not hidden environment variables.
