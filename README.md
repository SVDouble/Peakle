# Peakle

Peakle is a synthetic mountain peak annotation demo. It generates terrain,
renders a pinhole-camera image, extracts a skyline contour, estimates camera
extrinsics from a noisy pose prior, annotates visible peaks, and writes a
browser-based WebGL viewer.

## Quick Start

```bash
uv sync
uv run peakle demo run --output data/demo
uv run peakle web serve --artifact-dir data/demo
```

Then open the URL printed by the server.

Or use Make:

```bash
make sync
make run
```

`make run` regenerates the demo and starts the browser viewer. Use `make demo`
to generate artifacts without starting the server, and `make serve` to serve an
existing artifact directory.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ty check
make check
make build
```

Planning notes live in `docs/development/`.
