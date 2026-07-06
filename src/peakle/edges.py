"""Learned edge / boundary detection for photos (pluggable, optional).

A learned boundary detector finds ridge silhouettes — near *and* faint distant
ranges — while suppressing rock texture and lit slopes, which the classical
depth/tonal response cannot do as cleanly. The default is DexiNed (via `kornia`),
the SOTA generalist edge model; it is optional, so when `kornia`+`torch` are not
installed, ridge extraction falls back to the classical depth-aware response.

A systematic single-filter comparison (results/edges/filters/) found DexiNed by far
the cleanest for this task: classical first-derivative edges, ridge filters
(Sato/Meijering/Frangi) and Canny either drown in rock texture or miss the faint
background; the depth gradient is clean but blind to the (depth-flattened) distant
ranges. The learned detector handles both and is then *weighted* by depth.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class EdgeDetector(Protocol):
    """Produces a dense boundary-strength map in [0, 1] at the input resolution."""

    name: str

    def detect(self, rgb: NDArray[np.float64]) -> NDArray[np.float64]: ...


class LearnedEdges:
    """SOTA learned boundary detector (DexiNed via `kornia`), texture-suppressing."""

    name = "dexined"
    max_width = 1280  # run the model at a capped width for speed/memory
    # DexiNed was trained on BGR images with the ImageNet mean subtracted (no /255
    # scaling). Feeding raw RGB [0, 1] or [0, 255] gives a large DC offset the first
    # conv never saw → a sparse, degraded edge map. These are the official channel
    # means (BGR) used by the reference implementation / IPOL demo.
    mean_bgr = np.array([103.939, 116.779, 123.68], dtype=np.float32)

    def __init__(self) -> None:
        import torch  # noqa: PLC0415 - optional heavy dependency
        from kornia.filters.dexined import DexiNed  # noqa: PLC0415

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = DexiNed(pretrained=True).to(self._device).eval()

    def detect(self, rgb: NDArray[np.float64]) -> NDArray[np.float64]:
        torch = self._torch
        height, width = rgb.shape[:2]
        run_width = min(self.max_width, width)
        run_height = max(1, round(height * run_width / width))
        resized = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
        small = np.asarray(resized.resize((run_width, run_height), Image.LANCZOS), dtype=np.float32)  # RGB [0, 255]
        bgr = small[..., ::-1] - self.mean_bgr  # → BGR, mean-subtracted (official preprocessing)
        tensor = torch.from_numpy(np.ascontiguousarray(bgr.transpose(2, 0, 1)[None])).to(self._device)
        with torch.no_grad():
            edge = torch.sigmoid(self._model(tensor)).squeeze().detach().cpu().numpy().astype(np.float64)
        del tensor
        if self._device == "cuda":
            torch.cuda.empty_cache()  # a long-lived server process must not hoard the shared GPU
        if edge.shape != (height, width):
            scaled = Image.fromarray((np.clip(edge, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
            edge = np.asarray(scaled.resize((width, height), Image.BILINEAR), dtype=np.float64) / 255.0
        return edge


def load_learned_edges() -> EdgeDetector | None:
    """Returns the learned detector if `kornia`+`torch` are installed, else None."""

    try:
        import kornia  # noqa: F401, PLC0415
        import torch  # noqa: F401, PLC0415
    except ImportError:
        return None
    try:
        return LearnedEdges()
    except Exception:  # noqa: BLE001 - model download / GPU init can fail; degrade gracefully
        return None


class TeedEdges:
    """TEED (Tiny and Efficient Edge Detector, xavysp/TEED) — lighter, slightly crisper than
    DexiNed. Loads the cloned repo's `TED` model; BGR + dataset-mean preprocessing."""

    name = "teed"
    max_width = 1024
    mean_bgr = np.array([104.007, 116.669, 122.679], dtype=np.float32)

    def __init__(self, repo_dir: str, ckpt: str) -> None:
        import sys  # noqa: PLC0415

        import torch  # noqa: PLC0415

        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from ted import TED  # noqa: PLC0415 - from the TEED repo

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = TED().to(self._device).eval()
        self._model.load_state_dict(torch.load(ckpt, map_location=self._device))

    def detect(self, rgb: NDArray[np.float64]) -> NDArray[np.float64]:
        torch = self._torch
        h, w = rgb.shape[:2]
        rw = min(self.max_width, w) // 8 * 8
        rh = max(8, round(h * rw / w) // 8 * 8)
        small = np.asarray(
            Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((rw, rh), Image.LANCZOS),
            np.float32,
        )
        bgr = small[..., ::-1] - self.mean_bgr
        t = torch.from_numpy(np.ascontiguousarray(bgr.transpose(2, 0, 1)[None])).to(self._device)
        with torch.no_grad():
            fused = torch.sigmoid(self._model(t)[-1]).squeeze().cpu().numpy().astype(np.float64)
        if fused.shape != (h, w):
            im = Image.fromarray((np.clip(fused, 0, 1) * 255).astype(np.uint8), "L").resize((w, h), Image.BILINEAR)
            fused = np.asarray(im, np.float64) / 255.0
        return fused


class PrecomputedEdges:
    """Reads precomputed edge maps by image name from a directory — the practical way to use a
    heavy model (e.g. DiffusionEdge, run offline via its repo) inside the pipeline."""

    name = "precomputed"

    def __init__(self, maps_dir: str, name: str = "precomputed") -> None:
        self._dir = maps_dir
        self.name = name

    def detect_named(self, image_name: str, shape: tuple[int, int]) -> NDArray[np.float64] | None:
        import os  # noqa: PLC0415

        for ext in (".png", ".jpg"):
            p = os.path.join(self._dir, image_name + ext)
            if os.path.exists(p):
                e = Image.open(p).convert("L").resize((shape[1], shape[0]), Image.LANCZOS)
                return np.asarray(e, np.float64) / 255.0
        return None

    def detect(self, rgb: NDArray[np.float64]) -> NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError("PrecomputedEdges needs detect_named(image_name, shape)")


def load_edge_detector(kind: str = "dexined", **kwargs) -> EdgeDetector | None:
    """Factory for pluggable contour models: 'dexined' (kornia, default), 'teed' (needs
    repo_dir+ckpt), 'diffusionedge'/'precomputed' (needs maps_dir). Returns None if unavailable.
    """
    kind = kind.lower()
    try:
        if kind == "dexined":
            return load_learned_edges()
        if kind == "teed":
            return TeedEdges(kwargs["repo_dir"], kwargs["ckpt"])
        if kind in ("diffusionedge", "precomputed"):
            return PrecomputedEdges(kwargs["maps_dir"], name=kind)
    except Exception:  # noqa: BLE001 - missing repo / ckpt / deps → degrade gracefully
        return None
    raise ValueError(f"unknown edge detector kind: {kind!r}")


def estimate_edges(
    rgb: NDArray[np.float64], detector: EdgeDetector | None = None
) -> NDArray[np.float64] | None:
    """Runs the detector if given, else returns None (classical response is used)."""

    return detector.detect(rgb) if detector is not None else None
