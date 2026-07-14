"""SAM 3 text-promptable segmentation backend (gated `facebook/sam3`).

SAM 3 (Meta, 2025) segments *concepts* from a text prompt rather than points, so a single
prompt like ``"sky"`` yields a clean sky mask and ``"mountain"`` yields one instance per
distinct range/peak. It implements the two `Segmenter` primitives (`sky_mask`,
`instance_masks`); the shared `silhouette_map` / `terrain_layers` come from the base class.

Gated model: requires a Hugging Face token with access to `facebook/sam3`
(``huggingface-cli login`` or ``HF_TOKEN``). When transformers/torch or the weights are
unavailable, `load_segmenter` returns None and callers fall back to a point-prompted backend.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from peakle.segmenters import Segmenter

MODEL_ID = "facebook/sam3"


class Sam3Segmenter(Segmenter):
    """Text-promptable sky / terrain-instance segmentation via `facebook/sam3`."""

    name = "sam3"

    def __init__(self, model_id: str = MODEL_ID, concept: str = "mountain") -> None:
        import torch  # noqa: PLC0415
        from transformers import Sam3Model, Sam3Processor  # noqa: PLC0415

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Sam3Processor.from_pretrained(model_id)
        # The optional transformers model uses a decorated ``to`` method whose
        # runtime overloads are not represented correctly by its current stubs.
        model = cast(Any, Sam3Model.from_pretrained(model_id))
        self._model = model.to(self._device).eval()
        self.concept = concept  # text prompt used for terrain instances

    def _to_pil(self, rgb: NDArray[np.float64]):
        from PIL import Image  # noqa: PLC0415

        return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), "RGB")

    def _instances(self, rgb: NDArray[np.float64], text: str, threshold: float):
        """(masks[N,H,W] bool, scores[N]) for the concept ``text`` at ``threshold``."""
        torch = self._torch
        image = self._to_pil(rgb)
        inputs = self._processor(images=image, text=text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        res = self._processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5, target_sizes=[image.size[::-1]]
        )[0]
        masks = res["masks"]
        masks = masks.cpu().numpy().astype(bool) if hasattr(masks, "cpu") else np.asarray(masks, bool)
        if masks.ndim == 2:
            masks = masks[None]
        return masks

    # ---- Segmenter primitives ----
    def sky_mask(self, rgb: NDArray[np.float64], threshold: float = 0.3) -> NDArray[np.bool_]:
        masks = self._instances(rgb, "sky", threshold)
        if len(masks) == 0:
            return np.zeros(rgb.shape[:2], bool)
        sky = masks.any(0)
        from scipy.ndimage import label  # noqa: PLC0415

        lab, n = label(sky)  # keep only sky connected to the top of the frame
        if n:
            top = set(np.unique(lab[:3])) - {0}
            sky = np.isin(lab, list(top)) if top else sky
        return sky

    def instance_masks(self, rgb: NDArray[np.float64], threshold: float) -> list[NDArray[np.bool_]]:
        return list(self._instances(rgb, self.concept, threshold))


def load_sam3() -> Sam3Segmenter | None:
    """Back-compat helper; prefer `peakle.segmenters.load_segmenter('sam3')`."""
    try:
        import torch  # noqa: F401, PLC0415
        import transformers  # noqa: F401, PLC0415
    except ImportError:
        return None
    try:
        return Sam3Segmenter()
    except Exception:  # noqa: BLE001 - gated/missing weights, no token, OOM → degrade gracefully
        return None
