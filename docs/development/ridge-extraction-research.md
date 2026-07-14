# Ridge / skyline outline extraction — methods survey

> **Historical methods survey (non-normative).** This note preserves extraction research options;
> it does not set the current implementation sequence. See the
> [research and development program](../research-and-development.md).

For matching a photo to a DEM, peakle needs to annotate the photo's outlines:
the **skyline** (sky/terrain boundary) and, more valuably, the **internal
ridgelines** (mountain-on-mountain occlusion edges at different depths). This
surveys state-of-the-art and classical methods for both, and what fits peakle.

## The two sub-problems
1. **Skyline** — the topmost sky/terrain boundary. A 1-D curve per column.
2. **Internal ridges / depth layers** — where a nearer ridge is silhouetted
   against farther terrain (a depth discontinuity). Several stacked curves;
   far richer for disambiguating viewpoint, but much harder to extract.

## Non-ML (classical) methods
- **Dynamic programming / shortest path skyline** — the workhorse. Build a
  per-pixel cost (gradient, contrast, "skyness") and find the optimal sky→ground
  boundary as a min-cost left→right path with a smoothness constraint
  (Lie et al. 2005, *robust DP to extract skyline for navigation*). **peakle's
  `extract_dp`.** Robust on clear skies; struggles with haze/cloud and foreground
  texture.
- **Sky segmentation by colour/region** — blue/brightness threshold (Otsu),
  flood-fill from the top, graph-cut/MRF (sky vs ground), GrabCut. Cheap; fails
  on hazy/overcast skies (distant ranges read as sky).
- **Horizon *line* (straight) via edges + Hough/RANSAC** — marine/aerial flat
  horizons; not for mountains.
- **Internal ridges / depth (classical)** — atmospheric perspective: distant
  terrain is washed toward the airlight, so **dark-channel-prior dehazing**
  (He et al.) gives a transmission ≈ depth map, whose **depth discontinuities are
  the ridge crests**. Also curvature/"lapel-point" detection along the skyline,
  morphological ridge filters. **peakle's `depth.HazeDepth` + depth-drop ridges**
  (`segmentation._depth_drop_ridges`), symmetric with the DEM's
  `SyntheticRenderer.ridge_layers`.

## ML methods (state of the art)
- **Semantic sky segmentation** — FCN, SegNet, U-Net, DeepLabv3+. A 2018
  comparison found **FCN** best for mountainous sky segmentation; the skyline is
  the mask boundary. Robust to haze/cloud (learned). Practical SOTA for the
  skyline; the upgrade for cases where peakle's DP fails (e.g. the cloudy
  Gornergrat photo).
- **Dedicated skyline nets** — Porzi et al. (VGG horizon line), heatmap/
  deconvolution regression, *Resource-Efficient Mountainous Skyline Extraction
  using Shallow Learning* (2021, edge classifier + DP hybrid), and **YUNet
  (2025, YOLOv11-based skyline detection)** — recent SOTA.
- **Monocular depth estimation — the biggest lever for internal ridges.**
  **Depth Anything V2** (2024, ViT, SOTA relative depth, excellent on outdoor
  landscapes), **MiDaS/DPT**, **Marigold** (diffusion), **ZoeDepth/Metric3D**
  (metric). A dense depth map → *all* ridge layers as depth discontinuities,
  cleanly separating receding ridges from foreground texture. **peakle wires this
  via `depth.LearnedDepth`** (HF `transformers` depth-estimation, Depth-Anything
  V2 small) as a drop-in for `HazeDepth`.
- **Foundation segmentation** — **SAM / SAM2** (promptable masks) for sky and
  mountain layers; not depth-ordered by itself.
- **Learned edges** — HED, PiDiNet, DexiNed, **EDTER** (transformer) give cleaner
  ridge edges than Sobel; pair with DP.

## Datasets / benchmarks
- **GeoPose3K** (Brejcha & Čadík) — ~3k mountain photos with camera pose +
  rendered DEM depth/silhouettes; the standard for skyline-based geolocalization
  and photo↔render (cross-domain) matching.
- **CH1/CH2** (Baatz, Saurer — Swiss alpine), and the classic **Baboud et al.
  2011** mountain skyline↔DEM alignment. **Saurer et al.** large-scale visual
  geo-localization with skyline + ridge edges. **LandscapeAR / Brejcha**
  cross-domain descriptors (photo↔synthetic).

## Recommendation for peakle
1. **Depth: Depth Anything V2** as the depth source → clean multi-ridge layers
   (already wired via `LearnedDepth`; classical `HazeDepth` is the fallback). The
   single biggest accuracy upgrade for real photos.
2. **Skyline: a learned sky-segmentation** (U-Net/FCN, or SAM2 sky prompt) for
   hazy/cloudy photos where DP grabs the wrong boundary; keep DP as the fast
   default.
3. **Matching:** keep the geometric, confidence-weighted `multi_ridge_residual`;
   consider learned cross-domain descriptors (Brejcha/Saurer) for photo↔render
   robustness later.

## Sources
- [Comparison of Semantic Segmentation Approaches for Horizon/Sky Line Detection (arXiv 1805.08105)](https://arxiv.org/abs/1805.08105)
- [YUNet: Improved YOLOv11 Network for Skyline Detection (arXiv 2502.12449)](https://arxiv.org/html/2502.12449v1)
- [Resource Efficient Mountainous Skyline Extraction using Shallow Learning (arXiv 2107.10997)](https://arxiv.org/pdf/2107.10997)
- [Horizon line detection using supervised learning and edge cues (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1077314218302030)
- [A robust dynamic programming algorithm to extract skyline in images for navigation (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0167865504002302)
- [Camera Geolocation Using Digital Elevation Models in Hilly Area (MDPI)](https://www.mdpi.com/2076-3417/10/19/6661)
- [Camera Geolocation From Mountain Images (GMU)](https://c4i.gmu.edu/~pcosta/F15/data/fileserver/file/472116/filename/Paper_1570111401.pdf)
- Depth Anything V2 — https://depth-anything-v2.github.io ; GeoPose3K — Brejcha & Čadík.
