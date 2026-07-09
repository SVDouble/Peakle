# High-Precision Reference Data

This note tracks data sources that can help explain GT/photo/DEM contour
discrepancies. The key distinction:

- Better annotated photo corpora test the whole pose pipeline.
- Better terrain, surface, name, and orthophoto data test whether the existing
  GT labels are physically reproducible.

As of the current search, GeoPose3K still appears to be the only practical
public mountain-photo corpus with thousands of camera poses plus rendered depth,
normals, illumination, and semantic labels. I did not find a stronger public
corpus that gives hand-verified high-precision mountain skylines, internal
occlusion lines, and calibrated cameras. So the sensible path is not replacing
GeoPose3K immediately; it is auditing it against better reference geometry.

## Recommended Reference Stack

### Switzerland

- `swissALTI3D`: primary bare-earth DTM for Swiss terrain. Use 2 m by default,
  with 0.5 m tiles for pathological sharp ridges if storage and runtime allow.
  Source: https://www.swisstopo.admin.ch/en/height-model-swissalti3d
- `swissSURFACE3D`: classified LiDAR point cloud. Use it to separate bare
  terrain failures from surface-object/vegetation/building occlusions. The
  advertised precision is roughly 20 cm planimetric and 10 cm altimetric.
  Source: https://www.swisstopo.admin.ch/en/height-model-swisssurface3d
- `SWISSIMAGE`: orthophotos at 10 cm in plains/main valleys and 25 cm in the
  Alps. Use it to inspect whether an apparent line in the photo is terrain,
  snow/ice, infrastructure, cloud, or a foreground object.
  Source: https://www.swisstopo.admin.ch/en/orthoimages
- `swissTLM3D`: high-accuracy 3D vector landscape model. Use for named
  landforms, hydrology, buildings, transport, and other features that can
  create non-terrain edges.
  Source: https://www.swisstopo.admin.ch/en/landscape-model-swisstlm3d
- `swissNAMES3D`: official georeferenced names. Use it to replace OSM as the
  Swiss peak/landform naming source. It explicitly includes landform names such
  as peaks and passes.
  Source: https://www.swisstopo.admin.ch/en/landscape-model-swissnames3d

### Cross-Border Alps

Matterhorn-style views are not purely Swiss. A Swiss-only DTM can still miss or
smooth Italian/French background ridges and border transitions.

- `swissALTIRegio`: 10 m cross-border DTM extending at least 100 km beyond the
  Swiss border. It blends swissALTI3D with national models from neighbouring
  countries, so it is useful as a robust fallback when a view crosses borders.
  Source: https://www.swisstopo.admin.ch/en/height-model-swissaltiregio-20240405
- Valle d'Aosta DTM/DSM: regional LiDAR-derived terrain/surface models,
  including 2005/2008 products and download tooling. This is the first place to
  look for the Italian side of the Matterhorn.
  Source: https://geoportale.regione.vda.it/download/dtm/
- Piemonte DTM 5: LiDAR-derived 5 m DTM with stated vertical accuracy around
  +/-0.30 m, or +/-0.60 m in lower-precision wooded/urban areas. Useful for
  wider Italian Alpine coverage.
  Source:
  https://www.geoportale.piemonte.it/geonetwork/srv/api/records/r_piemon%3A224de2ac-023e-441c-9ae0-ea493b217a8e
- Italy HR-DTM-5m: seamless 5 m national DTM integrating LiDAR-derived DTMs and
  TINITALY. It prioritizes morphological continuity over strict point accuracy,
  so it is better as a coverage fallback than as the highest-confidence local
  oracle.
  Source: https://zenodo.org/records/18872933
- France RGE ALTI / LiDAR HD: IGN offers 1 m/5 m terrain data and a national
  LiDAR HD program. This matters for samples whose visible horizon crosses into
  France.
  Source: https://www.data.gouv.fr/datasets/rge-alti-r

## What This Lets Us Diagnose

With these sources wired in, each GT miss can be classified more honestly:

- `terrain_resolution_limit`: Copernicus/SRTM smoothed a sharp ridge, but
  LiDAR DTM reproduces the photo/GeoPose depth line.
- `surface_occluder`: bare-earth DTM fails, DSM/LiDAR surface explains the
  occlusion. This includes buildings, trees, lift pylons, ridgeline structures,
  and sometimes glacier/snow surface effects.
- `border_data_blend`: the error appears near a national-source transition, or
  only when using a Swiss-only patch.
- `photo_not_terrain`: orthophotos/DSM cannot explain the line; likely cloud,
  haze, crop border, watermark, snow contrast, or foreground object.
- `pose_or_camera_label_error`: high-resolution DTM/DSM still cannot reproduce
  the observed skyline under the provided camera model and pose.

## Textured Maps and Synthetic Photos

Textured 3D maps exist, but they should not be treated as a drop-in
replacement for camera-calibrated ground-truth photos.

- `SWISSIMAGE` can texture terrain at 10 cm in plains/main alpine valleys and
  25 cm in the Alps. This is excellent for map-view diagnosis and for seeing
  whether a line is snow, rock, road, or infrastructure, but it is nadir
  orthophoto texture. It will not faithfully reproduce oblique mountain faces,
  shadow direction, clouds, seasonal snow, or the exact historical appearance
  of a Flickr/GeoPose image.
- swisstopo's 3D data packages can combine swissALTI3D terrain, buildings,
  bridge axes, road/rail textures, and optional SWISSIMAGE texturing. This is a
  good base for internal synthetic render tests where we control the camera.
- `swissSURFACE3D Raster` is the useful complement when the photo line is a
  surface occluder rather than bare ground. Render DTM and DSM separately: DTM
  tells us terrain-pose geometry; DSM tells us whether visible permanent surface
  objects explain the photo.
- Google Photorealistic 3D Tiles / Cesium can produce very realistic textured
  oblique meshes in supported areas, but they are an external service with
  licensing/API constraints and incomplete metadata control. They are useful
  for qualitative visualization, not for an auditable solver benchmark unless
  the terms and capture metadata are explicitly compatible.

Therefore the practical plan is:

1. Use orthophoto/3D textured terrain in the web viewer as a diagnostic layer.
2. Keep solver scoring against geometry-derived outlines from DTM/DSM.
3. Use synthetic textured renders only as additional robustness tests, not as
   truth for real GeoPose samples.

## Implementation Plan

1. Add a `ReferenceTerrainProvider` abstraction that can return a prioritized
   stack per bounding box: Swiss LiDAR/ALTI, regional Italian/French data,
   swissALTIRegio, then Copernicus.
2. Extend the GT alignment audit to render the same pose against each available
   source and report both pixel and meter-space deltas.
3. Add a source-sensitivity score: if Copernicus is bad but LiDAR is good, the
   GT is probably usable; if all sources are bad, the GT/camera/photo is suspect.
4. Add optional DSM/surface rendering for occlusion diagnostics, separate from
   the bare-earth DTM used for actual terrain pose scoring.
5. Use official names (`swissNAMES3D`, then OSM fallback) for view search and
   labels, but keep label confidence separate from geometry confidence.

## Expected Outcome

The goal is not to trust high-resolution data blindly. The goal is to stop
mixing failure modes. A sample should only be allowed into solver evaluation
when its photo-derived skyline and internal occlusions are reproducible from at
least one high-confidence reference source under the same camera model.
