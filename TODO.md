# peakle — consolidated notes (fix / to-do)

Collected from the app-feedback session. Grouped by status. Newest feedback at top of each list.

## 🐛 To fix (open bugs)

- [x] **Matterhorn blue DEM skyline too coarse.** The focused Copernicus map grid is denser
      but capped for browser performance; GT POV skylines now use cached swissALTI3D at 5 m
      ray-march resolution when the current map origin is in Switzerland, with adaptive ray
      distances so close terrain is sampled more densely than the far horizon.
- [x] **Peak labels overwhelm the map.** Peak labels/markers now use distance-based LOD: far
      zoom shows only major named peaks, then progressively reveals less prominent and spot-height
      labels as you zoom in.
- [x] **3D map should feel continuous without an infinite mesh.** Panning near the terrain edge
      now loads the next bounded terrain window after pan end, preserving the orbit view while
      avoiding a giant always-loaded heightfield.
- [x] **Duplicate map/geometry helpers removed.** Coordinate conversion, terrain height lookup,
      angular distance, terrain resolution, and scene-refresh flows now use shared helpers.
- [x] **GT photo could be selected on a terrain window where its target mountain was not present.**
      Selecting a GT row or map thumbnail now recenters first when the current terrain is not
      centered on that photo, so the POV/skyline overlay is not drawn against the wrong map chunk.
- [x] **Views panel controls were inconsistent and crowded.** Header actions, solve controls,
      rows, metrics and the selected-item editor were compacted so the list remains readable.
- [x] **Views panel: scroll bar with the lower ~50% empty.** Layout containment fixed; the
      unified list owns the only scrollbar.
- [x] **Peaks sometimes labelled `Pt <height> m` instead of their real name.** OSM peak matching
      now uses nearest-pair assignment with a confident pass and an extended shoulder pass.
- [x] **Map GT image spot needs quick recentering.** Double-clicking a visible photo chip centers
      the map there.

## 🔭 To do (features / redesign)

- [x] **Search views by visible peak names.** GT samples now expose approximate visible-peak tags
      ranked by camera direction/centrality; placed views derive the same tags from the current
      map peaks, and the unified Views search matches those names.
- [x] **One list: merge GT-data list and Views list into a single list.** (Reiterated.)
      The unification is done underneath (a GT sample opens as a real View); this is the
      panel-level merge — one list showing corpus samples + placed/opened views.
- [x] **POV controls → `MAP / POV` + a "solutions" table.** Replace True/Predicted with a
      Map|POV toggle and a small table listing every pose for the image (ground truth + each
      solve) that you pick to look through.
- [x] **Solve arbitrary (non-corpus) photos.** `POST /api/views/from-photo` plus the "Localize
      photo" UI create photo-backed, solvable views from an image + location + FOV.
- [x] **Adjust ANY mutable view uniformly** (not GT catalogue samples). Materialized
      GT/photo/placed views use the same pose sliders; raw GT samples stay immutable baselines.
- [x] **Cleaner interface / remove scattered buttons.** List rows are selection-only; per-item
      actions live in contextual editor groups.
- [x] **(Optional) Make the 2D minimap a separate window/panel** instead of a corner overlay.

## ✅ Done this session

- [x] **GT selection now focuses before rendering overlays** — list rows and 3D photo thumbnails
      both route through the same "center-if-needed" guard, avoiding Matterhorn contours on the
      wrong terrain chunk.
- [x] **Inspect compares two poses for the selected view** — the DEM/refined-pose baseline and
      solver outputs share one candidate model; two dropdown columns compare map-fit metrics only.
- [x] **Inspect is the single pose list** — the duplicate pose rows were removed from Solve;
      Solve only runs solvers and shows diagnostics for the selected solver pose.
- [x] **Solvers sped up for large maps** — objective scoring now uses adaptive strided terrain
      point clouds, global uses bounded coarse candidate polishing, and horizon is the startup default.
- [x] **Added a modern adaptive black-box solver** — CMA-ES is now available as a full-pose
      solver, seeded by the domain-specific horizon orientation and polished locally.
- [x] **Added a contour-database seeding strategy** — projected skyline snapshots are sampled
      from ring viewpoints around the massif, encoded as compact normalized contour signatures,
      ranked against the observed outline, and locally polished with the selected pose priors.
- [x] **GT views expose both built-in poses** — each GT view now lists the immutable
      ground-truth/depth reference separately from the DEM/refined pose before solver outputs.
- [x] **View/pose/camera-model terminology clarified** — the left list is Views
      (image/crop/photo inputs), the second list is Poses (extrinsics), and camera model means
      intrinsics/projection. GT backing views stay hidden and solver runs append poses there.
- [x] **Solver priors are selectable** — position and orientation priors are independent toggles
      where the selected strategy supports them.
- [x] **Opening editable GT views avoids unnecessary terrain reloads** — if the GT camera is
      already inside the current terrain frame, it materializes in that frame instead of recentering.
- [x] **Peak-name search shipped** — visible peak tags are returned for GT samples, shown under
      rows, and used by the Views filter.
- [x] **Views peak search is fuzzy** — typo-tolerant matching now scores view labels and visible
      mountain names, weighted by peak relevance so prominent central peaks rank higher.
- [x] **Views panel compacted** — consistent native buttons/selects, wrapped metrics, peak tags,
      and a shorter editor keep the left column usable.
- [x] **Views panel overflow fixed** — the page/Dockview root no longer scrolls through empty
      space; the unified list owns the only scrollbar.
- [x] **Unified Views list styled and shipped** — placed/opened views and GT corpus samples live
      in one list with provider chips, search, rebuild, center, solve and editor flows.
- [x] **GT map spot recentering added** — double-clicking a visible photo chip centers the map.
- [x] **OSM peak-name fallback tightened** — nearest-pair assignment plus an extended second pass
      reduced the live scene from 7 spot-height labels to 1 legitimate unmatched local maximum.
- [x] **Map POV controls simplified** — the map now has a `Map | POV` toggle plus a compact pose
      table for the selected view's ground-truth/dataset pose and solver poses.
- [x] **Arbitrary photo localization shipped** — raw-image upload endpoint + "Localize photo"
      UI create photo-backed views with extracted skylines, broad yaw/pitch priors, and default
      to the horizon solver.
- [x] **Any view can be moved in one editor** — materialized GT/photo/placed views all expose
      East/North/Yaw/Pitch/Eye sliders, and GT catalogue samples open directly as editable views.
- [x] **Views list row actions consolidated** — rows are selection-only now; duplicate/delete,
      open-editable and center-map actions live in the selected item's editor.
- [x] **2D overview map moved into its own panel** — MapLibre now lives in a docked Overview
      panel, leaving the 3D map panel uncluttered.
- [x] **Map focusing now has clear selection semantics** — GT focus preserves the selected
      sample; generic map focus clears stale GT/view selection.
- [x] **Photo upload path hardened** — arbitrary-photo uploads are streamed with a 12 MB limit
      and rejected before image processing; the UI blocks oversized files before upload.
- [x] **Docs and stale UI code cleaned up** — README now describes the unified Views workflow,
      Localize photo, Map|POV and Overview panel; the old unused GT panel module/CSS is gone.
- [x] **Workbench store events tightened** — scene rebuild clears stale GT selection; photo/GT
      view creation emits `views`; deleting an active view selects the next remaining view.
- [x] Adjust-pose now **moves the 3D POV camera** live (drag → terrain re-aims).
- [x] **Overlay the photo on the 3D terrain** in POV with an opacity slider + a discoverable
      inspector checkbox — for eyeball alignment.
- [x] **Camera marker glyph** shrunk (was ~1 km on the larger scene box).
- [x] **40 km 3D map on startup** (was a small ~14 km chunk); the 2D overview now lives in
      its own docked panel.
- [x] **Rename + duplicate any view** (fork a placed or GT-opened view under a custom name).
- [x] **Open a GT sample as a real, solvable View** (photo + refined pose; cyltan solve).
- [x] `sp_belleetoile2` **is correctly CLEAN** — the green (GT) skyline is right; the blue
      (DEM) is smoother only on the sharp telephoto peaks (7% of columns, ≤11 px) because the
      30 m DEM undersamples them. A resolution limit, not a labeling error → no gate change.

## Notes

- The solver never reports a wrong pose as CONFIRMED; uncertain solves read AMBIGUOUS.
- Solve accuracy in-app is bounded by the focused terrain (40 km / grid 720, coarser than the
  benchmark's 90 km / grid 3000). Good views solve well; harder ones honestly read AMBIGUOUS.
