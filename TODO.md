# peakle — consolidated notes (fix / to-do)

Collected from the app-feedback session. Grouped by status. Newest feedback at top of each list.

## 🐛 To fix (open bugs)

- [x] **GT photo could be selected on a terrain window where its target mountain was not present.**
      Selecting a GT row or map thumbnail now recenters first when the current terrain is not
      centered on that photo, so the POV/skyline overlay is not drawn against the wrong map chunk.
- [x] **Views panel controls were inconsistent and crowded.** Header actions, solve controls,
      rows, metrics and the selected-item editor were compacted so the list remains readable.
- [x] **Views panel: scroll bar with the lower ~50% empty.** Layout containment fixed; the
      unified list owns the only scrollbar.
- [x] **Peaks sometimes labelled `Pt <height> m` instead of their real name.** OSM peak matching
      now uses nearest-pair assignment with a confident pass and an extended shoulder pass.
- [x] **Map GT image spot has no "center map here" button.** Every visible photo chip now has a
      small center-map action.

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
- [x] **Adjust ANY view uniformly** (not just GT). Placed views have pose sliders; GT samples
      have the inspector adjust. Fold into one "move this view" flow now that duplicate+rename
      exist.
- [x] **Cleaner interface / remove scattered buttons.** List rows are selection-only; per-item
      actions live in contextual editor groups.
- [x] **(Optional) Make the 2D minimap a separate window/panel** instead of a corner overlay.

## ✅ Done this session

- [x] **GT selection now focuses before rendering overlays** — list rows and 3D photo thumbnails
      both route through the same "center-if-needed" guard, avoiding Matterhorn contours on the
      wrong terrain chunk.
- [x] **Peak-name search shipped** — visible peak tags are returned for GT samples, shown under
      rows, and used by the Views filter.
- [x] **Views panel compacted** — consistent native buttons/selects, wrapped metrics, peak tags,
      and a shorter editor keep the left column usable.
- [x] **Views panel overflow fixed** — the page/Dockview root no longer scrolls through empty
      space; the unified list owns the only scrollbar.
- [x] **Unified Views list styled and shipped** — placed/opened views and GT corpus samples live
      in one list with provider chips, search, rebuild, center, solve and editor flows.
- [x] **GT map spot center button added** — every visible photo chip has a small ⌖ action.
- [x] **OSM peak-name fallback tightened** — nearest-pair assignment plus an extended second pass
      reduced the live scene from 7 spot-height labels to 1 legitimate unmatched local maximum.
- [x] **Map POV controls simplified** — the map now has a `Map | POV` toggle plus a compact pose
      table for the selected image's ground-truth/dataset pose and solves.
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
