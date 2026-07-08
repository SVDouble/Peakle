"use strict";

// Inspect panel: for a placed view, the rendered image with observed vs
// predicted skyline; for a selected GT sample, the photograph with the dataset's
// outline layers (GT / DEM / photo edges / depth) toggleable on top.

import { api } from "../api.js";
import { el, fitContainBox, setBox, svgElement } from "../format.js";

// Per-family layer toggles (matching the GT Lab), one checkbox per outline family
// so each can be shown/hidden independently. Each entry: [layer, label, swatch].
// Colors mirror gtlab.py _COLORS (GT = warm/green family, DEM = cool family).
const GT_LAYER_GROUPS = [
  {
    title: "Ground truth (from depth)",
    layers: [
      ["gt_sky", "Skyline", "#00e65a"],
      ["gt_occ", "Occlusion", "#ff961e"],
      ["gt_rib", "Ribs / spurs", "#ffeb3b"],
      ["gt_cou", "Couloirs", "#e86edc"],
    ],
  },
  {
    title: "DEM @ refined pose",
    layers: [
      ["dem_sky", "Skyline", "#00c8ff"],
      ["dem_occ", "Occlusion", "#ff4646"],
      ["dem_rib", "Ribs / spurs", "#50aaff"],
      ["dem_cou", "Couloirs", "#aa5aff"],
    ],
  },
  {
    title: "Other",
    layers: [
      ["edges", "Photo edges", "#f5f5f5"],
      ["gt_depth", "GT depth", "#5aa0d0"],
      ["dem_depth", "DEM depth", "#2a5a80"],
    ],
  },
];
export const GT_LAYER_NAMES = GT_LAYER_GROUPS.flatMap((g) => g.layers.map(([layer]) => layer));

export function setupCameraPanel(store, root) {
  const image = el("img", { class: "camera-image", alt: "Rendered view" });
  const overlay = svgElement("svg", { class: "camera-overlay" });
  overlay.setAttribute("preserveAspectRatio", "none");
  const gtOverlays = el("div", { class: "gt-overlays" });
  const box = el("div", { class: "frame-box camera-box" }, [image, overlay, gtOverlays]);
  const placeholder = el("p", { class: "control-hint", text: "Select a view or a GT sample to inspect it." });
  const legend = el("div", { class: "camera-image-legend" }, [
    el("span", { html: '<i class="observed-swatch"></i>Observed (truth)' }),
    el("span", { html: '<i class="predicted-swatch"></i>Predicted' }),
    el("span", { html: '<i class="diff-swatch"></i>Difference' }),
  ]);
  // Toggle for overlaying the photograph on the 3D terrain in POV (opacity fine-tuned by the map
  // HUD slider). Sits in the inspector so it is discoverable alongside the outline toggles.
  const photoOverlayInput = el("input", { type: "checkbox" });
  syncPhotoOverlayInput();
  photoOverlayInput.addEventListener("change", () => store.setPhotoOpacity(photoOverlayInput.checked ? 0.5 : 0));
  const photoOverlayToggle = el("label", { class: "toggle" }, [
    photoOverlayInput,
    el("span", { class: "swatch photo-overlay-swatch" }),
    el("span", { text: "Overlay photo on 3D map (POV)" }),
  ]);

  const gtControls = el("div", { class: "gt-inspect-controls" }, [
    el(
      "div",
      { class: "gt-toggle-groups" },
      GT_LAYER_GROUPS.map((group) =>
        el("div", { class: "gt-toggle-group" }, [
          el("span", { class: "gt-toggle-title", text: group.title }),
          ...group.layers.map(([layer, label, color]) => {
            const input = el("input", { type: "checkbox" });
            input.checked = !!store.gtDisplay[layer];
            input.addEventListener("change", () => store.setGtDisplay({ [layer]: input.checked }));
            return el("label", { class: "toggle" }, [
              input,
              el("span", { class: "swatch", style: `background:${color}` }),
              el("span", { text: label }),
            ]);
          }),
          ...(group.title === "Other" ? [photoOverlayToggle] : []),
        ]),
      ),
    ),
    el("div", { class: "gt-inspect-meta" }),
  ]);
  gtControls.hidden = true;
  const cameraFrame = el("div", { class: "camera-frame" }, [box]);
  root.replaceChildren(cameraFrame, legend, gtControls, placeholder);
  const gtMeta = gtControls.querySelector(".gt-inspect-meta");

  function syncPhotoOverlayInput() {
    photoOverlayInput.checked = store.photoOpacity > 0;
  }

  function poseNonce(view) {
    const e = view.true_extrinsics;
    return e ? `${e.position.east_m}_${e.position.north_m}_${e.yaw_deg}_${e.pitch_deg}_${view.eye_height_m}` : "0";
  }

  function layoutAndDraw() {
    // One selection source (the unified camera); a camera with precomputed outline layers (a GT
    // sample) renders its photo + layers, otherwise the placed-view render + skyline SVG.
    const cam = store.selectedCamera();
    if (cam?.hasLayers) {
      layoutGtSample(cam.sample);
      return;
    }
    gtControls.hidden = true;
    gtOverlays.replaceChildren();
    const view = store.selectedView();
    if (!view) {
      box.hidden = true;
      legend.hidden = true;
      placeholder.hidden = false;
      return;
    }
    box.hidden = false;
    legend.hidden = false;
    placeholder.hidden = true;

    const frame = box.parentElement;
    const aspect = view.intrinsics.width_px / view.intrinsics.height_px;
    const fit = fitContainBox(frame.clientWidth, frame.clientHeight, aspect);
    setBox(box, fit);
    overlay.setAttribute("viewBox", `0 0 ${fit.width} ${fit.height}`);

    // A GT-derived view shows its PHOTOGRAPH as the base image (with the observed/predicted
    // skyline SVG on top); a synthetic view shows its DEM render.
    const desiredSrc = view.photo_url
      ? view.photo_url
      : `${api.viewImageUrl(view.id)}?p=${encodeURIComponent(poseNonce(view))}`;
    if (image.dataset.src !== desiredSrc) {
      image.dataset.src = desiredSrc;
      image.src = desiredSrc;
    }
    drawOverlay(view, fit);
  }

  function layoutGtSample(sample) {
    box.hidden = false;
    legend.hidden = true;
    placeholder.hidden = true;
    gtControls.hidden = false;

    const frame = box.parentElement;
    const fit = fitContainBox(frame.clientWidth, frame.clientHeight, (sample.width || 4) / (sample.height || 3));
    setBox(box, fit);

    const desiredSrc = api.gtLayerUrl(sample.name, "photo");
    if (image.dataset.src !== desiredSrc) {
      image.dataset.src = desiredSrc;
      image.src = desiredSrc;
    }

    const layers = GT_LAYER_NAMES.filter((layer) => store.gtDisplay[layer]);
    const want = layers.map((layer) => api.gtLayerUrl(sample.name, layer));
    const have = [...gtOverlays.children].map((node) => node.dataset.src);
    if (want.join("|") !== have.join("|")) {
      gtOverlays.replaceChildren(
        ...want.map((src) => {
          const img = el("img", { class: "gt-layer", alt: "" });
          img.dataset.src = src;
          img.src = src;
          return img;
        }),
      );
    }

    const num = (value, digits = 1) => (value === null || value === undefined ? "–" : value.toFixed(digits));
    const reasons = (sample.reasons ?? []).length ? `<div class="gt-reasons">${sample.reasons.join("; ")}</div>` : "";
    gtMeta.innerHTML =
      `<div class="gt-name-row"><span class="gt-inspect-name">${sample.name}</span>` +
      `<span class="chip ${sample.quality}">${sample.quality}</span></div>` +
      `<div class="gt-metrics-row">` +
      `sky <b>${num(sample.sky_cons_px)}px</b> vs ${sample.obs_source ?? "pfm"}` +
      (sample.pfm_cons_px != null ? ` · pfm <b>${num(sample.pfm_cons_px)}px</b>` : "") +
      ` · ct <b>${num(sample.contour_cons_px)}px</b> · Δyaw <b>${num(sample.dyaw_deg)}°</b>` +
      ` · fov <b>${num(sample.fov_deg, 0)}°</b>` +
      ` · pos <b>${num(sample.de_m, 0)}E/${num(sample.dn_m, 0)}N m</b>` +
      (Number.isFinite(sample.lat) ? ` · <b>${num(sample.lat, 4)}, ${num(sample.lon, 4)}</b>` : "") +
      `</div>${reasons}`;
  }

  function drawOverlay(view, fit) {
    const elements = [];
    const observed = view.contour;
    const observedPoints =
      observed && observed.points.length > 1
        ? observed.points.map((p) => [(p.x_px / observed.image_width_px) * fit.width, (p.y_px / observed.image_height_px) * fit.height])
        : null;

    let predictedPoints = null;
    const solve = store.selectedSolve();
    if (solve && view.solves.some((s) => s.id === solve.id)) {
      const result = solve.result;
      // Full-resolution prediction so the contour is as crisp as the observed one.
      predictedPoints = profilePoints(result.predicted_profile_full, view.intrinsics.width_px, view.intrinsics.height_px, fit);
    }

    // Shade the difference between ground-truth (observed) and predicted skylines.
    if (observedPoints && predictedPoints && predictedPoints.length > 1) {
      const polygon = observedPoints.concat([...predictedPoints].reverse());
      elements.push(svgElement("polygon", { class: "diff-fill", points: toPointString(polygon) }));
    }
    if (observedPoints) {
      elements.push(svgElement("polyline", { class: "observed-line", points: toPointString(observedPoints) }));
    }
    if (predictedPoints && predictedPoints.length > 1) {
      elements.push(svgElement("polyline", { class: "predicted-line", points: toPointString(predictedPoints) }));
    }
    overlay.replaceChildren(...elements);
  }

  function profilePoints(profile, sampleWidth, sampleHeight, fit) {
    const points = [];
    for (let col = 0; col < profile.length; col += 1) {
      const value = profile[col];
      if (value === null || value === undefined) {
        continue;
      }
      points.push([(col / (sampleWidth - 1)) * fit.width, (value / sampleHeight) * fit.height]);
    }
    return points;
  }

  function toPointString(points) {
    return points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  }

  const resizeObserver = new ResizeObserver(() => layoutAndDraw());
  resizeObserver.observe(root);
  // The frame shrinks when the GT controls unhide below it — re-fit the box then,
  // or the photo overflows the panel and covers the controls.
  resizeObserver.observe(cameraFrame);
  store.on("selection", layoutAndDraw);
  store.on("views", layoutAndDraw);
  store.on("gt", layoutAndDraw);
  store.on("gt-display", layoutAndDraw);
  store.on("photo-opacity", syncPhotoOverlayInput);
  layoutAndDraw();
}
