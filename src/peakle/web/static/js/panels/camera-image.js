"use strict";

// Inspect panel: for a placed view, the rendered image with observed vs
// predicted skyline; for a selected GT sample, the photograph with the dataset's
// outline layers (GT / DEM / photo edges / depth) toggleable on top.

import { api } from "../api.js";
import { el, fitContainBox, setBox, svgElement } from "../format.js";

// Layer groups exposed as toggles; each maps to the GT Lab layer PNGs.
const GT_GROUPS = [
  ["gt", "GT outlines", ["gt_sky", "gt_occ", "gt_rib", "gt_cou"]],
  ["dem", "DEM outlines", ["dem_sky", "dem_occ", "dem_rib", "dem_cou"]],
  ["edges", "Photo edges", ["edges"]],
  ["depth", "DEM depth", ["dem_depth"]],
];

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
  const gtControls = el("div", { class: "gt-inspect-controls" }, [
    el(
      "div",
      { class: "gt-toggles" },
      GT_GROUPS.map(([key, label]) => {
        const input = el("input", { type: "checkbox" });
        input.checked = store.gtDisplay[key];
        input.addEventListener("change", () => store.setGtDisplay({ [key]: input.checked }));
        return el("label", { class: `toggle toggle-${key}` }, [input, el("span", { text: label })]);
      }),
    ),
    el("div", { class: "gt-inspect-meta" }),
  ]);
  gtControls.hidden = true;
  root.replaceChildren(el("div", { class: "camera-frame" }, [box]), legend, gtControls, placeholder);
  const gtMeta = gtControls.querySelector(".gt-inspect-meta");

  function poseNonce(view) {
    const e = view.true_extrinsics;
    return e ? `${e.position.east_m}_${e.position.north_m}_${e.yaw_deg}_${e.pitch_deg}_${view.eye_height_m}` : "0";
  }

  function layoutAndDraw() {
    const gtSample = store.selectedGtSample();
    if (gtSample) {
      layoutGtSample(gtSample);
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

    const desiredSrc = `${api.viewImageUrl(view.id)}?p=${encodeURIComponent(poseNonce(view))}`;
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
    overlay.replaceChildren();

    const frame = box.parentElement;
    const fit = fitContainBox(frame.clientWidth, frame.clientHeight, (sample.width || 4) / (sample.height || 3));
    setBox(box, fit);

    const desiredSrc = api.gtLayerUrl(sample.name, "photo");
    if (image.dataset.src !== desiredSrc) {
      image.dataset.src = desiredSrc;
      image.src = desiredSrc;
    }

    const layers = GT_GROUPS.filter(([key]) => store.gtDisplay[key]).flatMap(([, , names]) => names);
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
    gtMeta.innerHTML =
      `<span class="chip ${sample.quality}">${sample.quality}</span> ` +
      `sky <b>${num(sample.sky_cons_px)}px</b> vs ${sample.obs_source ?? "pfm"}` +
      (sample.pfm_cons_px != null ? ` · pfm <b>${num(sample.pfm_cons_px)}px</b>` : "") +
      ` · ct <b>${num(sample.contour_cons_px)}px</b> · Δyaw <b>${num(sample.dyaw_deg)}°</b>` +
      ` · fov <b>${num(sample.fov_deg, 0)}°</b>`;
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
  store.on("selection", layoutAndDraw);
  store.on("views", layoutAndDraw);
  store.on("gt", layoutAndDraw);
  store.on("gt-display", layoutAndDraw);
  layoutAndDraw();
}
