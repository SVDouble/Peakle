"use strict";

// Camera panel: shows the selected view's rendered image with the observed
// skyline contour and, when a solve is selected, the predicted skyline overlaid.

import { api } from "../api.js";
import { el, fitContainBox, setBox, svgElement } from "../format.js";

export function setupCameraPanel(store, root) {
  const image = el("img", { class: "camera-image", alt: "Rendered view" });
  const overlay = svgElement("svg", { class: "camera-overlay" });
  overlay.setAttribute("preserveAspectRatio", "none");
  const box = el("div", { class: "frame-box camera-box" }, [image, overlay]);
  const placeholder = el("p", { class: "control-hint", text: "Select or place a view to see its image." });
  const legend = el("div", { class: "camera-image-legend" }, [
    el("span", { html: '<i class="observed-swatch"></i>Observed contour' }),
    el("span", { html: '<i class="predicted-swatch"></i>Predicted contour' }),
  ]);
  root.replaceChildren(el("div", { class: "camera-frame" }, [box]), legend, placeholder);

  function poseNonce(view) {
    const e = view.true_extrinsics;
    return e ? `${e.position.east_m}_${e.position.north_m}_${e.yaw_deg}_${e.pitch_deg}_${view.eye_height_m}` : "0";
  }

  function layoutAndDraw() {
    const view = store.selectedView();
    if (!view) {
      box.hidden = true;
      placeholder.hidden = false;
      return;
    }
    box.hidden = false;
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

  function drawOverlay(view, fit) {
    const elements = [];
    const observed = view.contour;
    if (observed && observed.points.length > 1) {
      const points = observed.points
        .map((p) => `${((p.x_px / observed.image_width_px) * fit.width).toFixed(1)},${((p.y_px / observed.image_height_px) * fit.height).toFixed(1)}`)
        .join(" ");
      elements.push(svgElement("polyline", { class: "observed-line", points }));
    }

    const solve = store.selectedSolve();
    if (solve && view.solves.some((s) => s.id === solve.id)) {
      const result = solve.result;
      const frame = result.trace[result.trace.length - 1];
      const predicted = profilePolyline(frame.profile, result.sample_width, result.sample_height, fit);
      if (predicted) {
        elements.push(svgElement("polyline", { class: "predicted-line", points: predicted }));
      }
    }
    overlay.replaceChildren(...elements);
  }

  function profilePolyline(profile, sampleWidth, sampleHeight, fit) {
    const points = [];
    for (let col = 0; col < profile.length; col += 1) {
      const value = profile[col];
      if (value === null || value === undefined) {
        continue;
      }
      const x = (col / (sampleWidth - 1)) * fit.width;
      const y = (value / sampleHeight) * fit.height;
      points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return points.length > 1 ? points.join(" ") : null;
  }

  const resizeObserver = new ResizeObserver(() => layoutAndDraw());
  resizeObserver.observe(root);
  store.on("selection", layoutAndDraw);
  store.on("views", layoutAndDraw);
  layoutAndDraw();
}
