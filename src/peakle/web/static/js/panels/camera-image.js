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
    el("span", { html: '<i class="observed-swatch"></i>Observed (truth)' }),
    el("span", { html: '<i class="predicted-swatch"></i>Predicted' }),
    el("span", { html: '<i class="diff-swatch"></i>Difference' }),
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
  layoutAndDraw();
}
