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
  // Manual pose adjustment: deltas on top of the record's refined pose, previewed
  // live as a dashed skyline; Save persists them to the record + a sidecar the
  // builder seeds from on future rebuilds.
  const ADJUSTS = [
    ["dyaw", "Δyaw °", -4, 4, 0.05],
    ["de", "ΔE m", -400, 400, 5],
    ["dn", "ΔN m", -400, 400, 5],
    ["dv", "Δv px", -60, 60, 1],
  ];
  const adj = { dyaw: 0, de: 0, dn: 0, dv: 0 };
  const adjInputs = {};
  let adjTimer = null;
  let layerBust = 0;
  const adjStatus = el("span", { class: "adjust-status" });
  const saveButton = el("button", { type: "button", class: "adjust-save", text: "Save adjustment" });
  const resetButton = el("button", { type: "button", class: "adjust-reset", text: "Reset" });
  const adjustBlock = el("details", { class: "gt-adjust" }, [
    el("summary", { text: "Adjust pose" }),
    ...ADJUSTS.map(([key, label, min, max, step]) => {
      const output = el("output", { text: "0" });
      const input = el("input", {
        type: "range", min: String(min), max: String(max), step: String(step), value: "0",
        oninput: () => {
          adj[key] = Number(input.value);
          output.textContent = String(adj[key]);
          store.setGtAdjust({ [key]: adj[key] }); // moves the 3D True-POV camera live
          scheduleAdjustPreview();
        },
      });
      adjInputs[key] = { input, output };
      return el("label", { class: "field" }, [el("span", { text: label }), input, output]);
    }),
    el("div", { class: "adjust-actions" }, [saveButton, resetButton, adjStatus]),
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
        ]),
      ),
    ),
    el("div", { class: "gt-inspect-meta" }),
    adjustBlock,
  ]);
  gtControls.hidden = true;
  const cameraFrame = el("div", { class: "camera-frame" }, [box]);
  root.replaceChildren(cameraFrame, legend, gtControls, placeholder);
  const gtMeta = gtControls.querySelector(".gt-inspect-meta");

  function resetAdjust() {
    clearTimeout(adjTimer);
    for (const key of Object.keys(adj)) {
      adj[key] = 0;
      adjInputs[key].input.value = "0";
      adjInputs[key].output.textContent = "0";
    }
    store.resetGtAdjust();
    adjStatus.textContent = "";
    overlay.replaceChildren();
  }

  function scheduleAdjustPreview() {
    clearTimeout(adjTimer);
    adjTimer = setTimeout(drawAdjustPreview, 400);
  }

  async function drawAdjustPreview() {
    const sample = store.selectedGtSample();
    if (!sample) {
      return;
    }
    adjStatus.textContent = "rendering…";
    try {
      const q = `dyaw=${adj.dyaw}&de=${adj.de}&dn=${adj.dn}&dv=${adj.dv}`;
      const res = await fetch(`/api/gt/samples/${encodeURIComponent(sample.name)}/skyline?${q}`);
      const { rows } = await res.json();
      if (store.selectedGtName !== sample.name) {
        return;
      }
      const fit = { width: box.clientWidth, height: box.clientHeight };
      const points = [];
      for (let col = 0; col < rows.length; col += 1) {
        if (rows[col] !== null) {
          points.push(`${((col / (rows.length - 1)) * fit.width).toFixed(1)},${((rows[col] / sample.height) * fit.height).toFixed(1)}`);
        }
      }
      overlay.setAttribute("viewBox", `0 0 ${fit.width} ${fit.height}`);
      overlay.replaceChildren(svgElement("polyline", { class: "adjust-line", points: points.join(" ") }));
      adjStatus.textContent = "";
    } catch (error) {
      adjStatus.textContent = error.message;
    }
  }

  saveButton.addEventListener("click", async () => {
    const sample = store.selectedGtSample();
    if (!sample) {
      return;
    }
    saveButton.disabled = true;
    adjStatus.textContent = "saving…";
    try {
      const res = await fetch(`/api/gt/samples/${encodeURIComponent(sample.name)}/adjust`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(adj),
      });
      if (!res.ok) {
        throw new Error(`save failed: ${res.status}`);
      }
      const rec = await res.json();
      adjStatus.textContent = `saved · sky ${rec.sky_cons_px}px · ${rec.quality}`;
      layerBust += 1;
      resetButton.disabled = false;
      // metrics changed server-side: reload the corpus list and re-render layers
      store.gtSamples = null;
      await store.loadGtSamples();
    } catch (error) {
      adjStatus.textContent = error.message;
    } finally {
      saveButton.disabled = false;
    }
  });
  resetButton.addEventListener("click", resetAdjust);

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
    if (shownGtName !== null) {
      shownGtName = null;
      resetAdjust();
    }
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

  let shownGtName = null;

  function layoutGtSample(sample) {
    box.hidden = false;
    legend.hidden = true;
    placeholder.hidden = true;
    gtControls.hidden = false;
    if (sample.name !== shownGtName) {
      shownGtName = sample.name;
      resetAdjust();
    }

    const frame = box.parentElement;
    const fit = fitContainBox(frame.clientWidth, frame.clientHeight, (sample.width || 4) / (sample.height || 3));
    setBox(box, fit);

    const bust = layerBust ? `?v=${layerBust}` : "";
    const desiredSrc = api.gtLayerUrl(sample.name, "photo") + bust;
    if (image.dataset.src !== desiredSrc) {
      image.dataset.src = desiredSrc;
      image.src = desiredSrc;
    }

    const layers = GT_LAYER_NAMES.filter((layer) => store.gtDisplay[layer]);
    const want = layers.map((layer) => api.gtLayerUrl(sample.name, layer) + bust);
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
  layoutAndDraw();
}
