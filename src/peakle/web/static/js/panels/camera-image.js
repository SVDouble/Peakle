"use strict";

// Inspect panel: for a placed view, the rendered image with observed vs
// predicted skyline; for a selected GT sample, the photograph with the dataset's
// outline layers (GT / DEM / photo edges / depth) toggleable on top.

import { api } from "../api.js";
import { el, fitContainBox, setBox, svgElement } from "../format.js";
import { candidatePrimaryKey, comparisonText, defaultCompareKeys, poseCandidates, poseTargetKey, selectedPoseTarget } from "../pose-candidates.js";

// The PFM/source-depth skyline is the reference track. GT-v2 substituted outlines and their
// refined-pose DEM reconstruction are diagnostics only and start disabled.
const SOURCE_DEPTH_REFERENCE_LAYERS = [
  ["pfm_sky", "Source-depth / PFM skyline", "#ffd84a"],
];
const GT_V2_DIAGNOSTIC_LAYERS = [
  ["gt_sky", "Substituted skyline", "#00e65a"],
  ["gt_occ", "Occlusion", "#ff961e"],
  ["gt_rib", "Ribs / spurs", "#ffeb3b"],
  ["gt_cou", "Couloirs", "#e86edc"],
  ["dem_sky", "GT-v2 DEM skyline", "#00c8ff"],
  ["dem_occ", "GT-v2 DEM occlusion", "#ff4646"],
  ["dem_rib", "GT-v2 DEM ribs / spurs", "#50aaff"],
  ["dem_cou", "GT-v2 DEM couloirs", "#aa5aff"],
];
const SOLVED_POSE_LAYERS = [
  ["sky", "Skyline", "#ff8c42"],
  ["occ", "Occlusion", "#ff5252"],
  ["rib", "Ribs / spurs", "#ffcc40"],
  ["cou", "Couloirs", "#d768ff"],
  ["depth", "Depth", "#5aa0d0"],
];
const OTHER_LAYERS = [
  ["edges", "Photo edges", "#f5f5f5"],
  ["gt_depth", "GT depth", "#5aa0d0"],
  ["dem_depth", "DEM depth", "#2a5a80"],
];
export const GT_LAYER_NAMES = [...SOURCE_DEPTH_REFERENCE_LAYERS, ...GT_V2_DIAGNOSTIC_LAYERS, ...OTHER_LAYERS].map(([layer]) => layer);

export function setupCameraPanel(store, root) {
  const image = el("img", { class: "camera-image", alt: "Rendered view" });
  const overlay = svgElement("svg", { class: "camera-overlay" });
  overlay.setAttribute("preserveAspectRatio", "none");
  const gtOverlays = el("div", { class: "gt-overlays" });
  const imageLoading = el("div", { class: "camera-image-loading", hidden: true }, [
    el("span", { class: "app-loading-spinner" }),
    el("span", { text: "Loading image..." }),
  ]);
  const box = el("div", { class: "frame-box camera-box" }, [image, overlay, gtOverlays, imageLoading]);
  const placeholder = el("p", { class: "control-hint", text: "Select a view to inspect it." });
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

  const gtToggleGroups = el("div", { class: "gt-toggle-groups" });
  const gtControls = el("div", { class: "gt-inspect-controls" }, [gtToggleGroups, el("div", { class: "gt-inspect-meta" })]);
  gtControls.hidden = true;
  const compareList = el("ul", { class: "inspect-compare-list" });
  const compareA = el("select", { class: "solver-select compare-select", "aria-label": "First pose to compare" });
  const compareB = el("select", { class: "solver-select compare-select", "aria-label": "Second pose to compare" });
  const compareMetricA = el("span", { class: "pose-metrics", text: "map fit -" });
  const compareMetricB = el("span", { class: "pose-metrics", text: "map fit -" });
  const compareStatus = el("p", { class: "control-hint compare-status" });
  const comparePickers = el("div", { class: "inspect-compare-pickers" }, [
    el("label", {}, [el("span", { class: "compare-column-title", text: "Pose A" }), compareA, compareMetricA]),
    el("label", {}, [el("span", { class: "compare-column-title", text: "Pose B" }), compareB, compareMetricB]),
  ]);
  const compareControls = el("div", { class: "inspect-compare-controls" }, [
    el("span", { class: "control-eyebrow", text: "Poses for this view" }),
    compareList,
    comparePickers,
    compareStatus,
  ]);
  compareControls.hidden = true;
  const cameraFrame = el("div", { class: "camera-frame" }, [box]);
  root.replaceChildren(cameraFrame, legend, gtControls, compareControls, placeholder);
  const gtMeta = gtControls.querySelector(".gt-inspect-meta");
  let compareKeys = [];
  let lastCompareTargetKey = "";
  compareA.addEventListener("change", () => {
    updateCompareKeys("a");
    layoutAndDraw();
  });
  compareB.addEventListener("change", () => {
    updateCompareKeys("b");
    layoutAndDraw();
  });
  image.addEventListener("load", () => {
    imageLoading.hidden = true;
  });
  image.addEventListener("error", () => {
    imageLoading.hidden = true;
  });

  function syncPhotoOverlayInput() {
    photoOverlayInput.checked = store.photoOpacity > 0;
  }

  function poseNonce(view) {
    const e = view.true_extrinsics;
    return e ? `${e.position.east_m}_${e.position.north_m}_${e.yaw_deg}_${e.pitch_deg}_${view.eye_height_m}` : "0";
  }

  function layoutAndDraw() {
    renderCandidateCompare();
    // One selection source: a selected view descriptor with camera model and poses. GT catalogue
    // views render their immutable photo + layers; mutable views render their image/SVG overlay.
    const cam = store.selectedViewDescriptor();
    if (cam?.hasLayers) {
      layoutGtSample(cam.sample);
      return;
    }
    const selectedView = store.selectedView();
    const selectedViewSample = selectedView?.gt_name ? store.gtByName(selectedView.gt_name) : null;
    if (selectedViewSample) {
      layoutGtSample(selectedViewSample);
      return;
    }
    gtControls.hidden = true;
    gtOverlays.replaceChildren();
    const view = selectedView;
    if (!view) {
      cameraFrame.hidden = true;
      box.hidden = true;
      legend.hidden = true;
      compareControls.hidden = true;
      placeholder.hidden = false;
      return;
    }
    cameraFrame.hidden = false;
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
      imageLoading.hidden = false;
      image.src = desiredSrc;
    }
    drawOverlay(view, fit);
  }

  function renderCandidateCompare() {
    const targetInfo = selectedPoseTarget(store);
    const targetKey = poseTargetKey(targetInfo);
    const candidates = poseCandidates(store, targetInfo);
    const targetChanged = targetKey !== lastCompareTargetKey;
    lastCompareTargetKey = targetKey;
    syncComparisonKeys(candidates, targetChanged);
    renderGtToggleGroups(candidates, targetInfo);
    compareControls.hidden = candidates.length === 0;
    if (!candidates.length) {
      compareList.replaceChildren();
      compareMetricA.textContent = "map fit -";
      compareMetricB.textContent = "map fit -";
      compareStatus.textContent = "";
      return;
    }
    compareList.replaceChildren(...candidates.map((candidate) => compareRow(candidate, targetInfo)));
    const canCompare = candidates.length >= 2;
    comparePickers.hidden = !canCompare;
    compareStatus.hidden = !canCompare;
    if (!canCompare) {
      compareMetricA.textContent = "map fit -";
      compareMetricB.textContent = "map fit -";
      compareStatus.textContent = "";
      return;
    }
    renderComparePickers(candidates);
    const selected = compareKeys.map((key) => candidates.find((candidate) => candidate.key === key)).filter(Boolean);
    compareStatus.textContent = selected.length === 2 ? comparisonText(selected[0], selected[1]) : "Choose two poses to compare map-fit metrics.";
  }

  function syncComparisonKeys(candidates, targetChanged) {
    const valid = new Set(candidates.map((candidate) => candidate.key));
    compareKeys = compareKeys.filter((key) => valid.has(key)).slice(-2);
    if (targetChanged || compareKeys.length < Math.min(candidates.length, 2)) {
      compareKeys = defaultCompareKeys(candidates, candidatePrimaryKey(store));
    }
  }

  function updateCompareKeys(changed) {
    const targetInfo = selectedPoseTarget(store);
    const candidates = poseCandidates(store, targetInfo);
    let a = compareA.value;
    let b = compareB.value;
    if (a === b && candidates.length > 1) {
      const alternate = candidates.find((candidate) => candidate.key !== (changed === "a" ? a : b))?.key;
      if (changed === "a") {
        b = alternate ?? b;
      } else {
        a = alternate ?? a;
      }
    }
    compareKeys = [a, b].filter(Boolean);
  }

  function renderGtToggleGroups(candidates, targetInfo) {
    const activeKey = activeCandidateKey();
    const groups = candidates
      .map((candidate) => {
        const prefix = candidate.key === activeKey ? "Selected · " : "";
        return poseToggleGroup(`${prefix}${candidate.label}`, poseLayerDefs(candidate, targetInfo));
      })
      .filter(Boolean);
    const diagnostics = diagnosticLayerDefs(targetInfo);
    if (diagnostics.length) {
      groups.push(toggleGroup("GT-v2 diagnostics · not solve evidence", diagnostics));
    }
    groups.push(toggleGroup("Other", OTHER_LAYERS.map(layerDef), [photoOverlayToggle]));
    gtToggleGroups.replaceChildren(...groups);
  }

  function poseToggleGroup(title, layers) {
    return layers.length ? toggleGroup(title, layers) : null;
  }

  function toggleGroup(title, layers, extraNodes = []) {
    return el("div", { class: "gt-toggle-group" }, [
      el("span", { class: "gt-toggle-title", text: title }),
      ...layers.map((layer) => {
        const input = el("input", { type: "checkbox" });
        input.checked = layerChecked(layer);
        input.addEventListener("change", () => store.setGtDisplay({ [layer.key]: input.checked }));
        return el("label", { class: "toggle" }, [
          input,
          el("span", { class: "swatch", style: `background:${layer.color}` }),
          el("span", { text: layer.label }),
        ]);
      }),
      ...extraNodes,
    ]);
  }

  function poseLayerDefs(candidate, targetInfo) {
    if (!candidate) {
      return [];
    }
    if (candidate.key === "gt-depth") {
      return SOURCE_DEPTH_REFERENCE_LAYERS.map(layerDef);
    }
    if (candidate.key === "truth") {
      const view = poseLayerView(targetInfo);
      return view ? SOLVED_POSE_LAYERS.map(([family, label, color]) => poseApiLayer(view.id, "truth", family, label, color)) : [];
    }
    if (candidate.key.startsWith("solve:")) {
      const view = poseLayerView(targetInfo);
      return view
        ? SOLVED_POSE_LAYERS.map(([family, label, color]) => poseApiLayer(view.id, candidate.key, family, label, color))
        : [];
    }
    return [];
  }

  function layerDef([key, label, color, source = "png"]) {
    return { key, label, color, source };
  }

  function poseApiLayer(viewId, poseKey, family, label, color) {
    return {
      key: `${viewId}:${poseKey}:${family}`,
      label,
      color,
      source: "pose",
      viewId,
      poseKey,
      family,
      defaultChecked: family === "sky" && poseKey === activeCandidateKey(),
    };
  }

  function layerChecked(layer) {
    return store.gtDisplay[layer.key] ?? Boolean(layer.defaultChecked);
  }

  function diagnosticLayerDefs(targetInfo) {
    return targetInfo?.sample || targetInfo?.view?.gt_name ? GT_V2_DIAGNOSTIC_LAYERS.map(layerDef) : [];
  }

  function poseLayerView(targetInfo) {
    if (targetInfo?.kind === "gt") {
      return targetInfo.view ?? store.gtViewForSample?.(targetInfo.sample.name) ?? null;
    }
    return targetInfo?.view ?? null;
  }

  function enabledOverlayLayers(candidates, targetInfo) {
    return [
      ...candidates.flatMap((candidate) => poseLayerDefs(candidate, targetInfo)),
      ...diagnosticLayerDefs(targetInfo),
      ...OTHER_LAYERS.map(layerDef),
    ].filter(layerChecked);
  }

  function compareRow(candidate, targetInfo) {
    const selected = candidate.key === activeCandidateKey();
    const select = selectCandidateHandler(candidate, targetInfo);
    const content = [
      el("div", { class: "inspect-compare-copy" }, [
        el("span", { class: "solve-strategy", text: candidate.label }),
        el("span", { class: "solve-stat", text: candidate.stat }),
      ]),
      el("span", { class: "pose-metrics", text: candidate.metricText ?? "map fit -" }),
    ];
    const remove = deleteCandidateButton(candidate, targetInfo);
    if (remove) {
      content.push(remove);
    }
    return el(
      "li",
      {
        class: `inspect-compare-row${selected ? " selected" : ""}${candidate.kind === "truth" ? " truth-row" : ""}${select ? " clickable" : ""}`,
        onclick: select,
      },
      content,
    );
  }

  function activeCandidateKey() {
    return store.activePoseKey?.() ?? (store.selectedSolveId ? `solve:${store.selectedSolveId}` : "truth");
  }

  function selectCandidateHandler(candidate, targetInfo) {
    return () => {
      store.selectPose(targetInfo, candidate.key).catch((error) => {
        compareStatus.textContent = error.message;
      });
    };
  }

  function deleteCandidateButton(candidate, targetInfo) {
    if (candidate.kind !== "solve") {
      return null;
    }
    const view = targetInfo?.kind === "gt" ? targetInfo.view ?? store.gtViewForSample(targetInfo.sample.name) : targetInfo?.view;
    if (!view) {
      return null;
    }
    const remove = el("button", { type: "button", class: "icon-button", title: "Delete solver pose", text: "✕" });
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      store.deleteSolve(view.id, candidate.id);
    });
    return remove;
  }

  function renderComparePickers(candidates) {
    const options = candidates.map((candidate) => el("option", { value: candidate.key, text: candidate.label }));
    compareA.replaceChildren(...options.map((option) => option.cloneNode(true)));
    compareB.replaceChildren(...options.map((option) => option.cloneNode(true)));
    compareA.value = compareKeys[0] ?? candidates[0]?.key ?? "";
    compareB.value = compareKeys[1] ?? candidates[1]?.key ?? candidates[0]?.key ?? "";
    compareMetricA.textContent = candidateMetricText(candidates, compareA.value);
    compareMetricB.textContent = candidateMetricText(candidates, compareB.value);
  }

  function candidateMetricText(candidates, key) {
    return candidates.find((candidate) => candidate.key === key)?.metricText ?? "map fit -";
  }

  function layoutGtSample(sample) {
    cameraFrame.hidden = false;
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
      imageLoading.hidden = false;
      image.src = desiredSrc;
    }

    const targetInfo = selectedPoseTarget(store);
    const layers = enabledOverlayLayers(poseCandidates(store, targetInfo), targetInfo);
    const want = layers.map((layer) => layerSignature(sample, layer));
    const have = [...gtOverlays.children].map((node) => node.dataset.src);
    if (want.join("|") !== have.join("|")) {
      gtOverlays.replaceChildren(
        ...layers.map((layer) => createGtOverlayLayer(sample, layer)).filter(Boolean),
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

  function layerSignature(sample, layer) {
    if (layer.source === "pose") {
      return api.viewPoseLayerUrl(layer.viewId, layer.poseKey, layer.family);
    }
    return api.gtLayerUrl(sample.name, layer.key);
  }

  function createGtOverlayLayer(sample, layer) {
    const src = layer.source === "pose" ? api.viewPoseLayerUrl(layer.viewId, layer.poseKey, layer.family) : api.gtLayerUrl(sample.name, layer.key);
    const img = el("img", { class: "gt-layer", alt: "" });
    img.dataset.src = src;
    img.src = src;
    return img;
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
