"use strict";

// Views panel — ONE list of views, whatever provider they come from. A placed
// camera and a ground-truth sample are the same kind of thing (an image + a pose
// to work with); GT is just a different *provider*, tagged with a chip, not a
// separate corpus in a separate tab. Selecting any row drives the map + inspector;
// the editor below runs solvers and lists that view's solves. A GT view is opened
// (materialized so it can be solved) transparently the first time you solve it.

import { el, formatNumber } from "../format.js";
import { angleDeltaDeg, geoToLocal } from "../geometry.js";

const PATCH_DEBOUNCE_MS = 250;
const CORPUS_CAP = 400;
const PHOTO_MAX_UPLOAD_BYTES = 12 * 1024 * 1024;

export function setupViewsPanel(store, root) {
  root.classList.add("views-panel", "library-panel");

  let patchTimer = null;
  let editorKey = undefined;
  let strategyChoice = null;
  let positionPriorChoice = true;
  let solvesHost = null;
  let runStatus = null;
  let rebuildTimer = null;

  const placeButton = el("button", { type: "button", class: "primary lib-action", text: "Place camera" });
  placeButton.addEventListener("click", () => store.setPlacing(!store.placing));
  const gtLab = el("a", { class: "gt-lab-link", href: "/gt", target: "_blank", text: "GT Lab ↗" });
  const rebuildBtn = el("button", { type: "button", class: "secondary lib-action", text: "Rebuild", title: "Re-derive metrics/pose for the filtered GT views" });
  rebuildBtn.addEventListener("click", onRebuild);
  const search = el("input", { class: "lib-search", placeholder: "Search views or peaks…", type: "text" });
  const photoFile = el("input", { class: "photo-file", type: "file", accept: "image/*", required: true });
  const photoLabel = el("input", { class: "photo-field", type: "text", placeholder: "Label" });
  const photoLat = el("input", { class: "photo-field", type: "number", step: "0.000001", min: "-90", max: "90", placeholder: "Lat", required: true });
  const photoLon = el("input", { class: "photo-field", type: "number", step: "0.000001", min: "-180", max: "180", placeholder: "Lon", required: true });
  const photoFov = el("input", { class: "photo-field", type: "number", step: "0.1", min: "1", max: "179", value: "55", placeholder: "FOV", required: true });
  const photoEye = el("input", { class: "photo-field", type: "number", step: "0.1", min: "0", max: "5000", value: "2", placeholder: "Eye m" });
  const photoSubmit = el("button", { type: "submit", class: "primary", text: "Create view" });
  const photoStatus = el("p", { class: "control-hint" });
  const photoForm = el("form", { class: "photo-import-form", onsubmit: onPhotoSubmit }, [
    photoFile,
    el("div", { class: "photo-import-grid" }, [
      labelField("Label", photoLabel),
      labelField("Lat", photoLat),
      labelField("Lon", photoLon),
      labelField("FOV", photoFov),
      labelField("Eye", photoEye),
    ]),
    photoSubmit,
    photoStatus,
  ]);
  const photoImport = el("details", { class: "photo-import" }, [el("summary", { text: "Localize photo" }), photoForm]);
  const hint = el("p", { class: "control-hint" });
  const list = el("ul", { class: "lib-list" });
  const scroll = el("div", { class: "lib-scroll" }, [list, hint]);
  const editor = el("div", { class: "view-editor" });

  root.replaceChildren(
    el("div", { class: "library-head" }, [
      el("div", { class: "library-head-row" }, [el("span", { class: "control-eyebrow", text: "Views" }), gtLab]),
      el("div", { class: "library-actions" }, [placeButton, rebuildBtn]),
      search,
      photoImport,
    ]),
    scroll,
    editor,
  );

  // --- unified item model -------------------------------------------------
  // Placed/opened views first (your active working set), then the GT provider's
  // catalogue (worst-first). A GT sample already opened as a view is shown once,
  // as the view — never doubled.

  function items() {
    const filter = search.value.trim().toLowerCase();
    const materializedGt = new Set(store.views.filter((v) => v.source === "gt" && v.gt_name).map((v) => v.gt_name));
    const views = store.views
      .map((v) => ({ kind: "view", key: `v:${v.id}`, provider: v.source ?? "placed", label: v.label, view: v, peaks: viewPeakTags(v) }))
      .filter((item) => matchesFilter(item, filter));
    const corpus = (store.gtSamples ?? [])
      .filter((s) => !materializedGt.has(s.name))
      .map((s) => ({ kind: "gt", key: `g:${s.name}`, provider: "gt", label: s.name, sample: s, peaks: s.visible_peaks ?? [] }))
      .filter((item) => matchesFilter(item, filter));
    return { views, corpus };
  }

  function matchesFilter(item, filter) {
    if (!filter) {
      return true;
    }
    const haystack = [item.label, ...(item.peaks ?? []).map((peak) => peak.name)].join(" ").toLowerCase();
    return filter
      .split(/\s+/)
      .filter(Boolean)
      .every((token) => haystack.includes(token));
  }

  function inBounds(sample) {
    return store.terrain ? geoToLocal(store.terrain, sample.lat, sample.lon) !== null : false;
  }

  function providerChip(provider) {
    const label = provider === "gt" ? "gt" : provider === "photo" ? "photo" : "cam";
    return el("span", { class: `provider-chip ${provider}`, text: label });
  }

  function labelField(label, input) {
    return el("label", { class: "photo-label" }, [el("span", { text: label }), input]);
  }

  function metricSpan(label, value, gate) {
    if (value === null || value === undefined) {
      return "";
    }
    const text = Math.abs(value) >= 10 ? Math.round(value) : value.toFixed(1);
    return `<span${value > gate ? ' class="over"' : ""}>${label} ${text}</span>`;
  }

  function viewPeakTags(view, limit = 6) {
    if (view.source === "gt" && view.gt_name) {
      return store.gtByName(view.gt_name)?.visible_peaks ?? [];
    }
    const ext = view.true_extrinsics;
    if (!ext || !store.peaks?.length) {
      return [];
    }
    const fov = view.image_camera?.horizontal_fov_deg ?? store.scene?.config?.horizontal_fov_deg ?? 55;
    const halfFov = Math.max(1, fov / 2);
    const scored = [];
    for (const peak of store.peaks) {
      const dx = peak.local_position.east_m - ext.position.east_m;
      const dy = peak.local_position.north_m - ext.position.north_m;
      const distance = Math.hypot(dx, dy);
      if (distance < 250) {
        continue;
      }
      const bearing = (Math.atan2(dx, dy) * 180) / Math.PI;
      const delta = Math.abs(angleDeltaDeg(bearing, ext.yaw_deg));
      if (delta > halfFov) {
        continue;
      }
      const centrality = Math.max(0, 1 - delta / halfFov);
      const relief = Math.max(1, peak.prominence_m || peak.elevation_m / 20 || 1);
      const weight = centrality * centrality * relief / (1 + distance / 10000);
      scored.push({ name: peak.name, weight });
    }
    return scored.sort((a, b) => b.weight - a.weight || a.name.localeCompare(b.name)).slice(0, limit);
  }

  function selectedKey() {
    if (store.selectedViewId) {
      return `v:${store.selectedViewId}`;
    }
    if (store.selectedGtName) {
      return `g:${store.selectedGtName}`;
    }
    return null;
  }

  function viewRow(item) {
    const view = item.view;
    const selected = item.key === selectedKey();
    return el("li", { class: selected ? "lib-row selected" : "lib-row" }, [
      providerChip(item.provider),
      el("div", { class: "lib-main" }, [
        el("button", { type: "button", class: "lib-name", text: view.label, onclick: () => store.selectView(view.id) }),
        libPeaks(item.peaks, `${view.solves.length} solve${view.solves.length === 1 ? "" : "s"}`),
      ]),
    ]);
  }

  function gtRow(item) {
    const s = item.sample;
    const selected = item.key === selectedKey();
    const select = (event) => {
      event?.stopPropagation();
      store.selectGtSample(s.name, { focus: true });
    };
    const chips =
      `<span class="chip ${s.quality}">${s.quality}</span>` +
      metricSpan("sky", s.sky_cons_px, 15) +
      metricSpan("ct", s.contour_cons_px, 25) +
      (inBounds(s) ? '<span class="on-map">on map</span>' : "");
    return el("li", { class: selected ? "lib-row selected" : "lib-row", onclick: select }, [
      providerChip("gt"),
      el("div", { class: "lib-main" }, [
        el("button", { type: "button", class: "lib-name", text: s.name, onclick: select }),
        libPeaks(item.peaks),
        el("span", { class: "gt-metrics", html: chips }),
      ]),
    ]);
  }

  function libPeaks(peaks, fallback = "") {
    const names = (peaks ?? []).slice(0, 4).map((peak) => peak.name);
    return el("span", { class: "lib-peaks", text: names.length ? names.join(" · ") : fallback });
  }

  function renderList() {
    const { views, corpus } = items();
    const rows = [];
    for (const item of views) {
      rows.push(viewRow(item));
    }
    if (views.length && corpus.length) {
      rows.push(el("li", { class: "lib-divider", text: "GT provider" }));
    }
    const onMap = corpus.filter((c) => inBounds(c.sample)).length;
    for (const item of corpus.slice(0, CORPUS_CAP)) {
      rows.push(gtRow(item));
    }
    list.replaceChildren(...rows);
    if (!views.length && !corpus.length) {
      list.append(el("li", { class: "view-empty", text: store.gtSamples ? "No views match." : "Loading views…" }));
    }
    const capped = corpus.length > CORPUS_CAP ? ` · showing worst ${CORPUS_CAP}` : "";
    hint.textContent = corpus.length
      ? `${views.length} placed · ${corpus.length} GT (${onMap} on the current map)${capped}`
      : `${views.length} view${views.length === 1 ? "" : "s"}`;
  }

  // --- GT provider rebuild (re-derive metrics/pose for the filtered set) ---

  async function pollRebuild() {
    try {
      const st = await (await fetch("/api/gt/rebuild")).json();
      if (st.running) {
        hint.textContent = `rebuilding ${st.done.length + st.failed.length}/${st.queue.length} — ${st.current ?? "…"}`;
        rebuildTimer = setTimeout(pollRebuild, 3000);
        return;
      }
      rebuildBtn.disabled = false;
      rebuildBtn.textContent = "Rebuild filtered";
      store.gtSamples = null;
      await store.loadGtSamples();
    } catch {
      rebuildTimer = setTimeout(pollRebuild, 5000);
    }
  }

  async function onRebuild() {
    const filter = search.value.trim().toLowerCase();
    const names = (store.gtSamples ?? []).filter((s) => s.name.toLowerCase().includes(filter)).map((s) => s.name).slice(0, 50);
    if (!names.length) {
      return;
    }
    if (names.length > 1 && !window.confirm(`Rebuild ${names.length} GT views (~${Math.round(names.length * 0.6)} min)?`)) {
      return;
    }
    rebuildBtn.disabled = true;
    rebuildBtn.textContent = "Rebuilding…";
    try {
      const res = await fetch("/api/gt/rebuild", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ names }),
      });
      if (!res.ok) {
        throw new Error((await res.json()).detail ?? `rebuild failed: ${res.status}`);
      }
      clearTimeout(rebuildTimer);
      pollRebuild();
    } catch (error) {
      rebuildBtn.disabled = false;
      rebuildBtn.textContent = "Rebuild filtered";
      hint.textContent = error.message;
    }
  }

  async function onPhotoSubmit(event) {
    event.preventDefault();
    const file = photoFile.files?.[0];
    if (!file) {
      photoStatus.textContent = "Choose a photo first.";
      return;
    }
    if (file.size > PHOTO_MAX_UPLOAD_BYTES) {
      photoStatus.textContent = "Photo must be 12 MB or smaller.";
      return;
    }
    const lat = Number(photoLat.value);
    const lon = Number(photoLon.value);
    const fov = Number(photoFov.value);
    const eye = Number(photoEye.value || 2.0);
    if (![lat, lon, fov, eye].every(Number.isFinite)) {
      photoStatus.textContent = "Fill in numeric location and FOV.";
      return;
    }
    if (store.scene?.strategies?.some((strategy) => strategy.name === "horizon")) {
      strategyChoice = "horizon";
    }
    photoSubmit.disabled = true;
    photoStatus.textContent = "Creating photo view…";
    try {
      const view = await store.createPhotoView(file, {
        lat_deg: lat,
        lon_deg: lon,
        horizontal_fov_deg: fov,
        eye_height_m: eye,
        label: photoLabel.value.trim() || file.name.replace(/\.[^.]+$/, ""),
      });
      photoStatus.textContent = `Created ${view.label}.`;
      photoImport.open = false;
    } catch (error) {
      photoStatus.textContent = error.message;
    } finally {
      photoSubmit.disabled = false;
    }
  }

  // --- editor (pose + solve) ----------------------------------------------

  function poseSlider(labelText, value, min, max, step, unit, viewId, toChange) {
    const output = el("output", { text: formatNumber(value, unit) });
    const input = el("input", {
      type: "range",
      min: String(min),
      max: String(max),
      step: String(step),
      value: String(value),
      oninput: () => {
        output.textContent = formatNumber(Number(input.value), unit);
        schedulePatch(viewId, toChange(Number(input.value)));
      },
    });
    return el("label", { class: "field" }, [el("span", { text: labelText }), input, output]);
  }

  function positionSlider(labelText, axis, ext, viewId) {
    const terrain = store.terrain;
    const position = ext.position;
    const min = axis === "east_m" ? terrain?.x_min_m : terrain?.y_min_m;
    const max = axis === "east_m" ? terrain?.x_max_m : terrain?.y_max_m;
    const value = position[axis];
    return poseSlider(
      labelText,
      value,
      Math.floor(min ?? value - 1000),
      Math.ceil(max ?? value + 1000),
      10,
      "m",
      viewId,
      (v) => ({ [axis]: v }),
    );
  }

  function solverControls(onRun) {
    const strategies = store.scene?.strategies ?? [];
    if (!strategyChoice || !strategies.some((s) => s.name === strategyChoice)) {
      strategyChoice = store.scene?.config.default_strategy ?? strategies[0]?.name ?? "powell";
    }
    const strategySelect = el(
      "select",
      { class: "solver-select", value: strategyChoice },
      strategies.map((s) => el("option", { value: s.name, text: s.label })),
    );
    strategySelect.value = strategyChoice;
    strategySelect.addEventListener("change", () => (strategyChoice = strategySelect.value));

    const runButton = el("button", { type: "button", class: "primary", text: "Run solver" });
    runButton.addEventListener("click", () => onRun(strategySelect, runButton));

    const priorInput = el("input", { type: "checkbox", checked: positionPriorChoice });
    const priorSwitch = el("label", { class: "toggle solve-option-toggle" }, [
      priorInput,
      el("span", { text: "Position prior" }),
      el("small", { text: "Off: recover position from the skyline alone" }),
    ]);
    priorInput.addEventListener("change", () => (positionPriorChoice = priorInput.checked));

    return { strategySelect, runButton, priorSwitch };
  }

  function rebuildViewEditor(view) {
    const ext = view.true_extrinsics;
    runStatus = el("p", { class: "control-hint" });
    solvesHost = el("ul", { class: "solve-list" });
    if (view.source === "photo" && !view.solves.length && store.scene?.strategies?.some((strategy) => strategy.name === "horizon")) {
      strategyChoice = "horizon";
    }
    const { strategySelect, runButton, priorSwitch } = solverControls((sel, btn) => runSolve(view.id, sel, btn));

    const nameInput = el("input", { class: "view-name-edit", type: "text", value: view.label });
    nameInput.addEventListener("change", () => {
      const label = nameInput.value.trim();
      if (label && label !== view.label) {
        store.patchView(view.id, { label }).catch((error) => (runStatus.textContent = error.message));
      }
    });
    const dupButton = el("button", { type: "button", class: "icon-button", title: "Duplicate this view", text: "⎘" });
    dupButton.addEventListener("click", async () => {
      const label = window.prompt("Name for the duplicate", `${view.label} copy`);
      if (label !== null) {
        try {
          await store.duplicateView(view.id, label.trim() || undefined);
        } catch (error) {
          runStatus.textContent = error.message;
        }
      }
    });
    const deleteButton = el("button", { type: "button", class: "icon-button", title: "Delete this view", text: "✕" });
    deleteButton.addEventListener("click", async () => {
      deleteButton.disabled = true;
      try {
        await store.deleteView(view.id);
      } catch (error) {
        runStatus.textContent = error.message;
        deleteButton.disabled = false;
      }
    });

    editor.replaceChildren(
      el("div", { class: "control-block compact" }, [
        el("div", { class: "editor-header" }, [nameInput, dupButton, deleteButton]),
        positionSlider("East", "east_m", ext, view.id),
        positionSlider("North", "north_m", ext, view.id),
        poseSlider("Yaw", ext.yaw_deg, -180, 180, 1, "deg", view.id, (v) => ({ yaw_deg: v })),
        poseSlider("Pitch", ext.pitch_deg, -30, 30, 0.5, "deg", view.id, (v) => ({ pitch_deg: v })),
        poseSlider("Eye height", view.eye_height_m, 0, 1000, 10, "m", view.id, (v) => ({ eye_height_m: v })),
        el("div", { class: "editor-divider" }),
        el("span", { class: "control-eyebrow", text: "Solve" }),
        el("div", { class: "solve-actions" }, [strategySelect, runButton]),
        el("div", { class: "solve-option" }, [priorSwitch]),
        runStatus,
        solvesHost,
      ]),
    );
    renderSolves(view);
  }

  function rebuildGtEditor(sample) {
    runStatus = el("p", { class: "control-hint" });
    solvesHost = null;
    const openButton = el("button", { type: "button", class: "primary", text: "Open editable view" });
    openButton.addEventListener("click", async () => {
      openButton.disabled = true;
      runStatus.textContent = `Opening ${sample.name}…`;
      try {
        await store.openGtView(sample.name);
      } catch (error) {
        runStatus.textContent = error.message;
        openButton.disabled = false;
      }
    });
    const centerButton = el("button", { type: "button", class: "secondary", text: "Center map" });
    centerButton.addEventListener("click", async () => {
      centerButton.disabled = true;
      runStatus.textContent = `Centering ${sample.name}…`;
      try {
        await store.focusGtSample(sample);
      } catch (error) {
        runStatus.textContent = error.message;
      } finally {
        centerButton.disabled = false;
      }
    });
    const { strategySelect, runButton, priorSwitch } = solverControls((sel, btn) => solveGt(sample, sel, btn));
    editor.replaceChildren(
      el("div", { class: "control-block compact" }, [
        el("div", { class: "editor-header" }, [el("span", { class: "editor-title", text: sample.name }), providerChip("gt")]),
        el("div", { class: "editor-actions" }, [openButton, centerButton]),
        el("div", { class: "editor-divider" }),
        el("span", { class: "control-eyebrow", text: "Solve" }),
        el("div", { class: "solve-actions" }, [strategySelect, runButton]),
        el("div", { class: "solve-option" }, [priorSwitch]),
        runStatus,
      ]),
    );
  }

  function rebuildEditor() {
    const view = store.selectedView();
    if (view && view.true_extrinsics) {
      rebuildViewEditor(view);
      return;
    }
    const sample = store.selectedGtSample?.() ?? (store.selectedGtName ? store.gtByName(store.selectedGtName) : null);
    if (sample) {
      rebuildGtEditor(sample);
      return;
    }
    editor.replaceChildren();
    solvesHost = null;
  }

  function renderSolves(view) {
    if (!solvesHost) {
      return;
    }
    solvesHost.replaceChildren(
      ...view.solves.map((summary) => {
        const selected = summary.id === store.selectedSolveId;
        const remove = el("button", { type: "button", class: "icon-button", title: "Delete solve", text: "✕" });
        remove.addEventListener("click", (event) => {
          event.stopPropagation();
          store.deleteSolve(view.id, summary.id);
        });
        const yawErr = summary.metrics.yaw_error_deg;
        const mae = summary.metrics.contour_mae_px;
        return el("li", { class: selected ? "solve-row selected" : "solve-row" }, [
          el("button", { type: "button", class: "solve-name", onclick: () => store.selectSolve(view.id, summary.id) }, [
            el("span", { class: "solve-strategy", text: strategyLabel(summary.strategy) }),
            el("span", { class: "solve-stat", text: `yaw ${formatNumber(yawErr, "°")} · fit ${formatNumber(mae, "px")}` }),
          ]),
          remove,
        ]);
      }),
    );
    if (!view.solves.length) {
      solvesHost.append(el("li", { class: "view-empty", text: "No solves yet — run one above." }));
    }
  }

  function strategyLabel(name) {
    return store.scene?.strategies.find((s) => s.name === name)?.label ?? name;
  }

  async function runSolve(viewId, strategySelect, runButton) {
    runButton.disabled = true;
    runStatus.textContent = `Solving with ${strategyLabel(strategySelect.value)}…`;
    try {
      const solve = await store.runSolve(viewId, strategySelect.value, { position_prior: positionPriorChoice });
      runStatus.textContent = `Converged in ${solve.result.evaluations} evaluations.`;
    } catch (error) {
      runStatus.textContent = error.message;
    } finally {
      runButton.disabled = false;
    }
  }

  // Solve a GT view: open it as a scene view (recenters the map), then run — from
  // here it is an ordinary view, and the editor rebuilds itself around it.
  async function solveGt(sample, strategySelect, runButton) {
    runButton.disabled = true;
    runStatus.textContent = `Opening ${sample.name} as a view…`;
    try {
      const view = await store.openGtView(sample.name);
      runStatus.textContent = `Solving with ${strategyLabel(strategySelect.value)}…`;
      await store.runSolve(view.id, strategySelect.value, { position_prior: positionPriorChoice });
    } catch (error) {
      if (runStatus) {
        runStatus.textContent = error.message;
      }
    } finally {
      runButton.disabled = false;
    }
  }

  function schedulePatch(viewId, changes) {
    clearTimeout(patchTimer);
    patchTimer = setTimeout(() => {
      store.patchView(viewId, changes).catch((error) => {
        if (runStatus) {
          runStatus.textContent = error.message;
        }
      });
    }, PATCH_DEBOUNCE_MS);
  }

  function renderPlacing() {
    placeButton.textContent = store.placing ? "Click the map to place…" : "Place camera";
    placeButton.classList.toggle("active", store.placing);
  }

  function render() {
    renderList();
    const key = selectedKey();
    if (key !== editorKey) {
      editorKey = key;
      rebuildEditor();
    } else if (store.selectedView()) {
      renderSolves(store.selectedView());
    }
  }

  search.addEventListener("input", renderList);
  store.on("placing", renderPlacing);
  store.on("views", render);
  store.on("selection", render);
  store.on("gt", render);
  store.on("scene", render);
  store.loadGtSamples();
  renderPlacing();
  render();
}
