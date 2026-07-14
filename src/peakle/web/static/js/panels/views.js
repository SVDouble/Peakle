"use strict";

// Views panel: a left-side library of localized images/crops/photos. GT catalogue rows stay
// immutable; any materialized solver backing view and solver outputs live under that view
// as poses.

import { el, formatNumber } from "../format.js";
import { fuzzySearchScore, hasFuzzyMatch } from "../fuzzy-search.js";
import { angleDeltaDeg, geoToLocal } from "../geometry.js";

const PATCH_DEBOUNCE_MS = 250;
const CORPUS_CAP = 400;
const PHOTO_MAX_UPLOAD_BYTES = 12 * 1024 * 1024;
const SORT_OPTIONS = [
  { key: "relevance", label: "Matches", defaultDirection: "desc" },
  { key: "name", label: "Name", defaultDirection: "asc" },
  { key: "on-map", label: "Map", defaultDirection: "desc" },
  { key: "peaks", label: "Peaks", defaultDirection: "desc" },
  { key: "solves", label: "Poses", defaultDirection: "desc" },
];

export function setupViewsPanel(store, root) {
  root.classList.add("views-panel", "library-panel");

  let patchTimer = null;
  let editorKey = undefined;
  let runStatus = null;
  let sortKey = "on-map";
  let sortDirection = "desc";
  let searching = false;
  let browseSort = { key: sortKey, direction: sortDirection };

  const placeButton = el("button", { type: "button", class: "primary lib-action", text: "Place view" });
  placeButton.addEventListener("click", () => store.setPlacing(!store.placing));
  const gtLab = el("a", { class: "gt-lab-link", href: "/gt", target: "_blank", text: "GT Lab ↗" });
  const benchmarks = el("a", {
    class: "gt-lab-link",
    href: "/bench",
    target: "_blank",
    text: "Bench ↗",
    title: "Open pose benchmark results and dataset compatibility statistics",
  });
  const search = el("input", {
    class: "lib-search",
    placeholder: "Search views and peaks…",
    type: "search",
    "aria-label": "Search views",
  });
  const sortButtons = new Map();
  const sortRail = el("div", { class: "lib-sort-rail", role: "toolbar", "aria-label": "Sort views" });
  for (const option of SORT_OPTIONS) {
    const button = el("button", { type: "button", class: "lib-sort-button", text: option.label });
    button.addEventListener("click", () => activateSort(option));
    sortButtons.set(option.key, button);
    sortRail.append(button);
  }
  const sortControl = el("div", { class: "lib-sort-control" }, [
    el("span", { class: "lib-sort-label", text: "Sort" }),
    sortRail,
  ]);
  const resultSummary = el("span", { class: "library-summary", "aria-live": "polite" });
  const photoFile = el("input", { class: "photo-file", type: "file", accept: "image/*", required: true });
  const photoLabel = el("input", { class: "photo-field", type: "text", placeholder: "Label" });
  const photoLat = el("input", { class: "photo-field", type: "number", step: "0.000001", min: "-90", max: "90", placeholder: "Lat", required: true });
  const photoLon = el("input", { class: "photo-field", type: "number", step: "0.000001", min: "-180", max: "180", placeholder: "Lon", required: true });
  const photoFov = el("input", { class: "photo-field", type: "number", step: "0.1", min: "1", max: "179", value: "55", placeholder: "FOV", required: true });
  const photoEye = el("input", { class: "photo-field", type: "number", step: "0.1", min: "0", max: "5000", value: "2", placeholder: "Eye m" });
  const photoSubmit = el("button", { type: "submit", class: "primary", text: "Create view" });
  const photoCancel = el("button", { type: "button", class: "secondary", text: "Cancel" });
  const photoStatus = el("p", { class: "control-hint" });
  const photoForm = el("form", { class: "photo-import-form", onsubmit: onPhotoSubmit }, [
    labelField("Photo", photoFile, "photo-file-label"),
    el("div", { class: "photo-import-grid" }, [
      labelField("Label", photoLabel),
      labelField("Lat", photoLat),
      labelField("Lon", photoLon),
      labelField("FOV", photoFov),
      labelField("Eye", photoEye),
    ]),
    photoStatus,
    el("div", { class: "photo-dialog-actions" }, [photoCancel, photoSubmit]),
  ]);
  const photoDialog = el("dialog", { class: "photo-dialog", "aria-labelledby": "photoDialogTitle", "aria-describedby": "photoDialogDescription" }, [
    el("div", { class: "photo-dialog-shell" }, [
      el("h2", { id: "photoDialogTitle", text: "Add photo view" }),
      el("p", {
        id: "photoDialogDescription",
        class: "photo-dialog-description",
        text: "Choose a mountain photo and provide its approximate location and camera settings.",
      }),
      photoForm,
    ]),
  ]);
  const addPhotoButton = el("button", { type: "button", class: "secondary lib-action", text: "Add photo" });
  addPhotoButton.addEventListener("click", () => {
    photoStatus.textContent = "";
    photoDialog.showModal();
    requestAnimationFrame(() => photoFile.focus());
  });
  photoCancel.addEventListener("click", () => photoDialog.close());
  photoDialog.addEventListener("close", () => addPhotoButton.focus());
  photoDialog.addEventListener("click", (event) => {
    if (event.target === photoDialog) {
      photoDialog.close();
    }
  });
  const hint = el("p", { class: "control-hint", role: "status", "aria-live": "polite" });
  const list = el("ul", { class: "lib-list" });
  const scroll = el("div", { class: "lib-scroll" }, [list, hint]);
  const editor = el("div", { class: "view-editor" });

  root.replaceChildren(
    el("div", { class: "library-head" }, [
      el("div", { class: "library-head-row" }, [
        el("span", { class: "control-eyebrow", text: "Views" }),
        el("nav", { class: "library-head-links", "aria-label": "View tools" }, [gtLab, benchmarks]),
      ]),
      el("div", { class: "library-actions" }, [placeButton, addPhotoButton]),
      search,
      el("div", { class: "library-list-toolbar" }, [resultSummary, sortControl]),
    ]),
    scroll,
    editor,
    photoDialog,
  );

  // --- unified view model -------------------------------------------------
  // Placed/photo views are localized images/crops/photos. GT catalogue samples are immutable
  // views. Materialized GT views are hidden backing poses, not duplicate rows in the view library.

  function items() {
    const filter = search.value.trim();
    const views = store.views
      .filter((v) => v.source !== "gt")
      .map((v) => ({ kind: "view", key: `v:${v.id}`, provider: v.source ?? "placed", label: v.label, view: v, peaks: viewPeakTags(v) }))
      .map((item) => itemWithSearchScore(item, filter))
      .filter((item) => matchesFilter(item, filter));
    const corpus = (store.gtSamples ?? [])
      .map((s) => ({ kind: "gt", key: `g:${s.name}`, provider: "gt", label: s.name, sample: s, peaks: s.visible_peaks ?? [] }))
      .map((item) => itemWithSearchScore(item, filter))
      .filter((item) => matchesFilter(item, filter));
    return {
      rows: sortItems([...views, ...corpus]),
      viewCount: views.length,
      corpusCount: corpus.length,
    };
  }

  function sortItems(rows) {
    return [...rows].sort(compareItems);
  }

  function compareItems(a, b) {
    const direction = sortDirection === "asc" ? 1 : -1;
    let result = 0;
    if (sortKey === "relevance") {
      result = a.searchScore - b.searchScore;
    } else if (sortKey === "name") {
      result = a.label.localeCompare(b.label);
    } else if (sortKey === "on-map") {
      result = Number(isItemOnMap(a)) - Number(isItemOnMap(b));
    } else if (sortKey === "peaks") {
      result = peakWeight(a) - peakWeight(b);
    } else if (sortKey === "solves") {
      result = solveCount(a) - solveCount(b);
    }
    return direction * result || a.label.localeCompare(b.label);
  }

  function activateSort(option) {
    if (sortKey === option.key) {
      if (option.key !== "relevance") {
        sortDirection = sortDirection === "asc" ? "desc" : "asc";
      }
    } else {
      sortKey = option.key;
      sortDirection = option.defaultDirection;
    }
    if (!searching) {
      browseSort = { key: sortKey, direction: sortDirection };
    }
    updateSortControls();
    renderList();
  }

  function sortDescription(key, direction) {
    if (key === "relevance") {
      return "Best matches first";
    }
    if (key === "name") {
      return direction === "asc" ? "Name A to Z" : "Name Z to A";
    }
    if (key === "on-map") {
      return direction === "desc" ? "On-map views first" : "Off-map views first";
    }
    if (key === "peaks") {
      return direction === "desc" ? "Most relevant peaks first" : "Least relevant peaks first";
    }
    return direction === "desc" ? "Most poses first" : "Fewest poses first";
  }

  function updateSortControls() {
    for (const option of SORT_OPTIONS) {
      const button = sortButtons.get(option.key);
      const active = sortKey === option.key;
      button.hidden = option.key === "relevance" && !searching;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      const direction = active ? sortDirection : option.defaultDirection;
      button.title = `${sortDescription(option.key, direction)}${active && option.key !== "relevance" ? "; click to reverse" : ""}`;
      button.setAttribute("aria-label", button.title);
      button.replaceChildren(
        el("span", { text: option.label }),
        ...(active ? [el("span", { class: "lib-sort-direction", text: direction === "asc" ? "↑" : "↓" })] : []),
      );
    }
  }

  function isItemOnMap(item) {
    return item.sample ? inBounds(item.sample) : true;
  }

  function peakWeight(item) {
    return item.peaks?.[0]?.weight ?? 0;
  }

  function solveCount(item) {
    if (item.sample) {
      return store.gtViewForSample(item.sample.name)?.solves?.length ?? 0;
    }
    return item.view?.solves?.length ?? 0;
  }

  function matchesFilter(item, filter) {
    if (!filter) {
      return true;
    }
    return hasFuzzyMatch(filter, itemSearchFields(item));
  }

  function itemWithSearchScore(item, filter) {
    return { ...item, searchScore: filter ? fuzzySearchScore(filter, itemSearchFields(item)) : 1 };
  }

  function itemSearchFields(item) {
    return [
      { text: item.label, weight: 0.85 },
      ...(item.peaks ?? []).map((peak) => ({
        text: peak.name,
        weight: 1 + Math.min(0.6, Math.log1p(Math.max(0, peak.weight ?? 0)) / 7),
      })),
    ];
  }

  function inBounds(sample) {
    return store.terrain ? geoToLocal(store.terrain, sample.lat, sample.lon) !== null : false;
  }

  function providerChip(provider) {
    const label = provider === "gt" ? "gt" : provider === "photo" ? "photo" : "cam";
    return el("span", { class: `provider-chip ${provider}`, text: label });
  }

  function labelField(label, input, extraClass = "") {
    return el("label", { class: `photo-label${extraClass ? ` ${extraClass}` : ""}` }, [el("span", { text: label }), input]);
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
      el("div", { class: "lib-row-head" }, [
        providerChip(item.provider),
        el("button", { type: "button", class: "lib-name", text: rowTitle(item), title: view.label, onclick: () => store.selectView(view.id) }),
      ]),
      rowDetail(item, view.label),
      el("span", { class: "gt-metrics compact", text: `${view.solves.length} pose${view.solves.length === 1 ? "" : "s"}` }),
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
      `<span class="source-tag">${s.manual ? "MANUAL" : "AUTO"}</span>` +
      (inBounds(s) ? '<span class="on-map">on map</span>' : "");
    return el("li", { class: selected ? "lib-row selected" : "lib-row", onclick: select }, [
      el("div", { class: "lib-row-head" }, [
        providerChip("gt"),
        el("button", { type: "button", class: "lib-name", text: rowTitle(item), title: s.name, onclick: select }),
      ]),
      rowDetail(item, s.name),
      el("span", { class: "gt-metrics", html: chips }),
    ]);
  }

  function rowTitle(item) {
    const names = (item.peaks ?? []).slice(0, 2).map((peak) => peak.name);
    return names.length ? names.join(" · ") : item.label;
  }

  function rowDetail(item, fallback = "") {
    const names = (item.peaks ?? []).slice(2, 6).map((peak) => peak.name);
    const detail = names.length ? `${names.join(" · ")} · ${fallback}` : fallback;
    return el("span", { class: "lib-peaks", text: detail, title: detail });
  }

  function renderList() {
    const { rows: sortedRows, viewCount, corpusCount } = items();
    let shownCorpus = 0;
    const visibleRows = sortedRows.filter((item) => {
      if (item.kind !== "gt") {
        return true;
      }
      shownCorpus += 1;
      return shownCorpus <= CORPUS_CAP;
    });
    list.replaceChildren(...visibleRows.map((item) => (item.kind === "gt" ? gtRow(item) : viewRow(item))));
    if (!sortedRows.length) {
      list.append(el("li", { class: "view-empty", text: store.gtSamples ? "No views match." : "Loading views…" }));
    }
    const onMap = sortedRows.filter((item) => item.kind === "gt" && inBounds(item.sample)).length;
    const shown = visibleRows.length;
    const total = sortedRows.length;
    resultSummary.textContent = total > shown ? `${shown} of ${total} views` : `${total} view${total === 1 ? "" : "s"}`;
    resultSummary.title = corpusCount
      ? `${viewCount} placed · ${corpusCount} GT · ${onMap} on the current map`
      : `${viewCount} placed view${viewCount === 1 ? "" : "s"}`;
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
      photoDialog.close();
      photoForm.reset();
      photoStatus.textContent = "";
      hint.textContent = `Created ${view.label}.`;
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

  function rebuildViewEditor(view) {
    const ext = view.true_extrinsics;
    runStatus = el("p", { class: "control-hint" });

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
        runStatus,
      ]),
    );
  }

  function rebuildGtEditor(sample) {
    runStatus = el("p", { class: "control-hint" });
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
    editor.replaceChildren(
      el("div", { class: "control-block compact" }, [
        el("div", { class: "editor-header" }, [el("span", { class: "editor-title", text: sample.name }), providerChip("gt")]),
        el("div", { class: "editor-actions" }, [centerButton]),
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
    placeButton.textContent = store.placing ? "Click map to place view…" : "Place view";
    placeButton.classList.toggle("active", store.placing);
  }

  function render() {
    renderList();
    const key = selectedKey();
    if (key !== editorKey) {
      editorKey = key;
      rebuildEditor();
    }
  }

  search.addEventListener("input", () => {
    const nextSearching = Boolean(search.value.trim());
    if (nextSearching && !searching) {
      browseSort = { key: sortKey, direction: sortDirection };
      sortKey = "relevance";
      sortDirection = "desc";
    } else if (!nextSearching && searching && sortKey === "relevance") {
      sortKey = browseSort.key;
      sortDirection = browseSort.direction;
    }
    searching = nextSearching;
    updateSortControls();
    renderList();
  });
  store.on("placing", renderPlacing);
  store.on("views", render);
  store.on("selection", render);
  store.on("gt", render);
  store.on("scene", render);
  store.loadGtSamples();
  updateSortControls();
  renderPlacing();
  render();
}
