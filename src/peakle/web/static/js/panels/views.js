"use strict";

// Views panel: toggle the place-camera tool, list views, select/delete, and
// edit the selected view's pose (yaw / pitch / eye height) which re-renders the
// image server-side and clears its solves.

import { el, formatNumber } from "../format.js";

const PATCH_DEBOUNCE_MS = 250;

export function setupViewsPanel(store, root) {
  const placeButton = el("button", { type: "button", class: "primary", onclick: () => togglePlacing() });
  const hint = el("p", { class: "control-hint" });
  const list = el("ul", { class: "view-list" });
  const editor = el("div", { class: "view-editor" });

  root.replaceChildren(
    el("div", { class: "control-block" }, [
      el("span", { class: "control-eyebrow", text: "Views" }),
      placeButton,
      hint,
      list,
      editor,
    ]),
  );

  let patchTimer = null;

  function togglePlacing() {
    store.setPlacing(!store.placing);
  }

  function renderPlacing() {
    placeButton.textContent = store.placing ? "Placing… (click the map)" : "Place camera";
    placeButton.classList.toggle("active", store.placing);
    hint.textContent = store.placing
      ? "Click the 3D map to drop a camera looking outward."
      : "Place cameras on the map, then solve each view.";
  }

  function renderList() {
    list.replaceChildren(
      ...store.views.map((view) => {
        const selected = view.id === store.selectedViewId;
        return el("li", { class: selected ? "view-row selected" : "view-row" }, [
          el("button", { type: "button", class: "view-name", text: view.label, onclick: () => store.selectView(view.id) }),
          el("span", { class: "view-meta", text: `${view.solves.length} solve${view.solves.length === 1 ? "" : "s"}` }),
          el("button", { type: "button", class: "icon-button", title: "Delete", text: "✕", onclick: () => store.deleteView(view.id) }),
        ]);
      }),
    );
    if (!store.views.length) {
      list.append(el("li", { class: "view-empty", text: "No views yet." }));
    }
  }

  function renderEditor() {
    const view = store.selectedView();
    if (!view || !view.true_extrinsics) {
      editor.replaceChildren();
      return;
    }
    const ext = view.true_extrinsics;
    const slider = (labelText, value, min, max, step, unit, onChange) => {
      const output = el("output", { text: `${formatNumber(value, unit)}` });
      const input = el("input", {
        type: "range",
        min: String(min),
        max: String(max),
        step: String(step),
        value: String(value),
        oninput: () => {
          output.textContent = formatNumber(Number(input.value), unit);
          schedulePatch(view.id, onChange(Number(input.value)));
        },
      });
      return el("label", { class: "field" }, [el("span", { text: labelText }), input, output]);
    };

    editor.replaceChildren(
      el("div", { class: "control-block compact" }, [
        el("strong", { text: view.label }),
        slider("Yaw", ext.yaw_deg, -180, 180, 1, "deg", (v) => ({ yaw_deg: v })),
        slider("Pitch", ext.pitch_deg, -30, 30, 0.5, "deg", (v) => ({ pitch_deg: v })),
        slider("Eye height", view.eye_height_m, 0, 1000, 10, "m", (v) => ({ eye_height_m: v })),
        el("p", { class: "control-hint", text: `East ${Math.round(ext.position.east_m)} m · North ${Math.round(ext.position.north_m)} m` }),
      ]),
    );
  }

  function schedulePatch(viewId, changes) {
    clearTimeout(patchTimer);
    patchTimer = setTimeout(() => {
      store.patchView(viewId, changes).catch((error) => {
        hint.textContent = error.message;
      });
    }, PATCH_DEBOUNCE_MS);
  }

  store.on("placing", renderPlacing);
  store.on("views", renderList);
  store.on("views", renderEditor);
  store.on("selection", renderList);
  store.on("selection", renderEditor);
  renderPlacing();
  renderList();
  renderEditor();
}
