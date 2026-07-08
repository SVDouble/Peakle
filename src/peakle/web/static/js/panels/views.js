"use strict";

// Views panel: place cameras, pick/delete views, edit the selected view's pose,
// and — right next to the view — run solvers and pick among that view's solves.
// Selecting a solve drives the map's predicted camera and the Camera/Solve panels.

import { el, formatNumber } from "../format.js";

const PATCH_DEBOUNCE_MS = 250;

export function setupViewsPanel(store, root) {
  root.classList.add("views-panel");

  const placeButton = el("wa-button", { variant: "brand", appearance: "accent", size: "s", text: "Place camera" });
  placeButton.addEventListener("click", () => store.setPlacing(!store.placing));
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
  let editorViewId = undefined;
  let strategyChoice = null;
  let positionPriorChoice = true;
  let solvesHost = null;
  let runStatus = null;

  function renderPlacing() {
    placeButton.textContent = store.placing ? "Click the map to place…" : "Place camera";
    placeButton.setAttribute("variant", store.placing ? "neutral" : "brand");
    hint.textContent = store.placing
      ? "Click the 3D map to drop a camera looking outward."
      : "Place cameras on the map, then solve each view.";
  }

  function renderList() {
    list.replaceChildren(
      ...store.views.map((view) => {
        const selected = view.id === store.selectedViewId;
        const remove = el("button", { type: "button", class: "icon-button", title: "Delete view", text: "✕" });
        remove.addEventListener("click", (event) => {
          event.stopPropagation();
          store.deleteView(view.id);
        });
        return el("li", { class: selected ? "view-row selected" : "view-row" }, [
          el("button", { type: "button", class: "view-name", text: view.label, onclick: () => store.selectView(view.id) }),
          el("span", { class: "view-meta", text: `${view.solves.length} solve${view.solves.length === 1 ? "" : "s"}` }),
          remove,
        ]);
      }),
    );
    if (!store.views.length) {
      list.append(el("li", { class: "view-empty", text: "No views yet — place one on the map." }));
    }
  }

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

  function rebuildEditor(view) {
    if (!view || !view.true_extrinsics) {
      editor.replaceChildren();
      solvesHost = null;
      return;
    }
    const ext = view.true_extrinsics;

    const strategies = store.scene?.strategies ?? [];
    if (!strategyChoice || !strategies.some((s) => s.name === strategyChoice)) {
      strategyChoice = store.scene?.config.default_strategy ?? strategies[0]?.name ?? "powell";
    }
    const strategySelect = el(
      "wa-select",
      { label: "Solver", size: "s", value: strategyChoice },
      strategies.map((s) => el("wa-option", { value: s.name, text: s.label })),
    );
    strategySelect.addEventListener("change", () => {
      strategyChoice = strategySelect.value;
    });

    const runButton = el("wa-button", { variant: "brand", appearance: "accent", size: "s", text: "Run solver" });
    runButton.addEventListener("click", () => runSolve(view.id, strategySelect, runButton));

    const priorSwitch = el("wa-switch", { size: "s", text: "Position prior", hint: "Off: recover position from the skyline alone" });
    if (positionPriorChoice) {
      priorSwitch.setAttribute("checked", "");
    }
    priorSwitch.addEventListener("change", () => {
      positionPriorChoice = priorSwitch.checked;
    });

    runStatus = el("p", { class: "control-hint" });
    solvesHost = el("ul", { class: "solve-list" });

    // Editable name (rename it yourself) + a Duplicate action, grouped in the editor header.
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

    editor.replaceChildren(
      el("div", { class: "control-block compact" }, [
        el("div", { class: "editor-header" }, [nameInput, dupButton]),
        poseSlider("Yaw", ext.yaw_deg, -180, 180, 1, "deg", view.id, (v) => ({ yaw_deg: v })),
        poseSlider("Pitch", ext.pitch_deg, -30, 30, 0.5, "deg", view.id, (v) => ({ pitch_deg: v })),
        poseSlider("Eye height", view.eye_height_m, 0, 1000, 10, "m", view.id, (v) => ({ eye_height_m: v })),
        el("p", { class: "control-hint", text: `East ${Math.round(ext.position.east_m)} m · North ${Math.round(ext.position.north_m)} m` }),
        el("wa-divider"),
        el("span", { class: "control-eyebrow", text: "Solve" }),
        el("div", { class: "solve-actions" }, [strategySelect, runButton]),
        el("div", { class: "solve-option" }, [priorSwitch]),
        runStatus,
        solvesHost,
      ]),
    );
    renderSolves(view);
  }

  function renderSolves(view) {
    if (!solvesHost) {
      return;
    }
    const solves = view.solves;
    solvesHost.replaceChildren(
      ...solves.map((summary) => {
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
    if (!solves.length) {
      solvesHost.append(el("li", { class: "view-empty", text: "No solves yet — run one above." }));
    }
  }

  function strategyLabel(name) {
    return store.scene?.strategies.find((s) => s.name === name)?.label ?? name;
  }

  async function runSolve(viewId, strategySelect, runButton) {
    runButton.disabled = true;
    runButton.setAttribute("loading", "");
    runStatus.textContent = `Solving with ${strategyLabel(strategySelect.value)}…`;
    try {
      const solve = await store.runSolve(viewId, strategySelect.value, { position_prior: positionPriorChoice });
      runStatus.textContent = `Converged in ${solve.result.evaluations} evaluations.`;
    } catch (error) {
      runStatus.textContent = error.message;
    } finally {
      runButton.disabled = false;
      runButton.removeAttribute("loading");
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

  function render() {
    renderList();
    const view = store.selectedView();
    if (view?.id !== editorViewId) {
      editorViewId = view?.id;
      rebuildEditor(view);
    } else if (view) {
      renderSolves(view);
    }
  }

  store.on("placing", renderPlacing);
  store.on("views", render);
  store.on("selection", render);
  renderPlacing();
  render();
}
