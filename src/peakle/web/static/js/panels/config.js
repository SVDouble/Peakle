"use strict";

// Setup panel (Web Awesome web components): pick the map provider, seed, render
// size, and default solver, then rebuild the scene (which clears all views
// server-side). A live "Appearance" section switches the map's surface shading
// and contour lines without a rebuild.

import { el } from "../format.js";
import { SHADING_MODES } from "./map/terrain-mesh.js";

function select(label, value, options, onChange) {
  const node = el(
    "wa-select",
    { label, value: String(value), size: "s" },
    options.map(([optValue, optLabel]) => el("wa-option", { value: String(optValue), text: optLabel })),
  );
  if (onChange) {
    node.addEventListener("change", () => onChange(node.value));
  }
  return node;
}

function numberInput(label, value, { min, max, step }) {
  return el("wa-input", {
    label,
    type: "number",
    size: "s",
    value: String(value),
    min: String(min),
    max: String(max),
    step: String(step),
  });
}

export function setupConfigPanel(store, root) {
  root.classList.add("config-panel");
  const form = el("div", { class: "config-form" });
  const status = el("p", { class: "control-hint" });
  root.replaceChildren(form, status);

  function render() {
    const scene = store.scene;
    if (!scene) {
      return;
    }

    const provider = select("Map provider", scene.config.provider, scene.providers.map((kind) => [kind, kind]));
    const seed = numberInput("Seed", scene.config.seed, { min: 0, max: 999999, step: 1 });
    const width = numberInput("Width px", scene.config.image_width, { min: 160, max: 4096, step: 20 });
    const height = numberInput("Height px", scene.config.image_height, { min: 120, max: 4096, step: 20 });
    const fov = numberInput("Horizontal FOV", scene.config.horizontal_fov_deg, { min: 10, max: 170, step: 1 });
    const strategy = select("Default solver", scene.config.default_strategy, scene.strategies.map((entry) => [entry.name, entry.label]));

    const rebuild = el("wa-button", { variant: "brand", appearance: "accent", size: "s", text: "Rebuild scene" });
    rebuild.addEventListener("click", async () => {
      rebuild.disabled = true;
      rebuild.setAttribute("loading", "");
      status.textContent = "Rebuilding scene…";
      try {
        await store.rebuildScene({
          provider: provider.value,
          seed: Number(seed.value),
          image_width: Number(width.value),
          image_height: Number(height.value),
          horizontal_fov_deg: Number(fov.value),
          default_strategy: strategy.value,
        });
        status.textContent = "Scene rebuilt; views cleared.";
      } catch (error) {
        status.textContent = error.message;
      } finally {
        rebuild.disabled = false;
        rebuild.removeAttribute("loading");
      }
    });

    const shading = select(
      "Surface",
      store.display.shadingMode,
      SHADING_MODES.map((mode) => [mode.id, mode.label]),
      (value) => store.setDisplay({ shadingMode: value }),
    );
    const contours = el("wa-switch", { size: "s", text: "Contour lines" });
    if (store.display.contours) {
      contours.setAttribute("checked", "");
    }
    contours.addEventListener("change", () => store.setDisplay({ contours: contours.checked }));

    form.replaceChildren(
      el("span", { class: "control-eyebrow", text: "Scene setup" }),
      provider,
      el("div", { class: "field-row" }, [width, height]),
      seed,
      fov,
      strategy,
      rebuild,
      el("span", { class: "control-eyebrow appearance-eyebrow", text: "Appearance" }),
      shading,
      contours,
    );
  }

  store.on("scene", render);
  render();
}
