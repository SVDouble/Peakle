"use strict";

// Config panel: pick the map provider, seed, render size, and default solver,
// then rebuild the scene. Rebuilding clears all views (server-side).

import { el } from "../format.js";

export function setupConfigPanel(store, root) {
  const fields = {};
  const field = (labelText, control) => el("label", { class: "field" }, [el("span", { text: labelText }), control]);

  fields.provider = el("select");
  fields.seed = el("input", { type: "number", min: "0", step: "1" });
  fields.width = el("input", { type: "number", min: "160", max: "4096", step: "20" });
  fields.height = el("input", { type: "number", min: "120", max: "4096", step: "20" });
  fields.fov = el("input", { type: "number", min: "10", max: "170", step: "1" });
  fields.strategy = el("select");
  const status = el("p", { class: "control-hint", text: "" });
  const rebuild = el("button", {
    class: "primary",
    type: "button",
    text: "Rebuild scene",
    onclick: () => rebuildScene(),
  });

  root.replaceChildren(
    el("div", { class: "control-block" }, [
      el("span", { class: "control-eyebrow", text: "Scene setup" }),
      field("Map provider", fields.provider),
      field("Seed", fields.seed),
      el("div", { class: "field-row" }, [field("Width px", fields.width), field("Height px", fields.height)]),
      field("Horizontal FOV", fields.fov),
      field("Default solver", fields.strategy),
      el("div", { class: "align-actions" }, [rebuild]),
      status,
    ]),
  );

  async function rebuildScene() {
    rebuild.disabled = true;
    status.textContent = "Rebuilding scene…";
    try {
      await store.rebuildScene({
        provider: fields.provider.value,
        seed: Number(fields.seed.value),
        image_width: Number(fields.width.value),
        image_height: Number(fields.height.value),
        horizontal_fov_deg: Number(fields.fov.value),
        default_strategy: fields.strategy.value,
      });
      status.textContent = "Scene rebuilt; views cleared.";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      rebuild.disabled = false;
    }
  }

  function render() {
    const scene = store.scene;
    if (!scene) {
      return;
    }
    fields.provider.replaceChildren(...scene.providers.map((kind) => new Option(kind, kind)));
    fields.provider.value = scene.config.provider;
    fields.strategy.replaceChildren(...scene.strategies.map((s) => new Option(s.label, s.name)));
    fields.strategy.value = scene.config.default_strategy;
    fields.seed.value = scene.config.seed;
    fields.width.value = scene.config.image_width;
    fields.height.value = scene.config.image_height;
    fields.fov.value = scene.config.horizontal_fov_deg;
  }

  store.on("scene", render);
  render();
}
