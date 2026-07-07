"use strict";

// GT dataset panel: the ground-truth corpus inside the main app. Lists every
// sample worst-first with its quality metrics, filters by name, and drives the
// map + inspector: clicking a row selects the sample (photo + layers in the
// Inspect tab, spot highlighted on the map); the ⌖ button recenters the 3D map
// on the sample's location so its spot — and its neighbours — become clickable.

import { el } from "../format.js";
import { geoToLocal } from "./map/gt-spots.js";

const LIST_CAP = 400;

export function setupGtPanel(store, root) {
  root.classList.add("gt-panel");

  const search = el("input", { class: "gt-search", placeholder: "Filter samples…", type: "text" });
  const hint = el("p", { class: "control-hint", text: "Loading GT dataset…" });
  const list = el("ul", { class: "gt-list" });
  const rebuildButton = el("button", { type: "button", class: "gt-rebuild", text: "Rebuild filtered" });

  root.replaceChildren(
    el("div", { class: "control-block gt-block" }, [
      el("div", { class: "gt-head" }, [
        el("span", { class: "control-eyebrow", text: "GT dataset" }),
        rebuildButton,
        el("a", { class: "gt-lab-link", href: "/gt", target: "_blank", text: "GT Lab ↗" }),
      ]),
      search,
      hint,
      list,
    ]),
  );

  // Rebuild the filtered set server-side (pose polish + metrics + tier, manual
  // sidecars respected); the live list picks up fresh records as they land.
  let rebuildTimer = null;

  function filteredNames() {
    const filter = search.value.trim().toLowerCase();
    return (store.gtSamples ?? []).filter((s) => s.name.toLowerCase().includes(filter)).map((s) => s.name);
  }

  async function pollRebuild() {
    try {
      const st = await (await fetch("/api/gt/rebuild")).json();
      if (st.running) {
        const total = st.queue.length;
        const n = st.done.length + st.failed.length;
        hint.textContent = `rebuilding ${n}/${total} — ${st.current ?? "…"}`;
        rebuildTimer = setTimeout(pollRebuild, 3000);
        return;
      }
      rebuildButton.disabled = false;
      rebuildButton.textContent = "Rebuild filtered";
      const failed = st.failed?.length ? ` · ${st.failed.length} failed` : "";
      hint.textContent = `rebuild finished: ${st.done?.length ?? 0} samples${failed}`;
      store.gtSamples = null;
      await store.loadGtSamples();
    } catch {
      rebuildTimer = setTimeout(pollRebuild, 5000);
    }
  }

  rebuildButton.addEventListener("click", async () => {
    const names = filteredNames().slice(0, 50);
    if (!names.length) {
      return;
    }
    if (names.length > 1 && !window.confirm(`Rebuild ${names.length} samples (~${Math.round(names.length * 0.6)} min)?`)) {
      return;
    }
    rebuildButton.disabled = true;
    rebuildButton.textContent = "Rebuilding…";
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
      rebuildButton.disabled = false;
      rebuildButton.textContent = "Rebuild filtered";
      hint.textContent = error.message;
    }
  });

  function inBounds(sample) {
    return store.terrain ? geoToLocal(store.terrain, sample.lat, sample.lon) !== null : false;
  }

  function metricSpan(label, value, gate) {
    if (value === null || value === undefined) {
      return "";
    }
    const text = Math.abs(value) >= 10 ? Math.round(value) : value.toFixed(1);
    return `<span${value > gate ? ' class="over"' : ""}>${label} ${text}</span>`;
  }

  function render() {
    const samples = store.gtSamples;
    if (!samples) {
      return;
    }
    if (store.gtError) {
      hint.textContent = store.gtError;
      list.replaceChildren();
      return;
    }
    const filter = search.value.trim().toLowerCase();
    const rows = samples.filter((s) => s.name.toLowerCase().includes(filter));
    const shown = rows.slice(0, LIST_CAP);
    const onMap = rows.filter(inBounds).length;
    hint.textContent =
      `${rows.length} samples · ${onMap} on the current map` + (rows.length > shown.length ? ` · showing worst ${LIST_CAP}` : "");

    list.replaceChildren(
      ...shown.map((s) => {
        const li = el("li", { class: s.name === store.selectedGtName ? "gt-row sel" : "gt-row" });
        const focus = el("button", {
          type: "button",
          class: "icon-button",
          title: "Center the 3D map here",
          text: "⌖",
        });
        focus.addEventListener("click", async (event) => {
          event.stopPropagation();
          focus.disabled = true;
          try {
            store.selectGtSample(s.name);
            await store.focusScene(s.lat, s.lon);
          } catch (error) {
            hint.textContent = error.message;
          } finally {
            focus.disabled = false;
          }
        });
        // Open the sample as a real scene View (photo + refined pose) — then it lists, POVs,
        // adjusts and SOLVES like any placed camera.
        const openView = el("button", {
          type: "button",
          class: "icon-button",
          title: "Open as a camera view (solvable)",
          text: "→",
        });
        openView.addEventListener("click", async (event) => {
          event.stopPropagation();
          openView.disabled = true;
          hint.textContent = `opening ${s.name} as a view…`;
          try {
            await store.openGtView(s.name);
            hint.textContent = `opened ${s.name} — see the Views tab`;
          } catch (error) {
            hint.textContent = error.message;
          } finally {
            openView.disabled = false;
          }
        });
        li.append(
          el("div", { class: "gt-row-main" }, [
            el("span", { class: "gt-name", text: s.name }),
            el("span", {
              class: "gt-metrics",
              html:
                `<span class="chip ${s.quality}">${s.quality}</span>` +
                metricSpan("sky", s.sky_cons_px, 15) +
                metricSpan("pfm", s.pfm_cons_px, 15) +
                metricSpan("ct", s.contour_cons_px, 25) +
                (inBounds(s) ? '<span class="on-map">on map</span>' : ""),
            }),
          ]),
          el("div", { class: "gt-row-actions" }, [focus, openView]),
        );
        li.addEventListener("click", () => store.selectGtSample(s.name));
        return li;
      }),
    );
    if (!shown.length) {
      list.append(el("li", { class: "view-empty", text: "No samples match." }));
    }
  }

  search.addEventListener("input", render);
  store.on("gt", render);
  store.on("scene", render);
  store.loadGtSamples();
  render();
}
