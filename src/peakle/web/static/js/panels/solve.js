"use strict";

// Solve inspector: read-only detail for the currently selected solve. Animates
// the backend's convergence trace, shows fit/error metrics, and overlays the
// observed vs. predicted skyline. Running solvers and picking among them lives
// in the Views panel.

import { angleDeltaDeg, distanceMeters } from "../geometry.js";
import { el, formatDistance, formatNumber, svgElement } from "../format.js";

export function setupSolvePanel(store, root) {
  root.classList.add("solve-panel");
  const blurb = el("p", { class: "control-hint" });
  const status = el("p", { class: "control-hint", text: "Select a solve in the Views panel to inspect it." });

  const readouts = {
    confidence: metric("Confidence"),
    yaw: metric("Yaw error"),
    pitch: metric("Pitch error"),
    position: metric("Position"),
    fit: metric("Fit MAE"),
    evals: metric("Evaluations"),
  };
  const plot = svgElement("svg", { class: "solve-plot" });
  plot.setAttribute("preserveAspectRatio", "none");

  root.replaceChildren(
    el("div", { class: "control-block" }, [
      el("span", { class: "control-eyebrow", text: "Solve inspector" }),
      blurb,
      status,
      el("dl", { class: "metric-grid" }, Object.values(readouts).map((m) => m.node)),
      el("div", { class: "solve-plot-wrap" }, [
        plot,
        el("div", { class: "solve-plot-legend" }, [
          el("span", { html: '<i class="observed-swatch"></i>Observed' }),
          el("span", { html: '<i class="predicted-swatch"></i>Predicted' }),
        ]),
      ]),
    ]),
  );

  let animation = 0;
  let shownSolveId = null;

  function metric(label) {
    const value = el("dd", { text: "-" });
    return { node: el("div", {}, [el("dt", { text: label }), value]), value };
  }

  function strategyBlurb(name) {
    return store.scene?.strategies.find((s) => s.name === name)?.blurb ?? "";
  }

  function clearReadouts() {
    for (const readout of Object.values(readouts)) {
      readout.value.textContent = "-";
    }
    plot.replaceChildren();
  }

  function animate(solve, view) {
    cancelAnimationFrame(animation);
    const frames = solve.result.trace;
    let index = 0;
    const step = () => {
      showFrame(frames[index], solve.result, view);
      if (index < frames.length - 1) {
        index += 1;
        animation = requestAnimationFrame(step);
      } else {
        showFinal(solve, view);
      }
    };
    step();
  }

  function showFrame(frame, result, view) {
    const truth = view.true_extrinsics;
    if (truth) {
      readouts.yaw.value.textContent = formatNumber(angleDeltaDeg(frame.yaw_deg, truth.yaw_deg), "°");
      readouts.pitch.value.textContent = formatNumber(Math.abs(frame.pitch_deg - truth.pitch_deg), "°");
      readouts.position.value.textContent = formatDistance(
        distanceMeters({ east_m: frame.east_m, north_m: frame.north_m, up_m: frame.up_m }, truth.position),
      );
    }
    readouts.fit.value.textContent = formatNumber(frame.score, "px");
    readouts.evals.value.textContent = String(frame.evaluations);
    drawPlot(result.observed_profile, frame.profile, result.sample_width, result.sample_height);
  }

  function showFinal(solve, view) {
    const m = solve.result.estimate.metrics;
    readouts.confidence.value.textContent = m.confidence == null ? "-" : `${Math.round(m.confidence * 100)}%`;
    readouts.yaw.value.textContent = m.yaw_error_deg == null ? "-" : formatNumber(m.yaw_error_deg, "°");
    readouts.pitch.value.textContent = m.pitch_error_deg == null ? "-" : formatNumber(m.pitch_error_deg, "°");
    readouts.position.value.textContent = m.position_error_m == null ? "-" : formatDistance(m.position_error_m);
    readouts.fit.value.textContent = formatNumber(m.contour_mae_px, "px");
    readouts.evals.value.textContent = String(solve.result.evaluations);
    drawPlot(solve.result.observed_profile, solve.result.predicted_profile, solve.result.sample_width, solve.result.sample_height);
  }

  function drawPlot(observed, predicted, sampleWidth, sampleHeight) {
    const width = Math.max(1, plot.clientWidth);
    const height = Math.max(1, plot.clientHeight);
    plot.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const line = (profile, className) => {
      const points = [];
      for (let col = 0; col < profile.length; col += 1) {
        const value = profile[col];
        if (value === null || value === undefined) {
          continue;
        }
        points.push(`${((col / (sampleWidth - 1)) * width).toFixed(1)},${((value / sampleHeight) * height).toFixed(1)}`);
      }
      return points.length > 1 ? svgElement("polyline", { class: className, points: points.join(" ") }) : null;
    };
    plot.replaceChildren(...[line(observed, "observed-line"), line(predicted, "predicted-line")].filter(Boolean));
  }

  function renderSelection() {
    const view = store.selectedView();
    const solve = store.selectedSolve();
    if (view && solve && view.solves.some((s) => s.id === solve.id)) {
      blurb.textContent = strategyBlurb(solve.strategy);
      const d = solve.result.diagnostics;
      status.textContent =
        `${strategyLabel(solve.strategy)} · ${solve.result.evaluations} evaluations` +
        (d ? ` · ${d.verdict} (alias ${d.alias_ratio} · snr ${d.snr} · well ${d.well_width_deg}°)` : "");
      if (solve.id !== shownSolveId) {
        shownSolveId = solve.id;
        animate(solve, view);
      }
    } else {
      shownSolveId = null;
      cancelAnimationFrame(animation);
      blurb.textContent = "";
      status.textContent = view ? "Run or pick a solve in the Views panel to inspect it." : "Select a view first.";
      clearReadouts();
    }
  }

  function strategyLabel(name) {
    return store.scene?.strategies.find((s) => s.name === name)?.label ?? name;
  }

  store.on("scene", renderSelection);
  store.on("selection", renderSelection);
  renderSelection();
}
