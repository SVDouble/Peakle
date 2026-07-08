"use strict";

// Solve view: run solvers for the selected view/pose target and inspect the selected
// solver pose's convergence trace/fit. Pose selection itself lives in Inspect.

import { angleDeltaDeg, distanceMeters } from "../geometry.js";
import { el, formatDistance, formatNumber, svgElement } from "../format.js";
import { poseTargetKey, selectedPoseTarget, strategyLabel } from "../pose-candidates.js";

export function setupSolvePanel(store, root) {
  root.classList.add("solve-panel");
  const target = el("p", { class: "control-hint" });
  const runStatus = el("p", { class: "control-hint" });
  const strategySelect = el("select", { class: "solver-select" });
  const runButton = el("button", { type: "button", class: "primary", text: "Run solver" });
  const positionPriorInput = el("input", { type: "checkbox", checked: true });
  const orientationPriorInput = el("input", { type: "checkbox", checked: true });
  const positionPriorSwitch = el("label", { class: "toggle solve-option-toggle" }, [
    positionPriorInput,
    el("span", { text: "Position prior" }),
    el("small", { text: "Off: recover position from the skyline alone" }),
  ]);
  const orientationPriorSwitch = el("label", { class: "toggle solve-option-toggle" }, [
    orientationPriorInput,
    el("span", { text: "Orientation prior" }),
    el("small", { text: "Off: search full yaw/pitch bounds" }),
  ]);
  const blurb = el("p", { class: "control-hint" });
  const status = el("p", { class: "control-hint", text: "Select a solver pose in Inspect." });

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
      el("span", { class: "control-eyebrow", text: "Run solver" }),
      target,
      el("div", { class: "solve-actions" }, [strategySelect, runButton]),
      el("div", { class: "solve-option" }, [positionPriorSwitch, orientationPriorSwitch]),
      runStatus,
    ]),
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
  let strategyChoice = null;
  let positionPriorChoice = true;
  let orientationPriorChoice = true;
  let running = false;
  let lastTargetKey = "";

  positionPriorInput.addEventListener("change", () => {
    positionPriorChoice = positionPriorInput.checked;
  });
  orientationPriorInput.addEventListener("change", () => {
    orientationPriorChoice = orientationPriorInput.checked;
  });
  strategySelect.addEventListener("change", () => {
    strategyChoice = strategySelect.value;
    renderSolverControls();
  });
  runButton.addEventListener("click", runSelectedSolver);

  function metric(label) {
    const value = el("dd", { text: "-" });
    return { node: el("div", {}, [el("dt", { text: label }), value]), value };
  }

  function strategyBlurb(name) {
    return store.scene?.strategies.find((s) => s.name === name)?.blurb ?? "";
  }

  function syncStrategySelect(targetInfo) {
    const strategies = store.scene?.strategies ?? [];
    if (targetInfo?.kind === "view" && targetInfo.view.source === "photo" && !targetInfo.view.solves.length && strategies.some((s) => s.name === "horizon")) {
      strategyChoice = "horizon";
    }
    if (!strategyChoice || !strategies.some((s) => s.name === strategyChoice)) {
      strategyChoice = store.scene?.config.default_strategy ?? strategies[0]?.name ?? "";
    }
    const previous = strategySelect.value;
    strategySelect.replaceChildren(...strategies.map((s) => el("option", { value: s.name, text: s.label })));
    strategySelect.value = strategyChoice || previous;
    strategyChoice = strategySelect.value;
  }

  function renderSolverControls() {
    const targetInfo = selectedPoseTarget(store);
    const key = poseTargetKey(targetInfo);
    const targetChanged = key !== lastTargetKey;
    if (targetChanged && !running) {
      runStatus.textContent = "";
    }
    lastTargetKey = key;
    syncStrategySelect(targetInfo);
    syncPriorInputs(targetInfo);
    strategySelect.disabled = running || !targetInfo || !strategySelect.options.length;
    runButton.disabled = running || !targetInfo || !strategySelect.options.length;
    if (!targetInfo) {
      target.textContent = "Select a view first.";
    } else if (targetInfo.kind === "gt") {
      target.textContent = `${targetInfo.label} · solvers use the DEM/refined-pose backing pose.`;
    } else {
      target.textContent = targetInfo.label;
    }
    return targetInfo;
  }

  function syncPriorInputs(targetInfo) {
    const strategy = strategyChoice || strategySelect.value;
    const priorFree = strategy === "global";
    const horizon = strategy === "horizon";
    positionPriorInput.disabled = running || !targetInfo || priorFree || horizon;
    orientationPriorInput.disabled = running || !targetInfo || priorFree || horizon;
    positionPriorInput.checked = priorFree ? false : horizon ? true : positionPriorChoice;
    orientationPriorInput.checked = priorFree || horizon ? false : orientationPriorChoice;
  }

  async function runSelectedSolver() {
    const targetInfo = selectedPoseTarget(store);
    if (!targetInfo || !strategySelect.value) {
      return;
    }
    running = true;
    renderSolverControls();
    try {
      let view = targetInfo.view;
      if (targetInfo.kind === "gt") {
        runStatus.textContent = `Preparing ${targetInfo.label} DEM/refined pose...`;
        view = await store.openGtView(targetInfo.sample.name);
      }
      runStatus.textContent = `Solving with ${strategyLabel(store, strategySelect.value)}...`;
      const solve = await store.runSolve(view.id, strategySelect.value, {
        position_prior: positionPriorInput.checked,
        orientation_prior: orientationPriorInput.checked,
      });
      runStatus.textContent = `Converged in ${solve.result.evaluations} evaluations.`;
    } catch (error) {
      runStatus.textContent = error.message;
    } finally {
      running = false;
      renderSolverControls();
    }
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
    const targetInfo = renderSolverControls();
    const view = store.selectedView() ?? targetInfo?.view ?? null;
    const solve = store.selectedSolve();
    if (view && solve && view.solves.some((s) => s.id === solve.id)) {
      blurb.textContent = strategyBlurb(solve.strategy);
      const d = solve.result.diagnostics;
      status.textContent =
        `${strategyLabel(store, solve.strategy)} · ${solve.result.evaluations} evaluations` +
        (d ? ` · ${d.verdict} (alias ${d.alias_ratio} · snr ${d.snr} · well ${d.well_width_deg}°)` : "");
      if (solve.id !== shownSolveId) {
        shownSolveId = solve.id;
        animate(solve, view);
      }
    } else {
      shownSolveId = null;
      cancelAnimationFrame(animation);
      blurb.textContent = "";
      status.textContent = view ? "Run a solver here or pick a solver pose in Inspect." : "Select a view first.";
      clearReadouts();
    }
  }

  store.on("scene", renderSelection);
  store.on("views", renderSelection);
  store.on("gt", renderSelection);
  store.on("selection", renderSelection);
  renderSelection();
}
