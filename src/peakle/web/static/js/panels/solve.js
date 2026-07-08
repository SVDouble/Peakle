"use strict";

// Solve view: run solvers for the selected view/GT sample, pick among previous
// solves, and inspect the selected solve's convergence trace/fit.

import { angleDeltaDeg, distanceMeters } from "../geometry.js";
import { el, formatDistance, formatNumber, svgElement } from "../format.js";

export function setupSolvePanel(store, root) {
  root.classList.add("solve-panel");
  const target = el("p", { class: "control-hint" });
  const runStatus = el("p", { class: "control-hint" });
  const strategySelect = el("select", { class: "solver-select" });
  const runButton = el("button", { type: "button", class: "primary", text: "Run solver" });
  const priorInput = el("input", { type: "checkbox", checked: true });
  const priorSwitch = el("label", { class: "toggle solve-option-toggle" }, [
    priorInput,
    el("span", { text: "Position prior" }),
    el("small", { text: "Off: recover position from the skyline alone" }),
  ]);
  const solvesHost = el("ul", { class: "solve-list" });
  const compareStatus = el("p", { class: "control-hint compare-status" });
  const blurb = el("p", { class: "control-hint" });
  const status = el("p", { class: "control-hint", text: "Select a solve to inspect it." });

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
      el("span", { class: "control-eyebrow", text: "Pose candidates" }),
      target,
      el("div", { class: "solve-actions" }, [strategySelect, runButton]),
      el("div", { class: "solve-option" }, [priorSwitch]),
      runStatus,
      solvesHost,
      compareStatus,
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
  let running = false;
  let lastTargetKey = "";
  let compareKeys = [];

  priorInput.addEventListener("change", () => {
    positionPriorChoice = priorInput.checked;
  });
  strategySelect.addEventListener("change", () => {
    strategyChoice = strategySelect.value;
  });
  runButton.addEventListener("click", runSelectedSolver);

  function metric(label) {
    const value = el("dd", { text: "-" });
    return { node: el("div", {}, [el("dt", { text: label }), value]), value };
  }

  function strategyBlurb(name) {
    return store.scene?.strategies.find((s) => s.name === name)?.blurb ?? "";
  }

  function selectedTarget() {
    const view = store.selectedView();
    if (view?.true_extrinsics) {
      return { kind: "view", label: view.label, view };
    }
    const sample = store.selectedGtSample?.() ?? (store.selectedGtName ? store.gtByName(store.selectedGtName) : null);
    if (sample) {
      return { kind: "gt", label: sample.name, sample };
    }
    return null;
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
    const targetInfo = selectedTarget();
    const key = targetInfo ? `${targetInfo.kind}:${targetInfo.kind === "view" ? targetInfo.view.id : targetInfo.sample.name}` : "";
    const targetChanged = key !== lastTargetKey;
    if (targetChanged && !running) {
      runStatus.textContent = "";
    }
    lastTargetKey = key;
    syncStrategySelect(targetInfo);
    priorInput.checked = positionPriorChoice;
    strategySelect.disabled = running || !targetInfo || !strategySelect.options.length;
    runButton.disabled = running || !targetInfo || !strategySelect.options.length;
    if (!targetInfo) {
      target.textContent = "Select a view or GT sample first.";
    } else if (targetInfo.kind === "gt") {
      target.textContent = `${targetInfo.label} · will open as an editable view before solving.`;
    } else {
      target.textContent = targetInfo.label;
    }
    const candidates = candidateSummaries(targetInfo);
    syncComparisonKeys(candidates, targetChanged);
    renderSolves(targetInfo, candidates);
    renderComparison(candidates);
    return targetInfo;
  }

  function candidateSummaries(targetInfo) {
    if (targetInfo?.kind === "gt") {
      return [gtCandidate(targetInfo.sample)];
    }
    const view = targetInfo?.view;
    if (!view) {
      return [];
    }
    const candidates = view.true_extrinsics ? [viewCandidate(view)] : [];
    candidates.push(...view.solves.map((summary) => solveCandidate(summary)));
    return candidates;
  }

  function renderSolves(targetInfo, candidates) {
    solvesHost.replaceChildren();
    if (targetInfo?.kind === "gt") {
      solvesHost.append(candidateRow(candidates[0], { selected: true, compareable: false }));
      solvesHost.append(el("li", { class: "view-empty", text: "Run solver to create an editable view." }));
      return;
    }
    const view = targetInfo?.view ?? null;
    if (!view) {
      solvesHost.append(el("li", { class: "view-empty", text: "No editable view selected." }));
      return;
    }
    solvesHost.replaceChildren(
      ...candidates.map((candidate) => {
        const compare = {
          compareable: candidates.length >= 2,
          compareChecked: compareKeys.includes(candidate.key),
          onCompare: (checked) => toggleCompareKey(candidate.key, checked),
        };
        if (candidate.kind === "truth") {
          return candidateRow(candidate, { ...compare, selected: !store.selectedSolveId, onclick: () => store.selectViewTruth(view.id) });
        }
        const remove = el("button", { type: "button", class: "icon-button", title: "Delete solve", text: "✕" });
        remove.addEventListener("click", (event) => {
          event.stopPropagation();
          store.deleteSolve(view.id, candidate.id);
        });
        return candidateRow(candidate, { ...compare, selected: candidate.id === store.selectedSolveId, onclick: () => store.selectSolve(view.id, candidate.id), action: remove });
      }),
    );
    if (!view.solves.length) {
      solvesHost.append(el("li", { class: "view-empty", text: "No solves yet." }));
    }
  }

  function viewCandidate(view) {
    const sample = view.gt_name ? store.gtByName(view.gt_name) : null;
    const label = view.source === "gt" ? "GT refined pose" : view.source === "photo" ? "Photo metadata prior" : "Dataset pose";
    return {
      key: "truth",
      kind: "truth",
      label,
      stat: sample
        ? `sky ${formatNumber(sample.sky_cons_px, "px")} · ct ${formatNumber(sample.contour_cons_px, "px")}`
        : `yaw ${formatNumber(view.true_extrinsics.yaw_deg, "°")} · pitch ${formatNumber(view.true_extrinsics.pitch_deg, "°")}`,
      metrics: {
        fit: sample?.contour_cons_px ?? sample?.sky_cons_px ?? null,
        yaw: null,
        pitch: null,
        position: null,
      },
    };
  }

  function gtCandidate(sample) {
    return {
      key: "truth",
      kind: "truth",
      label: "GT refined pose",
      stat: `sky ${formatNumber(sample.sky_cons_px, "px")} · ct ${formatNumber(sample.contour_cons_px, "px")}`,
      metrics: { fit: sample.contour_cons_px ?? sample.sky_cons_px ?? null, yaw: null, pitch: null, position: null },
    };
  }

  function solveCandidate(summary) {
    const baseline = store.selectedView()?.gt_name ? store.gtByName(store.selectedView().gt_name) : null;
    const fit = summary.metrics.contour_mae_px;
    const baselineFit = baseline?.contour_cons_px ?? baseline?.sky_cons_px ?? null;
    const delta = baselineFit == null ? "" : ` · Δ ${formatNumber(fit - baselineFit, "px")}`;
    return {
      key: `solve:${summary.id}`,
      kind: "solve",
      id: summary.id,
      label: strategyLabel(summary.strategy),
      stat: `yaw ${formatNumber(summary.metrics.yaw_error_deg, "°")} · fit ${formatNumber(fit, "px")}${delta}`,
      metrics: {
        fit,
        yaw: summary.metrics.yaw_error_deg,
        pitch: summary.metrics.pitch_error_deg,
        position: summary.metrics.position_error_m,
      },
    };
  }

  function candidateRow(candidate, { selected = false, onclick = null, action = null, compareable = false, compareChecked = false, onCompare = null } = {}) {
    let compareNode = null;
    if (compareable && onCompare) {
      const input = el("input", { type: "checkbox", checked: compareChecked, "aria-label": `Compare ${candidate.label}` });
      input.addEventListener("click", (event) => event.stopPropagation());
      input.addEventListener("change", () => onCompare(input.checked));
      compareNode = el("label", { class: "candidate-compare", title: "Compare candidate" }, [input]);
    }
    return el("li", { class: `${selected ? "solve-row selected" : "solve-row"} ${candidate.kind === "truth" ? "truth-row" : ""}` }, [
      compareNode,
      el("button", { type: "button", class: "solve-name", onclick }, [
        el("span", { class: "solve-strategy", text: candidate.label }),
        el("span", { class: "solve-stat", text: candidate.stat }),
      ]),
      action,
    ]);
  }

  function syncComparisonKeys(candidates, targetChanged) {
    const valid = new Set(candidates.map((candidate) => candidate.key));
    compareKeys = compareKeys.filter((key) => valid.has(key)).slice(-2);
    if (targetChanged) {
      compareKeys = defaultCompareKeys(candidates);
    }
  }

  function defaultCompareKeys(candidates) {
    if (candidates.length < 2) {
      return [];
    }
    const primaryKey = store.selectedSolveId ? `solve:${store.selectedSolveId}` : "truth";
    const keys = [];
    if (candidates.some((candidate) => candidate.key === primaryKey)) {
      keys.push(primaryKey);
    }
    for (const candidate of candidates) {
      if (!keys.includes(candidate.key)) {
        keys.push(candidate.key);
      }
      if (keys.length === 2) {
        break;
      }
    }
    return keys;
  }

  function toggleCompareKey(key, checked) {
    if (checked) {
      compareKeys = compareKeys.filter((candidateKey) => candidateKey !== key);
      compareKeys.push(key);
      compareKeys = compareKeys.slice(-2);
    } else {
      compareKeys = compareKeys.filter((candidateKey) => candidateKey !== key);
    }
    renderSolverControls();
  }

  function renderComparison(candidates) {
    const selected = compareKeys.map((key) => candidates.find((candidate) => candidate.key === key)).filter(Boolean);
    compareStatus.hidden = candidates.length < 2;
    if (candidates.length < 2) {
      compareStatus.textContent = "";
      return;
    }
    compareStatus.textContent = selected.length === 2 ? comparisonText(selected[0], selected[1]) : "Tick two candidates to compare metrics.";
  }

  function comparisonText(primary, other) {
    const parts = [
      metricDelta("fit", primary.metrics.fit, other.metrics.fit, "px"),
      metricDelta("yaw", primary.metrics.yaw, other.metrics.yaw, "°"),
      metricDelta("pitch", primary.metrics.pitch, other.metrics.pitch, "°"),
      metricDelta("pos", primary.metrics.position, other.metrics.position, "m"),
    ].filter(Boolean);
    return parts.length ? `${primary.label} vs ${other.label}: ${parts.join(" · ")}` : `${primary.label} vs ${other.label}: no comparable metrics yet.`;
  }

  function metricDelta(label, a, b, unit) {
    if (a === null || a === undefined || b === null || b === undefined || Number.isNaN(a) || Number.isNaN(b)) {
      return "";
    }
    return `Δ${label} ${formatNumber(a - b, unit)}`;
  }

  async function runSelectedSolver() {
    const targetInfo = selectedTarget();
    if (!targetInfo || !strategySelect.value) {
      return;
    }
    running = true;
    renderSolverControls();
    try {
      let view = targetInfo.view;
      if (targetInfo.kind === "gt") {
        runStatus.textContent = `Opening ${targetInfo.label} as a view...`;
        view = await store.openGtView(targetInfo.sample.name);
      }
      runStatus.textContent = `Solving with ${strategyLabel(strategySelect.value)}...`;
      const solve = await store.runSolve(view.id, strategySelect.value, { position_prior: positionPriorChoice });
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
    const view = store.selectedView();
    const solve = store.selectedSolve();
    const targetInfo = renderSolverControls();
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
      status.textContent = targetInfo?.kind === "gt" ? "Run a solver above to open this GT sample as a view." : view ? "Run or pick a solve above to inspect it." : "Select a view first.";
      clearReadouts();
    }
  }

  function strategyLabel(name) {
    return store.scene?.strategies.find((s) => s.name === name)?.label ?? name;
  }

  store.on("scene", renderSelection);
  store.on("views", renderSelection);
  store.on("gt", renderSelection);
  store.on("selection", renderSelection);
  renderSelection();
}
