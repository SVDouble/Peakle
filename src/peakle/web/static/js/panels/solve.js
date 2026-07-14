"use strict";

// Solve view: run solvers for the selected view/pose target and inspect the selected
// solver pose's convergence trace/fit. Pose selection itself lives in Inspect.

import { angleDeltaDeg, distanceMeters } from "../geometry.js";
import { el, formatDistance, formatNumber, svgElement } from "../format.js";
import { poseCandidates, poseTargetKey, selectedPoseTarget, strategyLabel } from "../pose-candidates.js";
import { api } from "../api.js";

export function setupSolvePanel(store, root) {
  root.classList.add("solve-panel");
  const target = el("p", { class: "control-hint" });
  const runStatus = el("p", { class: "control-hint" });
  const strategySelect = el("select", { class: "solver-select" });
  const priorSourceSelect = el("select", { class: "solver-select prior-source-select", title: "Prior source" });
  const evidenceSourceSelect = el("select", {
    class: "solver-select evidence-source-select",
    title: "Skyline evidence used by this solve",
  });
  const runButton = el("button", { type: "button", class: "primary", text: "Run solver" });
  const runAllButton = el("button", {
    type: "button",
    class: "secondary",
    text: "Run loaded views",
    title: "Queue this solver for every loaded view",
  });
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
      el("div", { class: "solve-actions" }, [evidenceSourceSelect, strategySelect, priorSourceSelect, runButton, runAllButton]),
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
  let priorSourceChoice = "metadata";
  let evidenceSourceChoice = null;
  let positionPriorChoice = true;
  let orientationPriorChoice = true;
  let running = false;
  let batchRunning = false;
  let batchJobTimer = 0;
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
  priorSourceSelect.addEventListener("change", () => {
    priorSourceChoice = priorSourceSelect.value;
    renderSolverControls();
  });
  evidenceSourceSelect.addEventListener("change", () => {
    evidenceSourceChoice = evidenceSourceSelect.value;
    renderSolverControls();
  });
  runButton.addEventListener("click", runSelectedSolver);
  runAllButton.addEventListener("click", runBatchSolver);

  function metric(label) {
    const value = el("dd", { text: "-" });
    return { node: el("div", {}, [el("dt", { text: label }), value]), value };
  }

  function strategyBlurb(name) {
    return store.scene?.strategies.find((s) => s.name === name)?.blurb ?? "";
  }

  function syncStrategySelect(targetInfo) {
    const projection = targetInfo?.kind === "gt" || targetInfo?.view?.source === "gt" ? "cyltan" : "pinhole";
    const strategies = (store.scene?.strategies ?? []).filter(
      (strategy) => !strategy.projections || strategy.projections.includes(projection),
    );
    if (targetInfo?.kind === "view" && targetInfo.view.source === "photo" && !targetInfo.view.solves.length && strategies.some((s) => s.name === "horizon")) {
      strategyChoice = "horizon";
    }
    if (!strategyChoice || !strategies.some((s) => s.name === strategyChoice)) {
      strategyChoice = store.scene?.config.default_strategy ?? strategies[0]?.name ?? "";
      if (!strategies.some((s) => s.name === strategyChoice)) {
        strategyChoice = strategies[0]?.name ?? "";
      }
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
    syncPriorSourceSelect(targetInfo, targetChanged);
    syncEvidenceSourceSelect(targetInfo, targetChanged);
    syncPriorInputs(targetInfo);
    const selectedEvidence = evidenceSourceOptions(targetInfo).find((option) => option.id === evidenceSourceSelect.value);
    const evidenceUsable = Boolean(selectedEvidence?.available ?? false);
    const loadedViews = eligibleSolveViews(evidenceSourceSelect.value);
    const batchPriorReusable = !priorSourceSelect.value.startsWith("pose:solve:");
    strategySelect.disabled = running || batchRunning || !strategySelect.options.length;
    priorSourceSelect.disabled = running || batchRunning || !targetInfo || !priorSourceSelect.options.length;
    evidenceSourceSelect.disabled = running || batchRunning || !targetInfo || !evidenceSourceSelect.options.length;
    runButton.disabled = running || batchRunning || !targetInfo || !strategySelect.options.length || !evidenceUsable;
    runAllButton.disabled = running || batchRunning || !loadedViews.length || !strategySelect.options.length || !batchPriorReusable;
    runAllButton.title = batchPriorReusable ? "Queue this solver for every loaded view" : "Batch runs cannot reuse a selected solver pose as the prior";
    runAllButton.textContent = loadedViews.length ? `Run ${loadedViews.length} loaded` : "Run loaded views";
    if (!targetInfo) {
      target.textContent = "Select a view first.";
    } else if (targetInfo.kind === "gt" || targetInfo.view?.source === "gt") {
      target.textContent = `${targetInfo.label} · original GT pose; vertical crop offset makes physical pitch error non-comparable.`;
    } else {
      target.textContent = targetInfo.label;
    }
    return targetInfo;
  }

  function syncEvidenceSourceSelect(targetInfo, targetChanged) {
    const options = evidenceSourceOptions(targetInfo);
    const defaultOption = options.find((option) => option.default) ?? options[0];
    if (targetChanged || !options.some((option) => option.id === evidenceSourceChoice)) {
      evidenceSourceChoice = defaultOption?.id ?? null;
    }
    evidenceSourceSelect.replaceChildren(
      ...options.map((option) =>
        el("option", {
          value: option.id,
          text: `Evidence: ${option.label}${option.available ? "" : " · unavailable"}`,
        }),
      ),
    );
    evidenceSourceSelect.value = evidenceSourceChoice ?? "";
    const selected = options.find((option) => option.id === evidenceSourceSelect.value);
    evidenceSourceSelect.title = selected?.available
      ? `${selected.label}${selected.diagnostic ? " — diagnostic/oracle track" : ""}`
      : selected?.provenance?.reason ?? "This evidence track is unavailable; choose another track explicitly.";
  }

  function evidenceSourceOptions(targetInfo) {
    const view = targetInfo?.view ?? (targetInfo?.kind === "gt" ? store.gtViewForSample?.(targetInfo.sample.name) : null);
    if (view?.evidence_sources?.length) {
      return view.evidence_sources;
    }
    if (targetInfo?.kind === "gt") {
      return [
        { id: "photo_auto", label: "Photo skyline (automatic)", available: true, diagnostic: false, default: true },
        { id: "pfm_oracle", label: "PFM/source-depth oracle (diagnostic)", available: true, diagnostic: true, default: false },
      ];
    }
    if (view) {
      const source = view.default_evidence_source ?? "rendered_skyline";
      return [{ id: source, label: source.replaceAll("_", " "), available: true, diagnostic: false, default: true }];
    }
    return [];
  }

  function syncPriorSourceSelect(targetInfo, targetChanged) {
    const options = priorSourceOptions(targetInfo);
    if (targetChanged || !options.some((option) => option.value === priorSourceChoice)) {
      priorSourceChoice = options[0]?.value ?? "metadata";
    }
    priorSourceSelect.replaceChildren(...options.map((option) => el("option", { value: option.value, text: option.label })));
    priorSourceSelect.value = priorSourceChoice;
  }

  function priorSourceOptions(targetInfo) {
    if (!targetInfo) {
      return [{ value: "metadata", label: "Prior: view metadata" }];
    }
    const gtTarget = targetInfo.kind === "gt" || targetInfo.view?.source === "gt";
    const options = [{ value: "metadata", label: gtTarget ? "Prior: GT metadata" : "Prior: view metadata" }];
    if (!gtTarget) {
      options.push({ value: "pose:truth", label: "Prior: baseline pose" });
    }
    for (const candidate of poseCandidates(store, targetInfo)) {
      if (candidate.kind === "solve") {
        options.push({ value: `pose:${candidate.key}`, label: `Prior: ${candidate.label}` });
      }
    }
    const seen = new Set();
    return options.filter((option) => {
      if (seen.has(option.value)) {
        return false;
      }
      seen.add(option.value);
      return true;
    });
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

  function eligibleSolveViews(evidenceSource = null) {
    return store.views
      .filter(
        (view) =>
          view.prior &&
          view.true_extrinsics &&
          (!evidenceSource || view.evidence_sources?.some((source) => source.id === evidenceSource && source.available)),
      )
      .map((view) => view.id);
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
        runStatus.textContent = `Preparing ${targetInfo.label} from its original metadata...`;
        view = await store.openGtView(targetInfo.sample.name);
      }
      const evidence = view.evidence_sources?.find((source) => source.id === evidenceSourceSelect.value);
      if (!evidence?.available) {
        throw new Error(
          evidence?.provenance?.reason ??
            `Evidence ${evidenceSourceSelect.value} is unavailable. Select the PFM oracle explicitly only for a diagnostic solve.`,
        );
      }
      runStatus.textContent = `Solving with ${strategyLabel(store, strategySelect.value)}...`;
      const solve = await store.runSolve(view.id, strategySelect.value, {
        position_prior: positionPriorInput.checked,
        orientation_prior: orientationPriorInput.checked,
        prior_source: priorSourceSelect.value,
        evidence_source: evidenceSourceSelect.value,
      });
      runStatus.textContent = `Converged in ${solve.result.evaluations} evaluations.`;
    } catch (error) {
      runStatus.textContent = error.message;
    } finally {
      running = false;
      renderSolverControls();
    }
  }

  async function runBatchSolver() {
    const viewIds = eligibleSolveViews(evidenceSourceSelect.value);
    if (!viewIds.length || !strategySelect.value) {
      return;
    }
    batchRunning = true;
    renderSolverControls();
    try {
      const job = await store.runSolveJob(viewIds, strategySelect.value, {
        position_prior: positionPriorInput.checked,
        orientation_prior: orientationPriorInput.checked,
        prior_source: priorSourceSelect.value,
        evidence_source: evidenceSourceSelect.value,
      });
      runStatus.textContent = `Queued ${viewIds.length} solve task${viewIds.length === 1 ? "" : "s"}.`;
      clearTimeout(batchJobTimer);
      pollSolveJob(job.id);
    } catch (error) {
      batchRunning = false;
      runStatus.textContent = error.message;
      renderSolverControls();
    }
  }

  async function pollSolveJob(jobId) {
    try {
      const job = await api.getJob(jobId);
      const finished = (job.done ?? 0) + (job.failed ?? 0);
      const runningLabels = (job.running_tasks ?? []).map((task) => task.label).join(", ");
      if (job.status === "queued" || job.status === "running") {
        runStatus.textContent = `Solving ${finished}/${job.total} — ${runningLabels || job.status}`;
        batchJobTimer = setTimeout(() => pollSolveJob(jobId), 1500);
        return;
      }
      batchRunning = false;
      await store.refreshSceneState();
      store.emitSceneAndViews();
      store.emit("selection");
      runStatus.textContent =
        job.failed > 0 ? `Batch solve finished with ${job.failed} failed task${job.failed === 1 ? "" : "s"}.` : "Batch solve finished.";
      renderSolverControls();
    } catch (error) {
      batchRunning = false;
      runStatus.textContent = error.message;
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
      readouts.pitch.value.textContent =
        view.pitch_comparable === false ? "n/a · cropped image" : formatNumber(Math.abs(frame.pitch_deg - truth.pitch_deg), "°");
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
    readouts.pitch.value.textContent =
      view.pitch_comparable === false ? "n/a · cropped image" : m.pitch_error_deg == null ? "-" : formatNumber(m.pitch_error_deg, "°");
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
        `${strategyLabel(store, solve.strategy)} · ${solve.params?.evidence_source ?? "unknown evidence"} · ${solve.result.evaluations} evaluations` +
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
