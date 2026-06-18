"use strict";

// Solve inspector: pick a strategy, run it, and watch the backend's convergence
// trace animate (no solver runs in the browser). Lists the view's past solves
// so different strategies can be compared on the same view.

import { angleDeltaDeg, distanceMeters } from "../geometry.js";
import { el, formatNumber, svgElement } from "../format.js";

export function setupSolvePanel(store, root) {
  const strategy = el("select");
  const blurb = el("p", { class: "control-hint" });
  const runButton = el("button", { type: "button", class: "primary", text: "Run solver", onclick: () => run() });
  const status = el("p", { class: "control-hint", text: "Select a view, then run a solver." });

  const readouts = {
    yaw: metric("Yaw error"),
    pitch: metric("Pitch error"),
    position: metric("Position error"),
    score: metric("Objective"),
    evals: metric("Evaluations"),
  };
  const plot = svgElement("svg", { class: "solve-plot" });
  plot.setAttribute("preserveAspectRatio", "none");
  const solveList = el("ul", { class: "solve-list" });

  strategy.addEventListener("change", () => updateBlurb());

  root.replaceChildren(
    el("div", { class: "control-block" }, [
      el("span", { class: "control-eyebrow", text: "Solve inspector" }),
      el("label", { class: "field" }, [el("span", { text: "Strategy" }), strategy]),
      blurb,
      el("div", { class: "align-actions" }, [runButton]),
      status,
      el("dl", { class: "metric-grid" }, Object.values(readouts).map((m) => m.node)),
      el("div", { class: "solve-plot-wrap" }, [plot]),
      el("div", { class: "control-eyebrow", text: "Solves" }),
      solveList,
    ]),
  );

  let animation = 0;

  function metric(label) {
    const value = el("dd", { text: "-" });
    return { node: el("div", {}, [el("dt", { text: label }), value]), value };
  }

  function updateStrategies() {
    if (!store.scene) {
      return;
    }
    const previous = strategy.value;
    strategy.replaceChildren(...store.scene.strategies.map((s) => new Option(s.label, s.name)));
    strategy.value = previous || store.scene.config.default_strategy;
    updateBlurb();
  }

  function updateBlurb() {
    const meta = store.scene?.strategies.find((s) => s.name === strategy.value);
    blurb.textContent = meta ? meta.blurb : "";
  }

  async function run() {
    const view = store.selectedView();
    if (!view) {
      status.textContent = "Place or select a view first.";
      return;
    }
    runButton.disabled = true;
    status.textContent = `Solving with ${strategy.value}…`;
    try {
      const solve = await store.runSolve(view.id, strategy.value, {});
      animate(solve, view);
      status.textContent = `Converged in ${solve.result.evaluations} evaluations.`;
    } catch (error) {
      status.textContent = error.message;
    } finally {
      runButton.disabled = false;
    }
  }

  function animate(solve, view) {
    cancelAnimationFrame(animation);
    const frames = solve.result.trace;
    let index = 0;
    const step = () => {
      const frame = frames[index];
      showFrame(frame, solve.result, view);
      if (index < frames.length - 1) {
        index += 1;
        animation = requestAnimationFrame(step);
      }
    };
    step();
  }

  function showFrame(frame, result, view) {
    const truth = view.true_extrinsics;
    if (truth) {
      readouts.yaw.value.textContent = formatNumber(angleDeltaDeg(frame.yaw_deg, truth.yaw_deg), "deg");
      readouts.pitch.value.textContent = formatNumber(Math.abs(frame.pitch_deg - truth.pitch_deg), "deg");
      readouts.position.value.textContent = formatNumber(
        distanceMeters({ east_m: frame.east_m, north_m: frame.north_m, up_m: frame.up_m }, truth.position),
        "m",
      );
    }
    readouts.score.value.textContent = formatNumber(frame.score, "px");
    readouts.evals.value.textContent = String(frame.evaluations);
    drawPlot(result.observed_profile, frame.profile, result.sample_width, result.sample_height);
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

  function renderSolveList() {
    const view = store.selectedView();
    const solves = view ? view.solves : [];
    solveList.replaceChildren(
      ...solves.map((summary) => {
        const selected = summary.id === store.selectedSolveId;
        const yawErr = summary.metrics.yaw_error_deg;
        return el("li", { class: selected ? "solve-row selected" : "solve-row" }, [
          el("button", {
            type: "button",
            class: "solve-name",
            text: `${summary.strategy} · yaw ${formatNumber(yawErr, "deg")}`,
            onclick: () => store.selectSolve(view.id, summary.id),
          }),
          el("button", { type: "button", class: "icon-button", title: "Delete", text: "✕", onclick: () => store.deleteSolve(view.id, summary.id) }),
        ]);
      }),
    );
    if (!solves.length) {
      solveList.append(el("li", { class: "view-empty", text: "No solves yet." }));
    }
  }

  function renderSelection() {
    renderSolveList();
    const view = store.selectedView();
    const solve = store.selectedSolve();
    if (view && solve && view.solves.some((s) => s.id === solve.id)) {
      cancelAnimationFrame(animation);
      const result = solve.result;
      showFrame(result.trace[result.trace.length - 1], result, view);
    } else if (!solve) {
      for (const readout of Object.values(readouts)) {
        readout.value.textContent = "-";
      }
      plot.replaceChildren();
    }
  }

  store.on("scene", updateStrategies);
  store.on("views", renderSolveList);
  store.on("selection", renderSelection);
  updateStrategies();
  renderSelection();
}
