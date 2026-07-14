# Peakle development documents

Peakle is an active mountain-view localization and annotation research workbench. The original
synthetic MVP has grown into a real-terrain web application, a localization library, and an
experiment harness. Some documents in this directory describe the design at the time of a specific
study; they are retained for reproducibility, not as parallel roadmaps.

## Start here

- [Research and development program](../research-and-development.md) — the **only normative source**
  for the mission, accepted evidence, benchmark contract, technical direction, and current roadmap.
- [Machine-readable evidence ledger](../research-evidence.json) — artifact/run hashes, parent chains,
  archive availability, and compact outcomes backing the program's human evidence table.
- [Validation strategy](../validation-strategy.md) — durable methodological lessons and incident
  history. Current eligibility and priorities still come from the program above.
- [Root README](../../README.md) — installation, entry points, data locations, and user-facing
  behaviour.

When a result changes project status, update the research program and its evidence ledger in the
same commit. Do not put a new “next strategy” in a dated study report or duplicate detailed result
claims in the root README.

## Current architecture references

- [Algorithms](algorithms.md) — geometry, rendering, extraction, optimization, and annotation
  concepts.
- [Library decisions](library-decisions.md) — package and browser-visualization choices.
- [Project structure](project-structure.md) — the original intended module boundaries; compare it
  with the consolidation target in the normative research program before treating it as current.
- [Synthetic demo pipeline](synthetic-demo-pipeline.md) — original synthetic pipeline and artifact
  contracts.

Camera semantics belong to `peakle.domain.camera` and `peakle.domain.projection`. The browser keeps
only the presentation-side mirror needed by Three.js. Scoring, terrain ray-casting, pose comparison,
evaluation, and persisted solver outputs belong in Python services. Numeric acceleration may later
sit behind those contracts; it does not require a web or domain-model rewrite.

## Historical study records

- [Camera-fitting research](camera-fitting-research.md), [ridge-extraction research](ridge-extraction-research.md),
  and [high-precision reference data](high-precision-reference-data.md) — historical methods/data
  surveys whose present-tense recommendations are non-normative.
- [Pose-localization strategy after the refined-pose reset](pose-localization-strategy.md) — dated
  experiment design and render-match/PnP history.
- [High-compute pose-atlas study](pose-atlas-study.md) — atlas, proposal-coverage, and geometry-rerank
  evidence.
- [Synthetic localization benchmark](synthetic-localization-benchmark.md) — shared-renderer stage
  ceilings and truth-boundary design.
- [Original roadmap and risks](roadmap-and-risks.md) — retired MVP milestone plan.

Historical reports should remain factual and reproducible. Correct wrong numbers in place, but put
new decisions, status labels, and sequencing only in the normative research program.

## Development checks

```bash
uv run pytest
uv run ruff check .
uv run ty check src
```

Default tests must remain offline. Model-, GPU-, real-data-, and long-running studies belong in
explicit optional lanes and must record their input/model hashes and network policy.
