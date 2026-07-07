"""First-class CLI entry points and analysis/report runners for the peakle corpus.

These were top-level scripts/ files; they live in the package so they are importable and runnable
as ``python -m peakle.scripts.<name>``. Reusable domain logic lives in peakle.localize
(gtbuild, bench, gtquality); these modules are the thin CLIs and the presentation/experiment
runners over it.
"""
