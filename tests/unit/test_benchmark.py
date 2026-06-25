"""Benchmark harness smoke test."""

from peakle.benchmark import PriorLevel, format_table, run_benchmark


def test_benchmark_runs_and_aggregates() -> None:
    rows = run_benchmark(
        map_sizes_km=(14.0,),
        views_per_map=1,
        levels=(PriorLevel("prior x1", 1.0, "powell"), PriorLevel("no prior", None, "global")),
        seed=7,
    )
    assert len(rows) == 2
    assert all(row.views == 1 for row in rows)
    assert rows[0].pos_err_med_m >= 0.0
    assert 0.0 <= rows[0].success_rate <= 1.0
    assert format_table(rows)
