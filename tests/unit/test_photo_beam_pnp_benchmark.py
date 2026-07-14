from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.localize import photo_beam_pnp_benchmark as module
from peakle.localize import photo_beam_render_pnp as bridge_module
from peakle.localize.photo_beam_pnp_benchmark import (
    evaluate_photo_beam_pnp_benchmark,
    freeze_photo_beam_pnp_benchmark,
    validate_frozen_photo_beam_pnp_benchmark,
)
from peakle.localize.photo_beam_render_pnp import PhotoBeamRenderSeed, PhotoBeamRenderSeedBridge
from peakle.localize.render_match_pnp import (
    RENDER_SEED_BATCH_SCHEMA,
    RenderMatchBatchResult,
    RenderMatchPoseResult,
    RenderMatchSeedResult,
)


def _pose(east_m: float, yaw_deg: float) -> CameraExtrinsics:
    return CameraExtrinsics(
        position=LocalPoint(east_m=east_m, north_m=0.0, up_m=1000.0),
        yaw_deg=yaw_deg,
        pitch_deg=0.0,
        roll_deg=0.0,
    )


def _bridge() -> PhotoBeamRenderSeedBridge:
    seeds = tuple(
        PhotoBeamRenderSeed(
            candidate_id=candidate_id,
            beam_rank=rank,
            verifier_rank=rank,
            original_estimator_rank=rank,
            east_m=east_m,
            north_m=0.0,
            up_m=1000.0,
            yaw_deg=yaw_deg,
            query_initial_pitch_deg=0.0,
            discarded_atlas_roll_nuisance_deg=0.0,
        )
        for rank, (candidate_id, east_m, yaw_deg) in enumerate(
            (("seed-a", 200.0, 20.0), ("seed-b", 50.0, 3.0), ("seed-c", 300.0, 30.0)),
            start=1,
        )
    )
    observation_json = (
        '{"candidate":"photo","evidence_generated_at_reference_pose":false,'
        '"selection_uses_reference_truth":false,"source":"photo_auto",'
        '"source_atlas_sha256":"' + "a" * 64 + '","track":"photo_auto"}'
    )
    kwargs: dict[str, Any] = {
        "source_verifier_archive_sha256": "b" * 64,
        "source_atlas_sha256": "a" * 64,
        "photo_rgb_sha256": "c" * 64,
        "observed_skyline_sha256": "d" * 64,
        "edge_model_sha256": "e" * 64,
        "depth_model_sha256": "f" * 64,
        "observation_provenance_json": observation_json,
        "seeds": seeds,
    }
    basis = bridge_module._bridge_basis(**kwargs)
    return PhotoBeamRenderSeedBridge(**kwargs, bridge_sha256=bridge_module._canonical_sha256(basis))


def _diagnostics(
    candidate_id: str,
    rng_seed: int,
    *,
    pre_candidate: CameraExtrinsics | None,
    validation_passed: bool,
) -> dict[str, Any]:
    validation = {
        "enabled": True,
        "passed": validation_passed,
        "status": "passed" if validation_passed else "failed",
        "failures": [] if validation_passed else ["heldout_joint_consensus_below_acceptance_gate"],
        "uses_reference_truth": False,
    }
    return {
        "batch": {"candidate_id": candidate_id, "candidate_rng_seed": rng_seed},
        "frames": [
            {
                "match_stage_counts": {
                    "worker_selected": 40,
                    "after_query_padding_rejection": 38,
                    "after_render_lifting": 30,
                    "after_spatial_cap": 24,
                },
                "pnp": None,
            }
        ],
        "final_pnp": ({"candidate_pose": pre_candidate.model_dump(mode="json")} if pre_candidate is not None else None),
        "candidate_validation": validation,
        "rejected_candidate_pose": (
            pre_candidate.model_dump(mode="json") if pre_candidate is not None and not validation_passed else None
        ),
    }


def _batch(bridge: PhotoBeamRenderSeedBridge) -> RenderMatchBatchResult:
    rng_seeds = {"seed-a": 101, "seed-b": 102, "seed-c": 103}
    pre_candidates = {"seed-a": _pose(20.0, 2.0), "seed-b": None, "seed-c": _pose(80.0, 4.0)}
    accepted = {"seed-a": None, "seed-b": None, "seed-c": _pose(80.0, 4.0)}
    results = []
    for hypothesis in bridge.render_seeds:
        candidate_id = hypothesis.candidate_id
        output = accepted[candidate_id]
        result = RenderMatchPoseResult(
            status="solved" if output is not None else "abstained",
            extrinsics=output,
            diagnostics=_diagnostics(
                candidate_id,
                rng_seeds[candidate_id],
                pre_candidate=pre_candidates[candidate_id],
                validation_passed=output is not None,
            ),
            candidates=(output,) if output is not None else (),
        )
        results.append(
            RenderMatchSeedResult(
                candidate_id=candidate_id,
                render_seed=hypothesis.render_seed,
                rng_seed=rng_seeds[candidate_id],
                result=result,
            )
        )
    ids = list(bridge.ordered_candidate_ids)
    return RenderMatchBatchResult(
        results=tuple(results),
        provenance={
            "schema": RENDER_SEED_BATCH_SCHEMA,
            "uses_reference_truth": False,
            "ordered_candidate_ids": ids,
            "candidate_count": len(ids),
            "rng": {"candidate_root_seeds": rng_seeds},
        },
    )


def _frozen_record() -> dict[str, Any]:
    bridge = _bridge()
    return freeze_photo_beam_pnp_benchmark(
        bridge,
        _batch(bridge),
        lane_label="prior_regularized_fixed_radius",
        lane_config={"use_position_prior": True, "radius_m": 100.0},
    ).to_record()


def _rehash(record: dict[str, Any]) -> None:
    basis = dict(record)
    basis.pop("archive_sha256", None)
    record["archive_sha256"] = module._canonical_sha256(basis)


def test_freezes_exact_ordered_linkage_and_detaches_json_records() -> None:
    bridge = _bridge()
    batch = _batch(bridge)

    archive = freeze_photo_beam_pnp_benchmark(
        bridge,
        batch,
        lane_label="prior_regularized_fixed_radius",
        lane_config={"radius_m": 100.0, "passes": (0, 1)},
    )
    record = archive.to_record()

    assert record["uses_reference_truth"] is False
    assert record["ordered_candidate_ids"] == ["seed-a", "seed-b", "seed-c"]
    assert [result["candidate_id"] for result in record["results"]] == record["ordered_candidate_ids"]
    assert record["results"][2]["status"] == "solved"
    assert record["results"][2]["accepted_extrinsics"] == _pose(80.0, 4.0).model_dump(mode="json")
    assert record["lane"]["config"]["passes"] == [0, 1]
    batch.provenance["ordered_candidate_ids"].reverse()
    record["results"][0]["diagnostics"]["batch"]["candidate_id"] = "mutated"
    assert archive.to_record()["ordered_candidate_ids"] == ["seed-a", "seed-b", "seed-c"]
    assert archive.to_record()["results"][0]["diagnostics"]["batch"]["candidate_id"] == "seed-a"


@pytest.mark.parametrize("mutation", ["order", "count", "id", "seed"])
def test_freeze_rejects_bridge_batch_linkage_mismatches(mutation: str) -> None:
    bridge = _bridge()
    batch = _batch(bridge)
    results = list(batch.results)
    provenance = deepcopy(batch.provenance)
    if mutation == "order":
        results.reverse()
    elif mutation == "count":
        results.pop()
    elif mutation == "id":
        results[0] = replace(results[0], candidate_id="wrong")
    else:
        results[0] = replace(results[0], render_seed=bridge.render_seeds[1].render_seed)
    changed = RenderMatchBatchResult(results=tuple(results), provenance=provenance)

    with pytest.raises(ValueError, match="order or count|candidate ID|input seed"):
        freeze_photo_beam_pnp_benchmark(bridge, changed, lane_label="lane", lane_config={})


def test_freeze_rejects_non_json_diagnostics() -> None:
    bridge = _bridge()
    batch = _batch(bridge)
    bad_diagnostics = {**batch.results[0].result.diagnostics, "not_json": object()}
    bad_result = replace(batch.results[0].result, diagnostics=bad_diagnostics)
    changed = replace(batch, results=(replace(batch.results[0], result=bad_result), *batch.results[1:]))

    with pytest.raises(ValueError, match="diagnostics.*finite canonical JSON"):
        freeze_photo_beam_pnp_benchmark(bridge, changed, lane_label="lane", lane_config={})


def test_validator_hash_checks_detaches_and_rejects_structural_rehash() -> None:
    record = _frozen_record()

    validated = validate_frozen_photo_beam_pnp_benchmark(record)
    record["results"][0]["rng_seed"] = 999
    assert validated["results"][0]["rng_seed"] == 101

    tampered = deepcopy(validated)
    tampered["results"][0]["rng_seed"] = 999
    with pytest.raises(ValueError, match="SHA-256"):
        validate_frozen_photo_beam_pnp_benchmark(tampered)

    tampered = deepcopy(validated)
    tampered["results"][0]["candidate_id"] = "seed-b"
    _rehash(tampered)
    with pytest.raises(ValueError, match="ID/rank linkage"):
        validate_frozen_photo_beam_pnp_benchmark(tampered)


def test_post_freeze_evaluation_reports_stage_recall_counts_validation_and_oracles() -> None:
    record = _frozen_record()

    evaluation = evaluate_photo_beam_pnp_benchmark(record, _pose(0.0, 0.0))

    assert evaluation["target"] == {
        "horizontal_position_m_lte": 100.0,
        "absolute_yaw_deg_lte": 5.0,
        "pitch_used": False,
        "roll_used": False,
    }
    assert evaluation["input_seed"]["first_target_beam_rank"] == 2
    assert evaluation["pre_validation_pnp"]["first_target_beam_rank"] == 1
    assert evaluation["accepted_output"]["first_target_beam_rank"] == 3
    assert evaluation["input_seed"]["target_candidate_count"] == 1
    assert evaluation["pre_validation_pnp"]["target_candidate_count"] == 2
    assert evaluation["accepted_output"]["target_candidate_count"] == 1
    assert evaluation["per_seed"][0]["pre_validation_pnp"]["errors"]["horizontal_position_m"] == 20.0
    assert evaluation["per_seed"][0]["accepted_output"] is None
    assert evaluation["per_seed"][0]["match_counts"]["after_render_lifting"] == 30
    assert evaluation["per_seed"][0]["candidate_validation"]["passed"] is False
    assert evaluation["gt_oracles"]["evaluation_only"] is True
    assert evaluation["gt_oracles"]["used_by_estimator"] is False
    assert evaluation["gt_oracles"]["changes_estimator_selection"] is False
    assert evaluation["gt_oracles"]["input_seed"]["candidate_id"] == "seed-b"
    assert evaluation["gt_oracles"]["pre_validation_pnp"]["candidate_id"] == "seed-a"
    assert evaluation["estimator_selection_performed"] is False


def test_truth_can_only_enter_the_separate_evaluator() -> None:
    freeze_parameters = list(__import__("inspect").signature(freeze_photo_beam_pnp_benchmark).parameters)
    evaluate_parameters = list(__import__("inspect").signature(evaluate_photo_beam_pnp_benchmark).parameters)

    assert freeze_parameters == ["bridge", "batch", "lane_label", "lane_config"]
    assert evaluate_parameters == ["archive_record", "truth"]
