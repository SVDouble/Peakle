from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from peakle.scripts import roma_match_worker as worker


class ArrayTensor:
    """Small Torch-tensor stand-in for testing post-processing without importing Torch."""

    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def detach(self) -> ArrayTensor:
        return self

    def float(self) -> ArrayTensor:
        return self

    def cpu(self) -> ArrayTensor:
        return self

    def __array__(self, dtype: np.dtype[object] | None = None) -> np.ndarray:
        return np.asarray(self.value, dtype=dtype)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _manifest_fixture(tmp_path: Path, matcher_id: str = "roma_outdoor") -> tuple[Path, dict[str, object]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "worker-test@example.invalid")
    _git(repo, "config", "user.name", "Worker Test")
    (repo / "README.md").write_text("pinned checkout\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "--quiet", "-m", "fixture")

    dependencies = tmp_path / "deps"
    dependencies.mkdir()
    (dependencies / "dependency.py").write_text("VERSION = 1\n")

    models = tmp_path / "models"
    models.mkdir()
    roma = models / "matcher.pth"
    dino = models / "dino.pth"
    roma.write_bytes(b"matcher weights")
    dino.write_bytes(b"backbone weights")
    source: dict[str, object] = {
        "schema": worker.MANIFEST_SCHEMA,
        "matcher_id": matcher_id,
        "repository": {
            "path": "../repo",
            "commit": _git(repo, "rev-parse", "HEAD"),
            "require_clean": True,
            "python_paths": ["../deps"],
        },
        "artifacts": [
            {
                "role": worker.ROMA_WEIGHTS_ROLE,
                "path": roma.name,
                "sha256": _sha256(roma),
                "size_bytes": roma.stat().st_size,
                "source": "fixture://matcher",
            },
            {
                "role": worker.DINO_WEIGHTS_ROLE,
                "path": dino.name,
                "sha256": _sha256(dino),
                "size_bytes": dino.stat().st_size,
                "source": "fixture://backbone",
            },
        ],
        "inference": {
            "device": "cpu",
            "coarse_res": 560,
            "upsample_res": 864,
            "max_matches": 32,
            "selection_cell_px": 16,
            "min_confidence": 0.1,
            "amp_dtype": "float32",
            "symmetric": True,
            "use_custom_corr": False,
            "upsample_preds": True,
        },
        "licenses": {"code": "fixture"},
    }
    manifest_path = models / "manifest.json"
    manifest_path.write_text(json.dumps(source, indent=2) + "\n")
    normalized = json.loads(json.dumps(source))
    for record in normalized["artifacts"]:
        record["path"] = str((models / record["path"]).resolve())
    normalized["manifest_path"] = str(manifest_path.resolve())
    return manifest_path, normalized


@pytest.mark.parametrize("matcher_id", sorted(worker.ALLOWED_MATCHER_IDS))
def test_manifest_verification_pins_source_repository_artifacts_and_matcher(
    tmp_path: Path,
    matcher_id: str,
) -> None:
    _, normalized = _manifest_fixture(tmp_path, matcher_id)

    verified = worker._verify_manifest(normalized, matcher_id)

    assert verified.matcher_id == matcher_id
    assert verified.repository.commit == normalized["repository"]["commit"]
    assert set(verified.artifacts) == worker.REQUIRED_ARTIFACT_ROLES
    assert verified.repository.python_path_fingerprints[0]["file_count"] == 1


def test_manifest_verification_rejects_normalized_copy_tampering(tmp_path: Path) -> None:
    _, normalized = _manifest_fixture(tmp_path)
    normalized["inference"]["max_matches"] = 31

    with pytest.raises(worker.WorkerInputError, match="differs from its source file"):
        worker._verify_manifest(normalized, "roma_outdoor")


def test_manifest_verification_rejects_dirty_repository(tmp_path: Path) -> None:
    _, normalized = _manifest_fixture(tmp_path)
    (tmp_path / "repo" / "untracked.txt").write_text("dirty\n")

    with pytest.raises(worker.MissingModelError, match="repository is dirty"):
        worker._verify_manifest(normalized, "roma_outdoor")


def test_request_header_uses_an_exact_matcher_allowlist() -> None:
    base = {
        "schema": worker.REQUEST_SCHEMA,
        "seed": 7,
        "model_manifest": {},
        "query": {},
        "contract": {
            "network_allowed": False,
            "implicit_model_downloads_allowed": False,
            "coordinates": "original-resolution top-left x/y pixel centres",
        },
    }
    for matcher_id in worker.ALLOWED_MATCHER_IDS:
        assert worker._validate_request_header({**base, "matcher_id": matcher_id}) == matcher_id

    with pytest.raises(worker.WorkerInputError, match="only accepts matcher_id"):
        worker._validate_request_header({**base, "matcher_id": "arbitrary_roma_checkpoint"})


def test_deterministic_matches_return_original_resolution_pixel_centres() -> None:
    config = worker.InferenceConfig(
        coarse_res=560,
        upsample_res=864,
        max_matches=8,
        selection_cell_px=16,
        min_confidence=0.1,
        amp_dtype="float32",
        symmetric=True,
        use_custom_corr=False,
        upsample_preds=True,
        device="cpu",
    )
    warp = ArrayTensor(
        [
            [-0.75, -0.50, 0.00, 0.50],
            [-0.75, -0.50, 0.00, 0.50],  # symmetric duplicate
            [0.50, 0.20, -0.50, -0.20],
            [1.00, 0.00, 0.00, 0.00],  # RoMa's invalid clamp sentinel
            [0.00, 0.00, 0.00, 0.00],  # below confidence threshold
        ]
    )
    certainty = ArrayTensor([0.9, 0.8, 0.7, 0.95, 0.05])

    matches = worker._deterministic_matches(
        warp,
        certainty,
        query_shape=(100, 200, 3),
        render_shape=(50, 80, 3),
        config=config,
    )

    np.testing.assert_allclose(matches.query_xy_px, [[25.0, 25.0], [150.0, 60.0]])
    np.testing.assert_allclose(matches.render_xy_px, [[40.0, 37.5], [20.0, 20.0]])
    np.testing.assert_allclose(matches.confidence, [0.9, 0.7])
    assert matches.dense_candidates == 5
    assert matches.valid_candidates == 3
    assert matches.unique_candidates == 2


def test_joint_grid_selection_is_deterministic_and_spreads_both_images() -> None:
    query = np.asarray([[1, 1], [2, 2], [21, 1], [41, 1], [61, 1]], dtype=np.float64)
    render = np.asarray([[1, 1], [21, 1], [2, 2], [41, 1], [61, 1]], dtype=np.float64)
    confidence = np.asarray([0.99, 0.98, 0.97, 0.96, 0.95], dtype=np.float64)

    selected, unique = worker._joint_grid_select(
        query,
        render,
        confidence,
        max_matches=3,
        cell_px=16,
    )

    assert selected.tolist() == [0, 3, 4]
    assert unique == 5


def test_torch_hub_entry_points_are_blocked_and_restored() -> None:
    original_state_dict = object()
    original_load = object()
    torch = SimpleNamespace(
        hub=SimpleNamespace(
            load_state_dict_from_url=original_state_dict,
            load=original_load,
        )
    )

    with worker._blocked_torch_downloads(torch):
        with pytest.raises(RuntimeError, match="downloads are disabled"):
            torch.hub.load_state_dict_from_url("https://example.invalid/model.pth")
        with pytest.raises(RuntimeError, match="downloads are disabled"):
            torch.hub.load("owner/repository", "model")

    assert torch.hub.load_state_dict_from_url is original_state_dict
    assert torch.hub.load is original_load


def test_main_writes_structured_missing_model_result(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(
        json.dumps(
            {
                "schema": worker.REQUEST_SCHEMA,
                "matcher_id": "minima_roma",
                "seed": 3,
                "query": {},
                "renders": [{}],
                "outputs": {"result_json": result_path.name},
                "model_manifest": {
                    "schema": worker.MANIFEST_SCHEMA,
                    "matcher_id": "minima_roma",
                    "manifest_path": str(tmp_path / "missing-manifest.json"),
                },
                "contract": {
                    "coordinates": "original-resolution top-left x/y pixel centres",
                    "network_allowed": False,
                    "implicit_model_downloads_allowed": False,
                },
            }
        )
    )

    assert worker.main(["--request", str(request_path)]) == 0
    result = json.loads(result_path.read_text())
    assert result["schema"] == worker.RESULT_SCHEMA
    assert result["status"] == "SKIPPED_MISSING_MODEL"
    assert result["error_type"] == "MissingModelError"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("CUDA out of memory. Tried to allocate 1 GiB", True),
        ("cannot construct an optional CUDA extension", False),
        ("CPU out of memory", False),
    ],
)
def test_cuda_oom_classification(message: str, expected: bool) -> None:
    assert worker._is_cuda_oom(RuntimeError(message)) is expected
