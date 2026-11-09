"""Check the frozen run matrix against the historical evidence and sampling budgets."""
from __future__ import annotations

import copy
import json
import shutil

import pytest

import make_revision_manifest as M
import revision_task as T

# The reused runs of the historical cohorts are withheld from the anonymous archive.
HISTORICAL = M._source_dir(M.expected_cells()[1][0], M.HERE / "runs")
needs_history = pytest.mark.skipif(not HISTORICAL.exists(), reason="the historical run folders are withheld from the anonymous archive")


@pytest.fixture(scope="module")
def bundle():
    if not HISTORICAL.exists():
        pytest.skip("the historical run folders are withheld from the anonymous archive")
    dataset = T.build_dataset()
    return dataset, M.build_manifest(dataset)


def _rehash(manifest):
    manifest["sha256"] = M.manifest_digest(manifest)
    return manifest


@pytest.fixture
def copied_original(tmp_path):
    if not HISTORICAL.exists():
        pytest.skip("the historical run folders are withheld from the anonymous archive")
    record = M.expected_cells()[1][0]
    target = M._source_dir(record, tmp_path)
    shutil.copytree(M._source_dir(record, M.HERE / "runs"), target)
    return record, tmp_path, target


def test_full_study_has_fifty_legacy_and_thirty_five_strict_cells(bundle):
    dataset, manifest = bundle
    new, old = manifest["new_runs"], manifest["reused_runs"]
    assert len(new) == 68 and len(old) == 17
    assert [row["protocol"] for row in new] == ["legacy"] * 33 + ["strict"] * 35
    assert [row["row"] for row in new] == list(range(1, 69))
    legacy = [row for row in new + old if row["protocol"] == "legacy"]
    assert {(row["method"], row["config"]["w"], row["config"]["seed"]) for row in legacy} == {
        (method, weight, seed) for method in M.METHODS for weight in (5, 25) for seed in range(5)}
    strict = [row for row in new if row["protocol"] == "strict"]
    assert sum(row["config"]["group_size"] == 4 and row["config"]["reward"] == "binary" for row in strict) == 20
    assert sum(row["config"]["reward"] == "graded" for row in strict) == 5
    assert sum(row["config"]["group_size"] == 8 for row in strict) == 10
    assert all(row["training_completions"] == 7200 for row in new + old)
    assert all(row["config"]["max_new_tokens"] == 128 for row in new + old)
    assert all(row["config"]["reward_scale"] == 26 for row in strict)
    assert all(row["config"]["score_microbatch"] == 4 for row in strict)
    assert dataset["common_above_crossover_weight"] == 17
    assert manifest["frozen_common"]["verdict_weights"] == [5, 25]
    assert all(row["config"]["epochs"] == 4 and row["config"]["no_ref"] for row in strict)
    assert all(not row["config"]["skip_heldout"] for row in strict)
    amendment = manifest["prelaunch_amendments"][0]
    assert amendment["stage"] == "before_any_training"
    assert amendment["previous_value"] == 1 and amendment["new_value"] == 4
    assert amendment["prior_manifest_sha256"] == "7505882e05ca2d3d03135fe5980a77abe92ba3e9c4523e30556a00444c6b33d2"


def test_historical_reuse_records_include_exact_artifacts_and_explicit_waivers(bundle):
    dataset, manifest = bundle
    assert M.validate_manifest(manifest, dataset, runs_dir=M.HERE / "runs") is manifest
    reused = manifest["reused_runs"]
    assert all(set(row["artifact_sha256"]) == set(M.ARTIFACTS) for row in reused)
    waived = [row for row in reused if row["provenance_waiver"]["required"]]
    assert [row["run_id"] for row in waived] == [f"legacy_plain_w5_s{seed}" for seed in range(3)]
    assert all(row["recorded_training_sources_digest"] is None for row in waived)
    assert all(row["provenance_waiver"]["id"] == "F4" for row in waived)
    assert all(not row["source_path"].startswith("/") for row in reused)


@pytest.mark.parametrize("change, match", [
    ("duplicate", "duplicate"), ("missing", "68 new"), ("budget", "budget"),
    ("weight", "matrix"), ("waiver", "F4"), ("source", "historical source"),
])
def test_rehashed_semantic_errors_are_rejected(bundle, change, match):
    dataset, original = bundle
    changed = copy.deepcopy(original)
    if change == "duplicate":
        changed["new_runs"][1] = copy.deepcopy(changed["new_runs"][0])
    elif change == "missing":
        changed["new_runs"].pop()
    elif change == "budget":
        changed["new_runs"][-1]["config"]["steps"] = 300
    elif change == "weight":
        changed["new_runs"][33]["config"]["weight"] = 17
    elif change == "waiver":
        changed["reused_runs"][0]["provenance_waiver"]["required"] = False
    else:
        changed["reused_runs"][0]["source_path"] = changed["reused_runs"][1]["source_path"]
    with pytest.raises(ValueError, match=match):
        M.validate_manifest(_rehash(changed), dataset)


def test_unrehashed_tampering_is_rejected(bundle):
    dataset, original = bundle
    changed = copy.deepcopy(original)
    changed["new_runs"][0]["config"]["lr"] *= 2
    with pytest.raises(ValueError, match="digest"):
        M.validate_manifest(changed, dataset)


@pytest.mark.parametrize("change, match", [
    ("missing", "missing"), ("nonterminal", "terminal"),
    ("incomplete", "budget"), ("duplicate_step", "duplicate"),
    ("missing_step", "missing"), ("group_budget", "group budget"),
    ("config_budget", "configuration"), ("provenance_hash", "config hash"),
])
def test_reuse_requires_complete_original_records(copied_original, change, match):
    record, runs_dir, path = copied_original
    if change == "missing":
        (path / "metrics.jsonl").unlink()
    elif change == "nonterminal":
        (path / "DONE").write_text("failed\n")
    elif change in ("incomplete", "provenance_hash"):
        p = path / "provenance.json"
        value = json.loads(p.read_text())
        value["steps_completed" if change == "incomplete" else "config_hash"] = 299 if change == "incomplete" else "wrong"
        p.write_text(json.dumps(value))
    elif change == "config_budget":
        p = path / "config.json"
        value = json.loads(p.read_text())
        value["group_size"] = 8
        p.write_text(json.dumps(value))
    else:
        p = path / "metrics.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        index = next(index for index, row in enumerate(rows) if row["phase"] == "train")
        if change == "duplicate_step":
            rows.insert(index, copy.deepcopy(rows[index]))
        elif change == "missing_step":
            rows.pop(index)
        else:
            rows[index]["groups"][0]["selections"].pop()
        p.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match=match):
        M.inspect_reused_run(record, runs_dir)


def test_reuse_hashes_detect_changed_metrics_without_a_budget_change(copied_original):
    record, runs_dir, path = copied_original
    original = M.inspect_reused_run(record, runs_dir)
    p = path / "metrics.jsonl"
    rows = p.read_text().splitlines()
    first = json.loads(rows[0])
    first["eval_mean_n_selected"] = 123.0
    rows[0] = json.dumps(first)
    p.write_text("\n".join(rows) + "\n")
    changed = M.inspect_reused_run(record, runs_dir)
    assert changed["artifact_sha256"]["metrics.jsonl"] != original["artifact_sha256"]["metrics.jsonl"]
    assert changed["artifact_sha256"]["config.json"] == original["artifact_sha256"]["config.json"]


def test_existing_different_artifacts_are_never_overwritten(bundle, tmp_path):
    dataset, manifest = bundle
    M.write_bundle(tmp_path, dataset, manifest)
    original_dataset = (tmp_path / "revision_dataset.json").read_bytes()
    target = tmp_path / "revision_manifest.json"
    target.write_text("A different frozen study must remain intact.\n")
    with pytest.raises(FileExistsError, match="Refusing"):
        M.write_bundle(tmp_path, dataset, manifest)
    assert target.read_text() == "A different frozen study must remain intact.\n"
    assert (tmp_path / "revision_dataset.json").read_bytes() == original_dataset


@needs_history
def test_cli_can_create_and_recheck_a_scratch_bundle_without_rewriting(tmp_path, capsys):
    assert M.main(["--out-dir", str(tmp_path)]) == 0
    paths = [tmp_path / name for name in ("revision_dataset.json", "revision_manifest.json")]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    assert M.main(["--out-dir", str(tmp_path)]) == 0
    assert before == [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(output) == 2 and output[0] == output[1]
    assert output[0]["new_runs"] == 68 and output[0]["reused_runs"] == 17
