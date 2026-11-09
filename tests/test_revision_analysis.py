"""Exercise independent evidence checks with portable synthetic test records."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import statistics

import numpy as np
import pytest

import analyze_revision as A
import make_revision_manifest as M
import revision_task as T
import rewards
import run_revision_cell as runner
from train_revision import summarize_records

# The reused runs of the historical cohorts are withheld from the anonymous archive.
HISTORICAL = M._source_dir(M.expected_cells()[1][0], M.HERE / "runs")
needs_history = pytest.mark.skipif(not HISTORICAL.exists(), reason="the historical run folders are withheld from the anonymous archive")


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def reseal(path):
    provenance = A.read_json(path / "provenance.json")
    cfg = A.read_json(path / "config.json")
    provenance["config_sha256"] = A.digest({k: v for k, v in cfg.items() if k not in ("out", "dataset")})
    provenance["artifact_sha256"] = {name: A.file_hash(path / name) for name in provenance["artifact_sha256"]}
    write_json(path / "provenance.json", provenance)
    write_json(path / "DONE", {"status": "complete", "steps_completed": cfg["steps"],
                                "provenance_sha256": A.file_hash(path / "provenance.json")})


@pytest.fixture(scope="module")
def dataset():
    return T.build_dataset(heldout_count=2)


def create_synthetic_strict(path, dataset, method="ppo"):
    """Create format fixtures without representing them as trained models or study evidence."""
    path.mkdir()
    cfg = {**M.strict_config(method, "binary", 25.0, 0, 4), "group_size": 2, "steps": 2,
           "epochs": 2, "eval_every": 1, "eval_samples": 2, "max_new_tokens": 16,
           "tiny": True, "device": "cpu", "out": str(path), "dataset": str(path / "dataset.json"),
           "protocol_version": A.STRICT_PROTOCOL, "dataset_sha256": dataset["sha256"],
           "kl_beta": 0.0, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
           "repetition_penalty": 1.0, "optimizer_steps": 4, "training_completions": 24,
           "critic_type": "learned_prompt_value" if method == "ppo" else None}
    write_json(path / "config.json", cfg)
    write_json(path / "dataset.json", dataset)
    write_json(path / "prompt_tokens.json", {split: [{"instance_id": row["instance_id"], "prompt_token_ids": [1, 2]}
                                                      for row in dataset[split]] for split in ("train", "heldout")})
    schedule = [("eval", "train", 0), ("train", "train", 1), ("eval", "train", 1),
                ("train", "train", 2), ("eval", "train", 2), ("eval", "heldout", 2)]
    records, metrics = [], []
    values = np.zeros(6)
    for phase, split, step in schedule:
        event, groups = [], []
        for instance_index, instance in enumerate(dataset[split]):
            feasible_item = next(index for index, weight in enumerate(instance["weights"]) if weight <= instance["capacity"])
            group = []
            for sample in range(2):
                text = json.dumps([feasible_item]) if sample == 0 else ("[]" if split == "train" else "invalid")
                if method == "grpo" and phase == "train":
                    text = "invalid"
                score = T.score_completion(text, instance, "binary", 25)
                row = {**score, "phase": phase, "split": split, "step": step, "sample_index": sample,
                       "scaled_reward": score["reward"] / 26, "completion_token_ids": [42, 151645],
                       "n_completion_tokens": 2}
                if phase == "train":
                    row.update({"behavior_log_prob": -1.0, "old_value": float(values[instance_index])})
                group.append(row)
            if phase == "train":
                returns = np.asarray([row["scaled_reward"] for row in group])
                advantage = (rewards.advantage(returns).astype(np.float32) if method == "grpo" else
                             returns.astype(np.float32) - np.float32(values[instance_index]))
                for row, advantage in zip(group, advantage):
                    row["advantage"] = float(advantage)
            event.extend(group)
            groups.append(group)
        metric = {"phase": phase, "split": split, "step": step, **summarize_records(event)}
        if phase == "eval":
            metric["per_instance"] = [{"instance_id": instance["instance_id"], **summarize_records(group)}
                                      for instance, group in zip(dataset[split], groups)]
        else:
            metric.update({"s_per_step": 0.01,
                           "all_violating_fraction": float(np.mean([not any(row["useful_feasible"] for row in group) for group in groups])),
                           "zero_advantage_group_fraction": float(np.mean([all(row["advantage"] == 0 for row in group) for group in groups])),
                           "group_reward_std_mean": float(np.mean([np.std([row["scaled_reward"] for row in group], ddof=1) for group in groups])),
                           "actor_epochs": [{"epoch": epoch, "policy_loss": -float(np.mean([row["advantage"] for row in event])),
                                             "clip_fraction": 0.0, "active_clip_fraction": 0.0,
                                             "approximate_behavior_kl": 0.0, "ratio_min": 1.0,
                                             "ratio_max": 1.0, "grad_norm_preclip": 0.0} for epoch in (1, 2)]})
            if method == "ppo":
                targets = np.asarray([[row["scaled_reward"] for row in group] for group in groups], dtype=np.float32)
                initial = float(np.mean((values[:, None] - targets) ** 2))
                for _ in range(cfg["critic_steps"]):
                    values = (values - (2 * cfg["critic_lr"] / 6) * (values - targets.mean(axis=1))).astype(np.float32)
                metric.update({"critic_fit_steps": cfg["critic_steps"], "critic_mse_before": initial,
                               "critic_mse_after": float(np.mean((values[:, None] - targets) ** 2)),
                               "critic_values": values.astype(float).tolist()})
            else:
                metric["critic_fit_steps"] = 0
        records.extend(event)
        metrics.append(metric)
    write_jsonl(path / "completions.jsonl", records)
    write_jsonl(path / "metrics.jsonl", metrics)
    if method == "ppo":
        write_json(path / "critic.json", {"state_dict": {"values.weight": [[float(value)] for value in values]}})
    sources = {name: A.file_hash(A.HERE / name) for name in A.STRICT_SOURCES}
    provenance = {"protocol_version": A.STRICT_PROTOCOL, "dataset_sha256": dataset["sha256"],
                  "config_sha256": "placeholder", "steps_completed": 2, "end_time": "synthetic_fixture",
                  "source_files": sources, "source_sha256": A.digest(sources),
                  "device": "cpu", "device_name": "synthetic_fixture", "slurm_job_id": None,
                  "completion_mask_stop_ids": [151645],
                  "effective_generation_config": {"do_sample": True, "temperature": 1.0, "top_p": 1.0,
                       "top_k": 0, "repetition_penalty": 1.0, "num_beams": 1, "max_new_tokens": 16,
                       "eos_token_id": [151645]},
                  "artifact_sha256": {entry.name: "placeholder" for entry in path.iterdir()}}
    write_json(path / "provenance.json", provenance)
    reseal(path)
    return path


@pytest.fixture
def strict_fixture(tmp_path, dataset):
    return create_synthetic_strict(tmp_path / "run", dataset)


def test_strict_independent_replay_covers_both_splits_and_critic(strict_fixture, dataset):
    result = A.validate_strict_run(strict_fixture, dataset=dataset, smoke=True)
    assert result["training_completions"] == 24
    assert len(result["finals"]["train"]) == 12 and len(result["finals"]["heldout"]) == 4
    assert A.strict_summary(result["finals"]["heldout"])["useful_feasible_rate"] == 0.5
    assert A.strict_summary(result["finals"]["heldout"])["parse_fail_rate"] == 0.5


def test_zero_advantage_groups_are_valid_recorded_outcomes(tmp_path, dataset):
    path = create_synthetic_strict(tmp_path / "zero", dataset, method="grpo")
    result = A.validate_strict_run(path, smoke=True)
    assert result["zero_advantage_groups"] == result["training_groups"] == 12


def test_smoke_records_cannot_enter_final_endpoint_tables(strict_fixture):
    result = A.validate_strict_run(strict_fixture, smoke=True)
    with pytest.raises(A.ValidationError, match="cannot enter final study"):
        A.result_rows({"protocol": "strict"}, result)


@pytest.mark.parametrize("change, match", [
    ("score", "raw_score"), ("scaled", "scaled_reward"), ("duplicate", "duplicate completion"),
    ("missing", "budget"), ("critic", "old_value"), ("advantage", "advantage"),
    ("tokens", "token count"), ("stop", "stop token"),
    ("sample_type", "sample index"),
])
def test_resealed_raw_record_tampering_is_rejected(strict_fixture, change, match):
    path = strict_fixture / "completions.jsonl"
    rows = A.read_jsonl(path)
    index = next(index for index, row in enumerate(rows) if row["phase"] == "train")
    if change == "score":
        rows[index]["useful_value"] += 0.2
    elif change == "scaled":
        rows[index]["scaled_reward"] += 0.2
    elif change == "duplicate":
        rows.insert(index, copy.deepcopy(rows[index]))
    elif change == "missing":
        rows.pop(index)
    elif change == "critic":
        rows[index]["old_value"] = 0.5
    elif change == "advantage":
        rows[index]["advantage"] += 0.2
    elif change == "tokens":
        rows[index]["n_completion_tokens"] = 129
    elif change == "sample_type":
        rows[index]["sample_index"] = 0.0
    else:
        rows[index]["completion_token_ids"] = [151645, 42]
    write_jsonl(path, rows)
    reseal(strict_fixture)
    with pytest.raises(A.ValidationError, match=match):
        A.validate_strict_run(strict_fixture, smoke=True)


@pytest.mark.parametrize("change, match", [
    ("summary", "summary"), ("instance", "per_instance"), ("heldout_early", "schedule"),
    ("missing_epoch", "epoch budget"), ("critic_fit", "critic_values"),
])
def test_resealed_metrics_tampering_is_rejected(strict_fixture, change, match):
    path = strict_fixture / "metrics.jsonl"
    rows = A.read_jsonl(path)
    if change == "summary":
        rows[-1]["useful_feasible_rate"] = 1.0
    elif change == "instance":
        rows[-1]["per_instance"][0]["n_completions"] = 99
    elif change == "heldout_early":
        rows[-1]["step"] = 1
    elif change == "missing_epoch":
        rows[1]["actor_epochs"].pop()
    else:
        rows[1]["critic_values"][0] = 0.9
    write_jsonl(path, rows)
    reseal(strict_fixture)
    with pytest.raises(A.ValidationError, match=match):
        A.validate_strict_run(strict_fixture, smoke=True)


def test_hash_and_failure_marker_gates_precede_score_replay(strict_fixture):
    original = (strict_fixture / "completions.jsonl").read_text()
    (strict_fixture / "completions.jsonl").write_text(original + "\n")
    with pytest.raises(A.ValidationError, match="artifact"):
        A.validate_strict_run(strict_fixture, smoke=True)
    (strict_fixture / "FAILED").write_text("failure")
    with pytest.raises(A.ValidationError, match="failure markers"):
        A.validate_strict_run(strict_fixture, smoke=True)


@needs_history
def test_historical_replay_keeps_empty_feasibility_separate_from_useful_value():
    manifest = A.read_json(M.HERE / "configs/revision_manifest.json")
    record = next(row for row in manifest["reused_runs"] if row["run_id"] == "legacy_drgrpo_w5_s0")
    path = M._source_dir(record, M.HERE / "runs")
    result = A.validate_legacy_run(path, record, reused=True)
    quality = A.legacy_quality(result["finals"]["train"], 0.03)
    assert result["training_completions"] == 7200
    assert quality["recorded_verdict_failure_rate"] == 0
    assert quality["empty_selection_rate"] == 1
    assert quality["useful_value_mean"] == 0 and quality["useful_feasible_rate"] == 0


@pytest.fixture
def launch_fixture(tmp_path):
    manifest_path, dataset_path = M.HERE / "configs/revision_manifest.json", M.HERE / "configs/revision_dataset.json"
    manifest = A.read_json(manifest_path)
    cell = next(row for row in manifest["new_runs"] if row["protocol"] == "strict")
    path = tmp_path / "study" / cell["run_id"] / "job_123"
    path.mkdir(parents=True)
    out = Path("/frozen/runs/study") / cell["run_id"] / "job_123"
    remote_dataset = Path("/frozen/configs/revision_dataset.json")
    cfg = {**cell["config"], "out": str(out), "dataset": str(remote_dataset), "device": "cuda"}
    write_json(path / "config.json", cfg)
    names = {"configs/revision_manifest.json": A.file_hash(manifest_path),
             "configs/revision_dataset.json": A.file_hash(dataset_path),
             **{name: A.file_hash(A.HERE / name) for name in runner.REQUIRED_SOURCES}}
    entries = [{"path": name, "sha256": value} for name, value in sorted(names.items())]
    snapshot = {"source_sha256": A.digest(entries), "base_commit": "test",
                "snapshot_type": "uncommitted-content-frozen-copy", "files": entries}
    snapshot_hash = A.hashlib.sha256((json.dumps(snapshot, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
    launch = {"run_id": cell["run_id"], "row": cell["row"], "phase": "study", "slurm_job_id": "123",
              "command": runner.make_command(cell, Path("/frozen"), remote_dataset, out),
              "manifest_file_sha256": A.file_hash(manifest_path), "dataset_file_sha256": A.file_hash(dataset_path),
              "snapshot_file_sha256": snapshot_hash, "snapshot": snapshot, "cell": cell,
              "started_unix": 1, "ended_unix": 2, "exit_code": 0, "done_marker_exists": True}
    write_json(path.parent / "launch_123.json", launch)
    return path, cell, manifest_path, dataset_path


def test_launch_contract_binds_command_phase_and_frozen_content(launch_fixture):
    path, cell, manifest, dataset = launch_fixture
    assert A.validate_launch(path, cell, manifest, dataset)["record"]["phase"] == "study"


def test_rehashed_snapshot_cannot_omit_the_launcher(launch_fixture):
    path, cell, manifest, dataset = launch_fixture
    record_path = path.parent / "launch_123.json"
    record = A.read_json(record_path)
    snapshot = record["snapshot"]
    snapshot["files"] = [item for item in snapshot["files"] if item["path"] != "run_revision_cell.py"]
    snapshot["source_sha256"] = A.digest(snapshot["files"])
    record["snapshot_file_sha256"] = A.hashlib.sha256(
        (json.dumps(snapshot, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
    write_json(record_path, record)
    with pytest.raises(A.ValidationError, match="inventory is incomplete"):
        A.validate_launch(path, cell, manifest, dataset)


@pytest.mark.parametrize("change, match", [("phase", "phase"), ("command", "command"),
                                          ("manifest", "manifest_file"), ("snapshot", "snapshot_file"),
                                          ("exit", "exit_code")])
def test_invalid_launch_evidence_is_rejected(launch_fixture, change, match):
    path, cell, manifest, dataset = launch_fixture
    record_path = path.parent / "launch_123.json"
    launch = A.read_json(record_path)
    if change == "phase":
        launch["phase"] = "validation"
    elif change == "command":
        index = launch["command"].index("--steps")
        launch["command"][index + 1] = "20"
    elif change == "manifest":
        launch["manifest_file_sha256"] = "0" * 64
    elif change == "snapshot":
        launch["snapshot_file_sha256"] = "0" * 64
    else:
        launch["exit_code"] = 1
    write_json(record_path, launch)
    with pytest.raises(A.ValidationError, match=match):
        A.validate_launch(path, cell, manifest, dataset)


def test_status_does_not_call_missing_or_terminal_markers_validated_success(tmp_path):
    manifest = A.read_json(M.HERE / "configs/revision_manifest.json")
    status = A.study_status(manifest, tmp_path)
    assert status["counts"]["missing"] == 68 and status["study_results_validated"] is False
    run = manifest["new_runs"][0]["run_id"]
    path = tmp_path / "study" / run / "job_1"
    path.mkdir(parents=True)
    (path / "DONE").write_text("ok")
    write_json(path.parent / "launch_1.json", {"phase": "study", "slurm_job_id": "1", "exit_code": 0, "done_marker_exists": True})
    status = A.study_status(manifest, tmp_path)
    assert status["counts"]["completed_unvalidated"] == 1
    assert status["study_results_validated"] is False
    with pytest.raises(A.ValidationError, match="complete matrix"):
        A.select_attempts(status, tmp_path)


def test_duplicate_completed_attempts_require_an_explicit_selection(tmp_path):
    status = {"unexpected_study_directories": [], "cells": [{"run_id": "example", "attempts": [
        {"attempt": "job_1", "state": "completed_unvalidated"}, {"attempt": "job_2", "state": "completed_unvalidated"}]}]}
    with pytest.raises(A.ValidationError, match="multiple completed"):
        A.select_attempts(status, tmp_path)
    selected = A.select_attempts(status, tmp_path, {"selections": {"example": "job_1"}})
    assert selected["example"].name == "job_1"


def synthetic_per_seed():
    manifest = A.read_json(M.HERE / "configs/revision_manifest.json")
    rows = []
    for cell in manifest["new_runs"] + manifest["reused_runs"]:
        cfg = cell["config"]
        strict = cell["protocol"] == "strict"
        reward, weight = (cfg["reward"], cfg["weight"]) if strict else (cfg["arm"], cfg["w"])
        condition = A.condition_id(cell["protocol"], cell["method"], reward, weight, cfg["group_size"])
        for split in (("train", "heldout") if strict else ("train",)):
            value = cfg["seed"] / 10 + (0.1 if cell["method"] == "ppo" else 0)
            rows.append({"condition_id": condition, "protocol": cell["protocol"], "split": split,
                         "seed": cfg["seed"], "evaluation_completions": 384 if split == "heldout" else 96,
                         **{key: (None if key == "completion_tokens_mean" and not strict else value)
                            for key in A.ENDPOINT_METRICS}})
    return rows


def test_seed_aggregation_uses_five_replicates_and_sample_standard_deviation():
    summary, paired = A.aggregate_seed_results(synthetic_per_seed())
    assert len(summary) == 274 and len(paired) == 290
    row = next(row for row in summary if row["condition_id"] == "strict_grpo_binary_w5_k4"
               and row["split"] == "heldout" and row["metric"] == "useful_value_mean")
    assert row["n_seeds"] == 5 and row["mean"] == pytest.approx(0.2)
    assert row["sample_sd"] == pytest.approx(statistics.stdev([0, 0.1, 0.2, 0.3, 0.4]))
    comparison = [row for row in paired if row["comparison"] == "strict_ppo_minus_grpo"]
    assert comparison and all(row["difference"] == pytest.approx(0.1) for row in comparison)


def test_seed_aggregation_rejects_duplicate_or_missing_training_replicates():
    rows = synthetic_per_seed()
    rows[0]["seed"] = rows[1]["seed"]
    with pytest.raises(A.ValidationError, match="five distinct"):
        A.aggregate_seed_results(rows)


def test_incomplete_study_analysis_emits_no_result_tables(tmp_path):
    output = tmp_path / "analysis"
    with pytest.raises(A.ValidationError, match="complete matrix"):
        A.analyze_study(M.HERE / "configs/revision_manifest.json", M.HERE / "configs/revision_dataset.json",
                        tmp_path / "runs", M.HERE / "runs", output)
    assert not output.exists()


def policy_evidence_fixture():
    result = []
    for protocol in ("legacy", "strict"):
        for seed in (0, 1):
            result.append({"protocol": protocol, "reused": protocol == "legacy" and seed == 0,
                           "source_sha256": "a" * 64 if protocol == "strict" else "b" * 64,
                           "artifact_sha256": {"prompt_tokens.json": "c" * 64},
                           "initial_policy": {"model_id": M.MODEL_ID,
                               "initial_param_checksum": "d" * 64 if protocol == "strict" else None,
                               "model_revision": "1" * 40 if protocol == "strict" else None,
                               "tokenizer_revision": "2" * 40 if protocol == "strict" else None}})
    return result


def test_initial_policy_checks_preserve_missing_historical_evidence():
    evidence = policy_evidence_fixture()
    report = A.initial_policy_evidence(evidence)
    assert report["strict"]["initial_weight_equality_supported"] is True
    assert report["legacy"]["initial_weight_equality_supported"] is False
    assert report["legacy"]["identifiers"]["initial_param_checksum"]["missing_runs"] == 2
    assert "cannot cryptographically prove" in report["legacy"]["evidence_limit"]
    assert A.shared_source_hashes(evidence) == {"strict": "a" * 64, "new_legacy": "b" * 64}


@pytest.mark.parametrize("field", ["initial_param_checksum", "model_revision", "tokenizer_revision"])
def test_initial_policy_checks_reject_observed_strict_mismatches(field):
    evidence = policy_evidence_fixture()
    evidence[-1]["initial_policy"][field] = "f" * (64 if field == "initial_param_checksum" else 40)
    with pytest.raises(A.ValidationError, match="mismatch"):
        A.initial_policy_evidence(evidence)


def test_initial_policy_checks_reject_observed_legacy_mismatches_and_strict_missing_evidence():
    evidence = policy_evidence_fixture()
    evidence[0]["initial_policy"]["model_revision"] = "1" * 40
    evidence[1]["initial_policy"]["model_revision"] = "2" * 40
    with pytest.raises(A.ValidationError, match="mismatch"):
        A.initial_policy_evidence(evidence)
    evidence = policy_evidence_fixture()
    for row in evidence[2:]:
        for key in ("initial_param_checksum", "model_revision", "tokenizer_revision"):
            row["initial_policy"][key] = None
    with pytest.raises(A.ValidationError, match="Strict cells require"):
        A.initial_policy_evidence(evidence)


def test_shared_source_checks_reject_mixed_strict_training_code():
    evidence = policy_evidence_fixture()
    evidence[-1]["source_sha256"] = "e" * 64
    with pytest.raises(A.ValidationError, match="strict cells"):
        A.shared_source_hashes(evidence)


def test_approved_attempt_map_is_bound_to_manifest_and_requires_a_reason(tmp_path):
    manifest = A.read_json(M.HERE / "configs/revision_manifest.json")
    path = tmp_path / "selection.json"
    selected = {"schema_version": "revision_attempt_selection_v1", "manifest_sha256": manifest["sha256"],
                "approved": True, "rationale": "The operator approved the first complete attempt.",
                "selections": {manifest["new_runs"][0]["run_id"]: "job_123"}}
    write_json(path, selected)
    assert A.load_attempt_selection(path, manifest) == selected
    selected["approved"] = False
    write_json(path, selected)
    with pytest.raises(A.ValidationError, match="explicitly approve"):
        A.load_attempt_selection(path, manifest)


@needs_history
def test_complete_matrix_writer_keeps_seed_denominators_and_missing_legacy_token_lengths(tmp_path, monkeypatch):
    manifest_path, dataset_path = M.HERE / "configs/revision_manifest.json", M.HERE / "configs/revision_dataset.json"
    manifest, frozen = A.load_study(manifest_path, dataset_path)
    chosen = {row["run_id"]: tmp_path / "study" / row["run_id"] / "job_fixture" for row in manifest["new_runs"]}
    monkeypatch.setattr(A, "select_attempts", lambda *args: chosen)
    monkeypatch.setattr(A, "validate_launch", lambda *args: {"file_sha256": "e" * 64})

    def verified_fixture(path, cell, dataset=None, **kwargs):
        strict = cell["protocol"] == "strict"
        cfg = cell["config"]
        finals = {}
        if strict:
            for split in ("train", "heldout"):
                finals[split] = [{**T.score_completion("invalid", instance, cfg["reward"], cfg["weight"]),
                                  "scaled_reward": T.score_completion("invalid", instance, cfg["reward"], cfg["weight"])["reward"] / 26,
                                  "n_completion_tokens": 2}
                                 for instance in frozen[split] for _ in range(16)]
        else:
            finals["train"] = [(index,) for index in range(6) for _ in range(16)]
        provenance = {"source_sha256" if strict else "training_sources_digest": "a" * 64 if strict else "b" * 64,
                      "initial_param_checksum": "d" * 64 if strict else None,
                      "model_revision": "1" * 40 if strict else None, "tokenizer_revision": "2" * 40 if strict else None}
        return {"config": cfg, "provenance": provenance, "finals": finals,
                "training_completions": 7200, "training_groups": 7200 // cfg["group_size"],
                "zero_advantage_groups": 0, "artifact_sha256": {"prompt_tokens.json": "c" * 64} if strict else {}}

    monkeypatch.setattr(A, "validate_strict_run", verified_fixture)
    monkeypatch.setattr(A, "validate_legacy_run", verified_fixture)
    output = tmp_path / "analysis"
    report = A.analyze_study(manifest_path, dataset_path, tmp_path, M.HERE / "runs", output)
    assert report["new_cells"] == 68 and report["reused_cells"] == 17
    assert report["per_seed_rows"] == 120 and report["per_instance_rows"] == 1350
    assert report["initial_policy_evidence"]["legacy"]["initial_weight_equality_supported"] is False
    with (output / "per_seed.csv").open() as stream:
        rows = list(A.csv.DictReader(stream))
    assert len(rows) == 120
    assert all(row["completion_tokens_mean"] == "" for row in rows if row["protocol"] == "legacy")
    assert all(float(row["completion_tokens_mean"]) == 2 for row in rows if row["protocol"] == "strict")
    assert all(float(row["physical_capacity_violation_rate_over_all_completions"]) == 0 for row in rows)
    assert sum(int(row["evaluation_completions"]) for row in rows) == 21600
    assert all(row["recorded_verdict_failure_rate"] == "0.0" for row in rows if row["protocol"] == "legacy")
    assert all(row["recorded_verdict_failure_rate"] == "1.0" for row in rows if row["protocol"] == "strict")
    assert (output / "validation_sources.json").is_file()


def test_finite_ratio_underflow_is_not_relabelled_as_execution_failure(strict_fixture):
    metrics = A.read_jsonl(strict_fixture / "metrics.jsonl")
    metric = next(row for row in metrics if row["phase"] == "train")
    metric["actor_epochs"][1]["ratio_min"] = 0.0
    write_jsonl(strict_fixture / "metrics.jsonl", metrics)
    reseal(strict_fixture)
    assert A.validate_strict_run(strict_fixture, smoke=True)["smoke_only"]


def test_the_historical_digest_uses_the_frozen_fifteen_source_list():
    """The yoked-control commit appends three training sources. A historical-protocol run was made by
    the frozen trainer at COMMIT_4, which hashed fifteen names, so its recomputed digest must not
    change when the checkout's list grows."""
    import hashlib
    import subprocess

    import train_speed_grpo as trainer
    assert A.LEGACY_TRAINING_SOURCES == tuple(trainer.TRAINING_SOURCES[:15])
    assert A._legacy_training_sources() == A.LEGACY_TRAINING_SOURCES
    prefix = "experiments/cancellation/"
    snapshot, root = {}, Path(__file__).resolve().parents[1]
    for name in A.LEGACY_TRAINING_SOURCES:
        r = subprocess.run(["git", "show", f"COMMIT_4:{name}"],
                           cwd=root, capture_output=True)
        if r.returncode != 0:
            pytest.skip("the commit COMMIT_4 is not in this copy's git history")
        snapshot[name[len(prefix):]] = hashlib.sha256(r.stdout).hexdigest()
    assert A._legacy_source_digest(snapshot) == \
        "d5ea80637093bf77d98730fcbfe33043c57a6dbb987fbc253d2cb989037d1751"
