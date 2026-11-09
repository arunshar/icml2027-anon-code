#!/usr/bin/env python3
"""Verify the frozen revision study and summarize training seeds as replicates.

The status command reports execution coverage without accepting results. The analyze
command requires all 68 new cells and all 17 admitted historical cells. It verifies
launch records, frozen sources, budgets, raw strict scores, and reported aggregates
before writing tables. A failed policy, including an all-zero-advantage policy, is a
valid scientific outcome when its execution and records satisfy these checks.

No excluded cells or significance tests are supported. An explicit attempt-selection
record is required when a cell has more than one completed attempt. Its schema is
revision_attempt_selection_v1, and it must contain manifest_sha256, approved=true,
a nonempty rationale, and selections mapping run IDs to job directory names.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np

import capacity_task as legacy_task
import make_revision_manifest as study
import revision_task as strict_task
import rewards
import run_revision_cell as runner

HERE = Path(__file__).resolve().parent
STRICT_PROTOCOL = "strict_capacity_training_v1"
STRICT_SOURCES = ("train_revision.py", "revision_task.py", "revision_ppo.py",
                  "train_speed_grpo.py", "rewards.py", "tasks.py", "capacity_task.py", "speed_task.py")
FAILURE_MARKERS = {"FAILED", "FAILURE", "ERROR", "CANCELLED", "ABORTED", "failed.json"}
ENDPOINT_METRICS = ("parse_fail_rate", "empty_selection_rate", "nonempty_rate",
                    "useful_feasible_rate", "useful_value_mean", "useful_failure_rate",
                    "recorded_verdict_failure_rate", "physical_capacity_violation_rate_over_all_completions",
                    "completion_tokens_mean")
PRIMARY_METRICS = ("useful_feasible_rate", "useful_value_mean")


class ValidationError(ValueError):
    """Reject evidence that does not meet the frozen study's verification rules."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"A JSON object repeats the field {key}.")
        result[key] = value
    return result


def _parse_json(text):
    def reject_constant(value):
        raise ValidationError(f"A JSON record contains the nonfinite constant {value}.")
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant)


def read_json(path):
    return _parse_json(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as stream:
        return [_parse_json(line) for line in stream]


def equal(actual, expected, label, *, tolerance=1e-9):
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected, f"{label} disagrees with recomputation.")
    elif isinstance(expected, (int, float, np.number)):
        require(isinstance(actual, (int, float, np.number)) and not isinstance(actual, bool)
                and math.isfinite(float(actual))
                and math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance),
                f"{label} disagrees with recomputation.")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"{label} has the wrong length.")
        for index, (left, right) in enumerate(zip(actual, expected)):
            equal(left, right, f"{label}[{index}]", tolerance=tolerance)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict), f"{label} must be an object.")
        for key, value in expected.items():
            require(key in actual, f"{label} is missing {key}.")
            equal(actual[key], value, f"{label}.{key}", tolerance=tolerance)
    else:
        raise TypeError(f"The checker cannot compare {type(expected)}.")


def _finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "A recorded number is not finite.")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _terminal(path):
    require(path.is_dir(), f"The run directory is missing at {path}.")
    failures = [name for name in FAILURE_MARKERS if (path / name).exists()]
    require(not failures, f"The run has failure markers {failures}.")
    require((path / "DONE").is_file(), "The run has no terminal DONE marker.")


def load_study(manifest_path, dataset_path):
    dataset = strict_task.load_manifest(dataset_path)
    manifest = read_json(manifest_path)
    study.validate_manifest(manifest, dataset)
    return manifest, dataset


def validate_launch(path, cell, manifest_path, dataset_path, *, validation_steps=None):
    """Bind a scheduled study attempt to its command and complete source snapshot."""
    require(re.fullmatch(r"job_[A-Za-z0-9_]+", path.name) is not None, "The job directory name is invalid.")
    job_id = path.name[4:]
    launch_path = path.parent / f"launch_{job_id}.json"
    require(launch_path.is_file(), "The scheduled attempt has no launch record.")
    launch = read_json(launch_path)
    phase = "validation" if validation_steps is not None else "study"
    equal(launch, {"run_id": cell["run_id"], "row": cell["row"], "phase": phase,
                   "slurm_job_id": job_id, "exit_code": 0, "done_marker_exists": True,
                   "cell": cell}, "launch")
    require(launch["cell"] == cell, "The launch cell has unexpected fields or settings.")
    require(launch.get("ended_unix", 0) >= launch.get("started_unix", math.inf), "The launch has no valid terminal time.")
    equal(launch["manifest_file_sha256"], file_hash(manifest_path), "launch.manifest_file_sha256")
    equal(launch["dataset_file_sha256"], file_hash(dataset_path), "launch.dataset_file_sha256")
    snapshot = launch["snapshot"]
    entries = snapshot["files"]
    files = {}
    for entry in entries:
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "The source snapshot has an unsafe path.")
        require(entry["path"] not in files, "The source snapshot repeats a file.")
        require(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None, "A source snapshot hash is malformed.")
        files[entry["path"]] = entry["sha256"]
    require(runner.REQUIRED_SOURCES <= files.keys(), "The launch source inventory is incomplete.")
    equal(snapshot["source_sha256"], digest(entries), "snapshot.source_sha256")
    snapshot_bytes = (json.dumps(snapshot, sort_keys=True, indent=2) + "\n").encode()
    equal(launch["snapshot_file_sha256"], hashlib.sha256(snapshot_bytes).hexdigest(), "launch.snapshot_file_sha256")
    equal(files.get("configs/revision_manifest.json"), launch["manifest_file_sha256"], "snapshot.manifest")
    equal(files.get("configs/revision_dataset.json"), launch["dataset_file_sha256"], "snapshot.dataset")
    command = launch["command"]
    require(isinstance(command, list) and len(command) > 2 and all(isinstance(x, str) for x in command),
            "The launch command is malformed.")
    cfg = read_json(path / "config.json")
    output_key = "out" if cell["protocol"] == "strict" else "out_dir"
    remote_out = Path(cfg[output_key])
    require(remote_out.parts[-3:] == (phase, cell["run_id"], path.name),
            "The launch output is outside the expected study cell and attempt.")
    remote_dataset = Path(cfg.get("dataset", "unused"))
    expected = runner.make_command(cell, Path(command[1]).parent, remote_dataset, remote_out, validation_steps)
    require(command[1:] == expected[1:], "The launch command differs from the frozen final-study configuration.")
    require(Path(command[0]).name.startswith("python"), "The launch did not invoke a Python interpreter.")
    return {"record": launch, "file_sha256": file_hash(launch_path), "source_files": files}


def strict_summary(records):
    """Recompute every reported denominator without importing the training summarizer."""
    require(bool(records), "An evaluation or rollout has no completions.")
    count = len(records)
    valid = [row for row in records if row["format_valid"]]
    nonempty = [row for row in records if row["nonempty"]]
    useful = [row for row in records if row["useful_feasible"]]
    mean = lambda field, rows: statistics.fmean(row[field] for row in rows) if rows else None
    violating = sum(not row["capacity_feasible"] for row in valid)
    return {"n_completions": count, "n_format_valid": len(valid), "n_nonempty": len(nonempty),
            "n_useful_feasible": len(useful), "format_valid_rate": len(valid) / count,
            "parse_fail_rate": 1 - len(valid) / count, "nonempty_rate": len(nonempty) / count,
            "empty_selection_rate": sum(row["format_valid"] and not row["nonempty"] for row in records) / count,
            "useful_feasible_rate": len(useful) / count, "useful_failure_rate": 1 - len(useful) / count,
            "capacity_violation_rate_given_format_valid": violating / len(valid) if valid else None,
            "n_capacity_violating": violating, "completion_tokens_mean": mean("n_completion_tokens", records),
            "useful_value_mean": mean("useful_value", records),
            "normalized_value_mean_given_format_valid": mean("normalized_value", valid),
            "useful_value_mean_given_useful_feasible": mean("useful_value", useful),
            "capacity_residual_mean_given_format_valid": mean("capacity_residual", valid),
            "reward_mean": mean("reward", records), "scaled_reward_mean": mean("scaled_reward", records)}


def _strict_schedule(cfg):
    keys = [("eval", "train", 0)]
    for step in range(1, cfg["steps"] + 1):
        keys.append(("train", "train", step))
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            keys.append(("eval", "train", step))
    if not cfg.get("skip_heldout", False):
        keys.append(("eval", "heldout", cfg["steps"]))
    return keys


def _validate_generation(provenance, cfg):
    generation = provenance["effective_generation_config"]
    equal(generation, {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                       "repetition_penalty": 1.0, "num_beams": 1,
                       "max_new_tokens": cfg["max_new_tokens"]}, "generation")
    identity_defaults = {"typical_p": (None, 1.0), "min_p": (None, 0.0), "epsilon_cutoff": (None, 0.0),
                         "eta_cutoff": (None, 0.0), "no_repeat_ngram_size": (None, 0),
                         "min_length": (None, 0), "min_new_tokens": (None, 0)}
    for key, allowed in identity_defaults.items():
        require(generation.get(key) in allowed, f"Generation uses an undeclared {key} transform.")
    for key in ("bad_words_ids", "force_words_ids", "forced_bos_token_id", "forced_eos_token_id",
                "suppress_tokens", "begin_suppress_tokens", "sequence_bias", "watermarking_config",
                "guidance_scale", "constraints", "stop_strings", "penalty_alpha", "top_h"):
        require(generation.get(key) is None, f"Generation uses an undeclared {key} transform.")
    stop = generation.get("eos_token_id")
    stop = [stop] if isinstance(stop, int) else (stop or [])
    equal(provenance["completion_mask_stop_ids"], sorted(set(stop)), "completion_mask_stop_ids")


def _validate_actor_epochs(metric, cfg, records):
    epochs = metric["actor_epochs"]
    require([row["epoch"] for row in epochs] == list(range(1, cfg["epochs"] + 1)), "The actor epoch budget is incomplete or duplicated.")
    for row in epochs:
        require({"epoch", "policy_loss", "clip_fraction", "active_clip_fraction", "approximate_behavior_kl",
                 "ratio_min", "ratio_max", "grad_norm_preclip"} <= set(row), "An actor epoch is missing required diagnostics.")
        _finite_tree(row)
        require(0 <= row["active_clip_fraction"] <= row["clip_fraction"] + 1e-6 <= 1 + 1e-6,
                "The actor clipping fractions are inconsistent.")
        # A finite float32 exponential can underflow to zero. The trainer accepts
        # that finite ratio, so the replay must not relabel it as a failed outcome.
        require(0 <= row["ratio_min"] <= row["ratio_max"] and row["grad_norm_preclip"] >= 0,
                "The actor ratios or gradient norm are invalid.")
        require(row["approximate_behavior_kl"] >= -1e-6, "The actor behavior-divergence estimate is invalid.")
    equal(epochs[0]["ratio_min"], 1.0, "first_epoch.ratio_min", tolerance=1e-4)
    equal(epochs[0]["ratio_max"], 1.0, "first_epoch.ratio_max", tolerance=1e-4)
    equal(epochs[0]["clip_fraction"], 0.0, "first_epoch.clip_fraction", tolerance=1e-6)
    equal(epochs[0]["active_clip_fraction"], 0.0, "first_epoch.active_clip_fraction", tolerance=1e-6)
    equal(epochs[0]["approximate_behavior_kl"], 0.0, "first_epoch.approximate_behavior_kl", tolerance=1e-6)
    equal(epochs[0]["policy_loss"], -statistics.fmean(row["advantage"] for row in records),
          "first_epoch.policy_loss", tolerance=2e-5)


def validate_strict_run(path, cell=None, dataset=None, *, smoke=False, launch=None):
    """Verify strict artifacts, then replay raw scores, advantages, and every aggregate."""
    path = Path(path)
    _terminal(path)
    cfg, provenance, done = (read_json(path / name) for name in ("config.json", "provenance.json", "DONE"))
    frozen_dataset = strict_task.load_manifest(path / "dataset.json")
    if dataset is None:
        require(smoke, "Study validation requires the external frozen dataset.")
        dataset = frozen_dataset
    require(frozen_dataset == dataset, "The run's dataset differs from the frozen external dataset.")
    require(smoke or cell is not None, "Study validation requires a frozen manifest cell.")
    if cell is not None:
        expected_config = dict(cell["config"])
        if smoke:
            require(0 < cfg["steps"] < cell["config"]["steps"], "A timing smoke must be shorter than its retained study cell.")
            expected_config.update(steps=cfg["steps"], eval_every=cfg["steps"], skip_heldout=True)
        equal(cfg, expected_config, "config")
    if not smoke:
        require(not cfg["tiny"] and not cfg["skip_heldout"], "A tiny model or timing smoke cannot enter the study.")
        require(cfg["training_completions"] == 7200 and len(dataset["train"]) == 6 and len(dataset["heldout"]) == 24,
                "The strict study has an unmatched instance or sample budget.")
        require(launch is not None, "A final strict study run requires verified launch evidence.")
        require(str(provenance["device"]).startswith("cuda") and "a100" in provenance["device_name"].lower(),
                "A final strict study run must record the matched A100 device.")
        equal(cfg.get("evaluation_generation_batch_size"), 4, "evaluation_generation_batch_size")
    equal(cfg, {"protocol_version": STRICT_PROTOCOL, "dataset_sha256": dataset["sha256"],
                "kl_beta": 0.0, "no_ref": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                "repetition_penalty": 1.0, "optimizer_steps": cfg["steps"] * cfg["epochs"],
                "training_completions": cfg["steps"] * cfg["group_size"] * len(dataset["train"]),
                "critic_type": "learned_prompt_value" if cfg["method"] == "ppo" else None}, "config")
    equal(provenance, {"protocol_version": STRICT_PROTOCOL, "dataset_sha256": dataset["sha256"],
                       "steps_completed": cfg["steps"],
                       "config_sha256": digest({k: v for k, v in cfg.items() if k not in ("out", "dataset")})}, "provenance")
    require(bool(provenance.get("end_time")), "The strict provenance is not terminal.")
    equal(done, {"status": "complete", "steps_completed": cfg["steps"],
                 "provenance_sha256": file_hash(path / "provenance.json")}, "DONE")
    expected_artifacts = {"config.json", "dataset.json", "prompt_tokens.json", "metrics.jsonl", "completions.jsonl"}
    if cfg["method"] == "ppo":
        expected_artifacts.add("critic.json")
    require(set(provenance["artifact_sha256"]) == expected_artifacts, "The strict artifact inventory is incomplete or unexpected.")
    for name, expected in provenance["artifact_sha256"].items():
        require((path / name).is_file(), f"The strict run is missing {name}.")
        equal(file_hash(path / name), expected, f"artifact.{name}")
    require(set(provenance["source_files"]) == set(STRICT_SOURCES), "The strict training source inventory differs.")
    equal(provenance["source_sha256"], digest(provenance["source_files"]), "source_sha256")
    for name in ("revision_task.py", "rewards.py", "capacity_task.py", "tasks.py"):
        equal(provenance["source_files"][name], file_hash(HERE / name), f"replay_source.{name}")
    if launch is not None:
        equal(launch["record"]["phase"], "validation" if smoke else "study", "launch.phase")
        for name, expected in provenance["source_files"].items():
            equal(launch["source_files"].get(name), expected, f"launch_source.{name}")
        equal(str(provenance["slurm_job_id"]), launch["record"]["slurm_job_id"], "slurm_job_id")
    _validate_generation(provenance, cfg)
    prompt_tokens = read_json(path / "prompt_tokens.json")
    for split in ("train", "heldout"):
        require([row["instance_id"] for row in prompt_tokens[split]] == [row["instance_id"] for row in dataset[split]],
                "The prompt token record does not cover the frozen instance order.")
        require(all(row["prompt_token_ids"] and all(type(token) is int and token >= 0 for token in row["prompt_token_ids"])
                    for row in prompt_tokens[split]), "The prompt tokens are malformed.")
    metrics, records = read_jsonl(path / "metrics.jsonl"), read_jsonl(path / "completions.jsonl")
    schedule = _strict_schedule(cfg)
    require(all(type(row.get("step")) is int for row in metrics), "Every metric step must be an integer.")
    require([(row["phase"], row["split"], row["step"]) for row in metrics] == schedule,
            "The strict metrics have missing, duplicate, or unscheduled evaluations or updates.")
    instances = {row["instance_id"]: row for split in ("train", "heldout") for row in dataset[split]}
    grouped, seen, completion_order = defaultdict(list), set(), []
    for row in records:
        _finite_tree(row)
        require(type(row.get("step")) is int and type(row.get("sample_index")) is int,
                "Every raw completion step and sample index must be an integer.")
        key = (row["phase"], row["split"], row["step"])
        identity = (*key, row["instance_id"], row["sample_index"])
        require(identity not in seen, "The raw log contains a duplicate completion observation.")
        seen.add(identity)
        require(row["instance_id"] in instances, "The raw log names an unknown instance.")
        instance = instances[row["instance_id"]]
        require(row["split"] == instance["split"], "A completion is assigned to the wrong dataset split.")
        require(isinstance(row["raw_text"], str), "A raw completion must be text.")
        equal(row, strict_task.score_completion(row["raw_text"], instance, cfg["reward"], cfg["weight"]), "raw_score")
        equal(row["scaled_reward"], row["reward"] / cfg["reward_scale"], "scaled_reward")
        tokens = row["completion_token_ids"]
        require(type(row["n_completion_tokens"]) is int and 1 <= row["n_completion_tokens"] <= cfg["max_new_tokens"],
                "The completion token count is outside its declared budget.")
        require(isinstance(tokens, list) and len(tokens) == row["n_completion_tokens"]
                and all(type(token) is int and token >= 0 for token in tokens), "The completion token record is malformed.")
        require(not any(token in provenance["completion_mask_stop_ids"] for token in tokens[:-1]),
                "A completion contains tokens after its first declared stop token.")
        grouped[key].append(row)
        completion_order.append(identity)
    expected_order = []
    for phase, split, step in schedule:
        count = cfg["group_size"] if phase == "train" else cfg["eval_samples"]
        expected_order.extend((phase, split, step, instance["instance_id"], sample)
                              for instance in dataset[split] for sample in range(count))
    require(completion_order == expected_order, "The raw completion budget, instance order, or evaluation schedule differs.")
    critic_values = np.zeros(len(dataset["train"]), dtype=np.float64)
    zero_groups, total_groups = 0, 0
    for metric, key in zip(metrics, schedule):
        event = grouped[key]
        equal(metric, strict_summary(event), "summary")
        by_instance = [[row for row in event if row["instance_id"] == instance["instance_id"]] for instance in dataset[key[1]]]
        if key[0] == "eval":
            expected = [{"instance_id": instance["instance_id"], **strict_summary(group)}
                        for instance, group in zip(dataset[key[1]], by_instance)]
            equal(metric["per_instance"], expected, "per_instance")
            continue
        for index, group in enumerate(by_instance):
            scaled = np.asarray([row["scaled_reward"] for row in group], dtype=np.float64)
            for row in group:
                require(row["behavior_log_prob"] <= 1e-5, "A behavior log probability is positive.")
                equal(row["old_value"], critic_values[index], "old_value", tolerance=2e-5)
            if cfg["method"] == "grpo":
                expected_advantage = rewards.advantage(scaled).astype(np.float32).astype(float)
            else:
                expected_advantage = (scaled.astype(np.float32) - np.float32(critic_values[index])).astype(float)
            equal([row["advantage"] for row in group], expected_advantage.tolist(), "advantage", tolerance=2e-5)
            zero_groups += all(row["advantage"] == 0 for row in group)
            total_groups += 1
        equal(metric["all_violating_fraction"], statistics.fmean(not any(row["useful_feasible"] for row in group)
                                                                for group in by_instance), "all_violating_fraction")
        equal(metric["zero_advantage_group_fraction"], statistics.fmean(all(row["advantage"] == 0 for row in group)
                                                                       for group in by_instance), "zero_advantage_group_fraction")
        equal(metric["group_reward_std_mean"], statistics.fmean(float(np.std([row["scaled_reward"] for row in group], ddof=1))
                                                                for group in by_instance), "group_reward_std_mean")
        _validate_actor_epochs(metric, cfg, event)
        if cfg["method"] == "ppo":
            equal(metric["critic_fit_steps"], cfg["critic_steps"], "critic_fit_steps")
            targets = np.asarray([[row["scaled_reward"] for row in group] for group in by_instance], dtype=np.float32).astype(float)
            equal(metric["critic_mse_before"], float(np.mean((critic_values[:, None] - targets) ** 2)), "critic_mse_before", tolerance=2e-5)
            for _ in range(cfg["critic_steps"]):
                critic_values -= cfg["critic_lr"] * 2 / len(by_instance) * (critic_values - targets.mean(axis=1))
            equal(metric["critic_values"], critic_values.tolist(), "critic_values", tolerance=2e-5)
            equal(metric["critic_mse_after"], float(np.mean((critic_values[:, None] - targets) ** 2)), "critic_mse_after", tolerance=2e-5)
            critic_values = np.asarray(metric["critic_values"], dtype=float)
        else:
            equal(metric["critic_fit_steps"], 0, "critic_fit_steps")
    if cfg["method"] == "ppo":
        final_values = read_json(path / "critic.json")["state_dict"]["values.weight"]
        equal(final_values, [[value] for value in critic_values], "final_critic", tolerance=2e-5)
    finals = {split: grouped[("eval", split, cfg["steps"])] for split in ("train", "heldout")
              if ("eval", split, cfg["steps"]) in grouped}
    return {"config": cfg, "provenance": provenance, "finals": finals,
            "training_completions": sum(len(grouped[key]) for key in schedule if key[0] == "train"),
            "zero_advantage_groups": zero_groups, "training_groups": total_groups,
            "artifact_sha256": {name: file_hash(path / name) for name in expected_artifacts | {"provenance.json", "DONE"}},
            "smoke_only": smoke, "declared_study_cell": cell}


def _legacy_selection(value, instance):
    if value is None:
        return None
    require(isinstance(value, list) and value and all(type(index) is int for index in value),
            "A historical parsed selection is malformed.")
    require(value[0] == instance and value[1:] == sorted(set(value[1:]))
            and all(0 <= index < 16 for index in value[1:]),
            "A historical selection has invalid indices or the wrong prompt identity.")
    return tuple(value)


def legacy_quality(selections, ratio):
    """Use the same parsed-selection definitions as the historical standalone audit."""
    count = len(selections)
    require(count > 0, "A historical evaluation has no completions.")
    present = [row for row in selections if row is not None]
    nonempty = [row for row in present if len(row) > 1]
    useful = [row for row in nonempty if legacy_task.verdict(row, ratio)]
    violating = sum(not legacy_task.verdict(row, ratio) for row in selections)
    return {"n_completions": count, "n_format_valid": len(present), "n_nonempty": len(nonempty),
            "n_useful_feasible": len(useful), "n_capacity_violating": sum(not legacy_task.verdict(row, ratio) for row in present),
            "parse_fail_rate": (count - len(present)) / count,
            "empty_selection_rate": (len(present) - len(nonempty)) / count,
            "nonempty_rate": len(nonempty) / count, "useful_feasible_rate": len(useful) / count,
            "useful_failure_rate": 1 - len(useful) / count,
            "useful_value_mean": sum(legacy_task.task_term(row, ratio) for row in useful) / count,
            "recorded_verdict_failure_rate": violating / count,
            "physical_capacity_violation_rate_over_all_completions": sum(not legacy_task.verdict(row, ratio) for row in present) / count,
            "completion_tokens_mean": None}


# This tuple is the training-source list of the frozen trainer at COMMIT_4, which ran the
# historical-protocol rows. That trainer computed its digest over these fifteen names only. The checkout's list later gained
# the yoked-control files at its end, so reading the checkout's list alone would add three missing
# entries and change the recomputed digest of every historical-protocol run.
LEGACY_TRAINING_SOURCES = (
    "experiments/cancellation/train_speed_grpo.py",
    "experiments/cancellation/rewards.py",
    "experiments/cancellation/tasks.py",
    "experiments/cancellation/speed_task.py",
    "experiments/cancellation/capacity_task.py",
    "experiments/cancellation/make_capacity_instances.py",
    "experiments/cancellation/configs/base.yaml",
    "experiments/cancellation/configs/arms.yaml",
    "experiments/cancellation/configs/regimes.yaml",
    "experiments/cancellation/slurm/manifest.csv",
    "experiments/cancellation/slurm/manifest_capacity.csv",
    "experiments/cancellation/slurm/manifest_speed_remedy.csv",
    "experiments/cancellation/slurm/manifest_capacity_remedy.csv",
    "experiments/cancellation/slurm/manifest_capacity_kl.csv",
    "experiments/cancellation/slurm/manifest_fingerprint_speed.csv",
)


def _legacy_training_sources():
    """Return the frozen trainer's list, after checking that the checkout's list still starts with it."""
    tree = ast.parse((HERE / "train_speed_grpo.py").read_text())
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "TRAINING_SOURCES"
                                                    for target in statement.targets):
            current = tuple(ast.literal_eval(statement.value))
            require(current[:len(LEGACY_TRAINING_SOURCES)] == LEGACY_TRAINING_SOURCES,
                    "The checkout's training source list no longer starts with the historical list.")
            return LEGACY_TRAINING_SOURCES
    raise ValidationError("The historical training source list is unavailable.")


def _legacy_source_digest(snapshot_files):
    hasher = hashlib.sha256()
    prefix = "experiments/cancellation/"
    for name in _legacy_training_sources():
        hasher.update(name.encode() + b"\0")
        hasher.update(snapshot_files.get(name[len(prefix):], "MISSING").encode() + b"\0")
    return hasher.hexdigest()


def validate_legacy_run(path, cell, *, reused=False, launch=None, smoke=False):
    """Replay historical rewards while retaining the limits of its permissive parser."""
    path = Path(path)
    _terminal(path)
    require((path / "DONE").read_text().strip() == "ok", "The historical terminal marker is invalid.")
    cfg, provenance = read_json(path / "config.json"), read_json(path / "provenance.json")
    cfg = {**study.LEGACY_DEFAULTS, **cfg}
    expected_config = dict(cell["config"])
    if smoke:
        require(not reused and 0 < cfg["steps"] < cell["config"]["steps"],
                "A timing smoke must be a new run shorter than its retained study cell.")
        expected_config.update(steps=cfg["steps"], eval_every=cfg["steps"])
    equal(cfg, expected_config, "legacy.config")
    actual_cfg = read_json(path / "config.json")
    cfg_hash = hashlib.sha256(json.dumps({key: value for key, value in actual_cfg.items() if key != "out_dir"},
                                       sort_keys=True).encode()).hexdigest()[:16]
    equal(provenance["config_hash"], cfg_hash, "legacy.config_hash")
    equal(provenance["steps_completed"], cfg["steps"], "legacy.steps_completed")
    require(bool(provenance.get("end_time")), "The historical provenance is not terminal.")
    if reused:
        for name, expected in cell["artifact_sha256"].items():
            equal(file_hash(path / name), expected, f"reused.{name}")
    else:
        require(launch is not None, "A new historical-protocol run requires verified launch evidence.")
        equal(launch["record"]["phase"], "validation" if smoke else "study", "legacy.launch.phase")
        require(str(provenance["device"]).startswith("cuda") and "a100" in provenance["device_name"].lower(),
                "A new historical-protocol run must record the matched A100 device.")
        equal(str(provenance["slurm_job_id"]), launch["record"]["slurm_job_id"], "legacy.slurm_job_id")
        equal(provenance["training_sources_digest"], _legacy_source_digest(launch["source_files"]), "legacy.training_sources_digest")
        for name in ("rewards.py", "capacity_task.py", "tasks.py"):
            equal(launch["source_files"].get(name), file_hash(HERE / name), f"legacy.replay_source.{name}")
    records = read_jsonl(path / "metrics.jsonl")
    expected = [("eval", 0)]
    for step in range(1, cfg["steps"] + 1):
        expected.append(("train", step))
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            expected.append(("eval", step))
    require([(row["phase"], row["step"]) for row in records] == expected,
            "The historical metrics contain missing, duplicate, or unscheduled observations.")
    total, total_groups, zero_groups, kept_groups = 0, 0, 0, 0
    final = None
    for rec in records:
        _finite_tree(rec)
        if rec["phase"] == "eval":
            raw = rec["eval_selections"]
            require(len(raw) == cfg["batch_prompts"] * cfg["eval_samples"] == rec["eval_n"],
                    "A historical evaluation has an unmatched completion budget.")
            selections = [_legacy_selection(value, index // cfg["eval_samples"]) for index, value in enumerate(raw)]
            quality = legacy_quality(selections, cfg["vmax"])
            present = [value for value in selections if value is not None]
            equal(rec["eval_violation_rate"], quality["recorded_verdict_failure_rate"], "legacy.eval_violation_rate")
            equal(rec["eval_parse_fail_rate"], quality["parse_fail_rate"], "legacy.eval_parse_fail_rate")
            equal(rec["eval_mean_n_selected"], statistics.fmean(len(value) - 1 for value in present) if present else None,
                  "legacy.eval_mean_n_selected")
            equal(rec["eval_mean_residual"], statistics.fmean(legacy_task.residual(value, cfg["vmax"]) for value in present) if present else None,
                  "legacy.eval_mean_residual")
            final = selections
            continue
        groups = rec["groups"]
        require(len(groups) == cfg["batch_prompts"], "A historical step has an unmatched prompt budget.")
        flat, all_rewards, all_advantages, verdicts_by_group, kept = [], [], [], [], []
        for index, group in enumerate(groups):
            selections = [_legacy_selection(value, index) for value in group["selections"]]
            require(len(selections) == cfg["group_size"], "A historical group has an unmatched sample budget.")
            computed_rewards = rewards.group_rewards(selections, cfg["arm"], cfg["w"], cfg["vmax"],
                                                     spec=legacy_task.SPEC, w_task=cfg["w_task"])
            anchor = rewards.reward_anchor(cfg["adv_variant"], cfg["arm"], cfg["w"], cfg["vmax"],
                                           spec=legacy_task.SPEC, w_task=cfg["w_task"], instance=index)
            advantages = rewards.advantage_variant(computed_rewards, cfg["adv_variant"], anchor)
            verdicts = rewards.group_verdicts(selections, cfg["vmax"], spec=legacy_task.SPEC)
            keep = rewards.keep_group(cfg["group_filter"], computed_rewards, verdicts)
            equal(group["rewards"], computed_rewards.tolist(), "legacy.rewards")
            equal(group["adv"], advantages.tolist(), "legacy.advantages")
            equal(group["all_violating"], not any(verdicts), "legacy.all_violating")
            equal(group["has_violator"], not all(verdicts), "legacy.has_violator")
            if "kept" in group or not reused:
                equal(group.get("kept"), keep, "legacy.kept")
                equal(group.get("anchor"), anchor, "legacy.anchor")
                equal(group.get("prompt_index"), index, "legacy.prompt_index")
            flat.extend(selections)
            all_rewards.append(computed_rewards)
            all_advantages.extend(advantages)
            verdicts_by_group.append(verdicts)
            kept.append(keep)
            total += len(selections)
            total_groups += 1
            zero_groups += bool(np.count_nonzero(advantages) == 0)
            kept_groups += keep
        present = [value for value in flat if value is not None]
        failures = np.asarray([not legacy_task.verdict(value, cfg["vmax"]) for value in flat])
        advantages = np.asarray(all_advantages)
        expected = {"violation_rate": float(failures.mean()),
                    "parse_fail_rate": (len(flat) - len(present)) / len(flat),
                    "all_violating_fraction": statistics.fmean(not any(values) for values in verdicts_by_group),
                    "mean_n_selected": statistics.fmean(len(value) - 1 for value in present) if present else None,
                    "mean_residual": statistics.fmean(legacy_task.residual(value, cfg["vmax"]) for value in present) if present else None,
                    "reward_mean": float(np.mean(all_rewards)),
                    "group_reward_std_mean": statistics.fmean(float(np.std(value, ddof=1)) for value in all_rewards),
                    "violator_mean_adv": float(advantages[failures].mean()) if failures.any() else None,
                    "frac_violators_pos_adv": float((advantages[failures] > 0).mean()) if failures.any() else None}
        equal(rec, expected, "legacy.train_summary")
        if "n_kept_groups" in rec or not reused:
            equal(rec, {"n_kept_groups": sum(kept), "opt_step_taken": any(kept),
                        "n_mixed_groups": sum(any(values) and not all(values) for values in verdicts_by_group),
                        "n_all_feasible_groups": sum(all(values) for values in verdicts_by_group)}, "legacy.filter_summary")
    expected_samples = cfg["steps"] * cfg["group_size"] * cfg["batch_prompts"] if smoke else cell["training_completions"]
    require(total == expected_samples and (smoke or expected_samples == 7200), "The historical training sample budget is incomplete.")
    return {"config": cfg, "provenance": provenance, "finals": {"train": final},
            "training_completions": total, "zero_advantage_groups": zero_groups,
            "training_groups": total_groups, "kept_groups": kept_groups,
            "artifact_sha256": {name: file_hash(path / name) for name in study.ARTIFACTS},
            "smoke_only": smoke, "declared_study_cell": cell}


def attempt_status(path):
    job = path.name[4:]
    launch_path = path.parent / f"launch_{job}.json"
    result = {"attempt": path.name, "state": "incomplete", "launch_present": launch_path.is_file(),
              "done_present": (path / "DONE").is_file()}
    if any((path / name).exists() for name in FAILURE_MARKERS):
        result["state"] = "failed"
        return result
    if not launch_path.is_file():
        result["state"] = "invalid" if result["done_present"] else "incomplete"
        return result
    try:
        launch = read_json(launch_path)
        if launch.get("phase") != "study" or launch.get("slurm_job_id") != job:
            result["state"] = "invalid"
        elif launch.get("exit_code") is not None and launch["exit_code"] != 0:
            result["state"] = "failed"
        elif result["done_present"] and launch.get("exit_code") == 0 and launch.get("done_marker_exists") is True:
            result["state"] = "completed_unvalidated"
    except (ValueError, OSError, TypeError) as error:
        result["state"] = "invalid"
        result["reason"] = str(error)
    return result


def study_status(manifest, runs_root):
    """Report inventory only, and never equate a terminal marker with accepted evidence."""
    base = Path(runs_root) / "study"
    expected = {row["run_id"] for row in manifest["new_runs"]}
    unexpected = sorted(path.name for path in base.iterdir() if path.is_dir() and path.name not in expected) if base.exists() else []
    cells = []
    for record in manifest["new_runs"]:
        directory = base / record["run_id"]
        names = set()
        if directory.exists():
            names.update(path.name for path in directory.glob("job_*") if path.is_dir())
            names.update("job_" + path.stem[len("launch_"):] for path in directory.glob("launch_*.json"))
        attempts = [attempt_status(directory / name) for name in sorted(names)]
        completed = sum(row["state"] == "completed_unvalidated" for row in attempts)
        state = ("missing" if not attempts else "ambiguous" if completed > 1 else
                 "completed_unvalidated" if completed == 1 else "failed" if all(row["state"] == "failed" for row in attempts)
                 else "incomplete_or_invalid")
        cells.append({"run_id": record["run_id"], "row": record["row"], "state": state, "attempts": attempts})
    counts = {state: sum(row["state"] == state for row in cells)
              for state in ("missing", "ambiguous", "completed_unvalidated", "failed", "incomplete_or_invalid")}
    return {"status": "inventory_only", "study_results_validated": False,
            "expected_new_cells": 68, "expected_reused_cells": 17, "counts": counts,
            "unexpected_study_directories": unexpected, "cells": cells,
            "note": "Terminal markers describe execution coverage. Only a successful complete-matrix analysis accepts study evidence."}


def load_attempt_selection(path, manifest):
    if path is None:
        return None
    selection = read_json(path)
    require(selection.get("schema_version") == "revision_attempt_selection_v1", "The attempt selection schema is unsupported.")
    require(selection.get("manifest_sha256") == manifest["sha256"] and selection.get("approved") is True,
            "The attempt selection must explicitly approve this frozen manifest.")
    require(isinstance(selection.get("rationale"), str) and selection["rationale"].strip(), "The attempt selection needs its documented rationale.")
    known = {row["run_id"] for row in manifest["new_runs"]}
    require(isinstance(selection.get("selections"), dict) and set(selection["selections"]) <= known,
            "The attempt selection names an unknown study cell.")
    require(all(isinstance(value, str) and re.fullmatch(r"job_[A-Za-z0-9_]+", value)
                for value in selection["selections"].values()), "An approved attempt name is unsafe.")
    return selection


def select_attempts(status, runs_root, selection=None):
    require(not status["unexpected_study_directories"], "The study directory contains runs outside the frozen matrix.")
    chosen, problems = {}, []
    approved = selection["selections"] if selection else {}
    for cell in status["cells"]:
        completed = [row["attempt"] for row in cell["attempts"] if row["state"] == "completed_unvalidated"]
        requested = approved.get(cell["run_id"])
        if requested is not None:
            if requested not in completed:
                problems.append(f"{cell['run_id']} has no completed approved attempt {requested}")
            else:
                chosen[cell["run_id"]] = Path(runs_root) / "study" / cell["run_id"] / requested
        elif len(completed) == 1:
            chosen[cell["run_id"]] = Path(runs_root) / "study" / cell["run_id"] / completed[0]
        elif len(completed) > 1:
            problems.append(f"{cell['run_id']} has multiple completed attempts without an approved mapping")
        else:
            problems.append(f"{cell['run_id']} has no completed study attempt")
    require(not problems, "The complete matrix is unavailable: " + "; ".join(problems))
    return chosen


def condition_id(protocol, method, reward, weight, group_size):
    return f"{protocol}_{method}_{reward}_w{float(weight):g}_k{group_size}"


def result_rows(cell, verified, *, reused=False, attempt="historical"):
    require(not verified.get("smoke_only", False), "Timing or test-model evidence cannot enter final study endpoint tables.")
    cfg = verified["config"]
    protocol = cell["protocol"]
    reward, weight = (cfg["arm"], cfg["w"]) if protocol == "legacy" else (cfg["reward"], cfg["weight"])
    common = {"run_id": cell["run_id"], "condition_id": condition_id(protocol, cell["method"], reward, weight, cfg["group_size"]),
              "protocol": protocol, "method": cell["method"], "reward": reward, "weight": weight,
              "group_size": cfg["group_size"], "steps": cfg["steps"], "seed": cfg["seed"],
              "reused": reused, "attempt": attempt, "training_completions": verified["training_completions"],
              "training_groups": verified["training_groups"], "zero_advantage_groups": verified["zero_advantage_groups"],
              "zero_advantage_group_fraction": verified["zero_advantage_groups"] / verified["training_groups"],
              "format_evidence": "strict_raw_text" if protocol == "strict" else "historical_parsed_selection_only"}
    seeds, instances = [], []
    for split, records in verified["finals"].items():
        if protocol == "strict":
            summary = strict_summary(records)
            quality = {**summary, "recorded_verdict_failure_rate": summary["useful_failure_rate"],
                       "physical_capacity_violation_rate_over_all_completions": summary["n_capacity_violating"] / len(records)}
            identifiers = list(dict.fromkeys(row["instance_id"] for row in records))
            partitions = [(name, [row for row in records if row["instance_id"] == name]) for name in identifiers]
        else:
            quality = legacy_quality(records, cfg["vmax"])
            partitions = [(f"train_{index:03d}", records[index * cfg["eval_samples"]:(index + 1) * cfg["eval_samples"]])
                          for index in range(cfg["batch_prompts"])]
        counts = {"evaluation_completions": quality["n_completions"], "evaluation_instances": len(partitions),
                  "evaluation_samples_per_instance": cfg["eval_samples"], "n_parsed": quality["n_format_valid"],
                  "n_nonempty": quality["n_nonempty"], "n_useful_feasible": quality["n_useful_feasible"],
                  "n_capacity_violating_among_parsed": quality["n_capacity_violating"]}
        seeds.append({**common, "split": split, **counts, **{name: quality[name] for name in ENDPOINT_METRICS},
                      "all_outputs_useful_feasible": quality["n_useful_feasible"] == quality["n_completions"],
                      "all_outputs_pass_recorded_verdict": quality["recorded_verdict_failure_rate"] == 0})
        for instance, samples in partitions:
            if protocol == "strict":
                values = strict_summary(samples)
                values["recorded_verdict_failure_rate"] = values["useful_failure_rate"]
                values["physical_capacity_violation_rate_over_all_completions"] = values["n_capacity_violating"] / len(samples)
            else:
                values = legacy_quality(samples, cfg["vmax"])
            instances.append({"run_id": cell["run_id"], "condition_id": common["condition_id"],
                              "protocol": protocol, "seed": cfg["seed"], "split": split, "instance_id": instance,
                              "evaluation_completions": len(samples), "n_parsed": values["n_format_valid"],
                              "n_nonempty": values["n_nonempty"], "n_useful_feasible": values["n_useful_feasible"],
                              **{name: values[name] for name in ENDPOINT_METRICS}})
    return seeds, instances


def planned_contrasts():
    """Keep all comparisons within one protocol and declare their direction in advance."""
    comparisons = []
    legacy = lambda method, weight: condition_id("legacy", method, "binary", weight, 4)
    strict = lambda method, reward, weight, k: condition_id("strict", method, reward, weight, k)
    for method in study.METHODS:
        comparisons.append(("legacy_weight_25_minus_5", legacy(method, 25), legacy(method, 5)))
    for method in study.METHODS[1:]:
        for weight in (5, 25):
            comparisons.append(("legacy_remedy_minus_plain", legacy(method, weight), legacy("plain", weight)))
    for weight, k in ((5, 4), (25, 4), (25, 8)):
        comparisons.append(("strict_ppo_minus_grpo", strict("ppo", "binary", weight, k), strict("grpo", "binary", weight, k)))
    for method in ("grpo", "ppo"):
        comparisons.append(("strict_weight_25_minus_5", strict(method, "binary", 25, 4), strict(method, "binary", 5, 4)))
        comparisons.append(("strict_k8_minus_k4", strict(method, "binary", 25, 8), strict(method, "binary", 25, 4)))
    comparisons.append(("strict_graded_minus_binary", strict("grpo", "graded", 5, 4), strict("grpo", "binary", 5, 4)))
    return comparisons


def aggregate_seed_results(per_seed):
    groups = defaultdict(list)
    for row in per_seed:
        groups[(row["condition_id"], row["split"])].append(row)
    summary, paired = [], []
    for (condition, split), rows in sorted(groups.items()):
        require(sorted(row["seed"] for row in rows) == list(study.SEEDS), "Every condition requires exactly five distinct training seeds.")
        require(len({row["evaluation_completions"] for row in rows}) == 1, "Evaluation denominators differ across matched seeds.")
        for metric in ENDPOINT_METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            require(not values or len(values) == 5, "An endpoint has missing data for only some training seeds.")
            summary.append({"type": "condition", "comparison": "", "condition_id": condition,
                            "left_condition": "", "right_condition": "", "protocol": rows[0]["protocol"],
                            "split": split, "metric": metric, "n_seeds": len(rows), "n_observed_seeds": len(values),
                            "mean": statistics.fmean(values) if values else None,
                            "sample_sd": statistics.stdev(values) if values else None,
                            "evaluation_completions_per_seed": rows[0]["evaluation_completions"]})
    for name, left, right in planned_contrasts():
        splits = ("train", "heldout") if left.startswith("strict_") else ("train",)
        for split in splits:
            require((left, split) in groups and (right, split) in groups, "A planned paired comparison is missing a condition.")
            lhs = {row["seed"]: row for row in groups[(left, split)]}
            rhs = {row["seed"]: row for row in groups[(right, split)]}
            for metric in PRIMARY_METRICS:
                differences = []
                for seed in study.SEEDS:
                    difference = lhs[seed][metric] - rhs[seed][metric]
                    differences.append(difference)
                    paired.append({"comparison": name, "left_condition": left, "right_condition": right,
                                   "protocol": lhs[seed]["protocol"], "split": split, "metric": metric,
                                   "seed": seed, "left_value": lhs[seed][metric], "right_value": rhs[seed][metric],
                                   "difference": difference,
                                   "left_evaluation_completions": lhs[seed]["evaluation_completions"],
                                   "right_evaluation_completions": rhs[seed]["evaluation_completions"]})
                summary.append({"type": "paired_difference", "comparison": name, "condition_id": "",
                                "left_condition": left, "right_condition": right, "protocol": lhs[0]["protocol"],
                                "split": split, "metric": metric, "n_seeds": 5, "n_observed_seeds": 5,
                                "mean": statistics.fmean(differences), "sample_sd": statistics.stdev(differences),
                                "evaluation_completions_per_seed": lhs[0]["evaluation_completions"]})
    return summary, paired


def _write_csv(path, rows):
    require(bool(rows), f"The requested CSV {path.name} would contain no observations.")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shared_source_hashes(evidence):
    groups = {"strict": set(), "new_legacy": set()}
    for row in evidence:
        if row["protocol"] == "strict":
            groups["strict"].add(row["source_sha256"])
        elif not row["reused"]:
            groups["new_legacy"].add(row["source_sha256"])
    for name, hashes in groups.items():
        require(len(hashes) == 1 and None not in hashes, f"The {name} cells do not share one frozen training source digest.")
    return {name: next(iter(hashes)) for name, hashes in groups.items()}


def initial_policy_evidence(evidence):
    """Reject observed base-policy differences and describe missing historical evidence."""
    report = {}
    fields = ("initial_param_checksum", "model_revision", "tokenizer_revision")
    for protocol in ("legacy", "strict"):
        rows = [row for row in evidence if row["protocol"] == protocol]
        require(bool(rows), f"The {protocol} protocol has no initial-policy evidence records.")
        identities = [row["initial_policy"] for row in rows]
        require({identity["model_id"] for identity in identities} == {study.MODEL_ID},
                f"The {protocol} cells do not name the frozen base model.")
        available = {}
        for field in fields:
            values = {identity[field] for identity in identities if identity[field] is not None}
            require(len(values) <= 1, f"The {protocol} cells record an initial checkpoint or tokenizer mismatch in {field}.")
            if field == "initial_param_checksum":
                require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in values),
                        f"A {protocol} initial parameter checksum is malformed.")
            else:
                require(all(isinstance(value, str) and value.strip() for value in values),
                        f"A {protocol} checkpoint revision is malformed.")
            available[field] = {"recorded_runs": sum(identity[field] is not None for identity in identities),
                                "missing_runs": sum(identity[field] is None for identity in identities),
                                "distinct_values": sorted(values)}
        complete_hashes = available["initial_param_checksum"]["missing_runs"] == 0
        complete_revisions = all(available[field]["missing_runs"] == 0 for field in ("model_revision", "tokenizer_revision"))
        if protocol == "strict":
            require(complete_hashes or complete_revisions,
                    "Strict cells require matching initial parameter checksums or complete matching model and tokenizer revisions.")
            prompt_hashes = {row["artifact_sha256"]["prompt_tokens.json"] for row in rows}
            require(len(prompt_hashes) == 1, "The strict cells do not share identical encoded training and held-out prompts.")
        report[protocol] = {"runs": len(rows), "model_id": study.MODEL_ID, "identifiers": available,
                            "initial_weight_equality_supported": complete_hashes or complete_revisions,
                            "evidence_limit": ("The historical records have incomplete initial parameter checksums and resolved model/tokenizer revisions. "
                                "Matching model names, configurations, or training sources cannot cryptographically prove initial checkpoint equality."
                                if protocol == "legacy" and not (complete_hashes or complete_revisions) else
                                "The recorded identifiers agree across all available cells. Recorded parameter hashes describe the loaded initial model.")}
    return report


def analyze_study(manifest_path, dataset_path, runs_root, reused_root, out, selection_path=None):
    manifest, dataset = load_study(manifest_path, dataset_path)
    status = study_status(manifest, runs_root)
    selection = load_attempt_selection(selection_path, manifest)
    chosen = select_attempts(status, runs_root, selection)
    study.validate_manifest(manifest, dataset, runs_dir=reused_root)
    per_seed, per_instance, evidence = [], [], []
    for cell in manifest["new_runs"] + manifest["reused_runs"]:
        reused = "source_path" in cell
        path = study._source_dir(cell, reused_root) if reused else chosen[cell["run_id"]]
        launch = None if reused else validate_launch(path, cell, manifest_path, dataset_path)
        if cell["protocol"] == "strict":
            result = validate_strict_run(path, cell, dataset, launch=launch)
        else:
            result = validate_legacy_run(path, cell, reused=reused, launch=launch)
        seeds, instances = result_rows(cell, result, reused=reused, attempt="historical" if reused else path.name)
        per_seed.extend(seeds)
        per_instance.extend(instances)
        evidence.append({"run_id": cell["run_id"], "protocol": cell["protocol"], "reused": reused,
                         "attempt": "historical" if reused else path.name,
                         "artifact_sha256": result["artifact_sha256"],
                         "launch_sha256": None if reused else launch["file_sha256"],
                         "source_sha256": result["provenance"].get("source_sha256", result["provenance"].get("training_sources_digest")),
                         "initial_policy": {"model_id": result["config"]["model_id"],
                                            **{field: result["provenance"].get(field) for field in
                                               ("initial_param_checksum", "model_revision", "tokenizer_revision")}},
                         "training_completions": result["training_completions"]})
    source_hashes = shared_source_hashes(evidence)
    policy_evidence = initial_policy_evidence(evidence)
    require(len(per_seed) == 120 and len(per_instance) == 1350, "The complete study has an unexpected endpoint denominator.")
    summary, paired = aggregate_seed_results(per_seed)
    destination = Path(out)
    require(not destination.exists() or not any(destination.iterdir()), "The analysis destination must be new or empty to prevent stale table reuse.")
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "per_seed.csv", per_seed)
    _write_csv(destination / "per_instance.csv", per_instance)
    _write_csv(destination / "summary.csv", summary)
    _write_csv(destination / "paired_differences.csv", paired)
    sources = {name: file_hash(HERE / name) for name in
               ("analyze_revision.py", "make_revision_manifest.py", "revision_task.py", "rewards.py",
                "capacity_task.py", "tasks.py", "run_revision_cell.py")}
    report = {"status": "validated_complete_matrix", "study_results_validated": True,
              "manifest_sha256": manifest["sha256"], "dataset_sha256": dataset["sha256"],
              "manifest_file_sha256": file_hash(manifest_path), "dataset_file_sha256": file_hash(dataset_path),
              "new_cells": 68, "reused_cells": 17, "per_seed_rows": len(per_seed),
              "per_instance_rows": len(per_instance), "replicate_unit": "training_seed",
              "seeds_per_condition": 5, "sample_sd_ddof": 1, "attempt_inventory": status,
              "numerical_tolerances": {"score_and_summary_relative_and_absolute": 1e-9,
                                       "float32_actor_and_critic_relative_and_absolute": 2e-5,
                                       "initial_behavior_ratio_relative_and_absolute": 1e-4},
              "approved_attempt_selection": selection, "analysis_source_sha256": digest(sources),
              "strict_training_source_sha256": source_hashes["strict"],
              "new_legacy_training_source_sha256": source_hashes["new_legacy"],
              "initial_policy_evidence": policy_evidence,
              "metric_definitions": {
                  "physical_capacity_violation_rate_over_all_completions": "The numerator counts parsed outputs whose selected weight exceeds capacity. The denominator contains every completion, including parse failures and empty selections.",
                  "recorded_verdict_failure_rate": "This rate uses the verdict recorded by each protocol. The strict verdict includes nonempty output, while the historical verdict allows empty parsed selections.",
                  "useful_failure_rate": "The numerator counts every output that is not parsed, nonempty, and physically capacity-feasible. The denominator contains every completion.",
                  "completion_tokens_mean": "Strict records supply the emitted completion token count, including a terminating stop token. Historical records have no comparable token log, so this value is null.",
              },
              "limitations": [
                  "Historical logs preserve parsed selections and cannot establish strict validity of the original completion text.",
                  "All seventeen reused records lack initial parameter checksums and resolved model/tokenizer revisions. Initial checkpoint equality across the historical bridge cannot be cryptographically established from those logs.",
                  "Raw strict scores are recomputed from text. Token counts and stopping positions are checked, but decoding is not independently replayed without the matching tokenizer.",
                  "Later actor objectives cannot be fully reconstructed because the logs do not retain every epoch's per-completion log probabilities or model checkpoints.",
                  "The analysis reports seed means, sample standard deviations, and planned paired differences without significance tests.",
                  "The strict useful verdict requires nonempty output. The historical constraint verdict admits an empty parsed selection.",
              ], "runs": evidence}
    report["table_sha256"] = {name: file_hash(destination / name) for name in
                              ("per_seed.csv", "per_instance.csv", "summary.csv", "paired_differences.csv")}
    (destination / "validation_sources.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
    (destination / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--manifest", type=Path, default=HERE / "configs/revision_manifest.json")
    parser.add_argument("--dataset", type=Path, default=HERE / "configs/revision_dataset.json")
    parser.add_argument("--runs-root", type=Path, default=HERE / "runs/revision_20260915")
    parser.add_argument("--reused-root", type=Path, default=HERE / "runs")
    parser.add_argument("--attempt-selection", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.status:
            manifest, dataset = load_study(args.manifest, args.dataset)
            study.validate_manifest(manifest, dataset, runs_dir=args.reused_root)
            report = study_status(manifest, args.runs_root)
            report["reused_records_fingerprint_verified"] = 17
            if args.out:
                args.out.mkdir(parents=True, exist_ok=True)
                (args.out / "status.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            require(args.out is not None, "The analyze command requires an output directory.")
            report = analyze_study(args.manifest, args.dataset, args.runs_root, args.reused_root,
                                   args.out, args.attempt_selection)
            print(json.dumps({key: report[key] for key in ("status", "new_cells", "reused_cells", "per_seed_rows")}, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "rejected", "study_results_validated": False, "reason": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
