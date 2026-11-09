#!/usr/bin/env python3
"""Freeze the revision dataset and run matrix without launching any training.

The manifest records 68 new runs and 17 explicitly reused completed runs. The two
verdict weights and the strict reward scale are fixed design choices. Held-out
instances supply an audit of that design and never select its hyperparameters.

Run this script with --out-dir DIRECTORY to create a separate review bundle.
Existing files may be reused byte for byte, but different files are never replaced.
The launcher must freeze training sources in a separate record before submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import revision_task as task

HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = "revision_experiment_manifest_v1"
STUDY_ID = "capacity_revision_2026_09_16"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = tuple(range(5))
WEIGHTS = (5.0, 25.0)
STRICT_REWARD_SCALE = 26.0
STRICT_SCORE_MICROBATCH = 4
TRAINING_COMPLETIONS = 7200
METHODS = ("plain", "drgrpo", "ngrpo_feasible", "vfilter", "sfilter")
ARTIFACTS = ("config.json", "metrics.jsonl", "provenance.json", "DONE")
LEGACY_DEFAULTS = {"adv_variant": "grpo", "group_filter": "none", "kl_beta": 0.0}
PRELAUNCH_AMENDMENTS = [{
    "amendment_id": "strict_score_microbatch_four",
    "prior_manifest_sha256": "7505882e05ca2d3d03135fe5980a77abe92ba3e9c4523e30556a00444c6b33d2",
    "stage": "before_any_training",
    "scope": "All thirty-five strict runs use the same scoring chunk size.",
    "field": "score_microbatch", "previous_value": 1, "new_value": 4,
    "reason": "Four-sequence scoring chunks reduce execution overhead while preserving the defined mean-gradient accumulation and sampled-completion budget.",
    "result_selection": "No training jobs or study results preceded this amendment.",
}]


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def manifest_digest(manifest):
    return hashlib.sha256(_encoded({k: v for k, v in manifest.items() if k != "sha256"})).hexdigest()


def _json_text(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def legacy_config(method, weight, seed):
    variants = {"plain": ("grpo", "none"), "drgrpo": ("drgrpo", "none"),
                "ngrpo_feasible": ("ngrpo_feasible", "none"),
                "vfilter": ("grpo", "verdict_mixed"), "sfilter": ("grpo", "reward_spread")}
    variant, group_filter = variants[method]
    return {"task": "capacity", "arm": "binary", "w": weight, "w_task": 1.0,
            "vmax": 0.03, "regime": "H", "seed": seed, "adv_variant": variant,
            "group_filter": group_filter, "kl_beta": 0.0, "steps": 300,
            "group_size": 4, "batch_prompts": 6, "eval_every": 10, "eval_samples": 16,
            "eval_seed": 12345, "lr": 1e-5, "max_new_tokens": 128, "temperature": 1.0,
            "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.0, "weight_decay": 0.0,
            "clip_coef": 0.2, "grad_clip": 1.0, "model_id": MODEL_ID,
            "backward_per_group": True, "no_ref": False, "tiny": False}


def strict_config(method, reward, weight, seed, group_size):
    return {"method": method, "reward": reward, "weight": weight,
            "reward_scale": STRICT_REWARD_SCALE, "seed": seed, "group_size": group_size,
            "steps": 300 if group_size == 4 else 150, "epochs": 4,
            "eval_every": 30 if group_size == 4 else 15, "eval_samples": 16,
            "eval_seed": 12345, "score_microbatch": STRICT_SCORE_MICROBATCH, "max_new_tokens": 128,
            "model_id": MODEL_ID, "lr": 1e-5, "critic_lr": 0.3, "critic_steps": 4,
            "grad_clip": 1.0, "weight_decay": 0.0, "clip_coef": 0.2,
            "no_ref": True, "tiny": False, "skip_heldout": False}


def expected_cells():
    """Return the ordered new cells and the exact historical cells admitted for reuse."""
    new, reused = [], []
    for method_index, method in enumerate(METHODS):
        for weight in WEIGHTS:
            for seed in SEEDS:
                cell = {"run_id": f"legacy_{method}_w{int(weight)}_s{seed}",
                        "protocol": "legacy", "method": method,
                        "config": legacy_config(method, weight, seed),
                        "training_completions": TRAINING_COMPLETIONS}
                old_path = None
                if weight == 5 and method == "plain":
                    old_path = (f"572991/{4 + seed}_H_binary_s{seed}" if seed < 3
                                else f"878511/{36 + seed}_H_binary_s{seed}")
                elif weight == 5 and seed < 3:
                    arm = {"drgrpo": "drgrpo", "ngrpo_feasible": "ngrpo",
                           "vfilter": "vfilter", "sfilter": "sfilter"}[method]
                    row = (method_index - 1) * 3 + seed + 1
                    old_path = f"759612/{row}_H_binary_{arm}_s{seed}"
                if old_path is None:
                    new.append(cell)
                else:
                    reused.append({**cell, "source_path": f"experiments/cancellation/runs/{old_path}"})
    strict = [(method, "binary", weight, 4) for method in ("grpo", "ppo") for weight in WEIGHTS]
    strict += [("grpo", "graded", 5.0, 4)]
    strict += [(method, "binary", 25.0, 8) for method in ("grpo", "ppo")]
    for method, reward, weight, group_size in strict:
        for seed in SEEDS:
            new.append({"run_id": f"strict_{method}_{reward}_w{int(weight)}_k{group_size}_s{seed}",
                        "protocol": "strict", "method": method,
                        "config": strict_config(method, reward, weight, seed, group_size),
                        "training_completions": TRAINING_COMPLETIONS})
    for row, cell in enumerate(new, start=1):
        cell["row"] = row
    return new, reused


def _source_dir(record, runs_dir):
    prefix = "experiments/cancellation/runs/"
    relative = record["source_path"]
    if not relative.startswith(prefix) or ".." in Path(relative).parts:
        raise ValueError("A reused source path is outside the declared run directory.")
    return Path(runs_dir) / relative[len(prefix):]


def inspect_reused_run(record, runs_dir):
    """Check the original configuration, complete sampled budget, and terminal artifacts."""
    directory = _source_dir(record, runs_dir)
    missing = [name for name in ARTIFACTS if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"The reused run {record['run_id']} is missing {', '.join(missing)}.")
    if (directory / "DONE").read_text().strip() != "ok":
        raise ValueError(f"The reused run {record['run_id']} has no successful terminal marker.")
    actual = json.loads((directory / "config.json").read_text())
    normalized = {**LEGACY_DEFAULTS, **actual}
    mismatches = [key for key, value in record["config"].items() if normalized.get(key) != value]
    if mismatches:
        raise ValueError(f"The reused run {record['run_id']} has unmatched configuration fields {mismatches}.")
    provenance = json.loads((directory / "provenance.json").read_text())
    cfg_hash = hashlib.sha256(json.dumps({k: v for k, v in actual.items() if k != "out_dir"},
                                       sort_keys=True).encode()).hexdigest()[:16]
    if provenance.get("config_hash") != cfg_hash:
        raise ValueError(f"The reused run {record['run_id']} has a mismatched provenance config hash.")
    if provenance.get("steps_completed") != 300 or not provenance.get("end_time"):
        raise ValueError(f"The reused run {record['run_id']} did not complete its declared budget.")
    train_steps, eval_steps, completions = [], [], 0
    with (directory / "metrics.jsonl").open() as stream:
        for line in stream:
            observation = json.loads(line)
            if observation.get("phase") == "train":
                train_steps.append(observation["step"])
                groups = observation.get("groups", [])
                if len(groups) != 6 or any(len(group.get(key, [])) != 4
                                           for group in groups for key in ("selections", "rewards", "adv")):
                    raise ValueError(f"The reused run {record['run_id']} has an unmatched sampled group budget.")
                completions += sum(len(group["selections"]) for group in groups)
            elif observation.get("phase") == "eval":
                eval_steps.append(observation["step"])
                if observation.get("eval_n") != 96 or len(observation.get("eval_selections", [])) != 96:
                    raise ValueError(f"The reused run {record['run_id']} has an unmatched evaluation budget.")
            else:
                raise ValueError(f"The reused run {record['run_id']} contains an unknown observation phase.")
    if train_steps != list(range(1, 301)) or eval_steps != list(range(0, 301, 10)) or completions != TRAINING_COMPLETIONS:
        raise ValueError(f"The reused run {record['run_id']} has missing, duplicate, or incomplete observations.")
    digest = provenance.get("training_sources_digest")
    is_original_plain = record["method"] == "plain" and record["config"]["seed"] < 3
    if not digest and not is_original_plain:
        raise ValueError(f"The reused run {record['run_id']} is missing its training source digest.")
    if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"The reused run {record['run_id']} has an invalid training source digest.")
    waiver = {"required": is_original_plain, "id": "F4" if is_original_plain else None,
              "note": ("The original plain run lacks a training_sources_digest. Its reuse requires the recorded F4 drift_pass waiver. "
                       "That waiver does not establish bitwise reproducibility or certify new source changes."
                       if is_original_plain else "The historical run records its training source digest.")}
    return {**record, "artifact_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                          for name in ARTIFACTS},
            "recorded_training_sources_digest": digest, "provenance_waiver": waiver,
            "verified_terminal": True, "verified_training_steps": 300,
            "verified_training_completions": completions}


def build_manifest(dataset, runs_dir=HERE / "runs"):
    task.validate_manifest(dataset)
    new, reused = expected_cells()
    if len(dataset["train"]) != 6 or len(dataset["heldout"]) != 24 or dataset["capacity_ratio"] != 0.03:
        raise ValueError("This study requires six training instances and twenty-four held-out instances at capacity ratio 0.03.")
    resolved = [_source_dir(record, runs_dir).resolve() for record in reused]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Two reused records resolve to the same source directory.")
    manifest = {
        "schema_version": SCHEMA_VERSION, "study_id": STUDY_ID,
        "dataset_sha256": dataset["sha256"], "dataset_file": "revision_dataset.json",
        "prelaunch_amendments": [dict(record) for record in PRELAUNCH_AMENDMENTS],
        "frozen_common": {
            "model_id": MODEL_ID, "seeds": list(SEEDS), "verdict_weights": list(WEIGHTS),
            "strict_reward_scale": STRICT_REWARD_SCALE, "training_completions_per_run": TRAINING_COMPLETIONS,
            "training_prompts": 6, "heldout_prompts": 24,
            "weight_selection": "The weights 5 and 25 preserve the historical bridge and were fixed from the training design. Held-out thresholds are audit metadata only.",
            "reward_scale_selection": "The strict reward scale 26 is the fixed value 1 + 25. It is shared by every strict arm and seed.",
            "strict_sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0},
            "strict_update_epochs": 4, "strict_kl_beta": 0.0,
            "source_freeze": "The launcher must record and verify a separate source snapshot before any training submission.",
            "legacy_scope": "The historical trainer, parser, prompts, reward scale, nucleus sampling, and single update remain unchanged.",
            "strict_scope": "The strict task, explicit capacities, disjoint evaluation instances, full-softmax sampling, and four update epochs define a separate protocol.",
        },
        "new_runs": new, "reused_runs": [inspect_reused_run(record, runs_dir) for record in reused],
    }
    manifest["sha256"] = manifest_digest(manifest)
    validate_manifest(manifest, dataset)
    return manifest


def validate_manifest(manifest, dataset, runs_dir=None):
    """Reject missing cells, duplicated cells, changed budgets, and changed frozen content."""
    task.validate_manifest(dataset)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("study_id") != STUDY_ID:
        raise ValueError("The experiment manifest schema or study identifier is unsupported.")
    if manifest.get("sha256") != manifest_digest(manifest):
        raise ValueError("The experiment manifest digest does not match its contents.")
    if manifest.get("dataset_sha256") != dataset["sha256"]:
        raise ValueError("The experiment manifest refers to a different frozen dataset.")
    if manifest.get("prelaunch_amendments") != PRELAUNCH_AMENDMENTS:
        raise ValueError("The manifest must preserve its recorded prelaunch amendment.")
    new, reused = manifest["new_runs"], manifest["reused_runs"]
    all_records = new + reused
    if len({row["run_id"] for row in all_records}) != len(all_records):
        raise ValueError("The experiment manifest contains duplicate run identifiers.")
    if len(new) != 68 or len(reused) != 17:
        raise ValueError("The experiment manifest must contain 68 new and 17 reused runs.")
    for record in all_records:
        cfg = record["config"]
        prompts = cfg["batch_prompts"] if record["protocol"] == "legacy" else len(dataset["train"])
        if cfg["steps"] * cfg["group_size"] * prompts != TRAINING_COMPLETIONS or record["training_completions"] != TRAINING_COMPLETIONS:
            raise ValueError("An experiment cell has an unmatched training completion budget.")
    expected_new, expected_reused = expected_cells()
    if new != expected_new:
        raise ValueError("The new run matrix differs from the fixed methods, weights, seeds, settings, or order.")
    for record, expected in zip(reused, expected_reused):
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("A reused cell differs from its explicitly admitted historical source.")
        if (set(record["artifact_sha256"]) != set(ARTIFACTS)
                or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in record["artifact_sha256"].values())):
            raise ValueError("A reused cell has missing or malformed artifact hashes.")
        original_plain = record["method"] == "plain" and record["config"]["seed"] < 3
        if record["provenance_waiver"]["required"] != original_plain or (original_plain and record["provenance_waiver"]["id"] != "F4"):
            raise ValueError("A reused original plain run must retain its explicit F4 waiver requirement.")
        if record.get("verified_terminal") is not True or record.get("verified_training_completions") != TRAINING_COMPLETIONS:
            raise ValueError("A reused run lacks terminal or sampled-budget verification.")
        if runs_dir is not None and record != inspect_reused_run(expected, runs_dir):
            raise ValueError("A reused source has changed since the manifest was frozen.")
    common = manifest["frozen_common"]
    if (common.get("verdict_weights") != list(WEIGHTS) or common.get("strict_reward_scale") != STRICT_REWARD_SCALE
            or common.get("training_prompts") != 6 or common.get("training_completions_per_run") != TRAINING_COMPLETIONS):
        raise ValueError("The manifest's shared design choices have changed.")
    return manifest


def write_bundle(out_dir, dataset, manifest):
    """Check both targets before writing either file, and refuse different frozen content."""
    validate_manifest(manifest, dataset)
    target = Path(out_dir)
    contents = {"revision_dataset.json": _json_text(dataset), "revision_manifest.json": _json_text(manifest)}
    for name, content in contents.items():
        path = target / name
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f"Refusing to replace a different frozen artifact at {path}.")
    target.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        path = target / name
        if not path.exists():
            with path.open("x") as stream:
                stream.write(content)
    return target / "revision_manifest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "configs")
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    args = parser.parse_args(argv)
    dataset_path = args.out_dir / "revision_dataset.json"
    dataset = task.load_manifest(dataset_path) if dataset_path.exists() else task.build_dataset()
    manifest = build_manifest(dataset, args.runs_dir)
    target = write_bundle(args.out_dir, dataset, manifest)
    print(json.dumps({"manifest": str(target), "new_runs": len(manifest["new_runs"]),
                      "reused_runs": len(manifest["reused_runs"]), "sha256": manifest["sha256"],
                      "dataset_sha256": dataset["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
