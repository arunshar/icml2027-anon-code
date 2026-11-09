#!/usr/bin/env python3
"""Read every run, refuse unmatched comparisons, recompute counterfactual advantages, check
P1 and P2 on the sampled groups, and write tables and figures.

Usage: python analyze.py --runs <dir with one subdir per run> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

import rewards as RW
import tasks

# Fields that may legitimately differ between runs in one comparison.
# Fields that define an arm, a regime, or a seed, and may therefore differ inside one
# comparison. `w_task` belongs here because it is part of what an arm IS: the verdict-only and
# residual-only arms are the same penalty at a task weight of zero.
#
# `task` is deliberately absent and must stay absent. A matched-budget comparison holds one task
# fixed, and letting two tasks into one table is exactly what this gate exists to prevent.
#
# The remedy arms add three more fields of the same kind. The advantage variant, the group filter and
# the KL weight each decide what an arm is, so an arm that differs from its baseline only in one of
# them is the comparison the remedy tables exist to make.
ALLOWED_CONFIG_DIFF = {"arm", "w", "w_task", "regime", "vmax", "seed", "out_dir",
                       "adv_variant", "group_filter", "kl_beta"}
# The yoked filter controls add `keep_schedule` for the same reason. It names the schedule of kept
# groups that makes a control run what it is, and each draw is its own arm. The field sits in its own
# set so that ALLOWED_CONFIG_DIFF, which the paper counts and whose fields it names, keeps its
# meaning. A table that reports the controls must count this field too.
YOKED_CONFIG_DIFF = {"keep_schedule"}
# The bounded speed task of the ICML 2027 blocks (November 2026) trains a fixed-reference arm beside the
# centered arms. The shift `fixedref_c` names that arm the way `adv_variant` names the others, so it may
# differ between runs of one cohort. It sits in its own set for the same reason the yoked field does:
# the paper counts ALLOWED_CONFIG_DIFF and names each of its fields. The group size is not
# admitted here, because a K = 8 run samples twice the budget of a K = 4 run and is analysed as its own
# cohort, as the strict protocol's K = 8 cells were.
BOUNDED_CONFIG_DIFF = {"fixedref_c"}
ALLOWED_PROV_DIFF = {"slurm_job_id", "slurm_array_job_id", "slurm_array_task_id", "hostname", "start_time",
                     "end_time", "s_per_step_median", "wall_seconds", "steps_completed", "final_param_checksum",
                     "config_hash", "git_dirty",
                     # A content hash of the training sources. Runs made before it existed do not carry
                     # it, and a difference in the sources is already governed by the git_sha rule and
                     # the fingerprint waiver, which checks this digest on the runs it admits.
                     "training_sources_digest"}
# name -> (arm, penalty weight, task weight)
ARM_SPECS = {"graded": ("graded", 5.0, 1.0), "binary": ("binary", 5.0, 1.0),
             "none": ("none", 0.0, 1.0), "binary_1000x": ("binary", 5000.0, 1.0)}
CAPACITY_ARM_SPECS = dict(ARM_SPECS, verdict_only=("binary", 5.0, 0.0),
                          graded_no_task=("graded", 5.0, 0.0))
# The verdict at five times its light weight. On the speed task that weight sits above the crossover,
# so an arm there separates a remedy that acts from a weight that is too low. It is named here and
# kept out of ARM_SPECS, because the P1 and P2 checks iterate ARM_SPECS and must not change.
BINARY_ABOVE_W = 5 * ARM_SPECS["binary"][1]
# The graded arm at this weight is the residual below its ordering threshold on the speed task. At
# this weight the residual penalty's slope in the magnitude is twice the task term's slope. The reward
# above the limit therefore falls at the rate at which the task term rises below it. The value lies
# inside the open window where the residual reward still decreases with the magnitude on each side of
# the task cap. Inside that window the residual also stops ranking every feasible completion above
# every violator. make_manifests.py recomputes the window from the reward code and refuses a value
# outside it.
GRADED_BELOW_W = 0.01
# The graded arm at this weight is the scale-matched residual at the hard level of each task. The
# weight equals the light verdict's weight times the smallest limit in units, divided by the residual's
# factor of 100. On the speed task the limit is 5, so the penalty at the smallest violation equals the
# light verdict's penalty of 5. On the capacity task the six capacities are 6, 6, 5, 5, 6 and 5, so the
# first-violation penalty is 5 on three instances and 4.17 on the other three. It never exceeds the
# light verdict's penalty on any instance. make_manifests.py recomputes the weight from the reward code
# and refuses any other value.
GRADED_MATCHED_W = 0.25
# The graded arm at this weight is the capacity residual below its ordering threshold on four of the
# six hard-level instances. The weight is twice the inclusion bound of 0.02. Above that bound, adding
# any item to an overfull selection lowers the reward on every instance. That is the speed task's rule
# applied to the capacity reward. The weight lies in the open window (0.036, 0.0432). Inside it the
# best feasible answer outranks every violator on every instance. The residual no longer ranks every
# feasible answer above every violator on instances 0, 1, 3 and 4. make_manifests.py recomputes the
# weight and these properties from the reward code and refuses any other value.
GRADED_BELOW_CAP_W = 0.04
NAMED_WEIGHTS = {"binary_5x": ("binary", BINARY_ABOVE_W, 1.0),
                 "graded_below": ("graded", GRADED_BELOW_W, 1.0),
                 "graded_matched": ("graded", GRADED_MATCHED_W, 1.0),
                 "graded_below_cap": ("graded", GRADED_BELOW_CAP_W, 1.0)}
# A named weight listed here names a run only on these tasks. The capacity design's weight has no
# role on the speed task, so a speed run of the graded arm at 0.04 keeps its generic name. A config
# without a task is a speed run, as everywhere else in this file. Every other named weight
# applies on both tasks, which keeps each existing name unchanged.
NAMED_WEIGHT_TASKS = {"graded_below_cap": ("capacity",)}

# Keys a run's config.json may lack because it was written before the flag existed. Each value is the
# trainer's default for that flag, so filling it in describes exactly what the older run did. A test
# pins every value against the argument parser. Only absent keys are filled.
LEGACY_CONFIG_DEFAULTS = {"task": "speed", "w_task": 1.0, "backward_per_group": False,
                          "adv_variant": "grpo", "group_filter": "none", "kl_beta": 0.0,
                          "keep_schedule": None,
                          # Added 2026-11-06 with the ICML 2027 blocks. A run written before these two
                          # flags existed could not select grpo_fixedref, which is the only reader of
                          # the shift, and it never skipped a zero-advantage group. Both values are the
                          # trainer's parser defaults, and tests/test_remedy_analysis.py pins them to the parser.
                          "fixedref_c": 2.0, "skip_zero_adv": False}

VARIANT_SUFFIX = {"grpo": "", "drgrpo": "_drgrpo", "ngrpo_feasible": "_ngrpo", "ngrpo_sup": "_ngrposup",
                  # The fixed-reference variant of Block A' (ICML 2027, 2026-11-06). The suffix matches
                  # the arm_name column of slurm/manifest_speed_bounded.csv, which writes
                  # binary_w5_fixedref for the same arm.
                  "grpo_fixedref": "_fixedref"}
FILTER_SUFFIX = {"none": "", "verdict_mixed": "_vfilter", "reward_spread": "_sfilter"}
# This pattern matches the id of a yoked control's schedule, for example speed_w25_s1_d5.
# make_keep_schedules.py writes the same form. The pattern refuses leading zeros, so one run has
# exactly one id and one name.
KEEP_SCHEDULE_ID = re.compile(r"(speed|capacity)_w(0|[1-9][0-9]*)_s(0|[1-9][0-9]*)_d(0|[1-9][0-9]*)")


def _load_run(d: Path):
    cfg = json.loads((d / "config.json").read_text())
    for k, v in LEGACY_CONFIG_DEFAULTS.items():
        cfg.setdefault(k, v)
    prov = json.loads((d / "provenance.json").read_text())
    train, evals = [], []
    for line in (d / "metrics.jsonl").read_text().splitlines():
        rec = json.loads(line)
        (evals if rec.get("phase") == "eval" else train).append(rec)
    return {"dir": d, "cfg": cfg, "prov": prov, "train": train, "evals": evals}


def load_runs(root: Path):
    runs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (d / "DONE").exists():
            print(f"[skip] {d.name}: no DONE marker", file=sys.stderr)
            continue
        runs.append(_load_run(d))
    return runs


def _remedy_suffix(cfg) -> str:
    variant = cfg.get("adv_variant", "grpo")
    group_filter = cfg.get("group_filter", "none")
    if variant not in VARIANT_SUFFIX:
        raise ValueError(f"unknown adv_variant {variant!r} in a run config")
    if group_filter not in FILTER_SUFFIX:
        raise ValueError(f"unknown group_filter {group_filter!r} in a run config")
    kl = "_kl" if float(cfg.get("kl_beta", 0.0)) > 0 else ""
    return VARIANT_SUFFIX[variant] + FILTER_SUFFIX[group_filter] + kl + _yoked_suffix(cfg)


def keep_schedule_draw(cfg):
    """Return the draw index of a yoked control run, or None for any other run.

    The id must have the strict form and must name the run's own task, weight and seed, because an
    id that describes another run would give this run a wrong name."""
    sid = cfg.get("keep_schedule")
    if sid is None:
        return None
    m = KEEP_SCHEDULE_ID.fullmatch(sid) if isinstance(sid, str) else None
    if m is None:
        raise ValueError(f"malformed keep_schedule {sid!r} in a run config")
    task, w, seed, draw = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    if task != cfg.get("task", "speed") or float(w) != float(cfg["w"]) or (
            "seed" in cfg and seed != int(cfg["seed"])):
        raise ValueError(f"keep_schedule {sid!r} does not describe this run's task, weight and seed")
    return draw


def _yoked_suffix(cfg) -> str:
    draw = keep_schedule_draw(cfg)
    return "" if draw is None else f"_yoked_d{draw}"


def arm_name(cfg):
    """The arm's name: the reward design, then one suffix per remedy flag that is not at its default.

    A run at every default gets exactly the name it had before the remedy flags existed. A named
    weight that NAMED_WEIGHT_TASKS restricts names a run only on the tasks it lists."""
    task = cfg.get("task", "speed")
    specs = CAPACITY_ARM_SPECS if task != "speed" else ARM_SPECS
    wt = float(cfg.get("w_task", 1.0))
    base = None
    for name, (arm, w, t) in list(specs.items()) + list(NAMED_WEIGHTS.items()):
        if name in NAMED_WEIGHT_TASKS and task not in NAMED_WEIGHT_TASKS[name]:
            continue
        if cfg["arm"] == arm and float(cfg["w"]) == w and t == wt:
            base = name
            break
    if base is None:
        base = f'{cfg["arm"]}_w{cfg["w"]}_t{wt:g}'
    return base + _remedy_suffix(cfg)


# The files whose contents can change what a training run does. A commit that touches none of
# them cannot have changed a run, however much else it changed.
#
# This list exists because the array ran across two commits. The runs at the hard level and the
# runs at the easy level carry different `git_sha` values, and the strict rule refused the whole
# comparison. The strict rule was asking the wrong question. What the comparison needs is that
# the trainer, the reward, the task and the frozen constants were the same, and not that no one
# committed a ledger entry while the queue drained.
#
# Anything not on this list is analysis, tests, documentation or the paper, all of which run
# after a run finishes and none of which can reach into one. Adding a file here is a claim that
# it cannot affect training, so the list is deliberately short and explicit.
TRAINING_SOURCES = (
    "experiments/cancellation/train_speed_grpo.py",
    "experiments/cancellation/rewards.py",
    "experiments/cancellation/tasks.py",
    "experiments/cancellation/speed_task.py",
    "experiments/cancellation/capacity_task.py",
    "experiments/cancellation/make_capacity_instances.py",
    "experiments/cancellation/configs/base.yaml",
    "experiments/cancellation/configs/arms.yaml",
    "experiments/cancellation/configs/regimes.yaml",
    # The manifests set each run's arm, weights, limit and seed. Leaving them out let a commit
    # that edited a row count as touching no training source. The capacity manifest did change
    # inside the span this waiver covers, by appending three rows, which is why the append-only
    # test below exists rather than a flat refusal.
    "experiments/cancellation/slurm/manifest.csv",
    "experiments/cancellation/slurm/manifest_capacity.csv",
    # The revision's manifests set the remedy flags as well, so they count for the same reason.
    "experiments/cancellation/slurm/manifest_speed_remedy.csv",
    "experiments/cancellation/slurm/manifest_capacity_remedy.csv",
    "experiments/cancellation/slurm/manifest_capacity_kl.csv",
    "experiments/cancellation/slurm/manifest_fingerprint_speed.csv",
    # The yoked filter controls read their kept groups from the schedule file, and their manifests
    # name the schedule of each row, so all three decide what those runs do.
    "experiments/cancellation/configs/keep_schedules.json",
    "experiments/cancellation/slurm/manifest_speed_yoked.csv",
    "experiments/cancellation/slurm/manifest_capacity_yoked.csv",
    # Block A' of the ICML 2027 plan (added 2026-11-06). The bounded speed task module sets the task
    # term of every speed_bounded run, and its manifest sets each row's arm, weights, limit, seed,
    # advantage variant, fixed-reference shift and zero-advantage skip, so both decide what a run does.
    # They sit at the end so the first fifteen entries stay the historical list that
    # analyze_revision.LEGACY_TRAINING_SOURCES pins.
    "experiments/cancellation/speed_bounded_task.py",
    "experiments/cancellation/slurm/manifest_speed_bounded.csv",
)

_REPO_ROOT = Path(__file__).resolve().parent


def training_sources_between(sha_a: str, sha_b: str):
    """Training-relevant files that differ between two commits.

    Returns a sorted list of paths, or None when git cannot answer. None keeps the caller on the
    strict rule, because an unanswerable question is not a passed check.
    """
    if sha_a == sha_b:
        return []
    try:
        r = subprocess.run(["git", "diff", "--name-only", sha_a, sha_b, "--", *TRAINING_SOURCES],
                           cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    changed = sorted(x for x in r.stdout.splitlines() if x.strip())
    # A manifest that only gained rows cannot have changed a run that already existed. Appending
    # the easy-level baseline to manifest_capacity.csv is exactly that case. Anything else in a
    # manifest, and anything at all in the other files, still counts as a change.
    return [f for f in changed if not (f.endswith(".csv") and _appended_only(sha_a, sha_b, f))]


def _appended_only(sha_a: str, sha_b: str, path: str) -> bool:
    """True when `path` at sha_b starts with exactly its content at sha_a.

    Rows already in the file are then untouched, so a run built from one of them is unaffected.
    Any doubt returns False, because an unanswerable question is not a passed check.
    """
    def show(sha):
        try:
            r = subprocess.run(["git", "show", f"{sha}:{path}"], cwd=_REPO_ROOT,
                               capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout if r.returncode == 0 else None

    before, after = show(sha_a), show(sha_b)
    if before is None or after is None:
        return False
    return after.startswith(before) and len(after) >= len(before)


# ---------------------------------------------------------------- fingerprint waiver
#
# The remedy flags changed the trainer, the reward module and the task modules, so every run made
# after them differs in a training source from every run made before. The strict rule would then
# refuse any table that puts a new remedy arm beside an old baseline.
#
# The waiver admits that comparison only on evidence from real runs. A fingerprint reruns one old
# run at the new commit and again at the old commit, and compares all three logs:
#   pass              the new run equals the reference run
#   conditional_pass  the new run differs from the reference but equals the rerun at the old commit,
#                     so the difference is job nondeterminism and not the code change
#   drift_pass        (only where the fingerprint allows drift, which the capacity task needs) the
#                     new run and the rerun at the old commit both leave the reference, and at every
#                     step where two runs sample the same answers their rewards and advantages agree,
#                     so the difference is floating-point drift that happens without the code change
#   stop              the new run differs from the rerun at the old commit, so the change altered
#                     the default path, and nothing may be compared across it
#
# equivalence.json is analysis-side. It is not a training source, because no run reads it. Every
# path in it is repo-relative, and the analysis recomputes every recorded outcome from the run
# directories before it honours one.
EQUIVALENCE_PATH = Path(__file__).resolve().parent / "equivalence.json"
FINGERPRINT_OUTCOMES = ("pass", "conditional_pass", "drift_pass", "stop")


class EquivalenceError(ValueError):
    """equivalence.json is malformed, or it names a path that is not repo-relative."""


def _repo_relative(p) -> bool:
    return (isinstance(p, str) and p.strip() == p and p != "" and not p.startswith(("/", "~"))
            and "\\" not in p and ".." not in Path(p).parts and not Path(p).is_absolute())


def _no_absolute_strings(obj, where="$"):
    if isinstance(obj, str):
        if obj.startswith(("/", "~")):
            raise EquivalenceError(f"{where} holds an absolute path; only repo-relative paths are allowed")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _no_absolute_strings(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _no_absolute_strings(v, f"{where}[{i}]")


def load_equivalence(path=None):
    """The list of fingerprint waivers. A missing file means none. A malformed file raises."""
    path = Path(path) if path is not None else EQUIVALENCE_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise EquivalenceError(f"{path.name} is not valid JSON: {e}")
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise EquivalenceError(f"{path.name} must be an object with schema 1")
    _no_absolute_strings(data)
    waivers = data.get("waivers")
    if not isinstance(waivers, list):
        raise EquivalenceError(f"{path.name} must hold a list under 'waivers'")
    for i, wv in enumerate(waivers):
        for k in ("base_sha", "new_sha", "new_training_sources_digest", "fingerprints"):
            if not wv.get(k):
                raise EquivalenceError(f"waiver {i} lacks {k}")
        for fp in wv["fingerprints"]:
            for k in ("name", "reference", "new", "rerun_at_base", "steps", "outcome"):
                if k not in fp:
                    raise EquivalenceError(f"waiver {i} fingerprint lacks {k}")
            for k in ("reference", "new", "rerun_at_base"):
                if not _repo_relative(fp[k]):
                    raise EquivalenceError(f"waiver {i} fingerprint {fp['name']}: {k} is not a "
                                           f"repo-relative path")
            if fp["outcome"] not in FINGERPRINT_OUTCOMES:
                raise EquivalenceError(f"waiver {i} fingerprint {fp['name']}: outcome "
                                       f"{fp['outcome']!r} is not one of {FINGERPRINT_OUTCOMES}")
            if not isinstance(fp["steps"], int) or fp["steps"] < 1:
                raise EquivalenceError(f"waiver {i} fingerprint {fp['name']}: steps must be a "
                                       f"positive integer")
    return waivers


def _fingerprint_view(run, steps: int):
    """What two runs must share to count as the same run over their first `steps` steps."""
    tspec = tasks.get(run["cfg"].get("task", "speed"))
    train = [t for t in run["train"] if t.get("step", 0) <= steps]
    if len(train) != steps:
        raise EquivalenceError(f'{run["dir"].name} logged {len(train)} of the {steps} steps compared')
    keys = (tspec.ITEMS_KEY, "rewards", "adv", "all_violating", "has_violator")
    view = {"groups": [[{k: g.get(k) for k in keys} for g in t["groups"]] for t in train],
            "evals": [(e["step"], e.get(tspec.EVAL_ITEMS_KEY)) for e in run["evals"]
                      if e["step"] <= steps]}
    if int(run["cfg"]["steps"]) == steps:
        view["checksum"] = run["prov"].get("final_param_checksum")
    return view


def _same_run(a, b) -> bool:
    if a["groups"] != b["groups"] or a["evals"] != b["evals"]:
        return False
    if "checksum" in a and "checksum" in b:
        return a["checksum"] == b["checksum"]
    return True


def _logic_break(a_view, b_view, items_key):
    """The first step at which two runs sample the same answers and still log different rewards or
    advantages, or None. That is a difference no floating-point drift in the weights can explain."""
    for i, (sa, sb) in enumerate(zip(a_view["groups"], b_view["groups"])):
        if all(x[items_key] == y[items_key] for x, y in zip(sa, sb)) and any(
                x[k] != y[k] for x, y in zip(sa, sb) for k in ("rewards", "adv")):
            return i + 1
    return None


def fingerprint_outcome(reference, new, rerun_at_base, steps: int, allow_drift: bool = False) -> str:
    """The ledger entry of 2026-09-14 fixed the drift rule before the capacity rerun was read.

    Capacity training is not bit-reproducible across GPU jobs: the rerun at the old commit leaves its
    own reference at the same step the new run does. Drift is accepted only when both runs leave the
    reference and no pair of runs ever logs different rewards or advantages for identical answers."""
    ref_v, new_v, base_v = (_fingerprint_view(x, steps) for x in (reference, new, rerun_at_base))
    if _same_run(new_v, ref_v):
        return "pass"
    if _same_run(new_v, base_v):
        return "conditional_pass"
    if not allow_drift or _same_run(base_v, ref_v):
        return "stop"
    key = tasks.get(new["cfg"].get("task", "speed")).ITEMS_KEY
    if any(_logic_break(x, y, key) for x, y in ((new_v, ref_v), (base_v, ref_v), (new_v, base_v))):
        return "stop"
    return "drift_pass"


def _fingerprint_waiver(pair, equivalence):
    """(waiver, None) when a recorded and re-verified waiver covers this commit pair, else
    (None, reason or None)."""
    for wv in equivalence:
        for old, new in ((pair[0], pair[1]), (pair[1], pair[0])):
            if new != wv["new_sha"] or training_sources_between(old, wv["base_sha"]) != []:
                continue
            for fp in wv["fingerprints"]:
                if fp["outcome"] == "stop":
                    return None, f"fingerprint {fp['name']} recorded stop"
                dirs = [_REPO_ROOT / fp[k] for k in ("reference", "new", "rerun_at_base")]
                absent = [fp[k] for k, d in zip(("reference", "new", "rerun_at_base"), dirs)
                          if not (d / "DONE").exists()]
                if absent:
                    return None, (f"fingerprint {fp['name']} cannot be re-verified, because "
                                  f"{', '.join(absent)} is absent")
                try:
                    got = fingerprint_outcome(*[_load_run(d) for d in dirs], fp["steps"],
                                              bool(fp.get("allow_drift", False)))
                except EquivalenceError as e:
                    return None, f"fingerprint {fp['name']}: {e}"
                if got != fp["outcome"]:
                    return None, (f"fingerprint {fp['name']} records {fp['outcome']} and the logs "
                                  f"give {got}")
            return wv, None
    return None, None


def matched_budget_check(runs, notes=None, equivalence=None):
    """Every run must agree on every config and provenance field outside the allowed sets.

    The one field treated specially is `git_sha`. Two commits that touch no training source
    produce identical runs, so a difference there is checked against the diff rather than
    refused outright. Any such waiver is appended to `notes` and reported, never silent.

    A difference that does touch a training source is still refused, unless a fingerprint waiver
    in equivalence.json covers the pair and its outcomes re-verify from the run directories. Every
    run made at the waiver's new commit must then carry the digest the waiver records.
    """
    problems = []
    seen_sha_pairs = {}
    seen_waivers = {}
    used_waivers = []
    ref_cfg, ref_prov = runs[0]["cfg"], runs[0]["prov"]
    for r in runs[1:]:
        for k in sorted(set(ref_cfg) | set(r["cfg"])):
            if k in ALLOWED_CONFIG_DIFF or k in YOKED_CONFIG_DIFF or k in BOUNDED_CONFIG_DIFF:
                continue
            if ref_cfg.get(k) != r["cfg"].get(k):
                problems.append(f'{r["dir"].name}: config.{k} = {r["cfg"].get(k)!r} vs {ref_cfg.get(k)!r}')
        for k in sorted(set(ref_prov) | set(r["prov"])):
            if k in ALLOWED_PROV_DIFF:
                continue
            if ref_prov.get(k) == r["prov"].get(k):
                continue
            if k == "git_sha":
                pair = (ref_prov.get(k), r["prov"].get(k))
                if pair not in seen_sha_pairs:
                    seen_sha_pairs[pair] = training_sources_between(*pair)
                changed = seen_sha_pairs[pair]
                if changed == []:
                    if notes is not None and pair not in [n.get("pair") for n in notes]:
                        notes.append({"pair": pair, "training_sources_changed": []})
                    continue
                if changed is None:
                    problems.append(
                        f'{r["dir"].name}: provenance.git_sha = {pair[1]!r} vs {pair[0]!r}, and '
                        f'git could not be asked which files differ')
                    continue
                if pair not in seen_waivers:
                    try:
                        if equivalence is None:
                            equivalence = load_equivalence()
                        seen_waivers[pair] = _fingerprint_waiver(pair, equivalence)
                    except EquivalenceError as e:
                        seen_waivers[pair] = (None, f"equivalence.json is invalid: {e}")
                waiver, reason = seen_waivers[pair]
                if waiver is not None:
                    if waiver not in used_waivers:
                        used_waivers.append(waiver)
                    if notes is not None and pair not in [n.get("pair") for n in notes]:
                        notes.append({"pair": pair, "training_sources_changed": changed,
                                      "fingerprint_waiver": {
                                          "base_sha": waiver["base_sha"], "new_sha": waiver["new_sha"],
                                          "outcomes": {fp["name"]: fp["outcome"]
                                                       for fp in waiver["fingerprints"]}}})
                    continue
                problems.append(
                    f'{r["dir"].name}: provenance.git_sha = {pair[1]!r} vs {pair[0]!r}, '
                    f'changing {", ".join(changed)}' + (f' ({reason})' if reason else ''))
                continue
            problems.append(f'{r["dir"].name}: provenance.{k} = {r["prov"].get(k)!r} vs {ref_prov.get(k)!r}')
    for waiver in used_waivers:
        for r in runs:
            if r["prov"].get("git_sha") != waiver["new_sha"]:
                continue
            got = r["prov"].get("training_sources_digest")
            if got != waiver["new_training_sources_digest"]:
                problems.append(f'{r["dir"].name}: training_sources_digest {got!r} is not the digest '
                                f'the fingerprint waiver for {waiver["new_sha"][:8]} records')
    for r in runs:
        if r["prov"]["steps_completed"] != r["cfg"]["steps"]:
            problems.append(f'{r["dir"].name}: steps_completed {r["prov"]["steps_completed"]} != steps {r["cfg"]["steps"]}')
    return problems


# The same object the group filter reads, so the counter and the filter agree about every group.
ZERO_VAR_TOL = RW.ZERO_SPREAD_TOL


def group_counters(runs):
    """Per-run group counts that separate cancellation from policy convergence.

    Computed from the per-group `rewards` already present in metrics.jsonl, so the runs
    that are already finished get these on the same footing as any new run, with no retrain.

    Three counts per run:
      all_violating          every member of the group violates the constraint
      zero_variance          every member earns the same reward, so the advantage is exactly
                             zero and a DAPO-style all-same-reward filter would drop the group
      all_violating_positive all-violating AND still has reward spread, which is the window in
                             which an added verdict could have acted before it cancelled
      window_last_step       the last optimizer step at which such a group occurred, and
      window_groups_in_step1 how many of those groups occurred in step 1. This is a count of
                             groups, not a step index. The paper said the window
                             "falls entirely in the first optimizer step", which is true of one
                             seed and false of the other two, where it reaches step 4. Counting
                             the steps here is what stops that sentence being written by hand.

    The zero-variance count conflates two mechanisms, cancellation and a converged policy
    emitting one answer, so it is only ever read next to the all-violating count.
    """
    FIELDS = ("groups", "zero_variance", "all_violating",
              "all_violating_zero_variance", "all_violating_positive_variance",
              "window_groups_in_step1", "window_last_step")
    rows = []
    for r in runs:
        # Seed every field. A defaultdict only creates keys it is asked for, so a run with no
        # all-violating groups at all would otherwise write a short CSV row.
        c = defaultdict(int, {k: 0 for k in FIELDS})
        for rec in r["train"]:
            for g in rec["groups"]:
                rw = np.asarray(g["rewards"], dtype=np.float64)
                zero = bool(np.ptp(rw) <= ZERO_VAR_TOL)
                c["groups"] += 1
                c["zero_variance"] += zero
                if g["all_violating"]:
                    c["all_violating"] += 1
                    c["all_violating_zero_variance"] += zero
                    c["all_violating_positive_variance"] += (not zero)
                    if not zero:
                        step = rec.get("step", 0)
                        c["window_last_step"] = max(c["window_last_step"], step)
                        c["window_groups_in_step1"] += (step == 1)
        c["steps"] = len(r["train"])
        # What the policy does after the last update that could have carried the verdict. The
        # paper describes this and the three numbers were typed by hand, which is the one thing
        # its AI use statement says no number of ours is.
        resid = [(t["step"], t.get("mean_residual")) for t in r["train"]
                 if t.get("mean_residual") is not None]
        if resid and c["window_last_step"]:
            close = c["window_last_step"]
            at_close = next((v for st, v in resid if st == close), None)
            after = next((v for st, v in resid if st == close + 1), None)
            peak = max(v for _, v in resid)
            peak_step = next(st for st, v in resid if v >= 0.99 * peak)
            c["residual_at_window_close"] = at_close
            c["residual_after_window"] = after
            c["residual_peak"] = peak
            c["residual_peak_step"] = peak_step
        # The counts are integers. The residual fields are floats and may be absent, so they are
        # not run through int(), which is what broke when they were added.
        FLOATISH = ("residual_at_window_close", "residual_after_window", "residual_peak")
        rows.append({"regime": r["cfg"].get("regime", ""), "arm": arm_name(r["cfg"]),
                     "seed": r["cfg"]["seed"],
                     **{k: (None if v is None else (float(v) if k in FLOATISH else int(v)))
                        for k, v in c.items()},
                     # Appended after every column that existed before the remedy arms, so the
                     # earlier columns keep their positions.
                     **_update_counts(r)})
    return rows


def _step_gradient(rec, kl_beta: float) -> str:
    """"skipped" when no optimizer step was taken, "zero" when one was taken on an exactly zero
    gradient, "nonzero" otherwise.

    A run that logs the pre-clip gradient norm is read from it. An older run is read from its
    logged advantages, because at beta = 0 a step whose every advantage is exactly zero has an
    exactly zero gradient, and every older run is at beta = 0."""
    if rec.get("opt_step_taken", True) is False:
        return "skipped"
    if rec.get("grad_norm_preclip") is not None:
        return "zero" if float(rec["grad_norm_preclip"]) == 0.0 else "nonzero"
    if kl_beta > 0:
        return "nonzero"
    kept = [g for g in rec.get("groups", []) if g.get("kept", True)]
    return "nonzero" if any(float(x) != 0.0 for g in kept for x in g.get("adv", [])) else "zero"


def _update_counts(r) -> dict:
    """Per-run counts of what the update did, from the per-group flags and per-step fields.

      mixed_groups         groups with at least one violator that are not all-violating
      all_feasible_groups  groups with no violator
      kept_groups          groups that entered the loss; every group in a run without a filter
      opt_steps_skipped    steps that took no optimizer step, because no group was kept
      momentum_only_steps  steps that took an optimizer step on an exactly zero gradient after an
                           earlier step with a nonzero one, so AdamW's first moment alone moved
                           the weights

    The two group-type counts are None for a log that lacks the has_violator flag."""
    kl_beta = float(r["cfg"].get("kl_beta", 0.0))
    out = {"mixed_groups": 0, "all_feasible_groups": 0, "kept_groups": 0,
           "opt_steps_skipped": 0, "momentum_only_steps": 0}
    flags_present, seen_nonzero = True, False
    for rec in r["train"]:
        for g in rec.get("groups", []):
            out["kept_groups"] += bool(g.get("kept", True))
            if "has_violator" not in g:
                flags_present = False
                continue
            out["mixed_groups"] += bool(g["has_violator"] and not g["all_violating"])
            out["all_feasible_groups"] += bool(not g["has_violator"])
        state = _step_gradient(rec, kl_beta)
        if state == "skipped":
            out["opt_steps_skipped"] += 1
        elif state == "zero":
            out["momentum_only_steps"] += seen_nonzero
        else:
            seen_nonzero = True
    if not flags_present:
        out["mixed_groups"] = out["all_feasible_groups"] = None
    return out


def remedy_timing(runs):
    """When each run's remedy acted, one row per run. Written for every analysis, because a plain run
    answers the same questions: it keeps every group, and its momentum-only steps are the ones the
    paper's Z1 correction is about."""
    rows = []
    for r in runs:
        cfg = r["cfg"]
        kl_beta = float(cfg.get("kl_beta", 0.0))
        groups = kept = allviol = 0
        first_kept = first_skipped = first_mixed = last_nonzero = None
        taken = 0
        for rec in r["train"]:
            st = rec.get("step")
            gs = rec.get("groups", [])
            n_kept = sum(1 for g in gs if g.get("kept", True))
            groups += len(gs)
            kept += n_kept
            allviol += sum(1 for g in gs if g.get("all_violating"))
            if n_kept and first_kept is None:
                first_kept = st
            if first_mixed is None and any(g.get("has_violator") and not g.get("all_violating")
                                           for g in gs):
                first_mixed = st
            state = _step_gradient(rec, kl_beta)
            if state == "skipped":
                first_skipped = st if first_skipped is None else first_skipped
            else:
                taken += 1
                if state == "nonzero":
                    last_nonzero = st
        counts = _update_counts(r)
        ev_last = r["evals"][-1] if r["evals"] else {}
        rows.append({
            "regime": cfg.get("regime", ""), "arm": arm_name(cfg), "seed": cfg["seed"],
            "adv_variant": cfg.get("adv_variant", "grpo"), "group_filter": cfg.get("group_filter", "none"),
            "kl_beta": kl_beta, "steps": len(r["train"]), "groups": groups,
            "kept_groups": kept, "kept_share": (kept / groups) if groups else None,
            # DAPO refills a batch by sampling more groups. This is how many groups it would have had to
            # draw per kept one. None when nothing was kept, where no amount of sampling would do.
            "oversampling_factor": (groups / kept) if kept else None,
            "all_violating_groups": allviol, "mixed_groups": counts["mixed_groups"],
            "opt_steps_taken": taken, "opt_steps_skipped": counts["opt_steps_skipped"],
            "momentum_only_steps": counts["momentum_only_steps"],
            "first_kept_step": first_kept, "first_skipped_step": first_skipped,
            "first_mixed_step": first_mixed, "last_nonzero_gradient_step": last_nonzero,
            "arrival_step": _kl_at_arrival(r)["arrival_step"],
            "eval_violation_rate_final": ev_last.get("eval_violation_rate"),
            "eval_parse_fail_rate_final": ev_last.get("eval_parse_fail_rate"),
        })
    return rows


def no_update_runs(runs):
    """Runs that never took an optimizer step. Their weights never moved, so every evaluation must
    equal the step-0 one exactly. A filter that keeps no group produces such a run, and this is the
    check that it really changed nothing."""
    rows = []
    for r in runs:
        if not r["train"] or any(t.get("opt_step_taken", True) for t in r["train"]):
            continue
        key = tasks.get(r["cfg"].get("task", "speed")).EVAL_ITEMS_KEY
        evs = r["evals"]
        same = bool(evs) and all(e.get(key) == evs[0].get(key) for e in evs[1:])
        rows.append({"run": r["dir"].name, "evals": len(evs), "eval_unchanged": same})
    return rows


ALGEBRA_TOL = 1e-9
# The verdict weights the algebra reads, by arm name: the light verdict, five times it and a thousand
# times it. The weights come from the arm tables, so none is typed here.
ALGEBRA_DESIGNS = {"binary": ARM_SPECS["binary"][1], "binary_5x": BINARY_ABOVE_W,
                   "binary_1000x": ARM_SPECS["binary_1000x"][1]}


def _algebra_one(items, vmax, spec, w_task, instance):
    """Per (variant, design): the largest advantage difference, and whether the signs agree."""
    none_r = RW.group_rewards(items, "none", 0.0, vmax, spec=spec, w_task=w_task)
    verdicts = RW.group_verdicts(items, vmax, spec=spec)
    diffs, light = {}, None
    for design, w in ALGEBRA_DESIGNS.items():
        br = RW.group_rewards(items, "binary", w, vmax, spec=spec, w_task=w_task)
        if design == "binary":
            light = br
        for v in RW.ADV_VARIANTS:
            ab = RW.advantage_variant(br, v, RW.reward_anchor(v, "binary", w, vmax, spec=spec,
                                                              w_task=w_task, instance=instance))
            an = RW.advantage_variant(none_r, v, RW.reward_anchor(v, "none", 0.0, vmax, spec=spec,
                                                                  w_task=w_task, instance=instance))
            diffs[(v, design)] = (float(np.max(np.abs(ab - an))),
                                  bool(np.array_equal(np.sign(ab), np.sign(an))))
    keeps = {f: RW.keep_group(f, light, verdicts) for f in RW.GROUP_FILTERS}
    return bool(np.ptp(none_r) <= ZERO_VAR_TOL), diffs, keeps


def variant_algebra(runs, tol=ALGEBRA_TOL):
    """Whether the verdict cancels under each advantage variant, on every logged all-violating group.

    For each group the added-verdict design and the no-penalty design are scored on the same logged
    answers, each under the variant with its own anchor. Each group lands in exactly one count:

      cancels               the two advantage vectors agree within `tol`
      acts_with_spread      the group has reward spread and the vectors differ
      acts_zero_spread      the group has no spread and the verdict flips the sign of the advantage
      eps_only_zero_spread  the group has no spread, the signs agree, and the vectors differ only
                            through the 1e-6 guard in the divisor

    The last two exist because of a property of the anchored variants. In a zero-spread group every
    member gets sign(r - anchor) / sqrt(K) apart from that guard, so the verdict changes the update
    only when it moves r across the anchor. `acts` is the sum of the two acting counts.

    Also counted: how many all-violating groups each filter would keep under the light verdict.
    Every count is reported per regime under `by_regime`, because the anchors and the crossover
    depend on the limit, and the totals over all regimes sit at the top level.

    The literal anchor is also reported under `literal_anchor_by_instance`, by regime, task weight
    and instance: its value under the light verdict and under no penalty, and how many all-violating
    groups it cancels in. That anchor is the design's own maximum reward, so it depends on the
    instance, and a prediction committed before the capacity remedy runs asked for it that way. The
    task weight is kept apart because an arm without a task term has a different anchor, and summing
    the two hid which design the acting groups came from.
    Identical groups recur once a policy settles, so each distinct group is computed once."""
    def empty():
        return {"all_violating_groups": 0, "all_violating_zero_spread": 0,
                "variants": {v: {d: {"cancels": 0, "acts": 0, "acts_with_spread": 0,
                                     "acts_zero_spread": 0, "eps_only_zero_spread": 0,
                                     "worst_diff": 0.0}
                                 for d in ALGEBRA_DESIGNS} for v in RW.ADV_VARIANTS},
                "filters_keep_all_violating": {f: 0 for f in RW.GROUP_FILTERS}}

    total, by_regime, cache = empty(), {}, {}
    literal, anchors = {}, {}
    for r in runs:
        vmax = float(r["cfg"]["vmax"])
        tspec = tasks.get(r["cfg"].get("task", "speed"))
        spec = None if tspec.name == "speed" else tspec
        w_task = float(r["cfg"].get("w_task", 1.0))
        key = tspec.ITEMS_KEY
        # Each regime also splits by task weight. An arm with no task term is a different design, and
        # pooling its groups with the arms that have one mixed two designs into one printed count.
        reg_block = by_regime.setdefault(str(r["cfg"].get("regime", "")), empty())
        wt_block = reg_block.setdefault("by_task_weight", {}).setdefault(f"{w_task:g}", empty())
        blocks = (total, reg_block, wt_block)
        for rec in r["train"]:
            for gi, g in enumerate(rec["groups"]):
                if not g["all_violating"]:
                    continue
                items = g[key] if key in g else g["speeds"]
                # JSON gives lists back. A capacity selection becomes the tuple the scoring functions
                # index, and a number (the speed tasks) passes through unchanged.
                items = [_retuple(x) for x in items]
                instance = tspec.instance_for(g.get("prompt_index", gi))
                ck = (tspec.name, vmax, w_task, instance, tuple(items))
                if ck not in cache:
                    cache[ck] = _algebra_one(items, vmax, spec, w_task, instance)
                zero, diffs, kept = cache[ck]
                for block in blocks:
                    block["all_violating_groups"] += 1
                    block["all_violating_zero_spread"] += zero
                    for (v, d), (x, same_sign) in diffs.items():
                        cell = block["variants"][v][d]
                        cell["worst_diff"] = max(cell["worst_diff"], x)
                        if x <= tol:
                            cell["cancels"] += 1
                        elif not zero:
                            cell["acts_with_spread"] += 1
                            cell["acts"] += 1
                        elif not same_sign:
                            cell["acts_zero_spread"] += 1
                            cell["acts"] += 1
                        else:
                            cell["eps_only_zero_spread"] += 1
                    for f, k in kept.items():
                        block["filters_keep_all_violating"][f] += k
                x_sup = diffs[("ngrpo_sup", "binary")][0]
                ak = (tspec.name, vmax, w_task, instance)
                if ak not in anchors:
                    anchors[ak] = tuple(
                        float(RW.reward_anchor("ngrpo_sup", arm, w, vmax, spec=spec, w_task=w_task,
                                               instance=instance))
                        for arm, w in (("binary", ALGEBRA_DESIGNS["binary"]), ("none", 0.0)))
                inst_cell = (literal.setdefault(str(r["cfg"].get("regime", "")), {})
                             .setdefault(f"{w_task:g}", {})
                             .setdefault(str(instance), {
                                 "anchor_verdict": anchors[ak][0], "anchor_none": anchors[ak][1],
                                 "all_violating_groups": 0, "cancels": 0, "acts": 0,
                                 "worst_diff": 0.0}))
                inst_cell["all_violating_groups"] += 1
                inst_cell["worst_diff"] = max(inst_cell["worst_diff"], x_sup)
                inst_cell["cancels" if x_sup <= tol else "acts"] += 1
    return {"tol": tol, "design_weights": dict(ALGEBRA_DESIGNS), **total,
            "by_regime": dict(sorted(by_regime.items())),
            "literal_anchor_by_instance": {
                reg: {wt: dict(sorted(cells.items(), key=lambda kv: int(kv[0])))
                      for wt, cells in sorted(by_wt.items())}
                for reg, by_wt in sorted(literal.items())}}


def final_answers(runs):
    """What the final evaluation's answers were, counted per run.

    On the capacity task an answer that parses and selects no item is always feasible and scores no
    value. Every capacity remedy run that reached feasibility got there that way, or nearly so, so a
    feasibility count read alone overstates what those runs learned. This table puts the count of
    empty answers beside it. A task whose answers are not selections has no empty answer, and its
    column is left blank. The table is its own file because the capacity snapshot pins every column
    of final_per_run.csv."""
    rows = []
    for r in runs:
        if not r["evals"]:
            continue
        ev = r["evals"][-1]
        tspec = tasks.get(r["cfg"].get("task", "speed"))
        items = ev.get(tspec.EVAL_ITEMS_KEY)
        parsed = None if items is None else [_retuple(x) for x in items if x is not None]
        empty = None
        if parsed is not None and tspec.ITEMS_KEY == "selections":
            empty = sum(1 for x in parsed if tspec.summary_stat([x]) == 0.0)
        rows.append({"regime": r["cfg"].get("regime", ""), "arm": arm_name(r["cfg"]),
                     "seed": r["cfg"]["seed"], "eval_step_final": ev.get("step"),
                     "eval_n_final": ev.get("eval_n"),
                     "eval_unparseable_final": None if items is None else len(items) - len(parsed),
                     "eval_empty_selections_final": empty})
    return rows


def unparseable_collapse(runs):
    """The trace of a run that stops producing a parseable answer and never starts again.

    Six capacity runs end with nothing parseable, and two of them carry a residual, so this is
    not a property only of the arms whose verdict cancels. The paper describes the sequence
    step by step, so the steps are read from the log rather than typed.

    Reported per collapsed run: the step at which every sampled answer stopped parsing, the
    divergence on either side of it, the last step before it that carried any reward spread and
    how much, the size of the group that carried that spread and how many of its members did not
    parse, and the mean residual at the start and just before the collapse.
    """
    rows = []
    for r in runs:
        items_key = tasks.get(r["cfg"].get("task", "speed")).ITEMS_KEY
        evs = [e for e in r["evals"] if e.get("eval_parse_fail_rate") is not None]
        if not evs or evs[-1]["eval_parse_fail_rate"] < 1.0:
            continue
        train = [t for t in r["train"] if t.get("step") is not None]
        by_step = {t["step"]: t for t in train}
        # The step after which the run never parses again, which is not the first step at which
        # it fails to parse. Three of the six collapsed runs spike to a parse-failure rate of 1.0
        # and recover: the no-penalty seed at the hard level first hits 1.0 at step 76, is back to
        # 0.0 by step 86, and only stops for good at step 125. Taking the first occurrence
        # anchored every other field in the row on a step the run recovered from. The one run the
        # paper quotes happens to collapse on its first failure, so reading that row by hand
        # could not catch this.
        rates = [(t["step"], t.get("parse_fail_rate")) for t in train
                 if t.get("parse_fail_rate") is not None]
        if not rates or rates[-1][1] != 1.0:
            continue
        last_below = max((st for st, v in rates if v != 1.0), default=None)
        collapse = rates[0][0] if last_below is None else last_below + 1
        if collapse is None:
            continue

        def group_spread(g):
            return float(np.ptp(np.asarray(g["rewards"], dtype=np.float64)))

        def spread_of(t):
            return max((group_spread(g) for g in t.get("groups", [])), default=0.0)

        def widest_group(t):
            """Return the group with the largest reward range at this step, or None when it has none."""
            groups = t.get("groups", [])
            return max(groups, key=group_spread) if groups else None

        before = [t for t in train if t["step"] < collapse]
        spreads = [(t["step"], spread_of(t)) for t in before]
        live = [(st, sp) for st, sp in spreads if sp > ZERO_VAR_TOL]
        # These are the members of the group that carried the last spread, so the paper can say
        # how many of its completions parsed at that step. The count is None when no step before the
        # collapse had any spread, and also when the log predates the per-group item list.
        spike_group = widest_group(by_step[live[-1][0]]) if live else None
        members = None if spike_group is None else spike_group.get(items_key)
        # The first step at which no group has any spread. A later step may re-acquire a little,
        # and on the residual seed that is exactly what ends the run, so this is deliberately the
        # first such step and not the start of an uninterrupted flat run.
        flat = next((st for st, sp in spreads if sp <= ZERO_VAR_TOL), None)
        rows.append({
            "regime": r["cfg"]["regime"], "arm": arm_name(r["cfg"]),
            "seed": int(r["cfg"]["seed"]),
            "collapse_step": collapse,
            "kl_before": (by_step.get(collapse - 1) or {}).get("kl_seq_mean"),
            "kl_after": by_step[collapse].get("kl_seq_mean"),
            "last_spread_step": live[-1][0] if live else None,
            "last_spread_value": live[-1][1] if live else None,
            "last_spread_group_members": None if members is None else len(members),
            "last_spread_group_unparseable": (None if members is None
                                              else sum(m is None for m in members)),
            "first_flat_step": flat,
            "residual_first_step": (train[0].get("mean_residual") if train else None),
            "residual_before_collapse": (by_step.get(collapse - 1) or {}).get("mean_residual"),
        })
    return rows


def fallback_group_counts(runs):
    """How the unparseable fallback sat beside the parsed members of each logged group, per run.

    The fallback prices an unparseable member at the task floor plus the arm's penalty on a stand-in
    built for the group's worst parsed residual, so its raw reward depends on its siblings. Two
    objections to that rule are counted here rather than argued. The first said that in an
    arm with no task term a group whose parsed members are all feasible gives the unparseable member
    the same reward as the rest, so the group has zero spread. The second said that an unparseable
    member can earn more than a parsed violator when the stand-in falls short of the target.

    Read from the logged items and rewards, which counterfactual_checks has already matched against
    the recomputed rewards. Per run:
      groups_with_unparseable                    groups holding at least one unparseable member
      all_feasible_with_unparseable              of those, groups with at least one parsed member and
                                                 every parsed member feasible
      all_feasible_with_unparseable_zero_spread  of those, groups whose rewards all tie
      unparseable_outranks_parsed_violator       groups in which some unparseable member's reward is
                                                 strictly above some parsed violator's reward
      unparseable_beside_residual_above_reach    groups holding an unparseable member and a parsed
                                                 member whose residual exceeds stand_in_reach
      stand_in_reach                             the largest residual the stand-in can reach at this
                                                 level. On the capacity task it is the take-all
                                                 residual of the fixed instance. The speed stand-in
                                                 is unbounded, and the field is then None.
    The row carries the penalty arm and the task weight from the config, so a generator can select
    the arms it reports without parsing arm names.
    """
    FIELDS = ("groups_with_unparseable", "all_feasible_with_unparseable",
              "all_feasible_with_unparseable_zero_spread",
              "unparseable_outranks_parsed_violator", "unparseable_beside_residual_above_reach")
    rows = []
    for r in runs:
        vmax = float(r["cfg"]["vmax"])
        tspec = tasks.get(r["cfg"].get("task", "speed"))
        items_key = tspec.ITEMS_KEY
        reach = float(tspec.residual(tspec.synthesize(float("inf"), vmax), vmax))
        c = {k: 0 for k in FIELDS}
        for rec in r["train"]:
            for g in rec["groups"]:
                items = [_retuple(x) for x in (g[items_key] if items_key in g else g["speeds"])]
                rw = np.asarray(g["rewards"], dtype=np.float64)
                unp = [i for i, x in enumerate(items) if x is None]
                if not unp:
                    continue
                c["groups_with_unparseable"] += 1
                parsed = [i for i, x in enumerate(items) if x is not None]
                feas = {i: bool(tspec.verdict(items[i], vmax)) for i in parsed}
                if parsed and all(feas.values()):
                    c["all_feasible_with_unparseable"] += 1
                    c["all_feasible_with_unparseable_zero_spread"] += int(np.ptp(rw) <= ZERO_VAR_TOL)
                violators = [i for i in parsed if not feas[i]]
                if violators and rw[unp].max() > rw[violators].min():
                    c["unparseable_outranks_parsed_violator"] += 1
                if parsed and max(float(tspec.residual(items[i], vmax)) for i in parsed) > reach:
                    c["unparseable_beside_residual_above_reach"] += 1
        rows.append({"regime": r["cfg"].get("regime", ""), "arm": arm_name(r["cfg"]),
                     "seed": int(r["cfg"]["seed"]), "penalty": r["cfg"]["arm"],
                     "w_task": float(r["cfg"].get("w_task", 1.0)),
                     "stand_in_reach": reach if np.isfinite(reach) else None, **c})
    return rows


def verdict_none_agreement(runs):
    """How far a trained added-verdict run and a trained no-penalty run stay the same run.

    The closed-form identity says the verdict cancels inside an all-violating group. The
    strongest trained version of that is two runs that make the same choices step after step,
    and the speed task produced one seed where they matched for all 300 steps. A level whose
    all-violating fraction is below one will not do that, because a mixed group appears from
    time to time and the verdict acts inside it.

    So this measures rather than asserts: per regime and seed, the number of steps on which the
    two runs sampled identical answers, and the first step at which they differed. Both numbers
    are reported whatever they are, including the case where they never diverge.
    """
    by = {}
    for r in runs:
        name = arm_name(r["cfg"])
        if name in ("binary", "none"):
            by.setdefault((r["cfg"]["regime"], int(r["cfg"]["seed"])), {})[name] = r
    rows = []
    for (regime, seed), pair in sorted(by.items()):
        if set(pair) != {"binary", "none"}:
            continue
        a, b = pair["binary"], pair["none"]
        key = tasks.get(a["cfg"].get("task", "speed")).ITEMS_KEY

        def picks(run, rec):
            return [g[key] if key in g else g["speeds"] for g in rec["groups"]]

        n = min(len(a["train"]), len(b["train"]))
        same, first_diff, worst_adv = 0, None, 0.0
        for i in range(n):
            if picks(a, a["train"][i]) == picks(b, b["train"][i]):
                same += 1
                # While the two runs are still sampling the same answers, any difference in their
                # advantages is the verdict failing to cancel. Reporting the largest one says how
                # exact the cancellation is on a trained run, rather than on a sampled group.
                if first_diff is None:
                    for ga, gb in zip(a["train"][i]["groups"], b["train"][i]["groups"]):
                        d = np.abs(np.asarray(ga["adv"], dtype=np.float64)
                                   - np.asarray(gb["adv"], dtype=np.float64))
                        worst_adv = max(worst_adv, float(d.max()) if d.size else 0.0)
            elif first_diff is None:
                first_diff = a["train"][i].get("step", i)
        # Groups in the verdict run that had a feasible member. Where this is zero the verdict was
        # constant in every group of the run, so it never entered a single update.
        mixed = sum(1 for rec in a["train"] for g in rec["groups"] if not g["all_violating"])
        rows.append({"regime": regime, "seed": seed, "steps_compared": n,
                     "identical_steps": same, "first_divergent_step": first_diff,
                     "mixed_groups_in_verdict_run": mixed,
                     "max_adv_diff_while_identical": worst_adv})
    return rows


# This file holds the yoked controls' schedules. The analysis reads the same committed file the
# trainer read, and the tests point this constant at a temporary file.
KEEP_SCHEDULES_PATH = Path(__file__).resolve().parent / "configs" / "keep_schedules.json"


def load_keep_schedule(schedule_id, path=None):
    """Return a yoked control's scheduled kept prompts as one frozenset per step.

    A missing id raises, because a control whose schedule cannot be read cannot be checked."""
    path = Path(path) if path is not None else KEEP_SCHEDULES_PATH
    entries = json.loads(path.read_text()).get("schedules", {})
    if schedule_id not in entries:
        raise ValueError(f"keep_schedule {schedule_id!r} is not an entry of {path.name}")
    return [frozenset(s) for s in entries[schedule_id]["kept_sets"]]


def counterfactual_checks(runs, tol=1e-9):
    """P1, P2, and trainer self-consistency on every logged group.

    A yoked control keeps exactly its scheduled prompts, so for such a run the expected kept flag is
    membership in the scheduled set and not the output of a filter."""
    stats = defaultdict(int)
    worst = defaultdict(float)
    for r in runs:
        vmax = float(r["cfg"]["vmax"])
        tspec = tasks.get(r["cfg"].get("task", "speed"))
        spec = None if tspec.name == "speed" else tspec   # speed keeps the frozen scalar path
        items_key = tspec.ITEMS_KEY
        variant = r["cfg"].get("adv_variant", "grpo")
        group_filter = r["cfg"].get("group_filter", "none")
        w_task = float(r["cfg"].get("w_task", 1.0))
        sid = r["cfg"].get("keep_schedule")
        scheduled = load_keep_schedule(sid) if sid is not None else None
        for rec in r["train"]:
            if scheduled is not None:
                i = int(rec["step"]) - 1
                planned = scheduled[i] if 0 <= i < len(scheduled) else None
                if planned is None or rec.get("keep_schedule_n") != len(planned):
                    stats["keep_schedule_n_mismatch"] += 1
            for gi, g in enumerate(rec["groups"]):
                speeds = g[items_key] if items_key in g else g["speeds"]
                # JSON gives lists back. A capacity selection becomes the tuple the scoring functions
                # index, and a number (the speed task, and the bounded speed task of Block A') passes
                # through unchanged, so the frozen scalar path of the speed task is untouched.
                speeds = [_retuple(x) for x in speeds]
                # trainer self-consistency: logged advantages equal the recomputed ones for its own
                # arm, under its own advantage variant and anchor. At the default variant this is
                # the frozen advantage() applied to the same rewards.
                own_r = RW.group_rewards(speeds, r["cfg"]["arm"], float(r["cfg"]["w"]), vmax,
                                         spec=spec, w_task=w_task)
                prompt_index = g.get("prompt_index", gi)
                if prompt_index != gi:
                    stats["prompt_index_mismatch"] += 1
                verdicts_now = RW.group_verdicts(speeds, vmax, spec=spec)
                if variant == "grpo_fixedref":
                    # ICML 2027 (2026-11-06). The fixed-reference variant reads the group's verdicts as
                    # well as its rewards, so it is recomputed from the logged verdicts, as
                    # replay_bounded_scale.py does. A logged verdict that disagrees with the
                    # recomputed one is counted below, because an advantage that reproduces from a
                    # wrong verdict would otherwise pass this check.
                    if "verdicts" not in g:
                        raise ValueError(f'{r["dir"].name}: a grpo_fixedref run must log each group\'s verdicts')
                    anchor = None
                    own = RW.advantage_fixedref(own_r, g["verdicts"], float(r["cfg"]["fixedref_c"]))
                else:
                    anchor = RW.reward_anchor(variant, r["cfg"]["arm"], float(r["cfg"]["w"]), vmax,
                                              spec=spec, w_task=w_task,
                                              instance=tspec.instance_for(prompt_index))
                    own = RW.advantage_variant(own_r, variant, anchor)
                if "verdicts" in g and [bool(x) for x in g["verdicts"]] != verdicts_now:
                    stats["verdict_mismatch"] += 1
                if "anchor" in g and g["anchor"] != anchor:
                    stats["anchor_mismatch"] += 1
                if scheduled is not None:
                    want_kept = planned is not None and prompt_index in planned
                else:
                    want_kept = RW.keep_group(group_filter, own_r, verdicts_now)
                if bool(g.get("kept", True)) != want_kept:
                    stats["kept_mismatch"] += 1
                d = float(np.max(np.abs(own - np.array(g["adv"]))))
                worst["self"] = max(worst["self"], d)
                stats["groups"] += 1
                armspecs = CAPACITY_ARM_SPECS if spec is not None else ARM_SPECS
                cf = {name: RW.advantage(RW.group_rewards(speeds, arm, w, vmax, spec=spec, w_task=t))
                      for name, (arm, w, t) in armspecs.items()}
                verdicts = {not tspec.verdict(s, vmax) for s in speeds}
                if g["all_violating"]:
                    stats["all_violating"] += 1
                    d1 = max(float(np.max(np.abs(cf["binary"] - cf["none"]))), float(np.max(np.abs(cf["binary_1000x"] - cf["none"]))))
                    worst["P1"] = max(worst["P1"], d1)
                    if d1 > tol:
                        stats["P1_fail"] += 1
                if len(verdicts) == 1:
                    stats["constant_verdict"] += 1
                    d2 = float(np.max(np.abs(cf["binary"] - cf["binary_1000x"])))
                    worst["P2"] = max(worst["P2"], d2)
                    if d2 > tol:
                        stats["P2_fail"] += 1
                else:
                    stats["mixed"] += 1
                    feas = np.array([tspec.verdict(s, vmax) for s in speeds])
                    if cf["binary_1000x"][feas].min() <= cf["binary_1000x"][~feas].max():
                        stats["P2_boundary_fail"] += 1
                    if (cf["binary"][~feas] > cf["binary"][feas].max()).any():
                        stats["binary_w5_violator_outranks_feasible"] += 1
                if g["has_violator"]:
                    stats["has_violator"] += 1
                    viol = np.array([not tspec.verdict(s, vmax) for s in speeds])
                    for name in armspecs:
                        if (cf[name][viol] > 0).any():
                            stats[f"{name}_reinforces_violator"] += 1
    return dict(stats), dict(worst)


def _mean_task_term(ev: dict, tspec, vmax) -> "float | None":
    """Mean task reward over the completions that parsed, at one eval.

    The paper's boundary claim is about how much task value a design ends up with, not about
    how many items it selected or how fast it drove. Item count and speed are task-specific and
    are not comparable to an optimum; the task term is, because every task defines it against
    its own reachable best. So this is the one column that means the same thing in both tables.

    Unparseable completions are excluded rather than scored at the floor, because a mean that
    silently mixes achieved value with a failure constant cannot be read. `final_table` always
    emits `eval_parse_fail_rate_final` beside this column, so the denominator is never missing.
    """
    items = ev.get(tspec.EVAL_ITEMS_KEY)
    if not items:
        return None
    vals = [tspec.task_term(_retuple(x), vmax) for x in items if x is not None]
    return float(np.mean(vals)) if vals else None


def _retuple(x):
    """Round-trip a parsed structure back from JSON. A capacity selection is a tuple that json
    wrote as a list, and the scoring functions index it positionally, so a list works; a speed
    value is a bare number and passes through."""
    return tuple(x) if isinstance(x, list) else x


def _kl_at_arrival(r) -> dict:
    """The divergence at the step where the evaluated violation rate first reaches zero.

    This is what "the cost of reaching feasibility" means, and it is not the divergence at the
    end of training. A policy that arrives at step 5 and then trains for 295 more steps keeps
    moving, and the end-of-training number charges it for all of that.

    Returns no arrival step for a run that never reaches zero, rather than a number that would
    quietly stand in for one.
    """
    evals = [e for e in r["evals"] if e.get("eval_violation_rate") is not None]
    arrival = next((e["step"] for e in evals if e["eval_violation_rate"] == 0.0), None)
    if arrival is None:
        return {"arrival_step": None, "kl_at_arrival": None}
    # the training record at or just before that evaluation
    prior = [t for t in r["train"] if t["step"] <= arrival and t.get("kl_seq_mean") is not None]
    return {"arrival_step": arrival,
            "kl_at_arrival": prior[-1]["kl_seq_mean"] if prior else None}


def final_table(runs, out: Path):
    rows = []
    for r in runs:
        if not r["train"]:
            print(f'[skip] {r["dir"].name}: no training steps logged', file=sys.stderr)
            continue
        last = r["train"][-1]
        ev_last = r["evals"][-1] if r["evals"] else {}
        ev_first = r["evals"][0] if r["evals"] else {}
        tspec = tasks.get(r["cfg"].get("task", "speed"))
        rows.append({
            "regime": r["cfg"].get("regime", ""), "arm": arm_name(r["cfg"]), "seed": r["cfg"]["seed"],
            "vmax": r["cfg"]["vmax"], "steps": r["cfg"]["steps"],
            "eval_violation_rate_step0": ev_first.get("eval_violation_rate"),
            # The parse-failure rate before any training. Four capacity runs end unable to
            # produce a parseable answer, and the claim that this is caused by training rather
            # than by too short a generation budget rests on this number being small.
            "eval_parse_fail_rate_step0": ev_first.get("eval_parse_fail_rate"),
            "eval_violation_rate_final": ev_last.get("eval_violation_rate"),
            "eval_mean_residual_step0": ev_first.get("eval_mean_residual"),
            "eval_mean_residual_final": ev_last.get("eval_mean_residual"),
            "eval_parse_fail_rate_final": ev_last.get("eval_parse_fail_rate"),
            # The pricing section says it reads the divergence where the violation rate first
            # reaches zero. It was reading the divergence at the end of training, which is a
            # different and larger number, because the policy keeps moving after it arrives.
            **_kl_at_arrival(r),
            "eval_task_scalar_final": ev_last.get(tspec.EVAL_SCALAR_KEY),
            "eval_task_term_step0": _mean_task_term(ev_first, tspec, r["cfg"]["vmax"]),
            "eval_task_term_final": _mean_task_term(ev_last, tspec, r["cfg"]["vmax"]),
            "train_all_violating_fraction_step1": r["train"][0]["all_violating_fraction"],
            "train_all_violating_fraction_final": last["all_violating_fraction"],
            "kl_seq_mean_final": last.get("kl_seq_mean"),
            "s_per_step_median": r["prov"].get("s_per_step_median"),
            "device_name": r["prov"].get("device_name"), "git_sha": r["prov"].get("git_sha"),
        })
    if not rows:
        return None, None
    with (out / "final_per_run.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # seed aggregate
    agg = defaultdict(list)
    for row in rows:
        agg[(row["regime"], row["arm"])].append(row)
    arows = []
    for (regime, arm), rs in sorted(agg.items()):
        def ms(key):
            v = np.array([x[key] for x in rs if x[key] is not None], dtype=float)
            return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)) if v.size else (None, None, 0)
        vr = ms("eval_violation_rate_final"); res = ms("eval_mean_residual_final"); kl = ms("kl_seq_mean_final")
        tt = ms("eval_task_term_final"); pf = ms("eval_parse_fail_rate_final")
        ka = ms("kl_at_arrival"); ar = ms("arrival_step")
        arows.append({"regime": regime, "arm": arm, "n_seeds": vr[2],
                      "violation_rate_final_mean": vr[0], "violation_rate_final_std": vr[1],
                      "mean_residual_final_mean": res[0], "mean_residual_final_std": res[1],
                      "task_term_final_mean": tt[0], "task_term_final_std": tt[1],
                      "parse_fail_final_mean": pf[0], "parse_fail_final_std": pf[1],
                      "kl_at_arrival_mean": ka[0], "kl_at_arrival_std": ka[1],
                      "arrival_seeds": ka[2], "arrival_step_mean": ar[0],
                      "kl_final_mean": kl[0], "kl_final_std": kl[1]})
    with (out / "final_by_arm.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arows[0].keys())); w.writeheader(); w.writerows(arows)
    return rows, arows


def curves(runs, out: Path):
    keys = ["violation_rate", "all_violating_fraction", "mean_residual", "kl_seq_mean",
            "violator_mean_adv", "frac_violators_pos_adv", "group_reward_std_mean"]
    with (out / "curves_train.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["regime", "arm", "seed", "step"] + keys)
        for r in runs:
            for rec in r["train"]:
                w.writerow([r["cfg"].get("regime", ""), arm_name(r["cfg"]), r["cfg"]["seed"], rec["step"]] + [rec.get(k) for k in keys])
    with (out / "curves_eval.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["regime", "arm", "seed", "step", "eval_violation_rate", "eval_mean_residual", "eval_parse_fail_rate"])
        for r in runs:
            for rec in r["evals"]:
                w.writerow([r["cfg"].get("regime", ""), arm_name(r["cfg"]), r["cfg"]["seed"], rec["step"],
                            rec["eval_violation_rate"], rec["eval_mean_residual"], rec["eval_parse_fail_rate"]])


# The order arms appear in every figure and table, so a reader compares the same lines
# in the same positions across the paper.
ARM_ORDER = ["graded", "binary", "binary_1000x", "none", "verdict_only", "graded_no_task"]

ARM_STYLE = {
    "graded":         dict(color="#1b6ca8", ls="-",  label="residual $+$ task"),
    "binary":         dict(color="#c0392b", ls="-",  label="verdict $+$ task"),
    "binary_1000x":   dict(color="#e67e22", ls="--", label=r"verdict $\times 1000$ $+$ task"),
    "none":           dict(color="#7f8c8d", ls=":",  label="task only"),
    "verdict_only":   dict(color="#8e44ad", ls="-.", label="verdict only"),
    "graded_no_task": dict(color="#16a085", ls="--", label="residual only"),
}


NAMED_LABELS = {"binary_5x": r"verdict $\times 5$ $+$ task",
                "graded_below": r"residual $+$ task, below the ordering threshold",
                "graded_matched": r"residual $+$ task, matched weight",
                "graded_below_cap": r"residual $+$ task, below the ordering threshold on four instances"}
VARIANT_LABEL = {"drgrpo": "Dr. GRPO", "ngrpo_feasible": "NGRPO, feasible anchor",
                 "ngrpo_sup": "NGRPO, literal anchor", "grpo_fixedref": "fixed reference"}
# The table's names must be the paper's names. "verdict filter" and "spread filter" left a reader
# guessing which row the prose's verifier filter and identical-reward filter were.
FILTER_LABEL = {"verdict_mixed": "verifier filter", "reward_spread": "identical-reward filter"}


def compose_arm_label(cfg) -> str:
    """The label of an arm outside ARM_STYLE: its reward design's label, then each remedy flag.

    The plain copy also clears the keep schedule. Otherwise a yoked control's base name would keep its
    draw suffix, miss both label tables, and print as its raw name."""
    plain = dict(cfg, adv_variant="grpo", group_filter="none", kl_beta=0.0, keep_schedule=None)
    base = arm_name(plain)
    parts = [ARM_STYLE[base]["label"] if base in ARM_STYLE else NAMED_LABELS.get(base, base)]
    if cfg.get("adv_variant", "grpo") != "grpo":
        parts.append(VARIANT_LABEL[cfg["adv_variant"]])
    if cfg.get("group_filter", "none") != "none":
        parts.append(FILTER_LABEL[cfg["group_filter"]])
    if float(cfg.get("kl_beta", 0.0)) > 0:
        parts.append(f"KL $\\beta = {float(cfg['kl_beta']):g}$")
    draw = keep_schedule_draw(cfg)
    if draw is not None:
        parts.append(f"yoked control, draw {draw}")
    return ", ".join(parts)


# The phrase the paper uses for each regime. The speed task calls its regimes limits and the
# capacity task calls them levels, and a figure title that said "regime A" matched neither.
REGIME_WORDS = {"A": "hard limit", "B": "easy limit", "H": "hard level", "E": "easy level"}


def regime_title(regime, runs) -> str:
    """The title of a regime's panel: the paper's phrase for it, then the limit its runs share.

    The limit is read from the run configs, so the title cannot disagree with the runs it
    describes. The speed task stores an envelope and the capacity task a fraction of each
    instance's total weight, and each gets the form the paper uses. When the runs at this regime
    do not agree on one limit, the phrase stands alone, because one number would misdescribe
    some of them.
    """
    word = REGIME_WORDS.get(regime, str(regime))
    word = word[:1].upper() + word[1:]
    at = [r for r in runs if r["cfg"].get("regime", "") == regime]
    limits = {(r["cfg"].get("task", "speed"), r["cfg"].get("vmax")) for r in at}
    if len(limits) != 1 or None in {v for _, v in limits}:
        return word
    (task, v), = limits
    v = float(v)
    # The bounded speed task of Block A' (2026-11-06) stores an envelope as the speed task does, so
    # the form is chosen by what the task parses and not by its name.
    if tasks.get(task).ITEMS_KEY == "speeds":
        return f"{word}, $v_{{\\max}} = {v:g}$"
    return f"{word}, capacity fraction = {v:g}"


def _plain_label(s: str) -> str:
    """A label's plain-text form, for the figures' own text engine.

    The arm labels are LaTeX because the table generators typeset them. A figure renders its
    legend and axis text itself, so it gets the same words with the multiplication sign and the
    Greek letter written directly and the math delimiters removed.
    """
    return (s.replace("$+$", "+").replace("\\times", "\u00d7").replace("\\beta", "\u03b2")
            .replace("$", ""))


def arm_labels(runs=()) -> dict:
    labels = {arm: st["label"] for arm, st in ARM_STYLE.items()}
    for r in runs:
        name = arm_name(r["cfg"])
        if name not in labels:
            labels[name] = compose_arm_label(r["cfg"])
    return labels


def write_arm_labels(out: Path, runs=()) -> None:
    """Publish the arm labels as an artifact of the analysis.

    The figures and the paper's tables named the same arms differently, because each carried its
    own dictionary. A reader comparing a table row to a curve had to work out that "graded
    residual" and "residual + task" were the same design. The label belongs to the arm, not to
    whichever program is drawing it, so it is written once here and read by the table generator.
    A remedy arm's label is composed from its design's label and its flags, and it is written only
    when such an arm is present.
    """
    json.dump(arm_labels(runs), (out / "arm_labels.json").open("w"), indent=2, sort_keys=True)


SEED_LINESTYLES = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)))


def seed_evaluation_series(run):
    """Keep each seed's evaluation values and its parseable denominator together.

    An empty selection has zero task value. An evaluation with no parseable answer has
    undefined task value, so its plotted line must break rather than fall to zero.
    """
    spec = tasks.get(run["cfg"].get("task", "speed"))
    records = []
    for ev in run["evals"]:
        items = ev.get(spec.EVAL_ITEMS_KEY)
        if items is None:
            raise ValueError("seed figures require the recorded evaluation answers")
        total = len(items)
        parsed = sum(item is not None for item in items)
        if total == 0 or not np.isclose(ev["eval_parse_fail_rate"], 1 - parsed / total):
            raise ValueError("the recorded parse-failure rate disagrees with the evaluation answers")
        value = _mean_task_term(ev, spec, run["cfg"]["vmax"])
        records.append({
            "step": ev["step"], "violation": ev["eval_violation_rate"],
            "parse_failure": ev["eval_parse_fail_rate"], "task_value": value,
            "parseable_completions": parsed, "evaluation_completions": total,
        })
    return records


def _save_figure(fig, path):
    """A rebuild preserves identical PDF bytes when the plotted evidence has not changed."""
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})


def _capacity_seed_figure(runs, regime, out, plt):
    """Show the individual outcomes that a mean and deviation can hide."""
    from matplotlib.lines import Line2D

    arms = [arm for arm in ARM_ORDER if any(arm_name(r["cfg"]) == arm for r in runs)]
    fig, axes = plt.subplots(len(arms), 3, figsize=(7.5, 7.9), sharex=True, squeeze=False)
    columns = (("violation", "Evaluation violation"),
               ("parse_failure", "Parse-failure rate"),
               ("task_value", "Value / feasible optimum"))
    records_out = []
    for row, arm in enumerate(arms):
        arm_runs = sorted((r for r in runs if arm_name(r["cfg"]) == arm),
                          key=lambda r: r["cfg"]["seed"])
        style = ARM_STYLE[arm]
        for run in arm_runs:
            seed = run["cfg"]["seed"]
            records = seed_evaluation_series(run)
            records_out.extend(dict(regime=regime, arm=arm, seed=seed, **rec) for rec in records)
            for col, (key, _) in enumerate(columns):
                ax = axes[row, col]
                xs = [r["step"] for r in records]
                ys = [np.nan if r[key] is None else r[key] for r in records]
                ax.plot(xs, ys, color=style["color"], ls=SEED_LINESTYLES[seed % len(SEED_LINESTYLES)],
                        lw=1.35, alpha=0.85, zorder=2 + seed)
                # The endpoint makes an overlapping final trajectory visible as a real observation.
                if np.isfinite(ys[-1]):
                    ax.plot(xs[-1], ys[-1], marker="o", ms=2.8, color=style["color"], zorder=10)
        for col, (key, title) in enumerate(columns):
            ax = axes[row, col]
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.9", lw=0.6)
            ax.tick_params(labelsize=10)
            ax.set_xticks([0, 100, 200, 300])
            if key != "task_value":
                ax.set_ylim(-0.04, 1.04)
                ax.set_yticks([0, 0.5, 1])
            else:
                # Infeasible answers can exceed the feasible optimum, so the axis is never clipped at one.
                ax.set_ylim(bottom=-0.04)
                ax.axhline(1, color="0.65", lw=0.8, ls=":", zorder=1)
            if row == 0:
                ax.set_title(title, fontsize=11, pad=9)
            if row == len(arms) - 1:
                ax.set_xlabel("Optimizer step", fontsize=11)
        label = _plain_label(style["label"])
        if arm == "binary_1000x":
            # The heavy verdict's label is the widest, so it breaks onto two lines.
            label = label.replace(" + ", "\n+ ")
        axes[row, 0].set_ylabel(f"{label}\n({len(arm_runs)} seeds)", fontsize=10, labelpad=9)
    # Every task-value panel uses the same scale within this level.
    task_top = max(ax.get_ylim()[1] for ax in axes[:, 2])
    for ax in axes[:, 2]:
        ax.set_ylim(-0.04, task_top)
    seeds = sorted({r["cfg"]["seed"] for r in runs})
    handles = [Line2D([0], [0], color="0.25", lw=1.6,
                       ls=SEED_LINESTYLES[s % len(SEED_LINESTYLES)], label=f"Seed {s}") for s in seeds]
    fig.legend(handles=handles, loc="upper center", ncol=len(seeds), frameon=False,
               fontsize=10, bbox_to_anchor=(0.56, 1.02))
    fig.tight_layout(h_pad=0.7, w_pad=1.3)
    _save_figure(fig, out / f"seed_evaluation_regime{regime}.pdf")
    plt.close(fig)
    with (out / f"seed_evaluation_regime{regime}.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records_out[0]))
        writer.writeheader()
        writer.writerows(records_out)


def _seed_kl_figure(runs, regime, out, plt):
    """Separate reward designs and mark the direction of each seed's training trajectory."""
    from matplotlib.lines import Line2D

    labels = arm_labels(runs)
    arms = [arm for arm in ARM_ORDER if any(arm_name(r["cfg"]) == arm for r in runs)]
    arms += sorted({arm_name(r["cfg"]) for r in runs} - set(arms))
    ncols = 2 if len(arms) <= 4 else 3
    nrows = (len(arms) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 2.05 * nrows), squeeze=False,
                             sharex=True, sharey=True)
    for ax, arm in zip(axes.flat, arms):
        arm_runs = sorted((r for r in runs if arm_name(r["cfg"]) == arm),
                          key=lambda r: r["cfg"]["seed"])
        style = ARM_STYLE.get(arm, {"color": "0.35"})
        for run in arm_runs:
            seed = run["cfg"]["seed"]
            xs = [rec["kl_seq_mean"] for rec in run["train"]]
            ys = [rec["violation_rate"] for rec in run["train"]]
            ax.plot(xs, ys, color=style["color"], ls=SEED_LINESTYLES[seed % len(SEED_LINESTYLES)],
                    lw=1.2, alpha=0.85)
            ax.plot(xs[0], ys[0], marker="o", ms=4, mfc="white", mec=style["color"], zorder=4)
            ax.plot(xs[-1], ys[-1], marker="s", ms=4, color=style["color"], zorder=4)
        ax.set_title(_plain_label(labels[arm]), fontsize=10)
        ax.set_ylim(-0.04, 1.04)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", color="0.9", lw=0.6)
    for ax in list(axes.flat)[len(arms):]:
        ax.set_visible(False)
    fig.supxlabel("Reference log-ratio (nats per completion, token sum)", fontsize=10)
    fig.supylabel("Training-batch violation rate", fontsize=10)
    seeds = sorted({run["cfg"]["seed"] for run in runs})
    handles = [Line2D([0], [0], color="0.25", lw=1.4,
                       ls=SEED_LINESTYLES[s % len(SEED_LINESTYLES)], label=f"Seed {s}") for s in seeds]
    handles += [Line2D([0], [0], color="0.25", marker="o", mfc="white", ls="", label="First step"),
                Line2D([0], [0], color="0.25", marker="s", ls="", label="Final step")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.53, 1.02))
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.95))
    _save_figure(fig, out / f"violation_vs_kl_regime{regime}.pdf")
    plt.close(fig)


def figures(runs, out: Path):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    try:
        import matplotlib
        matplotlib.use("Agg")
        # Venues reject a PDF containing Type 3 fonts, and matplotlib embeds them by default.
        # 42 selects TrueType. The paper's build gate checks the final PDF for Type 3 and it
        # caught this, which is the only reason the figures were not shipped with them.
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib is optional
        print(f"[figures] skipped: {e}", file=sys.stderr)
        return
    labels = arm_labels(runs)
    regimes = sorted({r["cfg"].get("regime", "") for r in runs})
    for regime in regimes:
        for key, ylabel in (("eval_violation_rate", "violation rate (eval)"), ("eval_mean_residual", "mean residual (eval)")):
            fig, ax = plt.subplots(figsize=(3.3, 2.3))
            by_arm = defaultdict(list)
            for r in runs:
                if r["cfg"].get("regime", "") != regime:
                    continue
                xs = [e["step"] for e in r["evals"]]; ys = [e[key] if e[key] is not None else np.nan for e in r["evals"]]
                by_arm[arm_name(r["cfg"])].append((xs, ys))
            for arm in [a for a in ARM_ORDER if a in by_arm] + [a for a in sorted(by_arm) if a not in ARM_ORDER]:
                series = by_arm[arm]
                Y = np.array([q[1] for q in series], dtype=float); X = series[0][0]
                m = np.nanmean(Y, axis=0); sd = np.nanstd(Y, axis=0, ddof=1)
                st = ARM_STYLE.get(arm, dict(label=labels.get(arm, arm)))
                ax.plot(X, m, color=st.get("color"), ls=st.get("ls", "-"),
                        label=f'{_plain_label(st.get("label", arm))} (n={len(series)})', lw=1.6)
                lower = np.maximum(m - sd, 0)
                upper = np.minimum(m + sd, 1) if "violation" in key else m + sd
                ax.fill_between(X, lower, upper, color=st.get("color"), alpha=0.15, lw=0)
            ax.set_xlabel("Optimizer step"); ax.set_ylabel(ylabel)
            ax.set_title(regime_title(regime, runs), fontsize=8, pad=4)
            ax.set_ylim(-0.03, 1.03) if "violation" in key else None
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=6.5, frameon=False, loc="best")
            fig.tight_layout(); _save_figure(fig, out / f"{key}_regime{regime}.pdf"); plt.close(fig)
        regime_runs = [run for run in runs if run["cfg"].get("regime", "") == regime]
        _seed_kl_figure(regime_runs, regime, out, plt)
        if all(run["cfg"].get("task") == "capacity" and arm_name(run["cfg"]) in ARM_STYLE
               for run in regime_runs):
            _capacity_seed_figure(regime_runs, regime, out, plt)


def main_fingerprint(argv) -> int:
    """Record one fingerprint in equivalence.json, with the outcome computed from the three runs.

    Usage: python analyze.py fingerprint --name F1 --reference <run> --new <run>
               --rerun-at-base <run> --steps 300 --base-sha <sha> --new-sha <sha>
    Every run path is repo-relative. A stop outcome is recorded too, and it blocks the waiver."""
    ap = argparse.ArgumentParser(prog="analyze.py fingerprint")
    ap.add_argument("--name", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--rerun-at-base", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--base-sha", required=True)
    ap.add_argument("--new-sha", required=True)
    ap.add_argument("--equivalence", default=str(EQUIVALENCE_PATH))
    ap.add_argument("--allow-drift", action="store_true",
                    help="admit drift_pass, as the ledger entry of 2026-09-14 allows for the capacity task")
    a = ap.parse_args(argv)
    paths = {"reference": a.reference, "new": a.new, "rerun_at_base": a.rerun_at_base}
    bad = [k for k, p in paths.items() if not _repo_relative(p)]
    if bad:
        print(f"REFUSED: {', '.join(bad)} must be repo-relative paths", file=sys.stderr)
        return 2
    runs = {}
    for k, p in paths.items():
        d = _REPO_ROOT / p
        if not (d / "DONE").exists():
            print(f"REFUSED: {p} has no DONE marker", file=sys.stderr)
            return 2
        runs[k] = _load_run(d)
    if runs["new"]["prov"].get("git_sha") != a.new_sha:
        print("REFUSED: the new run was not made at --new-sha", file=sys.stderr)
        return 2
    digest = runs["new"]["prov"].get("training_sources_digest")
    if not digest:
        print("REFUSED: the new run carries no training_sources_digest", file=sys.stderr)
        return 2
    if training_sources_between(runs["rerun_at_base"]["prov"].get("git_sha", ""), a.base_sha) != []:
        print("REFUSED: the rerun was not made at a commit whose training sources equal --base-sha",
              file=sys.stderr)
        return 2
    try:
        outcome = fingerprint_outcome(runs["reference"], runs["new"], runs["rerun_at_base"], a.steps,
                                      a.allow_drift)
        path = Path(a.equivalence)
        waivers = load_equivalence(path)
    except EquivalenceError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    data = json.loads(path.read_text()) if path.exists() else {"schema": 1, "waivers": []}
    data["waivers"] = waivers
    wv = next((w for w in waivers if w["base_sha"] == a.base_sha and w["new_sha"] == a.new_sha), None)
    if wv is None:
        wv = {"base_sha": a.base_sha, "new_sha": a.new_sha,
              "new_training_sources_digest": digest, "fingerprints": []}
        waivers.append(wv)
    elif wv["new_training_sources_digest"] != digest:
        print("REFUSED: this new run's digest differs from the one the waiver already records",
              file=sys.stderr)
        return 2
    wv["fingerprints"] = [fp for fp in wv["fingerprints"] if fp["name"] != a.name]
    wv["fingerprints"].append({"name": a.name, **paths, "steps": a.steps, "outcome": outcome,
                               "allow_drift": bool(a.allow_drift)})
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fingerprint": a.name, "outcome": outcome}))
    return 5 if outcome == "stop" else 0


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "fingerprint":
        return main_fingerprint(argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unmatched", action="store_true", help="tests and smoke only")
    ap.add_argument("--min-cases", type=int, default=1,
                    help="a check with fewer cases than this is reported vacuous and does not pass")
    args = ap.parse_args(argv)
    runs = load_runs(Path(args.runs))
    if not runs:
        print("no completed runs", file=sys.stderr); return 2
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seen_tasks = sorted({r["cfg"].get("task", "speed") for r in runs})
    if len(seen_tasks) > 1:
        print(f"REFUSED: runs span more than one task: {seen_tasks}. "
              "A matched-budget comparison holds one task fixed.", file=sys.stderr)
        return 3
    sha_notes = []
    problems = matched_budget_check(runs, notes=sha_notes)
    if problems and not args.allow_unmatched:
        print("REFUSED: runs are not at a matched budget:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 3
    stats, worst = counterfactual_checks(runs)
    # A check that examined nothing has not passed. Zero failures out of zero groups is the
    # same shape of mistake as measuring a constant and reporting it as a result, so every
    # pass flag below requires that the check actually had cases to look at.
    n_groups = stats.get("groups", 0)
    n_allviol = stats.get("all_violating", 0)
    n_const = stats.get("constant_verdict", 0)
    n_mixed = stats.get("mixed", 0)
    vacuous = [name for name, n in (("P1", n_allviol), ("P2", n_const), ("P2_boundary", n_mixed),
                                    ("self_consistency", n_groups)) if n < args.min_cases]
    # A waived commit difference is reported, never silent. If the paper compares runs built
    # from two commits, a reader is entitled to see that and to see why it was allowed.
    for n in sha_notes:
        if n.get("fingerprint_waiver"):
            print(f"[note] runs span commits {n['pair'][0][:8]} and {n['pair'][1][:8]}, which differ "
                  f"in {', '.join(n['training_sources_changed'])}; admitted by fingerprints "
                  f"{n['fingerprint_waiver']['outcomes']}", file=sys.stderr)
            continue
        print(f"[note] runs span commits {n['pair'][0][:8]} and {n['pair'][1][:8]}, "
              f"which differ in no training source", file=sys.stderr)
    # A commit hash describes a run only when the tree that produced it was clean. Every run in
    # both arrays recorded a dirty tree, so the hash is a coarse label and not a full description.
    # What actually establishes that the reward code is unchanged is the recomputation below,
    # which rebuilds every logged group's advantages with today's code. Saying this out loud is
    # the difference between a provenance claim and a provenance habit.
    n_dirty = sum(1 for r in runs if r["prov"].get("git_dirty"))
    if n_dirty:
        print(f"[note] {n_dirty} of {len(runs)} runs recorded a dirty working tree, so the commit "
              f"hash does not fully describe them; the recomputation check below is what "
              f"establishes the reward code is unchanged", file=sys.stderr)
    report = {"n_runs": len(runs), "matched_budget_problems": problems,
              "commit_spans_waived": sha_notes, "runs_with_dirty_tree": n_dirty,
              "counterfactual": stats,
              "worst_abs_diff": worst, "min_cases_required": args.min_cases, "vacuous_checks": vacuous,
              "P1_pass": n_allviol >= args.min_cases and stats.get("P1_fail", 0) == 0,
              "P2_pass": n_const >= args.min_cases and stats.get("P2_fail", 0) == 0,
              "P2_boundary_pass": n_mixed >= args.min_cases and stats.get("P2_boundary_fail", 0) == 0,
              "trainer_self_consistent": (n_groups >= args.min_cases and worst.get("self", 0.0) < 1e-6
                                          and stats.get("kept_mismatch", 0) == 0
                                          and stats.get("verdict_mismatch", 0) == 0
                                          and stats.get("anchor_mismatch", 0) == 0
                                          and stats.get("prompt_index_mismatch", 0) == 0
                                          and stats.get("keep_schedule_n_mismatch", 0) == 0)}
    unchanged = no_update_runs(runs)
    report["no_update_runs"] = unchanged
    report["no_update_eval_pass"] = all(x["eval_unchanged"] for x in unchanged)
    report["variant_algebra"] = variant_algebra(runs)
    (out / "checks.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    collapses = unparseable_collapse(runs)
    if collapses:
        report["unparseable_collapse"] = collapses
        with (out / "unparseable_collapse.csv").open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(collapses[0].keys()))
            wtr.writeheader()
            wtr.writerows(collapses)
    agreement = verdict_none_agreement(runs)
    if agreement:
        report["verdict_none_agreement"] = agreement
        with (out / "verdict_none_agreement.csv").open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(agreement[0].keys()))
            wtr.writeheader()
            wtr.writerows(agreement)
    counters = group_counters(runs)
    with (out / "group_counters.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(counters[0].keys()))
        w.writeheader(); w.writerows(counters)
    report["group_counters"] = counters
    fallback = fallback_group_counts(runs)
    report["fallback_group_counts"] = fallback
    with (out / "fallback_group_counts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fallback[0].keys()))
        w.writeheader(); w.writerows(fallback)
    (out / "checks.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    timing = remedy_timing(runs)
    with (out / "remedy_timing.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(timing[0].keys()))
        w.writeheader(); w.writerows(timing)
    answers = final_answers(runs)
    if answers:
        with (out / "final_answers.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(answers[0].keys()))
            w.writeheader(); w.writerows(answers)

    rows, arows = final_table(runs, out)
    if rows is None:
        print("no run had any training step", file=sys.stderr)
        return 4
    curves(runs, out)
    write_arm_labels(out, runs)
    figures(runs, out)
    print(json.dumps({k: report[k] for k in ("n_runs", "P1_pass", "P2_pass", "P2_boundary_pass",
                                              "trainer_self_consistent", "vacuous_checks", "worst_abs_diff")}))
    for a in arows:
        print(json.dumps(a))
    for c in counters:
        print(json.dumps(c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
