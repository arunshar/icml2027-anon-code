#!/usr/bin/env python3
"""Write configs/keep_schedules.json, the kept-group schedules of the yoked filter controls.

A yoked control copies one verifier-filter run's update schedule. At every step where the filter
run kept n groups, the control keeps n groups and takes one optimizer step. At every other step it
keeps no group and takes no step. This script decides which prompts' groups each control keeps.

The rule reads only the number of groups each filter run kept at each step. It takes those counts
from the `kept` and `prompt_index` fields of the filter run's training records. It never reads a
reward, an advantage, a verdict, an evaluation record or a final outcome.

The rule was fixed on 22 September 2026, before any control run existed.
  1. The constant is 20260922. The task code is 0 for speed and 1 for capacity.
  2. At the filter run's first update step with n kept groups, list every set of n prompts in
     lexicographic order. numpy's default_rng, seeded with the list [constant, task code, int(w),
     seed], draws one permutation of that list. Draw d keeps the set at position d of the
     permutation, so the draws of one seed keep distinct sets at that step.
  3. Draw d also seeds default_rng with [constant, task code, int(w), seed, d + 1]. That generator
     draws one permutation of the prompts at every step, whether or not the step updates. At every
     later update step with n kept groups, draw d keeps the first n entries of that step's
     permutation, sorted.

numpy does not promise the same permutations across versions, so the committed file records the
numpy version that wrote it, and the committed file is the reference. The trainer reads that file
and never calls a random generator for the schedule.

Usage: python make_keep_schedules.py            write configs/keep_schedules.json
       python make_keep_schedules.py --check    exit 1 when the committed file differs from this output
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE
OUT_REL = "configs/keep_schedules.json"
SOURCE_PREFIX = "experiments/cancellation/"

THINNING_CONSTANT = 20260922
TASK_CODE = {"speed": 0, "capacity": 1}
DRAWS = {"speed": 6, "capacity": 2}
# This pattern matches the id of one control run. It refuses leading zeros, so every run has
# exactly one id.
SCHEDULE_ID = re.compile(r"(speed|capacity)_w(0|[1-9][0-9]*)_s(0|[1-9][0-9]*)_d(0|[1-9][0-9]*)")
# The trainer checks these fields against its own arguments before it trusts an entry.
CHECKED_FIELDS = ("task", "regime", "arm", "w", "w_task", "vmax", "seed", "steps", "batch_prompts")

# The controls copy these nine verifier-filter runs, listed in manifest order. Each metrics.jsonl
# hash was read on 22 September 2026 from the run directories, and a run whose log differs is
# refused.
SOURCES = (
    {"task": "speed", "w": 5.0, "seed": 0, "manifest": "manifest_speed_remedy.csv", "task_id": 10,
     "run": "experiments/cancellation/runs/759611/10_A_binary_vfilter_s0",
     "metrics_sha256": "7dfc19d487e78631f2cbfba44dabc200ffdb1f98c6dadf22f83710d9817e61b2"},
    {"task": "speed", "w": 5.0, "seed": 1, "manifest": "manifest_speed_remedy.csv", "task_id": 11,
     "run": "experiments/cancellation/runs/759611/11_A_binary_vfilter_s1",
     "metrics_sha256": "f2c84eb25dc27f5cc1a62267aae69cc94cd33747065590e9ea30c091e5c22587"},
    {"task": "speed", "w": 5.0, "seed": 2, "manifest": "manifest_speed_remedy.csv", "task_id": 12,
     "run": "experiments/cancellation/runs/759611/12_A_binary_vfilter_s2",
     "metrics_sha256": "8ce9b94c7eef1a2aa3d006451bb744e5691a5484fd4dad8069faaa8297f16109"},
    {"task": "speed", "w": 25.0, "seed": 0, "manifest": "manifest_speed_remedy.csv", "task_id": 22,
     "run": "experiments/cancellation/runs/759611/22_A_binary_5x_vfilter_s0",
     "metrics_sha256": "41a70f30026a4e0514c193956fc3b9c8fbfe85bf29dcd86286794aa88ee746ab"},
    {"task": "speed", "w": 25.0, "seed": 1, "manifest": "manifest_speed_remedy.csv", "task_id": 23,
     "run": "experiments/cancellation/runs/759611/23_A_binary_5x_vfilter_s1",
     "metrics_sha256": "c69911158b578cc8c4a1afea5b6da6a5411a863db3fb852d00cdb9d762be19f2"},
    {"task": "speed", "w": 25.0, "seed": 2, "manifest": "manifest_speed_remedy.csv", "task_id": 24,
     "run": "experiments/cancellation/runs/759611/24_A_binary_5x_vfilter_s2",
     "metrics_sha256": "0ca150dc7d65839c75a62fb9525e79f9a9cabebb350adae6ef34be342155f913"},
    {"task": "capacity", "w": 5.0, "seed": 0, "manifest": "manifest_capacity_remedy.csv", "task_id": 7,
     "run": "experiments/cancellation/runs/759612/7_H_binary_vfilter_s0",
     "metrics_sha256": "aacbaf90647551765e36ddd8f2f4a9b03ebff09c6644f3ae988536f83c46e194"},
    {"task": "capacity", "w": 5.0, "seed": 1, "manifest": "manifest_capacity_remedy.csv", "task_id": 8,
     "run": "experiments/cancellation/runs/759612/8_H_binary_vfilter_s1",
     "metrics_sha256": "ec462912af3a0754ca83c89f348de53128aa64d36216c600b33e7ce3c4aa8a42"},
    {"task": "capacity", "w": 5.0, "seed": 2, "manifest": "manifest_capacity_remedy.csv", "task_id": 9,
     "run": "experiments/cancellation/runs/759612/9_H_binary_vfilter_s2",
     "metrics_sha256": "530ee5350d2e05039b925cb38035e498686e9ba373138c12fc9ba04fb1913017"},
)


class SourceError(ValueError):
    """A source run is missing, unfinished, changed, or not the filter run the rule expects."""


def schedule_id(task: str, w: float, seed: int, draw: int) -> str:
    """Return the id of one control run, for example speed_w25_s1_d5."""
    if not float(w).is_integer():
        raise ValueError(f"the weight {w!r} is not an integer, so int(w) would change it")
    sid = f"{task}_w{int(w)}_s{int(seed)}_d{int(draw)}"
    if SCHEDULE_ID.fullmatch(sid) is None:
        raise ValueError(f"{sid!r} is not a valid schedule id")
    return sid


def source_kept_sets(records, steps: int, batch_prompts: int) -> list:
    """Return the filter run's kept prompts at every step, as sorted lists.

    Only the `phase` and `step` keys of a record and the `kept` and `prompt_index` keys of a group
    are read. A test hands this function records that raise on any other key."""
    by_step = {}
    for rec in records:
        if rec["phase"] != "train":
            continue
        step = rec["step"]
        if step in by_step:
            raise SourceError(f"step {step} is logged twice")
        indices, kept = [], []
        for g in rec["groups"]:
            indices.append(g["prompt_index"])
            if g["kept"]:
                kept.append(g["prompt_index"])
        if indices != list(range(batch_prompts)):
            raise SourceError(f"step {step} logs prompts {indices}, not 0 to {batch_prompts - 1} in order")
        by_step[step] = sorted(kept)
    if sorted(by_step) != list(range(1, steps + 1)):
        raise SourceError(f"the training records do not cover steps 1 to {steps} exactly once")
    return [by_step[t] for t in range(1, steps + 1)]


def control_kept_sets(task: str, w: float, seed: int, counts, n_draws: int, batch_prompts: int) -> list:
    """Return the kept sets of draws 0 to n_draws - 1, one list per draw and one set per step.

    `counts` holds the filter run's number of kept groups at steps 1, 2 and so on. The function
    receives nothing else about the filter run, so the schedule cannot depend on what it read."""
    counts = [int(n) for n in counts]
    if not float(w).is_integer():
        raise ValueError(f"the weight {w!r} is not an integer, so int(w) would change it")
    if any(n < 0 or n > batch_prompts for n in counts):
        raise ValueError(f"a kept count lies outside 0 to {batch_prompts}")
    taken = [t for t, n in enumerate(counts, start=1) if n > 0]
    if not taken:
        raise ValueError("the source run took no update, so the rule has no first update step")
    t1 = taken[0]
    sets = list(itertools.combinations(range(batch_prompts), counts[t1 - 1]))
    if n_draws > len(sets):
        raise ValueError(f"{n_draws} draws need {n_draws} distinct first-step sets and only "
                         f"{len(sets)} exist")
    base = [THINNING_CONSTANT, TASK_CODE[task], int(w), int(seed)]
    order = np.random.default_rng(base).permutation(len(sets))
    draws = []
    for d in range(n_draws):
        gen = np.random.default_rng(base + [d + 1])
        kept = []
        for t, n in enumerate(counts, start=1):
            # The permutation is drawn at every step, so the stream never depends on the counts.
            perm = gen.permutation(batch_prompts)
            if t == t1:
                kept.append([int(i) for i in sets[int(order[d])]])
            elif n > 0:
                kept.append(sorted(int(i) for i in perm[:n]))
            else:
                kept.append([])
        draws.append(kept)
    return draws


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source(src: dict, repo_root: Path):
    """Return the source run's config and its kept sets, after the refusal checks."""
    d = repo_root / src["run"]
    if not (d / "DONE").exists():
        raise SourceError(f"{src['run']} has no DONE marker")
    got = _sha256(d / "metrics.jsonl")
    if got != src["metrics_sha256"]:
        raise SourceError(f"{src['run']}/metrics.jsonl has SHA-256 {got}, and the rule was fixed "
                          f"on {src['metrics_sha256']}")
    cfg = json.loads((d / "config.json").read_text())
    want = {"task": src["task"], "arm": "binary", "w": src["w"], "seed": src["seed"],
            "adv_variant": "grpo", "group_filter": "verdict_mixed", "kl_beta": 0.0}
    wrong = [k for k, v in want.items() if cfg.get(k) != v]
    if wrong:
        raise SourceError(f"{src['run']} is not the expected verifier-filter run: {wrong}")
    records = (json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines())
    kept = source_kept_sets(records, int(cfg["steps"]), int(cfg["batch_prompts"]))
    return cfg, kept


def build_document(sources=SOURCES, repo_root: Path = REPO_ROOT) -> dict:
    """Return the schedule document for every source run and every draw."""
    schedules = {}
    for src in sources:
        cfg, source_sets = _load_source(src, Path(repo_root))
        counts = [len(s) for s in source_sets]
        draws = control_kept_sets(src["task"], src["w"], src["seed"], counts, DRAWS[src["task"]],
                                  int(cfg["batch_prompts"]))
        for d, kept in enumerate(draws):
            entry = {k: cfg[k] for k in CHECKED_FIELDS}
            entry.update({"draw": d, "source_run": src["run"],
                          "source_metrics_sha256": src["metrics_sha256"],
                          "kept_sets": kept, "source_kept_sets": source_sets})
            schedules[schedule_id(src["task"], src["w"], src["seed"], d)] = entry
    return {
        "about": ("Kept-group schedules of the yoked filter controls. Each entry copies one "
                  "verifier-filter run's kept counts per step and names the prompts the control "
                  "keeps. Written by make_keep_schedules.py, which reads only the kept and "
                  "prompt_index fields of the filter runs' training records. This committed file "
                  "is the reference, and the trainer reads it without drawing any randomness."),
        "schema": 1,
        "numpy_version": np.__version__,
        "rule": {"constant": THINNING_CONSTANT, "task_codes": dict(TASK_CODE),
                 "draws_per_seed": dict(DRAWS),
                 "first_update_step": "default_rng([constant, task_code, int(w), seed]).permutation "
                                      "of the lexicographic list of n-prompt sets, and draw d keeps "
                                      "the set at position d",
                 "later_update_steps": "default_rng([constant, task_code, int(w), seed, d + 1]) "
                                       "draws permutation(batch_prompts) at every step, and draw d "
                                       "keeps the first n entries, sorted"},
        "schedules": schedules,
    }


def render(doc: dict) -> str:
    """Return the file text. Each list of kept sets sits on one line, so the file stays short."""
    tokens = {}

    def compact(obj):
        if isinstance(obj, dict):
            return {k: (_token(v) if k in ("kept_sets", "source_kept_sets") else compact(v))
                    for k, v in obj.items()}
        return obj

    def _token(sets):
        key = f"@@SETS{len(tokens)}@@"
        tokens[key] = json.dumps(sets, separators=(",", ":"))
        return key

    text = json.dumps(compact(doc), indent=1, sort_keys=True)
    for key, value in tokens.items():
        text = text.replace(f'"{key}"', value)
    return text + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    try:
        text = render(build_document())
    except SourceError as e:
        print(f"REFUSED: {e}. Nothing was written.", file=sys.stderr)
        return 2
    path = HERE / OUT_REL
    if a.check:
        if not path.exists() or path.read_text() != text:
            print(f"STALE: {OUT_REL} differs from make_keep_schedules.py output", file=sys.stderr)
            return 1
        return 0
    path.write_text(text)
    print(f"wrote {OUT_REL}: {len(json.loads(text)['schedules'])} schedules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
