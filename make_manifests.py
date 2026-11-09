#!/usr/bin/env python3
"""Write the revision's run manifests from the frozen ones, so no row is typed by hand.

Four new manifests use the v2 layout. It names the task and carries the three remedy flags, and the
submission script passes those flags only for this layout:
  slurm/manifest_speed_remedy.csv       speed hard level: the remedy arms at the light verdict, and
                                        three arms at five times that weight
  slurm/manifest_capacity_remedy.csv    capacity hard level: the remedy arms at the light verdict
  slurm/manifest_capacity_kl.csv        capacity hard level: residual, verdict and no penalty, each
                                        with the KL penalty
  slurm/manifest_fingerprint_speed.csv  one speed run with every new flag at its default, for the
                                        equivalence gate
slurm/manifest_capacity.csv keeps its 36 rows and gains rows 37-60, seeds 3 and 4 of every capacity
arm. Hard-level rows come first, then the easy-level arms with a task term, then the easy-level arms
without one, so a cut from the end removes the rows the ledger ranks last.

A fifth v2 manifest, slurm/manifest_speed_below.csv, holds three speed hard-level runs of the
residual arm at a weight below its ordering threshold. `_below_weight` computes that weight from the
reward code. In the window of allowed weights the residual reward still decreases with the speed on
each side of the task cap. In the same window the residual no longer ranks every feasible completion
above every violator. `_below_weight` refuses any weight outside that window.

Two more v2 manifests hold the scale-matched residual at the hard level of each task.
slurm/manifest_speed_matched.csv holds three speed runs, and slurm/manifest_capacity_matched.csv
holds five capacity runs. Both run the graded arm at the weight `_matched_weight` computes from the
reward code. At that weight the residual's penalty at the smallest violation never exceeds the light
verdict's penalty, and it equals that penalty on the speed task and on at least one capacity
instance. Every other field equals the residual and light-verdict rows of the same task and seed.

One more v2 manifest, slurm/manifest_capacity_below.csv, holds five capacity hard-level runs of the
residual arm at a weight below its ordering threshold on four of the six instances.
`_capacity_below_weight` computes that weight from the reward code as twice the inclusion bound. It
checks four properties by enumeration and refuses the weight when one fails. The best feasible answer
must outrank every violator on every instance. Adding any item to an overfull selection must lower the
reward. No violator may tie the worst feasible answer. The residual must stop ranking every feasible
answer above every violator on exactly the instances in CAPACITY_BELOW_DROPPED.

Two v3 manifests hold the yoked filter controls. The v3 layout is the v2 header followed by
`keep_schedule`, and the submission script passes that id as --keep-schedule for this layout only.
slurm/manifest_speed_yoked.csv holds 36 speed runs, six draws for each of seeds 0, 1 and 2 at w = 5 and
at w = 25. slurm/manifest_capacity_yoked.csv holds 6 capacity runs, two draws for each of seeds 0, 1
and 2 at w = 5. Each row equals its verifier-filter source row except in `task_id`, `arm_name`,
`group_filter` and `keep_schedule`. The schedule ids and the source rows come from
make_keep_schedules.py, and each row is checked against the committed configs/keep_schedules.json.

Every weight, limit and arm name comes from the frozen manifests, configs/regimes.yaml or
analyze.py. The one new constant is the KL weight, which PREDICTIONS.md fixes.

Usage: python make_manifests.py            write every manifest
       python make_manifests.py --check    exit 1 when a committed manifest differs from this output
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

import analyze
import capacity_task
import make_keep_schedules
import rewards

HERE = Path(__file__).resolve().parent

V2_HEADER = ("task_id", "task", "regime", "arm_name", "arm", "w", "w_task", "vmax", "seed",
             "adv_variant", "group_filter", "kl_beta")
V3_HEADER = V2_HEADER + ("keep_schedule",)
SPEED_V1_HEADER = ("task_id", "regime", "arm_name", "arm", "w", "seed")
CAPACITY_V1_HEADER = ("task_id", "regime", "arm_name", "arm", "w", "w_task", "vmax", "seed")

LEGACY_CAPACITY_ROWS = 36
REMEDY_SEEDS = (0, 1, 2)
NEW_CAPACITY_SEEDS = (3, 4)
HARD_SPEED_REGIME = "A"
HARD_CAPACITY_REGIME = "H"
# The matched residual uses these capacity seeds. They are the seeds of the existing hard-level
# residual and light-verdict capacity runs.
MATCHED_CAPACITY_SEEDS = REMEDY_SEEDS + NEW_CAPACITY_SEEDS
# Two first-violation penalties closer than this count as equal. At the matched weight the trainer's
# arithmetic gives 4.999999999999999 where the light verdict's penalty is exactly 5.
MATCHED_PENALTY_TOL = 1e-9
# The KL ablation's weight, as PREDICTIONS.md states it. DeepSeekMath trains with this value.
KL_BETA_ABLATION = 0.04
# The capacity below-threshold residual uses the seeds of the existing hard-level residual and
# light-verdict capacity runs.
CAPACITY_BELOW_SEEDS = REMEDY_SEEDS + NEW_CAPACITY_SEEDS
# At the capacity below-threshold weight the residual no longer ranks every feasible answer above every
# violator on these hard-level instances, and it still does on the others. The generator recomputes
# the set by enumeration and refuses a weight that gives a different set.
CAPACITY_BELOW_DROPPED = (0, 1, 3, 4)
# Two capacity rewards closer than this count as tied. A check that needs a strict inequality refuses
# a near tie, so no property can pass through rounding in the last digit.
CAPACITY_BELOW_MARGIN = 1e-9

# (advantage variant, group filter) per remedy arm, in the order the ledger lists them.
SPEED_LIGHT_CELLS = (("drgrpo", "none"), ("ngrpo_feasible", "none"), ("ngrpo_sup", "none"),
                     ("grpo", "verdict_mixed"), ("grpo", "reward_spread"))
SPEED_ABOVE_CELLS = (("grpo", "none"), ("ngrpo_feasible", "none"), ("grpo", "verdict_mixed"))
CAPACITY_LIGHT_CELLS = (("drgrpo", "none"), ("ngrpo_feasible", "none"),
                        ("grpo", "verdict_mixed"), ("grpo", "reward_spread"))
KL_ARMS = ("graded", "binary", "none")


def _read(name):
    with (HERE / "slurm" / name).open(newline="") as f:
        rows = list(csv.reader(f))
    return tuple(rows[0]), [dict(zip(rows[0], r)) for r in rows[1:]]


def _text(header, rows) -> str:
    return "\n".join([",".join(header)] + [",".join(str(r[k]) for k in header) for r in rows]) + "\n"


def _v2_row(task_id, task, regime, arm, w, w_task, vmax, seed, adv_variant="grpo",
            group_filter="none", kl_beta=0.0):
    cfg = {"task": task, "arm": arm, "w": float(w), "w_task": float(w_task),
           "adv_variant": adv_variant, "group_filter": group_filter, "kl_beta": float(kl_beta)}
    return {"task_id": task_id, "task": task, "regime": regime, "arm_name": analyze.arm_name(cfg),
            "arm": arm, "w": w, "w_task": w_task, "vmax": vmax, "seed": seed,
            "adv_variant": adv_variant, "group_filter": group_filter, "kl_beta": repr(float(kl_beta))}


def _speed_light_row():
    header, rows = _read("manifest.csv")
    assert header == SPEED_V1_HEADER, header
    return next(r for r in rows if r["regime"] == HARD_SPEED_REGIME and r["arm_name"] == "binary"
                and int(r["seed"]) == 0)


def _speed_vmax() -> str:
    regimes = yaml.safe_load((HERE / "configs" / "regimes.yaml").read_text())
    return repr(float(regimes[HARD_SPEED_REGIME]["v_max"]))


def _speed_w_task() -> str:
    return repr(float(analyze.ARM_SPECS["binary"][2]))


def _below_window(v_max: float):
    """Return the open interval of residual weights as (lower bound, ordering threshold).

    Below the task cap the residual reward is affine in the speed magnitude. Its slope is
    task_slope - factor * w / v_max. The slope is negative exactly when w exceeds
    task_slope * v_max / factor, and that value is the lower bound. The residual ranks every
    feasible completion above a violator v exactly when w exceeds
    (task(v) - worst feasible task) / (factor * residual(v)). The ordering threshold is the largest
    of those values over every parseable speed."""
    task_slope = float(rewards.task(1.0) - rewards.task(0.0))
    factor = -float(rewards.pen_graded(2.0 * v_max, v_max))
    table = rewards.speed_output_table(v_max)
    violator = ~table["verdict"]
    worst_feasible = float(table["task"][table["verdict"]].min())
    threshold = float(((table["task"][violator] - worst_feasible)
                       / (factor * table["residual"][violator])).max())
    return task_slope * v_max / factor, threshold


def _below_weight() -> float:
    """Return the residual weight at which the penalty's slope in the magnitude is twice the task slope."""
    v_max = float(_speed_vmax())
    task_slope = float(rewards.task(1.0) - rewards.task(0.0))
    factor = -float(rewards.pen_graded(2.0 * v_max, v_max))
    w = 2.0 * task_slope * v_max / factor
    named = analyze.NAMED_WEIGHTS["graded_below"][1]
    if w != named:
        raise ValueError(f"the weight from the reward code is {w!r} and analyze.py names {named!r}")
    lower, threshold = _below_window(v_max)
    if not lower < w < threshold:
        raise ValueError(f"the weight {w!r} is outside the window ({lower!r}, {threshold!r})")
    return w


def _residual_factor() -> float:
    """Return the residual's factor, read from the reward code as the penalty one full limit above it."""
    v_max = float(_speed_vmax())
    return -float(rewards.pen_graded(2.0 * v_max, v_max))


def _first_violation_penalties(task: str, w: float, limit: float) -> list:
    """Return the residual's penalty at weight w for the smallest violation of each instance.

    The speed task has one instance and the capacity task has six. Each value comes from enumerating
    every parseable speed or every subset of the instance, so no step assumes that the smallest
    violation is one unit above the limit."""
    factor = _residual_factor()
    if task == "speed":
        tables = [rewards.speed_output_table(limit)]
    else:
        tables = [capacity_task.output_table(limit, k) for k in range(len(capacity_task.INSTANCES))]
    return [w * factor * float(t["residual"][t["residual"] > 0].min()) for t in tables]


def _matched_weight(task: str, light_w: str, limit: float, limit_units: float) -> float:
    """Return the residual weight whose first-violation penalty matches the light verdict's.

    The residual's penalty at a violation of one unit is w * factor / limit_units, and the light
    verdict's penalty is its weight. Setting the two equal gives light_w * limit_units / factor. This
    closed form is exact in floating point, while dividing by the smallest residual leaves an error in
    the last digit. The function refuses a weight that analyze.py does not name. It also refuses a
    weight whose enumerated first-violation penalty exceeds the light verdict's on any instance, or
    equals it on none."""
    w = float(light_w) * limit_units / _residual_factor()
    named = analyze.NAMED_WEIGHTS["graded_matched"][1]
    if w != named:
        raise ValueError(f"the weight from the reward code is {w!r} and analyze.py names {named!r}")
    light = float(light_w)
    penalties = _first_violation_penalties(task, w, limit)
    if any(p > light + MATCHED_PENALTY_TOL for p in penalties):
        raise ValueError(f"at weight {w!r} a {task} first-violation penalty exceeds the light "
                         f"verdict's {light!r}: {penalties!r}")
    if not any(abs(p - light) <= MATCHED_PENALTY_TOL for p in penalties):
        raise ValueError(f"at weight {w!r} no {task} first-violation penalty equals the light "
                         f"verdict's {light!r}: {penalties!r}")
    return w


def capacity_legacy():
    header, rows = _read("manifest_capacity.csv")
    assert header == CAPACITY_V1_HEADER, header
    return rows[:LEGACY_CAPACITY_ROWS]


def capacity_rows():
    """Rows 1-36 as committed, then seeds 3 and 4 of every arm in priority order."""
    legacy = capacity_legacy()
    template = {(r["regime"], r["arm_name"]): r for r in legacy}

    def arm_order(regime):
        seen = []
        for r in legacy:
            if r["regime"] == regime and r["arm_name"] not in seen:
                seen.append(r["arm_name"])
        return seen

    easy = next(r["regime"] for r in legacy if r["regime"] != HARD_CAPACITY_REGIME)
    blocks = [(HARD_CAPACITY_REGIME, a) for a in arm_order(HARD_CAPACITY_REGIME)]
    blocks += [(easy, a) for a in arm_order(easy) if float(template[(easy, a)]["w_task"]) > 0]
    blocks += [(easy, a) for a in arm_order(easy) if float(template[(easy, a)]["w_task"]) == 0]
    rows = list(legacy)
    for regime, arm in blocks:
        for seed in NEW_CAPACITY_SEEDS:
            rows.append(dict(template[(regime, arm)], task_id=str(len(rows) + 1), seed=str(seed)))
    return rows


def speed_remedy():
    light = _speed_light_row()
    vmax, w_task, above = _speed_vmax(), _speed_w_task(), repr(float(analyze.BINARY_ABOVE_W))
    cells = [(light["w"], v, f) for v, f in SPEED_LIGHT_CELLS]
    cells += [(above, v, f) for v, f in SPEED_ABOVE_CELLS]
    rows = []
    for w, variant, group_filter in cells:
        for seed in REMEDY_SEEDS:
            rows.append(_v2_row(len(rows) + 1, "speed", HARD_SPEED_REGIME, "binary", w, w_task,
                                vmax, seed, variant, group_filter))
    return rows


def speed_below():
    """Return one row for each of seeds 0, 1 and 2 of the graded arm at the below-threshold weight."""
    w, vmax, w_task = repr(_below_weight()), _speed_vmax(), _speed_w_task()
    return [_v2_row(i + 1, "speed", HARD_SPEED_REGIME, "graded", w, w_task,
                    vmax, seed) for i, seed in enumerate(REMEDY_SEEDS)]


def speed_matched():
    """Return one row for each of seeds 0, 1 and 2 of the graded arm at the matched weight."""
    light = _speed_light_row()
    vmax = _speed_vmax()
    w = _matched_weight("speed", light["w"], float(vmax), float(vmax))
    arm = analyze.NAMED_WEIGHTS["graded_matched"][0]
    return [_v2_row(i + 1, "speed", HARD_SPEED_REGIME, arm, repr(w), _speed_w_task(), vmax, seed)
            for i, seed in enumerate(REMEDY_SEEDS)]


def _capacity_light_row():
    return next(r for r in capacity_legacy()
                if r["regime"] == HARD_CAPACITY_REGIME and r["arm_name"] == "binary")


def capacity_matched():
    """Return one row for each of seeds 0 to 4 of the graded arm at the matched weight.

    The six instances have different capacities, so one weight cannot match every instance. The
    weight uses the smallest capacity, so the first-violation penalty never exceeds the light
    verdict's on any instance and equals it on the instances with that capacity."""
    light = _capacity_light_row()
    ratio = float(light["vmax"])
    units = float(min(capacity_task.capacity_of(inst, ratio) for inst in capacity_task.INSTANCES))
    w = _matched_weight("capacity", light["w"], ratio, units)
    arm = analyze.NAMED_WEIGHTS["graded_matched"][0]
    return [_v2_row(i + 1, "capacity", HARD_CAPACITY_REGIME, arm, repr(w), light["w_task"],
                    light["vmax"], seed) for i, seed in enumerate(MATCHED_CAPACITY_SEEDS)]


def _capacity_inclusion_bound(ratio: float) -> float:
    """Return the largest weight at which adding some item to an overfull selection keeps the reward.

    Above the capacity, adding item j raises the residual by w_j / cap and the task term by
    v_j / opt. The reward therefore falls for every such addition exactly when the weight exceeds
    v_j * cap / (factor * opt * w_j). This function returns the largest of those values over every item
    of every instance, and the factor is read from the reward code."""
    factor = _residual_factor()
    bound = 0.0
    for inst in capacity_task.INSTANCES:
        cap = capacity_task.capacity_of(inst, ratio)
        opt = capacity_task.dp_optimum(inst["values"], inst["weights"], cap)
        for value, weight in zip(inst["values"], inst["weights"]):
            bound = max(bound, value * cap / (factor * opt * weight))
    return bound


def _capacity_rewards(ratio: float, instance: int, w: float):
    """Return the graded arm's reward at weight w and the verdict for every subset of one instance.

    The arrays are indexed by bitmask, as capacity_task.output_table indexes them. The reward uses the
    task weight 1 and the penalty shape of the reward code."""
    table = capacity_task.output_table(ratio, instance)
    reward = np.asarray(table["task"]) + w * rewards._table_penalty(table, "graded")
    return reward, np.asarray(table["verdict"], dtype=bool)


def _largest_addition_change(reward, verdict) -> float:
    """Return the largest change in reward when one item joins an overfull selection.

    Every overfull subset and every item it lacks are enumerated, so no step assumes that the change
    depends on the item alone."""
    masks = np.arange(reward.size)
    largest = -np.inf
    for item in range(capacity_task.N_ITEMS):
        base = masks[(((masks >> item) & 1) == 0) & ~verdict]
        largest = max(largest, float((reward[base | (1 << item)] - reward[base]).max()))
    return largest


def _capacity_below_weight() -> float:
    """Return the capacity below-threshold weight, which is twice the inclusion bound.

    The function refuses the weight in four cases. The weight may differ from the one analyze.py
    names. A violator may reach the best feasible reward on some instance. Adding an item to an
    overfull selection may fail to lower the reward. The instances on which some violator outranks the
    worst feasible answer may differ from CAPACITY_BELOW_DROPPED. It also refuses a violator that ties
    the worst feasible reward, because that tie would make the set depend on rounding. Every check
    enumerates all 65,536 subsets of each instance."""
    ratio = float(_capacity_light_row()["vmax"])
    w = 2.0 * _capacity_inclusion_bound(ratio)
    named = analyze.NAMED_WEIGHTS["graded_below_cap"][1]
    if w != named:
        raise ValueError(f"the weight from the reward code is {w!r} and analyze.py names {named!r}")
    dropped = []
    for k in range(len(capacity_task.INSTANCES)):
        reward, verdict = _capacity_rewards(ratio, k, w)
        feasible, violator = reward[verdict], reward[~verdict]
        if violator.max() >= feasible.max() - CAPACITY_BELOW_MARGIN:
            raise ValueError(f"at weight {w!r} a violator reaches the best feasible reward on "
                             f"instance {k}")
        if np.abs(violator - feasible.min()).min() <= CAPACITY_BELOW_MARGIN:
            raise ValueError(f"at weight {w!r} a violator ties the worst feasible reward on "
                             f"instance {k}")
        if _largest_addition_change(reward, verdict) >= -CAPACITY_BELOW_MARGIN:
            raise ValueError(f"at weight {w!r} adding an item to an overfull selection does not lower "
                             f"the reward on instance {k}")
        if violator.max() > feasible.min():
            dropped.append(k)
    if tuple(dropped) != CAPACITY_BELOW_DROPPED:
        raise ValueError(f"at weight {w!r} the ordering is dropped on instances {dropped!r}, "
                         f"expected {list(CAPACITY_BELOW_DROPPED)!r}")
    return w


def capacity_below():
    """Return one row for each of seeds 0 to 4 of the graded arm at the capacity below-threshold weight.

    Every field except the arm, the weight and the name comes from the hard-level light-verdict row,
    which equals the hard-level residual row in each of those fields."""
    light = _capacity_light_row()
    w = _capacity_below_weight()
    arm = analyze.NAMED_WEIGHTS["graded_below_cap"][0]
    return [_v2_row(i + 1, "capacity", HARD_CAPACITY_REGIME, arm, repr(w), light["w_task"],
                    light["vmax"], seed) for i, seed in enumerate(CAPACITY_BELOW_SEEDS)]


def capacity_remedy():
    light = next(r for r in capacity_legacy()
                 if r["regime"] == HARD_CAPACITY_REGIME and r["arm_name"] == "binary")
    rows = []
    for variant, group_filter in CAPACITY_LIGHT_CELLS:
        for seed in REMEDY_SEEDS:
            rows.append(_v2_row(len(rows) + 1, "capacity", HARD_CAPACITY_REGIME, light["arm"],
                                light["w"], light["w_task"], light["vmax"], seed, variant,
                                group_filter))
    return rows


def capacity_kl():
    legacy = capacity_legacy()
    rows = []
    for name in KL_ARMS:
        base = next(r for r in legacy if r["regime"] == HARD_CAPACITY_REGIME and r["arm_name"] == name)
        for seed in REMEDY_SEEDS:
            rows.append(_v2_row(len(rows) + 1, "capacity", HARD_CAPACITY_REGIME, base["arm"], base["w"],
                                base["w_task"], base["vmax"], seed, kl_beta=KL_BETA_ABLATION))
    return rows


def fingerprint_speed():
    light = _speed_light_row()
    return [_v2_row(1, "speed", HARD_SPEED_REGIME, light["arm"], light["w"], _speed_w_task(),
                    _speed_vmax(), int(light["seed"]))]


def _yoked_rows(task: str):
    """Return the v3 rows of one task's yoked controls, by cell, then seed, then draw.

    Each row copies its source row from the verifier-filter manifest and changes only the task id,
    the arm name, the group filter and the schedule id. The row must also agree with the committed
    schedule entry on every field the trainer checks, so a stale schedule file is refused here."""
    entries = json.loads((HERE / make_keep_schedules.OUT_REL).read_text())["schedules"]
    rows = []
    for src in make_keep_schedules.SOURCES:
        if src["task"] != task:
            continue
        header, source_rows = _read(src["manifest"])
        assert header == V2_HEADER, header
        base = next(r for r in source_rows if int(r["task_id"]) == src["task_id"])
        want = (task, "binary", src["w"], src["seed"], "grpo", "verdict_mixed", 0.0)
        got = (base["task"], base["arm"], float(base["w"]), int(base["seed"]), base["adv_variant"],
               base["group_filter"], float(base["kl_beta"]))
        if got != want:
            raise ValueError(f"{src['manifest']} row {src['task_id']} is {got}, not the source {want}")
        for draw in range(make_keep_schedules.DRAWS[task]):
            sid = make_keep_schedules.schedule_id(task, src["w"], src["seed"], draw)
            entry = entries[sid]
            row = dict(base, task_id=str(len(rows) + 1), group_filter="none", keep_schedule=sid)
            cfg = {"task": task, "arm": row["arm"], "w": float(row["w"]),
                   "w_task": float(row["w_task"]), "seed": int(row["seed"]), "adv_variant": "grpo",
                   "group_filter": "none", "kl_beta": float(row["kl_beta"]), "keep_schedule": sid}
            row["arm_name"] = analyze.arm_name(cfg)
            for k in ("task", "regime", "arm", "w", "w_task", "vmax", "seed"):
                same = (entry[k] == row[k]) if isinstance(entry[k], str) else (
                    float(entry[k]) == float(row[k]))
                if not same:
                    raise ValueError(f"schedule {sid} has {k} = {entry[k]!r} and its row has {row[k]!r}")
            rows.append(row)
    return rows


def speed_yoked():
    """Return the 36 speed control rows, six draws per seed at w = 5 and then at w = 25."""
    return _yoked_rows("speed")


def capacity_yoked():
    """Return the 6 capacity control rows, two draws per seed at w = 5."""
    return _yoked_rows("capacity")


def build_all() -> dict:
    return {
        "slurm/manifest_capacity.csv": _text(CAPACITY_V1_HEADER, capacity_rows()),
        "slurm/manifest_speed_remedy.csv": _text(V2_HEADER, speed_remedy()),
        "slurm/manifest_speed_below.csv": _text(V2_HEADER, speed_below()),
        "slurm/manifest_speed_matched.csv": _text(V2_HEADER, speed_matched()),
        "slurm/manifest_capacity_matched.csv": _text(V2_HEADER, capacity_matched()),
        "slurm/manifest_capacity_below.csv": _text(V2_HEADER, capacity_below()),
        "slurm/manifest_capacity_remedy.csv": _text(V2_HEADER, capacity_remedy()),
        "slurm/manifest_capacity_kl.csv": _text(V2_HEADER, capacity_kl()),
        "slurm/manifest_fingerprint_speed.csv": _text(V2_HEADER, fingerprint_speed()),
        "slurm/manifest_speed_yoked.csv": _text(V3_HEADER, speed_yoked()),
        "slurm/manifest_capacity_yoked.csv": _text(V3_HEADER, capacity_yoked()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    built = build_all()
    if a.check:
        stale = [rel for rel, text in built.items()
                 if not (HERE / rel).exists() or (HERE / rel).read_text() != text]
        for rel in stale:
            print(f"STALE: {rel} differs from make_manifests.py output", file=sys.stderr)
        return 1 if stale else 0
    for rel, text in built.items():
        (HERE / rel).write_text(text)
        print(f"wrote {rel}: {text.count(chr(10)) - 1} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
