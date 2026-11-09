#!/usr/bin/env python3
"""Replay the logged groups of Block A' runs under three designs and count what the paper needs.

Every reward in this harness is a pure function of the parsed speeds, so the `metrics.jsonl` of a
finished run holds enough to recompute every reward and every advantage after the fact. This script
does that for each training group of one or more run directories, under three designs.

  own          the row's own design, read from the run's config.json. The recomputed rewards,
               verdicts and advantages must equal the logged values exactly. A difference stops the
               script, because a log that the reward code cannot reproduce is not evidence.
  scale        the binary verdict under the plain estimator at the light weight and at the heavy
               weight (5 and 5000 by default). In an all-violating group every member carries the
               same penalty, the shift is group-constant, and group-mean centering removes it, so the
               two advantage vectors must agree. The script counts the all-violating groups whose
               two vectors differ by more than --tol anywhere. This is the scale control of Block A':
               the heavy verdict and the light verdict give the same advantages wherever the verdict
               is cancelled, so whatever the heavy arm does differently it does in the mixed groups.
  fixedref     the binary verdict at the light weight under grpo_fixedref with shift c (2 by
               default). This is the counterfactual signal: what the fixed-reference arm would have
               given the logged groups. In an all-violating group it is the plain advantage minus c,
               so it is nonzero even when the plain advantage is exactly zero.

The JSON summary holds, per run and in total:
  groups                                  training groups read
  all_violating                           groups in which no member satisfies the constraint
  all_violating_with_reward_spread        of those, groups whose own-design reward range exceeds
                                          rewards.ZERO_SPREAD_TOL
  groups_with_distinct_speeds             groups with at least two distinct parsed speeds
  identical_reward_among_distinct_speeds  of those, groups whose own-design rewards are identical
  all_violating_with_distinct_speeds      all-violating groups with at least two distinct parsed speeds
  all_violating_identical_reward_among_distinct_speeds
                                          of those, groups whose own-design rewards are identical,
                                          which is the numerator of prediction A0
  mixed                                   groups with at least one feasible member and one violator
  all_passing                             groups in which every member satisfies the constraint
  all_violating_scale_differ              all-violating groups whose light- and heavy-weight
                                          advantages differ by more than --tol
  max_abs_adv_diff_all_violating          the largest such difference seen
  all_violating_shift_differ              all-violating groups whose heavy-weight rewards are not the
                                          light-weight rewards minus (w_heavy - w_light) within --tol,
                                          which is the identity behind the scale control stated on
                                          the rewards, where rounding is at the last digit of w_heavy
  max_abs_shift_diff_all_violating        the largest such difference seen
  all_violating_zero_adv_own              all-violating groups whose own-design advantages are all
                                          zero within rewards.ZERO_ADV_TOL
  all_violating_nonzero_adv_fixedref      all-violating groups whose fixed-reference counterfactual
                                          advantages are not all zero

The two weights are compared in float64. The identity is exact in arithmetic, and the recomputation
differs only by the rounding of the heavy shift, so the default tolerance of 1e-9 is loose for groups
whose task values differ by more than about 1e-6. A group of speeds all near the parse ceiling has
smaller task differences than that, and its z-score divides the rounding error of the shift by that
small spread. Measured on 2026-11-06, the group (900000, 901000, 950000, 999999) differs by 5.2e-9
between the two weights for rounding alone. The maximum difference is reported, and the reward-shift
count checks the same identity on the rewards, where the rounding is not amplified, so a reader can
tell a rounding difference from a wrong identity.

Usage:
  python replay_bounded_scale.py RUN_DIR [RUN_DIR ...] --out summary.json
  python replay_bounded_scale.py RUN_DIR --out summary.json --w-light 5 --w-heavy 5000 --fixedref-c 2 --tol 1e-9

Exit status 0 when every run reproduces and no all-violating group fails the scale check, 3 when a
group fails the scale check (the summary is still written), and a ValueError when a log does not
reproduce.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import rewards as RW
import tasks

COUNT_KEYS = (
    "groups", "all_violating", "all_violating_with_reward_spread", "groups_with_distinct_speeds",
    "identical_reward_among_distinct_speeds", "all_violating_with_distinct_speeds",
    "all_violating_identical_reward_among_distinct_speeds", "mixed", "all_passing",
    "all_violating_scale_differ", "all_violating_shift_differ", "all_violating_zero_adv_own",
    "all_violating_nonzero_adv_fixedref",
)
MAX_KEYS = ("max_abs_adv_diff_all_violating", "max_abs_shift_diff_all_violating")
CONFIG_KEYS = ("task", "regime", "arm", "w", "w_task", "vmax", "seed", "adv_variant", "group_filter",
               "kl_beta", "fixedref_c", "skip_zero_adv", "group_size", "steps")


def _items(spec_arg, logged):
    """The logged parsed structures as the reward code takes them. A spec task's tuple structures
    come back from JSON as lists."""
    if spec_arg is None:
        return list(logged)
    return [None if x is None else (tuple(x) if isinstance(x, list) else x) for x in logged]


def _own_advantage(r, verdicts, cfg, spec_arg, tspec, prompt_index):
    """The advantage the trainer computed for this group, by the same branch it took."""
    variant = cfg.get("adv_variant", "grpo")
    if variant == "grpo_fixedref":
        return RW.advantage_fixedref(r, verdicts, cfg["fixedref_c"]), None
    anchor = RW.reward_anchor(variant, cfg["arm"], cfg["w"], cfg["vmax"], spec=spec_arg,
                              w_task=cfg.get("w_task", 1.0), instance=tspec.instance_for(prompt_index))
    return RW.advantage_variant(r, variant, anchor), anchor


def _distinct_speeds(items) -> int:
    return len({x for x in items if x is not None})


def replay_run(run_dir, *, w_light: float, w_heavy: float, fixedref_c: float, tol: float) -> dict:
    """Replay one run directory. Raises ValueError when a logged group does not reproduce."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    prov_path = run_dir / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    tspec = tasks.get(cfg["task"])
    spec_arg = None if tspec.name == "speed" else tspec
    vmax, w_task = float(cfg["vmax"]), float(cfg.get("w_task", 1.0))
    counts = {k: 0 for k in COUNT_KEYS}
    max_diff, max_shift = 0.0, 0.0
    shift = w_heavy - w_light
    with (run_dir / "metrics.jsonl").open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    for rec in records:
        if rec.get("phase") != "train":
            continue
        for gi, g in enumerate(rec["groups"]):
            where = f"{run_dir.name} step {rec['step']} group {gi}"
            items = _items(spec_arg, g[tspec.ITEMS_KEY])
            # own design
            r = RW.group_rewards(items, cfg["arm"], cfg["w"], vmax, spec=spec_arg, w_task=w_task)
            if r.tolist() != g["rewards"]:
                raise ValueError(f"{where}: recomputed rewards {r.tolist()} differ from the log {g['rewards']}")
            verdicts = RW.group_verdicts(items, vmax, spec=spec_arg)
            if "verdicts" in g and list(g["verdicts"]) != verdicts:
                raise ValueError(f"{where}: recomputed verdicts {verdicts} differ from the log {g['verdicts']}")
            a_own, anchor = _own_advantage(r, verdicts, cfg, spec_arg, tspec, g.get("prompt_index", gi))
            if a_own.tolist() != g["adv"]:
                raise ValueError(f"{where}: recomputed advantages {a_own.tolist()} differ from the log {g['adv']}")
            if "anchor" in g and g["anchor"] != anchor:
                raise ValueError(f"{where}: recomputed anchor {anchor!r} differs from the log {g['anchor']!r}")
            all_violating = not any(verdicts)
            all_passing = all(verdicts)
            if "all_violating" in g and bool(g["all_violating"]) != all_violating:
                raise ValueError(f"{where}: the log marks all_violating={g['all_violating']} and the verdicts say {all_violating}")
            # the verdict at two weights, and the counterfactual
            r_light = RW.group_rewards(items, "binary", w_light, vmax, spec=spec_arg, w_task=w_task)
            r_heavy = RW.group_rewards(items, "binary", w_heavy, vmax, spec=spec_arg, w_task=w_task)
            a_light, a_heavy = RW.advantage(r_light), RW.advantage(r_heavy)
            a_fixed = RW.advantage_fixedref(r_light, verdicts, fixedref_c)
            # counts
            counts["groups"] += 1
            distinct = _distinct_speeds(items) >= 2
            identical = bool(np.ptp(r) <= RW.ZERO_SPREAD_TOL)
            if distinct:
                counts["groups_with_distinct_speeds"] += 1
                if identical:
                    counts["identical_reward_among_distinct_speeds"] += 1
            if all_violating:
                counts["all_violating"] += 1
                if not identical:
                    counts["all_violating_with_reward_spread"] += 1
                if distinct:
                    counts["all_violating_with_distinct_speeds"] += 1
                    if identical:
                        counts["all_violating_identical_reward_among_distinct_speeds"] += 1
                diff = float(np.abs(a_light - a_heavy).max())
                max_diff = max(max_diff, diff)
                if diff > tol:
                    counts["all_violating_scale_differ"] += 1
                # every member carries the penalty -1, so the heavy rewards are the light ones minus the
                # difference of the weights
                sdiff = float(np.abs((r_light - shift) - r_heavy).max())
                max_shift = max(max_shift, sdiff)
                if sdiff > tol:
                    counts["all_violating_shift_differ"] += 1
                if RW.group_adv_is_zero(a_own):
                    counts["all_violating_zero_adv_own"] += 1
                if not RW.group_adv_is_zero(a_fixed):
                    counts["all_violating_nonzero_adv_fixedref"] += 1
            elif all_passing:
                counts["all_passing"] += 1
            else:
                counts["mixed"] += 1
    counts["max_abs_adv_diff_all_violating"] = max_diff
    counts["max_abs_shift_diff_all_violating"] = max_shift
    return {"run_dir": str(run_dir), "config": {k: cfg.get(k) for k in CONFIG_KEYS},
            "training_sources_digest": prov.get("training_sources_digest"),
            "train_records": sum(1 for r in records if r.get("phase") == "train"), **counts}


def replay(run_dirs, *, w_light: float = 5.0, w_heavy: float = 5000.0, fixedref_c: float = 2.0,
           tol: float = 1e-9) -> dict:
    runs = [replay_run(d, w_light=w_light, w_heavy=w_heavy, fixedref_c=fixedref_c, tol=tol)
            for d in run_dirs]
    total = {k: sum(r[k] for r in runs) for k in COUNT_KEYS}
    for k in MAX_KEYS:
        total[k] = max([r[k] for r in runs], default=0.0)
    return {"w_light": w_light, "w_heavy": w_heavy, "fixedref_c": fixedref_c, "tol": tol,
            "zero_spread_tol": RW.ZERO_SPREAD_TOL, "zero_adv_tol": RW.ZERO_ADV_TOL,
            "n_runs": len(runs), "runs": runs, "total": total}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="run directories holding config.json and metrics.jsonl")
    ap.add_argument("--out", required=True, help="where to write the JSON summary")
    ap.add_argument("--w-light", type=float, default=5.0)
    ap.add_argument("--w-heavy", type=float, default=5000.0)
    ap.add_argument("--fixedref-c", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="largest allowed difference between the light- and heavy-weight advantages "
                         "of an all-violating group")
    a = ap.parse_args(argv)
    summary = replay(a.run_dirs, w_light=a.w_light, w_heavy=a.w_heavy, fixedref_c=a.fixedref_c, tol=a.tol)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    t = summary["total"]
    print(json.dumps({"runs": summary["n_runs"], **{k: t[k] for k in COUNT_KEYS + MAX_KEYS}}))
    if t["all_violating_scale_differ"] or t["all_violating_shift_differ"]:
        print(f"SCALE CHECK FAILED: {t['all_violating_scale_differ']} all-violating group(s) differ in "
              f"advantage between w = {a.w_light} and w = {a.w_heavy} by more than {a.tol} (largest "
              f"{t['max_abs_adv_diff_all_violating']:.3e}), and {t['all_violating_shift_differ']} differ in "
              f"the reward shift (largest {t['max_abs_shift_diff_all_violating']:.3e}). The summary was "
              f"written to {out}.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
