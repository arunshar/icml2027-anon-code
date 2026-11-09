#!/usr/bin/env python3
"""Reproduce the revision audit with local CPU arithmetic and optional saved logs.

The audit preserves the recorded parsers and training rules. It checks the population
standard deviation used by NGRPO, demonstrates the parser's permissive behavior, and
verifies residual ordering on every capacity subset at the recorded settings. Optional
run analysis separates empty feasibility from nonempty feasible output and replays
verdict weights on recorded groups. Replay changes no policy and predicts no recovery.

Run `python audit_revision_checks.py --out audit` for arithmetic checks. Add
`--runs runs/capacity_remedy` to audit the recorded capacity runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np
import yaml

import analyze
import capacity_task as C
import rewards as RW
import tasks

HERE = Path(__file__).resolve().parent


def arithmetic_checks():
    """Independent formulas check the conventions that the manuscript states."""
    k, r, anchor = 4, 2.0, 0.0
    values = np.full(k, r)
    mean = (k * r + anchor) / (k + 1)
    population_sd = math.sqrt(k) * abs(r - anchor) / (k + 1)
    actual = RW.advantage_variant(values, "ngrpo_feasible", anchor)
    expected = np.full(k, (r - mean) / (population_sd + 1e-6))
    assert np.allclose(actual, expected, rtol=0, atol=1e-14)
    sample_sd = abs(r - anchor) / math.sqrt(k + 1)
    sample_result = (r - mean) / (sample_sd + 1e-6)
    assert not np.allclose(actual, sample_result, rtol=1e-3)

    parser_examples = {text: list(C.parse_indices(text))
                       for text in ("[]", "[foo]", "[99]", "[1.5]", "[0, 3, 7]")}
    assert parser_examples["[foo]"] == parser_examples["[99]"] == []
    assert parser_examples["[1.5]"] == [1, 5]

    group = np.array([1.0, 2.0, 3.0, 4.0])
    shift = -5.0
    baseline = lambda x: x.mean() / 2
    difference = ((group + shift) - baseline(group + shift)) - (group - baseline(group))
    assert np.allclose(difference, shift / 2)
    residuals, task_values = np.array([1.0, 2.0]), np.array([0.0, 100.0])
    combined = task_values - residuals
    assert combined[1] > combined[0]

    manifest = list(csv.DictReader((HERE / "slurm/manifest_capacity.csv").open()))
    settings = sorted({(float(row["vmax"]), float(row["w"]), float(row["w_task"]))
                       for row in manifest if row["arm"] == "graded" and float(row["w_task"]) > 0})
    ordering = []
    for ratio, weight, task_weight in settings:
        for instance in range(len(C.INSTANCES)):
            table = C.output_table(ratio, instance)
            scores = task_weight * table["task"] - weight * 100 * table["residual"]
            violating = ~table["verdict"]
            levels = np.unique(table["residual"][violating])
            buckets = [scores[table["residual"] == level] for level in levels]
            ordered = all(lower.min() > higher.max()
                          for lower, higher in zip(buckets, buckets[1:]))
            margin = float(scores[table["verdict"]].min() - scores[violating].max())
            assert ordered and margin > 0
            ordering.append(dict(capacity_ratio=ratio, weight=weight, instance=instance,
                                 strict_order_across_unequal_residuals=ordered,
                                 minimum_feasible_reward_margin=margin))

    arms = yaml.safe_load((HERE / "configs/arms.yaml").read_text())
    graded_weight = next(row["w"] for row in arms if row["name"] == "graded")
    regimes = yaml.safe_load((HERE / "configs/regimes.yaml").read_text())
    speed_slopes = {regime: 0.1 - 100 * graded_weight / row["v_max"]
                    for regime, row in regimes.items()}
    assert all(value < 0 for value in speed_slopes.values())
    return dict(ngrpo_population_advantage=float(actual[0]),
                hypothetical_sample_advantage=float(sample_result),
                parser_examples=parser_examples,
                non_equivariant_baseline_shift=difference.tolist(),
                monotone_residual_counterexample_rewards=combined.tolist(),
                speed_reward_slopes=speed_slopes, capacity_ordering=ordering)


def wilson_interval(successes, count, confidence=0.95):
    """The interval treats each training seed as one Bernoulli observation."""
    if not count:
        return None, None
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / count
    divisor = 1 + z * z / count
    center = (p + z * z / (2 * count)) / divisor
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / divisor
    return max(0.0, center - radius), min(1.0, center + radius)


def saved_run_checks(root):
    """The replay reports recorded selections without interpreting their original text."""
    rows, replays = [], []
    runs = analyze.load_runs(root)
    for run in runs:
        cfg = run["cfg"]
        if cfg.get("task", "speed") != "capacity":
            raise ValueError("The quality audit requires capacity runs.")
        ev = run["evals"][-1]
        samples = ev["eval_selections"]
        n = len(samples)
        parsed = [tuple(x) for x in samples if x is not None]
        useful = [x for x in parsed if len(x) > 1 and C.verdict(x, cfg["vmax"])]
        rows.append(dict(regime=cfg["regime"], arm=analyze.arm_name(cfg), seed=cfg["seed"],
                         evaluation_samples=n, unparseable=n - len(parsed),
                         parsed_empty=sum(len(x) == 1 for x in parsed),
                         parsed_nonempty_feasible=len(useful),
                         violation_rate=ev["eval_violation_rate"],
                         feasible_value_per_sample=sum(C.task_term(x, cfg["vmax"])
                                                       for x in useful) / n))
        if cfg["arm"] != "binary" or float(cfg.get("w_task", 1)) != 1:
            continue
        mixed, spreads = [], []
        for rec in run["train"]:
            for group in rec["groups"]:
                if group["has_violator"] and not group["all_violating"]:
                    mixed.append([None if x is None else tuple(x) for x in group["selections"]])
                if np.ptp(group["rewards"]) > RW.ZERO_SPREAD_TOL:
                    spreads.append(float(np.std(group["rewards"], ddof=1)))
        crossover = max(float(C.output_table(cfg["vmax"], i)["task"].max())
                        for i in range(len(C.INSTANCES)))
        weights = sorted({float(cfg["w"]), crossover, math.floor(crossover) + 1})
        for weight in weights:
            margins = []
            for items in mixed:
                rewards = RW.group_rewards(items, "binary", weight, cfg["vmax"],
                                           spec=tasks.get("capacity"))
                feasible = np.array([C.verdict(x, cfg["vmax"]) for x in items])
                margins.append(float(rewards[feasible].min() - rewards[~feasible].max()))
            replays.append(dict(regime=cfg["regime"], arm=analyze.arm_name(cfg), seed=cfg["seed"],
                                replay_weight=weight, mixed_groups=len(margins),
                                mixed_groups_without_strict_feasible_dominance=sum(m <= 0 for m in margins),
                                minimum_reward_margin=min(margins) if margins else None,
                                nonzero_spread_groups=len(spreads),
                                groups_with_sample_sd_below_one=sum(sd < 1 for sd in spreads),
                                minimum_nonzero_sd=min(spreads) if spreads else None,
                                median_nonzero_sd=float(np.median(spreads)) if spreads else None))
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[(row["regime"], row["arm"])].append(row)
    uncertainty = []
    for (regime, arm), arm_rows in sorted(by_arm.items()):
        success = sum(row["violation_rate"] == 0 for row in arm_rows)
        lower, upper = wilson_interval(success, len(arm_rows))
        uncertainty.append(dict(regime=regime, arm=arm, seeds=len(arm_rows),
                                final_zero_violation_seeds=success,
                                wilson_95_lower=lower, wilson_95_upper=upper))
    return rows, replays, uncertainty


def write_csv(path, rows):
    """Each CSV keeps one observation per row and retains its denominator."""
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runs", type=Path)
    args = parser.parse_args()
    report = {"arithmetic": arithmetic_checks()}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.runs:
        quality, replay, uncertainty = saved_run_checks(args.runs)
        write_csv(args.out / "capacity_quality.csv", quality)
        write_csv(args.out / "mixed_group_replay.csv", replay)
        write_csv(args.out / "seed_uncertainty.csv", uncertainty)
        report["recorded_runs"] = len(quality)
        report["fully_feasible_runs"] = sum(row["violation_rate"] == 0 for row in quality)
        report["fully_feasible_all_empty_runs"] = sum(row["violation_rate"] == 0 and
                                                     row["parsed_empty"] == row["evaluation_samples"]
                                                     for row in quality)
        report["inference_limits"] = [
            "Parsed selections cannot certify the original completion's format.",
            "Weight replay changes recorded rewards and does not predict trained-policy outcomes.",
            "Wilson intervals use training seeds as observations and do not establish remedy rankings.",
        ]
    (args.out / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Revision arithmetic checks passed.")
    if args.runs:
        print(f"The audit read {report['recorded_runs']} saved runs.")


if __name__ == "__main__":
    main()
