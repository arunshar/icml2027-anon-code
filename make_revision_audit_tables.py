#!/usr/bin/env python3
"""Generate the revision's task-quality table and anonymous audit records.

The generator reads completed capacity-hard runs through the standalone CPU audit.
It keeps training seeds as the observations for feasibility intervals. Task quality
uses every final evaluation completion, so invalid and infeasible outputs contribute
zero. The emitted files contain no filesystem paths or machine metadata.
The --records option rebuilds the table from the included anonymous records without
requiring the original training logs. The default run directory is runs/capacity_audit,
which holds the reward-design runs, the remedy runs, and the KL-penalty runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = next(path for path in (HERE.parent / "experiments/cancellation", HERE, HERE.parent)
           if (path / "audit_revision_checks.py").is_file())
sys.path.insert(0, str(EXP))

import audit_revision_checks as audit  # noqa: E402

ORDER = ["graded", "binary", "binary_1000x", "none", "verdict_only", "graded_no_task",
         "binary_drgrpo", "binary_ngrpo", "binary_vfilter", "binary_sfilter",
         "graded_kl", "binary_kl", "none_kl"]
LABELS = {
    "graded": r"residual $+$ task",
    "binary": r"verdict $+$ task",
    "binary_1000x": r"verdict $\times 1000$ $+$ task",
    "none": "task only",
    "verdict_only": "verdict only",
    "graded_no_task": "residual only",
    "binary_drgrpo": r"verdict $+$ task, Dr.~GRPO",
    "binary_ngrpo": "verdict $+$ task, NGRPO, feasible anchor",
    "binary_vfilter": "verdict $+$ task, verifier filter",
    "binary_sfilter": "verdict $+$ task, identical-reward filter",
    "graded_kl": "residual $+$ task, KL penalty",
    "binary_kl": "verdict $+$ task, KL penalty",
    "none_kl": "task only, KL penalty",
}
# LaTeX macro names hold letters only, so each arm maps to a word.
WORDS = {
    "graded": "Graded", "binary": "Binary", "binary_1000x": "BinaryHeavy", "none": "None",
    "verdict_only": "VerdictOnly", "graded_no_task": "ResidualOnly",
    "binary_drgrpo": "BinaryDrGrpo", "binary_ngrpo": "BinaryNgrpo",
    "binary_vfilter": "BinaryVFilter", "binary_sfilter": "BinarySFilter",
    "graded_kl": "GradedKl", "binary_kl": "BinaryKl", "none_kl": "NoneKl",
}


def arm_words():
    """Every arm in ORDER maps to its macro word, and an arm without one is refused."""
    missing = [arm for arm in ORDER if arm not in WORDS]
    if missing:
        raise ValueError(f"Every arm in ORDER needs a macro word. Missing: {missing}.")
    return {arm: WORDS[arm] for arm in ORDER}


def aggregate_quality(rows):
    """Each arm's aggregate uses its complete set of recorded seeds."""
    by_arm = defaultdict(list)
    identifiers = set()
    sample_counts = {row["evaluation_samples"] for row in rows}
    if len(sample_counts) != 1 or next(iter(sample_counts)) <= 0:
        raise ValueError("Every run must have the same positive evaluation sample count.")
    for row in rows:
        if row["regime"] != "H":
            raise ValueError("The revision table accepts capacity-hard runs only.")
        identifier = (row["arm"], row["seed"])
        if identifier in identifiers:
            raise ValueError("The run set contains a duplicate arm and seed.")
        identifiers.add(identifier)
        n = row["evaluation_samples"]
        counts = [row[name] for name in
                  ("seed", "evaluation_samples", "unparseable", "parsed_empty", "parsed_nonempty_feasible")]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Seed identifiers and completion counts must be nonnegative integers.")
        valid = row["parsed_empty"] + row["parsed_nonempty_feasible"]
        if valid + row["unparseable"] > n:
            raise ValueError("The completion categories exceed their evaluation denominator.")
        rate, value = row["violation_rate"], row["feasible_value_per_sample"]
        if not (math.isfinite(rate) and math.isfinite(value)):
            raise ValueError("The recorded rates and task values must be finite.")
        if not math.isclose(rate, (n - valid) / n, rel_tol=0, abs_tol=1e-12):
            raise ValueError("The violation rate disagrees with the feasible completion count.")
        if not 0 <= value <= row["parsed_nonempty_feasible"] / n + 1e-12:
            raise ValueError("The feasible task value exceeds the nonempty feasible fraction.")
        by_arm[row["arm"]].append(row)
    if set(by_arm) != set(ORDER):
        raise ValueError("The revision table requires every arm in ORDER among the recorded "
                         "hard-capacity runs, and no other arm.")

    aggregates = []
    for arm in ORDER:
        seeds = by_arm[arm]
        count = len(seeds)
        if {row["seed"] for row in seeds} != set(range(count)):
            raise ValueError("The seed observations must form a complete sequence from zero.")
        successes = sum(row["violation_rate"] == 0 for row in seeds)
        lower, upper = audit.wilson_interval(successes, count)
        samples = sum(row["evaluation_samples"] for row in seeds)
        nonempty = sum(row["parsed_nonempty_feasible"] for row in seeds)
        value = sum(row["feasible_value_per_sample"] * row["evaluation_samples"]
                    for row in seeds) / samples
        assert 0 <= nonempty <= samples and 0 <= value <= 1
        assert 0 <= lower <= upper <= 1
        aggregates.append(dict(arm=arm, seeds=count, zero_violation_seeds=successes,
                               wilson_95_lower=lower, wilson_95_upper=upper,
                               evaluation_samples=samples,
                               parsed_nonempty_feasible=nonempty,
                               nonempty_feasible_fraction=nonempty / samples,
                               normalized_feasible_value=value))
    return aggregates


def seed_extremes(quality_rows):
    """Each arm gets its smallest and largest per-seed nonempty-feasible fraction and feasible value.

    The table prints these beside each arm's mean so a reader can see how far the seeds spread.
    The archived CSV files keep their schemas, so this reads the per-seed rows directly."""
    by_arm = defaultdict(list)
    for row in quality_rows:
        by_arm[row["arm"]].append(row)
    extremes = {}
    for arm, rows in by_arm.items():
        fractions = [row["parsed_nonempty_feasible"] / row["evaluation_samples"] for row in rows]
        values = [row["feasible_value_per_sample"] for row in rows]
        extremes[arm] = dict(nonempty=(min(fractions), max(fractions)),
                             value=(min(values), max(values)))
    return extremes


def read_records_csv(path, fields, float_fields=(), nullable=()):
    """The archived records use a fixed schema with explicit numeric types."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"{path.name} has missing, repeated, or unexpected columns.")
        rows = []
        for raw in reader:
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"{path.name} contains an incomplete CSV observation.")
            row = {}
            for name, value in raw.items():
                if name in ("arm", "regime"):
                    row[name] = value
                elif name in nullable and value == "":
                    row[name] = None
                elif name in float_fields:
                    row[name] = float(value)
                    if not math.isfinite(row[name]):
                        raise ValueError(f"{path.name} contains a nonfinite value.")
                else:
                    row[name] = int(value)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} has no observations.")
    return rows


def check_archived_rows(actual, expected, keys):
    """The archived aggregate files must agree with recomputed per-seed values."""
    def index(rows):
        indexed = {}
        for row in rows:
            key = tuple(row[name] for name in keys)
            if key in indexed:
                raise ValueError("The archived CSV contains duplicate observations.")
            indexed[key] = row
        return indexed

    observed, recomputed = index(actual), index(expected)
    if observed.keys() != recomputed.keys():
        raise ValueError("The archived CSV has missing or unexpected observations.")
    for key in observed:
        if observed[key].keys() != recomputed[key].keys():
            raise ValueError("The archived CSV has inconsistent fields.")
        for name, value in observed[key].items():
            target = recomputed[key][name]
            equal = (math.isclose(value, target, rel_tol=0, abs_tol=1e-12)
                     if isinstance(target, float) else value == target)
            if not equal:
                raise ValueError("The archived aggregate disagrees with the per-seed observations.")


def load_records(directory):
    """The per-seed records supply all table values, while the other files check completeness."""
    quality_fields = ["regime", "arm", "seed", "evaluation_samples", "unparseable", "parsed_empty",
                      "parsed_nonempty_feasible", "violation_rate", "feasible_value_per_sample"]
    quality = read_records_csv(directory / "capacity_quality_per_seed.csv", quality_fields,
                               ("violation_rate", "feasible_value_per_sample"))
    aggregates = aggregate_quality(quality)
    aggregate_floats = ("wilson_95_lower", "wilson_95_upper", "nonempty_feasible_fraction",
                        "normalized_feasible_value")
    archived = read_records_csv(directory / "capacity_quality_by_arm.csv", list(aggregates[0]),
                                aggregate_floats)
    check_archived_rows(archived, aggregates, ("arm",))
    uncertainty = [dict(regime="H", arm=row["arm"], seeds=row["seeds"],
                        final_zero_violation_seeds=row["zero_violation_seeds"],
                        wilson_95_lower=row["wilson_95_lower"], wilson_95_upper=row["wilson_95_upper"])
                   for row in sorted(aggregates, key=lambda row: row["arm"])]
    archived_intervals = read_records_csv(directory / "seed_uncertainty.csv", list(uncertainty[0]),
                                          ("wilson_95_lower", "wilson_95_upper"))
    check_archived_rows(archived_intervals, uncertainty, ("regime", "arm"))

    replay_fields = ["regime", "arm", "seed", "replay_weight", "mixed_groups",
                     "mixed_groups_without_strict_feasible_dominance", "minimum_reward_margin",
                     "nonzero_spread_groups", "groups_with_sample_sd_below_one",
                     "minimum_nonzero_sd", "median_nonzero_sd"]
    nullable = ("minimum_reward_margin", "minimum_nonzero_sd", "median_nonzero_sd")
    replay = read_records_csv(directory / "mixed_group_replay.csv", replay_fields,
                              ("replay_weight",) + nullable, nullable)
    expected_seeds = {(row["arm"], row["seed"]) for row in quality if row["arm"].startswith("binary")}
    replay_keys, replay_seeds = set(), defaultdict(set)
    for row in replay:
        key = (row["arm"], row["seed"], row["replay_weight"])
        if key in replay_keys:
            raise ValueError("The replay CSV contains duplicate observations.")
        replay_keys.add(key)
        if row["regime"] != "H" or key[:2] not in expected_seeds or row["replay_weight"] < 0:
            raise ValueError("The replay CSV contains an unexpected observation.")
        replay_seeds[key[:2]].add(row["replay_weight"])
        if not (0 <= row["mixed_groups_without_strict_feasible_dominance"] <= row["mixed_groups"]
                and 0 <= row["groups_with_sample_sd_below_one"] <= row["nonzero_spread_groups"]):
            raise ValueError("The replay CSV has inconsistent group counts.")
        if (row["minimum_reward_margin"] is None) != (row["mixed_groups"] == 0):
            raise ValueError("The replay CSV omits a required reward margin.")
        if any((row[name] is None) != (row["nonzero_spread_groups"] == 0)
               for name in ("minimum_nonzero_sd", "median_nonzero_sd")):
            raise ValueError("The replay CSV omits a required spread statistic.")
    if set(replay_seeds) != expected_seeds or any(len(weights) != 3 for weights in replay_seeds.values()):
        raise ValueError("The replay CSV has missing seed or weight observations.")
    report = json.loads((directory / "audit_report.json").read_text())
    if not isinstance(report.get("summary"), dict):
        raise ValueError("The audit report does not contain its recorded summary.")
    return quality, replay, uncertainty, report["summary"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--runs", type=Path,
                        help="Read the original completed run directory "
                             "(default runs/capacity_audit beside the audit).")
    source.add_argument("--records", type=Path,
                        help="Rebuild from the included sanitized CSV and JSON records.")
    parser.add_argument("--out", type=Path, default=HERE / "tables")
    parser.add_argument("--audit-out", type=Path, default=HERE / "revision_audit")
    args = parser.parse_args(argv)
    arithmetic = audit.arithmetic_checks()
    archived_summary = None
    if args.records:
        quality, replay, uncertainty, archived_summary = load_records(args.records)
    else:
        quality, replay, uncertainty = audit.saved_run_checks(args.runs or EXP / "runs/capacity_audit")
    aggregated = aggregate_quality(quality)
    extremes = seed_extremes(quality)
    words = arm_words()

    feasible = [row for row in quality if row["violation_rate"] == 0]
    all_empty = [row for row in feasible
                 if row["parsed_empty"] == row["evaluation_samples"]]
    useful = [row for row in feasible if row["parsed_nonempty_feasible"] > 0]
    samples_per_seed = quality[0]["evaluation_samples"]
    summary = dict(recorded_runs=len(quality), arms=len(aggregated),
                   fully_feasible_runs=len(feasible),
                   fully_feasible_all_empty_runs=len(all_empty),
                   fully_feasible_runs_with_nonempty_outputs=len(useful),
                   evaluation_samples_per_seed=samples_per_seed,
                   nonempty_feasible_samples_all_runs=sum(row["parsed_nonempty_feasible"]
                                                          for row in quality),
                   total_evaluation_samples=sum(row["evaluation_samples"] for row in quality))
    if archived_summary is not None and (archived_summary != summary or
                                        any(type(value) is not int for value in archived_summary.values())):
        raise ValueError("The archived summary has missing observations or inconsistent counts.")
    args.out.mkdir(parents=True, exist_ok=True)
    args.audit_out.mkdir(parents=True, exist_ok=True)
    audit.write_csv(args.audit_out / "capacity_quality_per_seed.csv", quality)
    audit.write_csv(args.audit_out / "capacity_quality_by_arm.csv", aggregated)
    audit.write_csv(args.audit_out / "mixed_group_replay.csv", replay)
    audit.write_csv(args.audit_out / "seed_uncertainty.csv", uncertainty)
    report = dict(summary=summary, arithmetic=arithmetic, definitions={
        "seed_success": "A seed has zero violation across its complete final evaluation.",
        "wilson_interval": "Each training seed supplies one observation for the descriptive 95% Wilson interval.",
        "nonempty_feasible_fraction": "The numerator counts parsed nonempty feasible outputs. The denominator includes every evaluation completion.",
        "normalized_feasible_value": "Feasible parsed outputs receive their value divided by the instance optimum. Other outputs receive zero.",
        "parser_limit": "The saved parsed selections cannot certify the original output format.",
        "replay_limit": "Counterfactual weights rescore recorded groups and do not estimate retrained-policy outcomes.",
    })
    (args.audit_out / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # A justified p column hyphenated a wrapped label, so the label column is ragged right.
    lines = [r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{>{\raggedright\arraybackslash}p{4.8cm}ccc}", r"\toprule",
             r"\textbf{Reward design and component} & "
             r"\shortstack{\textbf{feasible seeds}\\\textbf{[95\% interval]}} & "
             r"\shortstack{\textbf{nonempty}\\\textbf{feasible}} & "
             r"\shortstack{\textbf{feasible}\\\textbf{value}} \\", r"\midrule"]
    for row in aggregated:
        interval = f'{row["wilson_95_lower"]:.2f}, {row["wilson_95_upper"]:.2f}'
        feasible_cell = f'{row["zero_violation_seeds"]}/{row["seeds"]} [{interval}]'
        low, high = extremes[row["arm"]]["nonempty"]
        nonempty = f'{row["nonempty_feasible_fraction"]:.4f} [{low:.2f}, {high:.2f}]'
        low, high = extremes[row["arm"]]["value"]
        value = f'{row["normalized_feasible_value"]:.4f} [{low:.2f}, {high:.2f}]'
        lines.append(f'{LABELS[row["arm"]]} & {feasible_cell} & {nonempty} & {value} ' + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (args.out / "revision_quality.tex").write_text("\n".join(lines) + "\n")

    macros = {
        "revAuditRuns": summary["recorded_runs"],
        "revAuditArms": summary["arms"],
        "revAuditFeasibleRuns": summary["fully_feasible_runs"],
        "revAuditEmptyFeasibleRuns": summary["fully_feasible_all_empty_runs"],
        "revAuditUsefulFeasibleRuns": summary["fully_feasible_runs_with_nonempty_outputs"],
        "revAuditEvalSamples": samples_per_seed,
        "revAuditNonemptySamples": summary["nonempty_feasible_samples_all_runs"],
        "revAuditTotalSamples": summary["total_evaluation_samples"],
        "revAuditConfidencePercent": 95,
    }
    if len(useful) == 1:
        macros["revAuditUsefulSeedNonempty"] = useful[0]["parsed_nonempty_feasible"]
        macros["revAuditUsefulSeedSamples"] = useful[0]["evaluation_samples"]
        macros["revAuditUsefulSeedValue"] = f'{useful[0]["feasible_value_per_sample"]:.4f}'
    for row in aggregated:
        word = words[row["arm"]]
        macros[f"revAuditFeasSeeds{word}"] = row["zero_violation_seeds"]
        macros[f"revAuditSeeds{word}"] = row["seeds"]
        macros[f"revAuditNonempty{word}"] = f'{row["nonempty_feasible_fraction"]:.4f}'
        macros[f"revAuditValue{word}"] = f'{row["normalized_feasible_value"]:.4f}'
    macro_text = "% The generator computes these values from the recorded capacity runs.\n"
    macro_text += "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items())
    (args.out / "revision_macros.tex").write_text(macro_text)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
