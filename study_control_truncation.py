#!/usr/bin/env python3
"""Split the strict parse failures of the verdict-only control of the GSM8K study into cut and finished completions.

This is a descriptive measurement made after the study was scored, and no scorer reads it. For each control row (arm
verdict_only) the script reads the two random held-out evaluations of metrics.jsonl, one at the first step and one at the
last step. Each evaluation lists its completions as items. A completion counts as cut when its logged finished flag is false,
which means that no end-of-text token came before the generation limit. A completion counts as parsed when its logged strict
value is not null. The script also counts the completions that have at least as many tokens as the limit. That count differs
from the cut count only when a completion ends exactly at the limit.

Usage (from experiments/cancellation):
  python study_control_truncation.py --runs runs/reasoning_icml_v2/runs/6322778 [--limit 512] [--steps 0 300]

The output is a table with one line for each seed and step. Exit status 2 means that a run folder or an evaluation is missing or
appears twice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = HERE / "runs" / "reasoning_icml_v2" / "runs" / "6322778"
SEEDS = (0, 1, 2)
COLUMNS = ("seed", "step", "completions", "parsed", "unparsed", "cut", "unparsed_cut", "finished", "parsed_of_finished",
           "at_token_limit", "mean_tokens", "cut_share", "parsed_share_of_finished")


class Refusal(Exception):
    """A run folder or an evaluation record is missing or appears twice."""


def control_dir(runs: Path, seed: int) -> Path:
    matches = sorted(Path(runs).glob(f"*_study_0.5b_verdict_only_s{seed}"))
    if len(matches) != 1:
        raise Refusal(f"{runs} holds {len(matches)} control folders for seed {seed}, and the study has 1")
    return matches[0]


def read_evaluation(metrics: Path, step: int) -> tuple:
    """Return the item fields and the items of the one random held-out evaluation that was logged at the given step."""
    found = []
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("phase") == "eval" and record.get("set") == "random" and record.get("step") == step:
            found.append(record)
    if len(found) != 1:
        raise Refusal(f"{metrics} holds {len(found)} random held-out evaluations at step {step}, and the run has 1")
    return found[0]["item_fields"], found[0]["items"]


def split(fields: list, items: list, limit: int) -> dict:
    """Count the parsed, cut and finished completions of one evaluation."""
    ix = {name: i for i, name in enumerate(fields)}
    for needed in ("n_tokens", "finished", "strict"):
        if needed not in ix:
            raise Refusal(f"the item fields {fields} do not hold {needed}")
    n = len(items)
    parsed = [it[ix["strict"]] is not None for it in items]
    cut = [not it[ix["finished"]] for it in items]
    finished = n - sum(cut)
    parsed_finished = sum(1 for p, c in zip(parsed, cut) if p and not c)
    return {"completions": n, "parsed": sum(parsed), "unparsed": n - sum(parsed), "cut": sum(cut),
            "unparsed_cut": sum(1 for p, c in zip(parsed, cut) if c and not p), "finished": finished,
            "parsed_of_finished": parsed_finished, "at_token_limit": sum(1 for it in items if it[ix["n_tokens"]] >= limit),
            "mean_tokens": sum(it[ix["n_tokens"]] for it in items) / n,
            "cut_share": sum(cut) / n, "parsed_share_of_finished": parsed_finished / finished if finished else float("nan")}


def table(runs: Path, steps: tuple, limit: int) -> list:
    rows = []
    for seed in SEEDS:
        for step in steps:
            fields, items = read_evaluation(control_dir(runs, seed) / "metrics.jsonl", step)
            rows.append({"seed": seed, "step": step, **split(fields, items, limit)})
    return rows


def render(rows: list) -> str:
    lines = [" ".join(COLUMNS)]
    for row in rows:
        cells = []
        for name in COLUMNS:
            value = row[name]
            if name == "mean_tokens":
                cells.append(f"{value:.4f}")
            elif name in ("cut_share", "parsed_share_of_finished"):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append(" ".join(cells))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=str(DEFAULT_RUNS), help="the folder of the twelve run folders")
    ap.add_argument("--limit", type=int, default=512, help="the generation limit in tokens")
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 300], help="the steps of the evaluations")
    args = ap.parse_args(argv)
    try:
        text = render(table(Path(args.runs), tuple(args.steps), args.limit))
    except (Refusal, OSError, KeyError, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
