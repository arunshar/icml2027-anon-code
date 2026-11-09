#!/usr/bin/env python3
"""Pair each yoked control run with its verifier-filter run, score the two gates, then report.

Each control run names its schedule in `config.json`. The schedule entry in
configs/keep_schedules.json names the filter run it copies and that run's metrics hash, so this
script pairs the two without a lookup table of its own.

Two gates come first, and nothing else is written until both hold.
  Y1  The control takes an optimizer step at exactly the steps its schedule lists, and at each of
      those steps it keeps exactly the listed prompts.
  Y2  The control logs the same answers, rewards and advantages as its filter run at every step up
      to and including the first update step at which the two keep different prompts. A control
      whose schedule keeps the filter's own prompts at every step must equal the filter run over
      every step, every evaluation and the final parameter checksum. Y2 is a gate on the speed task.
      On the capacity task a difference is admitted as drift only when no step at which the two
      runs sample the same answers logs different rewards or advantages.

The --gates-only mode prints the gate results and never prints an evaluation value. The full mode
writes, for every control run and every source run, the kept sets, the retained completions and
tokens, the step counts by gradient kind, the kept count per prompt, whether the first update keeps
a mixed group, and the final violation, parse-failure and empty-selection counts.

Usage:
  python analyze_yoked.py --gates-only --runs runs/filter_yoked_speed runs/filter_yoked_capacity
  python analyze_yoked.py --runs runs/filter_yoked_speed runs/filter_yoked_capacity --out runs/filter_yoked_analysis
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import analyze
import tasks

REPO_ROOT = analyze._REPO_ROOT
SCHEDULES_PATH = analyze.KEEP_SCHEDULES_PATH
# Two runs must share these per-group fields to count as the same run at one step. They are the
# fields of the fingerprint view in analyze.py, and the answers are compared as well.
COMPARED_GROUP_KEYS = ("rewards", "adv", "all_violating", "has_violator")
GATE_FAILED = 6


class PairingError(ValueError):
    """A control run cannot be paired with its source run, so nothing about it can be scored."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_entries(path=None) -> dict:
    path = Path(path) if path is not None else SCHEDULES_PATH
    return json.loads(path.read_text())["schedules"]


def logged_kept_sets(run) -> list:
    """Return the prompts a run kept at each logged training step, as sorted lists."""
    return [sorted(g["prompt_index"] for g in rec["groups"] if g.get("kept", True))
            for rec in run["train"]]


def pair(control, entries, repo_root=None):
    """Return (entry, source run) for one control run, or raise PairingError."""
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
    sid = control["cfg"].get("keep_schedule")
    if sid not in entries:
        raise PairingError(f"{control['dir'].name}: schedule {sid!r} is not in the schedule file")
    entry = entries[sid]
    d = repo_root / entry["source_run"]
    if not (d / "DONE").exists():
        raise PairingError(f"{control['dir'].name}: source run {entry['source_run']} has no DONE marker")
    if _sha256(d / "metrics.jsonl") != entry["source_metrics_sha256"]:
        raise PairingError(f"{control['dir'].name}: source run {entry['source_run']} has a "
                           f"metrics.jsonl that differs from the one the schedule was written from")
    source = analyze._load_run(d)
    if logged_kept_sets(source) != entry["source_kept_sets"]:
        raise PairingError(f"{control['dir'].name}: the source run's logged kept sets differ from "
                           f"the ones the schedule records")
    return entry, source


def score_y1(control, entry) -> dict:
    """Score Y1, which asks whether the control steps and keeps exactly as its schedule says."""
    planned = entry["kept_sets"]
    train = control["train"]
    if len(train) != len(planned):
        return {"pass": False, "first_failure_step": None,
                "reason": f"{len(train)} training records for {len(planned)} scheduled steps"}
    for i, (rec, want) in enumerate(zip(train, planned), start=1):
        indices = [g["prompt_index"] for g in rec["groups"]]
        kept = sorted(g["prompt_index"] for g in rec["groups"] if g.get("kept", True))
        problems = []
        if rec.get("step") != i:
            problems.append("step")
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            problems.append("prompt_index")
        if kept != list(want):
            problems.append("kept")
        if rec.get("opt_step_taken") is not bool(want):
            problems.append("opt_step_taken")
        if rec.get("n_kept_groups") != len(want):
            problems.append("n_kept_groups")
        if rec.get("keep_schedule_n") != len(want):
            problems.append("keep_schedule_n")
        if problems:
            return {"pass": False, "first_failure_step": i, "reason": ", ".join(problems)}
    return {"pass": True, "first_failure_step": None, "reason": None}


def _group_view(run, rec):
    key = tasks.get(run["cfg"].get("task", "speed")).ITEMS_KEY
    return [dict({k: g.get(k) for k in COMPARED_GROUP_KEYS}, items=g.get(key)) for g in rec["groups"]]


def _first_group_difference(a, b, through: int):
    """Return the first step up to `through` at which the two runs log different groups, or None."""
    for i, (ra, rb) in enumerate(zip(a["train"][:through], b["train"][:through]), start=1):
        if _group_view(a, ra) != _group_view(b, rb):
            return i
    if len(a["train"]) < through or len(b["train"]) < through:
        return min(len(a["train"]), len(b["train"])) + 1
    return None


def _logic_break(a, b):
    """Return the first step at which the two runs sample the same answers and still log different
    rewards or advantages, or None. No drift in the weights can explain such a step."""
    for i, (ra, rb) in enumerate(zip(a["train"], b["train"]), start=1):
        va, vb = _group_view(a, ra), _group_view(b, rb)
        if len(va) == len(vb) and all(x["items"] == y["items"] for x, y in zip(va, vb)) and any(
                x[k] != y[k] for x, y in zip(va, vb) for k in ("rewards", "adv")):
            return i
    return None


def _evals(run):
    key = tasks.get(run["cfg"].get("task", "speed")).EVAL_ITEMS_KEY
    return [(e["step"], e.get(key)) for e in run["evals"]]


def score_y2(control, source, entry) -> dict:
    """Score Y2, which asks whether the control equals its source run through the step at which
    their kept sets first differ.

    The shared window ends at that step, because both runs sample the step's answers from the same
    weights and then update differently. A control whose kept sets equal the source's at every step
    is a full copy and must equal the source over every step, every evaluation and the checksum."""
    task = control["cfg"].get("task", "speed")
    mine, theirs = entry["kept_sets"], entry["source_kept_sets"]
    first_update = next((t for t, s in enumerate(mine, start=1) if s), None)
    diverge = next((t for t, (a, b) in enumerate(zip(mine, theirs), start=1) if a != b), None)
    full_copy = diverge is None
    if full_copy:
        kind = "full_copy"
    elif first_update is not None and diverge > first_update:
        kind = "first_update_copy"
    else:
        kind = "distinct_first_update"
    through = len(mine) if full_copy else diverge
    first_diff = _first_group_difference(control, source, through)
    same = first_diff is None
    if full_copy and same:
        same = (_evals(control) == _evals(source)
                and control["prov"].get("final_param_checksum") == source["prov"].get("final_param_checksum"))
        if not same:
            first_diff = "evaluations or checksum"
    if same:
        result = "pass"
    elif task == "speed":
        result = "fail"
    else:
        result = "drift" if _logic_break(control, source) is None else "fail"
    return {"kind": kind, "compared_through_step": through, "first_update_step": first_update,
            "first_difference": first_diff, "result": result}


def gates(controls, entries, repo_root=None):
    """Score Y1 and Y2 for every control run. Returns (rows, pairs, problems)."""
    rows, pairs, problems = [], [], []
    for c in controls:
        try:
            entry, source = pair(c, entries, repo_root)
        except PairingError as e:
            problems.append(str(e))
            continue
        y1 = score_y1(c, entry)
        y2 = score_y2(c, source, entry)
        task = c["cfg"].get("task", "speed")
        y2_holds = y2["result"] == "pass" or (task != "speed" and y2["result"] == "drift")
        rows.append({"run": c["dir"].name, "keep_schedule": c["cfg"]["keep_schedule"], "task": task,
                     "source_run": entry["source_run"], "y1_pass": y1["pass"],
                     "y1_first_failure_step": y1["first_failure_step"], "y1_reason": y1["reason"],
                     "y2_kind": y2["kind"], "y2_result": y2["result"],
                     "y2_compared_through_step": y2["compared_through_step"],
                     "y2_first_difference": y2["first_difference"],
                     "gates_hold": bool(y1["pass"] and y2_holds)})
        pairs.append((c, source, entry))
    return rows, pairs, problems


def _final_counts(run) -> dict:
    """Return the final evaluation's rates and its unparseable and empty answers, as analyze.py
    counts them."""
    ev = run["evals"][-1] if run["evals"] else {}
    answers = analyze.final_answers([run])
    counts = answers[0] if answers else {}
    return {"eval_step_final": ev.get("step"),
            "eval_violation_rate_final": ev.get("eval_violation_rate"),
            "eval_parse_fail_rate_final": ev.get("eval_parse_fail_rate"),
            "feasible_final": (ev.get("eval_violation_rate") == 0.0) if ev else None,
            "eval_unparseable_final": counts.get("eval_unparseable_final"),
            "eval_empty_selections_final": counts.get("eval_empty_selections_final")}


def run_report(run, role: str, entry=None) -> dict:
    """Return one row of per-run facts. None of them is a gate, and every one comes from the log."""
    cfg = run["cfg"]
    kl_beta = float(cfg.get("kl_beta", 0.0))
    kept_sets = logged_kept_sets(run)
    update_steps = [rec["step"] for rec in run["train"] if analyze._step_gradient(rec, kl_beta) != "skipped"]
    states = [analyze._step_gradient(rec, kl_beta) for rec in run["train"]]
    retained_completions = retained_tokens = 0
    per_prompt = {}
    first_mixed = None
    for rec in run["train"]:
        for g in rec["groups"]:
            per_prompt.setdefault(g["prompt_index"], 0)
            if not g.get("kept", True):
                continue
            per_prompt[g["prompt_index"]] += 1
            retained_completions += len(g.get("rewards", []))
            retained_tokens += sum(g.get("n_tokens") or [])
    if update_steps:
        rec = run["train"][update_steps[0] - 1]
        first_mixed = any(g.get("kept", True) and g.get("has_violator") and not g.get("all_violating")
                          for g in rec["groups"])
    counts = analyze._update_counts(run)
    row = {"run": run["dir"].name, "role": role, "task": cfg.get("task", "speed"),
           "arm": analyze.arm_name(cfg), "w": cfg["w"], "seed": cfg["seed"],
           "keep_schedule": cfg.get("keep_schedule"),
           "draw": analyze.keep_schedule_draw(cfg),
           "source_run": entry["source_run"] if entry else None,
           "update_steps": " ".join(str(t) for t in update_steps),
           "n_update_steps": len(update_steps),
           "nonzero_gradient_steps": states.count("nonzero"),
           "zero_gradient_steps": states.count("zero"),
           "momentum_only_steps": counts["momentum_only_steps"],
           "skipped_steps": states.count("skipped"),
           "retained_groups": sum(len(s) for s in kept_sets),
           "retained_completions": retained_completions,
           "retained_tokens": retained_tokens,
           "kept_per_prompt": " ".join(f"{k}:{v}" for k, v in sorted(per_prompt.items())),
           "first_update_keeps_mixed_group": first_mixed}
    row.update(_final_counts(run))
    return row


def _load_controls(roots):
    controls = []
    for root in roots:
        for r in analyze.load_runs(Path(root)):
            if r["cfg"].get("keep_schedule") is not None:
                controls.append(r)
    return controls


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="assembled directories of runs")
    ap.add_argument("--out", help="directory for the full report, which --gates-only does not need")
    ap.add_argument("--gates-only", action="store_true",
                    help="score Y1 and Y2 and print them, without any evaluation value or file")
    ap.add_argument("--schedules", default=None, help="schedule file, by default the committed one")
    a = ap.parse_args(argv)
    if not a.gates_only and not a.out:
        ap.error("--out is required unless --gates-only is given")
    entries = load_entries(a.schedules)
    controls = _load_controls(a.runs)
    if not controls:
        print("REFUSED: no yoked control run was found", file=sys.stderr)
        return 2
    rows, pairs, problems = gates(controls, entries)
    if problems:
        print("REFUSED: some control runs cannot be paired with their source runs:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 2
    held = all(r["gates_hold"] for r in rows)
    summary = {"controls": len(rows), "y1_failures": sum(not r["y1_pass"] for r in rows),
               "y2_failures": sum(r["y2_result"] == "fail" for r in rows),
               "y2_capacity_drift": sum(r["y2_result"] == "drift" for r in rows),
               "gates_hold": held}
    if a.gates_only:
        for r in rows:
            print(json.dumps(r, sort_keys=True))
        print(json.dumps({"summary": summary}, sort_keys=True))
        return 0 if held else GATE_FAILED
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gates.json").write_text(json.dumps({"summary": summary, "runs": rows}, indent=2,
                                               sort_keys=True) + "\n")
    if not held:
        print("REFUSED: a gate failed, so only gates.json was written", file=sys.stderr)
        return GATE_FAILED
    report, kept, seen_sources = [], {}, set()
    for control, source, entry in pairs:
        report.append(run_report(control, "control", entry))
        kept[control["dir"].name] = {str(t): s for t, s in
                                     enumerate(logged_kept_sets(control), start=1) if s}
        if entry["source_run"] not in seen_sources:
            seen_sources.add(entry["source_run"])
            report.append(run_report(source, "source"))
            kept[entry["source_run"]] = {str(t): s for t, s in
                                         enumerate(logged_kept_sets(source), start=1) if s}
    with (out / "per_run.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    (out / "kept_sets.json").write_text(json.dumps(kept, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "rows": len(report), "out": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
