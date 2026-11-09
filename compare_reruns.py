#!/usr/bin/env python3
"""This script compares the single-commit capacity reruns with the originals of the same level, arm and seed.

The capacity seeds 0 to 2 were made at two older commits, and seeds 3 and 4 were made at a newer
one. The reruns repeat seeds 0 to 2 at the newer commit, so this script pairs each rerun with its
original and reports how the two runs relate. It has three modes, and each mode reads only what it
needs, so no outcome is read before the gate that guards it.

The mode --files-only checks the pulled rerun files against the manifest rows. It prints problem
names and provenance values only, and it never prints an evaluation metric.

The mode --gate-only computes gate SC0 and nothing else. SC0 holds when every finished pair logs
identical answers, rewards and advantages at step 1, and no step logs identical answers in every
group with different rewards or advantages. It writes gate.json and exits 1 when SC0 fails.

The full mode computes SC0 again and stops there when it fails. When SC0 holds, it writes one row
per pair to pairs.csv with the divergence steps, the checksum match and the final labels. It also
writes summary.json with the agreement counts, the reference values and the band of each count.

Usage:
  python compare_reruns.py --files-only --rerun runs/<ID> --expect-sha <sha> --expect-digest <digest>
      --expect-rows 1-36
  python compare_reruns.py --gate-only --original <dir> --rerun runs/<ID> --expect-sha <sha>
      --expect-digest <digest> --out runs/<gate dir>
  python compare_reruns.py --original <dir> --rerun runs/<ID> --reference-cohort <dir>
      --expect-sha <sha> --expect-digest <digest> --out runs/<compare dir>

A source is a directory, optionally followed by a colon and a shell pattern on run directory names,
as in assemble_runs.py. The script exits 0 on a pass, 1 when a file check or SC0 fails, and 2 when it
refuses its inputs. A refusal writes nothing.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import analyze
import assemble_runs
import tasks

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "slurm" / "manifest_capacity.csv"

# Every rerun must hold these config values, which are the values of all 56 published capacity runs
# of the reward-design cohort. The row's own arm, weights, level, limit and seed are checked
# separately against the manifest.
EXPECTED_CONFIG = {
    "task": "capacity", "steps": 300, "eval_every": 10, "max_new_tokens": 128,
    "backward_per_group": True, "adv_variant": "grpo", "group_filter": "none", "kl_beta": 0.0,
    "device": "cuda", "batch_prompts": 6, "group_size": 4, "lr": 1e-05, "weight_decay": 0.0,
    "grad_clip": 1.0, "clip_coef": 0.2, "temperature": 1.0, "top_p": 0.95, "top_k": 0,
    "repetition_penalty": 1.0, "eval_samples": 16, "eval_seed": 12345,
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct", "no_ref": False, "tiny": False,
}
# Every published capacity run records this environment. The analysis compares each of these
# fields exactly, so a rerun with any other value could not join the existing capacity arrays.
EXPECTED_ENVIRONMENT = {
    "device": "cuda", "device_name": "NVIDIA A100-SXM4-40GB", "torch": "2.14.0+cu130",
    "transformers": "5.17.0", "numpy": "2.4.6", "python": "3.11.15",
    "platform": "Linux-4.18.0-553.157.1.el8_10.x86_64-x86_64-with-glibc2.28",
}
# Each level's original seeds 0 to 2 were made at the commit below. Every pair spans two commits by
# design, so the pair check exempts git_sha and checks each side against its own expected commit.
ORIGINAL_SHA = {"H": "COMMIT_1",
                "E": "COMMIT_2"}
MANIFEST_FIELDS = ("arm", "w", "w_task", "regime", "vmax", "seed")

LABELS = ("feasibility", "unparseable", "outcome_class")
# The central 95 percent range runs from the smallest count whose cumulative probability reaches
# the low bound to the smallest count whose cumulative probability reaches the high bound.
RANGE_LOW = Fraction(1, 40)
RANGE_HIGH = Fraction(39, 40)
# The gate output must not carry these keys, because a label or a divergence step read before SC0
# is recorded would break the order the prediction entry fixes.
OUTCOME_KEYS = ("feasible", "unparseable", "outcome_class", "label", "agree", "band", "reference",
                "first_answer_difference_step", "identical_answer_steps", "checksums_equal")


class Refusal(ValueError):
    """The inputs cannot be compared, so the script writes nothing and exits 2."""


# ---------------------------------------------------------------- inputs


def parse_rows(text: str) -> list:
    """This function returns the sorted row numbers of a list such as 1-36,57-60."""
    rows = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            first, last = int(lo), int(hi) if sep else int(lo)
        except ValueError:
            raise Refusal(f"row list {text!r} is not a list of numbers and ranges")
        if first < 1 or last < first:
            raise Refusal(f"row range {part!r} is empty or starts below 1")
        rows.update(range(first, last + 1))
    if not rows:
        raise Refusal("the row list is empty")
    return sorted(rows)


def read_manifest(path: Path) -> dict:
    """This function returns the manifest rows keyed by their task id."""
    with open(path, newline="") as f:
        return {int(r["task_id"]): r for r in csv.DictReader(f)}


def parse_not_run(texts) -> set:
    """This function reads identities written as level, arm and seed joined by colons, as in H:graded:1."""
    out = set()
    for text in texts or []:
        parts = text.split(":")
        if len(parts) != 3 or not parts[2].isdigit():
            raise Refusal(f"not-run entry {text!r} is not written as level:arm:seed")
        out.add(("capacity", parts[0], parts[1], int(parts[2])))
    return out


def _run_dirs(text: str):
    """This function returns the run directories of a source that hold a config.json and match its pattern."""
    root, pattern = assemble_runs.parse_source(text)
    if not root.is_dir():
        raise Refusal(f"{root} is not a directory")
    dirs = [d for d in sorted(p for p in root.iterdir() if p.is_dir())
            if fnmatch.fnmatchcase(d.name, pattern) and (d / "config.json").exists()]
    if not dirs:
        raise Refusal(f"{text} matched no run directory")
    return dirs


def _identity(cfg: dict):
    return assemble_runs.run_identity(cfg)


def load_side(sources, side: str) -> dict:
    """This function returns every finished run of one side, keyed by identity.

    A run without DONE or a repeated identity is a refusal, because an unfinished or doubled run must
    not enter a comparison by accident."""
    runs = {}
    for text in sources or []:
        for d in _run_dirs(text):
            if not (d / "DONE").exists():
                raise Refusal(f"{side} {d.parent.name}/{d.name} has no DONE marker")
            run = analyze._load_run(d)
            ident = _identity(run["cfg"])
            if ident in runs:
                raise Refusal(f"two {side} runs share the identity {ident}, which are "
                              f"{runs[ident]['dir'].name} and {d.name}")
            runs[ident] = run
    return runs


# ---------------------------------------------------------------- labels


def run_labels(run) -> dict:
    """This function returns the final labels of one run, which it reads from the final evaluation.

    A run is feasible when its final violation rate is 0.0, and it is unparseable when its final
    parse-failure rate is 1.0. The trainer counts an unparseable completion as a violation, so a
    feasible run parsed every answer. The outcome class is U for an unparseable run, FE for a
    feasible run whose answers are all empty selections, FI for any other feasible run, and V
    otherwise."""
    if not run["evals"]:
        raise Refusal(f"{run['dir'].name} has no evaluation record")
    ev = run["evals"][-1]
    if ev.get("step") != run["cfg"]["steps"]:
        raise Refusal(f"{run['dir'].name} has no evaluation at its last step")
    row = analyze.final_answers([run])[0]
    n, empty = row["eval_n_final"], row["eval_empty_selections_final"]
    if n is None or empty is None:
        raise Refusal(f"{run['dir'].name} has no final selections to count")
    feasible = float(ev["eval_violation_rate"]) == 0.0
    unparseable = float(ev["eval_parse_fail_rate"]) == 1.0
    if unparseable:
        outcome = "U"
    elif feasible:
        outcome = "FE" if empty == n else "FI"
    else:
        outcome = "V"
    return {"feasibility": feasible, "unparseable": unparseable, "outcome_class": outcome,
            "eval_n": n, "empty_selections": empty}


# ---------------------------------------------------------------- pair checks


def pair_problems(original, rerun, expect_sha: str, expect_digest: str) -> list:
    """This function lists every reason the pair cannot be compared.

    The list is empty when the pair is admissible."""
    name = f"{rerun['dir'].parent.name}/{rerun['dir'].name}"
    problems = []
    if rerun["prov"].get("git_sha") != expect_sha:
        problems.append(f"{name}: git_sha {rerun['prov'].get('git_sha')!r} is not {expect_sha}")
    if rerun["prov"].get("training_sources_digest") != expect_digest:
        problems.append(f"{name}: training_sources_digest "
                        f"{rerun['prov'].get('training_sources_digest')!r} is not {expect_digest}")
    level = original["cfg"].get("regime")
    if original["prov"].get("git_sha") != ORIGINAL_SHA.get(level):
        problems.append(f"original {original['dir'].name}: git_sha {original['prov'].get('git_sha')!r} "
                        f"is not the commit of level {level}")
    a, b = original["cfg"], rerun["cfg"]
    for k in sorted(set(a) | set(b)):
        if k not in analyze.ALLOWED_CONFIG_DIFF and a.get(k) != b.get(k):
            problems.append(f"{name}: config.{k} = {b.get(k)!r} vs {a.get(k)!r}")
    a, b = original["prov"], rerun["prov"]
    for k in sorted(set(a) | set(b)):
        if k in analyze.ALLOWED_PROV_DIFF or k == "git_sha":
            continue
        if a.get(k) != b.get(k):
            problems.append(f"{name}: provenance.{k} = {b.get(k)!r} vs {a.get(k)!r}")
    if rerun["prov"].get("steps_completed") != rerun["cfg"].get("steps"):
        problems.append(f"{name}: steps_completed {rerun['prov'].get('steps_completed')!r} is not "
                        f"steps {rerun['cfg'].get('steps')!r}")
    return problems


def trajectory(original, rerun) -> dict:
    """This function compares the logged training groups of the two runs at every step."""
    steps = int(rerun["cfg"]["steps"])
    key = tasks.get(rerun["cfg"].get("task", "speed")).ITEMS_KEY
    try:
        va = analyze._fingerprint_view(original, steps)
        vb = analyze._fingerprint_view(rerun, steps)
    except analyze.EquivalenceError as e:
        raise Refusal(str(e))
    same = [len(sa) == len(sb) and all(x[key] == y[key] for x, y in zip(sa, sb))
            for sa, sb in zip(va["groups"], vb["groups"])]
    first = va["groups"][0], vb["groups"][0]
    step1 = len(first[0]) == len(first[1]) and all(
        x[k] == y[k] for x, y in zip(*first) for k in (key, "rewards", "adv"))
    checksum_a = original["prov"].get("final_param_checksum")
    checksum_b = rerun["prov"].get("final_param_checksum")
    return {"step1_identical": step1,
            "logic_break_step": analyze._logic_break(va, vb, key),
            "first_answer_difference_step": next((i + 1 for i, s in enumerate(same) if not s), None),
            "identical_answer_steps": sum(same),
            "checksums_equal": checksum_a is not None and checksum_a == checksum_b}


def form_pairs(originals: dict, reruns: dict, not_run: set):
    """This function returns the pairs, the identities declared not run, and the new seeds.

    A new seed is a rerun that has no original."""
    for ident in sorted(not_run):
        if ident not in originals:
            raise Refusal(f"not-run entry {ident} names no original run")
        if ident in reruns:
            raise Refusal(f"{ident} is declared not run and also has a finished rerun")
    missing = sorted(i for i in originals if i not in reruns and i not in not_run)
    if missing:
        raise Refusal("these originals have no rerun and are not declared not run: "
                      + ", ".join(f"{i[1]}:{i[2]}:{i[3]}" for i in missing))
    pairs = [(i, originals[i], reruns[i]) for i in sorted(originals) if i in reruns]
    new_seeds = sorted(i for i in reruns if i not in originals)
    return pairs, sorted(not_run), new_seeds


def check_pairs(pairs, expect_sha, expect_digest):
    problems = [p for _, o, r in pairs for p in pair_problems(o, r, expect_sha, expect_digest)]
    if problems:
        raise Refusal("the pairs are not admissible:\n  " + "\n  ".join(problems))


def check_new_seeds(reruns, new_seeds, expect_sha, expect_digest):
    """This function refuses a rerun without an original unless it carries the frozen commit and digest."""
    problems = []
    for ident in new_seeds:
        prov = reruns[ident]["prov"]
        if prov.get("git_sha") != expect_sha or prov.get("training_sources_digest") != expect_digest:
            problems.append(f"new seed {reruns[ident]['dir'].name} is not at the expected commit and digest")
    if problems:
        raise Refusal("\n  ".join(problems))


# ---------------------------------------------------------------- gate SC0


def gate(pairs) -> dict:
    """This function computes gate SC0 over every finished pair.

    It reads the logged training groups only."""
    failing = []
    step1 = breaks = 0
    for ident, original, rerun in pairs:
        t = trajectory(original, rerun)
        step1 += bool(t["step1_identical"])
        breaks += t["logic_break_step"] is not None
        if not t["step1_identical"] or t["logic_break_step"] is not None:
            failing.append({"regime": ident[1], "arm": ident[2], "seed": ident[3],
                            "step1_identical": t["step1_identical"],
                            "logic_break_step": t["logic_break_step"]})
    holds = bool(pairs) and not failing
    return {"sc0_holds": holds, "pairs": len(pairs), "pairs_step1_identical": step1,
            "pairs_with_logic_break": breaks, "failing_pairs": failing,
            "reason": None if holds else ("no finished pair" if not pairs else "a pair fails SC0")}


# ---------------------------------------------------------------- reference values


def reference_probabilities(ident, original_labels: dict, cohort: dict, cohort_labels: dict) -> dict:
    """This function returns the reference probability of each label for one pair.

    The probability is the share of the arm's other seeds that carry the original's label."""
    others = [i for i in cohort if i[:3] == ident[:3] and i[3] != ident[3]]
    if not others:
        raise Refusal(f"the reference cohort holds no other seed of {ident[1]}:{ident[2]}")
    return {lab: Fraction(sum(cohort_labels[i][lab] == original_labels[lab] for i in others),
                          len(others))
            for lab in LABELS}


def poisson_binomial(ps) -> list:
    """This function returns the exact distribution of the number of successes.

    Each draw succeeds independently with its own probability, and the sums use exact fractions."""
    dist = [Fraction(1)]
    for p in ps:
        p = Fraction(p)
        nxt = [Fraction(0)] * (len(dist) + 1)
        for k, q in enumerate(dist):
            nxt[k] += q * (1 - p)
            nxt[k + 1] += q * p
        dist = nxt
    return dist


def central_range(ps):
    """This function returns the central 95 percent range of the agreement count under independent draws."""
    lo = hi = None
    total = Fraction(0)
    for k, q in enumerate(poisson_binomial(ps)):
        total += q
        if lo is None and total >= RANGE_LOW:
            lo = k
        if hi is None and total >= RANGE_HIGH:
            hi = k
    return lo, hi


def band(agree: int, n: int, lo: int, hi: int) -> str:
    """Band R is agreement in every pair. M lies above the range, X lies inside it, and B lies below it."""
    if agree == n:
        return "R"
    if agree > hi:
        return "M"
    if agree >= lo:
        return "X"
    return "B"


def reference_summary(ps) -> dict:
    lo, hi = central_range(ps)
    return {"expected": float(sum(ps, Fraction(0))),
            "sd": math.sqrt(float(sum((p * (1 - p) for p in ps), Fraction(0)))),
            "range_low": lo, "range_high": hi, "pairs_at_probability_one": sum(p == 1 for p in ps)}


def reference_values(originals: dict, cohort: dict) -> dict:
    """This function returns the reference values of every label and the probability of every pair."""
    cohort_labels = {i: run_labels(r) for i, r in cohort.items()}
    per_pair = {i: reference_probabilities(i, run_labels(r), cohort, cohort_labels)
                for i, r in sorted(originals.items())}
    return {"per_pair": per_pair,
            "labels": {lab: reference_summary([per_pair[i][lab] for i in sorted(per_pair)])
                       for lab in LABELS}}


# ---------------------------------------------------------------- files-only mode


def files_check(sources, manifest: dict, rows: list, expect_sha: str, expect_digest: str) -> dict:
    """This function lists the problems of every expected row and every directory that no row names.

    The check reads configs, provenance and record counts, and it reads no evaluation metric."""
    found = {}
    bare = set()
    for text in sources:
        for d in _run_dirs(text):
            found.setdefault(d.name, []).append(d)
        root, pattern = assemble_runs.parse_source(text)
        bare.update(p.name for p in root.iterdir() if p.is_dir() and not (p / "config.json").exists()
                    and fnmatch.fnmatchcase(p.name, pattern))
    missing_rows = [r for r in rows if r not in manifest]
    if missing_rows:
        raise Refusal(f"the manifest has no row {missing_rows}")
    expected = {}
    for r in rows:
        m = manifest[r]
        expected[f"{r}_{m['regime']}_{m['arm_name']}_s{m['seed']}"] = r
    extra = sorted(n for n in found if n not in expected)
    report = {}
    n_train_expected = EXPECTED_CONFIG["steps"]
    n_eval_expected = EXPECTED_CONFIG["steps"] // EXPECTED_CONFIG["eval_every"] + 1
    for name, r in sorted(expected.items(), key=lambda kv: kv[1]):
        m = manifest[r]
        dirs = found.get(name, [])
        if not dirs:
            report[r] = {"dir": name, "problems": ["config.json" if name in bare else "missing"]}
            continue
        if len(dirs) > 1:
            report[r] = {"dir": name, "problems": ["duplicate directory in the sources"]}
            continue
        d = dirs[0]
        problems = []
        if not (d / "DONE").exists():
            problems.append("DONE")
        cfg = json.loads((d / "config.json").read_text())
        prov_path = d / "provenance.json"
        prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
        if not prov:
            problems.append("provenance.json")
        for k, v in EXPECTED_CONFIG.items():
            if cfg.get(k) != v:
                problems.append(f"config.{k}={cfg.get(k)!r}")
        want = {"arm": m["arm"], "regime": m["regime"], "seed": int(m["seed"]),
                "w": float(m["w"]), "w_task": float(m["w_task"]), "vmax": float(m["vmax"])}
        for k in MANIFEST_FIELDS:
            got = cfg.get(k)
            if k in ("w", "w_task", "vmax") and isinstance(got, (int, float)):
                got = float(got)
            if got != want[k]:
                problems.append(f"config.{k}={cfg.get(k)!r}")
        if "arm" in cfg and "w" in cfg and analyze.arm_name(cfg) != m["arm_name"]:
            problems.append(f"arm_name={analyze.arm_name(cfg)!r}")
        if prov:
            if prov.get("git_sha") != expect_sha:
                problems.append(f"git_sha={prov.get('git_sha')!r}")
            if prov.get("training_sources_digest") != expect_digest:
                problems.append(f"training_sources_digest={prov.get('training_sources_digest')!r}")
            if prov.get("steps_completed") != n_train_expected:
                problems.append(f"steps_completed={prov.get('steps_completed')!r}")
            for k, v in EXPECTED_ENVIRONMENT.items():
                if prov.get(k) != v:
                    problems.append(f"provenance.{k}={prov.get(k)!r}")
        metrics = d / "metrics.jsonl"
        if metrics.exists():
            phases = [json.loads(line).get("phase") for line in metrics.read_text().splitlines()
                      if line.strip()]
            n_eval = sum(p == "eval" for p in phases)
            if (len(phases) - n_eval, n_eval) != (n_train_expected, n_eval_expected):
                problems.append(f"records train={len(phases) - n_eval} eval={n_eval}")
        else:
            problems.append("metrics.jsonl")
        report[r] = {"dir": f"{d.parent.name}/{d.name}", "problems": problems,
                     "git_dirty": prov.get("git_dirty")}
    return {"rows": report, "unexpected": extra}


# ---------------------------------------------------------------- output


def outcome_keys_in(obj, where="$") -> list:
    """This function lists every key of a nested object that names a label or a divergence field."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(bad in str(k) for bad in OUTCOME_KEYS):
                found.append(f"{where}.{k}")
            found.extend(outcome_keys_in(v, f"{where}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(outcome_keys_in(v, f"{where}[{i}]"))
    return found


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _prepare_out(out) -> Path:
    if out is None:
        return None
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Refusal(f"{out} already exists and is not an empty directory")
    return out


def _ident_text(ident) -> str:
    return f"{ident[1]}:{ident[2]}:{ident[3]}"


def full_records(pairs, cohort: dict):
    """This function returns the per-pair rows and the summary, and it runs only after SC0 holds."""
    cohort_labels = {i: run_labels(r) for i, r in cohort.items()}
    rows = []
    for ident, original, rerun in pairs:
        t = trajectory(original, rerun)
        lo_, lr_ = run_labels(original), run_labels(rerun)
        p = reference_probabilities(ident, lo_, cohort, cohort_labels)
        rows.append({
            "regime": ident[1], "arm": ident[2], "seed": ident[3],
            "original": f"{original['dir'].parent.name}/{original['dir'].name}",
            "rerun": f"{rerun['dir'].parent.name}/{rerun['dir'].name}",
            "step1_identical": t["step1_identical"], "logic_break_step": t["logic_break_step"],
            "first_answer_difference_step": t["first_answer_difference_step"],
            "identical_answer_steps": t["identical_answer_steps"],
            "checksums_equal": t["checksums_equal"],
            "original_feasible": lo_["feasibility"], "rerun_feasible": lr_["feasibility"],
            "original_unparseable": lo_["unparseable"], "rerun_unparseable": lr_["unparseable"],
            "original_class": lo_["outcome_class"], "rerun_class": lr_["outcome_class"],
            "original_empty_selections": lo_["empty_selections"],
            "rerun_empty_selections": lr_["empty_selections"], "eval_n": lr_["eval_n"],
            "p_feasibility": float(p["feasibility"]), "p_unparseable": float(p["unparseable"]),
            "p_outcome_class": float(p["outcome_class"]),
            "_p": p, "_labels": (lo_, lr_),
        })
    n = len(rows)
    labels = {}
    for lab in LABELS:
        ps = [r["_p"][lab] for r in rows]
        ref = reference_summary(ps)
        agree = sum(r["_labels"][0][lab] == r["_labels"][1][lab] for r in rows)
        labels[lab] = dict(ref, agree=agree, pairs=n,
                           band=band(agree, n, ref["range_low"], ref["range_high"]))
    for lab in ("feasibility", "unparseable"):
        labels[lab]["original_true_rerun_false"] = sum(
            r["_labels"][0][lab] and not r["_labels"][1][lab] for r in rows)
        labels[lab]["original_false_rerun_true"] = sum(
            not r["_labels"][0][lab] and r["_labels"][1][lab] for r in rows)
    summary = {"pairs": n, "labels": labels,
               "pairs_with_equal_checksums": sum(r["checksums_equal"] for r in rows),
               "pairs_with_different_outcome_class": n - labels["outcome_class"]["agree"]}
    for r in rows:
        del r["_p"], r["_labels"]
    return rows, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--files-only", action="store_true")
    mode.add_argument("--gate-only", action="store_true")
    ap.add_argument("--rerun", action="append", required=True)
    ap.add_argument("--original", action="append")
    ap.add_argument("--reference-cohort")
    ap.add_argument("--expect-sha", required=True)
    ap.add_argument("--expect-digest", required=True)
    ap.add_argument("--expect-rows")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--not-run", action="append", default=[],
                    help="an original that has no finished rerun, written as level:arm:seed")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        if a.files_only:
            if not a.expect_rows:
                raise Refusal("--files-only needs --expect-rows")
            result = files_check(a.rerun, read_manifest(Path(a.manifest)), parse_rows(a.expect_rows),
                                 a.expect_sha, a.expect_digest)
            bad = 0
            for r, entry in result["rows"].items():
                status = "OK" if not entry["problems"] else "PROBLEMS " + " ".join(entry["problems"])
                print(f"row {r} {entry['dir']} git_dirty={entry.get('git_dirty')} {status}")
                bad += bool(entry["problems"])
            for name in result["unexpected"]:
                print(f"unexpected run directory {name}")
            dirty = sum(e.get("git_dirty") is True for e in result["rows"].values())
            print(f"rows checked {len(result['rows'])}, rows with problems {bad}, "
                  f"unexpected directories {len(result['unexpected'])}, git_dirty true {dirty}")
            return 1 if bad or result["unexpected"] else 0
        if not a.original:
            raise Refusal("--gate-only and the full mode need --original")
        out = _prepare_out(a.out)
        if not a.gate_only and (out is None or not a.reference_cohort):
            raise Refusal("the full mode needs --out and --reference-cohort")
        originals = load_side(a.original, "original")
        reruns = load_side(a.rerun, "rerun")
        pairs, not_run, new_seeds = form_pairs(originals, reruns, parse_not_run(a.not_run))
        check_pairs(pairs, a.expect_sha, a.expect_digest)
        check_new_seeds(reruns, new_seeds, a.expect_sha, a.expect_digest)
        cohort = load_side([a.reference_cohort], "reference") if not a.gate_only else None
        result = gate(pairs)
        result.update(expect_sha=a.expect_sha, expect_digest=a.expect_digest,
                      not_run=[_ident_text(i) for i in not_run],
                      new_seeds=[_ident_text(i) for i in new_seeds])
        leaked = outcome_keys_in(result)
        if leaked:
            raise Refusal(f"the gate record would carry outcome keys {leaked}")
        records = None
        if not a.gate_only and result["sc0_holds"]:
            records = full_records(pairs, cohort)
    except Refusal as e:
        print(f"REFUSED: nothing was written. {e}", file=sys.stderr)
        return 2
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "gate.json", result)
    print(json.dumps({k: result[k] for k in ("sc0_holds", "pairs", "pairs_step1_identical",
                                             "pairs_with_logic_break", "reason")}))
    if not result["sc0_holds"]:
        for f in result["failing_pairs"]:
            print(f"SC0 fails for {f['regime']}:{f['arm']}:{f['seed']} step1_identical="
                  f"{f['step1_identical']} logic_break_step={f['logic_break_step']}")
        return 1
    if records is not None:
        rows, summary = records
        with (out / "pairs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summary.update(not_run=result["not_run"], new_seeds=result["new_seeds"],
                       expect_sha=a.expect_sha, expect_digest=a.expect_digest)
        _write_json(out / "summary.json", summary)
        print(json.dumps({"written": [str(out / "gate.json"), str(out / "pairs.csv"),
                                      str(out / "summary.json")]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
