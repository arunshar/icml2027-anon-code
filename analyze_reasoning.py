#!/usr/bin/env python3
"""The WO-I analysis applies admission, scores the W0 gate, and builds the counters and the tables.

The script reads pulled run directories and the frozen study manifest. It first applies the
admission rule of Section 4 to every run. It refuses to build any table unless every manifest row
has an admitted run or is named in the not_run file. It then scores W0 by replaying every logged
group from its logged lengths and verdicts. When W0 fails, it reports W0 and stops, so no outcome
is read. Only when W0 holds does it build the audit counters, the endpoint tables and the scores
of W1 to W7. With --w0-only it stops after W0 in every case.

The replay uses reasoning_task.reasoning_rewards and the unchanged advantage_variant and keep_group
of rewards.py, so it recomputes each arm exactly as the trainer defines it.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import reasoning_task as RT
import rewards as RW

LOG_FAILURE = re.compile(r"Traceback|out of memory|OutOfMemoryError|DUE TO TIME LIMIT|CANCELLED AT", re.I)
# These settings are held equal by the plan (Section 2.2) and are not manifest fields, so admission
# checks them in config.json. A run made with the test flags, such as --tiny or --eval-limit, fails here.
DESIGN_CONFIG = {"mode": "manifest", "kind": "study", "tiny": False, "eval_limit": None, "device": "cuda",
                 "lr": 1e-5, "weight_decay": 0.0, "clip_coef": 0.2, "grad_clip": 1.0,
                 "eval_seed": RT.EVAL_SEED, "eval_chunk_sequences": RT.EVAL_CHUNK_SEQUENCES,
                 "mid_eval_every": RT.EVAL_EVERY, "micro_batch": RT.MICRO_BATCH, "w": RT.W,
                 "group_size": RT.GROUP_SIZE, "groups_per_step": RT.GROUPS_PER_STEP,
                 "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.0}}
TOL_SELF = 1e-6
TOL_TASK_ONLY = 1e-9
W2_SHARE = 0.99


class Refusal(RuntimeError):
    """The analysis cannot build a table from these files."""


# ------------------------------------------------------------------ loading


def read_manifest(path) -> List[Dict[str, str]]:
    raw = Path(path).read_bytes()
    if b"\r" in raw:
        raise Refusal("the manifest must be LF-only")
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    for row in rows:
        RT.validate_arm_row(row)
    return rows


def load_run(run_dir: Path) -> dict:
    recs = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line]
    return {"dir": run_dir, "cfg": json.loads((run_dir / "config.json").read_text()),
            "prov": json.loads((run_dir / "provenance.json").read_text()),
            "train": [r for r in recs if r["phase"] == "train"],
            "eval": [r for r in recs if r["phase"] == "eval"],
            "done": (run_dir / "DONE").exists()}


def find_runs(roots: List[Path]) -> Dict[int, List[dict]]:
    by_task: Dict[int, List[dict]] = {}
    for root in roots:
        for cfg in sorted(Path(root).rglob("config.json")):
            run = load_run(cfg.parent)
            if run["cfg"].get("mode") == "manifest" and run["cfg"].get("kind") == "study":
                by_task.setdefault(int(run["cfg"]["task_id"]), []).append(run)
    return by_task


def log_text(logs_dir: Optional[Path], prov: dict) -> Optional[str]:
    """Return the run's Slurm log text, None when no log directory was given, or "" when no log is found."""
    if logs_dir is None:
        return None
    job, task = prov.get("slurm_array_job_id"), prov.get("slurm_array_task_id")
    if not job or not task:
        return ""
    parts = [p.read_text(errors="replace") for p in sorted(Path(logs_dir).glob(f"*-{job}_{task}.*"))]
    return "\n".join(parts)


def design_config_reasons(cfg: dict, row: Dict[str, str]) -> List[str]:
    """Return the held-equal settings and arm fields in config.json that differ from the design."""
    reasons = [f"config.json has {key} = {cfg.get(key)!r}, and the design needs {want!r}"
               for key, want in DESIGN_CONFIG.items() if cfg.get(key) != want]
    spec = RT.ARMS.get(row["arm"], {})
    for key in ("form", "w_task", "adv_variant", "group_filter"):
        if cfg.get(key) != spec.get(key):
            reasons.append(f"config.json has {key} = {cfg.get(key)!r}, and the arm needs {spec.get(key)!r}")
    return reasons


def admit(run: dict, row: Dict[str, str], frozen: dict, logs_dir: Optional[Path]) -> List[str]:
    """Return the reasons a run fails the admission rule. An empty list admits the run."""
    reasons = []
    cfg, prov, steps = run["cfg"], run["prov"], int(row["steps"])
    if not run["done"]:
        reasons.append("no DONE marker")
    if prov.get("stopped_early"):
        # Design version 2 only. The early-finish guard ended the run before its step count, so it is
        # not a complete row, and the reason says so by name instead of leaving only the step-count gap.
        reasons.append(f"the run stopped early after step {prov.get('steps_completed')} of {steps} (stopped_early)")
    if [r["step"] for r in run["train"]] != list(range(1, steps + 1)):
        reasons.append("the training records do not cover every step once")
    want = [(s, name) for s in range(steps + 1) for name, _ in RT.eval_plan(s, steps, RT.EVAL_EVERY)]
    if [(r["step"], r["set"]) for r in run["eval"]] != want:
        reasons.append("an evaluation record is missing or unscheduled")
    if cfg.get("manifest_row") != row:
        reasons.append("the configuration differs from its manifest row")
    reasons += design_config_reasons(cfg, row)
    if prov.get("reasoning_sources_digest") != frozen["reasoning_sources_digest"]:
        reasons.append("the source digest differs from the frozen value")
    if prov.get("dataset_sha256") != row["dataset_sha256"] or prov.get("selection_sha256") != row["selection_sha256"]:
        reasons.append("the dataset or selection digest differs from the manifest")
    if prov.get("model_revision") != row["model_revision"] or prov.get("model_snapshot") != row["model_revision"]:
        reasons.append("the model revision differs from the manifest")
    if prov.get("device_name") != RT.REQUIRED_GPU_NAME:
        reasons.append("the device is not an NVIDIA A100-SXM4-40GB")
    text = log_text(logs_dir, prov)
    if text is not None and not text:
        reasons.append("no Slurm log was found for this run")
    if text is not None and LOG_FAILURE.search(text):
        reasons.append("the logs hold traceback, out-of-memory or time-limit text")
    return reasons


# ------------------------------------------------------------------ W0 and the replay


def replay_group(g: dict, cfg: dict) -> dict:
    """Recompute one logged group under the run's own arm, and return the recomputed values."""
    L, fails, n = int(cfg["L"]), g["F"], g["n_tokens"]
    r = RT.reasoning_rewards(fails, n, cfg["form"], float(cfg["w"]), float(cfg["w_task"]), L)
    a = RW.advantage_variant(r, cfg["adv_variant"])
    kept = RW.keep_group(cfg["group_filter"], r, [f == 0 for f in fails])
    task_only = RW.advantage_variant(RT.task_only_rewards(n, float(cfg["w_task"]), L), cfg["adv_variant"])
    return {"rewards": r, "adv": a, "kept": kept, "task_only_adv": task_only}


def score_w0(runs: List[dict]) -> dict:
    worst = {"a": 0.0, "b": 0.0, "c": 0.0}
    fails = {"a": 0, "b": 0, "c": 0, "kept": 0, "reward": 0}
    for run in runs:
        cfg = run["cfg"]
        for rec in run["train"]:
            for g in rec["groups"]:
                rp = replay_group(g, cfg)
                logged = np.asarray(g["adv"], dtype=np.float64)
                d = float(np.max(np.abs(rp["adv"] - logged)))
                worst["a"] = max(worst["a"], d)
                fails["a"] += d > TOL_SELF
                fails["reward"] += float(np.max(np.abs(rp["rewards"] - np.asarray(g["rewards"])))) > TOL_SELF
                fails["kept"] += bool(g["kept"]) != rp["kept"]
                if not all(f == 1 for f in g["F"]):
                    continue
                if cfg["arm"] in ("plain", "drgrpo"):
                    d = float(np.max(np.abs(logged - rp["task_only_adv"])))
                    worst["b"] = max(worst["b"], d)
                    fails["b"] += d > TOL_TASK_ONLY
                if cfg["arm"] in ("gate", "verdict_only"):
                    d = float(np.max(np.abs(logged)))
                    worst["c"] = max(worst["c"], d)
                    fails["c"] += d > TOL_TASK_ONLY
                if cfg["arm"] == "vfilter" and g["kept"]:
                    fails["c"] += 1
    holds = all(v == 0 for v in fails.values())
    return {"holds": holds, "worst": worst, "failures": fails, "runs": len(runs)}


# ------------------------------------------------------------------ counters and endpoints


def audit(run: dict, short_threshold: Optional[float]) -> dict:
    c = {"groups": 0, "all_fail": 0, "mixed": 0, "all_pass": 0, "zero_variance": 0,
         "all_fail_zero_variance": 0, "all_fail_positive_variance": 0, "window_last_step": 0,
         "window_groups_in_step1": 0, "identical_reward_filter_drops": 0, "kept_groups": 0,
         "opt_steps_taken": 0, "all_fail_short_completions": 0, "all_fail_completions": 0}
    mixed_order_fail, all_fail_len = 0, []
    for rec in run["train"]:
        c["opt_steps_taken"] += bool(rec["opt_step_taken"])
        lens = []
        for g in rec["groups"]:
            rw = np.asarray(g["rewards"], dtype=np.float64)
            zero = bool(np.ptp(rw) <= RW.ZERO_SPREAD_TOL)
            c["groups"] += 1
            c["zero_variance"] += zero
            c["kept_groups"] += bool(g["kept"])
            c["identical_reward_filter_drops"] += not RW.keep_group("reward_spread", rw, [])
            f = np.asarray(g["F"])
            if f.all():
                c["all_fail"] += 1
                c["all_fail_zero_variance"] += zero
                c["all_fail_positive_variance"] += not zero
                if not zero:
                    c["window_last_step"] = max(c["window_last_step"], rec["step"])
                    c["window_groups_in_step1"] += rec["step"] == 1
                lens += g["n_tokens"]
                c["all_fail_completions"] += len(g["n_tokens"])
                if short_threshold is not None:
                    c["all_fail_short_completions"] += sum(1 for n in g["n_tokens"] if n < short_threshold)
            elif not f.any():
                c["all_pass"] += 1
            else:
                c["mixed"] += 1
                if rw[f == 0].min() <= rw[f == 1].max():
                    mixed_order_fail += 1
        all_fail_len.append(float(np.mean(lens)) if lens else None)
    c["mixed_order_failures"] = mixed_order_fail
    c["all_fail_mean_length_by_step"] = all_fail_len
    c["all_fail_positive_variance_share"] = (c["all_fail_positive_variance"] / c["all_fail"]
                                             if c["all_fail"] else None)
    return c


def final_endpoints(run: dict, steps: int, short_threshold: Optional[float]) -> dict:
    final = {r["set"]: r for r in run["eval"] if r["step"] == steps}
    items = final["hard"]["items"] + final["random"]["items"]
    return {"final_hard_accuracy": final["hard"]["accuracy"], "final_random_accuracy": final["random"]["accuracy"],
            "final_hard_lenient_accuracy": final["hard"]["lenient_accuracy"],
            "final_heldout_mean_length": float(np.mean([it[2] for it in items])),
            "final_heldout_short_incorrect_rate": (float(np.mean([it[4] == 1 and it[2] < short_threshold
                                                                  for it in items]))
                                                   if short_threshold is not None else None),
            "final_heldout_truncation_rate": float(np.mean([not it[3] for it in items])),
            "final_heldout_strict_parse_fail_rate": float(np.mean([it[5] is None for it in items]))}


def label(n: int, k: int, w0: bool) -> str:
    if w0 and n == 3 and k == 3:
        return "A"
    if w0 and k == 0 and n >= 2:
        return "Z"
    return "N"


def score_predictions(table: List[dict], counters: Dict[int, dict], w0: bool, steps: int) -> dict:
    by = {(r["size"], r["arm"], r["seed"]): r for r in table}
    out: Dict[str, object] = {"W0": w0}
    out["W1"] = all(counters[r["task_id"]]["mixed_order_failures"] == 0 for r in table)
    # A plain run with no all-fail group does not count against W2, and it is listed instead.
    plain_runs = [r for r in table if r["arm"] == "plain"]
    shares = [counters[r["task_id"]]["all_fail_positive_variance_share"] for r in plain_runs]
    out["W2"] = all(x is None or x >= W2_SHARE for x in shares) if plain_runs else None
    out["W2_runs_without_all_fail_groups"] = [r["task_id"] for r, x in zip(plain_runs, shares) if x is None]
    sizes = sorted({r["size"] for r in table})
    # A size with no finished verifier-filter run leaves W3 not scored there, so it never holds vacuously.
    vfilter = {s: [r for r in table if r["size"] == s and r["arm"] == "vfilter"] for s in sizes}
    out["W3"] = {s: (all(counters[r["task_id"]]["opt_steps_taken"] < steps for r in vfilter[s])
                     if vfilter[s] else "not scored") for s in sizes}
    labels, w4, w5, vd = {}, {}, {}, {}
    for s in sizes:
        pairs = [(by.get((s, "gate", seed)), by.get((s, "plain", seed))) for seed in RT.SEEDS]
        pairs = [(g, p) for g, p in pairs if g and p]
        n = len(pairs)
        k = sum(1 for g, p in pairs if g["final_hard_accuracy"] > p["final_hard_accuracy"])
        labels[s] = {"n": n, "k": k, "label": label(n, k, w0)}
        w4[s] = n == 3 and k == 3
        w5[s] = n == 3 and all(p["final_heldout_short_incorrect_rate"] > g["final_heldout_short_incorrect_rate"]
                               for g, p in pairs)
        vd[s] = {name: sum(1 for seed in RT.SEEDS if by.get((s, arm, seed)) and by.get((s, "plain", seed))
                           and by[(s, arm, seed)]["final_hard_accuracy"] > by[(s, "plain", seed)]["final_hard_accuracy"])
                 for name, arm in (("V", "vfilter"), ("D", "drgrpo"))}
    out.update({"labels": labels, "W4": w4, "W5": w5, "V_and_D": vd})
    pv = [(by.get(("0.5b", "plain", s)), by.get(("0.5b", "verdict_only", s))) for s in RT.SEEDS]
    pv = [(p, v) for p, v in pv if p and v]
    out["W6"] = len(pv) == 3 and all(p["final_heldout_mean_length"] < v["final_heldout_mean_length"] for p, v in pv)
    small = labels.get("0.5b", {}).get("label")
    out["W7"] = (labels["1.5b"]["label"] == small) if (small in ("A", "Z") and "1.5b" in labels) else "not scored"
    return out


# ------------------------------------------------------------------ per-run curves (design version 2 additions)

# The per-step fields a curve can follow. completion_tokens_mean exists in every run. The other two
# are written by design version 2 runs only (ICML 2027, November 2026), so a version-1 run's curves
# hold them as None at every step.
CURVE_FIELDS = ("completion_tokens_mean", "entropy_tok_mean", "kl_seq_mean")


def run_curves(run: dict) -> dict:
    """Return one run's per-step curves of mean completion length, token entropy and sequence KL.

    Each curve is a list with one entry per logged training step, in step order, and an entry is None
    when the step does not log that field. The record also says whether the run stopped early and how
    many steps it completed, read from its provenance, so a pilot or study row that the early-finish
    guard stopped is visible here even when admission refuses it as incomplete."""
    cfg, prov = run["cfg"], run["prov"]
    train = run["train"]
    return {"task_id": cfg.get("task_id"), "kind": cfg.get("kind"), "arm": cfg.get("arm"), "seed": cfg.get("seed"),
            "design_version": cfg.get("design_version", RT.DESIGN_VERSION_V1),
            "steps_planned": cfg.get("steps"), "steps_completed": prov.get("steps_completed"),
            "stopped_early": bool(prov.get("stopped_early", False)), "done": run["done"],
            "step": [r["step"] for r in train],
            **{field: [r.get(field) for r in train] for field in CURVE_FIELDS}}


def curves_report(roots: List[Path]) -> List[dict]:
    """Return the curves of every run under the roots, whatever its kind, admitted or not.

    This reads the same run directories as find_runs, without the manifest filter, so a pilot run and a
    run that stopped early both appear. The main report is untouched: the curves are written to a
    separate file by --curves."""
    out = []
    for root in roots:
        for cfg in sorted(Path(root).rglob("config.json")):
            out.append({"dir": str(cfg.parent), **run_curves(load_run(cfg.parent))})
    return out


# ------------------------------------------------------------------ design version 2 pilot (ICML 2027, Block D')
#
# The pilot of 6 November 2026 runs three single-change rows of the verdict-only arm and scores each
# with the D0 rule of PREDICTIONS.md. This section holds the admission of a version-2 pilot row, the D0
# score and the D-choice. It is reached only through the pilot-d0 command line mode, so the version-1
# analysis above, its arguments and every file it writes stay as they were.

PILOT_ACC_TOL = 0.05
PILOT_PARSE_MIN = 0.80
# D0 compares a difference of two held-out accuracies with 0.05 as a number, and a difference such as
# 0.55 - 0.50 is 0.05000000000000004 in floating point. This slack keeps an exact boundary on the
# passing side. The accuracies move in steps of 1/400, so 1e-12 can never decide a real case.
PILOT_FLOAT_TOL = 1e-12
PILOT_ITEM_TOL = 1e-9
# The D-choice order of the pre-registration: the skip, then the KL penalty, then the learning rate.
PILOT_CHOICE_ORDER = ("skip", "kl", "lr")
PILOT_LOG_FAILURE = re.compile(r"REFUSED|Traceback")
PILOT_LOG_OTHER = re.compile(r"out of memory|OutOfMemoryError|DUE TO TIME LIMIT|CANCELLED AT", re.I)


def _kind_of(lr: float, kl: float, skip: int) -> str:
    if (lr, kl, skip) == (RT.LR_DEFAULT, 0.0, 1):
        return "skip"
    if lr == RT.LR_DEFAULT and kl > 0 and skip == 0:
        return "kl"
    if lr < RT.LR_DEFAULT and kl == 0 and skip == 0:
        return "lr"
    if {"lr": lr, "kl_beta": kl, "skip_zero_adv": skip} == RT.PILOT_RETRY_COMBINATION:
        return "combination"
    return "other"


def pilot_row_kind(row: Dict[str, str]) -> str:
    """Name the change a pilot row makes: skip, kl or lr when it changes that lever alone, else other.

    The skip row has skip_zero_adv 1 at lr 1e-5 and KL beta 0. The KL row has beta above 0 and nothing
    else changed. The learning-rate row has lr below 1e-5 and nothing else changed. The retry
    combination of the plan (the skip with beta 0.02) is named combination."""
    return _kind_of(float(row["lr"]), float(row["kl_beta"]), int(row["skip_zero_adv"]))


def pilot_is_first_wave(row: Dict[str, str]) -> bool:
    """Return True for rows 1 to 3 of the pilot: lr 1e-6 alone, the skip alone and beta 0.02 alone."""
    lr, kl, skip = float(row["lr"]), float(row["kl_beta"]), int(row["skip_zero_adv"])
    return (lr, kl, skip) in ((1e-6, 0.0, 0), (RT.LR_DEFAULT, 0.0, 1), (RT.LR_DEFAULT, 0.02, 0))


def _read_json_file(path: Path, name: str, problems: List[str]):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        problems.append(f"{name} is missing")
    except ValueError:
        problems.append(f"{name} is not valid JSON")
    return None


def load_pilot_run(run_dir: Path) -> dict:
    """Read one pilot run directory without raising, and list what could not be read as load_problems."""
    run_dir = Path(run_dir)
    load_problems: List[str] = []
    cfg = _read_json_file(run_dir / "config.json", "config.json", load_problems)
    prov = _read_json_file(run_dir / "provenance.json", "provenance.json", load_problems)
    recs: List[dict] = []
    try:
        for line in (run_dir / "metrics.jsonl").read_text().splitlines():
            if line:
                recs.append(json.loads(line))
    except FileNotFoundError:
        load_problems.append("metrics.jsonl is missing")
    except ValueError:
        load_problems.append("metrics.jsonl holds a line that is not valid JSON")
    return {"dir": run_dir, "cfg": cfg if isinstance(cfg, dict) else {}, "prov": prov if isinstance(prov, dict) else {},
            "train": [r for r in recs if r.get("phase") == "train"],
            "eval": [r for r in recs if r.get("phase") == "eval"],
            "done": (run_dir / "DONE").exists(), "load_problems": load_problems}


def _number_is(have, want) -> bool:
    """Return True when have is a plain number (not a bool or a string) equal to want."""
    return isinstance(have, (int, float)) and not isinstance(have, bool) and have == want


def _text_same(have, want: str) -> bool:
    """Compare a manifest field stored as text: equal strings, or two numbers that are equal."""
    if have == want:
        return True
    try:
        return isinstance(have, str) and float(have) == float(want)
    except ValueError:
        return False


def pilot_config_problems(cfg: dict, row: Dict[str, str]) -> List[str]:
    """Return how config.json differs from its manifest row. An empty list means it matches."""
    typed = RT.validate_arm_row(row)
    problems = []
    want = {"mode": "manifest", "kind": "pilot", "task_id": typed["task_id"], "arm": typed["arm"],
            "seed": typed["seed"], "steps": typed["steps"], "L": typed["L"], "lr": typed["lr"],
            "kl_beta": typed["kl_beta"], "design_version": RT.DESIGN_VERSION_V2,
            "size": typed["size"], "stage": typed["stage"], "fixedref_c": typed["fixedref_c"]}
    for key, value in want.items():
        have = cfg.get(key)
        same = (have == value) if isinstance(value, str) else _number_is(have, value)
        if not same:
            problems.append(f"config.json has {key} = {have!r}, and the manifest row needs {value!r}")
    skip = cfg.get("skip_zero_adv")
    if not (isinstance(skip, (bool, int)) and int(skip) == typed["skip_zero_adv"]):
        problems.append(f"config.json has skip_zero_adv = {skip!r}, and the manifest row needs {typed['skip_zero_adv']!r}")
    for key, value in (("tiny", False), ("eval_limit", None)):
        if cfg.get(key) is not value:
            problems.append(f"config.json has {key} = {cfg.get(key)!r}, and a pilot row needs {value!r}")
    stored = cfg.get("manifest_row")
    if not isinstance(stored, dict):
        problems.append("the configuration holds no manifest row")
    else:
        differs = [k for k in row if not _text_same(stored.get(k), row[k])]
        extra = sorted(set(stored) - set(row) - {"design_version"})
        if differs or extra or not _text_same(stored.get("design_version"), str(RT.DESIGN_VERSION_V2)):
            problems.append("the configuration differs from its manifest row"
                            + (f" in {', '.join(differs + extra)}" if differs or extra else ""))
    return problems


def pilot_provenance_problems(prov: dict, row: Dict[str, str], frozen: dict) -> List[str]:
    """Return how provenance.json differs from the frozen digests and from the manifest row."""
    problems = []
    if prov.get("design_version") != RT.DESIGN_VERSION_V2:
        problems.append(f"provenance.json has design_version = {prov.get('design_version')!r}, and the pilot needs 2")
    for key, name in (("reasoning_sources_digest", "reasoning"), ("training_sources_digest", "training")):
        if prov.get(key) != frozen.get(key):
            problems.append(f"the {name} source digest differs from the frozen value")
    if "frozen_v2" in prov and prov["frozen_v2"] != frozen:
        problems.append("the frozen record inside provenance.json differs from the frozen file")
    if prov.get("dataset_sha256") != row["dataset_sha256"] or prov.get("selection_sha256") != row["selection_sha256"]:
        problems.append("the dataset or selection digest differs from the manifest")
    if prov.get("model_revision") != row["model_revision"] or prov.get("model_snapshot") != row["model_revision"]:
        problems.append("the model revision differs from the manifest")
    return problems


def pilot_stop_state(run: dict, row: Dict[str, str]) -> dict:
    """Return steps_completed, the stopped_early flag and whether the run finished every step of its row."""
    steps = int(row["steps"])
    sc, stopped = run["prov"].get("steps_completed"), run["prov"].get("stopped_early")
    return {"steps_planned": steps, "steps_completed": sc, "stopped_early": stopped,
            "complete": isinstance(sc, int) and not isinstance(sc, bool) and sc == steps}


def pilot_log_files(logs_dir: Path, prov: dict) -> List[Path]:
    """Return the Slurm log files of one run, matched by array job id and array task id."""
    job, task = prov.get("slurm_array_job_id"), prov.get("slurm_array_task_id")
    if not job or not task:
        return []
    return sorted(Path(logs_dir).glob(f"*-{job}_{task}.*"))


def _items_summary(items: List[list]) -> dict:
    """Recompute accuracy and the strict parse failure rate of one evaluation from its logged items."""
    by_pid: Dict[str, List[list]] = {}
    for it in items:
        by_pid.setdefault(it[0], []).append(it)
    acc = float(np.mean([np.mean([it[4] == 0 for it in v]) for v in by_pid.values()])) if by_pid else None
    fail = sum(1 for it in items if it[5] is None) / len(items) if items else None
    return {"accuracy": acc, "strict_parse_fail_rate": fail}


def admit_pilot(run: dict, row: Dict[str, str], frozen: dict, logs_dir: Optional[Path]) -> List[str]:
    """Return the reasons a version-2 pilot run fails admission. An empty list admits the run.

    Admission needs the DONE marker, a config.json that equals the manifest row (arm, seed, steps, lr,
    KL beta, skip, design version and L), provenance digests that equal the frozen file, a step count
    that equals the row's, training records for every completed step, a random held-out evaluation at
    step 0 and at the last completed step, and Slurm logs with no REFUSED and no Traceback. A run that
    the early-finish guard stopped is not refused for its shorter step count. Its stopped_early flag
    and steps_completed are reported by pilot_stop_state, and it is scored on the evaluations it wrote."""
    problems = list(run.get("load_problems", []))
    if not run["done"]:
        problems.append("no DONE marker")
    cfg, prov = run["cfg"], run["prov"]
    if cfg:
        problems += pilot_config_problems(cfg, row)
    if prov:
        problems += pilot_provenance_problems(prov, row, frozen)
    steps = int(row["steps"])
    sc, stopped = prov.get("steps_completed"), prov.get("stopped_early")
    sc_ok = isinstance(sc, int) and not isinstance(sc, bool)
    if not isinstance(stopped, bool):
        problems.append("provenance.json has no stopped_early flag")
    if not sc_ok:
        problems.append("provenance.json has no steps_completed")
    elif sc > steps or (sc < steps and stopped is not True) or (sc == steps and stopped is True):
        problems.append(f"steps_completed is {sc}, stopped_early is {stopped!r}, and the row needs {steps} steps")
    elif sc == 0:
        problems.append("the run completed no training step")
    if sc_ok and 0 < sc <= steps:
        if [r.get("step") for r in run["train"]] != list(range(1, sc + 1)):
            problems.append("the training records do not cover every completed step once")
        rand = [r for r in run["eval"] if r.get("set") == "random"]
        if sorted(r.get("step") for r in rand) != [0, sc]:
            problems.append(f"the random held-out evaluations are not exactly at steps 0 and {sc}")
        for r in rand:
            if r.get("samples") != RT.EVAL_SAMPLES_RANDOM or r.get("n") != len(r.get("items", [])):
                problems.append(f"the random evaluation at step {r.get('step')} has a wrong sample or item count")
                continue
            got = _items_summary(r["items"])
            for key in ("accuracy", "strict_parse_fail_rate"):
                if r.get(key) is None or got[key] is None or abs(got[key] - r[key]) > PILOT_ITEM_TOL:
                    problems.append(f"the random evaluation at step {r['step']} has a {key} that its items do not give")
    if logs_dir is not None:
        files = pilot_log_files(logs_dir, prov)
        if not files:
            problems.append("no Slurm log was found for this run")
        else:
            text = "\n".join(p.read_text(errors="replace") for p in files)
            for word in ("REFUSED", "Traceback"):
                if word in text:
                    problems.append(f"the logs hold {word}")
    return problems


def pilot_d0(eval_records: List[dict], last_step: Optional[int] = None) -> dict:
    """Score the D0 rule from the random held-out evaluation records at step 0 and the last step.

    D0 passes when (a) the absolute difference of the two accuracies is at most 0.05 and (b) at least
    0.80 of the last step's completions parse strictly. The parse rate is one minus the logged strict
    parse failure rate. A rise of more than 0.05 is flagged separately and fails (a)."""
    by_step: Dict[int, List[dict]] = {}
    for r in eval_records:
        if r.get("set") == "random":
            by_step.setdefault(r["step"], []).append(r)
    if 0 not in by_step:
        raise ValueError("there is no random held-out evaluation at step 0")
    last = max(by_step) if last_step is None else last_step
    if last not in by_step or last == 0:
        raise ValueError("there is no random held-out evaluation at a last step after step 0")
    if len(by_step[0]) != 1 or len(by_step[last]) != 1:
        raise ValueError("a step holds more than one random held-out evaluation")
    first, final = by_step[0][0], by_step[last][0]
    for rec in (first, final):
        if rec.get("accuracy") is None or rec.get("strict_parse_fail_rate") is None:
            raise ValueError("a random held-out evaluation has no accuracy or no strict parse failure rate")
    diff = final["accuracy"] - first["accuracy"]
    parse_rate = 1.0 - final["strict_parse_fail_rate"]
    within = abs(diff) <= PILOT_ACC_TOL + PILOT_FLOAT_TOL
    parse_ok = parse_rate >= PILOT_PARSE_MIN - PILOT_FLOAT_TOL
    return {"first_step": 0, "last_step": last, "n_first": first.get("n"), "n_last": final.get("n"),
            "accuracy_first": first["accuracy"], "accuracy_last": final["accuracy"],
            "signed_difference": diff, "absolute_difference": abs(diff), "within_0_05": within,
            "rise_above_0_05": diff > PILOT_ACC_TOL + PILOT_FLOAT_TOL,
            "strict_parse_fail_rate_last": final["strict_parse_fail_rate"], "strict_parse_rate_last": parse_rate,
            "parse_at_least_0_80": parse_ok, "passes": bool(within and parse_ok)}


def pilot_d_choice(rows: List[dict]) -> dict:
    """Choose the study's single change among the passing rows by the pre-registered order.

    Each row needs task_id, lr, kl_beta, skip_zero_adv and passes. The order is the skip row, then the
    KL row, then the learning-rate row. A passing row outside those three kinds, such as the retry
    combination, is listed and never chosen. Two passing rows of one kind are listed in tied, and the
    lower task id is chosen."""
    order = {name: i for i, name in enumerate(PILOT_CHOICE_ORDER)}
    passing = [r for r in rows if r["passes"]]
    kinds = {r["task_id"]: _kind_of(float(r["lr"]), float(r["kl_beta"]), int(r["skip_zero_adv"])) for r in passing}
    ranked = sorted((r for r in passing if kinds[r["task_id"]] in order),
                    key=lambda r: (order[kinds[r["task_id"]]], r["task_id"]))
    chosen = ranked[0] if ranked else None
    tied = [r["task_id"] for r in ranked if chosen and kinds[r["task_id"]] == kinds[chosen["task_id"]]]
    return {"order": list(PILOT_CHOICE_ORDER), "passing_task_ids": sorted(r["task_id"] for r in passing),
            "chosen_task_id": chosen["task_id"] if chosen else None,
            "chosen_change": kinds[chosen["task_id"]] if chosen else None,
            "tied_task_ids": tied if len(tied) > 1 else [],
            "passing_outside_the_order": sorted(r["task_id"] for r in passing if kinds[r["task_id"]] not in order)}


def recheck_pilot_from_completions(run_dir: Path, dataset: dict, random_pids: List[str], steps: List[int]) -> dict:
    """Recompute the random held-out accuracy and strict parse rate at the given steps from completions.jsonl.

    Each completion is parsed with reasoning_task.strict_parse and judged against the reference answer
    with reasoning_task.verdict_fail. The result maps each step to n, accuracy and strict_parse_rate, and
    a problems list names any step whose completions do not cover the random set twice per problem."""
    refs = {c["pid"]: RT.parse_fraction_str(c["reference"])
            for c in dataset["train_candidates"] + dataset["test_candidates"]}
    want = set(random_pids)
    seen: Dict[int, Dict[str, List[int]]] = {s: {} for s in steps}
    parsed: Dict[int, List[bool]] = {s: [] for s in steps}
    problems: List[str] = []
    path = Path(run_dir) / "completions.jsonl"
    if not path.is_file():
        return {"problems": ["completions.jsonl is missing"], "steps": {}}
    with path.open() as fh:
        for line in fh:
            if '"eval"' not in line:
                continue
            rec = json.loads(line)
            if rec.get("phase") != "eval" or rec.get("set") != "random" or rec.get("step") not in seen:
                continue
            seen[rec["step"]].setdefault(rec["pid"], []).append(RT.verdict_fail(rec["text"], refs[rec["pid"]]))
            parsed[rec["step"]].append(RT.strict_parse(rec["text"]) is not None)
    out = {}
    for s in steps:
        by_pid = seen[s]
        if set(by_pid) != want or any(len(v) != RT.EVAL_SAMPLES_RANDOM for v in by_pid.values()):
            problems.append(f"the completions at step {s} do not hold {RT.EVAL_SAMPLES_RANDOM} samples of every random problem")
            continue
        out[s] = {"n": sum(len(v) for v in by_pid.values()),
                  "accuracy": float(np.mean([np.mean([f == 0 for f in v]) for v in by_pid.values()])),
                  "strict_parse_rate": sum(parsed[s]) / len(parsed[s])}
    return {"problems": problems, "steps": out}


def _recheck_problems(rechecked: dict, d0: dict) -> List[str]:
    problems = list(rechecked["problems"])
    for step, key in ((0, "accuracy_first"), (d0["last_step"], "accuracy_last")):
        got = rechecked["steps"].get(step)
        if got is None:
            continue
        if abs(got["accuracy"] - d0[key]) > PILOT_ITEM_TOL:
            problems.append(f"the completions at step {step} give accuracy {got['accuracy']}, and the record has {d0[key]}")
    got = rechecked["steps"].get(d0["last_step"])
    if got is not None and abs((1.0 - got["strict_parse_rate"]) - d0["strict_parse_fail_rate_last"]) > PILOT_ITEM_TOL:
        problems.append(f"the completions at step {d0['last_step']} give strict parse rate {got['strict_parse_rate']}, "
                        f"and the record has failure rate {d0['strict_parse_fail_rate_last']}")
    return problems


def _find_pilot_runs(runs_root: Path):
    by_task: Dict[int, List[dict]] = {}
    other = []
    for cfg_path in sorted(Path(runs_root).rglob("config.json")):
        run = load_pilot_run(cfg_path.parent)
        cfg = run["cfg"]
        if cfg.get("mode") == "manifest" and cfg.get("kind") == "pilot" and isinstance(cfg.get("task_id"), int):
            by_task.setdefault(cfg["task_id"], []).append(run)
        else:
            other.append(str(cfg_path.parent))
    return by_task, other


def pilot_report(manifest: Path, runs_root: Path, frozen: dict, logs_dir: Path,
                 dataset: Optional[dict] = None, selection: Optional[dict] = None) -> dict:
    """Admit and score every pilot row that has a run directory, and apply the D-choice.

    A manifest row without a run directory is left out of the report. A first-wave row (rows 1 to 3)
    without one is named in missing_first_wave_task_ids, because the D-choice then lacks a row."""
    rows = [r for r in read_manifest(manifest) if r["kind"] == "pilot"]
    by_task, other_dirs = _find_pilot_runs(runs_root)
    entries, missing = [], []
    for row in rows:
        tid = int(row["task_id"])
        runs = by_task.get(tid, [])
        if not runs:
            if pilot_is_first_wave(row):
                missing.append(tid)
            continue
        entry = {"task_id": tid, "change": pilot_row_kind(row), "lr": float(row["lr"]),
                 "kl_beta": float(row["kl_beta"]), "skip_zero_adv": int(row["skip_zero_adv"]),
                 "run_dir": str(runs[0]["dir"]), "d0": None, "recheck": "not run"}
        if len(runs) > 1:
            entry.update({"problems": ["more than one run directory for this row"], "admitted": False,
                          "eligible": False, "stopped_early": None, "steps_completed": None, "complete": False,
                          "passes": False})
            entries.append(entry)
            continue
        run = runs[0]
        problems = admit_pilot(run, row, frozen, logs_dir)
        entry.update(pilot_stop_state(run, row))
        try:
            entry["d0"] = pilot_d0(run["eval"], last_step=run["prov"].get("steps_completed"))
        except (ValueError, KeyError, TypeError) as exc:
            if not problems:
                problems.append(f"D0 cannot be scored: {exc}")
        if entry["d0"] is not None and dataset is not None and selection is not None:
            rechecked = recheck_pilot_from_completions(run["dir"], dataset, list(selection["random"]),
                                                       [0, entry["d0"]["last_step"]])
            more = _recheck_problems(rechecked, entry["d0"])
            problems += more
            entry["recheck"] = {"agrees": not more, "steps": rechecked["steps"]}
        entry["problems"] = problems
        entry["eligible"] = entry["d0"] is not None and not problems
        # admitted is True when the row has no problem and finished every step of its row.
        entry["admitted"] = not problems and entry["complete"]
        entry["passes"] = bool(entry["eligible"] and entry["d0"]["passes"])
        entries.append(entry)
    choice = pilot_d_choice([e for e in entries if e["eligible"]])
    complete_only = pilot_d_choice([e for e in entries if e["eligible"] and e["complete"]])
    unattributed = []
    matched = {p for r in [x for v in by_task.values() for x in v] for p in pilot_log_files(logs_dir, r["prov"])}
    for p in sorted(Path(logs_dir).rglob("*")):
        if p.is_file() and p not in matched:
            text = p.read_text(errors="replace")
            if PILOT_LOG_FAILURE.search(text):
                unattributed.append(p.name)
    return {"rule": {"accuracy_tolerance": PILOT_ACC_TOL, "parse_minimum": PILOT_PARSE_MIN,
                     "float_tolerance": PILOT_FLOAT_TOL, "d_choice_order": list(PILOT_CHOICE_ORDER)},
            "rows": entries, "missing_first_wave_task_ids": missing,
            "any_rise_above_0_05": any(e["d0"] is not None and e["d0"]["rise_above_0_05"] for e in entries),
            "d_choice": choice, "d_choice_complete_rows_only": complete_only,
            "log_files_with_REFUSED_or_Traceback_not_matched_to_a_row": unattributed,
            "run_directories_not_matched_to_a_pilot_row": other_dirs}


def pilot_table(report: dict) -> str:
    """Return a short plain-text table of the pilot report. It states the rule's outcomes and nothing more."""
    head = ("task", "change", "admitted", "stopped_early", "steps_done", "acc_step0", "acc_last", "signed_diff",
            "abs_diff<=0.05", "rise>0.05", "parse_last", "parse>=0.80", "D0")
    lines = ["\t".join(head)]
    for e in report["rows"]:
        d = e["d0"]
        if d is None:
            cells = ["-"] * 7
        else:
            cells = [f"{d['accuracy_first']:.4f}", f"{d['accuracy_last']:.4f}", f"{d['signed_difference']:+.4f}",
                     str(d["within_0_05"]), str(d["rise_above_0_05"]), f"{d['strict_parse_rate_last']:.4f}",
                     str(d["parse_at_least_0_80"])]
        verdict = "PASS" if e["passes"] else ("FAIL" if e["eligible"] else "NOT SCORED")
        lines.append("\t".join([str(e["task_id"]), e["change"], str(e["admitted"]), str(e["stopped_early"]),
                                str(e["steps_completed"])] + cells + [verdict]))
    for e in report["rows"]:
        for problem in e["problems"]:
            lines.append(f"task {e['task_id']} problem: {problem}")
    if report["missing_first_wave_task_ids"]:
        lines.append(f"no run directory for first-wave tasks {report['missing_first_wave_task_ids']}")
    c = report["d_choice"]
    if c["chosen_task_id"] is None:
        lines.append(f"D-choice: no scored row passes (passing rows {c['passing_task_ids']})")
    else:
        lines.append(f"D-choice: task {c['chosen_task_id']} ({c['chosen_change']} row), passing rows {c['passing_task_ids']}")
    lines.append(f"a rise of more than 0.05 occurs in a row: {report['any_rise_above_0_05']}")
    return "\n".join(lines) + "\n"


def pilot_main(argv) -> int:
    """The pilot-d0 mode. It writes pilot_d0.json and pilot_d0.txt to --out-dir."""
    ap = argparse.ArgumentParser(prog="analyze_reasoning.py pilot-d0",
                                 description="Admit the version-2 pilot rows, score D0 and apply the D-choice.")
    ap.add_argument("--manifest", type=Path, required=True, help="slurm/manifest_reasoning_pilot.csv")
    ap.add_argument("--frozen", type=Path, required=True, help="configs/reasoning_frozen_v2.json")
    ap.add_argument("--runs", type=Path, required=True, help="the directory that holds the pilot run directories")
    ap.add_argument("--logs", type=Path, required=True, help="the directory of the pulled Slurm logs")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="with --selection, also recompute the random-set scores from completions.jsonl")
    ap.add_argument("--selection", type=Path, default=None)
    args = ap.parse_args(argv)
    if (args.dataset is None) != (args.selection is None):
        ap.error("--dataset and --selection go together")
    try:
        frozen = json.loads(args.frozen.read_text())
        if (frozen.get("schema") != "wo_i_frozen_v2" or frozen.get("design_version") != RT.DESIGN_VERSION_V2
                or not all(isinstance(frozen.get(k), str) and len(frozen[k]) == 64
                           for k in ("reasoning_sources_digest", "training_sources_digest"))):
            raise Refusal("the frozen file is not a version-2 frozen file")
        dataset = json.loads(args.dataset.read_text()) if args.dataset else None
        selection = json.loads(args.selection.read_text()) if args.selection else None
        report = pilot_report(args.manifest, args.runs, frozen, args.logs, dataset, selection)
    except (Refusal, ValueError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pilot_d0.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (args.out_dir / "pilot_d0.txt").write_text(pilot_table(report))
    print(json.dumps({"out_dir": str(args.out_dir), "rows": [e["task_id"] for e in report["rows"]],
                      "d_choice_task_id": report["d_choice"]["chosen_task_id"]}))
    return 0


# ------------------------------------------------------------------ main


def analyze(manifest: Path, roots: List[Path], frozen: dict, selection: dict, not_run: Dict[str, str],
            logs_dir: Optional[Path] = None, w0_only: bool = False) -> dict:
    rows = read_manifest(manifest)
    found = find_runs(roots)
    admitted, refused = {}, {}
    for row in rows:
        tid = int(row["task_id"])
        cands = [r for r in found.get(tid, []) if r["done"]]
        if len(cands) > 1:
            refused[tid] = ["more than one finished run for this row"]
            continue
        if not cands:
            refused[tid] = ["no finished run"]
            continue
        reasons = admit(cands[0], row, frozen, logs_dir)
        if reasons:
            refused[tid] = reasons
        else:
            admitted[tid] = (row, cands[0])
    unexplained = sorted(t for t in refused if str(t) not in not_run)
    if unexplained:
        raise Refusal(f"rows {unexplained} have no admitted run and are not in the not_run list: "
                      + "; ".join(f"{t}: {', '.join(refused[t])}" for t in unexplained))
    report = {"admitted": sorted(admitted), "not_run": {k: v for k, v in sorted(not_run.items())},
              "W0": score_w0([run for _, run in admitted.values()])}
    if w0_only or not report["W0"]["holds"]:
        report["stopped_after_W0"] = True
        return report
    steps = {int(row["steps"]) for row, _ in admitted.values()}
    if len(steps) != 1:
        raise Refusal("the admitted runs do not share one step count")
    S = steps.pop()
    table, counters = [], {}
    for tid, (row, run) in sorted(admitted.items()):
        thr = selection["sizes"][row["size"]]["short_threshold"]
        counters[tid] = audit(run, thr)
        table.append({"task_id": tid, "size": row["size"], "arm": row["arm"], "seed": int(row["seed"]),
                      **final_endpoints(run, S, thr)})
    report.update({"table": table, "counters": {str(k): v for k, v in counters.items()},
                   "predictions": score_predictions(table, counters, True, S), "stopped_after_W0": False})
    return report


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "pilot-d0":
        return pilot_main(argv[1:])
    if argv and argv[0] == "study":
        # The study scorer lives in its own module. It is imported here, on demand, so the version-1 analysis
        # and the pilot mode never load it and the two modules cannot import each other in a cycle.
        import analyze_study2
        return analyze_study2.main(argv, prog="analyze_reasoning.py")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--runs", type=Path, action="append", required=True)
    ap.add_argument("--frozen", type=Path, required=True, help="JSON with reasoning_sources_digest")
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--not-run", type=Path, default=None)
    ap.add_argument("--logs", type=Path, required=True, help="the pulled Slurm logs, which admission reads")
    ap.add_argument("--w0-only", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--curves", type=Path, default=None,
                    help="also write the per-run curves of mean completion length, entropy_tok_mean and "
                         "kl_seq_mean of every run under --runs to this file; the main report is unchanged")
    args = ap.parse_args(argv)
    not_run = json.loads(args.not_run.read_text()) if args.not_run else {}
    if args.curves is not None:
        args.curves.write_text(json.dumps(curves_report(args.runs), indent=1, sort_keys=True) + "\n")
    try:
        report = analyze(args.manifest, args.runs, json.loads(args.frozen.read_text()),
                         json.loads(args.selection.read_text()), not_run, args.logs, args.w0_only)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "W0": report["W0"]["holds"],
                      "stopped_after_W0": report["stopped_after_W0"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
