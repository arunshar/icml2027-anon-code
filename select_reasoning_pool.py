#!/usr/bin/env python3
"""Apply the WO-I selection rules of Section 2.3 to calibration records, and later to smoke timing.

`select` reads the four calibration record sets and writes configs/reasoning_selection.json. It
sets L, checks the strict-parse rate, builds each size's training pool, hard held-out set and
prompt schedule, fixes the random held-out set, and computes the priors that the second
predictions entry records. It reads base-policy calibration records only.

`smoke-rules` reads only the timing.json file of each smoke, which holds step times, evaluation
times and peak memory and no outcome field. It applies the memory rule and the S rule and writes
configs/reasoning_smoke_rules.json.

Both commands stop with exit code 3 and a named reason when a stop rule of the plan fires, and they
write nothing in that case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import reasoning_task as RT

HERE = Path(__file__).resolve().parent
SELECTION_SCHEMA = "wo_i_reasoning_selection_v1"
SMOKE_RULES_SCHEMA = "wo_i_reasoning_smoke_rules_v1"


class Stop(RuntimeError):
    """A stop rule of the plan fired."""


# ------------------------------------------------------------------ calibration input


def load_records(path: Path) -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def check_calibration(dataset: dict, size: str, split: str, run_dir: Path,
                      require_device: Optional[str] = RT.REQUIRED_GPU_NAME) -> dict:
    """Return the records of one finished calibration task after checking them against the candidates.

    A calibration made with the tiny test model, or on a card other than the design's A100, is refused,
    because only the base models on that card set the rules of Section 2.3."""
    if not (run_dir / "DONE").exists():
        raise Stop(f"the {size} {split} calibration has no DONE marker")
    recs = load_records(run_dir / "records.jsonl")
    want = [c["pid"] for c in dataset[f"{split}_candidates"]]
    if [r["pid"] for r in recs] != want:
        raise Stop(f"the {size} {split} calibration does not cover the candidates in order")
    if any(len(r["samples"]) != RT.CAL_SAMPLES for r in recs):
        raise Stop(f"the {size} {split} calibration does not hold 8 samples per problem")
    prov = json.loads((run_dir / "provenance.json").read_text())
    if prov.get("tiny", False):
        raise Stop(f"the {size} {split} calibration used the tiny test model")
    if require_device is not None and prov.get("device_name") != require_device:
        raise Stop(f"the {size} {split} calibration ran on {prov.get('device_name')!r}, not {require_device}")
    snapshot = prov.get("model_snapshot")
    if snapshot is not None and not prov.get("tiny", False) and snapshot != prov["design"]["model_revision"]:
        raise Stop(f"the {size} {split} calibration loaded snapshot {snapshot}, not its pinned revision")
    check_attempts(prov, f"the {size} {split} calibration")
    return {"records": recs, "records_sha256": RT.file_sha256(run_dir / "records.jsonl"),
            "model_revision": prov["design"]["model_revision"], "model_snapshot": prov.get("model_snapshot"),
            "reasoning_sources_digest": prov.get("reasoning_sources_digest"),
            "bundle_digest": prov.get("bundle_digest"), "dataset_sha256": prov.get("dataset_sha256"),
            "attempts": [a.get("slurm_array_job_id") for a in prov.get("attempts", [])],
            "device_name": prov.get("device_name"), "tiny": prov.get("tiny", False)}


def check_attempts(prov: dict, what: str) -> None:
    """Refuse a calibration task whose attempts do not all match the first attempt.

    The top of provenance.json holds the first attempt's values, and every attempt records its own
    values of RT.CAL_ATTEMPT_FIELDS. A retry from another bundle, another source tree, another
    dataset file, another snapshot or another card therefore shows up here as a mismatch. An attempt
    that lacks a field, or that carries no source digest, is refused as well."""
    attempts = prov.get("attempts") or []
    if not attempts:
        raise Stop(f"{what} records no attempt")
    missing = [k for k in RT.CAL_ATTEMPT_FIELDS if k not in prov]
    if missing:
        raise Stop(f"the provenance of {what} does not record the first values of {', '.join(missing)}")
    for i, attempt in enumerate(attempts, start=1):
        missing = [k for k in RT.CAL_ATTEMPT_FIELDS if k not in attempt]
        if missing:
            raise Stop(f"attempt {i} of {what} does not record {', '.join(missing)}")
        if not attempt["reasoning_sources_digest"]:
            raise Stop(f"attempt {i} of {what} carries no source digest")
        differ = [k for k in RT.CAL_ATTEMPT_FIELDS if attempt[k] != prov[k]]
        if differ:
            raise Stop(f"the attempts of {what} carry different values of {', '.join(differ)}")


def exceeds(sample: dict, L: int) -> bool:
    """A completion exceeds L when it had no end token within 512 tokens or it is longer than L."""
    return (not sample["finished"]) or sample["n_tokens"] > L


def choose_L(all_samples: List[dict]) -> (Optional[int], Dict[str, float]):
    shares = {str(L): sum(exceeds(s, L) for s in all_samples) / len(all_samples) for L in RT.L_CANDIDATES}
    for L in RT.L_CANDIDATES:
        if shares[str(L)] <= RT.L_MAX_EXCEED_SHARE:
            return L, shares
    return None, shares


def correct_counts(recs: List[dict], L: int) -> Dict[str, int]:
    return {r["pid"]: sum(1 for s in r["samples"] if s["by_L"][str(L)]["F"] == 0) for r in recs}


def build_pool(train_cands: List[dict], c: Dict[str, int]) -> (List[str], float):
    """Return the largest prefix, sorted by c and then by candidate index, whose mean expected
    all-fail fraction is at least POOL_TARGET, together with that mean."""
    order = sorted(train_cands, key=lambda x: (c[x["pid"]], x["cand"]))
    q = [RT.expected_all_fail(c[x["pid"]]) for x in order]
    best, total = 0, 0.0
    for i, v in enumerate(q, start=1):
        total += v
        if total / i >= RT.POOL_TARGET - 1e-12:
            best = i
    pool = [x["pid"] for x in order[:best]]
    mean = float(np.mean(q[:best])) if best else 0.0
    return pool, mean


def build_hard(test_cands: List[dict], c: Dict[str, int], max_c: int, size: str) -> List[str]:
    eligible = [x for x in test_cands if c[x["pid"]] <= max_c]
    if len(eligible) > RT.HARD_CAP:
        rng = random.Random(f"{RT.CANDIDATE_SEED}:hard:{size}")
        eligible = sorted(rng.sample(eligible, RT.HARD_CAP), key=lambda x: x["cand"])
    return [x["pid"] for x in eligible]


def build_schedule(pool: List[str], size: str, steps: int = max(RT.S_CANDIDATES)) -> List[List[str]]:
    """Cycle one seeded shuffle of the pool, six problems per step."""
    order = list(pool)
    random.Random(f"{RT.CANDIDATE_SEED}:schedule:{size}").shuffle(order)
    k = RT.GROUPS_PER_STEP
    return [[order[(s * k + j) % len(order)] for j in range(k)] for s in range(steps)]


def w2_prior(recs: List[dict], pool: set, L: int) -> Optional[float]:
    """Return the share of all-fail base groups with a reward range above 1e-12 under the plain reward.
    The groups are samples 0 to 3 and samples 4 to 7 of each pool problem."""
    hits, total = 0, 0
    for r in recs:
        if r["pid"] not in pool:
            continue
        for part in (r["samples"][:4], r["samples"][4:]):
            fails = [s["by_L"][str(L)]["F"] for s in part]
            if not all(f == 1 for f in fails):
                continue
            n = [min(s["n_tokens"], L) for s in part]
            rw = RT.reasoning_rewards(fails, n, "linear", RT.W, 1.0, L)
            total += 1
            hits += bool(np.ptp(rw) > 1e-12)
    return hits / total if total else None


def w3_prior(schedule: List[List[str]], c: Dict[str, int], steps: int) -> float:
    """Return the expected number of steps with at least one mixed group, from each problem's c of 8."""
    expected = 0.0
    for pids in schedule[:steps]:
        none_mixed = 1.0
        for p in pids:
            mixed = 1.0 - RT.expected_all_fail(c[p]) - RT.expected_all_pass(c[p])
            none_mixed *= 1.0 - mixed
        expected += 1.0 - none_mixed
    return expected


def select(dataset: dict, dataset_sha256: str, cal: Dict[str, Dict[str, dict]]) -> dict:
    """Return the selection as a dict. cal[size][split] is the output of check_calibration.

    Each calibration must have read the same dataset file as this selection, because its verdicts
    were graded against that file's references."""
    for size in RT.SIZES:
        for split in ("train", "test"):
            if cal[size][split]["dataset_sha256"] != dataset_sha256:
                raise Stop(f"the {size} {split} calibration read a dataset file other than the one given "
                           "to the selection")
    all_samples = [s for size in RT.SIZES for split in ("train", "test")
                   for r in cal[size][split]["records"] for s in r["samples"]]
    L, shares = choose_L(all_samples)
    if L is None:
        raise Stop("no L candidate has at most 5 percent of calibration completions above it")
    sizes = {}
    for size in RT.SIZES:
        samples = [s for split in ("train", "test") for r in cal[size][split]["records"] for s in r["samples"]]
        parse_rate = sum(1 for s in samples if s["by_L"][str(L)]["strict"] is not None) / len(samples)
        if parse_rate < RT.MIN_PARSE_RATE:
            raise Stop(f"the {size} strict-parse rate is {parse_rate:.4f}, under 0.5")
        c_train = correct_counts(cal[size]["train"]["records"], L)
        c_test = correct_counts(cal[size]["test"]["records"], L)
        pool, pool_mean = build_pool(dataset["train_candidates"], c_train)
        max_c = max((c_train[p] for p in pool), default=-1)
        hard = build_hard(dataset["test_candidates"], c_test, max_c, size) if pool else []
        correct_n = [min(s["n_tokens"], L) for s in samples if s["by_L"][str(L)]["F"] == 0]
        entry = {"parse_rate": parse_rate, "pool_size": len(pool), "pool_mean_expected_all_fail": pool_mean,
                 "pool_max_c": max_c, "hard_size": len(hard), "c_train": c_train, "c_test": c_test,
                 "short_threshold": float(np.percentile(correct_n, 5)) if correct_n else None,
                 "short_threshold_method": "numpy.percentile(q=5, linear) over correct completions at L",
                 "base_accuracy": 1.0 - float(np.mean([s["by_L"][str(L)]["F"] for s in samples])),
                 "model_revision": cal[size]["train"]["model_revision"]}
        reasons = []
        if len(pool) < RT.POOL_FLOOR:
            reasons.append(f"the pool holds {len(pool)} problems, under {RT.POOL_FLOOR}")
        if len(hard) < RT.HARD_FLOOR:
            reasons.append(f"the hard held-out set holds {len(hard)} problems, under {RT.HARD_FLOOR}")
        if reasons:
            entry.update({"status": "dropped", "reasons": reasons})
        else:
            schedule = build_schedule(pool, size)
            entry.update({"status": "ok", "pool": pool, "hard": hard, "schedule": schedule,
                          "w2_prior": w2_prior(cal[size]["train"]["records"], set(pool), L),
                          "w3_prior_by_S": {str(s): w3_prior(schedule, c_train, s) for s in RT.S_CANDIDATES}})
        sizes[size] = entry
    if sizes["0.5b"]["status"] != "ok":
        raise Stop("the 0.5B size fails its floors: " + "; ".join(sizes["0.5b"]["reasons"]))
    for size in RT.SIZES:
        if cal[size]["train"]["model_revision"] != cal[size]["test"]["model_revision"]:
            raise Stop(f"the {size} train and test calibrations used different model revisions")
    digests = {cal[size][split]["reasoning_sources_digest"] for size in RT.SIZES for split in ("train", "test")}
    if len(digests) != 1:
        raise Stop("the four calibration tasks ran from different source digests")
    return {
        "schema": SELECTION_SCHEMA,
        "candidates_sha256": dataset_sha256,
        "L": L,
        "exceed_share_by_L": shares,
        "random": [x["pid"] for x in dataset["test_candidates"][:RT.RANDOM_SET_SIZE]],
        "sizes": sizes,
        "calibration_sources": {size: {split: {k: v for k, v in cal[size][split].items() if k != "records"}
                                       for split in ("train", "test")} for size in RT.SIZES},
        "rules": {"L_candidates": list(RT.L_CANDIDATES), "L_max_exceed_share": RT.L_MAX_EXCEED_SHARE,
                  "pool_target": RT.POOL_TARGET, "pool_floor": RT.POOL_FLOOR, "hard_cap": RT.HARD_CAP,
                  "hard_floor": RT.HARD_FLOOR, "random_set_size": RT.RANDOM_SET_SIZE,
                  "min_parse_rate": RT.MIN_PARSE_RATE, "seed": RT.CANDIDATE_SEED},
    }


# ------------------------------------------------------------------ smoke rules

TIMING_KEYS = {"kind", "size", "arm", "L", "micro_batch", "steps", "selection_sha256", "device_name",
               "model_load_seconds", "step_seconds", "generation_seconds", "update_seconds", "eval_seconds",
               "peak_memory_reserved_bytes", "peak_memory_allocated_bytes", "steps_completed"}


def smoke_rules(selection: dict, selection_sha256: str, timings: Dict[str, dict],
                require_device: Optional[str] = RT.REQUIRED_GPU_NAME) -> dict:
    """Apply the memory rule and the S rule to smoke timing records only."""
    kept = [s for s in RT.SIZES if selection["sizes"][s]["status"] == "ok"]
    report, projected = {}, {}
    for size in kept:
        t = timings.get(size)
        if t is None:
            raise Stop(f"the {size} smoke timing is missing")
        if set(t) != TIMING_KEYS:
            raise Stop(f"the {size} timing file has unexpected fields")
        if (t["kind"], t["size"], t["arm"], t["L"], t["micro_batch"], t["steps"]) != (
                "smoke", size, "plain", selection["L"], RT.MICRO_BATCH, RT.SMOKE_STEPS):
            raise Stop(f"the {size} timing is not a smoke of this design")
        if t["selection_sha256"] != selection_sha256 or t["steps_completed"] != RT.SMOKE_STEPS:
            raise Stop(f"the {size} smoke did not finish on this selection")
        if require_device is not None and t["device_name"] != require_device:
            raise Stop(f"the {size} smoke ran on {t['device_name']}")
        peak = t["peak_memory_reserved_bytes"]
        entry = {"peak_memory_reserved_bytes": peak, "peak_memory_allocated_bytes": t["peak_memory_allocated_bytes"]}
        if peak is None or peak > RT.MEMORY_LIMIT_BYTES:
            entry.update({"status": "stopped", "reason": "peak memory above 36 GB or not measured"})
            report[size] = entry
            continue
        full = [e["seconds"] for e in t["eval_seconds"] if e["step"] in (0, RT.SMOKE_STEPS)]
        mid = [e["seconds"] for e in t["eval_seconds"] if e["step"] not in (0, RT.SMOKE_STEPS)]
        if len(full) != 4 or len(mid) != 1:
            raise Stop(f"the {size} smoke does not hold the expected evaluation timings")
        full_eval = sum(full) / 2.0
        entry["projected_minutes_by_S"] = {str(S): RT.project_minutes(t["model_load_seconds"], t["step_seconds"],
                                                                     full_eval, mid[0], S)
                                           for S in RT.S_CANDIDATES}
        entry["status"] = "ok"
        report[size] = entry
        projected[size] = entry["projected_minutes_by_S"]
    if report.get("0.5b", {}).get("status") != "ok":
        raise Stop("the 0.5B smoke stops the block (peak memory above 36 GB)")
    slowest = {S: max(projected[s][str(S)] for s in projected) for S in RT.S_CANDIDATES}
    S = RT.choose_steps(slowest)
    if S is None:
        raise Stop("S = 150 does not fit within 120 minutes at the slower size")
    return {"schema": SMOKE_RULES_SCHEMA, "selection_sha256": selection_sha256, "S": S,
            "sizes_kept": [s for s in kept if report[s]["status"] == "ok"],
            "sizes_stopped": [s for s in kept if report[s]["status"] != "ok"],
            "slowest_projected_minutes_by_S": {str(k): v for k, v in slowest.items()},
            "sizes": report, "memory_limit_bytes": RT.MEMORY_LIMIT_BYTES,
            "planning_minutes": RT.PLANNING_MINUTES}


# ------------------------------------------------------------------ main


def dumps(obj: dict) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("select")
    a.add_argument("--dataset", type=Path, default=HERE / "configs" / "reasoning_dataset.json")
    for size in RT.SIZES:
        for split in ("train", "test"):
            a.add_argument(f"--cal-{size.replace('.', 'p')}-{split}", type=Path, required=True)
    a.add_argument("--out", type=Path, default=HERE / "configs" / "reasoning_selection.json")
    b = sub.add_parser("smoke-rules")
    b.add_argument("--selection", type=Path, default=HERE / "configs" / "reasoning_selection.json")
    b.add_argument("--timing-0p5b", type=Path, required=True)
    b.add_argument("--timing-1p5b", type=Path, default=None)
    b.add_argument("--out", type=Path, default=HERE / "configs" / "reasoning_smoke_rules.json")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "select":
            dataset_sha = RT.file_sha256(args.dataset)
            dataset = json.loads(args.dataset.read_text())
            cal = {size: {split: check_calibration(dataset, size, split,
                                                   getattr(args, f"cal_{size.replace('.', 'p')}_{split}"))
                          for split in ("train", "test")} for size in RT.SIZES}
            payload = dumps(select(dataset, dataset_sha, cal))
        else:
            sel_sha = RT.file_sha256(args.selection)
            selection = json.loads(args.selection.read_text())
            timings = {"0.5b": json.loads(args.timing_0p5b.read_text())}
            if args.timing_1p5b is not None:
                timings["1.5b"] = json.loads(args.timing_1p5b.read_text())
            payload = dumps(smoke_rules(selection, sel_sha, timings))
    except Stop as exc:
        print(f"STOP: {exc}. Nothing was written, and the executor reports.", file=sys.stderr)
        return 3
    args.out.write_bytes(payload)
    print(json.dumps({"out": str(args.out), "sha256": hashlib.sha256(payload).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
