"""This module builds synthetic WO-I files for the tests, which are a dataset, a selection and runs.

Nothing here downloads a file or reads the real GSM8K data. Every value is built from a seed, so
the tests that use these helpers are deterministic.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

import reasoning_task as RT
import rewards as RW
import select_reasoning_pool as SEL


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj) -> str:
    raw = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def tiny_dataset(n_train=12, n_test=10) -> dict:
    train = [{"cand": i, "pid": f"tr{i}", "row": i, "split": "train", "question": f"What is {i} plus {i + 1}?",
              "prompt": RT.prompt_text(f"What is {i} plus {i + 1}?"), "reference": str(2 * i + 1)}
             for i in range(n_train)]
    test = [{"cand": i, "pid": f"te{i}", "row": i, "split": "test", "question": f"What is {i} times 3?",
             "prompt": RT.prompt_text(f"What is {i} times 3?"), "reference": str(3 * i)}
            for i in range(n_test)]
    return {"schema": "wo_i_reasoning_dataset_v1", "train_candidates": train, "test_candidates": test}


def tiny_selection(dataset_sha: str, dataset: dict, L=16, steps=10, short_threshold=5.0) -> dict:
    pool = [c["pid"] for c in dataset["train_candidates"]]
    tests = [c["pid"] for c in dataset["test_candidates"]]
    sched = [[pool[(s * 6 + j) % len(pool)] for j in range(6)] for s in range(steps)]
    return {"schema": "wo_i_reasoning_selection_v1", "candidates_sha256": dataset_sha, "L": L,
            "sizes": {"0.5b": {"status": "ok", "pool": pool, "hard": tests[:3], "schedule": sched,
                               "short_threshold": short_threshold, "model_revision": RT.MODELS["0.5b"]["revision"]},
                      "1.5b": {"status": "dropped", "reasons": ["test"]}},
            "random": tests[3:5]}


def write_tiny_design(tmp: Path, **kw):
    ds = tiny_dataset()
    ds_sha = write_json(tmp / "ds.json", ds)
    sel_sha = write_json(tmp / "sel.json", tiny_selection(ds_sha, ds, **kw))
    return tmp / "ds.json", ds_sha, tmp / "sel.json", sel_sha


# ------------------------------------------------------------------ calibration records


def sample_record(j, n, finished, F_by_L, parsed=True):
    return {"sample": j, "n_tokens": n, "finished": finished, "token_ids": [], "text": "",
            "strict": None, "lenient": None,
            "by_L": {str(L): {"F": F_by_L[L], "strict": "1" if parsed else None} for L in RT.L_CANDIDATES}}


def calibration_dir(root: Path, dataset: dict, split: str, size: str, correct_of, lengths_of=None,
                    revision=None, parsed=True) -> Path:
    """Write one finished calibration directory. correct_of(cand) gives c of 8 at every L."""
    d = root / f"{size}_{split}"
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for c in dataset[f"{split}_candidates"]:
        k = correct_of(c)
        samples = []
        for j in range(RT.CAL_SAMPLES):
            n, fin = lengths_of(c, j) if lengths_of else (100 + j, True)
            f = 0 if j < k else 1
            samples.append(sample_record(j, n, fin, {L: f for L in RT.L_CANDIDATES}, parsed))
        lines.append({"pid": c["pid"], "cand": c["cand"], "chunk_start": 0, "reference": c["reference"],
                      "samples": samples})
    (d / "records.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    revision = revision or RT.MODELS[size]["revision"]
    own = calibration_attempt_fields(revision)
    (d / "provenance.json").write_text(json.dumps({**own, "attempts": [{"slurm_array_job_id": "1", **own}]}))
    (d / "DONE").write_text("ok\n")
    return d


FAKE_SOURCE_DIGEST = "5" * 64
FAKE_BUNDLE_DIGEST = "b" * 64


def calibration_attempt_fields(revision: str, **kw) -> dict:
    """Return the fields that one calibration attempt records for itself, as the A100 run writes them."""
    own = {"reasoning_sources_digest": FAKE_SOURCE_DIGEST, "training_sources_digest": "7" * 64,
           "bundle_digest": FAKE_BUNDLE_DIGEST, "dataset_sha256": "d" * 64, "model_snapshot": revision,
           "device_name": RT.REQUIRED_GPU_NAME, "tiny": False, "design": {"model_revision": revision}}
    assert set(own) == set(RT.CAL_ATTEMPT_FIELDS)
    own.update(kw)
    return own


def synthetic_candidates(n_train=1000, n_test=400) -> dict:
    return {"train_candidates": [{"cand": i, "pid": f"tr{i}", "reference": "1"} for i in range(n_train)],
            "test_candidates": [{"cand": i, "pid": f"te{i}", "reference": "1"} for i in range(n_test)]}


def calibration_set(root: Path, dataset: dict, c_fn, **kw):
    return {size: {split: SEL.check_calibration(dataset, size, split,
                                                calibration_dir(root, dataset, split, size, c_fn, **kw))
                   for split in ("train", "test")} for size in RT.SIZES}


# ------------------------------------------------------------------ runs for the analysis


def fake_run(root: Path, row: dict, rng: random.Random, frozen_digest: str, *, tamper=False,
             all_fail_rate=0.5, acc_shift=0.0) -> Path:
    """Write a finished run directory whose logs follow the trainer's formats for one manifest row."""
    steps, L = int(row["steps"]), int(row["L"])
    spec = RT.ARMS[row["arm"]]
    d = root / f"{row['task_id']}_study_{row['size']}_{row['arm']}_s{row['seed']}"
    d.mkdir(parents=True)
    lines = []
    for step in range(steps + 1):
        for name, k in RT.eval_plan(step, steps, RT.EVAL_EVERY):
            items = []
            for p in range(4):
                for j in range(k):
                    f = int(rng.random() > 0.3 + acc_shift)
                    items.append([f"te{p}", j, rng.randint(1, L), True, f, None if f else "1", None, 0])
            acc = float(np.mean([1 - np.mean([it[4] for it in items if it[0] == f"te{p}"]) for p in range(4)]))
            lines.append({"step": step, "phase": "eval", "set": name, "samples": k, "accuracy": acc,
                          "lenient_accuracy": acc, "items": items})
        if step == 0:
            continue
        groups = []
        taken = False
        for g in range(6):
            if rng.random() < all_fail_rate:
                F = [1, 1, 1, 1]
            else:
                F = [int(rng.random() < 0.5) for _ in range(4)]
            n = [rng.randint(1, L) for _ in range(4)]
            r = RT.reasoning_rewards(F, n, spec["form"], RT.W, spec["w_task"], L)
            a = RW.advantage_variant(r, spec["adv_variant"])
            kept = RW.keep_group(spec["group_filter"], r, [f == 0 for f in F])
            taken = taken or kept
            if tamper and step == 1 and g == 0:
                a = a + 1e-3
            groups.append({"pid": f"tr{g}", "n_tokens": n, "F": F, "finished": [True] * 4,
                           "strict": [None] * 4, "lenient": [None] * 4, "rewards": r.tolist(),
                           "adv": a.tolist(), "kept": kept, "all_fail": all(F), "all_pass": not any(F)})
        lines.insert(len(lines) - len(RT.eval_plan(step, steps, RT.EVAL_EVERY)),
                     {"step": step, "phase": "train", "opt_step_taken": taken, "groups": groups})
    (d / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    cfg = {"mode": "manifest", "kind": "study", "task_id": int(row["task_id"]), "manifest_row": dict(row),
           "arm": row["arm"], "size": row["size"], "seed": int(row["seed"]), "L": L, "steps": steps,
           "form": spec["form"], "w": RT.W, "w_task": spec["w_task"], "adv_variant": spec["adv_variant"],
           "group_filter": spec["group_filter"],
           # These are the held-equal settings that the trainer writes on the cluster, typed out here.
           "tiny": False, "eval_limit": None, "device": "cuda", "lr": 1e-5, "weight_decay": 0.0,
           "clip_coef": 0.2, "grad_clip": 1.0, "eval_seed": 12345, "eval_chunk_sequences": 64,
           "mid_eval_every": 50, "micro_batch": 2, "group_size": 4, "groups_per_step": 6,
           "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.0}}
    (d / "config.json").write_text(json.dumps(cfg))
    prov = {"reasoning_sources_digest": frozen_digest, "dataset_sha256": row["dataset_sha256"],
            "selection_sha256": row["selection_sha256"], "model_revision": row["model_revision"],
            "model_snapshot": row["model_revision"], "device_name": RT.REQUIRED_GPU_NAME,
            "slurm_array_job_id": "900", "slurm_array_task_id": row["task_id"]}
    (d / "provenance.json").write_text(json.dumps(prov))
    (d / "DONE").write_text("ok\n")
    return d
