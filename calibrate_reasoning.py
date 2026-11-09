#!/usr/bin/env python3
"""This script samples the WO-I base-policy calibration at 8 samples per candidate and 512 tokens.

One task samples one model on one split's candidates with the trainer's sampler and batched
generation. Each problem gets one line in records.jsonl that holds its 8 completions, with token
ids, text, the token count through the first end token, whether an end token came, and both parses.
For every L candidate of Section 2.3 it also stores the strict verdict after truncating the
completion at L tokens, so the selection never needs a tokenizer.

The problems are sampled in fixed chunks of 8, and the random state is reseeded at the start of
each chunk from the row's seed and the chunk's first index. A chunk's lines are written in one
call and flushed. A retry of a task that timed out keeps every finished chunk, drops any partial
chunk, and resumes at the first unfinished problem with the same draws an uninterrupted task makes.

The run directory is <out-root>/calibration/<task>_<size>_<split>, the same for every attempt, so
a retry finds the records of the first attempt. Only calibration records feed the selection rules.

The top of provenance.json holds the values of the first attempt, and a retry never overwrites
them. Every attempt also records its own source digests, its bundle digest, its dataset digest,
its model snapshot, its card, its tiny flag and its design, which reasoning_task.CAL_ATTEMPT_FIELDS
names.
The bundle digest is the SHA-256 of the SHA256SUMS file beside this script, which is the value
that names the bundle, and it is null when the script runs from the repository. The selection
refuses a task whose attempts disagree on any of these fields, so the records of a retry from
another bundle cannot enter the selection.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import reasoning_task as RT
import train_reasoning_grpo as TR
import train_speed_grpo as TSG

HERE = Path(__file__).resolve().parent


def bundle_digest(exp_dir=None):
    """Return the SHA-256 of the SHA256SUMS file in exp_dir, or None when the directory has none.

    The directory defaults to the one that holds this script. package_reasoning_source.py names
    each bundle by this value, and the cluster checks every bundle file against SHA256SUMS before
    this script starts, so the value identifies the bundle."""
    sums = Path(HERE if exp_dir is None else exp_dir) / "SHA256SUMS"
    return RT.file_sha256(sums) if sums.is_file() else None


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="the frozen calibration manifest, which sets every field together with --row")
    ap.add_argument("--row", type=int)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--size", choices=RT.SIZES, default="0.5b")
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--samples", type=int, default=RT.CAL_SAMPLES)
    ap.add_argument("--max-new-tokens", type=int, default=RT.CAL_MAX_NEW_TOKENS)
    ap.add_argument("--chunk-problems", type=int, default=RT.CAL_CHUNK_PROBLEMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-sha256", default=None)
    ap.add_argument("--limit-problems", type=int, default=None, help="caps the candidates, for tests only")
    ap.add_argument("--stop-after-chunks", type=int, default=None,
                    help="stops after this many new chunks, as a timeout would, for tests only")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tiny", action="store_true", help="a random 2-layer model with the real tokenizer, for tests only")
    ap.add_argument("--require-gpu-name", default=None)
    ap.add_argument("--time-limit-minutes", type=int, default=None)
    return ap


def resolve(args) -> dict:
    if args.manifest:
        if args.row is None:
            raise TR.Refused("--manifest needs --row")
        row = RT.validate_calibration_row(RT.read_manifest_row(args.manifest, args.row))
        return {"mode": "manifest", "task_id": row["task_id"], "size": row["size"], "split": row["split"],
                "model_id": row["model_id"], "model_revision": row["model_revision"],
                "samples": row["samples"], "max_new_tokens": row["max_new_tokens"],
                "chunk_problems": row["chunk_problems"], "seed": row["seed"],
                "dataset_sha256": row["dataset_sha256"]}
    model = RT.MODELS[args.size]
    return {"mode": "explicit", "task_id": 0, "size": args.size, "split": args.split,
            "model_id": model["model_id"], "model_revision": model["revision"], "samples": args.samples,
            "max_new_tokens": args.max_new_tokens, "chunk_problems": args.chunk_problems,
            "seed": args.seed, "dataset_sha256": args.dataset_sha256}


def read_finished(path: Path, chunk: int) -> list:
    """Return the complete problem lines that form whole chunks, in order. Later lines are dropped."""
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text().split("\n"):
        if not raw:
            continue
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            break
    keep = (len(lines) // chunk) * chunk
    return lines[:keep]


def problem_line(tok, pid: str, cand: int, chunk_start: int, reference: str, seqs, plen: int, eos_ids) -> dict:
    recs, _ = TR.completion_records(tok, seqs, plen, eos_ids)
    ref = RT.parse_fraction_str(reference)
    samples = []
    for j, r in enumerate(recs):
        strict = RT.strict_parse(r["text"])
        by_L = {}
        for L in RT.L_CANDIDATES:
            ids = r["token_ids"][:min(r["n_tokens"], L)]
            text = tok.decode(ids, skip_special_tokens=True)
            parsed = RT.strict_parse(text)
            by_L[str(L)] = {"strict": RT.fraction_str(parsed),
                            "F": 0 if (parsed is not None and parsed == ref) else 1}
        samples.append({"sample": j, "n_tokens": r["n_tokens"], "finished": r["finished"],
                        "token_ids": r["token_ids"], "text": r["text"], "strict": RT.fraction_str(strict),
                        "lenient": RT.fraction_str(RT.lenient_parse(r["text"])), "by_L": by_L})
    return {"pid": pid, "cand": cand, "chunk_start": chunk_start, "reference": reference, "samples": samples}


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return _run(args)
    except (TR.Refused, ValueError) as exc:
        print(f"REFUSED: {exc}. Nothing was sampled.", file=sys.stderr, flush=True)
        return 2


def _run(args) -> int:
    d = resolve(args)
    TR.check_time_limit("calibration", args.time_limit_minutes)
    dataset = TR.load_json_checked(args.dataset, d["dataset_sha256"], TR.DATASET_SCHEMA)
    cands = dataset[f"{d['split']}_candidates"]
    if args.limit_problems is not None:
        cands = cands[:args.limit_problems]
    out = Path(args.out_root) / "calibration" / f"{d['task_id']}_{d['size']}_{d['split']}"
    if (out / "DONE").exists():
        print(json.dumps({"event": "already_done", "out": str(out)}), flush=True)
        return 0
    dev = TSG.pick_device(args.device)
    if args.require_gpu_name is not None and TSG.device_name(dev) != args.require_gpu_name:
        raise TR.Refused(f"the device is {TSG.device_name(dev)!r}, and the design needs {args.require_gpu_name!r}")
    records = out / "records.jsonl"
    chunk = d["chunk_problems"]
    finished = read_finished(records, chunk)
    expected = [c["pid"] for c in cands[:len(finished)]]
    if [x["pid"] for x in finished] != expected:
        raise TR.Refused("the existing records do not match this task's candidates")

    # The model loads before anything is written, so a missing model leaves no directory behind.
    tiny = RT.MODELS["0.5b"]
    t0 = time.time()
    TSG.seed_everything(d["seed"])  # this seed fixes the random weights of the tiny test model
    tok, model, info = TR.load_model(tiny["model_id"] if args.tiny else d["model_id"],
                                     tiny["revision"] if args.tiny else d["model_revision"], dev, tiny=args.tiny)
    out.mkdir(parents=True, exist_ok=True)
    with records.open("w") as fh:  # this rewrite keeps whole chunks and drops a partial one
        fh.write("".join(json.dumps(x) + "\n" for x in finished))
    load_seconds = time.time() - t0
    model.eval()
    eos_ids = TR.eos_ids_of(tok)
    sampler = TR.Sampler()
    prov_path = out / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {"attempts": []}
    own = {"reasoning_sources_digest": RT.reasoning_sources_digest(),
           "training_sources_digest": TSG.training_sources_digest(), "bundle_digest": bundle_digest(),
           "dataset_sha256": RT.file_sha256(args.dataset), "model_snapshot": info["snapshot"],
           "device_name": TSG.device_name(dev), "tiny": args.tiny, "design": d}
    first = {**own, "model_file_sha256": info["file_sha256"], "sampler": vars(sampler),
             "eos_ids": eos_ids, "n_candidates": len(cands)}
    for key, value in first.items():
        prov.setdefault(key, value)  # a retry keeps the first attempt's values here
    attempt = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
               "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "hostname": socket.gethostname(),
               "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "resumed_at_problem": len(finished),
               "model_load_seconds": load_seconds, **own}
    prov["attempts"].append(attempt)
    prov_path.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")

    new_chunks = 0
    with records.open("a") as fh:
        for start in range(len(finished), len(cands), chunk):
            if args.stop_after_chunks is not None and new_chunks >= args.stop_after_chunks:
                print(json.dumps({"event": "stopped_early", "at_problem": start}), flush=True)
                return 0
            part = cands[start:start + chunk]
            TSG.seed_everything(d["seed"] + start)
            groups = TR.batched_generate(model, tok, [TR.encode_prompt(tok, c["prompt"]) for c in part],
                                         d["samples"], sampler, d["max_new_tokens"], dev)
            lines = [problem_line(tok, c["pid"], c["cand"], start, c["reference"], seqs, plen, eos_ids)
                     for c, (seqs, plen) in zip(part, groups)]
            fh.write("".join(json.dumps(x) + "\n" for x in lines))
            fh.flush()
            os.fsync(fh.fileno())
            new_chunks += 1
            if start // chunk % 10 == 0:
                print(json.dumps({"event": "chunk", "at_problem": start, "elapsed": round(time.time() - t0, 1)}),
                      flush=True)
    attempt["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov_path.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    done = read_finished(records, 1)
    if [x["pid"] for x in done] != [c["pid"] for c in cands]:
        raise RuntimeError("the records do not cover every candidate")
    (out / "DONE").write_text("ok\n")
    print(json.dumps({"event": "done", "out": str(out), "problems": len(done)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
