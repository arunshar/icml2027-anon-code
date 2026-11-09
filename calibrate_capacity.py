#!/usr/bin/env python3
"""Choose the two capacity ratios by sampling the base policy. No training, no optimizer.

The control variable of the whole experiment is how often a group contains no feasible member.
The hard level wants that fraction at or above 0.95, so the verdict has almost no mixed group to
learn from. The easy level wants roughly 0.5, so it does.

This also reports the parse-failure rate, which is the risk that sinks the task: if the base model
rarely emits a bracketed list, every group is trivially all-violating for the wrong reason. If the
rate is above about 0.4 the answer format changes to a fixed-width bitstring before anything is
frozen.

Sampling only, so it runs in about 2 GB and does not need the cluster.

Usage: python calibrate_capacity.py --groups 40 --device mps --out runs/capacity_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

import numpy as np

import capacity_task as C
import tasks
import train_speed_grpo as T


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=40, help="groups of K per prompt")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--candidates", default="0.10,0.15,0.20,0.25,0.35,0.50,0.65,0.75,0.90")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import torch
    spec = tasks.get("capacity")
    dev = T.pick_device(a.device)
    args = T.build_arg_parser().parse_args(
        ["--task", "capacity", "--arm", "none", "--w", "0", "--vmax", "0.5",
         "--out-dir", "/dev/null", "--no-ref", "--device", dev,
         "--max-new-tokens", str(a.max_new_tokens)])
    T.seed_everything(a.seed)
    tok, model, _ = T.build_models(args, dev)
    model.eval()
    eos = sorted({i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None})

    # groups[p] is a list of groups for prompt p; each group is K parsed selections
    groups: List[List] = []
    n_tokens = []
    with torch.no_grad():
        for pi, prompt in enumerate(spec.PROMPTS):
            parse_p = spec.parser_for(pi)
            ids = T.chat_ids(tok, prompt, dev)
            plen = ids.shape[1]
            for _ in range(a.groups):
                gen = T.sample_completions(model, tok, ids, args, a.group_size)
                texts = [tok.decode(s[plen:], skip_special_tokens=True) for s in gen]
                n_tokens.append(float(np.mean([len(tok(t).input_ids) for t in texts])))
                groups.append([parse_p(t) for t in texts])

    flat = [s for g in groups for s in g]
    n = len(flat)
    parse_fail = sum(1 for s in flat if s is None) / n
    sizes = [len(s) - 1 for s in flat if s is not None]

    rows = []
    for ratio in [float(x) for x in a.candidates.split(",")]:
        allv = float(np.mean([all(not C.verdict(s, ratio) for s in g) for g in groups]))
        anyv = float(np.mean([any(not C.verdict(s, ratio) for s in g) for g in groups]))
        per_sample = float(np.mean([not C.verdict(s, ratio) for s in flat]))
        rows.append({"ratio": ratio, "all_violating_groups": round(allv, 4),
                     "groups_with_a_violator": round(anyv, 4),
                     "per_sample_violation": round(per_sample, 4)})
        print(json.dumps(rows[-1]))

    def pick(target, prefer_ge):
        ok = [r for r in rows if (r["all_violating_groups"] >= target if prefer_ge
                                  else r["all_violating_groups"] <= target)]
        if not ok:
            return None
        # the least extreme ratio that still meets the target
        return (max if prefer_ge else min)(ok, key=lambda r: r["ratio"])["ratio"]

    rep = {
        # Provenance, so a build can tell whether this artifact is older than the code that
        # wrote it. That went undetected once: the sampler was fixed and the JSON it had already
        # produced was left in place, and the ratios were read from the stale file.
        "git_sha": T.git_sha(), "git_dirty": T.git_dirty(),
        "sampler": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                    "max_new_tokens": args.max_new_tokens},
        "device": T.device_name(dev), "seed": a.seed, "groups": len(groups), "samples": n,
        "group_size": a.group_size, "max_new_tokens": a.max_new_tokens,
        "parse_fail_rate": round(parse_fail, 4),
        "mean_completion_tokens": round(float(np.mean(n_tokens)), 1),
        "selection_size": {
            "mean": round(float(np.mean(sizes)), 2) if sizes else None,
            "p10": float(np.percentile(sizes, 10)) if sizes else None,
            "p50": float(np.percentile(sizes, 50)) if sizes else None,
            "p90": float(np.percentile(sizes, 90)) if sizes else None,
        },
        "sweep": rows,
        "suggested_hard_ratio": pick(0.95, True),
        "suggested_easy_ratio": pick(0.55, False),
        "parse_fail_gate": "PASS" if parse_fail <= 0.4 else "FAIL: switch to a bitstring format",
    }
    print(json.dumps({k: rep[k] for k in
                      ("parse_fail_rate", "mean_completion_tokens", "selection_size",
                       "suggested_hard_ratio", "suggested_easy_ratio", "parse_fail_gate")},
                     indent=2))
    if a.out:
        open(a.out, "w").write(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
