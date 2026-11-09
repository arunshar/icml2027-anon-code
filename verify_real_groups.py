#!/usr/bin/env python3
"""Check P1 and P2 on groups the real policy actually produced.

Sampling only. No optimizer, no gradients, no reference copy, so this runs in about 2 GB and
fits on a laptop that cannot hold the full fine-tune. It answers the one question the closed
form cannot: do the completions this model really emits form groups where the identities hold.

Usage: python verify_real_groups.py --vmax 5 --groups 60 --out runs/real_groups_v5.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np

import rewards as RW
import train_speed_grpo as T
from speed_task import PROMPTS, parse_speed

ARM_SPECS = {"graded": ("graded", 5.0), "binary": ("binary", 5.0),
             "none": ("none", 0.0), "binary_1000x": ("binary", 5000.0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmax", type=float, required=True)
    ap.add_argument("--groups", type=int, default=60, help="groups per prompt")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import torch
    dev = T.pick_device(a.device)
    args = T.build_arg_parser().parse_args(["--arm", "none", "--w", "0", "--vmax", str(a.vmax),
                                            "--out-dir", "/dev/null", "--no-ref", "--device", dev])
    T.seed_everything(a.seed)
    tok, model, _ = T.build_models(args, dev)
    model.eval()

    groups = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = T.chat_ids(tok, p, dev)
            plen = ids.shape[1]
            for _ in range(a.groups):
                gen = T.sample_completions(model, tok, ids, args, a.group_size)
                groups.append([parse_speed(tok.decode(s[plen:], skip_special_tokens=True)) for s in gen])

    st = Counter()
    worst = {"P1": 0.0, "P2": 0.0}
    for sp in groups:
        cf = {n: RW.advantage(RW.group_rewards(sp, arm, w, a.vmax)) for n, (arm, w) in ARM_SPECS.items()}
        viol = np.array([s is None or abs(s) > a.vmax for s in sp])
        st["groups"] += 1
        if viol.all():
            st["all_violating"] += 1
            d = max(float(np.max(np.abs(cf["binary"] - cf["none"]))),
                    float(np.max(np.abs(cf["binary_1000x"] - cf["none"]))))
            worst["P1"] = max(worst["P1"], d)
            st["P1_fail"] += int(d > a.tol)
        if len(set(viol.tolist())) == 1:
            st["constant_verdict"] += 1
            d = float(np.max(np.abs(cf["binary"] - cf["binary_1000x"])))
            worst["P2"] = max(worst["P2"], d)
            st["P2_fail"] += int(d > a.tol)
        else:
            st["mixed"] += 1
            st["P2_boundary_fail"] += int(cf["binary_1000x"][~viol].min() <= cf["binary_1000x"][viol].max())
            st["binary_w5_violator_outranks_feasible"] += int((cf["binary"][viol] > cf["binary"][~viol].max()).any())
            st["graded_violator_outranks_feasible"] += int((cf["graded"][viol] > cf["graded"][~viol].max()).any())
        if viol.any():
            st["has_violator"] += 1
            for n in ARM_SPECS:
                st[f"{n}_reinforces_violator"] += int((cf[n][viol] > 0).any())

    vac = [k for k, n in (("P1", st["all_violating"]), ("P2", st["constant_verdict"]), ("P2_boundary", st["mixed"])) if n == 0]
    # Provenance, because this artifact went stale once and nothing noticed. The sampler was
    # fixed to pass every decoding knob explicitly, the script changed, and the JSON it had
    # already written was left in place. The paper kept citing numbers produced by the old
    # decoder. Recording the knobs and the commit makes that detectable instead of invisible.
    rep = {"device": T.device_name(dev), "v_max": a.vmax, "seed": a.seed, "tolerance": a.tol,
           "git_sha": T.git_sha(), "git_dirty": T.git_dirty(),
           "sampler": {"temperature": args.temperature, "top_p": args.top_p,
                       "top_k": args.top_k, "repetition_penalty": args.repetition_penalty,
                       "max_new_tokens": args.max_new_tokens,
                       "group_size": a.group_size},
           "counts": dict(st), "worst_abs_diff": worst, "vacuous_checks": vac,
           "P1_pass": st["all_violating"] > 0 and st["P1_fail"] == 0,
           "P2_pass": st["constant_verdict"] > 0 and st["P2_fail"] == 0,
           "P2_boundary_pass": st["mixed"] > 0 and st["P2_boundary_fail"] == 0,
           "parse_fail_rate": sum(s is None for g in groups for s in g) / (len(groups) * a.group_size)}
    # Counter values picked up numpy bool_ from the comparisons above; make them plain ints.
    rep["counts"] = {k: int(v) for k, v in rep["counts"].items()}
    txt = json.dumps(rep, indent=2, sort_keys=True, default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else int(o))
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)
    return 0 if not vac and rep["P1_pass"] and rep["P2_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
