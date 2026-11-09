#!/usr/bin/env python3
"""Measure the base model's speed distribution and the all-violating group fraction per v_max.

Regime A needs a v_max at which at least 80 percent of K=4 groups are all-violating at
step 0. This samples the untrained policy with the eval seed and reports, for candidate
v_max values, the per-sample violation rate p, the estimate p^4, and the exact fraction
over the sampled groups. No training happens here.
"""
import argparse
import json
import sys

import numpy as np

import train_speed_grpo as T
from speed_task import PROMPTS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--samples", type=int, default=32, help="per prompt")
    ap.add_argument("--candidates", default="5,8,10,12,15,20,25,30")
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args(argv)
    dev = T.pick_device(a.device)
    args = T.build_arg_parser().parse_args(["--arm", "none", "--w", "0", "--vmax", "30", "--out-dir", "/dev/null",
                                            "--no-ref", "--device", dev, "--eval-samples", str(a.samples),
                                            "--eval-seed", str(a.seed)])
    T.seed_everything(0)
    tok, model, _ = T.build_models(args, dev)
    eos = sorted({i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None})
    ev = T.evaluate(model, tok, dev, args, eos)
    speeds = ev["eval_speeds"]
    parsed = np.array([s for s in speeds if s is not None], dtype=float)
    n = len(speeds)
    print(json.dumps({"device": T.device_name(dev), "n": n, "parse_fail_rate": ev["eval_parse_fail_rate"],
                      "percentiles": {str(q): float(np.percentile(np.abs(parsed), q)) for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)}}))
    # groups of 4 in sampling order within each prompt (K consecutive samples per prompt)
    groups = [speeds[i:i + 4] for i in range(0, n - n % 4, 4)]
    rows = []
    for v in [float(x) for x in a.candidates.split(",")]:
        viol = np.array([s is None or abs(s) > v for s in speeds])
        p = float(viol.mean())
        allv = float(np.mean([all(s is None or abs(s) > v for s in g) for g in groups]))
        rows.append({"v_max": v, "p_violation": p, "p4_estimate": p ** 4, "all_violating_groups_exact": allv})
        print(json.dumps(rows[-1]))
    ok = [r for r in rows if r["all_violating_groups_exact"] >= 0.8]
    print(json.dumps({"largest_v_max_with_all_violating_ge_0.8": max(r["v_max"] for r in ok) if ok else None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
