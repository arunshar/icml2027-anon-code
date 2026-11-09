#!/usr/bin/env python3
"""Generate the capacity-task instances ONCE, offline, and print them as literals.

They must be literals in the task module, never sampled at run time: the eval's identity is
the number and order of generate calls made after the eval seed is set, so drawing instances
from a random stream during training would shift the global RNG and void the matched-start
check the ledger records. This script exists so the literals are reproducible, not so they
are produced on the fly.

Usage: python make_capacity_instances.py --seed 20260912 --n-items 16 --prompts 6
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def dp_optimum(values, weights, capacity):
    """Exact 0/1 knapsack optimum by dynamic programming over integer capacity."""
    best = [0] * (capacity + 1)
    for v, w in zip(values, weights):
        for c in range(capacity, w - 1, -1):
            cand = best[c - w] + v
            if cand > best[c]:
                best[c] = cand
    return best[capacity]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--n-items", type=int, default=16)
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--v-lo", type=int, default=1)
    ap.add_argument("--v-hi", type=int, default=20)
    ap.add_argument("--w-lo", type=int, default=1)
    ap.add_argument("--w-hi", type=int, default=20)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = []
    for _ in range(a.prompts):
        v = rng.integers(a.v_lo, a.v_hi + 1, size=a.n_items).tolist()
        w = rng.integers(a.w_lo, a.w_hi + 1, size=a.n_items).tolist()
        out.append({"values": v, "weights": w, "total_weight": int(sum(w))})
    print(json.dumps({"seed": a.seed, "n_items": a.n_items, "instances": out}, indent=2))
    # capacity is set per hardness level as a ratio of total weight, so report the optima
    # at a few ratios to show the task is non-trivial at each.
    for ratio in (0.25, 0.5, 0.75):
        line = []
        for inst in out:
            cap = max(1, int(round(ratio * inst["total_weight"])))
            line.append(dp_optimum(inst["values"], inst["weights"], cap))
        print(f"# ratio {ratio}: DP optima {line}")


if __name__ == "__main__":
    main()
