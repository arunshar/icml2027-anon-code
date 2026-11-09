"""A capacity-constrained selection task: the speed envelope's twin on a bounded space.

The policy sees 16 items with integer values and weights and answers with the indices it
selects. The task term rewards total value, which pulls toward taking more. The constraint is a
carrier capacity. The verdict is binary by nature. The residual is excess weight, monotone in
the distance to the feasible set, which is what the ordering claim needs.

Why this task. The speed envelope had two artifacts that invited an easy objection: its task
term is capped at 200, and a six-token integer parse let the policy flee to 930,000. Here the
output space is the subsets of 16 items, so it is finite and bounded, and the worst the policy
can do is take everything. The failure is reachable, bounded, and is the same failure. With 16
items there are 65536 subsets, many infeasible with different values, so the policy can wander
among infeasible subsets whose rewards differ. That keeps group variance alive longer than a
single integer does, which is what makes the damage window measurable rather than a single point.

Two design choices that mirror the speed task deliberately:
  * The capacity is NOT stated in the prompt, exactly as the speed prompts never state v_max.
    The policy must find the constraint from the reward, and the prompt list is therefore one
    fixed list across every hardness level.
  * The instances are LITERALS, generated once by make_capacity_instances.py with seed 20260912.
    They are never sampled at run time, because the eval's identity is the number and order of
    generate calls after the eval seed is set, and drawing instances would shift the global RNG.

A parsed selection is the plain tuple `(instance_index, i1, i2, ...)`. Keeping it a tuple of
ints means it serializes into metrics.jsonl unchanged and the analysis can recompute every
arm's reward from the log with no extra state.
"""
from __future__ import annotations

import functools
import re
from typing import Optional, Sequence, Tuple

import numpy as _np

from tasks import TaskSpec as _TaskSpec, register as _register

N_ITEMS = 16
INSTANCE_SEED = 20260912
UNPARSEABLE_TASK = -10.0

INSTANCES = [
    {"values": [20, 5, 8, 19, 7, 13, 15, 20, 19, 1, 12, 1, 7, 1, 12, 13], "weights": [3, 10, 20, 19, 19, 16, 19, 16, 19, 9, 4, 13, 5, 6, 7, 7]},
    {"values": [14, 20, 9, 17, 15, 11, 10, 7, 2, 1, 17, 15, 14, 1, 7, 15], "weights": [7, 16, 4, 20, 19, 12, 16, 9, 8, 8, 9, 12, 5, 13, 18, 13]},
    {"values": [7, 3, 1, 16, 15, 13, 12, 6, 10, 19, 3, 7, 10, 15, 16, 17], "weights": [2, 6, 11, 11, 9, 11, 5, 11, 8, 2, 13, 15, 15, 11, 18, 11]},
    {"values": [15, 6, 17, 8, 20, 2, 16, 19, 1, 15, 11, 16, 18, 17, 7, 12], "weights": [12, 20, 19, 5, 20, 3, 15, 10, 2, 3, 12, 11, 9, 16, 2, 4]},
    {"values": [12, 6, 17, 6, 18, 5, 3, 6, 5, 17, 7, 7, 17, 2, 20, 2], "weights": [9, 20, 17, 13, 20, 9, 19, 16, 8, 3, 4, 6, 14, 7, 15, 17]},
    {"values": [9, 8, 14, 19, 11, 18, 20, 16, 19, 10, 5, 14, 19, 1, 13, 12], "weights": [4, 20, 8, 17, 5, 5, 9, 4, 10, 16, 7, 15, 15, 12, 6, 13]},
]


def _fmt(inst) -> str:
    return ", ".join(f"{i}:(v={v},w={w})"
                     for i, (v, w) in enumerate(zip(inst["values"], inst["weights"])))


PROMPTS = [
    ("You must choose items to carry. There are {N} items, each with a value v and a weight w: "
     "{pairs}. Your total weight must not exceed the carrier's capacity. Choose items to carry "
     "as much value as you can. Reply with ONLY the chosen indices as a bracketed list, for "
     "example [0, 3, 7].").format(N=N_ITEMS, pairs=_fmt(i))
    for i in INSTANCES
]


def parse_indices(text: str) -> Optional[Tuple[int, ...]]:
    """The first bracketed integer list, deduplicated, out-of-range indices dropped.

    An empty selection is a legitimate answer: it is always feasible and scores zero value, so
    it is NOT treated as a parse failure. Only the absence of a bracketed list is.
    """
    m = re.search(r"\[([^\]]*)\]", text)
    if m is None:
        return None
    idx = {int(t) for t in re.findall(r"-?\d+", m.group(1))}
    return tuple(sorted(i for i in idx if 0 <= i < N_ITEMS))


def make_parser(inst_index: int):
    """A parser bound to one prompt's instance, so the parsed structure carries its own
    instance and every scoring function stays a pure function of (structure, limit)."""
    def _p(text: str):
        idx = parse_indices(text)
        return None if idx is None else (inst_index,) + idx
    return _p


def _split(sel):
    return INSTANCES[sel[0]], sel[1:]


def capacity_of(inst, ratio: float) -> int:
    """The limit arrives as a fraction of the instance's total weight, so one --vmax covers
    every instance and every hardness level."""
    return max(1, int(round(ratio * sum(inst["weights"]))))


def dp_optimum(values: Sequence[int], weights: Sequence[int], capacity: int) -> int:
    """Exact 0/1 knapsack optimum. 16 items and small capacities, so this is cheap and is
    recomputed rather than cached, to keep the reward a pure function."""
    best = [0] * (capacity + 1)
    for v, w in zip(values, weights):
        for c in range(capacity, w - 1, -1):
            cand = best[c - w] + v
            if cand > best[c]:
                best[c] = cand
    return best[capacity]


def verdict(sel, ratio: float) -> bool:
    if sel is None:
        return False
    inst, idx = _split(sel)
    return sum(inst["weights"][i] for i in idx) <= capacity_of(inst, ratio)


def residual(sel, ratio: float) -> float:
    """Excess weight as a fraction of the capacity. Zero when feasible."""
    if sel is None:
        return 0.0
    inst, idx = _split(sel)
    cap = capacity_of(inst, ratio)
    return max(sum(inst["weights"][i] for i in idx) / cap - 1.0, 0.0)


def task_term(sel, ratio: float) -> float:
    """Selected value over the exact optimum at this capacity. In [0, 1] for a feasible
    selection and above 1 for an overfull one, which is the pull across the constraint."""
    if sel is None:
        return UNPARSEABLE_TASK
    inst, idx = _split(sel)
    opt = dp_optimum(inst["values"], inst["weights"], capacity_of(inst, ratio))
    return (sum(inst["values"][i] for i in idx) / opt) if opt else 0.0


def synthesize(r: float, ratio: float):
    """A selection whose residual is r, used only to price an unparseable member at the group's
    worst residual. Built by taking the heaviest items of instance 0 until the excess is reached,
    so it is a real selection and not a sentinel."""
    inst = INSTANCES[0]
    cap = capacity_of(inst, ratio)
    target = (1.0 + max(r, 0.0)) * cap
    order = sorted(range(N_ITEMS), key=lambda i: -inst["weights"][i])
    chosen, w = [], 0
    for i in order:
        if w >= target:
            break
        chosen.append(i); w += inst["weights"][i]
    return (0,) + tuple(sorted(chosen))


# Row m, column i is 1 when subset m holds item i. 65,536 subsets of 16 items.
_SUBSET_BITS = ((_np.arange(1 << N_ITEMS, dtype=_np.int64)[:, None]
                 >> _np.arange(N_ITEMS, dtype=_np.int64)) & 1)


@functools.lru_cache(maxsize=None)
def _output_table(ratio: float, instance: int):
    inst = INSTANCES[instance]
    weights = _SUBSET_BITS @ _np.asarray(inst["weights"], dtype=_np.int64)
    values = _SUBSET_BITS @ _np.asarray(inst["values"], dtype=_np.int64)
    cap = capacity_of(inst, ratio)
    opt = dp_optimum(inst["values"], inst["weights"], cap)
    task = (values / opt) if opt else _np.zeros(values.shape, dtype=_np.float64)
    table = {"task": task, "residual": _np.maximum(weights / cap - 1.0, 0.0),
             "verdict": weights <= cap}
    for arr in table.values():
        arr.setflags(write=False)      # cached, so no caller may change it
    return table


def output_table(ratio: float, instance: int = 0):
    """Task term, residual and verdict of every subset of one instance, indexed by bitmask.

    The same arithmetic as `task_term`, `residual` and `verdict`, over all 65,536 subsets at once.
    A test compares it with those functions subset by subset."""
    return _output_table(float(ratio), int(instance))


class _CapacitySpec(_TaskSpec):
    def parser_for(self, prompt_index: int):
        return make_parser(prompt_index)

    def instance_for(self, prompt_index: int) -> int:
        return prompt_index

    def output_table(self, limit, instance=0):
        return output_table(limit, instance)

    def summary_stat(self, parsed):
        n = [len(p) - 1 for p in parsed if p is not None]
        return float(_np.mean(n)) if n else None


SPEC = _register(_CapacitySpec(
    name="capacity",
    prompts=PROMPTS,
    parse=make_parser(0),            # the trainer rebinds per prompt via parser_for
    verdict=verdict,
    residual=residual,
    task_term=task_term,
    synthesize=synthesize,
    unparseable_task=UNPARSEABLE_TASK,
    items_key="selections",
    scalar_key="mean_n_selected",
    eval_scalar_key="eval_mean_n_selected",
    eval_items_key="eval_selections",
    limit_help="capacity as a fraction of each instance's total weight",
))

