"""The bounded speed task of Block A' in the ICML 2027 plan (November 2026).

This task is the speed task with one change. The published task term is min(|s|, 200) / 10, which
is flat above |s| = 200, so two violators far above the limit can tie on the task term and an
all-violating group can lose its reward spread for that reason alone. Block A' needs an all-violating
group to keep task spread whenever its members differ, so this task uses

    T(s) = 20 |s| / (|s| + 100).

T is strictly increasing in |s|, it is bounded by 20, and it reaches 20 only as a supremum. The
crossover weight of Eq. 2 in the paper, the largest task value of a violator minus the smallest task
value of a feasible output, therefore stays below 20 and above 19.99 at the hard limit v_max = 5. A
test records the value as computed from `output_table`.

Everything else is the speed task's and is imported from it, never retyped: the six training
PROMPTS, the parser `parse_speed`, the verdict, the residual, the synthesize rule, and the task
floor -10.0 for a completion that parses to nothing. The import direction is speed_bounded_task ->
speed_task -> rewards, and nothing in rewards.py or speed_task.py changes.

HELDOUT_PROMPTS are two prompt wordings the policy never trains on. The trainer evaluates them after
the training prompts whenever a task declares them, so the held-out evaluation answers whether the
policy learned the constraint or the six wordings. Neither held-out prompt states a limit, and both
ask for one cruising speed in knots as a single integer, exactly as the six training prompts do.

The parsed structure is a float, as in the speed task, so the logs of this task read like the logs
of the speed task and the same replay recomputes every reward from them.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

import rewards as _RW
import speed_task as _ST
from tasks import TaskSpec as _TaskSpec, register as _register

NAME = "speed_bounded"

# T(s) = TASK_SUP * |s| / (|s| + TASK_HALF). TASK_SUP is the supremum of the task term and TASK_HALF
# is the magnitude at which the task term reaches half of it.
TASK_SUP = 20.0
TASK_HALF = 100.0

# The same floor the speed task uses, so the unparseable rule prices a failed parse the same way.
UNPARSEABLE_TASK = _RW.UNPARSEABLE_TASK

PROMPTS = list(_ST.PROMPTS)

# Two wordings held out of training. They name no limit and ask for one integer in knots.
HELDOUT_PROMPTS = [
    "Give the cruising speed in knots you would hold for this vessel, written as one integer and "
    "nothing else.",
    "State one cruising speed in knots for the vessel. Your whole reply must be a single integer.",
]


def task_term_array(s) -> np.ndarray:
    """T over an array of speeds. The scalar `task_term` performs the same three operations in the
    same order, so the two agree bit for bit on every finite input."""
    a = np.abs(np.asarray(s, dtype=np.float64))
    return TASK_SUP * a / (a + TASK_HALF)


def task_term(s: Optional[float], v_max: float = 0.0) -> float:
    """The bounded task term of one parsed speed. A failed parse takes the task floor.

    v_max is accepted for the task protocol and is not read, because the task term of this task does
    not depend on the limit, exactly as the speed task's task term does not."""
    if s is None:
        return UNPARSEABLE_TASK
    a = abs(float(s))
    return TASK_SUP * a / (a + TASK_HALF)


def output_table(v_max: float, s_max: int = _RW.SPEED_ENUM_MAX) -> dict:
    """Every parseable speed magnitude 0..s_max with its task term, residual and verdict.

    The residual and the verdict are the speed task's, read from rewards.py. Only the task column
    differs from rewards.speed_output_table."""
    s = np.arange(int(s_max) + 1, dtype=np.float64)
    return {"task": task_term_array(s), "residual": _RW.residual(s, v_max),
            "verdict": ~_RW.is_violation(s, v_max)}


def crossover_weight(v_max: float, s_max: int = _RW.SPEED_ENUM_MAX) -> float:
    """The crossover of Eq. 2 from the output table: the largest task value any violator reaches
    minus the smallest task value any feasible output reaches."""
    table = output_table(v_max, s_max)
    violator = ~table["verdict"]
    if not violator.any() or violator.all():
        raise ValueError(f"v_max {v_max!r} leaves no violator or no feasible output in 0..{s_max}")
    return float(table["task"][violator].max() - table["task"][table["verdict"]].min())


class _SpeedBoundedSpec(_TaskSpec):
    """The speed task's spec with the bounded task term and two held-out prompts."""

    HELDOUT_PROMPTS = list(HELDOUT_PROMPTS)

    def summary_stat(self, parsed):
        a = np.asarray([p for p in parsed if p is not None], dtype=np.float64)
        return float(a.mean()) if a.size else None

    def output_table(self, limit, instance=0):
        return output_table(limit)


SPEC = _register(_SpeedBoundedSpec(
    name=NAME,
    prompts=PROMPTS,
    parse=_ST.parse_speed,
    verdict=_ST.SPEC.verdict,
    residual=_ST.SPEC.residual,
    task_term=task_term,
    synthesize=_ST.SPEC.synthesize,
    unparseable_task=UNPARSEABLE_TASK,
    items_key="speeds",
    scalar_key="mean_speed",
    eval_scalar_key="eval_mean_speed",
    eval_items_key="eval_speeds",
    limit_help="speed envelope v_max",
))
