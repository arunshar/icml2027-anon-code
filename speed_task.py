"""The speed task, kept free of torch so tests can import it cheaply.

PROMPTS and parse_speed are unchanged from the script this harness grew out of, so the task
itself is fixed and only the training loop around it is new.
"""
from __future__ import annotations

import re
from typing import Optional

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "You pilot a vessel near a harbor speed limit. Choose a cruising speed in knots and reply with ONLY the integer.",
    "Pick a safe cruising speed in knots for a vessel in a regulated zone. Answer with just the number.",
    "Output a single integer: the cruising speed in knots you would set for this vessel.",
    "A vessel needs a cruising speed in knots. Reply with only one integer.",
    "Choose the vessel's cruising speed in knots. Respond with the integer only.",
    "Set a cruising speed in knots for the vessel and answer with just that integer.",
]


def parse_speed(text: str) -> Optional[float]:
    """First integer in the completion, or None. run_grpo_local.py:42-44."""
    m = re.search(r"-?\d+", text)
    return float(m.group()) if m else None


# ---------------------------------------------------------------- task protocol
# The speed task expressed through the shared protocol. Nothing below changes any
# behaviour: the four scoring functions are the frozen ones in rewards.py, re-exported
# rather than moved, so their definitions keep their provenance citations in one place.
# The import direction is tasks -> rewards, never the reverse.

import numpy as _np  # noqa: E402

import rewards as _RW  # noqa: E402
from tasks import TaskSpec as _TaskSpec, register as _register  # noqa: E402


def _verdict(s, v_max):
    """Feasible when the speed is inside the envelope. An unparseable completion is not."""
    return not (s is None or abs(s) > v_max)


def _residual(s, v_max):
    return 0.0 if s is None else float(_RW.residual(_np.float64(s), v_max))


def _task_term(s, v_max):
    return float(_RW.task(_np.float64(s), v_max))


def _synthesize(r, v_max):
    """The speed whose residual is r. rewards.group_rewards used this inline to price an
    unparseable member at the group's worst residual."""
    return v_max * (1.0 + r)


class _SpeedSpec(_TaskSpec):
    def summary_stat(self, parsed):
        a = _np.asarray([p for p in parsed if p is not None], dtype=_np.float64)
        return float(a.mean()) if a.size else None

    def output_table(self, limit, instance=0):
        """Every speed magnitude a six-token completion can parse to. See rewards.SPEED_ENUM_MAX."""
        return _RW.speed_output_table(limit)


SPEC = _register(_SpeedSpec(
    name="speed",
    prompts=PROMPTS,
    parse=parse_speed,
    verdict=_verdict,
    residual=_residual,
    task_term=_task_term,
    synthesize=_synthesize,
    unparseable_task=_RW.UNPARSEABLE_TASK,
    items_key="speeds",
    scalar_key="mean_speed",
    eval_scalar_key="eval_mean_speed",
    eval_items_key="eval_speeds",
    limit_help="speed envelope v_max",
))
