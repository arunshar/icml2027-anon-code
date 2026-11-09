"""Reward functions and the group-relative advantage for the cancellation experiment.

Every function here is a copy of code that already produced a saved result, so the
trainer and the analysis agree with the provenance numbers by construction.

Sources (repo-relative):
  task, pen_graded, pen_binary : provenance/objection_check.py lines 5-7
  advantage                    : provenance/run_groupnorm.py lines 36-39
The only change is that v_max is an explicit argument instead of a module constant.
"""
from __future__ import annotations

import functools
from typing import Optional, Sequence

import numpy as np

ARMS = ("graded", "binary", "none")

# The task value a completion gets when no integer can be parsed from it.
# run_grpo_local.py line 47 uses -10.0 and counts the completion as a violation.
UNPARSEABLE_TASK = -10.0


def task(s, v_max: float = 30.0):
    """Rater term: rewards higher speed, capped at 200. objection_check.py:5."""
    return np.minimum(np.abs(s), 200) / 10.0


def pen_graded(s, v_max: float = 30.0):
    """Graded residual penalty above the envelope. objection_check.py:6."""
    return -100.0 * np.maximum(np.abs(s) / v_max - 1.0, 0.0)


def pen_binary(s, v_max: float = 30.0):
    """Binary verdict: 0 if feasible, -1 if not. objection_check.py:7."""
    return np.where(np.abs(s) <= v_max, 0.0, -1.0)


def residual(s, v_max: float = 30.0):
    """Distance above the envelope in units of v_max. Zero when feasible."""
    return np.maximum(np.abs(s) / v_max - 1.0, 0.0)


def is_violation(s, v_max: float = 30.0):
    return np.abs(s) > v_max


def advantage(rewards: Sequence[float]) -> np.ndarray:
    """Group-relative advantage, matching run_grpo_local.py:96 under torch semantics.

    run_groupnorm.py:36-39, verbatim. Sample standard deviation (ddof=1) plus 1e-6.
    """
    r = np.asarray(rewards, dtype=np.float64)
    return (r - r.mean()) / (r.std(ddof=1) + 1e-6)


def penalty(s, arm: str, v_max: float):
    if arm == "graded":
        return pen_graded(s, v_max)
    if arm == "binary":
        return pen_binary(s, v_max)
    if arm == "none":
        return np.zeros_like(np.asarray(s, dtype=np.float64))
    raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")


def arm_reward(s, arm: str, w: float, v_max: float):
    """R = task(s) + w * penalty(s) for a parsed speed s (scalar or array)."""
    s = np.asarray(s, dtype=np.float64)
    return task(s, v_max) + w * penalty(s, arm, v_max)


def gate_reward(s, v_max: float):
    """Counterfactual only: multiplicative gate, task * 1[feasible]. Analysis, not an arm."""
    s = np.asarray(s, dtype=np.float64)
    return task(s, v_max) * (~is_violation(s, v_max)).astype(np.float64)


def group_rewards(speeds: Sequence[Optional[float]], arm: str, w: float, v_max: float,
                  *, spec=None, w_task: float = 1.0) -> np.ndarray:
    """Rewards for one group of K parsed speeds, applying the unparseable rule.

    An unparseable completion (None) takes the task value UNPARSEABLE_TASK and the arm's
    penalty evaluated at the largest residual present in the group. If no member parsed,
    that residual is taken as 1.0 (one full envelope above the limit). The rule is the
    same for every arm and is stated in the paper.
    """
    if spec is None and w_task != 1.0:
        raise ValueError("w_task is only supported on the task-spec path; the speed task is frozen")
    if spec is None:
        parsed = np.array([s for s in speeds if s is not None], dtype=np.float64)
        if parsed.size:
            r_max = float(residual(parsed, v_max).max())
            s_at_rmax = v_max * (1.0 + r_max)
        else:
            r_max, s_at_rmax = 1.0, 2.0 * v_max
        out = np.empty(len(speeds), dtype=np.float64)
        for i, s in enumerate(speeds):
            if s is None:
                out[i] = UNPARSEABLE_TASK + w * float(penalty(s_at_rmax, arm, v_max))
            else:
                out[i] = float(arm_reward(s, arm, w, v_max))
        return out

    # Task-supplied path. Same rule, expressed through the protocol: an unparseable member
    # takes the task floor plus this arm's penalty evaluated at the group's worst residual.
    present = [x for x in speeds if x is not None]
    r_max = max((spec.residual(x, v_max) for x in present), default=1.0)
    at_rmax = spec.synthesize(r_max, v_max)
    out = np.empty(len(speeds), dtype=np.float64)
    for i, x in enumerate(speeds):
        if x is None:
            out[i] = w_task * spec.UNPARSEABLE_TASK + w * _spec_penalty(at_rmax, arm, v_max, spec)
        else:
            out[i] = w_task * spec.task_term(x, v_max) + w * _spec_penalty(x, arm, v_max, spec)
    return out


def _spec_penalty(x, arm: str, v_max: float, spec) -> float:
    """The arm's penalty for a task-supplied structure. Same three arms, same shapes:
    a graded residual, a binary verdict, or nothing."""
    if arm == "graded":
        return -100.0 * float(spec.residual(x, v_max))
    if arm == "binary":
        return 0.0 if spec.verdict(x, v_max) else -1.0
    if arm == "none":
        return 0.0
    raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")


def group_is_all_violating(speeds: Sequence[Optional[float]], v_max: float, *, spec=None) -> bool:
    if spec is None:
        return all(s is None or abs(s) > v_max for s in speeds)
    return all(not spec.verdict(x, v_max) for x in speeds)


def group_has_violator(speeds: Sequence[Optional[float]], v_max: float, *, spec=None) -> bool:
    if spec is None:
        return any(s is None or abs(s) > v_max for s in speeds)
    return any(not spec.verdict(x, v_max) for x in speeds)


# ---------------------------------------------------------------- remedy arms
# Everything below is new with the remedy arms and has produced no saved result yet. Nothing above
# calls it, and `advantage()` stays exactly as it was, so a run at the default flags cannot reach
# this code except through the "grpo" variant, which returns `advantage()` itself.

ADV_VARIANTS = ("grpo", "drgrpo", "ngrpo_feasible", "ngrpo_sup")
GROUP_FILTERS = ("none", "verdict_mixed", "reward_spread")
ANCHORED_VARIANTS = ("ngrpo_feasible", "ngrpo_sup")

# ICML 2027 (November 2026). Variants that read the group's verdict vector as well as its rewards.
# They are kept out of ADV_VARIANTS on purpose: analyze.py enumerates ADV_VARIANTS through
# `advantage_variant(rewards, v, anchor)`, which has no verdicts to give, and the frozen replay of the
# published runs must keep working unchanged. The trainer accepts every name in TRAINER_ADV_VARIANTS.
VERDICT_VARIANTS = ("grpo_fixedref",)
TRAINER_ADV_VARIANTS = ADV_VARIANTS + VERDICT_VARIANTS

# A group whose largest absolute advantage is at or below this contributes a zero policy gradient.
# The trainer's --skip-zero-adv reads it, and the analysis can count the same groups from the logs.
ZERO_ADV_TOL = 1e-12

# A group whose reward range is at most this is treated as having no spread. analyze.py reads the
# same object, so the filter and the zero-variance counter cannot disagree about a group.
ZERO_SPREAD_TOL = 1e-12

# The largest speed magnitude a six-token completion can parse to is 999,999. Every speed reward is
# a function of |s|, the task term is flat above 200, and every penalty is non-increasing in |s|, so
# the maximum over this range is the maximum over every integer. A test extends the range to check.
SPEED_ENUM_MAX = 999_999


def advantage_variant(rewards: Sequence[float], variant: str = "grpo",
                      anchor: Optional[float] = None) -> np.ndarray:
    """The advantage under one of the remedy variants.

    grpo            `advantage()` itself, so the default is the frozen path by construction.
    drgrpo          R - mean(R). Dr. GRPO drops the standard-deviation divisor
                    (arXiv 2503.20783, Sec. 3.1). The mean stays, so a group-constant verdict still
                    cancels.
    ngrpo_feasible  NGRPO's virtual sample (arXiv 2509.18851, Sec. 4.1, Eq. 6). The anchor joins the
    ngrpo_sup       K real rewards, the mean and the population standard deviation (divisor K+1,
                    ddof=0) are taken over those K+1 values, and only the K real advantages are
                    returned. The two variants differ only in the anchor, see `reward_anchor`.
    """
    if variant == "grpo":
        if anchor is not None:
            raise ValueError("the grpo variant takes no anchor")
        return advantage(rewards)
    if variant in VERDICT_VARIANTS:
        raise ValueError(f"the {variant} variant needs verdicts; call advantage_fixedref(rewards, verdicts, c)")
    r = np.asarray(rewards, dtype=np.float64)
    if variant == "drgrpo":
        if anchor is not None:
            raise ValueError("the drgrpo variant takes no anchor")
        return r - r.mean()
    if variant in ANCHORED_VARIANTS:
        if anchor is None:
            raise ValueError(f"the {variant} variant needs an anchor")
        # NGRPO Eq. 6: mean and std over the G+1 values, std with divisor G+1.
        z = np.append(r, np.float64(anchor))
        return (r - z.mean()) / (z.std(ddof=0) + 1e-6)
    raise ValueError(f"unknown advantage variant {variant!r}, expected one of {ADV_VARIANTS}")


def advantage_fixedref(rewards: Sequence[float], verdicts: Sequence[bool], c: float) -> np.ndarray:
    """The frozen z-score with a fixed reference for the verifier component in homogeneous-verdict groups.

    z = advantage(rewards) is computed first and is never changed on a mixed group. When every member
    fails the verifier (all-violating) every advantage is z - c, and when every member passes
    (all-passing) every advantage is z + c. A group with at least one pass and at least one fail
    returns z unchanged, so this variant and the plain estimator agree on every mixed group.

    In expectation a group-constant shift of the advantage is a baseline shift of the score-function
    estimator, so it contributes zero expected gradient. Any training effect is therefore a
    finite-sample effect of pushing down every sampled completion of an all-violating group, or up
    every sampled completion of an all-passing group. This is the fixed-reference idea of Nie et al.
    2026 (arXiv 2605.07689) and RL-ZVP (arXiv 2509.21880) applied to the verifier component only,
    under a mixed reward.
    """
    v = [bool(x) for x in verdicts]
    n = len(np.asarray(rewards, dtype=np.float64))
    if not v or n == 0:
        raise ValueError("an empty group has no advantage")
    if len(v) != n:
        raise ValueError(f"{len(v)} verdicts for {n} rewards")
    z = advantage(rewards)
    c = float(c)
    if all(not x for x in v):
        return z - c
    if all(v):
        return z + c
    return z


def group_adv_is_zero(adv) -> bool:
    """True when the largest absolute advantage in the group is at or below ZERO_ADV_TOL."""
    a = np.abs(np.asarray(adv, dtype=np.float64))
    return bool(a.size == 0 or a.max() <= ZERO_ADV_TOL)


def group_verdicts(items: Sequence, v_max: float, *, spec=None) -> list:
    """Per member, True when it satisfies the constraint. An unparseable member does not."""
    if spec is None:
        return [not (s is None or abs(s) > v_max) for s in items]
    return [bool(spec.verdict(x, v_max)) for x in items]


def keep_group(group_filter: str, rewards: Sequence[float], verdicts: Sequence[bool]) -> bool:
    """Whether a group enters the loss under a filter.

    none           every group.
    verdict_mixed  DAPO's accuracy filter, drop-only (arXiv 2503.14476, Sec. 3.2, Eq. 11): keep a
                   group when at least one member passes the verifier and at least one fails. The
                   trainer does not over-sample to refill the batch.
    reward_spread  keep a group whose reward range exceeds ZERO_SPREAD_TOL, which is a filter on
                   identical rewards.
    """
    if group_filter == "none":
        return True
    if group_filter == "verdict_mixed":
        n_feasible = sum(1 for v in verdicts if v)
        return 0 < n_feasible < len(verdicts)
    if group_filter == "reward_spread":
        return bool(np.ptp(np.asarray(rewards, dtype=np.float64)) > ZERO_SPREAD_TOL)
    raise ValueError(f"unknown group filter {group_filter!r}, expected one of {GROUP_FILTERS}")


def speed_output_table(v_max: float, s_max: int = SPEED_ENUM_MAX) -> dict:
    """Every parseable speed magnitude with its task term, residual and verdict, as arrays."""
    s = np.arange(int(s_max) + 1, dtype=np.float64)
    return {"task": task(s, v_max), "residual": residual(s, v_max),
            "verdict": ~is_violation(s, v_max)}


def _table_penalty(table: dict, arm: str) -> np.ndarray:
    """The arm's penalty over an output table, in the shapes `penalty` and `_spec_penalty` use."""
    if arm == "graded":
        return -100.0 * table["residual"]
    if arm == "binary":
        return np.where(table["verdict"], 0.0, -1.0)
    if arm == "none":
        return np.zeros_like(table["residual"])
    raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")


def _output_table(v_max: float, spec, instance: int) -> dict:
    if spec is None:
        return speed_output_table(v_max)
    return spec.output_table(v_max, instance)


@functools.lru_cache(maxsize=None)
def _anchor_cached(variant: str, arm: str, w: float, v_max: float, spec, w_task: float,
                   instance: int) -> float:
    table = _output_table(v_max, spec, instance)
    if variant == "ngrpo_feasible":
        feasible = table["task"][table["verdict"]]
        if feasible.size == 0:
            raise ValueError("the output space has no feasible element, so there is no anchor")
        return float(w_task * feasible.max())
    # ngrpo_sup: the design's own largest reward over its whole output space. An unparseable
    # member scores the task floor plus a penalty that is never positive, so its best case is the
    # floor itself, and it is included so the maximum is over every reward the design can give.
    floor = UNPARSEABLE_TASK if spec is None else spec.UNPARSEABLE_TASK
    rewards = w_task * table["task"] + w * _table_penalty(table, arm)
    return float(max(rewards.max(), w_task * floor))


def reward_anchor(variant: str, arm: str, w: float, v_max: float, *, spec=None,
                  w_task: float = 1.0, instance: int = 0) -> Optional[float]:
    """The virtual sample an anchored variant appends, or None for a variant without one.

    ngrpo_feasible  w_task times the best task value any feasible output reaches. This is the
                    reward of the best feasible answer under every penalty arm, because each
                    penalty is zero on the feasible set.
    ngrpo_sup       the arm's own maximum reward over its output space, found by exhaustive
                    enumeration. When that maximum is infeasible the verdict shifts it exactly as it
                    shifts an all-violating group, so the verdict cancels again.

    The enumeration is cached per design and consumes no randomness. The capacity task's anchors
    depend on the instance, which is the prompt index.
    """
    if variant not in TRAINER_ADV_VARIANTS:
        raise ValueError(f"unknown advantage variant {variant!r}, expected one of {TRAINER_ADV_VARIANTS}")
    if variant not in ANCHORED_VARIANTS:
        return None
    if spec is None and w_task != 1.0:
        raise ValueError("w_task is only supported on the task-spec path; the speed task is frozen")
    return _anchor_cached(variant, arm, float(w), float(v_max), spec, float(w_task), int(instance))
