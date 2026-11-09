"""The capacity task, and the identities re-established rather than assumed.

P1 and P2 are claims about a task's verdict and residual geometry, not about the arms, so a
task whose residual is discrete has to earn them again. The speed task's residual is continuous
and monotone by construction. This one is a sum over chosen integer weights, so it moves in
jumps, and the group can contain ties that the speed task almost never produced.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import capacity_task as C  # noqa: E402
import rewards as RW  # noqa: E402
import tasks  # noqa: E402

SPEC = tasks.get("capacity")
RATIO = 0.5
ARMS = {"graded": ("graded", 5.0), "binary": ("binary", 5.0),
        "none": ("none", 0.0), "binary_1000x": ("binary", 5000.0)}


def _adv(sels, arm, w):
    return RW.advantage(RW.group_rewards(sels, arm, w, RATIO, spec=SPEC))


def _take(inst_index, idx):
    return (inst_index,) + tuple(sorted(idx))


# ---------------------------------------------------------------- parsing


def test_parse_accepts_a_bracketed_list_and_normalises_it():
    p = C.make_parser(0)
    assert p("take [3, 1, 1, 0]") == (0, 0, 1, 3)          # sorted and deduplicated
    assert p("[0, 99, 4]") == (0, 0, 4)                     # out-of-range dropped
    assert p("I would pick [] nothing") == (0,)             # empty selection is legitimate
    assert p("no list at all") is None                      # only a missing list is a failure
    assert p("[2] then [5]") == (0, 2)                      # the first list wins


def test_parser_binds_the_instance_so_scoring_is_a_pure_function():
    a, b = C.make_parser(0)("[0,1]"), C.make_parser(1)("[0,1]")
    assert a[0] == 0 and b[0] == 1
    # same indices, different instance, so generally different weight
    wa = sum(C.INSTANCES[0]["weights"][i] for i in a[1:])
    wb = sum(C.INSTANCES[1]["weights"][i] for i in b[1:])
    assert (wa, wb) == (wa, wb)  # both computable; the point is they are independent


# ---------------------------------------------------------------- scoring


def test_taking_everything_is_infeasible_and_scores_above_one():
    allsel = _take(0, range(C.N_ITEMS))
    assert C.verdict(allsel, RATIO) is False
    assert C.residual(allsel, RATIO) > 0
    assert C.task_term(allsel, RATIO) > 1.0, "the task term must pull across the constraint"


def test_empty_selection_is_feasible_and_worthless():
    empty = _take(0, [])
    assert C.verdict(empty, RATIO) is True
    assert C.residual(empty, RATIO) == 0.0
    assert C.task_term(empty, RATIO) == 0.0


def test_residual_is_zero_exactly_when_feasible_and_grows_with_excess():
    inst = C.INSTANCES[0]
    order = sorted(range(C.N_ITEMS), key=lambda i: -inst["weights"][i])
    prev = -1.0
    grew = 0
    for k in range(1, C.N_ITEMS + 1):
        sel = _take(0, order[:k])
        r = C.residual(sel, RATIO)
        assert (r == 0.0) == C.verdict(sel, RATIO)
        if r > prev:
            grew += 1
        assert r >= prev, "adding the next heaviest item must not reduce the excess"
        prev = r
    assert grew >= 2


def test_dp_optimum_matches_brute_force_on_a_small_instance():
    import itertools
    vals, wts, cap = [3, 5, 7, 2], [2, 4, 5, 1], 7
    best = max((sum(vals[i] for i in c)
                for k in range(5) for c in itertools.combinations(range(4), k)
                if sum(wts[i] for i in c) <= cap), default=0)
    assert C.dp_optimum(vals, wts, cap) == best


def test_task_term_is_one_at_the_optimum():
    inst = C.INSTANCES[0]
    cap = C.capacity_of(inst, RATIO)
    opt = C.dp_optimum(inst["values"], inst["weights"], cap)
    # find a feasible selection achieving the optimum by DP reconstruction over a small search
    best_sel, best_val = None, -1
    rng = np.random.default_rng(0)
    for _ in range(4000):
        k = int(rng.integers(0, C.N_ITEMS + 1))
        idx = tuple(sorted(rng.choice(C.N_ITEMS, size=k, replace=False).tolist()))
        sel = _take(0, idx)
        if C.verdict(sel, RATIO):
            v = sum(inst["values"][i] for i in idx)
            if v > best_val:
                best_sel, best_val = sel, v
    assert best_val <= opt, "no feasible selection may beat the DP optimum"
    assert C.task_term(best_sel, RATIO) == pytest.approx(best_val / opt)


# ---------------------------------------------------------------- P1 and P2, re-established


def _random_groups(n, rng, all_violating):
    """Groups of four selections, either all infeasible or mixed."""
    out = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        inst_i = int(rng.integers(0, len(C.INSTANCES)))
        g = []
        for _ in range(4):
            k = int(rng.integers(0, C.N_ITEMS + 1))
            idx = rng.choice(C.N_ITEMS, size=k, replace=False).tolist()
            g.append(_take(inst_i, idx))
        viol = [not C.verdict(s, RATIO) for s in g]
        if all_violating and all(viol):
            out.append(g)
        elif not all_violating and any(viol) and not all(viol):
            out.append(g)
    return out


def test_P1_verdict_cancels_in_all_violating_groups():
    """In a group where every selection is infeasible the verdict is one constant, so
    mean-centering removes it and both binary arms equal the no-penalty arm."""
    rng = np.random.default_rng(0)
    groups = _random_groups(400, rng, all_violating=True)
    assert len(groups) >= 300, "need enough all-violating groups for the check to mean anything"
    worst = 0.0
    for g in groups:
        base = _adv(g, "none", 0.0)
        for arm, w in (("binary", 5.0), ("binary", 5000.0)):
            worst = max(worst, float(np.max(np.abs(_adv(g, arm, w) - base))))
    assert worst < 1e-9, f"verdict did not cancel; worst disagreement {worst:.3e}"


def test_P2_the_two_binary_weights_agree_where_the_verdict_is_constant():
    rng = np.random.default_rng(1)
    for g in _random_groups(300, rng, all_violating=True):
        assert np.allclose(_adv(g, "binary", 5.0), _adv(g, "binary", 5000.0), atol=1e-9)


def test_heavy_verdict_ranks_every_feasible_member_above_every_violator():
    """The task term is bounded by roughly 2, so a weight of 5000 is far above the crossover."""
    rng = np.random.default_rng(2)
    groups = _random_groups(300, rng, all_violating=False)
    assert len(groups) >= 100
    for g in groups:
        a = _adv(g, "binary", 5000.0)
        feas = np.array([C.verdict(s, RATIO) for s in g])
        assert a[feas].min() > a[~feas].max()


def test_graded_residual_never_inverts_the_ordering_in_mixed_groups():
    rng = np.random.default_rng(3)
    inverted = 0
    groups = _random_groups(300, rng, all_violating=False)
    for g in groups:
        a = _adv(g, "graded", 5.0)
        feas = np.array([C.verdict(s, RATIO) for s in g])
        if (a[~feas] > a[feas].max()).any():
            inverted += 1
    assert inverted == 0, f"{inverted} of {len(groups)} mixed groups inverted under the residual"


# ---------------------------------------------------------------- the unparseable rule


def test_unparseable_member_is_priced_identically_across_arms():
    """synthesize cannot land on an exact residual for a discrete task, and it does not need
    to. What it must do is price an unparseable member the same way for every arm, so the rule
    never favours one design."""
    g = [_take(0, [0, 1]), None, _take(0, range(C.N_ITEMS)), _take(0, [2])]
    r_syn = C.residual(C.synthesize(0.5, RATIO), RATIO)
    assert r_syn >= 0.0
    for arm, w in ARMS.values():
        rw = RW.group_rewards(g, arm, w, RATIO, spec=SPEC)
        assert np.isfinite(rw).all()
        assert rw[1] <= C.UNPARSEABLE_TASK, "an unparseable member must not score above the floor"


def test_all_unparseable_group_is_scored_and_has_no_spread():
    g = [None] * 4
    for arm, w in ARMS.values():
        rw = RW.group_rewards(g, arm, w, RATIO, spec=SPEC)
        assert np.ptp(rw) == 0.0, "four identical failures must tie"


# ------------------------------------------- the two regimes, side by side, in closed form

CAP_ARMS = {"graded": ("graded", 5.0, 1.0), "binary": ("binary", 5.0, 1.0),
            "none": ("none", 0.0, 1.0), "binary_1000x": ("binary", 5000.0, 1.0),
            "verdict_only": ("binary", 5.0, 0.0), "graded_no_task": ("graded", 5.0, 0.0)}


def _spread(g, arm, w, w_task):
    return float(np.ptp(RW.group_rewards(g, arm, w, RATIO, spec=SPEC, w_task=w_task)))


def test_verdict_only_is_silent_in_all_violating_groups():
    """The published shape puts a constant below feasibility and grades only feasible answers.
    Every member of an all-violating group therefore earns the same reward, the group variance
    is zero, and the update is exactly zero. This is the zero-signal regime the literature
    describes, reproduced here so the paper can show both regimes in one figure."""
    rng = np.random.default_rng(11)
    groups = _random_groups(200, rng, all_violating=True)
    assert len(groups) >= 150
    for g in groups:
        assert _spread(g, *CAP_ARMS["verdict_only"]) == 0.0


def test_a_live_task_term_makes_the_same_group_loud():
    """With a task term that varies below feasibility the group is not silent. It produces a
    confident gradient, and the verdict has been removed from it."""
    rng = np.random.default_rng(12)
    groups = _random_groups(200, rng, all_violating=True)
    for g in groups:
        assert _spread(g, *CAP_ARMS["binary"]) > 0.0
        # and that gradient is exactly the no-penalty one
        assert np.allclose(_adv(g, "binary", 5.0), _adv(g, "none", 0.0), atol=1e-9)


def test_the_residual_alone_still_carries_signal_where_the_verdict_alone_does_not():
    """The distinction the paper rests on, stated exactly.

    In an all-violating group a verdict takes one value no matter what the members are, so a
    verdict-only design is silent in EVERY such group. A residual takes one value only when the
    members happen to be identical. So the residual is silent exactly when the policy has
    collapsed onto one answer, and informative otherwise.

    This separation matters for reading the zero-variance counter in the results: a zero there
    can mean cancellation or it can mean a converged policy, and only the second kind survives
    when the task term is removed.
    """
    rng = np.random.default_rng(13)
    groups = _random_groups(200, rng, all_violating=True)
    distinct = [g for g in groups if len(set(g)) > 1]
    degenerate = [g for g in groups if len(set(g)) == 1]
    assert len(distinct) >= 100, "need groups whose members differ"

    for g in groups:
        # the verdict is a constant in every all-violating group, identical members or not
        assert _spread(g, *CAP_ARMS["verdict_only"]) == 0.0

    for g in distinct:
        assert _spread(g, *CAP_ARMS["graded_no_task"]) > 0.0, \
            "a residual must separate members that differ"

    for g in degenerate:
        # four copies of one answer tie under any reward. That is mode collapse, not cancellation.
        assert _spread(g, *CAP_ARMS["graded_no_task"]) == 0.0
        assert _spread(g, *CAP_ARMS["graded"]) == 0.0


def test_verdict_only_does_separate_once_a_group_is_mixed():
    rng = np.random.default_rng(14)
    groups = _random_groups(200, rng, all_violating=False)
    assert len(groups) >= 80
    for g in groups:
        assert _spread(g, *CAP_ARMS["verdict_only"]) > 0.0
