"""The remedy variants, filters and anchors, checked on hand-computed groups.

Each variant is a published remedy with a named source. The tests pin the arithmetic the trainer and
the analysis share, and the one property the paper's argument turns on: whether a group-constant
verdict still cancels inside an all-violating group under that variant.
"""
import csv
import math
import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze  # noqa: E402
import capacity_task as C  # noqa: E402
import rewards as RW  # noqa: E402
import tasks  # noqa: E402

CAP = tasks.get("capacity")
SPEED = tasks.get("speed")


def _capacity_ratios():
    """The two capacity levels, read from the manifest that sets them rather than typed here."""
    rows = list(csv.DictReader(open(os.path.join(HERE, "slurm", "manifest_capacity.csv"))))
    return sorted({float(r["vmax"]) for r in rows})


def _speed_vmax(regime):
    import yaml
    return float(yaml.safe_load(open(os.path.join(HERE, "configs", "regimes.yaml")))[regime]["v_max"])


# ---------------------------------------------------------------- advantage variants


def test_the_grpo_variant_is_the_frozen_advantage_exactly():
    rng = np.random.default_rng(0)
    groups = [rng.normal(size=4) * 10 for _ in range(200)]
    groups += [np.full(4, -15.0), np.array([-10.0, -10.0, 0.4, -10.0])]
    for r in groups:
        assert np.array_equal(RW.advantage_variant(r, "grpo"), RW.advantage(r))


def test_drgrpo_on_a_three_way_tie_gives_minus_three_quarters_and_quarters():
    x, d = 1.0, 4.0
    got = RW.advantage_variant([x, x + d, x + d, x + d], "drgrpo")
    assert got.tolist() == [-3 * d / 4, d / 4, d / 4, d / 4]
    for d in (0.37, -2.5, 1e3):
        got = RW.advantage_variant([x, x + d, x + d, x + d], "drgrpo")
        assert np.allclose(got, [-3 * d / 4, d / 4, d / 4, d / 4], rtol=0, atol=1e-12)


def test_ngrpo_matches_the_hand_formula_with_population_std_over_k_plus_one():
    """NGRPO, arXiv 2509.18851, Sec. 4.1, Eq. 6: mean and std over the G+1 values, divisor G+1."""
    r, anchor = [1.0, 3.0, 3.0, 3.0], 0.5
    z = r + [anchor]
    mean = sum(z) / len(z)
    std = math.sqrt(sum((v - mean) ** 2 for v in z) / len(z))
    want = [(v - mean) / (std + 1e-6) for v in r]
    for variant in RW.ANCHORED_VARIANTS:
        got = RW.advantage_variant(r, variant, anchor)
        assert len(got) == len(r), "only the K real advantages are returned"
        assert np.allclose(got, want, rtol=1e-13, atol=0)
    # the sample deviation would give a different answer, so this pins the divisor
    sample = math.sqrt(sum((v - mean) ** 2 for v in z) / (len(z) - 1))
    assert not np.allclose(RW.advantage_variant(r, "ngrpo_sup", anchor),
                           [(v - mean) / (sample + 1e-6) for v in r], rtol=1e-6, atol=0)


def test_anchor_arguments_are_refused_where_they_mean_nothing():
    with pytest.raises(ValueError):
        RW.advantage_variant([1.0, 2.0], "grpo", 0.5)
    with pytest.raises(ValueError):
        RW.advantage_variant([1.0, 2.0], "drgrpo", 0.5)
    with pytest.raises(ValueError):
        RW.advantage_variant([1.0, 2.0], "ngrpo_feasible", None)
    with pytest.raises(ValueError):
        RW.advantage_variant([1.0, 2.0], "nope")


# ---------------------------------------------------------------- cancellation per variant


def _speed_all_violating(rng, vmax, n):
    lo = int(math.floor(vmax)) + 1
    out = []
    for _ in range(n):
        g = [float(v) for v in rng.integers(lo, 1000, size=4)]
        if rng.random() < 0.3:
            g[int(rng.integers(0, 4))] = None          # unparseable counts as failing
        out.append(g)
    return out


def _adv(items, variant, arm, w, vmax, spec=None, w_task=1.0, instance=0):
    r = RW.group_rewards(items, arm, w, vmax, spec=spec, w_task=w_task)
    a = RW.reward_anchor(variant, arm, w, vmax, spec=spec, w_task=w_task, instance=instance)
    return RW.advantage_variant(r, variant, a)


def test_ngrpo_sup_cancels_at_the_light_weight_and_not_above_the_crossover():
    vmax = _speed_vmax("A")
    light = analyze.ARM_SPECS["binary"][1]
    above = 5 * light
    rng = np.random.default_rng(1)
    for items in _speed_all_violating(rng, vmax, 300):
        assert RW.group_is_all_violating(items, vmax)
        none = _adv(items, "ngrpo_sup", "none", 0.0, vmax)
        assert np.max(np.abs(_adv(items, "ngrpo_sup", "binary", light, vmax) - none)) < 1e-9
        assert np.max(np.abs(_adv(items, "ngrpo_sup", "binary", above, vmax) - none)) > 1e-3


def test_ngrpo_sup_anchor_is_infeasible_at_the_light_weight_and_feasible_above():
    """Why the previous test comes out as it does, stated as relations between derived anchors."""
    vmax = _speed_vmax("A")
    light = analyze.ARM_SPECS["binary"][1]
    sup_none = RW.reward_anchor("ngrpo_sup", "none", 0.0, vmax)
    feasible = RW.reward_anchor("ngrpo_feasible", "binary", light, vmax)
    assert RW.reward_anchor("ngrpo_sup", "binary", light, vmax) == sup_none - light > feasible
    assert RW.reward_anchor("ngrpo_sup", "binary", 5 * light, vmax) == feasible


def _spread(items, arm, w, vmax, spec=None, w_task=1.0):
    return np.ptp(RW.group_rewards(items, arm, w, vmax, spec=spec, w_task=w_task)) > RW.ZERO_SPREAD_TOL


def test_ngrpo_feasible_does_not_cancel_in_any_all_violating_group_with_spread():
    light = analyze.ARM_SPECS["binary"][1]
    rng = np.random.default_rng(2)
    n_speed = 0
    for regime in ("A", "B"):
        vmax = _speed_vmax(regime)
        for items in _speed_all_violating(rng, vmax, 300):
            if not _spread(items, "binary", light, vmax):
                continue
            n_speed += 1
            d = _adv(items, "ngrpo_feasible", "binary", light, vmax) \
                - _adv(items, "ngrpo_feasible", "none", 0.0, vmax)
            assert np.max(np.abs(d)) > 1e-6, items
    assert n_speed > 100
    ratio = _capacity_ratios()[0]
    n = 0
    for _ in range(3000):
        inst = int(rng.integers(0, len(C.INSTANCES)))
        items = [(inst,) + tuple(sorted(rng.choice(C.N_ITEMS, size=int(rng.integers(8, 17)),
                                                   replace=False).tolist())) for _ in range(4)]
        if not RW.group_is_all_violating(items, ratio, spec=CAP):
            continue
        if not _spread(items, "binary", light, ratio, CAP):
            continue
        n += 1
        d = _adv(items, "ngrpo_feasible", "binary", light, ratio, CAP, 1.0, inst) \
            - _adv(items, "ngrpo_feasible", "none", 0.0, ratio, CAP, 1.0, inst)
        assert np.max(np.abs(d)) > 1e-6, items
    assert n > 100, "too few all-violating capacity groups to mean anything"


def test_ngrpo_on_a_zero_spread_group_gives_every_member_the_sign_over_root_k():
    """A property of Eq. 6, found while testing the claim above. When all K rewards equal r, the
    mean over the K+1 values sits at (Kr + a)/(K+1) and the population deviation is
    sqrt(K)|r - a|/(K+1), so every member gets sign(r - a)/sqrt(K) apart from the 1e-6 term.

    The verdict shifts r and leaves a in place. It therefore changes a zero-spread group's update
    only when it moves r across the anchor. Below that weight such a group is reinforced or
    suppressed as a whole whatever the verdict says."""
    vmax = _speed_vmax("A")
    light = analyze.ARM_SPECS["binary"][1]
    capped = [float(v) for v in (264, 309, 564, 659)]          # all four at the task cap
    none_r = RW.group_rewards(capped, "none", 0.0, vmax)
    assert np.ptp(none_r) == 0.0
    a = RW.reward_anchor("ngrpo_feasible", "binary", light, vmax)
    K = len(capped)
    for arm, w in (("none", 0.0), ("binary", light), ("binary", 5 * light)):
        r = RW.group_rewards(capped, arm, w, vmax)
        got = RW.advantage_variant(r, "ngrpo_feasible", a)
        want = np.sign(r[0] - a) / math.sqrt(K)
        assert np.allclose(got, want, rtol=0, atol=1e-6), (arm, w, got)
    light_d = _adv(capped, "ngrpo_feasible", "binary", light, vmax) \
        - _adv(capped, "ngrpo_feasible", "none", 0.0, vmax)
    assert np.max(np.abs(light_d)) < 1e-6, "the light verdict does not move r below the anchor"
    heavy_d = _adv(capped, "ngrpo_feasible", "binary", 5 * light, vmax) \
        - _adv(capped, "ngrpo_feasible", "none", 0.0, vmax)
    assert np.allclose(np.abs(heavy_d), 2 / math.sqrt(K), atol=1e-6), "this weight flips the sign"


@pytest.mark.parametrize("variant", ["grpo", "drgrpo"])
def test_the_unanchored_variants_still_cancel(variant):
    vmax = _speed_vmax("A")
    light = analyze.ARM_SPECS["binary"][1]
    for items in _speed_all_violating(np.random.default_rng(3), vmax, 300):
        d = _adv(items, variant, "binary", light, vmax) - _adv(items, variant, "none", 0.0, vmax)
        assert np.max(np.abs(d)) < 1e-9


# ---------------------------------------------------------------- filters


def test_keep_group_edge_cases():
    k = RW.keep_group
    # every member fails, one of them unparseable
    items = [None, 40.0, 60.0, 900.0]
    v = RW.group_verdicts(items, 5.0)
    assert v == [False, False, False, False]
    assert k("verdict_mixed", [-15.0, -1.0, -1.0, 15.0], v) is False
    # every member passes
    assert k("verdict_mixed", [0.1, 0.2, 0.3, 0.4], [True] * 4) is False
    # mixed
    assert k("verdict_mixed", [0.1, -15.0, 0.3, 0.4], [True, False, True, True]) is True
    assert k("verdict_mixed", [0.0, 0.0, 0.0, 0.0], [True, False, False, False]) is True
    # spread
    assert k("reward_spread", [1.0, 1.0, 1.0, 1.0 + 1e-15], [False] * 4) is False
    assert k("reward_spread", [1.0, 1.0, 1.0, 1.0 + 1e-9], [False] * 4) is True
    assert k("reward_spread", [-15.0] * 4, [True, False, True, False]) is False
    # none keeps everything
    assert k("none", [-15.0] * 4, [False] * 4) is True
    with pytest.raises(ValueError):
        k("dapo", [1.0], [True])


def test_the_spread_tolerance_is_the_analysis_tolerance():
    assert RW.ZERO_SPREAD_TOL is analyze.ZERO_VAR_TOL
    assert RW.ZERO_SPREAD_TOL == analyze.ZERO_VAR_TOL


def test_group_verdicts_agree_with_the_violation_flags():
    vmax = _speed_vmax("A")
    items = [None, 3.0, 5.0, 6.0, -5.0, -6.0]
    assert RW.group_verdicts(items, vmax) == [SPEED.verdict(x, vmax) for x in items]
    sel = [None, (0,), (0, 0, 1, 2, 3, 4, 5, 6, 7)]
    ratio = _capacity_ratios()[0]
    assert RW.group_verdicts(sel, ratio, spec=CAP) == [C.verdict(x, ratio) for x in sel]


# ---------------------------------------------------------------- anchors and output tables


def test_anchors_consume_no_randomness_and_are_cached():
    RW._anchor_cached.cache_clear()
    C._output_table.cache_clear()
    np.random.seed(123)
    random.seed(123)
    np_state, py_state = np.random.get_state(), random.getstate()
    vmax, ratio = _speed_vmax("A"), _capacity_ratios()[0]
    first = [RW.reward_anchor(v, "binary", 5.0, vmax) for v in RW.ANCHORED_VARIANTS]
    first += [RW.reward_anchor(v, "graded", 5.0, ratio, spec=CAP, instance=i)
              for v in RW.ANCHORED_VARIANTS for i in range(len(C.INSTANCES))]
    hits = RW._anchor_cached.cache_info().hits
    second = [RW.reward_anchor(v, "binary", 5.0, vmax) for v in RW.ANCHORED_VARIANTS]
    assert second == first[:2]
    assert RW._anchor_cached.cache_info().hits == hits + 2
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1]) \
        and after[2:] == np_state[2:]
    assert random.getstate() == py_state
    assert RW.reward_anchor("grpo", "binary", 5.0, vmax) is None
    assert RW.reward_anchor("drgrpo", "binary", 5.0, vmax) is None


def test_the_speed_enumeration_range_is_already_exhaustive():
    for vmax in (_speed_vmax("A"), _speed_vmax("B")):
        for arm, w in (("graded", 5.0), ("binary", 5.0), ("binary", 25.0), ("none", 0.0)):
            for s_max in (RW.SPEED_ENUM_MAX, 10 * RW.SPEED_ENUM_MAX + 9):
                t = RW.speed_output_table(vmax, s_max)
                got = float((t["task"] + w * RW._table_penalty(t, arm)).max())
                if s_max == RW.SPEED_ENUM_MAX:
                    base = got
                assert got == base


def test_the_speed_table_matches_the_scalar_functions():
    vmax = _speed_vmax("A")
    t = SPEED.output_table(vmax)
    for s in (0, 1, 4, 5, 6, 199, 200, 201, 999, 123456, RW.SPEED_ENUM_MAX):
        assert t["task"][s] == SPEED.task_term(float(s), vmax)
        assert t["residual"][s] == SPEED.residual(float(s), vmax)
        assert bool(t["verdict"][s]) == SPEED.verdict(float(s), vmax)
    assert SPEED.feasible_task_sup(vmax) == SPEED.task_term(float(math.floor(vmax)), vmax)


def test_the_sup_anchor_is_the_largest_reward_group_rewards_can_produce():
    vmax = _speed_vmax("A")
    rng = np.random.default_rng(4)
    for arm, w in (("graded", 5.0), ("binary", 5.0), ("none", 0.0)):
        sup = RW.reward_anchor("ngrpo_sup", arm, w, vmax)
        for _ in range(200):
            items = [None if rng.random() < 0.2 else float(rng.integers(-2000, 2000))
                     for _ in range(4)]
            assert RW.group_rewards(items, arm, w, vmax).max() <= sup
        best = RW.group_rewards([float(i) for i in range(0, 400)], arm, w, vmax).max()
        assert best == sup


@pytest.fixture
def _memo_dp(monkeypatch):
    """task_term recomputes the knapsack optimum on every call. The brute-force loop below calls it
    131,072 times, so the optimum is memoized for the test. The arithmetic is untouched."""
    real, memo = C.dp_optimum, {}

    def cached(values, weights, capacity):
        key = (tuple(values), tuple(weights), capacity)
        if key not in memo:
            memo[key] = real(values, weights, capacity)
        return memo[key]
    monkeypatch.setattr(C, "dp_optimum", cached)


def test_the_capacity_table_equals_a_brute_force_loop_on_instance_zero(_memo_dp):
    ratios = _capacity_ratios()
    assert len(ratios) == 2
    for ratio in ratios:
        t = CAP.output_table(ratio, 0)
        assert t["task"].shape == (1 << C.N_ITEMS,)
        for m in range(1 << C.N_ITEMS):
            sel = (0,) + tuple(i for i in range(C.N_ITEMS) if m >> i & 1)
            assert t["task"][m] == C.task_term(sel, ratio), (ratio, m)
            assert t["residual"][m] == C.residual(sel, ratio), (ratio, m)
            assert bool(t["verdict"][m]) == C.verdict(sel, ratio), (ratio, m)


def test_the_capacity_feasible_sup_is_the_knapsack_optimum_share():
    for ratio in _capacity_ratios():
        for i, inst in enumerate(C.INSTANCES):
            opt = C.dp_optimum(inst["values"], inst["weights"], C.capacity_of(inst, ratio))
            assert CAP.feasible_task_sup(ratio, i) == opt / opt
            assert CAP.instance_for(i) == i
    assert SPEED.instance_for(3) == 0


def test_the_capacity_table_is_read_only():
    t = CAP.output_table(_capacity_ratios()[0], 1)
    with pytest.raises(ValueError):
        t["task"][0] = 99.0
