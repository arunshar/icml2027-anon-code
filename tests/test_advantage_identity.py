"""P1 and P2 as algebra, plus the two provenance groups the paper prints."""
import numpy as np
import pytest

from rewards import advantage, arm_reward, group_is_all_violating

VMAX = 30.0
rng = np.random.default_rng(0)


def _adv(speeds, arm, w):
    return advantage(arm_reward(np.asarray(speeds, float), arm, w, VMAX))


def test_provenance_group_reproduces_saved_advantages():
    # groupnorm_results.json: advantages_w_hard_5 for (20, 25, 60, 200) at w=5
    got = _adv([20, 25, 60, 200], "graded", 5.0)
    assert np.allclose(got, [0.615134, 0.615505, 0.246518, -1.477157], atol=5e-7)
    assert got[2] > 0  # the 60-knot violator is reinforced


def test_provenance_saturation_moves_adv60_by_0_000335():
    a5 = _adv([20, 25, 60, 200], "graded", 5.0)[2]
    a5e6 = _adv([20, 25, 60, 200], "graded", 5e6)[2]
    assert abs(a5 - a5e6) == pytest.approx(0.000335, abs=1e-6)


def test_mixed_group_binary_alone_matches_verify_claims_line_40():
    # verify_claims_output.txt line 40: mixed group (20,60,100,200), binary verdict alone
    from rewards import pen_binary
    got = advantage(pen_binary(np.array([20, 60, 100, 200], float), VMAX))
    assert np.allclose(got, [1.5, -0.5, -0.5, -0.5], atol=1e-6)


def test_P1_binary_and_none_identical_in_all_violating_groups():
    n = 0
    for _ in range(20000):
        g = rng.integers(31, 201, size=4).astype(float)
        assert group_is_all_violating(list(g), VMAX)
        # exact in algebra; float64 rounding at w=5000 sits near 1e-12, so the gate is 1e-9
        assert np.allclose(_adv(g, "binary", 5.0), _adv(g, "none", 0.0), atol=1e-9)
        assert np.allclose(_adv(g, "binary", 5000.0), _adv(g, "none", 0.0), atol=1e-9)
        n += 1
    assert n == 20000


def test_P2_binary_weights_identical_where_verdict_is_constant():
    # all-violating and all-feasible groups: the verdict is a constant, so w cancels
    for _ in range(10000):
        g = rng.integers(31, 201, size=4).astype(float)
        assert np.allclose(_adv(g, "binary", 5.0), _adv(g, "binary", 5000.0), atol=1e-9)
        f = rng.integers(0, 31, size=4).astype(float)
        assert np.allclose(_adv(f, "binary", 5.0), _adv(f, "binary", 5000.0), atol=1e-9)


def test_P2_boundary_binary_weights_differ_in_mixed_groups_and_5000_owns_the_ordering():
    # The task range is 20, so at w=5 a fast violator can outrank a feasible member,
    # and at w=5000 it never can. This is the binary crossover; the two arms bracket it.
    differ = 0
    for _ in range(10000):
        g = np.concatenate([rng.integers(0, 31, size=2), rng.integers(31, 201, size=2)]).astype(float)
        a5, a5000 = _adv(g, "binary", 5.0), _adv(g, "binary", 5000.0)
        if not np.allclose(a5, a5000, atol=1e-9):
            differ += 1
        # at w=5000 every feasible member ranks above every violator
        assert a5000[:2].min() > a5000[2:].max()
    assert differ > 0


def test_graded_differs_from_none_in_all_violating_groups():
    # the residual varies within the group, so it survives mean-centering
    hits = 0
    for _ in range(2000):
        g = rng.integers(31, 201, size=4).astype(float)
        if not np.allclose(_adv(g, "graded", 5.0), _adv(g, "none", 0.0), atol=1e-6):
            hits += 1
    assert hits > 1900


def test_graded_reinforces_the_member_closest_to_the_envelope_when_all_violate():
    # objection_check_output.txt line 8: 20,000/20,000
    r = np.random.default_rng(1)
    for _ in range(20000):
        g = r.integers(31, 201, size=4).astype(float)
        a = _adv(g, "graded", 5.0)
        if (a > 0).any():
            assert g[np.argmax(a)] == g.min()
