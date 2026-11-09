import numpy as np
import pytest

from rewards import (ARMS, arm_reward, gate_reward, group_rewards, pen_binary,
                     pen_graded, residual, task, UNPARSEABLE_TASK)

VMAX = 30.0
S = np.arange(-250, 251, dtype=np.float64)


def test_task_is_monotone_up_to_200_then_flat():
    t = task(S, VMAX)
    pos = S[S >= 0]
    tp = task(pos, VMAX)
    assert np.all(np.diff(tp[pos <= 200]) >= 0)
    assert np.all(tp[pos >= 200] == 20.0)
    assert np.all(t >= 0)


def test_graded_is_zero_inside_and_monotone_decreasing_outside():
    p = pen_graded(S, VMAX)
    inside = np.abs(S) <= VMAX
    assert np.all(p[inside] == 0.0)
    assert np.all(p[~inside] < 0.0)
    pos = S[S > VMAX]
    assert np.all(np.diff(pen_graded(pos, VMAX)) < 0)


def test_binary_takes_only_two_values_and_matches_residual_sign():
    p = pen_binary(S, VMAX)
    assert set(np.unique(p).tolist()) == {0.0, -1.0}
    assert np.array_equal(p == -1.0, residual(S, VMAX) > 0)


def test_binary_at_1000x_scales_only_the_penalty_part():
    lhs = arm_reward(S, "binary", 5000.0, VMAX) - task(S, VMAX)
    rhs = 1000.0 * (arm_reward(S, "binary", 5.0, VMAX) - task(S, VMAX))
    assert np.allclose(lhs, rhs)


def test_none_arm_is_the_task_alone():
    assert np.array_equal(arm_reward(S, "none", 5.0, VMAX), task(S, VMAX))


def test_unknown_arm_raises():
    with pytest.raises(ValueError):
        arm_reward(10.0, "gate", 1.0, VMAX)


def test_gate_reward_zeroes_violators_only():
    g = gate_reward(S, VMAX)
    assert np.all(g[np.abs(S) > VMAX] == 0.0)
    assert np.array_equal(g[np.abs(S) <= VMAX], task(S[np.abs(S) <= VMAX], VMAX))


@pytest.mark.parametrize("arm", ARMS)
def test_unparseable_rule_is_identical_across_arms(arm):
    speeds = [20.0, None, 60.0, 200.0]
    r = group_rewards(speeds, arm, 5.0, VMAX)
    # the None row takes the task floor plus the arm's penalty at the group's max residual
    s_at_rmax = VMAX * (1.0 + float(residual(np.array([20.0, 60.0, 200.0]), VMAX).max()))
    expected = UNPARSEABLE_TASK + 5.0 * float(arm_reward(s_at_rmax, arm, 1.0, VMAX) - task(s_at_rmax, VMAX))
    assert r[1] == pytest.approx(expected)
    assert r[0] == pytest.approx(float(arm_reward(20.0, arm, 5.0, VMAX)))


def test_all_unparseable_group_uses_one_envelope_residual():
    r = group_rewards([None, None, None, None], "graded", 5.0, VMAX)
    assert np.allclose(r, UNPARSEABLE_TASK + 5.0 * (-100.0 * 1.0))
