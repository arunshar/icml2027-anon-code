"""This file checks the fallback-reward group counts against hand-picked groups and a sweep.

The counts describe how the unparseable fallback prices a completion beside its parsed
group-mates. A verdict arm's stand-in is always infeasible, so it never outranks a parsed
violator. A residual arm's stand-in can fall short of a violator's residual, so it sometimes
does.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze as A  # noqa: E402
import capacity_task as C  # noqa: E402
import rewards as RW  # noqa: E402
import tasks  # noqa: E402


def _full(i):
    """This returns the take-all selection on instance i. It chooses every item, so it
    reaches that instance's largest residual."""
    return (i,) + tuple(range(C.N_ITEMS))


def _run(groups, *, arm="graded", w=5.0, w_task=1.0, ratio=0.03, task="capacity", regime="H"):
    """This builds one training step from real groups. Each group is priced by the same
    reward and advantage code the trainer calls. The fallback counts then see real geometry
    instead of a hand-typed reward."""
    tspec = tasks.get(task)
    spec = None if task == "speed" else tspec
    gs = []
    for items in groups:
        rw = RW.group_rewards(items, arm, w, ratio, spec=spec, w_task=w_task)
        gs.append({
            tspec.ITEMS_KEY: items,
            "rewards": rw.tolist(),
            "adv": RW.advantage(rw).tolist(),
            "all_violating": RW.group_is_all_violating(items, ratio, spec=spec),
            "has_violator": RW.group_has_violator(items, ratio, spec=spec),
        })
    cfg = {"vmax": ratio, "task": task, "arm": arm, "w": w, "w_task": w_task,
           "seed": 0, "regime": regime}
    return {"cfg": cfg, "train": [{"step": 1, "groups": gs}], "evals": []}


# ---------------------------------------------------------------- hand-picked groups


def test_a_verdict_only_group_with_every_parsed_member_feasible_has_zero_spread_and_no_outrank():
    run = _run([[(0,), (0, 0), (0, 10), None]], arm="binary", w_task=0.0)
    row = A.fallback_group_counts([run])[0]
    assert row["groups_with_unparseable"] == 1
    assert row["all_feasible_with_unparseable"] == 1
    assert row["all_feasible_with_unparseable_zero_spread"] == 0
    assert row["unparseable_outranks_parsed_violator"] == 0


def test_overwriting_the_logged_rewards_with_a_tie_flips_the_zero_spread_flag():
    run = _run([[(0,), (0, 0), (0, 10), None]], arm="binary", w_task=0.0)
    run["train"][0]["groups"][0]["rewards"] = [1.0, 1.0, 1.0, 1.0]
    row = A.fallback_group_counts([run])[0]
    assert row["all_feasible_with_unparseable_zero_spread"] == 1


def test_a_parsed_violator_removes_the_group_from_the_all_feasible_count():
    run = _run([[_full(0), (0, 0), None]], arm="binary", w_task=0.0)
    row = A.fallback_group_counts([run])[0]
    assert row["groups_with_unparseable"] == 1
    assert row["all_feasible_with_unparseable"] == 0


def test_a_group_with_no_parsed_member_is_not_all_feasible_and_counts_nothing_else():
    run = _run([[None, None, None, None]], arm="binary", w_task=0.0)
    row = A.fallback_group_counts([run])[0]
    assert row["groups_with_unparseable"] == 1
    assert row["all_feasible_with_unparseable"] == 0
    assert row["all_feasible_with_unparseable_zero_spread"] == 0
    assert row["unparseable_outranks_parsed_violator"] == 0
    assert row["unparseable_beside_residual_above_reach"] == 0


# ---------------------------------------------------------------- property sweep, verdict arms


def _random_capacity_groups(seed, n=300, group_size=4):
    """This returns n groups, and each group holds one unparseable member first and
    group_size minus 1 random parsed selections after it, so every group has both kinds
    of member by construction."""
    rng = np.random.default_rng(seed)
    groups = []
    for _ in range(n):
        inst = int(rng.integers(0, len(C.INSTANCES)))
        items = [None]
        for _ in range(group_size - 1):
            k = int(rng.integers(0, C.N_ITEMS + 1))
            idx = tuple(sorted(rng.choice(C.N_ITEMS, size=k, replace=False).tolist()))
            items.append((inst,) + idx)
        groups.append(items)
    return groups


@pytest.mark.parametrize("arm,w,w_task", [
    ("binary", 5.0, 1.0), ("binary", 5.0, 0.0), ("binary", 5000.0, 1.0),
])
def test_a_verdict_arms_stand_in_never_outranks_a_parsed_violator(arm, w, w_task):
    groups = _random_capacity_groups(20260918)
    run = _run(groups, arm=arm, w=w, w_task=w_task)
    row = A.fallback_group_counts([run])[0]
    assert row["groups_with_unparseable"] == 300
    assert row["unparseable_outranks_parsed_violator"] == 0


# ---------------------------------------------------------------- residual arms and reach


def test_a_residual_arm_can_let_the_unparseable_fallback_outrank_a_violator_beyond_its_reach():
    run = _run([[_full(5), (5,), None, (5, 0)]], arm="graded")
    row = A.fallback_group_counts([run])[0]
    assert row["unparseable_beside_residual_above_reach"] == 1
    assert row["unparseable_outranks_parsed_violator"] == 1


def test_a_violator_at_exactly_the_stand_ins_reach_does_not_outrank_the_fallback():
    run = _run([[_full(0), (0,), None]], arm="graded")
    row = A.fallback_group_counts([run])[0]
    assert row["unparseable_beside_residual_above_reach"] == 0
    assert row["unparseable_outranks_parsed_violator"] == 0


def test_the_residual_only_arm_ties_the_fallback_with_that_same_violator():
    run = _run([[_full(0), (0,), None]], arm="graded", w_task=0.0)
    row = A.fallback_group_counts([run])[0]
    assert row["unparseable_outranks_parsed_violator"] == 0


def test_the_stand_in_reach_is_the_first_instances_take_all_residual_and_speed_has_none():
    cap_run = _run([[(0,), None]], ratio=0.03)
    cap_row = A.fallback_group_counts([cap_run])[0]
    assert cap_row["stand_in_reach"] == pytest.approx(C.residual(_full(0), 0.03))

    speed_run = _run([[None, 100000.0, 3.0]], task="speed", regime="A", ratio=5.0, arm="binary")
    speed_row = A.fallback_group_counts([speed_run])[0]
    assert speed_row["stand_in_reach"] is None
    assert speed_row["unparseable_beside_residual_above_reach"] == 0


def test_a_residual_only_row_is_named_and_flagged_by_the_arm_specs():
    run = _run([[(0,), None]], arm="graded", w_task=0.0)
    row = A.fallback_group_counts([run])[0]
    assert row["arm"] == "graded_no_task"
    assert row["penalty"] == "graded"
    assert row["w_task"] == 0.0
