"""How far a trained verdict run and a trained no-penalty run stay the same run.

The closed-form identity says an added verdict cancels inside an all-violating group. The
strongest trained form of that claim is two runs that sample the same answers step after step,
and the speed task produced exactly one such seed. The paper cites that count, so the count is
generated rather than typed, and these tests pin what it must report.

The case that matters most is the last one. A level where a few groups are mixed will diverge,
and a counter that quietly reported "identical" there would turn a partial result into a false
one.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402


def _run(arm, w, seed, picks_per_step, regime="A", task="speed", w_task=1.0,
         advs=None, all_violating=True):
    """A run whose logged groups hold exactly the given answers, one list per step.

    `advs` supplies per-step advantage vectors when a test cares about them, and defaults to a
    zero vector of the right length so the common case stays short.
    """
    key = "speeds" if task == "speed" else "selections"
    train = []
    for i, picks in enumerate(picks_per_step):
        adv = advs[i] if advs is not None else [0.0] * len(picks)
        train.append({"step": i + 1,
                      "groups": [{key: picks, "adv": adv,
                                  "all_violating": all_violating}]})
    return {"cfg": {"arm": arm, "w": w, "seed": seed, "regime": regime,
                    "task": task, "w_task": w_task, "vmax": 5.0},
            "train": train, "evals": []}


def test_identical_runs_report_every_step_and_no_divergence():
    picks = [[100.0, 200.0, 300.0, 400.0] for _ in range(300)]
    rows = A.verdict_none_agreement([_run("binary", 5.0, 2, picks),
                                     _run("none", 0.0, 2, picks)])
    assert len(rows) == 1
    assert rows[0]["identical_steps"] == 300
    assert rows[0]["steps_compared"] == 300
    assert rows[0]["first_divergent_step"] is None, \
        "a run that never diverges must report no divergent step, not step 0"


def test_the_first_divergent_step_is_the_first_one_and_not_a_later_one():
    a = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    b = [[1.0, 2.0], [3.0, 4.0], [9.0, 9.0], [7.0, 8.0]]
    rows = A.verdict_none_agreement([_run("binary", 5.0, 0, a), _run("none", 0.0, 0, b)])
    assert rows[0]["first_divergent_step"] == 3
    assert rows[0]["identical_steps"] == 3, "the runs re-converge on the fourth step"


def test_seeds_and_regimes_are_never_compared_across_each_other():
    """Pairing a seed against a different seed would compare two unrelated trajectories and
    report their disagreement as evidence about the verdict."""
    picks = [[1.0, 2.0]]
    rows = A.verdict_none_agreement([
        _run("binary", 5.0, 0, picks, regime="A"),
        _run("none", 0.0, 1, picks, regime="A"),
        _run("binary", 5.0, 0, [[9.0, 9.0]], regime="B"),
        _run("none", 0.0, 0, [[9.0, 9.0]], regime="B"),
    ])
    assert [(r["regime"], r["seed"]) for r in rows] == [("B", 0)], \
        "only the regime B pair shares a regime and a seed"


def test_an_unpaired_arm_is_dropped_rather_than_compared_against_nothing():
    rows = A.verdict_none_agreement([_run("binary", 5.0, 0, [[1.0]])])
    assert rows == []


def test_the_capacity_task_is_read_through_its_own_group_key():
    """Capacity groups log `selections`, not `speeds`. Reading the wrong key would raise, or
    worse, compare two empty lists on every step and call the runs identical."""
    a = [[[0, 1, 2], [0, 3]], [[0, 1], [0, 2]]]
    b = [[[0, 1, 2], [0, 3]], [[0, 5], [0, 6]]]
    rows = A.verdict_none_agreement([
        _run("binary", 5.0, 0, a, regime="H", task="capacity"),
        _run("none", 0.0, 0, b, regime="H", task="capacity"),
    ])
    assert rows[0]["identical_steps"] == 1 and rows[0]["first_divergent_step"] == 2


def test_a_shorter_run_limits_the_comparison_instead_of_padding_it():
    """A run that died early must shorten the window, never count its missing steps as agreement."""
    rows = A.verdict_none_agreement([
        _run("binary", 5.0, 0, [[1.0], [2.0], [3.0]]),
        _run("none", 0.0, 0, [[1.0], [2.0]]),
    ])
    assert rows[0]["steps_compared"] == 2 and rows[0]["identical_steps"] == 2


def test_the_verdict_run_s_mixed_groups_are_counted():
    """A group with a feasible member is one where the verdict could act. The paper explained a
    divergence by mixed groups before this existed, and the run it named has none of them."""
    picks = [[1.0, 2.0], [3.0, 4.0]]
    a = _run("binary", 5.0, 0, picks, all_violating=False)
    b = _run("none", 0.0, 0, picks)
    assert A.verdict_none_agreement([a, b])[0]["mixed_groups_in_verdict_run"] == 2
    a2 = _run("binary", 5.0, 0, picks, all_violating=True)
    assert A.verdict_none_agreement([a2, b])[0]["mixed_groups_in_verdict_run"] == 0


def test_the_advantage_gap_is_the_largest_one_seen_while_the_runs_still_agree():
    """This number is the cancellation's exactness on a trained run. Reading it past the point
    where the runs diverge would report a difference the verdict did not cause."""
    picks = [[1.0, 2.0]] * 3
    a = _run("binary", 5.0, 0, picks, advs=[[0.0, 0.0], [1e-9, 0.0], [0.0, 0.0]])
    b = _run("none", 0.0, 0, picks, advs=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    row = A.verdict_none_agreement([a, b])[0]
    assert row["max_adv_diff_while_identical"] == pytest.approx(1e-9)


def test_the_advantage_gap_stops_at_the_first_divergence():
    a = _run("binary", 5.0, 0, [[1.0], [2.0], [3.0]], advs=[[0.0], [0.0], [5.0]])
    b = _run("none", 0.0, 0, [[1.0], [2.0], [9.0]], advs=[[0.0], [0.0], [0.0]])
    row = A.verdict_none_agreement([a, b])[0]
    assert row["first_divergent_step"] == 3
    assert row["max_adv_diff_while_identical"] == 0.0, \
        "the third step's gap is after they diverged and must not be counted"
