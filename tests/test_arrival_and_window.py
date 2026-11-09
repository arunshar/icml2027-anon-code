"""The two quantities added to the analysis today, both of which the paper leads with.

`kl_at_arrival` is the divergence at the step where the evaluated violation rate first reaches
zero. The pricing section said it reported that and was reporting the divergence at the end of
training, which charges a design for every step it took after it had already arrived.

`window_last_step` and `window_groups_in_step1` count when the damage window closes. The abstract said
the window "falls entirely in the first optimizer step", which is true of one seed of three and
false of the others, where it reaches step four.

Both are now generated, so both need to be wrong loudly rather than quietly.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402


def _run(evals, trains):
    return {"evals": evals, "train": trains}


def test_arrival_is_the_first_zero_not_the_last():
    """A policy can leave the feasible set and come back. The cost of arriving is paid the first
    time, so a later zero must not overwrite it."""
    r = _run(
        [{"step": 0, "eval_violation_rate": 1.0}, {"step": 10, "eval_violation_rate": 0.0},
         {"step": 20, "eval_violation_rate": 0.5}, {"step": 30, "eval_violation_rate": 0.0}],
        [{"step": s, "kl_seq_mean": float(s)} for s in range(1, 31)],
    )
    out = A._kl_at_arrival(r)
    assert out["arrival_step"] == 10
    assert out["kl_at_arrival"] == 10.0


def test_a_run_that_never_arrives_reports_nothing_rather_than_a_number():
    """Returning the final divergence here would silently price an arm that never arrived as
    though it had, and it is exactly the arms that never arrive whose divergence is largest."""
    r = _run([{"step": s, "eval_violation_rate": 1.0} for s in (0, 10, 20)],
             [{"step": s, "kl_seq_mean": 99.0} for s in range(1, 21)])
    out = A._kl_at_arrival(r)
    assert out["arrival_step"] is None
    assert out["kl_at_arrival"] is None


def test_the_divergence_is_read_at_or_before_the_evaluation_that_found_zero():
    """Evaluation runs every ten steps, so the arrival step is an evaluation step. The divergence
    must come from a training record no later than it, never from a later one."""
    r = _run([{"step": 0, "eval_violation_rate": 1.0}, {"step": 10, "eval_violation_rate": 0.0}],
             [{"step": s, "kl_seq_mean": 5.0 if s <= 10 else 500.0} for s in range(1, 31)])
    assert A._kl_at_arrival(r)["kl_at_arrival"] == 5.0


def test_an_eval_with_no_violation_rate_does_not_count_as_an_arrival():
    r = _run([{"step": 0, "eval_violation_rate": None}, {"step": 10, "eval_violation_rate": None}],
             [{"step": 1, "kl_seq_mean": 1.0}])
    assert A._kl_at_arrival(r)["arrival_step"] is None


def _counter_run(groups_by_step, regime="A", arm="binary"):
    return {"cfg": {"regime": regime, "arm": arm, "w": 5.0, "seed": 0},
            "train": [{"step": s, "groups": g} for s, g in sorted(groups_by_step.items())]}


def test_the_window_records_when_it_closes_and_how_much_is_in_the_first_step():
    spread = {"rewards": [1.0, 2.0, 3.0, 4.0], "all_violating": True}
    tied = {"rewards": [1.0, 1.0, 1.0, 1.0], "all_violating": True}
    rows = A.group_counters([_counter_run({1: [spread, spread, tied], 2: [tied], 4: [spread],
                                           9: [tied]})])
    c = rows[0]
    assert c["all_violating_positive_variance"] == 3
    assert c["window_groups_in_step1"] == 2, "two of the three spread-carrying groups are in step 1"
    assert c["window_last_step"] == 4, "the last one is in step 4, not step 9"


def test_a_run_whose_window_is_empty_reports_zero_and_not_a_missing_field():
    tied = {"rewards": [2.0, 2.0, 2.0, 2.0], "all_violating": True}
    c = A.group_counters([_counter_run({1: [tied], 2: [tied]})])[0]
    assert c["all_violating_positive_variance"] == 0
    assert c["window_groups_in_step1"] == 0 and c["window_last_step"] == 0


def test_a_mixed_group_with_spread_is_not_in_the_window():
    """The window counts groups where a verdict could still have acted, which means groups that
    are all-violating. A mixed group already gives the verdict something to separate."""
    mixed = {"rewards": [1.0, 5.0, 2.0, 3.0], "all_violating": False}
    c = A.group_counters([_counter_run({1: [mixed], 7: [mixed]})])[0]
    assert c["all_violating_positive_variance"] == 0
    assert c["window_last_step"] == 0, "a mixed group must not extend the window"
