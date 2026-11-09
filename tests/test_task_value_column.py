"""The task-value column, and the rule that it never travels without its denominator.

A mean over "the completions that parsed" is a useful number and a dangerous one. If the parse
rate is not printed beside it, a run where one completion in a hundred parsed reads as though the
policy were doing well. These tests pin the exclusion rule, the JSON round-trip that lets a
capacity selection be scored after it has been through a log file, and the pairing of the two
columns in the emitted table.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402
import tasks  # noqa: E402

SPEED = tasks.get("speed")
CAP = tasks.get("capacity")


def test_a_selection_survives_the_round_trip_through_json():
    """metrics.jsonl writes a parsed selection as a list. The scoring functions index it
    positionally, so it has to come back as something they can read."""
    assert A._retuple([2, 0, 5]) == (2, 0, 5)
    assert A._retuple(37.0) == 37.0
    assert A._retuple(None) is None


def test_task_value_ignores_the_completions_that_did_not_parse():
    ev = {CAP.EVAL_ITEMS_KEY: [[0, 0, 1], None, [0, 2]]}
    got = A._mean_task_term(ev, CAP, 0.5)
    want = np.mean([CAP.task_term((0, 0, 1), 0.5), CAP.task_term((0, 2), 0.5)])
    assert got == pytest.approx(want)


def test_task_value_is_none_when_nothing_parsed_rather_than_zero():
    """Zero is a real task value: it is what the empty selection earns. A run where every
    completion failed to parse has no task value at all, and reporting that as 0.0 would put it
    in the same cell as a policy that answered and scored nothing."""
    assert A._mean_task_term({CAP.EVAL_ITEMS_KEY: [None, None]}, CAP, 0.5) is None
    assert A._mean_task_term({}, CAP, 0.5) is None


def test_task_value_works_on_the_frozen_speed_task_too():
    ev = {SPEED.EVAL_ITEMS_KEY: [5.0, None, 200.0]}
    got = A._mean_task_term(ev, SPEED, 5.0)
    assert got == pytest.approx(np.mean([SPEED.task_term(5.0, 5.0), SPEED.task_term(200.0, 5.0)]))


def test_the_table_prints_the_parse_rate_next_to_the_task_value():
    """Reading the source is the only way to pin this, and it is worth pinning: the two columns
    are only meaningful together."""
    src = open(os.path.join(HERE, "analyze.py")).read()
    per_run = src[src.index("def final_table"):]
    i_task = per_run.index('"eval_task_term_final"')
    i_fail = per_run.index('"eval_parse_fail_rate_final"')
    assert i_fail < i_task, "the parse rate must be emitted with the task value it qualifies"
    agg = per_run[per_run.index("# seed aggregate"):]
    assert '"task_term_final_mean"' in agg and '"parse_fail_final_mean"' in agg
