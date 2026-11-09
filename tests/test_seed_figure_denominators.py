"""Seed figures distinguish empty feasible answers from missing task-value observations."""
import numpy as np
import pytest

import analyze
import tasks


def run_with_evaluations(evaluations):
    return {"cfg": {"task": "capacity", "vmax": 0.03}, "evals": evaluations}


def test_empty_answers_have_zero_value_but_parser_collapse_breaks_the_curve():
    key = tasks.get("capacity").EVAL_ITEMS_KEY
    records = [
        {"step": 0, key: [[0], [1]], "eval_violation_rate": 0, "eval_parse_fail_rate": 0},
        {"step": 10, key: [None, None], "eval_violation_rate": 1, "eval_parse_fail_rate": 1},
    ]
    series = analyze.seed_evaluation_series(run_with_evaluations(records))
    assert series[0]["task_value"] == 0
    assert series[0]["parseable_completions"] == 2
    assert series[1]["task_value"] is None
    assert series[1]["parseable_completions"] == 0
    assert series[1]["parse_failure"] == 1


def test_partial_parsing_keeps_the_actual_denominator_and_recomputes_task_value():
    spec = tasks.get("capacity")
    record = {"step": 10, spec.EVAL_ITEMS_KEY: [[0, 0], None, [1]],
              "eval_violation_rate": 1 / 3, "eval_parse_fail_rate": 1 / 3}
    point = analyze.seed_evaluation_series(run_with_evaluations([record]))[0]
    assert point["evaluation_completions"] == 3
    assert point["parseable_completions"] == 2
    expected = np.mean([spec.task_term((0, 0), 0.03), spec.task_term((1,), 0.03)])
    assert point["task_value"] == pytest.approx(expected)


def test_a_false_parse_denominator_stops_figure_generation():
    spec = tasks.get("capacity")
    record = {"step": 10, spec.EVAL_ITEMS_KEY: [None, [0]],
              "eval_violation_rate": 0.5, "eval_parse_fail_rate": 0}
    with pytest.raises(ValueError, match="disagrees"):
        analyze.seed_evaluation_series(run_with_evaluations([record]))
