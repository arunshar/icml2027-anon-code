"""The 24 speed runs are published numbers, and this test is what keeps them published numbers.

The plan for this paper promised a frozen-result regression test and the repository did not have
one. In a single day `analyze.py` gained three functions, had a counter field renamed, and had its
matched-budget rule rewritten. Any of those could have moved a number the paper prints, and
nothing would have said so.

The values below were recorded on 2026-09-11, before any of that work, and are reproduced here
from the ledger's headline table rather than from today's analysis output. So this is a comparison
against an independent record and not a snapshot of whatever the code happens to emit now.

The runs are large and are not in the repository, so every test here skips when they are absent.
That keeps the shipped supplement's suite green without weakening the check where it matters.
"""
import csv
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

ANALYSIS = os.path.join(HERE, "runs", "430542", "analysis")
NO_RUNS = ("the frozen speed runs are not in this checkout, so there is nothing to regress "
           "against; run analyze.py on runs/430542 first")
requires_runs = pytest.mark.skipif(not os.path.isdir(ANALYSIS), reason=NO_RUNS)


def _rows(name):
    with open(os.path.join(ANALYSIS, name)) as f:
        return list(csv.DictReader(f))


def _by_seed(rows, regime, arm):
    got = [r for r in rows if r["regime"] == regime and r["arm"] == arm]
    return sorted(got, key=lambda r: int(r["seed"]))


# (regime, arm) -> (all_violating, zero_variance) per seed, recorded 2026-09-11
COUNTERS = {
    ("A", "graded"):       ([5, 4, 8],           [1777, 1777, 1778]),
    ("A", "binary"):       ([1799, 1797, 1799],  [1794, 1791, 1793]),
    ("A", "none"):         ([1799, 1798, 1799],  [1794, 1793, 1793]),
    ("A", "binary_1000x"): ([19, 4, 18],         [1778, 1787, 1763]),
}

# (regime, arm) -> (violation, final mean speed, KL) per seed, recorded 2026-09-11
FINALS = {
    ("A", "graded"):       ([0.0, 0.0, 0.0], [5.0, 5.0, 5.0],                     [6.89, 6.89, 7.78]),
    ("A", "binary"):       ([1.0, 1.0, 1.0], [918464.2, 608879.2, 633473.7],      [20.55, 22.23, 19.82]),
    ("A", "none"):         ([1.0, 1.0, 1.0], [908116.1, 485563.3, 633473.7],      [20.40, 22.41, 19.82]),
    ("A", "binary_1000x"): ([0.0, 0.0, 0.0], [5.0, 0.0, 2.0],                     [12.87, 17.82, 20.83]),
    ("B", "graded"):       ([0.0, 0.0, 0.0], [30.0, 30.0, 29.0],                  [24.25, 6.23, 32.32]),
    ("B", "binary"):       ([1.0, 1.0, 1.0], [858424.4, 412760.4, 100626.0],      [20.06, 16.15, 26.63]),
    ("B", "none"):         ([1.0, 1.0, 1.0], [908116.1, 485563.3, 633473.7],      [20.40, 22.41, 19.82]),
    ("B", "binary_1000x"): ([0.0, 0.0, 0.0], [30.0, 30.0, 0.0],                   [27.73, 20.62, 18.16]),
}


@requires_runs
@pytest.mark.parametrize("key", sorted(COUNTERS))
def test_the_group_counters_have_not_moved(key):
    """The all-violating column separates the designs and the zero-variance column does not.
    Both claims are counts, and the paper prints them, so both are pinned."""
    regime, arm = key
    want_av, want_zv = COUNTERS[key]
    rows = _by_seed(_rows("group_counters.csv"), regime, arm)
    assert [int(r["all_violating"]) for r in rows] == want_av
    assert [int(r["zero_variance"]) for r in rows] == want_zv
    assert all(int(r["groups"]) == 1800 for r in rows), "300 steps by 6 groups"


@requires_runs
@pytest.mark.parametrize("key", sorted(FINALS))
def test_the_final_evaluation_has_not_moved(key):
    regime, arm = key
    want_v, want_s, want_k = FINALS[key]
    rows = _by_seed(_rows("final_per_run.csv"), regime, arm)
    assert [float(r["eval_violation_rate_final"]) for r in rows] == pytest.approx(want_v)
    assert [float(r["eval_task_scalar_final"]) for r in rows] == pytest.approx(want_s, rel=1e-4)
    assert [float(r["kl_seq_mean_final"]) for r in rows] == pytest.approx(want_k, abs=5e-3)


@requires_runs
def test_the_no_penalty_arm_is_one_set_of_runs_read_at_both_limits():
    """The appendix says so and rests it on matching parameter checksums. The outputs match too,
    which is a stronger statement and the one a reader can check from the tables."""
    rows = _rows("final_per_run.csv")
    a = _by_seed(rows, "A", "none")
    b = _by_seed(rows, "B", "none")
    assert [r["eval_task_scalar_final"] for r in a] == [r["eval_task_scalar_final"] for r in b]
    assert [r["kl_seq_mean_final"] for r in a] == [r["kl_seq_mean_final"] for r in b]


@requires_runs
def test_the_zero_variance_count_does_not_depend_on_the_limit():
    """The same policy scored against two different limits gives the same zero-variance count,
    because that count asks whether the members agree and never asks what the constraint is. The
    appendix makes this point and it is the sharpest form of the paper's argument, so it is pinned
    rather than left to be noticed."""
    rows = _rows("group_counters.csv")
    a = _by_seed(rows, "A", "none")
    b = _by_seed(rows, "B", "none")
    assert [int(r["zero_variance"]) for r in a] == [int(r["zero_variance"]) for r in b]
    assert [int(r["all_violating"]) for r in a] != [int(r["all_violating"]) for r in b], \
        "the all-violating count must move with the limit, or the comparison proves nothing"


@requires_runs
def test_every_frozen_run_completed_its_full_budget():
    rows = _rows("final_per_run.csv")
    assert len(rows) == 24, f"expected 24 frozen speed runs, found {len(rows)}"
    assert {int(r["steps"]) for r in rows} == {300}
