"""Where a run stops producing a parseable answer for good.

This function shipped with a defect that a hand check could not have found. It reported the FIRST
step whose parse-failure rate reached 1.0, while its own docstring promised the step after which
the run never parses again. Three of the six collapsed capacity runs spike and recover, so three
of six rows described a step the run came back from, and every field anchored on that step moved
with it.

The one run the paper quotes collapses on its first failure, so reading that row by hand agreed
with the code. That is the trap these tests exist to close.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402
import tasks  # noqa: E402


class _Dir:
    def __init__(self, name):
        self.name = name


def _run(parse_rates, *, spreads=None, residuals=None, kls=None, members=None, arm="graded",
         regime="H", task="capacity"):
    """A run whose per-step parse-failure rate is given, with optional per-step extras.

    The final eval mirrors the last training step, because the function uses the last eval to
    decide whether a run collapsed at all. `members` is a per-step list of the single group's
    parsed completions, with None for a completion that did not parse. A step whose entry is
    None gets no item list at all, which is what a log written before that list existed looks
    like. The item list sits under the key the task logs it under, so a speed run carries speeds.
    """
    items_key = tasks.get(task).ITEMS_KEY
    n = len(parse_rates)
    train = []
    for i, pf in enumerate(parse_rates, start=1):
        sp = 0.0 if spreads is None else spreads[i - 1]
        group = {"rewards": [0.0, 0.0, 0.0, sp]}
        if members is not None and members[i - 1] is not None:
            group[items_key] = list(members[i - 1])
        train.append({
            "step": i,
            "parse_fail_rate": pf,
            "mean_residual": None if residuals is None else residuals[i - 1],
            "kl_seq_mean": float(i) if kls is None else kls[i - 1],
            "groups": [group],
        })
    return {"dir": _Dir("r"), "train": train,
            "evals": [{"step": n, "eval_parse_fail_rate": parse_rates[-1]}],
            "cfg": {"arm": arm, "w": 5.0, "w_task": 1.0, "regime": regime, "seed": 0,
                    "vmax": 0.03, "steps": n, "task": task},
            "prov": {}}


def test_a_spike_the_run_recovers_from_is_not_the_collapse():
    """The defect in one line. A run fails to parse at step 5, recovers at step 6, and dies at
    step 20. The collapse is step 20."""
    rates = [0.0] * 4 + [1.0] + [0.0] * 14 + [1.0]
    assert len(rates) == 20
    rows = A.unparseable_collapse([_run(rates)])
    assert len(rows) == 1
    assert rows[0]["collapse_step"] == 20, "the first failure at step 5 is not the collapse"


def test_several_spikes_still_give_the_last_unbroken_run_of_failures():
    rates = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    rows = A.unparseable_collapse([_run(rates)])
    assert rows[0]["collapse_step"] == 7


def test_a_run_that_never_parses_collapses_at_its_first_step():
    rows = A.unparseable_collapse([_run([1.0] * 6)])
    assert rows[0]["collapse_step"] == 1


def test_a_run_that_ends_parseable_is_not_reported_at_all():
    """A run that spikes and recovers and finishes fine has not collapsed, whatever it did in the
    middle."""
    assert A.unparseable_collapse([_run([0.0, 1.0, 1.0, 0.0, 0.0])]) == []


def test_the_fields_anchored_on_the_collapse_move_with_the_corrected_step():
    """This is what the defect actually cost. Every one of these read from the wrong step."""
    rates = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    resid = [9.0, 5.0, 4.0, 3.0, 0.0, None, None]
    kls = [1.0, 2.0, 3.0, 4.0, 7.0, 9.0, 9.5]
    spreads = [2.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0]
    rows = A.unparseable_collapse([_run(rates, residuals=resid, kls=kls, spreads=spreads)])
    r = rows[0]
    assert r["collapse_step"] == 6
    assert r["kl_before"] == 7.0 and r["kl_after"] == 9.0
    assert r["residual_before_collapse"] == 0.0, "the residual just before the real collapse"
    assert r["residual_first_step"] == 9.0
    assert r["last_spread_step"] == 5 and r["last_spread_value"] == pytest.approx(3.0)


def test_the_reported_step_is_never_one_the_run_came_back_from():
    """A property rather than a case: whatever the trace, no step after the reported collapse may
    have a parse-failure rate below 1.0."""
    for rates in ([0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0], [0.0] * 5 + [1.0],
                  [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]):
        rows = A.unparseable_collapse([_run(rates)])
        if not rows:
            continue
        c = rows[0]["collapse_step"]
        after = rates[c - 1:]
        assert all(v == 1.0 for v in after), \
            f"trace {rates} reported collapse at {c}, but it parses again afterwards"


def test_the_spike_group_reports_its_size_and_how_many_members_did_not_parse():
    """The paper says how many completions of the last group with any spread parsed. The count
    comes from the group with the widest reward range at that step, and from no other group."""
    rates = [0.0, 0.0, 0.0, 1.0, 1.0]
    spreads = [2.0, 0.0, 3.0, 0.0, 0.0]
    # One member of the spike group did not parse.
    members = [[[0], [1], [2], [0, 1]], None, [[0], None, [1], [0, 1]], None, None]
    r = A.unparseable_collapse([_run(rates, spreads=spreads, members=members)])[0]
    assert r["last_spread_step"] == 3
    assert r["last_spread_group_members"] == 4
    assert r["last_spread_group_unparseable"] == 1
    # Every member of the spike group parsed.
    members = [[[0], [1], [2], [0, 1]], None, [[0], [2], [1], [0, 1]], None, None]
    r = A.unparseable_collapse([_run(rates, spreads=spreads, members=members)])[0]
    assert r["last_spread_group_members"] == 4
    assert r["last_spread_group_unparseable"] == 0
    # The spike step holds a second, narrower group with more members and more failures. It is
    # not the group the row describes.
    run = _run(rates, spreads=spreads, members=members)
    run["train"][2]["groups"].append({"rewards": [0.0, 0.0, 0.0, 0.0, 1.0],
                                      "selections": [None, None, [0], [1], [2]]})
    r = A.unparseable_collapse([run])[0]
    assert r["last_spread_step"] == 3
    assert r["last_spread_group_members"] == 4
    assert r["last_spread_group_unparseable"] == 0
    # The wider group wins wherever it sits in the list.
    run = _run(rates, spreads=spreads, members=members)
    run["train"][2]["groups"].insert(0, {"rewards": [0.0, 0.0, 0.0, 0.0, 1.0],
                                         "selections": [None, None, [0], [1], [2]]})
    r = A.unparseable_collapse([run])[0]
    assert (r["last_spread_group_members"], r["last_spread_group_unparseable"]) == (4, 0)


def test_the_spike_group_reads_the_item_list_under_the_task_key():
    """A speed run logs its parsed items as speeds, and the count follows that key."""
    assert tasks.get("speed").ITEMS_KEY != tasks.get("capacity").ITEMS_KEY
    rates = [0.0, 0.0, 1.0, 1.0]
    members = [None, [3, None, 5, 4], None, None]
    r = A.unparseable_collapse([_run(rates, spreads=[0.0, 2.0, 0.0, 0.0], members=members,
                                     task="speed", regime="A")])[0]
    assert r["last_spread_step"] == 2
    assert r["last_spread_group_members"] == 4
    assert r["last_spread_group_unparseable"] == 1


def test_a_log_without_the_item_list_reports_no_spike_group_counts():
    """A log written before the per-group item list existed still has a spike step, and the
    two counts are None rather than zero, because zero would be a measurement."""
    rates = [0.0, 0.0, 1.0, 1.0]
    r = A.unparseable_collapse([_run(rates, spreads=[1.0, 2.0, 0.0, 0.0])])[0]
    assert r["last_spread_step"] == 2
    assert r["last_spread_group_members"] is None
    assert r["last_spread_group_unparseable"] is None


def test_a_run_with_no_spread_before_the_collapse_reports_no_spike_group_counts():
    """With no live spread there is no spike step, so there is no group to count, even when the
    log carries the item list."""
    rates = [0.0, 0.0, 1.0, 1.0]
    members = [[[0], None, [1], [2]]] * 4
    r = A.unparseable_collapse([_run(rates, members=members)])[0]
    assert r["last_spread_step"] is None
    assert r["last_spread_group_members"] is None
    assert r["last_spread_group_unparseable"] is None
