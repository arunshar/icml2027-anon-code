"""These tests cover the scale-matched residual in the analysis and in its two manifests.

The design trains the graded arm at the weight whose first-violation penalty matches the light
verdict's penalty at the hard level of each task. On the speed task the match is exact. On the capacity
task the weight uses the smallest capacity, so the penalty equals the light verdict's on three
instances and stays below it on the other three. Each test pins one fact that the design or its
drafted prediction entry relies on. Each expected value is recomputed here from the reward code, and
the tests compare that value with what make_manifests.py and analyze.py hold.
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402
import capacity_task as C  # noqa: E402
import make_manifests as M  # noqa: E402
import rewards as RW  # noqa: E402
import train_speed_grpo as T  # noqa: E402

SBATCH = os.path.join(HERE, "slurm", "cancel_arms.sbatch")
requires_sbatch = pytest.mark.skipif(
    not os.path.exists(SBATCH),
    reason="the cluster submission script is not shipped in the anonymous supplement, because it "
           "names a compute account and a site-specific scratch path")
SPEED_MANIFEST = os.path.join(HERE, "slurm", "manifest_speed_matched.csv")
CAPACITY_MANIFEST = os.path.join(HERE, "slurm", "manifest_capacity_matched.csv")
MANIFESTS = {"speed": SPEED_MANIFEST, "capacity": CAPACITY_MANIFEST}
SPEED_RUNS = os.path.join(HERE, "runs", "speed_matched")
CAPACITY_RUNS = os.path.join(HERE, "runs", "capacity_matched")
# Prediction M0 in the drafted entry uses this tolerance.
IDENTITY_TOL = 1e-6
# Two first-violation penalties closer than this count as equal.
PENALTY_TOL = 1e-9
LIGHT_W = A.ARM_SPECS["binary"][1]
TRAINED_W = A.ARM_SPECS["graded"][1]


def _vmax():
    return float(yaml.safe_load(open(os.path.join(HERE, "configs", "regimes.yaml")))["A"]["v_max"])


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _legacy_capacity_rows():
    return _rows(os.path.join(HERE, "slurm", "manifest_capacity.csv"))


def _ratio():
    """Return the hard capacity level, read from the committed capacity manifest."""
    ratios = {r["vmax"] for r in _legacy_capacity_rows() if r["regime"] == "H"}
    assert len(ratios) == 1
    return float(ratios.pop())


def _factor():
    v = _vmax()
    return -float(RW.pen_graded(2.0 * v, v))


def _capacities():
    return [C.capacity_of(inst, _ratio()) for inst in C.INSTANCES]


def _speed_table():
    return RW.speed_output_table(_vmax())


def _capacity_tables():
    return [C.output_table(_ratio(), k) for k in range(len(C.INSTANCES))]


def _table_rewards(table, arm, w):
    """Return the reward of every output in a table at task weight 1, through the reward code."""
    return np.asarray(table["task"]) + w * RW._table_penalty(table, arm)


def _penalties(table, w):
    """Return the residual's penalty magnitudes at weight w, sorted and without repeats or zeros."""
    p = np.unique(np.round(-w * RW._table_penalty(table, "graded"), 9))
    return p[p > 0]


def _first_penalty(table, w):
    return float(_penalties(table, w)[0])


def _crossover(table):
    """Return the verdict weight above which every feasible output outranks every violator."""
    task = np.asarray(table["task"])
    ok = np.asarray(table["verdict"])
    return float(task[~ok].max() - task[ok].min())


def _ordering_threshold(table):
    """Return the residual weight above which every feasible output outranks every violator."""
    task = np.asarray(table["task"])
    ok = np.asarray(table["verdict"])
    penalty = -RW._table_penalty(table, "graded")
    return float(((task[~ok] - task[ok].min()) / penalty[~ok]).max())


def _implied_v2(row, task):
    """Return a committed v1 row as the v2 row the submission script and the trainer give it.

    A v1 speed row carries no task weight and no limit, and the script sets them to 1.0 and the
    regime's limit. No v1 row carries the three remedy flags, so each takes the trainer's default."""
    defaults = vars(T.build_arg_parser().parse_args(
        ["--arm", "graded", "--w", "5", "--vmax", "5", "--out-dir", "unused"]))
    out = dict(row, task=task, adv_variant=defaults["adv_variant"],
               group_filter=defaults["group_filter"], kl_beta=repr(float(defaults["kl_beta"])))
    if task == "speed":
        out.update(w_task="1.0", vmax=repr(_vmax()))
    return out


def _source_rows(task, arm_name):
    """Return the committed hard-level rows of one arm, as v2 rows keyed by seed."""
    if task == "speed":
        rows = [r for r in _rows(os.path.join(HERE, "slurm", "manifest.csv"))
                if r["regime"] == "A" and r["arm_name"] == arm_name]
    else:
        rows = [r for r in _legacy_capacity_rows() if r["regime"] == "H" and r["arm_name"] == arm_name]
    return {r["seed"]: _implied_v2(r, task) for r in rows}


# ---------------------------------------------------------------- the arm's name and label


def test_the_graded_arm_at_the_matched_weight_is_named_graded_matched_on_both_tasks():
    cfg = {"arm": "graded", "w": 0.25, "w_task": 1.0}
    assert A.arm_name(cfg) == "graded_matched"
    assert A.arm_name(dict(cfg, task="speed")) == "graded_matched"
    assert A.arm_name(dict(cfg, task="capacity")) == "graded_matched"
    assert A.NAMED_WEIGHTS["graded_matched"] == ("graded", A.GRADED_MATCHED_W, 1.0)
    assert A.GRADED_MATCHED_W == 0.25


def test_the_name_stays_out_of_the_arm_specs_and_every_other_weight_keeps_its_name():
    assert "graded_matched" not in A.ARM_SPECS and "graded_matched" not in A.CAPACITY_ARM_SPECS
    assert A.arm_name({"arm": "graded", "w": 25}) == "graded_w25_t1"
    assert A.arm_name({"arm": "graded", "w": 5.0}) == "graded"
    assert A.arm_name({"arm": "graded", "w": 0.01, "w_task": 1.0}) == "graded_below"
    assert A.arm_name({"arm": "binary", "w": 0.25}) == "binary_w0.25_t1"
    assert A.arm_name({"arm": "graded", "w": 0.25, "w_task": 0.0, "task": "capacity"}) == \
        "graded_w0.25_t0"
    assert A.arm_name({"arm": "graded", "w": 0.3, "task": "capacity"}) == "graded_w0.3_t1"


def test_a_remedy_flag_adds_its_suffix_to_the_new_name():
    cfg = {"arm": "graded", "w": 0.25, "w_task": 1.0, "task": "capacity"}
    assert A.arm_name(dict(cfg, group_filter="verdict_mixed")) == "graded_matched_vfilter"
    assert A.arm_name(dict(cfg, kl_beta=0.04)) == "graded_matched_kl"


def test_the_arm_label_names_the_matched_weight():
    cfg = {"arm": "graded", "w": 0.25, "w_task": 1.0, "task": "capacity"}
    label = A.compose_arm_label(cfg)
    assert label == r"residual $+$ task, matched weight"
    assert A._plain_label(label) == "residual + task, matched weight"
    assert A.NAMED_LABELS["graded_matched"] == label
    published = A.arm_labels([{"cfg": cfg}])
    assert published["graded_matched"] == label
    assert published["graded"] == "residual $+$ task"


def test_the_new_manifests_are_not_training_sources():
    """The design leaves the training-source digest alone, and config.json records the weight."""
    for name in ("manifest_speed_matched.csv", "manifest_capacity_matched.csv"):
        assert not any(s.endswith(name) for s in A.TRAINING_SOURCES)
        assert not any(s.endswith(name) for s in T.TRAINING_SOURCES)


# ---------------------------------------------------------------- the weight


def test_the_weight_is_the_closed_form_value_on_both_tasks():
    v, factor = _vmax(), _factor()
    assert (v, factor) == (5.0, 100.0)
    assert _capacities() == [6, 6, 5, 5, 6, 5]
    assert LIGHT_W * v / factor == A.GRADED_MATCHED_W
    assert LIGHT_W * min(_capacities()) / factor == A.GRADED_MATCHED_W
    # Dividing by the smallest residual instead leaves an error in the last digit, which would reach
    # config.json and the arm name. That is why the generator uses the closed form.
    assert LIGHT_W / (factor * float(RW.residual(6.0, v))) != A.GRADED_MATCHED_W
    assert M._matched_weight("speed", "5.0", v, v) == A.GRADED_MATCHED_W
    assert M._matched_weight("capacity", "5.0", _ratio(), float(min(_capacities()))) == \
        A.GRADED_MATCHED_W


def test_the_capacity_penalty_uses_the_same_factor_as_the_speed_penalty():
    ratio = _ratio()
    for sel in [(1, 0, 1, 2), (5,) + tuple(range(C.N_ITEMS)), (0, 2)]:
        assert RW._spec_penalty(sel, "graded", ratio, C.SPEC) == \
            pytest.approx(-_factor() * C.residual(sel, ratio))


def test_the_smallest_violation_is_one_unit_on_the_speed_task_and_every_capacity_instance():
    speed = np.asarray(_speed_table()["residual"])
    assert float(speed[speed > 0].min()) * _vmax() == pytest.approx(1.0)
    for table, cap in zip(_capacity_tables(), _capacities()):
        res = np.asarray(table["residual"])
        assert float(res[res > 0].min()) * cap == pytest.approx(1.0)


def test_the_first_violation_penalty_matches_the_light_verdict_and_never_exceeds_it():
    w = A.GRADED_MATCHED_W
    speed = _first_penalty(_speed_table(), w)
    assert speed == pytest.approx(LIGHT_W, abs=PENALTY_TOL)
    capacity = [_first_penalty(t, w) for t in _capacity_tables()]
    assert capacity == pytest.approx([25 / 6, 25 / 6, 5.0, 5.0, 25 / 6, 5.0], abs=PENALTY_TOL)
    assert [k for k, p in enumerate(capacity) if abs(p - LIGHT_W) <= PENALTY_TOL] == [2, 3, 5]
    assert all(p <= LIGHT_W + PENALTY_TOL for p in [speed] + capacity)
    assert M._first_violation_penalties("speed", w, _vmax()) == pytest.approx([speed])
    assert M._first_violation_penalties("capacity", w, _ratio()) == pytest.approx(capacity)


def test_the_first_violation_penalties_sit_below_every_crossover():
    w = A.GRADED_MATCHED_W
    speed_cross = _crossover(_speed_table())
    assert speed_cross == 20.0
    assert _first_penalty(_speed_table(), w) < speed_cross
    crossovers = [_crossover(t) for t in _capacity_tables()]
    assert max(crossovers) == pytest.approx(12.5)
    assert int(np.argmax(crossovers)) == 1
    for table, cross in zip(_capacity_tables(), crossovers):
        assert _first_penalty(table, w) < cross


def test_the_match_holds_at_the_first_violation_unit_only():
    """The penalty passes the light verdict's from the second unit and later reaches the crossover."""
    w = A.GRADED_MATCHED_W
    s = np.arange(0, 1001, dtype=np.float64)
    penalty = -w * RW.pen_graded(s, _vmax())
    assert penalty[7] == pytest.approx(10.0)
    assert int(s[penalty >= 20.0 - PENALTY_TOL][0]) == 9
    assert penalty[10] == pytest.approx(25.0)
    for table, cap in zip(_capacity_tables(), _capacities()):
        p = _penalties(table, w)
        assert p[1] == pytest.approx(2 * LIGHT_W * min(_capacities()) / cap)
        assert p[1] > LIGHT_W
        reach = float(p[p >= 12.5 - PENALTY_TOL].min())
        assert reach == pytest.approx(12.5 if cap == 6 else 15.0)


def test_the_generator_refuses_a_weight_that_analyze_does_not_name(monkeypatch):
    named = dict(A.NAMED_WEIGHTS, graded_matched=("graded", 0.3, 1.0))
    monkeypatch.setattr(A, "NAMED_WEIGHTS", named)
    with pytest.raises(ValueError, match="analyze.py names"):
        M.speed_matched()
    with pytest.raises(ValueError, match="analyze.py names"):
        M.capacity_matched()


def test_the_generator_refuses_a_penalty_above_the_light_verdicts_or_one_that_never_equals_it(
        monkeypatch):
    monkeypatch.setattr(M, "_first_violation_penalties", lambda task, w, limit: [5.0, 6.0])
    with pytest.raises(ValueError, match="exceeds the light"):
        M.capacity_matched()
    monkeypatch.setattr(M, "_first_violation_penalties", lambda task, w, limit: [4.0])
    with pytest.raises(ValueError, match="equals the light"):
        M.speed_matched()


def test_the_alternative_capacity_weight_would_exceed_the_light_verdict_on_three_instances():
    """Decision 1's alternative matches the capacity-6 instances and overshoots the others."""
    alt = LIGHT_W * max(_capacities()) / _factor()
    assert alt == pytest.approx(0.3)
    pens = M._first_violation_penalties("capacity", alt, _ratio())
    assert pens == pytest.approx([5.0, 5.0, 6.0, 6.0, 5.0, 6.0])


# ---------------------------------------------------------------- what the weight does to the ordering


def test_the_matched_residual_ranks_no_violator_above_a_feasible_answer_on_either_task():
    w = A.GRADED_MATCHED_W
    for table in [_speed_table()] + _capacity_tables():
        r = _table_rewards(table, "graded", w)
        ok = np.asarray(table["verdict"])
        assert r[~ok].max() < r[ok].min()


def test_the_light_verdict_ranks_violators_above_the_best_feasible_answer_on_both_tasks():
    table = _speed_table()
    r = _table_rewards(table, "binary", LIGHT_W)
    ok = np.asarray(table["verdict"])
    s = np.arange(r.size, dtype=np.float64)
    above = s[(~ok) & (r > r[ok].max())]
    assert above.min() == 56.0 and above.size == 999_944
    counts = []
    for table in _capacity_tables():
        r = _table_rewards(table, "binary", LIGHT_W)
        ok = np.asarray(table["verdict"])
        counts.append(int(np.sum(r[~ok] > r[ok].max())))
    assert counts == [5_901, 35_845, 43, 7_997, 7_468, 28_689]


def test_no_weight_matches_the_first_violation_penalty_and_sits_below_the_ordering_threshold():
    speed = _ordering_threshold(_speed_table())
    assert speed == pytest.approx(0.03, abs=1e-12)
    capacity = [_ordering_threshold(t) for t in _capacity_tables()]
    assert [round(x, 4) for x in capacity] == [0.096, 0.06, 0.0298, 0.0432, 0.0847, 0.0361]
    assert A.GRADED_MATCHED_W > max([speed] + capacity)
    # The weight that matches the light verdict exactly on one capacity instance is 0.25 or 0.30, and
    # both exceed every instance's ordering threshold.
    for cap in _capacities():
        assert LIGHT_W * cap / _factor() > max(capacity)
    assert LIGHT_W * _vmax() / _factor() > speed


# ---------------------------------------------------------------- the identity on the speed task


# Every group is all-violating, every member parses, and every magnitude lies on one side of the task
# cap of 200. The third group has the smallest nonzero spread below the cap and the eighth has it
# above the cap, so they give the largest gap the normalizer's 1e-6 term can open.
IDENTITY_GROUPS = [
    (6.0, 7.0, 8.0, 9.0),
    (10.0, 20.0, 30.0, 60.0),
    (6.0, 6.0, 6.0, 7.0),
    (6.0, 50.0, 120.0, 199.0),
    (200.0, 400.0, 999_999.0, 1_000.0),
    (250.0, 300.0, 1_000.0, 205.0),
    (6.0, 6.0, 6.0, 6.0),
    (201.0, 201.0, 201.0, 202.0),
]


@pytest.mark.parametrize("speeds", IDENTITY_GROUPS)
def test_on_one_side_of_the_cap_the_advantages_equal_the_trained_residuals(speeds):
    v = _vmax()
    lo = RW.advantage(RW.group_rewards(list(speeds), "graded", A.GRADED_MATCHED_W, v))
    hi = RW.advantage(RW.group_rewards(list(speeds), "graded", TRAINED_W, v))
    assert np.abs(lo - hi).max() <= IDENTITY_TOL


def test_a_group_that_straddles_the_cap_breaks_the_identity_but_keeps_its_ranking():
    v = _vmax()
    speeds = [30.0, 150.0, 400.0, 900.0]
    lo = RW.advantage(RW.group_rewards(speeds, "graded", A.GRADED_MATCHED_W, v))
    hi = RW.advantage(RW.group_rewards(speeds, "graded", TRAINED_W, v))
    assert np.abs(lo - hi).max() > IDENTITY_TOL
    assert list(np.argsort(-lo)) == list(np.argsort(-hi))


@pytest.mark.parametrize("speeds", IDENTITY_GROUPS[:6])
def test_in_the_same_groups_the_light_verdict_gives_the_task_only_advantages(speeds):
    v = _vmax()
    light = RW.advantage(RW.group_rewards(list(speeds), "binary", LIGHT_W, v))
    task_only = RW.advantage(RW.group_rewards(list(speeds), "none", 0.0, v))
    assert np.abs(light - task_only).max() <= 1e-9


# ---------------------------------------------------------------- the unparseable fallback


def test_the_fallback_reward_with_no_parsed_member_under_each_arm_and_task():
    v, ratio = _vmax(), _ratio()
    assert RW.group_rewards([None] * 4, "binary", LIGHT_W, v)[0] == -15.0
    assert RW.group_rewards([None] * 4, "graded", A.GRADED_MATCHED_W, v)[0] == pytest.approx(-35.0)
    assert RW.group_rewards([None] * 4, "graded", TRAINED_W, v)[0] == pytest.approx(-510.0)
    cap = [RW.group_rewards([None] * 4, arm, w, ratio, spec=C.SPEC)[0]
           for arm, w in (("binary", LIGHT_W), ("graded", A.GRADED_MATCHED_W), ("graded", TRAINED_W))]
    assert cap == pytest.approx([-15.0, -10.0 - 175.0 / 3.0, -10.0 - 3500.0 / 3.0])


def test_beside_parsed_members_the_fallback_is_still_deeper_under_the_matched_residual():
    v, ratio = _vmax(), _ratio()
    assert RW.group_rewards([None, 10.0, 20.0, 30.0], "graded", A.GRADED_MATCHED_W, v)[0] == \
        pytest.approx(-135.0)
    assert RW.group_rewards([None, 10.0, 20.0, 30.0], "binary", LIGHT_W, v)[0] == -15.0
    take_all = (1,) + tuple(range(C.N_ITEMS))
    assert RW.group_rewards([None, take_all], "graded", A.GRADED_MATCHED_W, ratio,
                            spec=C.SPEC)[0] == pytest.approx(-772.5)
    assert RW.group_rewards([None, take_all], "binary", LIGHT_W, ratio, spec=C.SPEC)[0] == -15.0


def test_an_unparseable_member_outranks_instance_fives_take_all_at_either_residual_weight():
    ratio = _ratio()
    take_all = (5,) + tuple(range(C.N_ITEMS))
    assert C.residual(take_all, ratio) == pytest.approx(32.2)
    assert C.residual(C.synthesize(1e6, ratio), ratio) == pytest.approx(31.0)
    for w in (A.GRADED_MATCHED_W, TRAINED_W):
        fallback, parsed = RW.group_rewards([None, take_all], "graded", w, ratio, spec=C.SPEC)
        assert fallback > parsed
    fallback, parsed = RW.group_rewards([None, take_all], "graded", A.GRADED_MATCHED_W, ratio,
                                        spec=C.SPEC)
    assert (fallback, parsed) == pytest.approx((-785.0, -793.4444444444))
    fallback, parsed = RW.group_rewards([None, take_all], "binary", LIGHT_W, ratio, spec=C.SPEC)
    assert fallback < parsed


# ---------------------------------------------------------------- the manifests


def test_the_speed_manifest_holds_three_seeds_of_the_named_arm():
    rows = _rows(SPEED_MANIFEST)
    assert tuple(rows[0].keys()) == M.V2_HEADER
    assert [int(r["task_id"]) for r in rows] == [1, 2, 3]
    assert [int(r["seed"]) for r in rows] == [0, 1, 2]
    for r in rows:
        assert (r["task"], r["regime"], r["arm_name"], r["arm"]) == ("speed", "A", "graded_matched",
                                                                      "graded")
        assert r["w"] == "0.25" and float(r["w"]) == A.GRADED_MATCHED_W
        assert r["w_task"] == "1.0" and float(r["vmax"]) == _vmax()
        assert (r["adv_variant"], r["group_filter"], float(r["kl_beta"])) == ("grpo", "none", 0.0)


def test_the_capacity_manifest_holds_five_seeds_of_the_named_arm():
    rows = _rows(CAPACITY_MANIFEST)
    assert tuple(rows[0].keys()) == M.V2_HEADER
    assert [int(r["task_id"]) for r in rows] == [1, 2, 3, 4, 5]
    assert [int(r["seed"]) for r in rows] == [0, 1, 2, 3, 4]
    for r in rows:
        assert (r["task"], r["regime"], r["arm_name"], r["arm"]) == ("capacity", "H",
                                                                      "graded_matched", "graded")
        assert r["w"] == "0.25" and float(r["w"]) == A.GRADED_MATCHED_W
        assert r["w_task"] == "1.0" and float(r["vmax"]) == _ratio()
        assert (r["adv_variant"], r["group_filter"], float(r["kl_beta"])) == ("grpo", "none", 0.0)


@pytest.mark.parametrize("task", ["speed", "capacity"])
def test_the_manifests_are_lf_only_numbered_and_trimmed(task):
    raw = open(MANIFESTS[task], "rb").read()
    assert b"\r" not in raw and raw.endswith(b"\n")
    rows = _rows(MANIFESTS[task])
    assert [int(r["task_id"]) for r in rows] == list(range(1, len(rows) + 1))
    for r in rows:
        assert all(v == v.strip() and v != "" for v in r.values()), r


@pytest.mark.parametrize("task", ["speed", "capacity"])
def test_every_row_equals_the_residual_row_of_its_seed_except_weight_and_name(task):
    """The weight and the name are the only differences from the trained residual runs."""
    source = _source_rows(task, "graded")
    rows = _rows(MANIFESTS[task])
    assert sorted(source) == sorted(r["seed"] for r in rows)
    changed = ("task_id", "w", "arm_name")
    for r in rows:
        base = source[r["seed"]]
        assert (base["arm"], float(base["w"])) == ("graded", TRAINED_W)
        assert {k: v for k, v in r.items() if k not in changed} == \
            {k: base[k] for k in r if k not in changed}
        assert r["w"] != base["w"] and r["arm_name"] != base["arm_name"]


@pytest.mark.parametrize("task", ["speed", "capacity"])
def test_every_row_equals_the_light_verdict_row_of_its_seed_except_arm_weight_and_name(task):
    source = _source_rows(task, "binary")
    rows = _rows(MANIFESTS[task])
    assert sorted(source) == sorted(r["seed"] for r in rows)
    changed = ("task_id", "arm", "w", "arm_name")
    for r in rows:
        base = source[r["seed"]]
        assert (base["arm"], float(base["w"])) == ("binary", LIGHT_W)
        assert {k: v for k, v in r.items() if k not in changed} == \
            {k: base[k] for k in r if k not in changed}


def test_the_speed_rows_match_the_fingerprint_row_except_arm_weight_and_name():
    """The fingerprint row is the light verdict's seed-0 run in the v2 layout, written by the generator."""
    fingerprint = _rows(os.path.join(HERE, "slurm", "manifest_fingerprint_speed.csv"))[0]
    row = next(r for r in _rows(SPEED_MANIFEST) if r["seed"] == fingerprint["seed"])
    changed = ("arm", "w", "arm_name")
    assert {k: v for k, v in row.items() if k not in changed} == \
        {k: v for k, v in fingerprint.items() if k not in changed}


@pytest.mark.parametrize("task", ["speed", "capacity"])
def test_the_analysis_names_every_row_as_the_row_names_itself(task):
    for r in _rows(MANIFESTS[task]):
        cfg = {"task": r["task"], "arm": r["arm"], "w": float(r["w"]), "w_task": float(r["w_task"]),
               "adv_variant": r["adv_variant"], "group_filter": r["group_filter"],
               "kl_beta": float(r["kl_beta"])}
        assert A.arm_name(cfg) == r["arm_name"] == "graded_matched"


@pytest.mark.parametrize("task", ["speed", "capacity"])
def test_the_trainer_accepts_every_row(task):
    for r in _rows(MANIFESTS[task]):
        argv = ["--task", r["task"], "--arm", r["arm"], "--w", r["w"], "--w-task", r["w_task"],
                "--vmax", r["vmax"], "--seed", r["seed"], "--regime", r["regime"],
                "--adv-variant", r["adv_variant"], "--group-filter", r["group_filter"],
                "--kl-beta", r["kl_beta"], "--out-dir", "unused"]
        ap = T.build_arg_parser()
        args = ap.parse_args(argv)
        T.check_args(ap, args)
        assert args.w == A.GRADED_MATCHED_W and args.arm == "graded" and args.task == task


def test_the_generator_writes_the_committed_files():
    built = M.build_all()
    assert built["slurm/manifest_speed_matched.csv"] == open(SPEED_MANIFEST).read()
    assert built["slurm/manifest_capacity_matched.csv"] == open(CAPACITY_MANIFEST).read()
    assert M.main(["--check"]) == 0


def _parse_block():
    src = open(SBATCH).read()
    assert src.count("# >>> manifest-parse") == 1 and src.count("# <<< manifest-parse") == 1
    return src.split("# >>> manifest-parse")[1].split("# <<< manifest-parse")[0]


def _parse(manifest, task_id, task):
    script = "\n".join([
        "set -uo pipefail",
        f"SLURM_ARRAY_TASK_ID={task_id}", f"MANIFEST='{manifest}'", f"CANCEL_TASK={task}",
        f"PY='{sys.executable}'", f"EXP='{HERE}'",
        _parse_block(),
        'printf "%s|%s|%s|%s|%s|%s|%s|%s|%s\\n" "$TASK_ID" "$REGIME" "$ARM_NAME" "$ARM" "$W" '
        '"$W_TASK" "$VMAX" "$SEED" "$REMEDY_FLAGS"',
    ])
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


@requires_sbatch
@pytest.mark.parametrize("task", ["speed", "capacity"])
def test_the_submission_scripts_v2_branch_parses_every_row(task):
    for r in _rows(MANIFESTS[task]):
        p = _parse(MANIFESTS[task], r["task_id"], task)
        assert p.returncode == 0, p.stderr
        fields = p.stdout.strip().split("|")
        assert fields[:8] == [r["task_id"], r["regime"], r["arm_name"], r["arm"], r["w"], r["w_task"],
                              r["vmax"], r["seed"]]
        assert fields[8] == "--adv-variant grpo --group-filter none --kl-beta 0.0"


@requires_sbatch
@pytest.mark.parametrize("task,other", [("speed", "capacity"), ("capacity", "speed")])
def test_the_submission_script_refuses_each_manifest_for_the_other_task(task, other):
    p = _parse(MANIFESTS[task], "1", other)
    assert p.returncode == 2 and "REFUSED" in p.stderr


# ---------------------------------------------------------------- completeness of the run directories
# The directories exist only after the runs are pulled and assembled. A seed that did not finish must
# be named in the not_run list of the directory's assembled.json, so an undeclared gap fails here.


def _check_assembled(root, task, seeds):
    record = json.load(open(os.path.join(root, "assembled.json")))
    runs = record["runs"]
    assert {e["arm"] for e in runs} == {"graded", "binary", "graded_matched"}
    finished = [int(e["seed"]) for e in runs if e["arm"] == "graded_matched"]
    declared = [int(x["seed"]) if isinstance(x, dict) else int(x) for x in record.get("not_run", [])]
    assert len(set(finished)) == len(finished), "a matched seed appears twice"
    assert not set(finished) & set(declared), "a seed is both finished and listed as not run"
    assert set(finished) | set(declared) == set(seeds)
    for e in runs:
        cfg = json.load(open(os.path.join(root, e["link"], "config.json")))
        assert A.arm_name(dict({"task": task, "w_task": 1.0}, **cfg)) == e["arm"]


@pytest.mark.skipif(not os.path.isdir(SPEED_RUNS), reason="runs/speed_matched has not been assembled")
def test_the_assembled_speed_matched_directory_is_complete():
    _check_assembled(SPEED_RUNS, "speed", [int(r["seed"]) for r in _rows(SPEED_MANIFEST)])


@pytest.mark.skipif(not os.path.isdir(CAPACITY_RUNS),
                    reason="runs/capacity_matched has not been assembled")
def test_the_assembled_capacity_matched_directory_is_complete():
    _check_assembled(CAPACITY_RUNS, "capacity", [int(r["seed"]) for r in _rows(CAPACITY_MANIFEST)])
