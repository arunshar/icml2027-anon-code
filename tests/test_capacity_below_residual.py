"""These tests cover the capacity residual below its ordering threshold, in the analysis and the manifest.

The design trains the graded arm at weight 0.04 at the capacity task's hard level. At that weight the
best feasible answer outranks every violator on every instance, and adding any item to an overfull
selection lowers the reward. The residual no longer ranks every feasible answer above every violator on
instances 0, 1, 3 and 4, and it still does on instances 2 and 5. Each test pins one fact that the
design or its drafted prediction entry relies on. The bounds are recomputed here with exact fractions
from the instance literals, and the tests compare them with what make_manifests.py and analyze.py hold.
"""
import csv
import json
import os
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

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
MANIFEST = os.path.join(HERE, "slurm", "manifest_capacity_below.csv")
BELOW_RUNS = os.path.join(HERE, "runs", "capacity_below")
LIGHT_W = A.ARM_SPECS["binary"][1]
TRAINED_W = A.ARM_SPECS["graded"][1]
W_EXACT = Fraction(1, 25)
DROPPED = (0, 1, 3, 4)

# Each list holds one exact bound per instance, in the order of the instances. The lists are the
# plan's table of Section 2.1.
THRESHOLDS = [Fraction(12, 125), Fraction(3, 50), Fraction(31, 1040), Fraction(19, 440),
              Fraction(36, 425), Fraction(13, 360)]
FEASIBLE_TOP = [Fraction(9, 250), Fraction(9, 700), Fraction(3, 520), Fraction(3, 440),
                Fraction(21, 850), Fraction(1, 90)]
INCLUSION = [Fraction(1, 50), Fraction(3, 250), Fraction(19, 1040), Fraction(1, 88),
             Fraction(1, 50), Fraction(1, 90)]
PAIRWISE = [Fraction(141, 500), Fraction(243, 700), Fraction(9, 52), Fraction(37, 220),
            Fraction(273, 850), Fraction(101, 360)]
# This map lists, per instance, the violating selections that earn more than the empty selection at
# weight 1/25.
ABOVE_EMPTY = {0: [(0, 10), (0, 12)], 1: [(0,)], 2: [], 3: [(14, 15)], 4: [(9, 10)], 5: []}


def _rows(path=MANIFEST):
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
    return -float(RW.pen_graded(20.0, 10.0))


def _instance(k):
    """Return the capacity, the optimum and the integer weight and value of every subset."""
    inst = C.INSTANCES[k]
    cap = C.capacity_of(inst, _ratio())
    opt = C.dp_optimum(inst["values"], inst["weights"], cap)
    weights = C._SUBSET_BITS @ np.asarray(inst["weights"], dtype=np.int64)
    values = C._SUBSET_BITS @ np.asarray(inst["values"], dtype=np.int64)
    return cap, opt, weights, values


def _levels(k):
    """Return the largest and smallest value at each total weight above the capacity."""
    cap, _, weights, values = _instance(k)
    levels = sorted(int(x) for x in np.unique(weights[weights > cap]))
    vmax = {lv: int(values[weights == lv].max()) for lv in levels}
    vmin = {lv: int(values[weights == lv].min()) for lv in levels}
    return levels, vmax, vmin


def _exact_bounds(k):
    """Return the ordering threshold, feasible-top, inclusion and pairwise bounds as fractions."""
    cap, opt, _, _ = _instance(k)
    factor = int(_factor())
    levels, vmax, vmin = _levels(k)
    threshold = max(Fraction(vmax[lv] * cap, factor * opt * (lv - cap)) for lv in levels)
    top = max([Fraction((vmax[lv] - opt) * cap, factor * opt * (lv - cap))
               for lv in levels if vmax[lv] > opt] + [Fraction(0)])
    inst = C.INSTANCES[k]
    inclusion = max(Fraction(v * cap, factor * opt * w) for v, w in zip(inst["values"], inst["weights"]))
    pair = Fraction(0)
    for i, low in enumerate(levels):
        for high in levels[i + 1:]:
            if vmax[high] > vmin[low]:
                pair = max(pair, Fraction((vmax[high] - vmin[low]) * cap, factor * opt * (high - low)))
    return threshold, top, inclusion, pair


def _scaled_rewards(k, w=W_EXACT):
    """Return the reward times opt * cap * denominator(w) of every subset, as exact integers.

    The reward of a parsed selection is V / opt - w * factor * max(W - cap, 0) / cap. Scaling by the
    positive integer opt * cap * denominator(w) keeps every comparison exact."""
    cap, opt, weights, values = _instance(k)
    factor = int(_factor())
    over = np.maximum(weights - cap, 0)
    scaled = values * cap * w.denominator - w.numerator * factor * opt * over
    return scaled, weights <= cap, opt * cap * w.denominator


def _float_rewards(k, w):
    table = C.output_table(_ratio(), k)
    reward = np.asarray(table["task"]) + w * RW._table_penalty(table, "graded")
    return reward, np.asarray(table["verdict"], dtype=bool)


def _selection(mask):
    return tuple(i for i in range(C.N_ITEMS) if (mask >> i) & 1)


# ---------------------------------------------------------------- the arm's name and label


def test_the_graded_arm_at_the_named_weight_is_named_graded_below_cap_on_the_capacity_task():
    cfg = {"task": "capacity", "arm": "graded", "w": 0.04, "w_task": 1.0}
    assert A.arm_name(cfg) == "graded_below_cap"
    assert A.NAMED_WEIGHTS["graded_below_cap"] == ("graded", A.GRADED_BELOW_CAP_W, 1.0)
    assert A.GRADED_BELOW_CAP_W == 0.04


def test_a_speed_run_at_the_same_weight_keeps_its_generic_name():
    """The capacity design's weight names no speed run, and a config without a task is a speed run."""
    assert A.arm_name({"task": "speed", "arm": "graded", "w": 0.04, "w_task": 1.0}) == "graded_w0.04_t1"
    assert A.arm_name({"arm": "graded", "w": 0.04}) == "graded_w0.04_t1"
    assert A.NAMED_WEIGHT_TASKS == {"graded_below_cap": ("capacity",)}


def test_every_other_weight_keeps_its_name_on_both_tasks():
    assert "graded_below_cap" not in A.ARM_SPECS and "graded_below_cap" not in A.CAPACITY_ARM_SPECS
    assert A.arm_name({"arm": "graded", "w": 0.01, "w_task": 1.0}) == "graded_below"
    assert A.arm_name({"task": "speed", "arm": "graded", "w": 0.01, "w_task": 1.0}) == "graded_below"
    assert A.arm_name({"arm": "graded", "w": 25}) == "graded_w25_t1"
    assert A.arm_name({"task": "capacity", "arm": "graded", "w": 25.0}) == "graded_w25.0_t1"
    assert A.arm_name({"task": "capacity", "arm": "graded", "w": 5.0}) == "graded"
    for task in ("speed", "capacity"):
        assert A.arm_name({"task": task, "arm": "graded", "w": 0.25, "w_task": 1.0}) == "graded_matched"
    assert A.arm_name({"task": "capacity", "arm": "binary", "w": 0.04}) == "binary_w0.04_t1"
    assert A.arm_name({"task": "capacity", "arm": "graded", "w": 0.04, "w_task": 0.0}) == \
        "graded_w0.04_t0"


def test_a_remedy_flag_adds_its_suffix_to_the_new_name():
    cfg = {"arm": "graded", "w": 0.04, "w_task": 1.0, "task": "capacity"}
    assert A.arm_name(dict(cfg, group_filter="verdict_mixed")) == "graded_below_cap_vfilter"
    assert A.arm_name(dict(cfg, kl_beta=0.04)) == "graded_below_cap_kl"


def test_the_arm_label_names_the_four_instances():
    cfg = {"arm": "graded", "w": 0.04, "w_task": 1.0, "task": "capacity"}
    label = A.compose_arm_label(cfg)
    assert label == r"residual $+$ task, below the ordering threshold on four instances"
    assert A._plain_label(label) == "residual + task, below the ordering threshold on four instances"
    assert A.NAMED_LABELS["graded_below_cap"] == label
    published = A.arm_labels([{"cfg": cfg}])
    assert published["graded_below_cap"] == label
    assert published["graded"] == "residual $+$ task"


def test_the_new_manifest_is_not_a_training_source():
    """The design leaves the training-source digest alone, and config.json records the weight."""
    assert not any(s.endswith("manifest_capacity_below.csv") for s in A.TRAINING_SOURCES)
    assert not any(s.endswith("manifest_capacity_below.csv") for s in T.TRAINING_SOURCES)


# ---------------------------------------------------------------- the weight and the bounds


def test_the_weight_is_twice_the_inclusion_bound_read_from_the_reward_code():
    assert _factor() == 100.0
    assert M._capacity_inclusion_bound(_ratio()) == 0.02
    assert 2.0 * M._capacity_inclusion_bound(_ratio()) == A.GRADED_BELOW_CAP_W
    assert M._capacity_below_weight() == A.GRADED_BELOW_CAP_W
    assert Fraction(A.GRADED_BELOW_CAP_W).limit_denominator(1000) == W_EXACT
    assert M.CAPACITY_BELOW_DROPPED == DROPPED


def test_the_four_bounds_equal_the_drafted_fractions_on_every_instance():
    assert [C.capacity_of(inst, _ratio()) for inst in C.INSTANCES] == [6, 6, 5, 5, 6, 5]
    bounds = [_exact_bounds(k) for k in range(len(C.INSTANCES))]
    assert [b[0] for b in bounds] == THRESHOLDS
    assert [b[1] for b in bounds] == FEASIBLE_TOP
    assert [b[2] for b in bounds] == INCLUSION
    assert [b[3] for b in bounds] == PAIRWISE
    assert max(INCLUSION) == Fraction(1, 50) and 2 * max(INCLUSION) == W_EXACT


def test_no_weight_keeps_the_best_feasible_answer_on_top_and_drops_the_ordering_everywhere():
    assert max(FEASIBLE_TOP) == Fraction(9, 250) > min(THRESHOLDS) == Fraction(31, 1040)
    for threshold, pair in zip(THRESHOLDS, PAIRWISE):
        assert pair > threshold
    assert all(pair < TRAINED_W for pair in PAIRWISE)


def test_the_weight_lies_in_window_b_and_above_instance_fives_threshold():
    low = max(FEASIBLE_TOP)
    high = min(THRESHOLDS[k] for k in DROPPED)
    assert (low, high) == (Fraction(9, 250), Fraction(19, 440))
    assert low < W_EXACT < high
    assert THRESHOLDS[5] == Fraction(13, 360) < W_EXACT
    assert [k for k, t in enumerate(THRESHOLDS) if t > W_EXACT] == list(DROPPED)
    assert all(b < W_EXACT for b in INCLUSION)


def test_the_float_enumeration_agrees_with_the_exact_fractions_at_the_weight():
    for k in range(len(C.INSTANCES)):
        reward, ok = _float_rewards(k, A.GRADED_BELOW_CAP_W)
        scaled, ok_exact, _ = _scaled_rewards(k)
        assert np.array_equal(ok, ok_exact)
        assert np.array_equal(reward[~ok] > reward[ok].min(), scaled[~ok] > 0)
        assert (reward[~ok].max() < reward[ok].max()) and (scaled[~ok].max() < scaled[ok].max())


# ---------------------------------------------------------------- what the weight does, by integers


def test_five_violating_selections_outrank_the_empty_selection_and_none_the_optimum():
    for k in range(len(C.INSTANCES)):
        scaled, ok, optimum = _scaled_rewards(k)
        assert scaled[ok].min() == 0 and scaled[ok].max() == optimum
        above = [_selection(int(m)) for m in np.flatnonzero((~ok) & (scaled > 0))]
        assert sorted(above) == ABOVE_EMPTY[k]
        assert not np.any((~ok) & (scaled == 0)), "a violator ties the empty selection"
        assert not np.any((~ok) & (scaled >= optimum)), "a violator reaches the optimum"
    assert [len(ABOVE_EMPTY[k]) for k in range(6)] == [2, 1, 0, 1, 1, 0]


def test_adding_any_item_to_an_overfull_selection_costs_at_least_twice_its_value():
    ratios = []
    for k, inst in enumerate(C.INSTANCES):
        cap, opt, _, _ = _instance(k)
        factor = int(_factor())
        # The penalty for adding item j to an overfull selection is w * factor * w_j / cap, and the
        # task gain is v_j / opt.
        ratios.append(min(W_EXACT * factor * w * opt / (cap * v)
                          for v, w in zip(inst["values"], inst["weights"])))
    assert ratios == [2, Fraction(10, 3), Fraction(208, 95), Fraction(88, 25), 2, Fraction(18, 5)]
    for k in range(len(C.INSTANCES)):
        scaled, ok, _ = _scaled_rewards(k)
        masks = np.arange(scaled.size)
        for item in range(C.N_ITEMS):
            base = masks[(((masks >> item) & 1) == 0) & ~ok]
            assert np.all(scaled[base | (1 << item)] < scaled[base])


def test_the_generator_measures_the_same_largest_addition_change():
    for k in range(len(C.INSTANCES)):
        reward, ok = _float_rewards(k, A.GRADED_BELOW_CAP_W)
        assert M._largest_addition_change(reward, ok) < 0
    reward, ok = _float_rewards(0, A.GRADED_BELOW_CAP_W)
    # On instance 0 the densest item is item 0, and its penalty is exactly twice its task gain of 1.
    assert M._largest_addition_change(reward, ok) == pytest.approx(-1.0)


def test_the_first_violation_penalty_is_below_the_light_verdicts_on_every_instance():
    w = A.GRADED_BELOW_CAP_W
    penalties = []
    for k in range(len(C.INSTANCES)):
        table = C.output_table(_ratio(), k)
        residual = np.asarray(table["residual"])
        penalties.append(w * _factor() * float(residual[residual > 0].min()))
    assert penalties == pytest.approx([2 / 3, 2 / 3, 0.8, 0.8, 2 / 3, 0.8])
    assert all(p < LIGHT_W for p in penalties)
    # The penalty equals the light verdict's 5 at a residual of 1.25 and the crossover of 12.50 at a
    # residual of 3.125.
    assert w * _factor() * 1.25 == pytest.approx(LIGHT_W)
    assert w * _factor() * 3.125 == pytest.approx(12.5)


def test_the_light_verdict_ranks_violators_above_the_best_feasible_answer():
    counts = []
    for k in range(len(C.INSTANCES)):
        table = C.output_table(_ratio(), k)
        reward = np.asarray(table["task"]) + LIGHT_W * RW._table_penalty(table, "binary")
        ok = np.asarray(table["verdict"])
        counts.append(int(np.sum(reward[~ok] > reward[ok].max())))
    assert counts == [5_901, 35_845, 43, 7_997, 7_468, 28_689]


# ---------------------------------------------------------------- the unparseable fallback


def _largest_fallback_margin(w):
    """Return, per instance, the largest amount by which an unparseable member can outearn a parsed one.

    A group's fallback is priced at its worst parsed residual. The parsed member with the lowest reward
    in such a group has a residual at most that level, so the margin at a level compares the fallback
    with the lowest reward among subsets whose residual does not exceed it."""
    out = []
    for k in range(len(C.INSTANCES)):
        table = C.output_table(_ratio(), k)
        residual = np.asarray(table["residual"])
        reward = np.asarray(table["task"]) + w * RW._table_penalty(table, "graded")
        order = np.argsort(residual, kind="stable")
        prefix_min = np.minimum.accumulate(reward[order])
        levels, last = np.unique(residual[order][::-1], return_index=True)
        last = residual.size - 1 - last
        margin = -np.inf
        for level, idx in zip(levels, last):
            fallback = C.UNPARSEABLE_TASK + w * RW._spec_penalty(
                C.synthesize(float(level), _ratio()), "graded", _ratio(), C.SPEC)
            margin = max(margin, fallback - float(prefix_min[idx]))
        out.append(margin)
    return out


def test_an_unparseable_member_never_outearns_a_parsed_member_at_the_weight():
    margins = _largest_fallback_margin(A.GRADED_BELOW_CAP_W)
    assert max(margins) < 0
    assert max(margins) == pytest.approx(-10.35294117647, abs=1e-9)
    assert int(np.argmax(margins)) == 4


def test_at_the_matched_weight_the_fallback_can_outearn_a_parsed_member_on_instance_five():
    margins = _largest_fallback_margin(0.25)
    assert max(margins) == pytest.approx(8.4444444444, abs=1e-6)
    assert int(np.argmax(margins)) == 5


def test_the_fallback_reward_with_no_parsed_member_under_each_arm():
    ratio = _ratio()
    got = [RW.group_rewards([None] * 4, arm, w, ratio, spec=C.SPEC)[0]
           for arm, w in (("graded", A.GRADED_BELOW_CAP_W), ("binary", LIGHT_W), ("graded", TRAINED_W))]
    assert got == pytest.approx([-10.0 - 28.0 / 3.0, -15.0, -10.0 - 3500.0 / 3.0])
    assert [round(x, 2) for x in got] == [-19.33, -15.0, -1176.67]


# ---------------------------------------------------------------- the generator's refusals


def test_the_generator_refuses_a_weight_that_analyze_does_not_name(monkeypatch):
    named = dict(A.NAMED_WEIGHTS, graded_below_cap=("graded", 0.05, 1.0))
    monkeypatch.setattr(A, "NAMED_WEIGHTS", named)
    with pytest.raises(ValueError, match="analyze.py names"):
        M.capacity_below()


def test_the_generator_refuses_a_weight_at_which_a_violator_beats_the_optimum(monkeypatch):
    """At 0.025, inside window A, the selection of items 0 and 10 outranks the optimum on instance 0."""
    monkeypatch.setattr(M, "_capacity_inclusion_bound", lambda ratio: 0.0125)
    named = dict(A.NAMED_WEIGHTS, graded_below_cap=("graded", 0.025, 1.0))
    monkeypatch.setattr(A, "NAMED_WEIGHTS", named)
    with pytest.raises(ValueError, match="reaches the best feasible reward on instance 0"):
        M._capacity_below_weight()
    reward, _ = _float_rewards(0, 0.025)
    assert reward[(1 << 0) | (1 << 10)] == pytest.approx(1.18333333333)


def test_the_generator_refuses_a_weight_whose_dropped_set_differs(monkeypatch):
    monkeypatch.setattr(M, "CAPACITY_BELOW_DROPPED", (0, 1, 3, 4, 5))
    with pytest.raises(ValueError, match="ordering is dropped on instances"):
        M._capacity_below_weight()
    # Above 19/440 the ordering returns on instance 3, so the set differs there too.
    monkeypatch.undo()
    monkeypatch.setattr(M, "_capacity_inclusion_bound", lambda ratio: 0.022)
    named = dict(A.NAMED_WEIGHTS, graded_below_cap=("graded", 0.044, 1.0))
    monkeypatch.setattr(A, "NAMED_WEIGHTS", named)
    with pytest.raises(ValueError, match=r"dropped on instances \[0, 1, 4\]"):
        M._capacity_below_weight()


def test_the_generator_refuses_an_addition_that_keeps_the_reward(monkeypatch):
    monkeypatch.setattr(M, "_largest_addition_change", lambda reward, verdict: 0.0)
    with pytest.raises(ValueError, match="does not lower the reward"):
        M._capacity_below_weight()


def test_the_generator_refuses_a_near_tie_with_the_worst_feasible_answer(monkeypatch):
    """The closest violator on instance 0 earns 1/60 more than the empty selection."""
    monkeypatch.setattr(M, "CAPACITY_BELOW_MARGIN", 0.02)
    with pytest.raises(ValueError, match="ties the worst feasible reward on instance 0"):
        M._capacity_below_weight()


# ---------------------------------------------------------------- the manifest


def _implied_v2(row):
    """Return a committed v1 capacity row as the v2 row the submission script and the trainer give it.

    No v1 row carries the three remedy flags, so each takes the trainer's default."""
    defaults = vars(T.build_arg_parser().parse_args(
        ["--arm", "graded", "--w", "5", "--vmax", "5", "--out-dir", "unused"]))
    return dict(row, task="capacity", adv_variant=defaults["adv_variant"],
                group_filter=defaults["group_filter"], kl_beta=repr(float(defaults["kl_beta"])))


def _source_rows(arm_name):
    """Return the committed hard-level capacity rows of one arm, as v2 rows keyed by seed."""
    rows = [r for r in _legacy_capacity_rows() if r["regime"] == "H" and r["arm_name"] == arm_name]
    return {r["seed"]: _implied_v2(r) for r in rows}


def test_the_manifest_holds_five_seeds_of_the_named_arm():
    rows = _rows()
    assert tuple(rows[0].keys()) == M.V2_HEADER
    assert [int(r["task_id"]) for r in rows] == [1, 2, 3, 4, 5]
    assert [int(r["seed"]) for r in rows] == [0, 1, 2, 3, 4]
    for r in rows:
        assert (r["task"], r["regime"], r["arm_name"], r["arm"]) == ("capacity", "H", "graded_below_cap",
                                                                      "graded")
        assert r["w"] == "0.04" and float(r["w"]) == A.GRADED_BELOW_CAP_W
        assert r["w_task"] == "1.0" and float(r["vmax"]) == _ratio()
        assert (r["adv_variant"], r["group_filter"], r["kl_beta"]) == ("grpo", "none", "0.0")


def test_the_manifest_is_lf_only_numbered_and_trimmed():
    raw = open(MANIFEST, "rb").read()
    assert b"\r" not in raw and raw.endswith(b"\n")
    rows = _rows()
    assert [int(r["task_id"]) for r in rows] == list(range(1, len(rows) + 1))
    for r in rows:
        assert all(v == v.strip() and v != "" for v in r.values()), r


def test_every_row_equals_the_residual_row_of_its_seed_except_weight_and_name():
    """The weight and the name are the only differences from the trained residual runs."""
    source = _source_rows("graded")
    rows = _rows()
    assert sorted(source) == sorted(r["seed"] for r in rows)
    changed = ("task_id", "w", "arm_name")
    for r in rows:
        base = source[r["seed"]]
        assert (base["arm"], float(base["w"])) == ("graded", TRAINED_W)
        assert {k: v for k, v in r.items() if k not in changed} == \
            {k: base[k] for k in r if k not in changed}
        assert r["w"] != base["w"] and r["arm_name"] != base["arm_name"]


def test_every_row_equals_the_light_verdict_row_of_its_seed_except_arm_weight_and_name():
    source = _source_rows("binary")
    rows = _rows()
    assert sorted(source) == sorted(r["seed"] for r in rows)
    changed = ("task_id", "arm", "w", "arm_name")
    for r in rows:
        base = source[r["seed"]]
        assert (base["arm"], float(base["w"])) == ("binary", LIGHT_W)
        assert {k: v for k, v in r.items() if k not in changed} == \
            {k: base[k] for k in r if k not in changed}


def test_the_rows_differ_from_the_matched_rows_only_in_weight_and_name():
    matched = {r["seed"]: r for r in _rows(os.path.join(HERE, "slurm", "manifest_capacity_matched.csv"))}
    changed = ("w", "arm_name")
    for r in _rows():
        m = matched[r["seed"]]
        assert {k: v for k, v in r.items() if k not in changed} == \
            {k: v for k, v in m.items() if k not in changed}


def test_the_analysis_names_every_row_as_the_row_names_itself():
    for r in _rows():
        cfg = {"task": r["task"], "arm": r["arm"], "w": float(r["w"]), "w_task": float(r["w_task"]),
               "adv_variant": r["adv_variant"], "group_filter": r["group_filter"],
               "kl_beta": float(r["kl_beta"])}
        assert A.arm_name(cfg) == r["arm_name"] == "graded_below_cap"


def test_the_trainer_accepts_every_row():
    for r in _rows():
        argv = ["--task", r["task"], "--arm", r["arm"], "--w", r["w"], "--w-task", r["w_task"],
                "--vmax", r["vmax"], "--seed", r["seed"], "--regime", r["regime"],
                "--adv-variant", r["adv_variant"], "--group-filter", r["group_filter"],
                "--kl-beta", r["kl_beta"], "--out-dir", "unused"]
        ap = T.build_arg_parser()
        args = ap.parse_args(argv)
        T.check_args(ap, args)
        assert args.w == A.GRADED_BELOW_CAP_W and args.arm == "graded" and args.task == "capacity"


def test_the_generator_writes_the_committed_file():
    assert M.build_all()["slurm/manifest_capacity_below.csv"] == open(MANIFEST).read()
    assert M.main(["--check"]) == 0


def _parse_block():
    src = open(SBATCH).read()
    assert src.count("# >>> manifest-parse") == 1 and src.count("# <<< manifest-parse") == 1
    return src.split("# >>> manifest-parse")[1].split("# <<< manifest-parse")[0]


def _parse(task_id, task):
    script = "\n".join([
        "set -uo pipefail",
        f"SLURM_ARRAY_TASK_ID={task_id}", f"MANIFEST='{MANIFEST}'", f"CANCEL_TASK={task}",
        f"PY='{sys.executable}'", f"EXP='{HERE}'",
        _parse_block(),
        'printf "%s|%s|%s|%s|%s|%s|%s|%s|%s\\n" "$TASK_ID" "$REGIME" "$ARM_NAME" "$ARM" "$W" '
        '"$W_TASK" "$VMAX" "$SEED" "$REMEDY_FLAGS"',
    ])
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


@requires_sbatch
def test_the_submission_scripts_v2_branch_parses_every_row():
    for r in _rows():
        p = _parse(r["task_id"], "capacity")
        assert p.returncode == 0, p.stderr
        fields = p.stdout.strip().split("|")
        assert fields[:8] == [r["task_id"], r["regime"], r["arm_name"], r["arm"], r["w"], r["w_task"],
                              r["vmax"], r["seed"]]
        assert fields[8] == "--adv-variant grpo --group-filter none --kl-beta 0.0"


@requires_sbatch
def test_the_submission_script_refuses_the_manifest_for_the_speed_task():
    p = _parse("1", "speed")
    assert p.returncode == 2 and "REFUSED" in p.stderr


@requires_sbatch
def test_the_run_directory_name_carries_the_task_id_and_the_arm_name():
    assert "${TASK_ID}_${REGIME}_${ARM_NAME}_s${SEED}" in open(SBATCH).read()


# ---------------------------------------------------------------- completeness of the run directory
# The directory exists only after the runs are pulled and assembled. A seed that did not finish must be
# named in the not_run list of the directory's assembled.json, so an undeclared gap fails here.


@pytest.mark.skipif(not os.path.isdir(BELOW_RUNS), reason="runs/capacity_below has not been assembled")
def test_the_assembled_capacity_below_directory_is_complete():
    record = json.load(open(os.path.join(BELOW_RUNS, "assembled.json")))
    runs = record["runs"]
    assert {e["arm"] for e in runs} == {"graded", "binary", "graded_below_cap"}
    finished = [int(e["seed"]) for e in runs if e["arm"] == "graded_below_cap"]
    declared = [int(x["seed"]) if isinstance(x, dict) else int(x) for x in record.get("not_run", [])]
    assert len(set(finished)) == len(finished), "a below-threshold seed appears twice"
    assert not set(finished) & set(declared), "a seed is both finished and listed as not run"
    assert set(finished) | set(declared) == {int(r["seed"]) for r in _rows()}
    for e in runs:
        cfg = json.load(open(os.path.join(BELOW_RUNS, e["link"], "config.json")))
        assert A.arm_name(dict({"task": "capacity", "w_task": 1.0}, **cfg)) == e["arm"]


# ---------------------------------------------------------------- the replay generator's design macros
# The replay generator emits the four bounds per instance, the two windows, the weights, the five
# selections above the empty selection, the light verdict's counts above the best feasible answer and
# the fallback rewards without reading a run. These tests compare its strings with the exact fractions
# and the counts above. The paper directory is not part of the anonymous code supplement, so they skip
# there.

PAPER = os.path.join(HERE, "paper")
GENERATOR = os.path.join(PAPER, "make_capacity_below_replay_macros.py")
requires_generator = pytest.mark.skipif(
    not os.path.exists(GENERATOR),
    reason="the paper's generators are not part of the anonymous code supplement")


def _generator():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_capacity_below_replay_macros", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _design(mod):
    return mod.design_macros(A.GRADED_BELOW_CAP_W, LIGHT_W, TRAINED_W, _ratio())


def _four_places(fractions):
    return ", ".join(f"{float(x):.4f}" for x in fractions)


@requires_generator
def test_the_generator_emits_the_four_bounds_and_both_windows_from_the_exact_fractions():
    mod = _generator()
    m = _design(mod)
    assert m["capBelowThresholds"] == _four_places(THRESHOLDS)
    assert m["capBelowFeasibleTopBounds"] == _four_places(FEASIBLE_TOP)
    assert m["capBelowInclusionBounds"] == _four_places(INCLUSION)
    assert m["capBelowPairwiseBounds"] == _four_places(PAIRWISE)
    assert (m["capBelowWindowALow"], m["capBelowWindowAHigh"]) == \
        (f"{float(max(INCLUSION)):.4f}", f"{float(min(THRESHOLDS)):.4f}")
    assert (m["capBelowWindowLow"], m["capBelowWindowHigh"]) == \
        (f"{float(max(FEASIBLE_TOP)):.4f}", f"{float(min(THRESHOLDS[k] for k in DROPPED)):.4f}")
    assert (m["capBelowW"], m["capBelowInclusionBound"]) == ("0.04", "0.02")
    assert (m["capBelowLightW"], m["capBelowTrainedW"]) == ("5", "5")
    # The enumerated values agree with the fractions beyond the four printed digits.
    for got, want in zip(mod._inclusion_bounds(_ratio()), INCLUSION):
        assert got == pytest.approx(float(want), abs=1e-12)
    for got, want in zip(mod._pairwise_bounds(_ratio()), PAIRWISE):
        assert got == pytest.approx(float(want), abs=1e-12)
    assert mod.window_a(_ratio()) == pytest.approx((float(max(INCLUSION)), float(min(THRESHOLDS))))


@requires_generator
def test_the_generator_lists_the_five_selections_the_light_counts_and_the_fallbacks():
    m = _design(_generator())
    assert m["capBelowAboveEmptyCount"] == "5"
    assert m["capBelowAboveEmptyPerInstance"] == ", ".join(str(len(ABOVE_EMPTY[k])) for k in range(6))
    assert m["capBelowAboveEmptySelections"] == (
        "items 0 and 10 and items 0 and 12 on instance 0, item 0 on instance 1, "
        "items 14 and 15 on instance 3, and items 9 and 10 on instance 4")
    # The same counts and rewards that the two tests above compute from the reward code.
    assert m["capBelowLightAboveBestPerInstance"] == "5,901, 35,845, 43, 7,997, 7,468, 28,689"
    assert (m["capBelowFallbackBelow"], m["capBelowFallbackLight"], m["capBelowFallbackTrained"]) == \
        ("-19.33", "-15.00", "-1,176.67")
    assert m["capBelowFirstPenaltyLargeCap"] == "0.67" and m["capBelowFirstPenaltySmallCap"] == "0.80"
    assert all(name.startswith("capBelow") and name.isalpha() for name in m)


@requires_generator
def test_the_selection_phrase_groups_by_instance_in_the_entrys_wording():
    mod = _generator()
    assert mod.selection_phrase([]) == "none"
    assert mod.selection_phrase([(1, 0)]) == "item 0 on instance 1"
    assert mod.selection_phrase([(3, 14, 15), (0, 0, 12), (0, 0, 10)]) == \
        "items 0 and 10 and items 0 and 12 on instance 0 and items 14 and 15 on instance 3"
    assert mod.selection_phrase([(2, 1, 2, 3), (4, 9, 10), (5, 7)]) == \
        "items 1, 2 and 3 on instance 2, items 9 and 10 on instance 4, and item 7 on instance 5"


@requires_generator
def test_the_design_macros_refuse_a_pairwise_bound_that_is_not_above_the_threshold(monkeypatch):
    mod = _generator()
    monkeypatch.setattr(mod, "_pairwise_bounds", lambda ratio: [0.0] * len(C.INSTANCES))
    with pytest.raises(mod.RefusalError, match="pairwise bound"):
        _design(mod)
    monkeypatch.setattr(mod, "_pairwise_bounds", lambda ratio: [TRAINED_W] * len(C.INSTANCES))
    with pytest.raises(mod.RefusalError, match="pairwise bound"):
        _design(mod)


@requires_generator
def test_the_design_macros_refuse_an_inclusion_bound_that_disagrees_or_an_empty_window_a(monkeypatch):
    mod = _generator()
    monkeypatch.setattr(mod, "_inclusion_bounds", lambda ratio: [0.03] * len(C.INSTANCES))
    with pytest.raises(mod.RefusalError, match="inclusion bound"):
        _design(mod)
    monkeypatch.undo()
    monkeypatch.setattr(mod, "window_a", lambda ratio: (0.05, 0.03))
    with pytest.raises(mod.RefusalError, match="window A"):
        _design(mod)
