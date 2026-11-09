"""These tests cover the residual below its ordering threshold on the speed task, in the analysis and the
manifest.

The design trains the graded arm at a weight w' inside a window. Inside it the residual reward still
decreases with the speed magnitude on each side of the task cap, so an all-violating group keeps its
ordering. Inside it the residual also stops ranking every feasible completion above every violator.
Each test pins one fact that the design or its pre-registered entry relies on. Each fact is
recomputed here from the reward code, and no test reads make_manifests.py or analyze.py for the
number. None of these tests evaluates the enumeration at the upper end of the window, because at
that weight the float result and the exact tie disagree.
"""
import csv
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402
import make_manifests as M  # noqa: E402
import rewards as RW  # noqa: E402
import train_speed_grpo as T  # noqa: E402

SBATCH = os.path.join(HERE, "slurm", "cancel_arms.sbatch")
requires_sbatch = pytest.mark.skipif(
    not os.path.exists(SBATCH),
    reason="the cluster submission script is not shipped in the anonymous supplement, because it "
           "names a compute account and a site-specific scratch path")
MANIFEST = os.path.join(HERE, "slurm", "manifest_speed_below.csv")
BELOW_RUNS = os.path.join(HERE, "runs", "speed_below")
IDENTITY_TOL = 1e-4
LIGHT_W = A.ARM_SPECS["binary"][1]
TRAINED_W = A.ARM_SPECS["graded"][1]


def _vmax():
    return float(yaml.safe_load(open(os.path.join(HERE, "configs", "regimes.yaml")))["A"]["v_max"])


def _rows(path=MANIFEST):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _constants():
    """Return the task slope, the residual factor and the task cap, all read from the reward code."""
    v = _vmax()
    slope = float(RW.task(1.0, v) - RW.task(0.0, v))
    factor = -float(RW.pen_graded(2.0 * v, v))
    grid = np.arange(0, 1001, dtype=np.float64)
    cap = float(grid[int(np.argmax(RW.task(grid, v)))])
    return v, slope, factor, cap


def _window():
    """Return the open window for w' by enumeration, without using make_manifests.py."""
    v, slope, factor, _ = _constants()
    tab = RW.speed_output_table(v)
    viol = ~tab["verdict"]
    worst_feasible = tab["task"][tab["verdict"]].min()
    lower = slope * v / factor
    upper = float(((tab["task"][viol] - worst_feasible) / (factor * tab["residual"][viol])).max())
    return lower, upper


def _rewards_over_enumeration(arm, w):
    v = _vmax()
    tab = RW.speed_output_table(v)
    s = np.arange(tab["task"].size, dtype=np.float64)
    return s, tab["verdict"], RW.arm_reward(s, arm, w, v)


# ---------------------------------------------------------------- the arm's name and label


def test_the_graded_arm_at_the_named_weight_is_named_graded_below():
    cfg = {"arm": "graded", "w": 0.01, "w_task": 1.0}
    assert A.arm_name(cfg) == "graded_below"
    assert A.arm_name(dict(cfg, task="speed")) == "graded_below"
    assert A.NAMED_WEIGHTS["graded_below"] == ("graded", A.GRADED_BELOW_W, 1.0)
    assert A.GRADED_BELOW_W == 0.01


def test_the_name_stays_out_of_the_arm_specs_and_every_other_weight_keeps_its_name():
    assert "graded_below" not in A.ARM_SPECS and "graded_below" not in A.CAPACITY_ARM_SPECS
    assert A.arm_name({"arm": "graded", "w": 25}) == "graded_w25_t1"
    assert A.arm_name({"arm": "graded", "w": 5.0}) == "graded"
    assert A.arm_name({"arm": "binary", "w": 0.01}) == "binary_w0.01_t1"
    assert A.arm_name({"arm": "graded", "w": 0.01, "w_task": 0.0}) == "graded_w0.01_t0"


def test_a_remedy_flag_adds_its_suffix_to_the_new_name():
    cfg = {"arm": "graded", "w": 0.01, "w_task": 1.0, "task": "speed"}
    assert A.arm_name(dict(cfg, group_filter="verdict_mixed")) == "graded_below_vfilter"
    assert A.arm_name(dict(cfg, kl_beta=0.04)) == "graded_below_kl"


def test_the_arm_label_names_the_ordering_threshold():
    cfg = {"arm": "graded", "w": 0.01, "w_task": 1.0, "task": "speed"}
    label = A.compose_arm_label(cfg)
    assert label == r"residual $+$ task, below the ordering threshold"
    assert A._plain_label(label) == "residual + task, below the ordering threshold"
    assert A.NAMED_LABELS["graded_below"] == label
    published = A.arm_labels([{"cfg": cfg}])
    assert published["graded_below"] == label
    assert published["graded"] == "residual $+$ task"


def test_the_new_manifest_is_not_a_training_source():
    """The design leaves the training-source digest alone, and config.json records the weight."""
    assert not any(s.endswith("manifest_speed_below.csv") for s in A.TRAINING_SOURCES)
    assert not any(s.endswith("manifest_speed_below.csv") for s in T.TRAINING_SOURCES)


# ---------------------------------------------------------------- the weight and its window


def test_the_weight_is_the_slope_rule_value_and_lies_strictly_inside_the_window():
    v, slope, factor, _ = _constants()
    assert (v, slope, factor) == (5.0, 0.1, 100.0)
    assert 2.0 * slope * v / factor == A.GRADED_BELOW_W
    lower, upper = _window()
    assert lower == 0.005
    assert upper == pytest.approx(0.03, abs=1e-12)
    assert lower < A.GRADED_BELOW_W < upper
    # The generator recomputes the same window and the same weight.
    assert M._below_window(v) == (lower, upper)
    assert M._below_weight() == A.GRADED_BELOW_W


def test_the_first_violation_penalty_is_below_the_light_verdicts():
    v, _, factor, _ = _constants()
    first = A.GRADED_BELOW_W * factor * float(RW.residual(6.0, v))
    assert first == pytest.approx(0.2)
    assert first < LIGHT_W
    assert first == pytest.approx(-A.GRADED_BELOW_W * float(RW.pen_graded(6.0, v)))


def test_the_penalty_reaches_the_light_verdicts_at_speed_30_and_the_crossover_at_105():
    v, _, _, _ = _constants()
    s = np.arange(0, 1001, dtype=np.float64)
    penalty = -A.GRADED_BELOW_W * RW.pen_graded(s, v)
    assert int(s[penalty >= LIGHT_W - 1e-9][0]) == 30
    crossover = float(RW.task(1e6, v) - RW.task(0.0, v))
    assert crossover == 20.0
    assert int(s[penalty >= crossover - 1e-9][0]) == 105


def test_the_window_tracks_the_reward_code(monkeypatch):
    """A weight outside the recomputed window, or one that differs from the named weight, is refused."""
    # The first window lies above the weight, so it tests the lower bound. The second lies below it.
    for window in ((0.5, 0.6), (0.001, 0.005)):
        monkeypatch.setattr(M, "_below_window", lambda v_max, window=window: window)
        with pytest.raises(ValueError, match="outside the window"):
            M._below_weight()
    monkeypatch.undo()
    named = dict(A.NAMED_WEIGHTS, graded_below=("graded", 0.02, 1.0))
    monkeypatch.setattr(A, "NAMED_WEIGHTS", named)
    with pytest.raises(ValueError, match="analyze.py names"):
        M._below_weight()


def test_the_lower_bound_flattens_the_reward_below_the_cap_and_double_it_does_not():
    lower, _ = _window()
    v, _, _, cap = _constants()
    s, _, r = _rewards_over_enumeration("graded", lower)
    band = (s > v) & (s <= cap)
    assert np.ptp(r[band]) < 1e-12, "at the lower bound every sub-cap violator earns the same reward"
    _, _, doubled = _rewards_over_enumeration("graded", 2.0 * lower)
    assert np.all(np.diff(doubled[(s >= v) & (s <= cap)]) < 0)


# ---------------------------------------------------------------- what the weight does to the ordering


def test_four_violator_magnitudes_outrank_the_worst_feasible_answer_and_none_the_best():
    s, feasible, r = _rewards_over_enumeration("graded", A.GRADED_BELOW_W)
    violator = ~feasible
    assert s[violator & (r > r[feasible].min())].tolist() == [6.0, 7.0, 8.0, 9.0]
    assert not np.any(violator & (r > r[feasible].max()))
    assert r[feasible].max() == pytest.approx(0.5)


def test_the_reward_decreases_with_the_magnitude_on_each_side_of_the_task_cap():
    v, _, _, cap = _constants()
    s, feasible, r = _rewards_over_enumeration("graded", A.GRADED_BELOW_W)
    below_cap = (s >= v) & (s <= cap)
    above_cap = s >= cap
    assert np.all(np.diff(r[below_cap]) < 0)
    assert np.all(np.diff(r[above_cap]) < 0)


def test_the_light_verdict_puts_violators_above_the_best_feasible_answer():
    s, feasible, r = _rewards_over_enumeration("binary", LIGHT_W)
    above_best = s[(~feasible) & (r > r[feasible].max())]
    above_worst = s[(~feasible) & (r > r[feasible].min())]
    assert above_best.size >= 1 and above_best.min() == 56.0
    assert above_worst.min() == 51.0
    assert above_best.size == 999_944


# ---------------------------------------------------------------- the identity class


# The third and fourth groups hold the boundary speed 5.
# The fourth and sixth groups hold the task cap.
IDENTITY_GROUPS = [
    (10.0, 20.0, 30.0, 60.0),
    (6.0, 7.0, 9.0, 150.0),
    (5.0, 6.0, 7.0, 9.0),
    (5.0, 5.0, 12.0, 200.0),
    (250.0, 300.0, 1000.0, 205.0),
    (200.0, 400.0, 999_999.0, 1_000.0),
]


@pytest.mark.parametrize("speeds", IDENTITY_GROUPS)
def test_on_one_side_of_the_cap_the_advantages_equal_the_trained_residuals(speeds):
    v = _vmax()
    lo = RW.advantage(RW.group_rewards(list(speeds), "graded", A.GRADED_BELOW_W, v))
    hi = RW.advantage(RW.group_rewards(list(speeds), "graded", TRAINED_W, v))
    assert np.abs(lo - hi).max() <= IDENTITY_TOL


def test_the_normalizer_epsilon_is_why_the_tolerance_is_larger_than_1e_6():
    """The 1e-6 term is divided by the slope ratio, which is about one thousandth at this weight."""
    v = _vmax()
    ratio = (20.0 * A.GRADED_BELOW_W - 0.1) / 99.9
    assert ratio == pytest.approx(1.0 / 999.0, rel=1e-3)
    worst = 0.0
    for speeds in IDENTITY_GROUPS:
        lo = RW.advantage(RW.group_rewards(list(speeds), "graded", A.GRADED_BELOW_W, v))
        hi = RW.advantage(RW.group_rewards(list(speeds), "graded", TRAINED_W, v))
        worst = max(worst, float(np.abs(lo - hi).max()))
    assert worst < IDENTITY_TOL


def test_a_feasible_member_below_the_limit_beside_a_violator_breaks_the_identity():
    v = _vmax()
    speeds = [1.0, 6.0, 1.0, 1.0]
    lo = RW.advantage(RW.group_rewards(speeds, "graded", A.GRADED_BELOW_W, v))
    hi = RW.advantage(RW.group_rewards(speeds, "graded", TRAINED_W, v))
    assert np.abs(lo - hi).max() > 1.0
    assert int(np.argmax(lo)) == 1, "At the new weight the violator at 6 takes the top rank."
    assert int(np.argmin(hi)) == 1, "At the trained weight the violator at 6 takes the bottom rank."
    light = RW.advantage(RW.group_rewards(speeds, "binary", LIGHT_W, v))
    assert int(np.argmin(light)) == 1


def test_a_group_that_straddles_the_cap_may_differ_but_keeps_its_ranking():
    v = _vmax()
    speeds = [30.0, 150.0, 400.0, 900.0]
    lo = RW.group_rewards(speeds, "graded", A.GRADED_BELOW_W, v)
    hi = RW.group_rewards(speeds, "graded", TRAINED_W, v)
    assert list(np.argsort(-lo)) == list(np.argsort(-hi))


# ---------------------------------------------------------------- the unparseable fallback


def test_the_fallback_reward_with_no_parsed_member_under_each_arm():
    v = _vmax()
    assert RW.group_rewards([None] * 4, "graded", A.GRADED_BELOW_W, v)[0] == pytest.approx(-11.0)
    assert RW.group_rewards([None] * 4, "binary", LIGHT_W, v)[0] == -15.0
    assert RW.group_rewards([None] * 4, "graded", TRAINED_W, v)[0] == pytest.approx(-510.0)


def _random_group(rng):
    pool = [lambda: float(rng.randint(0, 9)), lambda: float(rng.randint(0, 40)),
            lambda: float(rng.randint(41, 400)), lambda: float(rng.randint(401, 999_999))]
    return [None if rng.random() < 0.4 else rng.choice(pool)() for _ in range(4)]


@pytest.mark.parametrize("arm,w", [("graded", A.GRADED_BELOW_W), ("binary", LIGHT_W),
                                   ("graded", TRAINED_W)])
def test_an_unparseable_member_never_earns_more_than_a_parsed_member(arm, w):
    v = _vmax()
    rng = random.Random(20260919)
    checked = 0
    for _ in range(3000):
        group = _random_group(rng)
        if all(x is not None for x in group) or all(x is None for x in group):
            continue
        rewards = RW.group_rewards(group, arm, w, v)
        parsed = [rewards[i] for i, x in enumerate(group) if x is not None]
        for i, x in enumerate(group):
            if x is None:
                assert rewards[i] <= min(parsed) + 1e-9, (group, rewards)
                checked += 1
    assert checked > 1000


# ---------------------------------------------------------------- the manifest


def test_the_manifest_holds_three_seeds_of_the_named_arm():
    rows = _rows()
    assert [int(r["task_id"]) for r in rows] == [1, 2, 3]
    assert [int(r["seed"]) for r in rows] == [0, 1, 2]
    assert tuple(rows[0].keys()) == M.V2_HEADER
    for r in rows:
        assert (r["task"], r["regime"], r["arm_name"], r["arm"]) == ("speed", "A", "graded_below",
                                                                      "graded")
        assert float(r["w"]) == A.GRADED_BELOW_W and r["w"] == "0.01"
        assert float(r["w_task"]) == 1.0 and float(r["vmax"]) == _vmax()
        assert (r["adv_variant"], r["group_filter"], float(r["kl_beta"])) == ("grpo", "none", 0.0)


def test_the_manifest_is_lf_only_numbered_and_trimmed():
    raw = open(MANIFEST, "rb").read()
    assert b"\r" not in raw and raw.endswith(b"\n")
    for r in _rows():
        assert all(v == v.strip() and v != "" for v in r.values()), r


def test_every_row_equals_the_light_verdict_row_of_its_seed_except_arm_weight_and_name():
    light_manifest = _rows(os.path.join(HERE, "slurm", "manifest.csv"))
    fingerprint = _rows(os.path.join(HERE, "slurm", "manifest_fingerprint_speed.csv"))[0]
    changed = ("arm", "w", "arm_name")
    for r in _rows():
        light = next(x for x in light_manifest
                     if x["regime"] == "A" and x["arm_name"] == "binary" and x["seed"] == r["seed"])
        assert (light["arm"], float(light["w"])) == ("binary", LIGHT_W)
        assert r["regime"] == light["regime"] and r["seed"] == light["seed"]
        assert r["w_task"] == repr(float(A.ARM_SPECS["binary"][2]))
        assert r["vmax"] == repr(_vmax())
        if r["seed"] == fingerprint["seed"]:
            assert {k: v for k, v in r.items() if k not in changed} == \
                {k: v for k, v in fingerprint.items() if k not in changed}


def test_the_analysis_names_every_row_as_the_row_names_itself():
    for r in _rows():
        cfg = {"task": r["task"], "arm": r["arm"], "w": float(r["w"]), "w_task": float(r["w_task"]),
               "adv_variant": r["adv_variant"], "group_filter": r["group_filter"],
               "kl_beta": float(r["kl_beta"])}
        assert A.arm_name(cfg) == r["arm_name"]


def test_the_trainer_accepts_every_row():
    for r in _rows():
        argv = ["--task", r["task"], "--arm", r["arm"], "--w", r["w"], "--w-task", r["w_task"],
                "--vmax", r["vmax"], "--seed", r["seed"], "--regime", r["regime"],
                "--adv-variant", r["adv_variant"], "--group-filter", r["group_filter"],
                "--kl-beta", r["kl_beta"], "--out-dir", "unused"]
        ap = T.build_arg_parser()
        args = ap.parse_args(argv)
        T.check_args(ap, args)
        assert args.w == A.GRADED_BELOW_W and args.arm == "graded"


def test_the_generator_writes_the_committed_file():
    assert M.build_all()["slurm/manifest_speed_below.csv"] == open(MANIFEST).read()
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
def test_the_submission_scripts_v2_branch_parses_every_row():
    for r in _rows():
        p = _parse(MANIFEST, r["task_id"], "speed")
        assert p.returncode == 0, p.stderr
        fields = p.stdout.strip().split("|")
        assert fields[:8] == [r["task_id"], r["regime"], r["arm_name"], r["arm"], r["w"], r["w_task"],
                              r["vmax"], r["seed"]]
        assert fields[8] == "--adv-variant grpo --group-filter none --kl-beta 0.0"


@requires_sbatch
def test_the_submission_script_refuses_the_manifest_for_the_other_task():
    p = _parse(MANIFEST, "1", "capacity")
    assert p.returncode == 2 and "REFUSED" in p.stderr


@requires_sbatch
def test_the_run_directory_name_carries_the_task_id_and_the_arm_name():
    src = open(SBATCH).read()
    assert "${TASK_ID}_${REGIME}_${ARM_NAME}_s${SEED}" in src


# ---------------------------------------------------------------- completeness of the run directory
# The gate that refuses an undeclared gap is in the paper's make_remedy_tables.py, and its tests are in
# test_paper_generators.py. The test below reads the assembled directory itself once it exists.


@pytest.mark.skipif(not os.path.isdir(BELOW_RUNS), reason="runs/speed_below has not been assembled")
def test_the_assembled_below_threshold_directory_is_complete():
    import json
    record = json.load(open(os.path.join(BELOW_RUNS, "assembled.json")))
    runs = record["runs"]
    assert {e["arm"] for e in runs} == {"graded", "binary", "graded_below"}
    finished = [int(e["seed"]) for e in runs if e["arm"] == "graded_below"]
    declared = [int(x["seed"]) if isinstance(x, dict) else int(x) for x in record.get("not_run", [])]
    assert len(set(finished)) == len(finished), "a below-threshold seed appears twice"
    assert not set(finished) & set(declared), "a seed is both finished and listed as not run"
    assert set(finished) | set(declared) == {0, 1, 2}
    for e in runs:
        cfg = json.load(open(os.path.join(BELOW_RUNS, e["link"], "config.json")))
        assert A.arm_name(dict({"task": "speed", "w_task": 1.0}, **cfg)) == e["arm"]
