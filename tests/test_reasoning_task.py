"""These tests check the WO-I parsers, verdict, brevity term, rewards and frozen source list."""
import json
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import reasoning_task as RT
import rewards as RW
import train_speed_grpo as TSG

HERE = Path(__file__).resolve().parent.parent
# The digest of the CURRENT training sources. The September 2026 value,
# 6afdaa1fa50c1126c3278aa33887d4be4ec47e5283097bda081bfa563e3300aa, is what every frozen run before
# November 2026 recorded, and the tests that read recorded provenance keep reading it from the runs.
# The pin moved on 2026-11-06 because rewards.py and train_speed_grpo.py gained the ICML 2027 blocks of
# November 2026 (the grpo_fixedref variant, --fixedref-c, --skip-zero-adv and held-out prompt
# evaluation), all off by default. It passed through two intermediate values that day,
# 7c8c42d450872fd021d87101d8a486443f6177a1725737cffc81777cd1deada9 after the first of those edits and
# 2ef23e5bd59b2880d0596a10fd79072fedb9df6b6795e95a0c2171a75cf787f2 after the rest, and it moved once
# more when TRAINING_SOURCES gained speed_bounded_task.py and slurm/manifest_speed_bounded.csv for Block
# A'. It moves again whenever a training source changes, and each move must be deliberate and dated here.
FROZEN = "65d67374f425580b8d4ac7e3d21e4d5fa08f705f6816d7ee8239c8d297d95690"


# ------------------------------------------------------------------ parsers


@pytest.mark.parametrize("text,value", [
    ("so the answer is \\boxed{72}.", Fraction(72)),
    ("\\boxed{1,234}", Fraction(1234)),
    ("\\boxed{1{,}000}", Fraction(1000)),
    ("\\boxed{1\\,000}", Fraction(1000)),
    ("\\boxed{\\$18}", Fraction(18)),
    ("\\boxed{$1,000.50}", Fraction(2001, 2)),
    ("\\boxed{ 18 }", Fraction(18)),
    ("\\boxed{18.00}", Fraction(18)),
    ("\\boxed{-5}", Fraction(-5)),
    ("\\boxed{.5}", Fraction(1, 2)),
    ("\\boxed{3/4}", Fraction(3, 4)),
    ("first \\boxed{3} then \\boxed{4}", Fraction(4)),
    ("\\boxed{12} and then {x}", Fraction(12)),
])
def test_strict_parser_reads_the_last_box(text, value):
    assert RT.strict_parse(text) == value


@pytest.mark.parametrize("text", [
    "no box here, the answer is 12",
    "\\boxed{12",                       # the completion was cut at the generation limit
    "\\boxed{\\frac{1}{2}}",            # a LaTeX fraction is not a plain number
    "\\boxed{18 \\text{ dollars}}",
    "\\boxed{1,23}",                    # this comma is not a thousands grouping
    "\\boxed{}",
    "\\boxed{3/0}",
    "\\boxed{25\\%}",
])
def test_strict_parser_refuses_anything_else(text):
    assert RT.strict_parse(text) is None


def test_nested_braces_are_balanced_before_parsing():
    assert RT.last_boxed("\\boxed{a{b{c}}d} tail") == "a{b{c}}d"
    assert RT.last_boxed("\\boxed{a{b}") is None


@pytest.mark.parametrize("text,value", [
    ("10-3=7 so 7 apples", Fraction(7)),
    ("the change is -3.", Fraction(-3)),
    ("she has 1,000 apples", Fraction(1000)),
    ("costs 2.50 dollars", Fraction(5, 2)),
    ("no digits at all", None),
])
def test_lenient_parser_takes_the_last_number(text, value):
    assert RT.lenient_parse(text) == value


def test_reference_answers_and_the_verdict():
    assert RT.reference_answer("steps <<2*3=6>>6\n#### 1,000") == Fraction(1000)
    assert RT.reference_answer("x\n#### -10") == Fraction(-10)
    with pytest.raises(ValueError):
        RT.reference_answer("no marker")
    with pytest.raises(ValueError):
        RT.reference_answer("#### twelve")
    assert RT.verdict_fail("\\boxed{18.0}", Fraction(18)) == 0
    assert RT.verdict_fail("\\boxed{17}", Fraction(18)) == 1
    assert RT.verdict_fail("18", Fraction(18)) == 1


def test_every_frozen_reference_answer_parses_and_is_an_integer():
    data = json.loads((HERE / "configs" / "reasoning_dataset.json").read_text())
    cands = data["train_candidates"] + data["test_candidates"]
    assert len(data["train_candidates"]) == RT.N_TRAIN_CANDIDATES
    assert len(data["test_candidates"]) == RT.N_TEST_CANDIDATES
    for c in cands:
        value = RT.parse_fraction_str(c["reference"])
        assert value.denominator == 1
        assert c["prompt"] == RT.prompt_text(c["question"])
        assert RT.verdict_fail(f"\\boxed{{{c['reference']}}}", value) == 0


# ------------------------------------------------------------------ rewards


def test_brevity_refuses_lengths_outside_one_to_L():
    assert RT.brevity([1, 384], 384).tolist() == [1 - 1 / 384, 0.0]
    with pytest.raises(ValueError):
        RT.brevity([0], 384)
    with pytest.raises(ValueError):
        RT.brevity([385], 384)


@pytest.mark.parametrize("L", RT.L_CANDIDATES)
@pytest.mark.parametrize("form", RT.FORMS)
def test_every_correct_member_outranks_every_incorrect_member_at_w2(L, form):
    rng = np.random.default_rng(L)
    for _ in range(2000):
        f = rng.integers(0, 2, size=4)
        if f.all() or not f.any():
            continue
        n = rng.integers(1, L + 1, size=4)
        r = RT.reasoning_rewards(f, n, form, RT.W, 1.0, L)
        assert r[f == 0].min() > r[f == 1].max()
    # The extreme case puts the longest correct answer against the shortest wrong one.
    r = RT.reasoning_rewards([0, 1], [L, 1], form, RT.W, 1.0, L)
    assert r[0] > r[1]


@pytest.mark.parametrize("variant", ["grpo", "drgrpo"])
def test_all_fail_groups_cancel_the_verdict_and_the_gate_zeroes_them(variant):
    rng = np.random.default_rng(7)
    for _ in range(500):
        L = int(rng.choice(RT.L_CANDIDATES))
        n = rng.integers(1, L + 1, size=4)
        fails = np.ones(4, dtype=int)
        linear = RW.advantage_variant(RT.reasoning_rewards(fails, n, "linear", RT.W, 1.0, L), variant)
        task_only = RW.advantage_variant(RT.task_only_rewards(n, 1.0, L), variant)
        assert np.max(np.abs(linear - task_only)) <= 1e-9
        gated = RW.advantage_variant(RT.reasoning_rewards(fails, n, "gated", RT.W, 1.0, L), variant)
        assert np.all(gated == 0.0)
        verdict_only = RW.advantage_variant(RT.reasoning_rewards(fails, n, "linear", RT.W, 0.0, L), variant)
        assert np.all(verdict_only == 0.0)


def test_reward_forms_match_their_definitions():
    r = RT.reasoning_rewards([0, 1], [10, 20], "linear", 2.0, 1.0, 40)
    assert r.tolist() == [1 - 10 / 40, (1 - 20 / 40) - 2.0]
    g = RT.reasoning_rewards([0, 1], [10, 20], "gated", 2.0, 1.0, 40)
    assert g.tolist() == [1 - 10 / 40, -2.0]
    with pytest.raises(ValueError):
        RT.reasoning_rewards([2], [1], "linear", 2.0, 1.0, 40)
    with pytest.raises(ValueError):
        RT.reasoning_rewards([1], [1], "product", 2.0, 1.0, 40)


# ------------------------------------------------------------------ design


def test_arms_follow_the_plan_table():
    assert RT.ARMS["plain"] == {"form": "linear", "w_task": 1.0, "adv_variant": "grpo", "group_filter": "none"}
    assert RT.ARMS["vfilter"]["group_filter"] == "verdict_mixed"
    assert RT.ARMS["drgrpo"]["adv_variant"] == "drgrpo"
    assert RT.ARMS["gate"]["form"] == "gated"
    assert RT.ARMS["verdict_only"]["w_task"] == 0.0
    assert len(RT.STUDY_ARMS["0.5b"]) * 3 + len(RT.STUDY_ARMS["1.5b"]) * 3 == 27
    # The version-1 arms read rewards only. The restored arm of design version 2 (ICML 2027, November
    # 2026) reads the verdict vector too, through the trainer's grpo_fixedref variant, which analyze.py
    # keeps out of ADV_VARIANTS on purpose.
    for name, spec in RT.ARMS.items():
        assert spec["adv_variant"] in RW.TRAINER_ADV_VARIANTS and spec["group_filter"] in RW.GROUP_FILTERS
        if name != "restored":
            assert spec["adv_variant"] in RW.ADV_VARIANTS
    assert RT.ARMS["restored"] == {"form": "linear", "w_task": 1.0, "adv_variant": "grpo_fixedref",
                                   "group_filter": "none"}
    assert "restored" not in RT.STUDY_ARMS["0.5b"] and "restored" not in RT.STUDY_ARMS["1.5b"]
    assert RT.STUDY_ARMS_V2 == ("verdict_only", "plain", "restored", "vfilter")
    assert RT.MIN_MINUTES["pilot"] == 60 and RT.MIN_MINUTES["study"] == 120


def test_expected_all_fail_is_the_unbiased_hypergeometric_estimate():
    assert RT.expected_all_fail(0) == 1.0
    assert RT.expected_all_fail(8) == 0.0
    assert RT.expected_all_fail(1) == pytest.approx(0.5)
    assert RT.expected_all_fail(4) == pytest.approx(1 / 70)
    for c in range(9):
        assert RT.expected_all_fail(c) + RT.expected_all_pass(c) <= 1.0


def test_eval_plan_and_the_step_projection():
    assert RT.eval_plan(0, 200) == [("hard", 4), ("random", 2)]
    assert RT.eval_plan(200, 200) == [("hard", 4), ("random", 2)]
    assert RT.eval_plan(50, 200) == [("hard", 1)]
    assert RT.eval_plan(51, 200) == []
    assert RT.eval_plan(10, 20, RT.SMOKE_MID_EVAL_EVERY) == [("hard", 1)]
    # The run loads in 60 s, takes 20 s per step after the first, 100 s per full evaluation and 25 s per middle one.
    minutes = RT.project_minutes(60, [99] + [20] * 19, 100, 25, 300)
    assert minutes == pytest.approx((60 + 300 * 20 + 200 + 5 * 25) / 60)
    assert RT.choose_steps({300: 130.0, 200: 119.0, 150: 90.0}) == 200
    assert RT.choose_steps({300: 200.0, 200: 150.0, 150: 121.0}) is None


def test_rows_outside_the_design_are_refused():
    base = {"task_id": "1", "kind": "study", "stage": "1", "size": "0.5b",
            "model_id": RT.MODELS["0.5b"]["model_id"], "model_revision": RT.MODELS["0.5b"]["revision"],
            "arm": "gate", "form": "gated", "w": "2.0", "w_task": "1.0", "adv_variant": "grpo",
            "group_filter": "none", "seed": "0", "steps": "200", "L": "384", "micro_batch": "2",
            "dataset_sha256": "a" * 64, "selection_sha256": "b" * 64}
    assert RT.validate_arm_row(dict(base))["steps"] == 200
    for key, bad in [("w", "0.25"), ("form", "linear"), ("micro_batch", "1"), ("L", "300"), ("steps", "250"),
                     ("seed", "3"), ("stage", "2"), ("model_revision", "main"), ("arm", "ngrpo")]:
        row = dict(base, **{key: bad})
        with pytest.raises((ValueError, KeyError)):
            RT.validate_arm_row(row)
    with pytest.raises(ValueError):
        RT.validate_arm_row(dict(base, size="1.5b", model_id=RT.MODELS["1.5b"]["model_id"], stage="2",
                                 arm="verdict_only", form="linear", w_task="0.0"))
    smoke = dict(base, kind="smoke", stage="0", arm="plain", form="linear", seed="100", steps="20")
    assert RT.validate_arm_row(smoke)["kind"] == "smoke"
    with pytest.raises(ValueError):
        RT.validate_arm_row(dict(smoke, seed="0"))
    # A version-1 row never admits the version-2 arm, kind or fields.
    with pytest.raises(ValueError):
        RT.validate_arm_row(dict(base, arm="restored", form="linear", adv_variant="grpo_fixedref"))
    with pytest.raises(ValueError):
        RT.validate_arm_row(dict(base, kind="pilot", stage="3"))
    with pytest.raises(ValueError):
        RT.validate_arm_row(dict(base, lr="1e-05"))
    assert RT.validate_arm_row(dict(base))["design_version"] == 1


def v2_row(**kw):
    row = {"task_id": "1", "kind": "pilot", "stage": "3", "size": "0.5b",
           "model_id": RT.MODELS["0.5b"]["model_id"], "model_revision": RT.MODELS["0.5b"]["revision"],
           "arm": "verdict_only", "form": "linear", "w": "2.0", "w_task": "0.0", "adv_variant": "grpo",
           "group_filter": "none", "seed": "0", "steps": "100", "L": "512", "micro_batch": "2",
           "dataset_sha256": "a" * 64, "selection_sha256": "b" * 64,
           "lr": "1e-06", "kl_beta": "0.0", "skip_zero_adv": "0", "fixedref_c": "0.0"}
    row.update(kw)
    return row


def test_version_2_pilot_rows_change_one_lever_or_the_retry_combination():
    assert tuple(v2_row().keys()) == RT.ARM_HEADER_V2
    out = RT.validate_arm_row(v2_row())
    assert out["design_version"] == 2 and out["lr"] == 1e-6 and out["kl_beta"] == 0.0
    assert out["skip_zero_adv"] == 0 and out["fixedref_c"] == 0.0 and out["steps"] == 100
    for change in ({"lr": "5e-07"}, {"lr": "1e-05", "kl_beta": "0.02"}, {"lr": "1e-05", "kl_beta": "0.05"},
                   {"lr": "1e-05", "skip_zero_adv": "1"}):
        RT.validate_arm_row(v2_row(**change))
    # The retry combination is the one two-change row the design allows.
    RT.validate_arm_row(v2_row(lr="1e-05", kl_beta="0.02", skip_zero_adv="1"))
    assert RT.PILOT_RETRY_COMBINATION == {"lr": 1e-5, "kl_beta": 0.02, "skip_zero_adv": 1}
    for bad in ({"lr": "1e-05"},                                            # no change at all
                {"lr": "1e-06", "kl_beta": "0.02"},                          # two changes, not the retry
                {"lr": "1e-05", "kl_beta": "0.05", "skip_zero_adv": "1"},    # two changes, not the retry
                {"lr": "2e-06"}, {"lr": "1e-05", "kl_beta": "0.1"}, {"skip_zero_adv": "2"},
                {"stage": "4"}, {"arm": "plain", "w_task": "1.0"}, {"seed": "1"}, {"steps": "300"},
                {"L": "384"}, {"fixedref_c": "2.0"}, {"kind": "smoke"}, {"kind": "study"}):
        with pytest.raises(ValueError):
            RT.validate_arm_row(v2_row(**bad))
    assert RT.pilot_changes(1e-6, 0.0, 0) == ["lr"] and RT.pilot_changes(1e-5, 0.02, 1) == ["kl_beta", "skip_zero_adv"]


def test_version_2_study_rows_are_stage_4_at_300_steps_and_the_restored_arm_carries_c():
    study = v2_row(kind="study", stage="4", lr="1e-06", steps="300")
    for arm in RT.STUDY_ARMS_V2:
        spec = RT.ARMS[arm]
        c = "2.0" if arm == "restored" else "0.0"
        row = dict(study, arm=arm, form=spec["form"], w_task=repr(float(spec["w_task"])),
                   adv_variant=spec["adv_variant"], group_filter=spec["group_filter"], fixedref_c=c)
        for seed in RT.SEEDS:
            out = RT.validate_arm_row(dict(row, seed=str(seed)))
            assert out["design_version"] == 2 and out["fixedref_c"] == float(c)
        with pytest.raises(ValueError):
            RT.validate_arm_row(dict(row, fixedref_c="2.0" if c == "0.0" else "0.0"))
    # A study row may combine every lever, because the pilot chose them together.
    RT.validate_arm_row(dict(study, lr="5e-07", kl_beta="0.05", skip_zero_adv="1"))
    for bad in ({"stage": "3"}, {"steps": "200"}, {"seed": "3"}, {"arm": "gate", "form": "gated", "w_task": "1.0"},
                {"arm": "drgrpo", "adv_variant": "drgrpo", "w_task": "1.0"}, {"L": "384"}):
        with pytest.raises(ValueError):
            RT.validate_arm_row(dict(study, **bad))
    with pytest.raises(ValueError):
        RT.validate_arm_row(dict(study, size="1.5b", model_id=RT.MODELS["1.5b"]["model_id"]))


# ------------------------------------------------------------------ frozen sources


def test_frozen_training_sources_are_unchanged():
    assert TSG.training_sources_digest() == FROZEN


@pytest.mark.skipif(not (HERE / "slurm" / "reasoning_arms.sbatch").is_file(),
                    reason="the cluster job scripts are not part of the anonymous code archive")
def test_reasoning_sources_list_the_frozen_sources_then_the_cluster_files_only():
    assert RT.REASONING_SOURCES[:len(TSG.TRAINING_SOURCES)] == TSG.TRAINING_SOURCES
    new = RT.REASONING_SOURCES[len(TSG.TRAINING_SOURCES):]
    # reasoning_arms_v2.sbatch joined the list for design version 2 (ICML 2027, November 2026).
    assert [p.rsplit("/", 1)[1] for p in new] == [
        "reasoning_task.py", "calibrate_reasoning.py", "train_reasoning_grpo.py",
        "download_reasoning.sbatch", "reasoning_calibrate.sbatch", "reasoning_arms.sbatch",
        "reasoning_arms_v2.sbatch"]
    for path in RT.REASONING_SOURCES:
        low = path.lower()
        assert not any(bad in low for bad in ("ledger", "predictions", "analy", "/tests/", "test_"))
        assert (HERE / path[len("experiments/cancellation/"):]).is_file()
    assert len(set(RT.REASONING_SOURCES)) == len(RT.REASONING_SOURCES)


@pytest.mark.skipif(not (HERE / "slurm" / "reasoning_arms.sbatch").is_file(),
                    reason="the cluster job scripts are not part of the anonymous code archive")
def test_the_source_digest_changes_with_any_source_byte(tmp_path):
    import shutil
    for rel in RT.REASONING_SOURCES:
        src = HERE / rel[len("experiments/cancellation/"):]
        dst = tmp_path / rel[len("experiments/cancellation/"):]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    assert RT.reasoning_sources_digest(tmp_path) == RT.reasoning_sources_digest()
    target = tmp_path / "reasoning_task.py"
    target.write_bytes(target.read_bytes() + b"\n")
    assert RT.reasoning_sources_digest(tmp_path) != RT.reasoning_sources_digest()
