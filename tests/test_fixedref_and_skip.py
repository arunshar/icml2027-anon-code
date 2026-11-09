"""The ICML 2027 additions to the trainer (November 2026), on the random tiny model.

Three things are checked, each off by default so the frozen path is untouched:
  * the fixed-reference variant for the verifier component, `grpo_fixedref`: the plain z-score minus
    c in an all-violating group, plus c in an all-passing group, unchanged on a mixed group, and what
    a run logs under it is what the reward code recomputes from the logged speeds and verdicts;
  * `--skip-zero-adv`: a kept group whose advantages are all zero does not enter the update, so a step
    with no other group takes no optimizer step and leaves the parameter checksum where it was, while
    the same step without the flag moves the weights through AdamW's momentum;
  * held-out prompt evaluation: a task that declares HELDOUT_PROMPTS is evaluated on them after the
    training prompts, and the training-prompt samples are identical with or without them.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

torch = pytest.importorskip("torch")

import rewards as RW  # noqa: E402
import tasks  # noqa: E402
import train_speed_grpo as T  # noqa: E402
from speed_task import SPEC as SPEED_SPEC  # noqa: E402

BASE = ["--arm", "binary", "--w", "5", "--vmax", "5", "--tiny", "--device", "cpu",
        "--eval-every", "0", "--batch-prompts", "2", "--seed", "0"]


def _read(out):
    recs = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    cfg = json.load(open(os.path.join(out, "config.json")))
    prov = json.load(open(os.path.join(out, "provenance.json")))
    return (cfg, prov, [r for r in recs if r["phase"] == "train"],
            [r for r in recs if r["phase"] == "eval"])


def _run(tmp_path, name, extra, steps=2):
    out = tmp_path / name
    assert T.main(BASE + ["--steps", str(steps), "--out-dir", str(out)] + list(extra)) == 0
    return _read(out)


def _strip(train):
    return [{k: v for k, v in r.items() if k != "s_per_step"} for r in train]


# ---------------------------------------------------------------- advantage_fixedref


@pytest.mark.parametrize("c", [0.5, 2.0, 7.0])
def test_an_all_violating_group_is_shifted_down_by_exactly_c(c):
    r = [1.0, 2.5, 3.0, 4.0]
    z = RW.advantage(r)
    got = RW.advantage_fixedref(r, [False] * 4, c)
    assert np.array_equal(got, z - c)
    assert got - z == pytest.approx(-c, abs=1e-12)
    assert got.sum() == pytest.approx(z.sum() - 4 * c, abs=1e-12)


@pytest.mark.parametrize("c", [0.5, 2.0, 7.0])
def test_an_all_passing_group_is_shifted_up_by_exactly_c(c):
    r = [1.0, 2.5, 3.0, 4.0]
    z = RW.advantage(r)
    got = RW.advantage_fixedref(r, [True] * 4, c)
    assert np.array_equal(got, z + c)
    assert got - z == pytest.approx(c, abs=1e-12)
    assert got.sum() == pytest.approx(z.sum() + 4 * c, abs=1e-12)


@pytest.mark.parametrize("verdicts", [[True, False, False, False], [False, True, True, True],
                                      [True, False, True, False]])
def test_a_mixed_group_is_the_frozen_z_score_itself(verdicts):
    r = [1.0, 2.5, 3.0, 4.0]
    got = RW.advantage_fixedref(r, verdicts, 2.0)
    assert np.array_equal(got, RW.advantage(r))
    assert np.array_equal(got, RW.advantage_variant(r, "grpo"))


def test_an_identical_reward_all_violating_group_gets_a_uniform_negative_advantage():
    """The case the variant exists for: the z-score of identical rewards is exactly zero, and the
    fixed reference turns it into -c on every member."""
    r = [-15.0] * 4
    assert np.array_equal(RW.advantage(r), np.zeros(4))
    assert np.array_equal(RW.advantage_fixedref(r, [False] * 4, 2.0), np.full(4, -2.0))
    assert np.array_equal(RW.advantage_fixedref(r, [True] * 4, 2.0), np.full(4, 2.0))


def test_advantage_fixedref_refuses_mismatched_or_empty_verdicts():
    with pytest.raises(ValueError):
        RW.advantage_fixedref([1.0, 2.0, 3.0], [True, False], 2.0)
    with pytest.raises(ValueError):
        RW.advantage_fixedref([], [], 2.0)


def test_the_old_signature_refuses_the_new_variant_and_asks_for_verdicts():
    with pytest.raises(ValueError, match="needs verdicts"):
        RW.advantage_variant([1.0, 2.0, 3.0, 4.0], "grpo_fixedref")
    with pytest.raises(ValueError, match="needs verdicts"):
        RW.advantage_variant([1.0, 2.0, 3.0, 4.0], "grpo_fixedref", None)


def test_the_new_variant_is_a_trainer_choice_with_no_anchor():
    assert "grpo_fixedref" in RW.TRAINER_ADV_VARIANTS
    assert "grpo_fixedref" in RW.VERDICT_VARIANTS
    assert "grpo_fixedref" not in RW.ANCHORED_VARIANTS
    # analyze.py enumerates ADV_VARIANTS through the rewards-only signature, so the verdict variant
    # stays out of that tuple and the frozen replay is unchanged.
    assert "grpo_fixedref" not in RW.ADV_VARIANTS
    assert RW.TRAINER_ADV_VARIANTS[:len(RW.ADV_VARIANTS)] == RW.ADV_VARIANTS
    assert RW.reward_anchor("grpo_fixedref", "binary", 5.0, 5.0) is None
    assert RW.reward_anchor("grpo_fixedref", "none", 0.0, 0.03, spec=tasks.get("capacity"),
                            instance=1) is None
    assert T.build_arg_parser().parse_args(BASE + ["--out-dir", "x", "--adv-variant",
                                                   "grpo_fixedref"]).adv_variant == "grpo_fixedref"


def test_group_adv_is_zero_reads_the_tolerance():
    assert RW.ZERO_ADV_TOL == 1e-12
    assert RW.group_adv_is_zero(np.zeros(4)) is True
    assert RW.group_adv_is_zero([0.0, 1e-13, -1e-13, 0.0]) is True
    assert RW.group_adv_is_zero([0.0, 0.0, 0.0, 1e-6]) is False
    assert RW.group_adv_is_zero([-5e-13]) is True
    assert RW.group_adv_is_zero([]) is True
    assert RW.group_adv_is_zero(RW.advantage([3.0, 3.0, 3.0, 3.0])) is True
    assert RW.group_adv_is_zero(RW.advantage([3.0, 3.0, 3.0, 4.0])) is False
    assert RW.group_adv_is_zero(RW.advantage_fixedref([3.0] * 4, [False] * 4, 2.0)) is False


# ---------------------------------------------------------------- the variant in the trainer


def _initial_checksum(argv):
    args = T.build_arg_parser().parse_args(argv)
    T.seed_everything(args.seed)
    _, model, _ = T.build_models(args, "cpu")
    return T.param_checksum(model)


@pytest.mark.parametrize("c_flag", [[], ["--fixedref-c", "0.5"]])
def test_a_fixedref_run_logs_what_the_reward_code_recomputes(tmp_path, c_flag):
    extra = ["--adv-variant", "grpo_fixedref"] + c_flag
    cfg, prov, train, _ = _run(tmp_path, "fixedref", extra)
    assert cfg["adv_variant"] == "grpo_fixedref"
    assert cfg["fixedref_c"] == (0.5 if c_flag else 2.0)
    assert cfg["skip_zero_adv"] is False
    assert [r["step"] for r in train] == [1, 2]
    homogeneous = 0
    for rec in train:
        for gi, g in enumerate(rec["groups"]):
            speeds = g["speeds"]
            r = RW.group_rewards(speeds, cfg["arm"], cfg["w"], cfg["vmax"])
            assert r.tolist() == g["rewards"]
            verdicts = RW.group_verdicts(speeds, cfg["vmax"])
            assert g["verdicts"] == verdicts
            assert g["anchor"] is None
            want = RW.advantage_fixedref(r, verdicts, cfg["fixedref_c"])
            assert want.tolist() == g["adv"]
            assert g["zero_adv"] is RW.group_adv_is_zero(want)
            assert g["kept"] is True and g["kept_for_update"] is True
            shift = np.array(g["adv"]) - RW.advantage(r)
            if all(verdicts) or not any(verdicts):
                homogeneous += 1
                sign = 1.0 if all(verdicts) else -1.0
                assert shift == pytest.approx(sign * cfg["fixedref_c"], abs=1e-12)
                assert g["zero_adv"] is False, "a shifted group never has a zero advantage"
            else:
                assert np.array_equal(shift, np.zeros_like(shift))
        assert rec["n_kept_groups"] == len(rec["groups"]) == rec["n_update_groups"]
        assert rec["n_zero_adv_groups"] == sum(g["zero_adv"] for g in rec["groups"])
        assert rec["opt_step_taken"] is True
    assert homogeneous > 0, "precondition: the tiny model emits at least one homogeneous group"
    # A shifted group has a nonzero advantage even without reward spread, so the run moves weights.
    assert prov["final_param_checksum"] != _initial_checksum(
        BASE + ["--steps", "2", "--out-dir", "x"] + extra)
    assert prov["training_sources_digest"] == T.training_sources_digest()


def test_a_fixedref_run_differs_from_the_plain_run_only_on_homogeneous_groups(tmp_path):
    """Both runs sample the same first step, because the seed and the weights agree there."""
    _, _, plain, _ = _run(tmp_path, "plain", [], steps=1)
    _, _, fixed, _ = _run(tmp_path, "fixed", ["--adv-variant", "grpo_fixedref"], steps=1)
    for gp, gf in zip(plain[0]["groups"], fixed[0]["groups"]):
        assert gp["speeds"] == gf["speeds"] and gp["rewards"] == gf["rewards"]
        assert gp["verdicts"] == gf["verdicts"]
        v = gf["verdicts"]
        if all(v) or not any(v):
            assert np.array(gf["adv"]) - np.array(gp["adv"]) == pytest.approx(
                (2.0 if all(v) else -2.0), abs=1e-12)
        else:
            assert gf["adv"] == gp["adv"]


@pytest.mark.parametrize("extra", [
    ["--kl-beta", "0.04", "--adv-variant", "grpo_fixedref"],
    ["--keep-schedule", "any", "--adv-variant", "grpo_fixedref"],
])
def test_fixedref_is_refused_with_a_kl_term_or_a_keep_schedule(tmp_path, extra):
    out = tmp_path / "refused"
    with pytest.raises(SystemExit) as e:
        T.main(BASE + ["--steps", "1", "--out-dir", str(out)] + extra)
    assert e.value.code == 2
    assert not out.exists()


def test_a_default_run_logs_the_new_fields_and_keeps_the_update_decision(tmp_path):
    cfg, _, train, _ = _run(tmp_path, "plain", [])
    assert cfg["adv_variant"] == "grpo" and cfg["fixedref_c"] == 2.0 and cfg["skip_zero_adv"] is False
    for rec in train:
        for g in rec["groups"]:
            assert len(g["verdicts"]) == cfg["group_size"]
            assert all(isinstance(v, bool) for v in g["verdicts"])
            assert g["verdicts"] == RW.group_verdicts(g["speeds"], cfg["vmax"])
            assert isinstance(g["zero_adv"], bool)
            assert g["zero_adv"] is RW.group_adv_is_zero(g["adv"])
            assert g["kept_for_update"] is g["kept"]
        assert rec["n_update_groups"] == rec["n_kept_groups"] == len(rec["groups"])
        assert rec["n_zero_adv_groups"] == sum(g["zero_adv"] for g in rec["groups"])
        # Off by default: a zero-advantage group still enters the update.
        assert rec["opt_step_taken"] is True


# ---------------------------------------------------------------- --skip-zero-adv

PUSH = np.array([1.5, -0.5, -0.5, -0.5])
GROUPS_PER_STEP = 2   # BASE uses --batch-prompts 2


def _force_advantages(monkeypatch, n_pushed_calls):
    """Replace the advantage with a pushed vector for the first calls and zeros afterwards.

    The trainer calls RW.advantage_variant once per group per step, so with two groups per step the
    first two calls are step one. This is the technique the momentum test in test_remedy_trainer.py
    uses on policy_update, here applied through a whole run."""
    calls = {"n": 0}

    def fake(rewards, variant="grpo", anchor=None):
        calls["n"] += 1
        return PUSH.copy() if calls["n"] <= n_pushed_calls else np.zeros(len(rewards))

    monkeypatch.setattr(RW, "advantage_variant", fake)


def test_skip_zero_adv_takes_no_step_and_leaves_the_checksum_where_it_was(tmp_path, monkeypatch):
    """Step one has pushed advantages in every run, so AdamW's first moment is nonzero afterwards.
    Step two has zero advantages in every group. With the skip on, no group enters the update and
    the weights stay at their step-one values. Without the skip, the zero-gradient step still moves
    the weights through the momentum, which is what the flag exists to prevent."""
    _force_advantages(monkeypatch, GROUPS_PER_STEP)
    _, prov_one, train_one, _ = _run(tmp_path, "one_step", [], steps=1)
    after_first = prov_one["final_param_checksum"]
    assert train_one[0]["opt_step_taken"] is True and train_one[0]["grad_norm_preclip"] > 0

    _force_advantages(monkeypatch, GROUPS_PER_STEP)
    cfg_s, prov_s, train_s, _ = _run(tmp_path, "skip", ["--skip-zero-adv"], steps=2)
    assert cfg_s["skip_zero_adv"] is True
    assert _strip(train_s[:1]) == _strip(train_one), "the two runs share the first step"
    second = train_s[1]
    assert second["opt_step_taken"] is False
    assert second["grad_norm_preclip"] is None and second["pg_loss"] is None
    assert second["n_kept_groups"] == GROUPS_PER_STEP
    assert second["n_zero_adv_groups"] == GROUPS_PER_STEP
    assert second["n_update_groups"] == 0
    for g in second["groups"]:
        assert g["adv"] == [0.0] * 4
        assert g["zero_adv"] is True
        assert g["kept"] is True, "the filter decision is untouched"
        assert g["kept_for_update"] is False
    assert prov_s["final_param_checksum"] == after_first, \
        "a skipped step must leave the weights exactly where they were"

    _force_advantages(monkeypatch, GROUPS_PER_STEP)
    cfg_m, prov_m, train_m, _ = _run(tmp_path, "momentum", [], steps=2)
    assert cfg_m["skip_zero_adv"] is False
    assert _strip(train_m[:1]) == _strip(train_one)
    second_m = train_m[1]
    assert second_m["opt_step_taken"] is True and second_m["grad_norm_preclip"] == 0.0
    assert second_m["n_update_groups"] == GROUPS_PER_STEP == second_m["n_zero_adv_groups"]
    assert all(g["kept_for_update"] is True and g["zero_adv"] is True for g in second_m["groups"])
    assert prov_m["final_param_checksum"] != after_first, \
        "a zero-gradient AdamW step after a nonzero one must move the weights"
    # The weights agreed after step one, so step two sampled the same completions in both runs.
    assert [g["speeds"] for g in second["groups"]] == [g["speeds"] for g in second_m["groups"]]
    assert [g["n_tokens"] for g in second["groups"]] == [g["n_tokens"] for g in second_m["groups"]]


def test_skip_zero_adv_removes_only_the_zero_groups_from_the_update(tmp_path, monkeypatch):
    # Three pushed calls: both groups of step one and the first group of step two.
    _force_advantages(monkeypatch, GROUPS_PER_STEP + 1)
    _, _, train, _ = _run(tmp_path, "partial", ["--skip-zero-adv"], steps=2)
    second = train[1]
    assert [g["zero_adv"] for g in second["groups"]] == [False, True]
    assert [g["kept"] for g in second["groups"]] == [True, True]
    assert [g["kept_for_update"] for g in second["groups"]] == [True, False]
    assert second["n_kept_groups"] == 2 and second["n_zero_adv_groups"] == 1
    assert second["n_update_groups"] == 1
    assert second["opt_step_taken"] is True and second["grad_norm_preclip"] > 0


def test_the_skip_composes_with_the_filter_decision(tmp_path, monkeypatch):
    """A group the filter drops is not an update group whether or not its advantage is zero, and
    the filter decision is what g["kept"] records."""
    _force_advantages(monkeypatch, 0)
    _, prov, train, _ = _run(tmp_path, "filtered", ["--skip-zero-adv", "--group-filter",
                                                     "verdict_mixed"], steps=1)
    rec = train[0]
    for g in rec["groups"]:
        v = g["verdicts"]
        assert g["kept"] is (0 < sum(v) < len(v))
        assert g["kept_for_update"] is (g["kept"] and not g["zero_adv"])
    assert rec["n_update_groups"] == sum(g["kept_for_update"] for g in rec["groups"])
    assert rec["opt_step_taken"] is (rec["n_update_groups"] > 0)


# ---------------------------------------------------------------- held-out prompts

HELDOUT = [
    "Name a cruising speed in knots for this vessel and reply with the integer alone.",
    "What cruising speed in knots should the vessel hold? Give only the integer.",
]


class _HeldoutSpeedSpec(type(SPEED_SPEC)):
    """The speed task with a held-out prompt list the test sets per run."""
    HELDOUT_PROMPTS: list = []


@pytest.fixture
def heldout_task():
    spec = _HeldoutSpeedSpec(
        name="speed_heldout_test", prompts=SPEED_SPEC.PROMPTS, parse=SPEED_SPEC.parse,
        verdict=SPEED_SPEC.verdict, residual=SPEED_SPEC.residual, task_term=SPEED_SPEC.task_term,
        synthesize=SPEED_SPEC.synthesize, unparseable_task=SPEED_SPEC.UNPARSEABLE_TASK,
        items_key=SPEED_SPEC.ITEMS_KEY, scalar_key=SPEED_SPEC.SCALAR_KEY,
        eval_scalar_key=SPEED_SPEC.EVAL_SCALAR_KEY, eval_items_key=SPEED_SPEC.EVAL_ITEMS_KEY)
    tasks.register(spec)
    try:
        yield spec
    finally:
        tasks._REGISTRY.pop(spec.name, None)


def test_neither_built_in_task_declares_held_out_prompts():
    for name in ("speed", "capacity"):
        assert not getattr(tasks.get(name), "HELDOUT_PROMPTS", None)


HELDOUT_KEYS = ("eval_heldout_n", "eval_heldout_violation_rate", "eval_heldout_parse_fail_rate",
                "eval_heldout_mean_residual", "eval_heldout_items")


def test_held_out_prompts_leave_the_training_prompt_samples_unchanged(tmp_path, heldout_task):
    extra = ["--task", heldout_task.name, "--eval-every", "2", "--eval-samples", "2"]
    heldout_task.HELDOUT_PROMPTS = list(HELDOUT)
    _, prov_h, train_h, evals_h = _run(tmp_path, "with_heldout", extra)
    heldout_task.HELDOUT_PROMPTS = []
    _, prov_0, train_0, evals_0 = _run(tmp_path, "without_heldout", extra)

    assert [e["step"] for e in evals_h] == [0, 2] == [e["step"] for e in evals_0]
    key = heldout_task.EVAL_ITEMS_KEY
    for eh, e0 in zip(evals_h, evals_0):
        assert eh[key] == e0[key], "the training-prompt eval items must not move"
        for k in ("eval_n", "eval_violation_rate", "eval_parse_fail_rate", "eval_mean_residual",
                  heldout_task.EVAL_SCALAR_KEY):
            assert eh[k] == e0[k]
        assert not any(k in e0 for k in HELDOUT_KEYS)
        assert all(k in eh for k in HELDOUT_KEYS)
        assert eh["eval_heldout_n"] == len(HELDOUT) * 2 == len(eh["eval_heldout_items"])
        items = eh["eval_heldout_items"]
        assert all(s is None or isinstance(s, float) for s in items)
        present = [s for s in items if s is not None]
        assert eh["eval_heldout_parse_fail_rate"] == (len(items) - len(present)) / len(items)
        assert eh["eval_heldout_violation_rate"] == (
            sum(1 for s in items if not heldout_task.verdict(s, 5.0)) / len(items))
        want_resid = ([heldout_task.residual(s, 5.0) for s in present] or None)
        if want_resid is None:
            assert eh["eval_heldout_mean_residual"] is None
        else:
            assert eh["eval_heldout_mean_residual"] == pytest.approx(float(np.mean(want_resid)))
    # The training RNG is restored after the eval, so the training steps agree too.
    assert _strip(train_h) == _strip(train_0)
    assert prov_h["final_param_checksum"] == prov_0["final_param_checksum"]


def test_evaluate_calls_generate_for_the_training_prompts_first(heldout_task, monkeypatch):
    heldout_task.HELDOUT_PROMPTS = list(HELDOUT)
    argv = BASE + ["--out-dir", os.devnull, "--no-ref", "--eval-samples", "2", "--task",
                   heldout_task.name]
    args = T.build_arg_parser().parse_args(argv)
    T.seed_everything(args.seed)
    tok, model, _ = T.build_models(args, "cpu")
    eos = sorted({i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None})
    seen = []
    real = T.sample_completions

    def spy(model_, tok_, ids, args_, n):
        seen.append(tok_.decode(ids[0], skip_special_tokens=True))
        return real(model_, tok_, ids, args_, n)

    monkeypatch.setattr(T, "sample_completions", spy)
    ev = T.evaluate(model, tok, "cpu", args, eos, heldout_task)
    assert len(seen) == args.batch_prompts + len(HELDOUT)
    for text, prompt in zip(seen, heldout_task.PROMPTS[: args.batch_prompts] + HELDOUT):
        assert prompt in text
    assert ev["eval_n"] == args.batch_prompts * 2 and ev["eval_heldout_n"] == len(HELDOUT) * 2
