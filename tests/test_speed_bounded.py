"""Block A' of the ICML 2027 plan (November 2026): the bounded speed task, its manifest and its replay.

Five kinds of check:
  * the registered task is the speed task in every part but the task term, and the task term is
    strictly increasing on every parseable magnitude and below 20, so the crossover of Eq. 2 computed
    from the output table sits just below 20 and no flat region exists;
  * a two-step run of each of the four arms on the random tiny model, with the zero-advantage skip
    on, logs rewards and advantages the reward code recomputes, and every eval record carries the
    held-out keys;
  * the replay script reproduces those logs, and on a constructed log with every kind of group it
    counts each kind exactly and finds the light- and heavy-weight advantages of all-violating groups
    identical;
  * the committed manifest is the generator's output, has 39 rows with the stated counts per arm and
    regime, and every row is a flag combination the trainer accepts;
  * the generator refuses a row the design does not define.
"""
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import make_bounded_manifest as MB  # noqa: E402
import replay_bounded_scale as RB  # noqa: E402
import rewards as RW  # noqa: E402
import speed_bounded_task as SB  # noqa: E402
import speed_task as ST  # noqa: E402
import tasks  # noqa: E402

MANIFEST = os.path.join(HERE, "slurm", "manifest_speed_bounded.csv")
HELDOUT_KEYS = ("eval_heldout_n", "eval_heldout_violation_rate", "eval_heldout_parse_fail_rate",
                "eval_heldout_mean_residual", "eval_heldout_items")


# ---------------------------------------------------------------- the task


def test_the_task_is_registered_as_the_speed_task_with_a_new_task_term():
    spec = tasks.get("speed_bounded")
    assert spec is SB.SPEC and "speed_bounded" in tasks.names()
    assert spec.PROMPTS == ST.PROMPTS and len(spec.PROMPTS) == 6
    assert spec.parse is ST.parse_speed
    assert spec.verdict is ST.SPEC.verdict and spec.residual is ST.SPEC.residual
    assert spec.synthesize is ST.SPEC.synthesize
    assert spec.UNPARSEABLE_TASK == -10.0 == RW.UNPARSEABLE_TASK
    assert spec.task_term is SB.task_term and spec.task_term is not ST.SPEC.task_term
    assert (spec.ITEMS_KEY, spec.SCALAR_KEY, spec.EVAL_SCALAR_KEY, spec.EVAL_ITEMS_KEY) == \
        ("speeds", "mean_speed", "eval_mean_speed", "eval_speeds")
    # the speed task itself is untouched
    assert tasks.get("speed") is ST.SPEC and ST.SPEC.task_term(300.0, 5.0) == 20.0
    assert not getattr(ST.SPEC, "HELDOUT_PROMPTS", None)


def test_the_held_out_prompts_are_two_new_wordings_that_name_no_limit():
    held = SB.SPEC.HELDOUT_PROMPTS
    assert held == SB.HELDOUT_PROMPTS and len(held) == 2 and len(set(held)) == 2
    for p in held:
        assert p not in ST.PROMPTS
        low = p.lower()
        assert "knots" in low and "integer" in low and "cruising speed" in low
        assert not any(ch.isdigit() for ch in p), "a held-out prompt must not state a limit"
        assert "limit" not in low and "maximum" not in low and "exceed" not in low


def test_the_task_term_is_strictly_increasing_on_every_parseable_magnitude_and_below_twenty():
    for v_max in (5.0, 30.0):
        table = SB.SPEC.output_table(v_max)
        t = table["task"]
        assert t.shape == (RW.SPEED_ENUM_MAX + 1,)
        assert t[0] == 0.0
        assert bool(np.all(np.diff(t) > 0)), "T has a flat step somewhere in 0..SPEED_ENUM_MAX"
        assert float(t.max()) < SB.TASK_SUP == 20.0
        # the residual and the verdict are the speed task's
        ref = RW.speed_output_table(v_max)
        assert np.array_equal(table["residual"], ref["residual"])
        assert np.array_equal(table["verdict"], ref["verdict"])
        assert not np.array_equal(table["task"], ref["task"])


def test_the_scalar_task_term_matches_the_table_and_depends_on_the_magnitude_only():
    table = SB.SPEC.output_table(5.0)["task"]
    for s in (0, 1, 4, 5, 6, 100, 200, 201, 999_999):
        assert SB.task_term(float(s), 5.0) == table[s]
        assert SB.task_term(-float(s), 5.0) == table[s]
        assert SB.task_term(float(s), 30.0) == table[s], "the task term does not read the limit"
    assert SB.task_term(100.0) == 10.0, "T reaches half its supremum at TASK_HALF"
    assert SB.task_term(None) == RW.UNPARSEABLE_TASK


def test_the_crossover_at_the_hard_limit_is_just_below_twenty():
    """Eq. 2: the largest task value of a violator minus the smallest task value of a feasible output.
    Computed from the output table at v_max = 5 on 2026-11-06, the value is 19.9980001979804, which
    is T(999999) - T(0) = 20 * 999999 / 1000099. The supremum of T is 20, so the crossover is 20 under
    a supremum, and this is the plan's statement that w* stays 20 under a supremum."""
    table = SB.SPEC.output_table(5.0)
    violator = ~table["verdict"]
    w_star = float(table["task"][violator].max() - table["task"][table["verdict"]].min())
    assert 19.99 < w_star < 20.0
    assert w_star == SB.crossover_weight(5.0)
    assert w_star == pytest.approx(19.9980001979804, abs=1e-12)
    assert w_star == SB.task_term(float(RW.SPEED_ENUM_MAX)) - SB.task_term(0.0)
    # the same value at the easy limit, because T does not read the limit and 0 is feasible at both
    assert SB.crossover_weight(30.0) == w_star
    # the light verdict sits below the crossover, the heavy verdict above it
    assert MB.W_LIGHT < w_star < MB.W_HEAVY


def test_an_all_violating_group_with_distinct_speeds_keeps_reward_spread_under_every_arm():
    """The reason this task exists: two violators far above the limit still differ on the task term."""
    speeds = [300.0, 1000.0, 50_000.0, 999_999.0]
    assert RW.group_is_all_violating(speeds, 5.0, spec=SB.SPEC)
    for arm, w in (("none", 0.0), ("binary", 5.0), ("binary", 5000.0), ("graded", 5.0)):
        r = RW.group_rewards(speeds, arm, w, 5.0, spec=SB.SPEC)
        assert np.ptp(r) > RW.ZERO_SPREAD_TOL
        assert len(set(r.tolist())) == 4
    # on the frozen speed task the same group ties on the task term under the verdict
    r_speed = RW.group_rewards(speeds, "binary", 5.0, 5.0)
    assert np.ptp(r_speed) == 0.0


def test_the_unparseable_rule_prices_a_failed_parse_as_the_speed_task_does():
    for arm, w in (("none", 0.0), ("binary", 5.0), ("binary", 5000.0)):
        r = RW.group_rewards([None, 4.0, None, 9.0], arm, w, 5.0, spec=SB.SPEC)
        pen = {"none": 0.0, "binary": -1.0}[arm]
        assert r[0] == r[2] == RW.UNPARSEABLE_TASK + w * pen
        assert r[1] == SB.task_term(4.0) and r[3] == SB.task_term(9.0) + w * pen
    r = RW.group_rewards([None] * 4, "binary", 5.0, 5.0, spec=SB.SPEC)
    assert r.tolist() == [-15.0] * 4


# ---------------------------------------------------------------- tiny runs

torch = pytest.importorskip("torch")

import train_speed_grpo as T  # noqa: E402

# The random model rarely emits a digit, so the runs use every prompt, a longer generation and the seed
# at which the golden speed case parses a group.
BASE = ["--task", "speed_bounded", "--vmax", "5", "--tiny", "--device", "cpu", "--eval-every", "2",
        "--eval-samples", "2", "--batch-prompts", "6", "--seed", "3", "--max-new-tokens", "24",
        "--skip-zero-adv", "--steps", "2"]
ARMS = {
    "none": ["--arm", "none", "--w", "0"],
    "binary_w5": ["--arm", "binary", "--w", "5"],
    "binary_w5000": ["--arm", "binary", "--w", "5000"],
    "binary_w5_fixedref": ["--arm", "binary", "--w", "5", "--adv-variant", "grpo_fixedref",
                           "--fixedref-c", "2.0"],
    "binary_w5_fixedref_k8": ["--arm", "binary", "--w", "5", "--adv-variant", "grpo_fixedref",
                              "--fixedref-c", "2.0", "--group-size", "8"],
}


def _read(out):
    recs = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    cfg = json.load(open(os.path.join(out, "config.json")))
    prov = json.load(open(os.path.join(out, "provenance.json")))
    return (cfg, prov, [r for r in recs if r["phase"] == "train"],
            [r for r in recs if r["phase"] == "eval"])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One two-step run per arm, shared by the tests below."""
    root = tmp_path_factory.mktemp("bounded_runs")
    out = {}
    for name, extra in ARMS.items():
        d = root / name
        assert T.main(BASE + ["--out-dir", str(d)] + extra) == 0
        out[name] = str(d)
    return out


def _check_logged_against_recomputation(cfg, train):
    spec = tasks.get(cfg["task"])
    assert spec is SB.SPEC
    for rec in train:
        assert len(rec["groups"]) == 6
        for gi, g in enumerate(rec["groups"]):
            items = g["speeds"]
            assert g["prompt_index"] == gi
            assert len(items) == cfg["group_size"] == len(g["rewards"]) == len(g["adv"])
            r = RW.group_rewards(items, cfg["arm"], cfg["w"], cfg["vmax"], spec=spec, w_task=cfg["w_task"])
            assert r.tolist() == g["rewards"]
            verdicts = RW.group_verdicts(items, cfg["vmax"], spec=spec)
            assert g["verdicts"] == verdicts
            if cfg["adv_variant"] == "grpo_fixedref":
                assert g["anchor"] is None
                assert RW.advantage_fixedref(r, verdicts, cfg["fixedref_c"]).tolist() == g["adv"]
                if not any(verdicts):
                    assert np.allclose(np.array(g["adv"]) - RW.advantage(r), -cfg["fixedref_c"])
            else:
                assert g["anchor"] is None
                assert RW.advantage_variant(r, cfg["adv_variant"], None).tolist() == g["adv"]
            assert g["all_violating"] is (not any(verdicts))
            assert g["has_violator"] is (not all(verdicts))
            assert g["kept"] is True
            assert g["zero_adv"] is RW.group_adv_is_zero(g["adv"])
            assert g["kept_for_update"] is (not g["zero_adv"])
        assert rec["n_update_groups"] == sum(g["kept_for_update"] for g in rec["groups"])
        assert rec["opt_step_taken"] is (rec["n_update_groups"] > 0)


@pytest.mark.parametrize("name", list(ARMS))
def test_a_two_step_run_of_each_arm_logs_what_the_reward_code_recomputes(runs, name):
    cfg, prov, train, evals = _read(runs[name])
    assert cfg["task"] == "speed_bounded" and cfg["skip_zero_adv"] is True
    assert cfg["group_size"] == (8 if name.endswith("_k8") else 4)
    assert [r["step"] for r in train] == [1, 2]
    _check_logged_against_recomputation(cfg, train)
    assert prov["training_sources_digest"] == T.training_sources_digest()
    assert "mean_speed" in train[0]


@pytest.mark.parametrize("name", list(ARMS))
def test_every_eval_record_carries_the_held_out_keys(runs, name):
    cfg, _, _, evals = _read(runs[name])
    assert [e["step"] for e in evals] == [0, 2]
    for e in evals:
        assert all(k in e for k in HELDOUT_KEYS)
        assert e["eval_heldout_n"] == len(SB.HELDOUT_PROMPTS) * cfg["eval_samples"] == 4
        assert len(e["eval_heldout_items"]) == e["eval_heldout_n"]
        assert e["eval_n"] == 6 * cfg["eval_samples"] and len(e["eval_speeds"]) == e["eval_n"]
        items = e["eval_heldout_items"]
        present = [s for s in items if s is not None]
        assert e["eval_heldout_parse_fail_rate"] == (len(items) - len(present)) / len(items)
        assert e["eval_heldout_violation_rate"] == \
            sum(1 for s in items if not SB.SPEC.verdict(s, cfg["vmax"])) / len(items)


def test_the_fixed_reference_arm_shares_the_light_arm_s_rewards_and_mixed_group_advantages(runs):
    """The two arms differ only in the advantage of homogeneous-verdict groups, so on the same seed they
    log the same first step and, on each group, the advantages differ by exactly minus c (all violating),
    plus c (all passing) or not at all (mixed)."""
    cfg_f, _, train_f, _ = _read(runs["binary_w5_fixedref"])
    cfg_p, _, train_p, _ = _read(runs["binary_w5"])
    gf, gp = train_f[0]["groups"], train_p[0]["groups"]
    assert [g["speeds"] for g in gf] == [g["speeds"] for g in gp], "step 1 samples from the same weights"
    for a, b in zip(gf, gp):
        assert a["rewards"] == b["rewards"]
        shift = np.array(a["adv"]) - np.array(b["adv"])
        v = a["verdicts"]
        want = -cfg_f["fixedref_c"] if not any(v) else (cfg_f["fixedref_c"] if all(v) else 0.0)
        assert np.allclose(shift, want, atol=1e-12)


# ---------------------------------------------------------------- the replay script


def test_the_replay_script_reproduces_the_tiny_runs(runs, tmp_path):
    out = tmp_path / "replay.json"
    dirs = [runs[n] for n in ARMS]
    assert RB.main(dirs + ["--out", str(out)]) == 0
    summary = json.loads(out.read_text())
    assert summary["n_runs"] == len(ARMS) and len(summary["runs"]) == len(ARMS)
    assert (summary["w_light"], summary["w_heavy"], summary["fixedref_c"], summary["tol"]) == (5.0, 5000.0, 2.0, 1e-9)
    t = summary["total"]
    assert t["groups"] == 6 * 2 * len(ARMS)
    assert t["all_violating"] + t["mixed"] + t["all_passing"] == t["groups"]
    assert t["all_violating_scale_differ"] == 0 and t["all_violating_shift_differ"] == 0
    assert t["max_abs_adv_diff_all_violating"] <= 1e-9 and t["max_abs_shift_diff_all_violating"] <= 1e-9
    assert t["all_violating_identical_reward_among_distinct_speeds"] <= t["all_violating_with_distinct_speeds"]
    assert t["identical_reward_among_distinct_speeds"] <= t["groups_with_distinct_speeds"]
    for r in summary["runs"]:
        assert r["train_records"] == 2 and r["groups"] == 12
        assert r["training_sources_digest"] == T.training_sources_digest()
        assert r["config"]["task"] == "speed_bounded" and r["config"]["skip_zero_adv"] is True
        assert r["all_violating_nonzero_adv_fixedref"] == r["all_violating"], \
            "every all-violating group gets a nonzero counterfactual signal of -c"
        assert sum(r[k] for k in ("all_violating", "mixed", "all_passing")) == 12


def test_the_replay_script_refuses_a_log_it_cannot_reproduce(runs, tmp_path):
    src = runs["binary_w5"]
    bad = tmp_path / "tampered"
    shutil.copytree(src, bad)
    lines = (bad / "metrics.jsonl").read_text().splitlines()
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec["phase"] == "train":
            rec["groups"][0]["rewards"][0] += 1e-6
            lines[i] = json.dumps(rec)
            break
    (bad / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="recomputed rewards"):
        RB.replay([str(bad)])
    # the output file is not written when a run does not reproduce
    out = tmp_path / "never.json"
    with pytest.raises(ValueError):
        RB.main([str(bad), "--out", str(out)])
    assert not out.exists()


def _write_fake_run(root, *, arm, w, adv_variant, fixedref_c, group_speeds, vmax=5.0):
    """A run directory whose logged rewards and advantages come from the reward code itself, so the
    replay has something with every kind of group to count."""
    root.mkdir()
    cfg = {"task": "speed_bounded", "arm": arm, "w": w, "w_task": 1.0, "vmax": vmax, "seed": 0,
           "adv_variant": adv_variant, "group_filter": "none", "kl_beta": 0.0, "fixedref_c": fixedref_c,
           "skip_zero_adv": True, "group_size": 4, "steps": 1, "regime": "A"}
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "provenance.json").write_text(json.dumps({"training_sources_digest": "x" * 64}))
    groups = []
    for pi, speeds in enumerate(group_speeds):
        r = RW.group_rewards(speeds, arm, w, vmax, spec=SB.SPEC)
        v = RW.group_verdicts(speeds, vmax, spec=SB.SPEC)
        a = RW.advantage_fixedref(r, v, fixedref_c) if adv_variant == "grpo_fixedref" else RW.advantage(r)
        groups.append({"speeds": speeds, "rewards": r.tolist(), "adv": a.tolist(), "verdicts": v,
                       "all_violating": not any(v), "has_violator": not all(v), "prompt_index": pi,
                       "anchor": None, "kept": True})
    rec = {"step": 1, "phase": "train", "groups": groups}
    (root / "metrics.jsonl").write_text(json.dumps({"step": 0, "phase": "eval"}) + "\n" + json.dumps(rec) + "\n")
    return root


FAKE_GROUPS = [
    [300.0, 1000.0, 50_000.0, 999_999.0],   # all violating, distinct, spread
    [7.0, 7.0, 7.0, 7.0],                   # all violating, identical speeds, no spread
    [None, None, None, None],               # all violating, unparseable, no spread
    [6.0, None, 6.0, None],                 # all violating, one parsed value, spread from the floor
    [-9.0, 9.0, 9.0, 9.0],                  # all violating, two distinct parsed values, identical rewards
    [2.0, 8.0, None, 3.0],                  # mixed
    [1.0, 4.0, 5.0, 0.0],                   # all passing, spread
    [5.0, 5.0, 5.0, 5.0],                   # all passing, no spread
]


@pytest.mark.parametrize("arm,w,variant,c", [("binary", 5.0, "grpo", 0.0), ("binary", 5000.0, "grpo", 0.0),
                                             ("none", 0.0, "grpo", 0.0), ("binary", 5.0, "grpo_fixedref", 2.0)])
def test_the_replay_counts_every_kind_of_group_on_a_constructed_log(tmp_path, arm, w, variant, c):
    run = _write_fake_run(tmp_path / "fake", arm=arm, w=w, adv_variant=variant, fixedref_c=c,
                          group_speeds=FAKE_GROUPS)
    s = RB.replay_run(run, w_light=5.0, w_heavy=5000.0, fixedref_c=2.0, tol=1e-9)
    assert s["groups"] == 8 and s["train_records"] == 1
    assert s["all_violating"] == 5 and s["mixed"] == 1 and s["all_passing"] == 2
    # spread under the row's own design: group 0 from distinct task values and group 3 from the
    # unparseable floor sitting below a parsed violator; groups 1, 2 and 4 tie under every arm
    assert s["all_violating_with_reward_spread"] == 2
    assert s["groups_with_distinct_speeds"] == 4          # groups 0, 4, 5, 6
    assert s["all_violating_with_distinct_speeds"] == 2    # groups 0 and 4
    assert s["all_violating_identical_reward_among_distinct_speeds"] == 1   # group 4, -9 and 9 tie
    assert s["identical_reward_among_distinct_speeds"] == 1
    assert s["all_violating_scale_differ"] == 0 and s["all_violating_shift_differ"] == 0
    assert s["max_abs_adv_diff_all_violating"] <= 1e-9 and s["max_abs_shift_diff_all_violating"] <= 1e-9
    # own-design zero advantages: the three tied all-violating groups under the plain estimator, and
    # none under the fixed reference
    assert s["all_violating_zero_adv_own"] == (0 if variant == "grpo_fixedref" else 3)
    assert s["all_violating_nonzero_adv_fixedref"] == 5
    assert s["config"]["arm"] == arm and s["config"]["w"] == w


def test_the_scale_control_holds_exactly_where_the_identity_says_it_does():
    """In an all-violating group every member carries the same verdict penalty, so the two weights
    give the same centered rewards up to rounding. In a mixed group they do not."""
    all_viol = [300.0, 1000.0, 50_000.0, 999_999.0]
    r5 = RW.group_rewards(all_viol, "binary", 5.0, 5.0, spec=SB.SPEC)
    r5000 = RW.group_rewards(all_viol, "binary", 5000.0, 5.0, spec=SB.SPEC)
    assert np.allclose(r5000, r5 - 4995.0, atol=1e-9)
    assert np.abs(RW.advantage(r5) - RW.advantage(r5000)).max() <= 1e-9
    mixed = [2.0, 8.0, None, 3.0]
    m5 = RW.advantage(RW.group_rewards(mixed, "binary", 5.0, 5.0, spec=SB.SPEC))
    m5000 = RW.advantage(RW.group_rewards(mixed, "binary", 5000.0, 5.0, spec=SB.SPEC))
    assert np.abs(m5 - m5000).max() > 1e-3


def test_a_group_near_the_parse_ceiling_passes_the_reward_shift_check_where_the_advantage_tolerance_is_tight(tmp_path):
    """The identity is exact on the rewards. The z-score divides the rounding of the heavy shift by the
    group's spread, and a group of speeds near the parse ceiling has a spread of about 1e-6, so its two
    advantage vectors can differ by more than 1e-9 for rounding alone (5.2e-9 on this group, measured
    2026-11-06). The reward-shift count stays at zero on the same group, which is how the summary tells
    rounding from a wrong identity."""
    near = [900_000.0, 901_000.0, 950_000.0, 999_999.0]
    run = _write_fake_run(tmp_path / "near", arm="binary", w=5.0, adv_variant="grpo", fixedref_c=0.0,
                          group_speeds=[near])
    s = RB.replay_run(run, w_light=5.0, w_heavy=5000.0, fixedref_c=2.0, tol=1e-9)
    assert s["all_violating"] == 1 and s["all_violating_with_reward_spread"] == 1
    assert s["all_violating_shift_differ"] == 0 and s["max_abs_shift_diff_all_violating"] <= 1e-9
    assert s["max_abs_adv_diff_all_violating"] < 1e-6
    # at a tolerance that absorbs the rounding the advantage count is zero too
    loose = RB.replay_run(run, w_light=5.0, w_heavy=5000.0, fixedref_c=2.0, tol=1e-6)
    assert loose["all_violating_scale_differ"] == 0


def test_the_replay_exits_three_when_a_group_fails_the_scale_check(tmp_path, capsys):
    """A tolerance of zero makes rounding count as a difference on the spread group, which exercises
    the failure path without a wrong identity."""
    run = _write_fake_run(tmp_path / "fake", arm="binary", w=5.0, adv_variant="grpo", fixedref_c=0.0,
                          group_speeds=[[300.0, 1000.0, 50_000.0, 999_999.0]])
    out = tmp_path / "s.json"
    rc = RB.main([str(run), "--out", str(out), "--tol", "0"])
    summary = json.loads(out.read_text())
    diff = summary["total"]["max_abs_adv_diff_all_violating"]
    if diff > 0.0:
        assert rc == 3 and summary["total"]["all_violating_scale_differ"] == 1
        assert "SCALE CHECK FAILED" in capsys.readouterr().err
    else:
        assert rc == 0 and summary["total"]["all_violating_scale_differ"] == 0


# ---------------------------------------------------------------- the manifest


def _rows():
    return MB.read_rows(MB.HERE / MB.OUT_REL)


def test_the_committed_manifest_is_the_generator_output():
    assert os.path.exists(MANIFEST)
    raw = open(MANIFEST, "rb").read()
    assert raw == MB.text().encode("ascii")
    assert b"\r" not in raw and raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert MB.main(["--check"]) == 0


def test_the_manifest_has_39_rows_with_the_stated_cells():
    rows = _rows()
    assert len(rows) == 39
    assert tuple(rows[0]) == MB.V4_HEADER == MB.V2_HEADER + ("fixedref_c", "skip_zero_adv", "group_size")
    assert [int(r["task_id"]) for r in rows] == list(range(1, 40))
    assert all(r["task"] == "speed_bounded" for r in rows)
    assert all(r["skip_zero_adv"] == "1" and r["kl_beta"] == "0.0" and r["w_task"] == "1.0"
               and r["group_filter"] == "none" for r in rows)
    assert all(v == v.strip() and v != "" for r in rows for v in r.values())
    counts = Counter((r["regime"], r["arm_name"]) for r in rows)
    assert counts == {
        ("A", "none"): 5, ("A", "binary_w5"): 5, ("A", "binary_w5000"): 5, ("A", "binary_w5_fixedref"): 5,
        ("A", "binary_w5_k8"): 5, ("A", "binary_w5_fixedref_k8"): 5,
        ("B", "none"): 3, ("B", "binary_w5"): 3, ("B", "binary_w5_fixedref"): 3,
    }
    assert {r["vmax"] for r in rows if r["regime"] == "A"} == {"5.0"}
    assert {r["vmax"] for r in rows if r["regime"] == "B"} == {"30.0"}
    assert MB.regime_limits() == {"A": 5.0, "B": 30.0}
    seeds = {}
    for r in rows:
        seeds.setdefault((r["regime"], r["arm_name"]), []).append(int(r["seed"]))
    for (regime, _), s in seeds.items():
        assert s == ([0, 1, 2, 3, 4] if regime == "A" else [0, 1, 2])
    for r in rows:
        name = r["arm_name"]
        assert r["group_size"] == ("8" if name.endswith("_k8") else "4")
        if "fixedref" in name:
            assert (r["adv_variant"], r["fixedref_c"], r["arm"], r["w"]) == ("grpo_fixedref", "2.0", "binary", "5.0")
        else:
            assert (r["adv_variant"], r["fixedref_c"]) == ("grpo", "0.0")
        if name == "none":
            assert (r["arm"], r["w"]) == ("none", "0.0")
        elif name.startswith("binary_w5000"):
            assert (r["arm"], r["w"]) == ("binary", "5000.0")
        else:
            assert (r["arm"], r["w"]) == ("binary", "5.0")
    # the paired arms differ only in the variant and the shift
    by = {(r["regime"], r["arm_name"], r["seed"]): r for r in rows}
    for regime, plain, fixed in (("A", "binary_w5", "binary_w5_fixedref"), ("A", "binary_w5_k8", "binary_w5_fixedref_k8"),
                                 ("B", "binary_w5", "binary_w5_fixedref")):
        for seed in ("0", "1", "2"):
            p, f = by[(regime, plain, seed)], by[(regime, fixed, seed)]
            skip = {"task_id", "arm_name", "adv_variant", "fixedref_c"}
            assert {k: v for k, v in p.items() if k not in skip} == {k: v for k, v in f.items() if k not in skip}


def _argv(r):
    """The flags the v4 branch of the submission script passes for a row."""
    argv = ["--task", r["task"], "--arm", r["arm"], "--w", r["w"], "--w-task", r["w_task"], "--vmax", r["vmax"],
            "--regime", r["regime"], "--seed", r["seed"], "--adv-variant", r["adv_variant"],
            "--group-filter", r["group_filter"], "--kl-beta", r["kl_beta"], "--fixedref-c", r["fixedref_c"],
            "--group-size", r["group_size"], "--out-dir", "unused"]
    if r["skip_zero_adv"] == "1":
        argv.append("--skip-zero-adv")
    return argv


def test_every_row_is_a_flag_combination_the_trainer_accepts():
    for r in _rows():
        ap = T.build_arg_parser()
        args = ap.parse_args(_argv(r))
        assert T.check_args(ap, args) is None
        assert args.skip_zero_adv is True and args.group_size in (4, 8)
        assert args.fixedref_c == float(r["fixedref_c"]) and args.adv_variant == r["adv_variant"]
        assert args.task == "speed_bounded" and tasks.get(args.task) is SB.SPEC


def test_every_variant_in_the_manifest_is_one_the_trainer_knows_and_one_is_new():
    rows = _rows()
    variants = {r["adv_variant"] for r in rows}
    assert variants == {"grpo", "grpo_fixedref"}
    assert variants <= set(RW.TRAINER_ADV_VARIANTS)
    assert "grpo_fixedref" not in RW.ADV_VARIANTS, "the analysis enumerates ADV_VARIANTS without verdicts"


@pytest.mark.parametrize("field,value,match", [
    ("arm", "gate", "not one of"),
    ("adv_variant", "ngrpo_mean", "not one of"),
    ("fixedref_c", "2.0", "under variant"),
    ("group_size", "6", "group size"),
    ("skip_zero_adv", "yes", "not 0 or 1"),
    ("kl_beta", "0.04", "KL weight"),
    ("vmax", "6.0", "configs/regimes.yaml gives"),
    ("regime", "C", "lacks"),
    ("task", "speed", "runs task"),
])
def test_the_generator_refuses_a_row_outside_the_design(field, value, match):
    limits = MB.regime_limits()
    row = dict(MB.rows()[5])   # a plain binary_w5 row at regime A
    row[field] = value
    with pytest.raises(ValueError, match=match):
        MB.validate_row(row, limits)
    fixed = dict(MB.rows()[15])   # a binary_w5_fixedref row
    fixed["fixedref_c"] = "0.0"
    with pytest.raises(ValueError, match="with a shift of"):
        MB.validate_row(fixed, limits)


def test_the_generator_is_deterministic():
    assert MB.text() == MB.text()
    assert MB.text().count("\n") == 40
