"""analyze.py admits a Block A' run (ICML 2027, November 2026): the bounded speed task under the
fixed-reference advantage variant with the zero-advantage skip on.

Four things are checked:
  * the two trainer flags of the ICML blocks, --fixedref-c and --skip-zero-adv, have a legacy default
    equal to the parser default, so a run written before they existed reads as what it was;
  * arm_name gives grpo_fixedref its own suffix and compose_arm_label its own words, every variant the
    trainer accepts has both, and the frozen algebra table still enumerates rewards.ADV_VARIANTS only;
  * the trainer self-consistency check recomputes a fixed-reference run's advantages from the logged
    verdicts, counts a logged verdict that disagrees with the recomputation, still catches a logged
    advantage that disagrees, and refuses a fixed-reference run that logged no verdicts;
  * a two-step run of the arm on the random tiny model is admitted end to end by analyze.py in a
    subprocess: the analysis directory is written, the counterfactual checks pass and the arm resolves.
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402
import rewards as RW  # noqa: E402
import speed_bounded_task as SB  # noqa: E402
import tasks  # noqa: E402
import train_speed_grpo as T  # noqa: E402

PYTHON = sys.executable


def _parser_defaults():
    return vars(T.build_arg_parser().parse_args(["--arm", "binary", "--w", "5", "--vmax", "5", "--out-dir", "x"]))


# A fixed-reference run with every kind of group, written from the reward code itself so the logged
# rewards, verdicts and advantages are what the trainer would have logged.
FAKE_GROUPS = [
    [300.0, 1000.0, 50_000.0, 999_999.0],   # all violating, distinct speeds, reward spread
    [7.0, 7.0, 7.0, 7.0],                   # all violating, identical speeds, no spread
    [None, None, None, None],               # all violating, unparseable, no spread
    [2.0, 8.0, None, 3.0],                  # mixed
    [1.0, 4.0, 5.0, 0.0],                   # all passing, spread
    [5.0, 5.0, 5.0, 5.0],                   # all passing, no spread
]


def _write_fixedref_run(root, *, c=2.0, steps=2, with_verdicts=True, vmax=5.0):
    root.mkdir(parents=True)
    cfg = {"task": "speed_bounded", "arm": "binary", "w": 5.0, "w_task": 1.0, "vmax": vmax, "regime": "A",
           "seed": 0, "steps": steps, "group_size": 4, "batch_prompts": len(FAKE_GROUPS), "lr": 1e-5,
           "max_new_tokens": 24, "backward_per_group": False, "adv_variant": "grpo_fixedref",
           "group_filter": "none", "kl_beta": 0.0, "keep_schedule": None, "fixedref_c": c,
           "skip_zero_adv": True, "out_dir": str(root)}
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "provenance.json").write_text(json.dumps({"steps_completed": steps, "git_sha": "synthetic",
                                                       "final_param_checksum": "c0"}))
    lines = [json.dumps({"step": 0, "phase": "eval", "eval_n": 4, "eval_violation_rate": 1.0,
                         "eval_parse_fail_rate": 0.0, "eval_mean_speed": 9.0, "eval_mean_residual": 4.0,
                         "eval_speeds": [9.0, 9.0, 9.0, 9.0]})]
    for step in range(1, steps + 1):
        groups = []
        for pi, speeds in enumerate(FAKE_GROUPS):
            r = RW.group_rewards(speeds, "binary", 5.0, vmax, spec=SB.SPEC)
            v = RW.group_verdicts(speeds, vmax, spec=SB.SPEC)
            a = RW.advantage_fixedref(r, v, c)
            g = {"speeds": speeds, "rewards": r.tolist(), "adv": a.tolist(), "all_violating": not any(v),
                 "has_violator": not all(v), "kept": True, "prompt_index": pi, "anchor": None,
                 "n_tokens": [3, 3, 3, 3], "zero_adv": RW.group_adv_is_zero(a),
                 "kept_for_update": not RW.group_adv_is_zero(a)}
            if with_verdicts:
                g["verdicts"] = v
            groups.append(g)
        lines.append(json.dumps({"step": step, "phase": "train", "all_violating_fraction": 0.5,
                                 "kl_seq_mean": 0.0, "opt_step_taken": True, "groups": groups}))
    (root / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    (root / "DONE").write_text("ok\n")
    return root


# ---------------------------------------------------------------- legacy defaults


def test_the_icml_flags_have_legacy_defaults_equal_to_the_parser_defaults(tmp_path):
    d = _parser_defaults()
    assert A.LEGACY_CONFIG_DEFAULTS["fixedref_c"] == d["fixedref_c"] == 2.0
    assert A.LEGACY_CONFIG_DEFAULTS["skip_zero_adv"] is d["skip_zero_adv"] is False
    run_dir = _write_fixedref_run(tmp_path / "r")
    cfg = json.loads((run_dir / "config.json").read_text())
    del cfg["fixedref_c"], cfg["skip_zero_adv"]
    cfg["adv_variant"] = "grpo"
    (run_dir / "config.json").write_text(json.dumps(cfg))
    loaded = A._load_run(run_dir)["cfg"]
    assert loaded["fixedref_c"] == 2.0 and loaded["skip_zero_adv"] is False
    # a value that is present is never overwritten
    loaded = A._load_run(_write_fixedref_run(tmp_path / "s", c=0.5))["cfg"]
    assert loaded["fixedref_c"] == 0.5 and loaded["skip_zero_adv"] is True


# ---------------------------------------------------------------- names and labels


def test_the_fixed_reference_variant_has_its_own_suffix_and_label():
    cfg = {"arm": "binary", "w": 5.0, "task": "speed_bounded", "vmax": 5.0, "adv_variant": "grpo_fixedref",
           "fixedref_c": 2.0, "skip_zero_adv": True}
    assert A.arm_name(cfg) == "binary_fixedref"
    assert A.compose_arm_label(cfg) == A.ARM_STYLE["binary"]["label"] + ", fixed reference"
    assert A.arm_name(dict(cfg, w=A.BINARY_ABOVE_W)) == "binary_5x_fixedref"
    assert A.arm_name(dict(cfg, group_filter="verdict_mixed")) == "binary_fixedref_vfilter"
    # the plain estimator at the same flags keeps the name it always had
    assert A.arm_name(dict(cfg, adv_variant="grpo")) == "binary"
    # every variant the trainer accepts can be named and labelled by the analysis
    assert set(A.VARIANT_SUFFIX) == set(RW.TRAINER_ADV_VARIANTS)
    assert set(A.VARIANT_LABEL) == set(RW.TRAINER_ADV_VARIANTS) - {"grpo"}
    assert len(set(A.VARIANT_SUFFIX.values())) == len(A.VARIANT_SUFFIX), "two variants share a suffix"
    # the variant stays out of the frozen enumeration the algebra table reads
    assert "grpo_fixedref" not in RW.ADV_VARIANTS


def test_a_regime_title_names_the_envelope_for_the_bounded_speed_task():
    runs = [{"cfg": {"task": "speed_bounded", "regime": "A", "vmax": 5.0}}]
    assert A.regime_title("A", runs) == "Hard limit, $v_{\\max} = 5$"
    assert A.regime_title("A", [{"cfg": {"task": "speed", "regime": "A", "vmax": 5.0}}]) == "Hard limit, $v_{\\max} = 5$"
    assert A.regime_title("H", [{"cfg": {"task": "capacity", "regime": "H", "vmax": 0.5}}]) == "Hard level, capacity fraction = 0.5"


# ---------------------------------------------------------------- the self-consistency check


def test_counterfactual_checks_recompute_a_fixed_reference_run_from_its_logged_verdicts(tmp_path):
    run = A._load_run(_write_fixedref_run(tmp_path / "ok"))
    stats, worst = A.counterfactual_checks([run])
    assert worst["self"] == 0.0
    assert stats["groups"] == 2 * len(FAKE_GROUPS)
    assert not {"verdict_mismatch", "anchor_mismatch", "kept_mismatch", "prompt_index_mismatch"} & set(stats)
    # a logged verdict that disagrees with the recomputation is counted, even when flipping it leaves
    # the group mixed and so leaves the logged advantage reproducible
    g = run["train"][0]["groups"][3]
    assert not all(g["verdicts"]) and any(g["verdicts"])
    g["verdicts"][0] = not g["verdicts"][0]
    stats, worst = A.counterfactual_checks([run])
    assert stats["verdict_mismatch"] == 1
    # a logged advantage that disagrees is still caught
    run = A._load_run(tmp_path / "ok")
    run["train"][1]["groups"][0]["adv"][0] += 1e-3
    stats, worst = A.counterfactual_checks([run])
    assert worst["self"] == pytest.approx(1e-3)
    # the all-violating groups carry the shift, so none has a zero advantage vector
    run = A._load_run(tmp_path / "ok")
    for rec in run["train"]:
        for g in rec["groups"]:
            if g["all_violating"]:
                assert not RW.group_adv_is_zero(g["adv"])
                plain = RW.advantage(g["rewards"])
                assert np.allclose(np.array(g["adv"]) - plain, -2.0)


def test_a_fixed_reference_run_that_logged_no_verdicts_is_refused(tmp_path):
    run = A._load_run(_write_fixedref_run(tmp_path / "nov", with_verdicts=False))
    with pytest.raises(ValueError, match="verdicts"):
        A.counterfactual_checks([run])


def test_the_frozen_algebra_table_still_enumerates_adv_variants_only(tmp_path):
    run = A._load_run(_write_fixedref_run(tmp_path / "alg"))
    algebra = A.variant_algebra([run])
    assert set(algebra["variants"]) == set(RW.ADV_VARIANTS)
    assert algebra["all_violating_groups"] == 2 * 3


# ---------------------------------------------------------------- end to end on the tiny model

torch = pytest.importorskip("torch")

BASE = ["--task", "speed_bounded", "--vmax", "5", "--tiny", "--device", "cpu", "--eval-every", "2",
        "--eval-samples", "2", "--batch-prompts", "6", "--seed", "3", "--max-new-tokens", "24",
        "--skip-zero-adv", "--steps", "2", "--arm", "binary", "--w", "5",
        "--adv-variant", "grpo_fixedref", "--fixedref-c", "2.0"]


def test_a_tiny_fixed_reference_run_is_admitted_by_the_analysis_end_to_end(tmp_path):
    root = tmp_path / "runs"
    run_dir = root / "16_A_binary_w5_fixedref_s3"
    assert T.main(BASE + ["--out-dir", str(run_dir)]) == 0
    assert (run_dir / "DONE").exists()
    cfg = A._load_run(run_dir)["cfg"]
    assert cfg["adv_variant"] == "grpo_fixedref" and cfg["fixedref_c"] == 2.0 and cfg["skip_zero_adv"] is True
    assert A.arm_name(cfg) == "binary_fixedref"
    out = tmp_path / "analysis"
    r = subprocess.run([PYTHON, os.path.join(HERE, "analyze.py"), "--runs", str(root), "--out", str(out)],
                       cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    checks = json.loads((out / "checks.json").read_text())
    assert checks["n_runs"] == 1 and checks["matched_budget_problems"] == []
    assert checks["trainer_self_consistent"] is True
    assert checks["worst_abs_diff"]["self"] == 0.0
    cf = checks["counterfactual"]
    assert cf["groups"] == 12
    assert not {"verdict_mismatch", "anchor_mismatch", "kept_mismatch", "prompt_index_mismatch"} & set(cf)
    assert set(checks["variant_algebra"]["variants"]) == set(RW.ADV_VARIANTS)
    labels = json.loads((out / "arm_labels.json").read_text())
    assert labels["binary_fixedref"] == A.ARM_STYLE["binary"]["label"] + ", fixed reference"
    with (out / "final_per_run.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["arm"] for row in rows] == ["binary_fixedref"]
    # a direct run carries no regime; the sbatch sets it from the manifest row
    assert rows[0]["regime"] == "" and rows[0]["steps"] == "2"
    for name in ("final_by_arm.csv", "group_counters.csv", "curves_train.csv", "curves_eval.csv"):
        assert (out / name).exists(), name
    assert tasks.get(cfg["task"]) is SB.SPEC


def test_a_centered_run_and_a_fixed_reference_run_share_one_cohort_but_a_k8_run_does_not(tmp_path):
    """The shift that names the fixed-reference arm may differ inside a cohort, as the advantage variant
    may. The group size may not, because a K = 8 run samples twice the budget and is its own cohort."""
    common = ["--task", "speed_bounded", "--vmax", "5", "--tiny", "--device", "cpu", "--eval-every", "0",
              "--batch-prompts", "2", "--seed", "0", "--steps", "2", "--arm", "binary", "--w", "5",
              "--skip-zero-adv"]
    root = tmp_path / "runs"
    plain = root / "1_A_binary_w5_s0"
    fixed = root / "2_A_binary_w5_fixedref_s0"
    k8 = root / "3_A_binary_w5_k8_s0"
    assert T.main(common + ["--out-dir", str(plain)]) == 0
    assert T.main(common + ["--adv-variant", "grpo_fixedref", "--out-dir", str(fixed)]) == 0
    assert T.main(common + ["--group-size", "8", "--out-dir", str(k8)]) == 0
    runs = [A._load_run(plain), A._load_run(fixed)]
    assert runs[0]["cfg"]["fixedref_c"] == runs[1]["cfg"]["fixedref_c"] == 2.0
    assert runs[0]["cfg"]["adv_variant"] != runs[1]["cfg"]["adv_variant"]
    problems = A.matched_budget_check(runs)
    assert [p for p in problems if "config." in p] == []
    assert "fixedref_c" in A.BOUNDED_CONFIG_DIFF and "group_size" not in A.BOUNDED_CONFIG_DIFF
    assert len(A.ALLOWED_CONFIG_DIFF) == 10 and not (A.BOUNDED_CONFIG_DIFF & A.ALLOWED_CONFIG_DIFF)
    mixed = A.matched_budget_check([A._load_run(plain), A._load_run(k8)])
    assert any("config.group_size" in p for p in mixed), mixed
