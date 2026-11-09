"""The revision audit table generator on synthetic capacity-hard runs.

The tests cover the arm order with the three KL-penalty arms, the refusal of a run set that lacks an arm,
the byte-identical rebuild from the archived records, the per-seed range beside each mean, the
ragged-right label column, the macro word of every arm, and the useful-seed macro guard.

The generator lives in the paper directory of the repository and beside analyze.py in the code
supplement, so the loader looks in both places.
"""
import importlib.util
import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402
import capacity_task as C  # noqa: E402
import rewards as RW  # noqa: E402
import tasks  # noqa: E402

PAPER = os.path.join(HERE, "paper")
GENERATOR = next((p for p in (os.path.join(PAPER, "make_revision_audit_tables.py"),
                              os.path.join(HERE, "make_revision_audit_tables.py"))
                  if os.path.exists(p)), None)
pytestmark = pytest.mark.skipif(
    GENERATOR is None, reason="make_revision_audit_tables.py is not in this copy of the code")

VMAX = 0.03
SPEC = tasks.get("capacity")
# Each entry overrides the base capacity config so that analyze.arm_name gives the ORDER name.
ARM_CFGS = {
    "graded": {"arm": "graded"},
    "binary": {"arm": "binary"},
    "binary_1000x": {"arm": "binary", "w": 5000.0},
    "none": {"arm": "none", "w": 0.0},
    "verdict_only": {"arm": "binary", "w_task": 0.0},
    "graded_no_task": {"arm": "graded", "w_task": 0.0},
    "binary_drgrpo": {"arm": "binary", "adv_variant": "drgrpo"},
    "binary_ngrpo": {"arm": "binary", "adv_variant": "ngrpo_feasible"},
    "binary_vfilter": {"arm": "binary", "group_filter": "verdict_mixed"},
    "binary_sfilter": {"arm": "binary", "group_filter": "reward_spread"},
    "graded_kl": {"arm": "graded", "kl_beta": 0.04},
    "binary_kl": {"arm": "binary", "kl_beta": 0.04},
    "none_kl": {"arm": "none", "w": 0.0, "kl_beta": 0.04},
}
# Instance 0 at this ratio has capacity 6. Item 0 weighs 3 and is the optimum. Item 9 weighs 9.
FEASIBLE, EMPTY, INFEASIBLE = (0, 0), (0,), (0, 9)
# Half the completions violate and none carries nonempty feasible value.
DEFAULT_EVAL = [EMPTY, EMPTY, INFEASIBLE, None]


def _gen():
    spec = importlib.util.spec_from_file_location("make_revision_audit_tables", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _as_json(items):
    return [list(x) if isinstance(x, tuple) else x for x in items]


def _capacity_run(tmp_path, arm, seed, eval_items=None):
    """This writes one finished capacity-hard run whose config names the given ORDER arm."""
    d = tmp_path / f"{arm}_s{seed}"
    d.mkdir(parents=True)
    cfg = {"task": "capacity", "arm": "binary", "w": 5.0, "w_task": 1.0, "vmax": VMAX,
           "regime": "H", "seed": seed, "steps": 1, "group_size": 4, "batch_prompts": 1,
           "lr": 1e-5, "max_new_tokens": 128, "backward_per_group": True, "adv_variant": "grpo",
           "group_filter": "none", "kl_beta": 0.0, "out_dir": str(d)}
    cfg.update(ARM_CFGS[arm])
    assert A.arm_name(cfg) == arm, (arm, A.arm_name(cfg))
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "provenance.json").write_text(json.dumps({"steps_completed": 1, "git_sha": "synthetic"}))
    items = [FEASIBLE, EMPTY, INFEASIBLE, None]
    rewards = RW.group_rewards(items, cfg["arm"], cfg["w"], VMAX, spec=SPEC, w_task=cfg["w_task"])
    group = {"selections": _as_json(items), "rewards": rewards.tolist(),
             "has_violator": RW.group_has_violator(items, VMAX, spec=SPEC),
             "all_violating": RW.group_is_all_violating(items, VMAX, spec=SPEC)}
    eval_items = DEFAULT_EVAL if eval_items is None else eval_items
    lines = [json.dumps({"step": 1, "phase": "train", "groups": [group]}),
             json.dumps({"step": 1, "phase": "eval", "eval_n": len(eval_items),
                         "eval_violation_rate": float(np.mean([not C.verdict(x, VMAX)
                                                               for x in eval_items])),
                         "eval_selections": _as_json(eval_items)})]
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    (d / "DONE").write_text("ok\n")
    return d


def _run_set(tmp_path, arms, seeds=(0, 1), eval_items=None):
    root = tmp_path / "runs"
    for arm in arms:
        for seed in seeds:
            items = None if eval_items is None else eval_items(arm, seed)
            _capacity_run(root, arm, seed, items)
    return root


def _build(gen, root, tmp_path, name="first"):
    out, audit = tmp_path / f"{name}_tables", tmp_path / f"{name}_audit"
    assert gen.main(["--runs", str(root), "--out", str(out), "--audit-out", str(audit)]) == 0
    return out, audit


def _macros(path):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", path.read_text()))


def _body_rows(path):
    text = path.read_text()
    body = text.split(r"\midrule")[1].split(r"\bottomrule")[0]
    return [line for line in body.strip().splitlines()]


def test_the_fixture_covers_every_arm_the_generator_orders():
    gen = _gen()
    assert set(ARM_CFGS) == set(gen.ORDER)
    assert set(gen.LABELS) >= set(gen.ORDER)


def test_every_arm_appears_in_order_with_the_kl_arms_last(tmp_path):
    gen = _gen()
    assert gen.ORDER[-3:] == ["graded_kl", "binary_kl", "none_kl"]
    out, _ = _build(gen, _run_set(tmp_path, gen.ORDER), tmp_path)
    rows = _body_rows(out / "revision_quality.tex")
    assert [row.split(" & ")[0] for row in rows] == [gen.LABELS[arm] for arm in gen.ORDER]
    labels = "\n".join(rows)
    for label in ("residual $+$ task, KL penalty", "verdict $+$ task, KL penalty",
                  "task only, KL penalty"):
        assert label in labels
    macros = _macros(out / "revision_macros.tex")
    assert macros["revAuditArms"] == str(len(gen.ORDER))
    assert macros["revAuditRuns"] == str(2 * len(gen.ORDER))


def test_a_run_set_missing_a_kl_arm_is_refused(tmp_path):
    gen = _gen()
    root = _run_set(tmp_path, [arm for arm in gen.ORDER if arm != "none_kl"])
    out, audit = tmp_path / "tables", tmp_path / "audit"
    with pytest.raises(ValueError, match="every arm in ORDER"):
        gen.main(["--runs", str(root), "--out", str(out), "--audit-out", str(audit)])


def test_the_records_rebuild_the_same_table_byte_for_byte(tmp_path):
    gen = _gen()
    out, audit = _build(gen, _run_set(tmp_path, gen.ORDER), tmp_path)
    again, audit_again = tmp_path / "again_tables", tmp_path / "again_audit"
    assert gen.main(["--records", str(audit), "--out", str(again),
                     "--audit-out", str(audit_again)]) == 0
    for name in ("revision_quality.tex", "revision_macros.tex"):
        assert (out / name).read_bytes() == (again / name).read_bytes()


def _spread_eval(arm, seed):
    if arm != "binary_sfilter":
        return None
    return [EMPTY] * 4 if seed == 0 else [FEASIBLE, FEASIBLE, EMPTY, EMPTY]


def test_value_cells_carry_the_per_seed_range(tmp_path):
    gen = _gen()
    out, audit = _build(gen, _run_set(tmp_path, gen.ORDER, eval_items=_spread_eval), tmp_path)
    rows = {row.split(" & ")[0]: row for row in _body_rows(out / "revision_quality.tex")}
    spread = rows[gen.LABELS["binary_sfilter"]]
    assert spread.endswith(r"& 2/2 [0.34, 1.00] & 0.2500 [0.00, 0.50] & 0.2500 [0.00, 0.50] \\")
    plain = rows[gen.LABELS["none_kl"]]
    assert plain.endswith(r"& 0/2 [0.00, 0.66] & 0.0000 [0.00, 0.00] & 0.0000 [0.00, 0.00] \\")
    macros = _macros(out / "revision_macros.tex")
    assert macros["revAuditNonemptyBinarySFilter"] == "0.2500"
    assert macros["revAuditValueBinarySFilter"] == "0.2500"
    assert macros["revAuditFeasSeedsBinarySFilter"] == "2"
    assert macros["revAuditSeedsBinarySFilter"] == "2"
    per_seed = (audit / "capacity_quality_per_seed.csv").read_text().splitlines()[0]
    assert per_seed == ("regime,arm,seed,evaluation_samples,unparseable,parsed_empty,"
                        "parsed_nonempty_feasible,violation_rate,feasible_value_per_sample")


def test_seed_extremes_is_the_min_and_max_over_seeds():
    gen = _gen()
    rows = [dict(arm="a", seed=0, evaluation_samples=4, parsed_nonempty_feasible=0,
                 feasible_value_per_sample=0.0),
            dict(arm="a", seed=1, evaluation_samples=4, parsed_nonempty_feasible=2,
                 feasible_value_per_sample=0.5),
            dict(arm="a", seed=2, evaluation_samples=4, parsed_nonempty_feasible=1,
                 feasible_value_per_sample=0.125),
            dict(arm="b", seed=0, evaluation_samples=4, parsed_nonempty_feasible=4,
                 feasible_value_per_sample=0.75)]
    assert gen.seed_extremes(rows) == {"a": {"nonempty": (0.0, 0.5), "value": (0.0, 0.5)},
                                       "b": {"nonempty": (1.0, 1.0), "value": (0.75, 0.75)}}


def test_the_label_column_is_ragged_right(tmp_path):
    gen = _gen()
    out, _ = _build(gen, _run_set(tmp_path, gen.ORDER), tmp_path)
    text = (out / "revision_quality.tex").read_text()
    assert r"\begin{tabular}{>{\raggedright\arraybackslash}p{4.8cm}ccc}" in text


def test_every_arm_has_a_letters_only_macro_word(tmp_path, monkeypatch):
    gen = _gen()
    words = gen.arm_words()
    assert list(words) == gen.ORDER
    assert all(re.fullmatch(r"[A-Za-z]+", word) for word in words.values())
    assert len(set(words.values())) == len(words)
    assert words["graded_kl"] == "GradedKl" and words["binary_1000x"] == "BinaryHeavy"
    out, _ = _build(gen, _run_set(tmp_path, gen.ORDER), tmp_path)
    macros = _macros(out / "revision_macros.tex")
    for word in words.values():
        for stem in ("revAuditFeasSeeds", "revAuditSeeds", "revAuditNonempty", "revAuditValue"):
            assert f"{stem}{word}" in macros
    monkeypatch.setattr(gen, "WORDS", {arm: word for arm, word in gen.WORDS.items()
                                       if arm != "none_kl"})
    with pytest.raises(ValueError, match="none_kl"):
        gen.arm_words()


def test_the_useful_seed_macros_need_exactly_one_useful_feasible_run(tmp_path):
    gen = _gen()
    out, _ = _build(gen, _run_set(tmp_path, gen.ORDER, eval_items=_spread_eval), tmp_path)
    macros = _macros(out / "revision_macros.tex")
    assert macros["revAuditUsefulFeasibleRuns"] == "1"
    assert (macros["revAuditUsefulSeedNonempty"], macros["revAuditUsefulSeedSamples"],
            macros["revAuditUsefulSeedValue"]) == ("2", "4", "0.5000")

    def two_useful(arm, seed):
        return [FEASIBLE, FEASIBLE, EMPTY, EMPTY] if arm == "binary_sfilter" else None

    out, _ = _build(gen, _run_set(tmp_path / "two", gen.ORDER, eval_items=two_useful),
                    tmp_path / "two")
    macros = _macros(out / "revision_macros.tex")
    assert macros["revAuditUsefulFeasibleRuns"] == "2"
    assert "revAuditUsefulSeedValue" not in macros
