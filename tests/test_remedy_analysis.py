"""The analysis side of the remedy arms.

Covers the legacy config defaults, the arm names, per-variant self-consistency and its failure
modes, the variant algebra, the new counters, remedy_timing.csv, the fingerprint waiver and its
refusals, assemble_runs.py, an end-to-end remedy and KL table through make_remedy_tables.py, and
the frozen snapshot of the 36 capacity rows.

Every synthetic run below computes its rewards, anchors, advantages and kept flags with the same
functions the trainer calls, so the analysis's recomputation has to pass for real.
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402
import assemble_runs  # noqa: E402
import capacity_task as C  # noqa: E402
import rewards as RW  # noqa: E402
import tasks  # noqa: E402
import train_speed_grpo as T  # noqa: E402

PAPER = os.path.join(HERE, "paper")
requires_paper = pytest.mark.skipif(
    not os.path.exists(os.path.join(PAPER, "make_remedy_tables.py")),
    reason="the paper's table generators are not part of the anonymous code supplement")
SNAPSHOT = os.path.join(HERE, "tests", "fixtures", "capacity36_final_per_run.csv")


# ---------------------------------------------------------------- synthetic runs


def _speed_pool(vmax):
    f = float(np.floor(vmax))
    return [None, 0.0, f, f + 1.0, 40.0, 250.0, 900.0]


def write_run(d, *, task="speed", arm="binary", w=5.0, w_task=1.0, vmax=5.0, regime="A", seed=0,
              steps=3, groups=3, adv_variant="grpo", group_filter="none", kl_beta=0.0,
              items_fn=None, final_feasible=False, git_sha="synthetic", digest=None, rng_seed=None):
    tspec = tasks.get(task)
    spec = None if task == "speed" else tspec
    rng = np.random.default_rng(rng_seed if rng_seed is not None else 7919 * seed + 31 * steps)
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"task": task, "arm": arm, "w": w, "w_task": w_task, "vmax": vmax, "regime": regime,
           "seed": seed, "steps": steps, "group_size": 4, "batch_prompts": groups, "lr": 1e-5,
           "max_new_tokens": 6 if task == "speed" else 128, "backward_per_group": task != "speed",
           "adv_variant": adv_variant, "group_filter": group_filter, "kl_beta": kl_beta,
           "out_dir": str(d)}
    prov = {"steps_completed": steps, "git_sha": git_sha, "torch": "1", "transformers": "1",
            "numpy": "1", "device": "cpu", "device_name": "cpu", "python": "3.12", "platform": "p",
            "s_per_step_median": 1.0, "final_param_checksum": f"checksum-{seed}"}
    if digest is not None:
        prov["training_sources_digest"] = digest
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "provenance.json").write_text(json.dumps(prov))

    def draw(gi):
        if items_fn is not None:
            return items_fn(rng, gi)
        if task == "speed":
            pool = _speed_pool(vmax)
            return [pool[int(i)] for i in rng.integers(0, len(pool), size=4)]
        inst = gi % len(C.INSTANCES)
        out = []
        for _ in range(4):
            if rng.random() < 0.1:
                out.append(None)
                continue
            k = int(rng.integers(0, C.N_ITEMS + 1))
            out.append((inst,) + tuple(sorted(rng.choice(C.N_ITEMS, size=k, replace=False).tolist())))
        return out

    def as_json(items):
        return [list(x) if isinstance(x, tuple) else x for x in items]

    def ev(step, items):
        present = [x for x in items if x is not None]
        return {"step": step, "phase": "eval", "eval_n": len(items),
                "eval_violation_rate": float(np.mean([not tspec.verdict(x, vmax) for x in items])),
                "eval_parse_fail_rate": float(np.mean([x is None for x in items])),
                tspec.EVAL_SCALAR_KEY: tspec.summary_stat(items),
                "eval_mean_residual": (float(np.mean([tspec.residual(x, vmax) for x in present]))
                                       if present else None),
                tspec.EVAL_ITEMS_KEY: as_json(items)}

    lines = [json.dumps(ev(0, [x for gi in range(groups) for x in draw(gi)]))]
    for st in range(1, steps + 1):
        gs = []
        for gi in range(groups):
            items = draw(gi)
            r = RW.group_rewards(items, arm, w, vmax, spec=spec, w_task=w_task)
            anchor = RW.reward_anchor(adv_variant, arm, w, vmax, spec=spec, w_task=w_task,
                                      instance=tspec.instance_for(gi))
            a = RW.advantage_variant(r, adv_variant, anchor)
            gs.append({tspec.ITEMS_KEY: as_json(items), "rewards": r.tolist(), "adv": a.tolist(),
                       "all_violating": RW.group_is_all_violating(items, vmax, spec=spec),
                       "has_violator": RW.group_has_violator(items, vmax, spec=spec),
                       "kept": RW.keep_group(group_filter, r, RW.group_verdicts(items, vmax, spec=spec)),
                       "prompt_index": gi, "anchor": anchor, "n_tokens": [3, 3, 3, 3]})
        n_kept = sum(g["kept"] for g in gs)
        nonzero = kl_beta > 0 or any(x != 0.0 for g in gs if g["kept"] for x in g["adv"])
        flat = [tuple(x) if isinstance(x, list) else x for g in gs for x in g[tspec.ITEMS_KEY]]
        present = [x for x in flat if x is not None]
        lines.append(json.dumps({
            "step": st, "phase": "train", "s_per_step": 1.0, "groups": gs,
            "violation_rate": float(np.mean([not tspec.verdict(x, vmax) for x in flat])),
            "parse_fail_rate": float(np.mean([x is None for x in flat])),
            "all_violating_fraction": float(np.mean([g["all_violating"] for g in gs])),
            tspec.SCALAR_KEY: tspec.summary_stat(flat),
            "mean_residual": (float(np.mean([tspec.residual(x, vmax) for x in present]))
                              if present else None),
            "reward_mean": 0.0, "group_reward_std_mean": 0.0, "violator_mean_adv": 0.0,
            "frac_violators_pos_adv": 0.0, "kl_seq_mean": 0.1 * st + seed,
            "pg_loss": 0.0 if n_kept else None, "opt_step_taken": n_kept > 0,
            "grad_norm_preclip": (1.0 if nonzero else 0.0) if n_kept else None,
            "n_kept_groups": n_kept, "kl_tok_mean": 0.01 * st,
            "kl_k3_tok_mean": 0.0 if kl_beta else None, "kl_loss": 0.0 if kl_beta else None}))
        eval_items = flat
        if st == steps and final_feasible:
            eval_items = [0.0] * len(flat) if task == "speed" else [(0,)] * len(flat)
        lines.append(json.dumps(ev(st, eval_items)))
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    (d / "DONE").write_text("ok\n")
    return d


def _analyze(root, out):
    return subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"), "--runs", str(root),
                           "--out", str(out)], cwd=HERE, capture_output=True, text=True)


# ---------------------------------------------------------------- legacy defaults


def _parser_defaults():
    return vars(T.build_arg_parser().parse_args(
        ["--arm", "binary", "--w", "5", "--vmax", "5", "--out-dir", "x"]))


def test_legacy_config_defaults_are_the_parser_defaults():
    d = _parser_defaults()
    for k, v in A.LEGACY_CONFIG_DEFAULTS.items():
        assert k in d, f"{k} is not a trainer flag"
        assert d[k] == v and type(d[k]) is type(v), f"{k}: parser {d[k]!r}, legacy {v!r}"


def test_every_flag_added_since_the_golden_recording_has_a_legacy_default():
    fx = os.path.join(HERE, "tests", "fixtures", "golden_tiny.json")
    if not os.path.exists(fx):
        pytest.skip("the golden fixture is not present in this copy of the code")
    recorded = set(json.load(open(fx))["cases"]["speed"]["config"])
    added = set(_parser_defaults()) - recorded - {"out_dir"}
    assert added and added <= set(A.LEGACY_CONFIG_DEFAULTS), added


@pytest.mark.parametrize("array", ["430542", "572991"])
def test_real_legacy_configs_lack_only_keys_that_have_a_legacy_default(array):
    root = os.path.join(HERE, "runs", array)
    if not os.path.isdir(root):
        pytest.skip("the published runs are not in this checkout")
    parser = set(_parser_defaults())
    seen = 0
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "config.json")
        if os.path.exists(p):
            seen += 1
            missing = parser - set(json.load(open(p)))
            assert missing <= set(A.LEGACY_CONFIG_DEFAULTS), (name, missing)
    assert seen >= 24


def test_legacy_defaults_fill_only_absent_keys(tmp_path):
    d = write_run(tmp_path / "r", adv_variant="drgrpo")
    cfg = json.loads((d / "config.json").read_text())
    for k in ("group_filter", "kl_beta", "task", "w_task", "backward_per_group"):
        del cfg[k]
    (d / "config.json").write_text(json.dumps(cfg))
    run = A._load_run(d)
    assert run["cfg"]["adv_variant"] == "drgrpo"
    assert (run["cfg"]["group_filter"], run["cfg"]["kl_beta"], run["cfg"]["task"],
            run["cfg"]["w_task"], run["cfg"]["backward_per_group"]) == ("none", 0.0, "speed", 1.0, False)


def test_the_remedy_flags_may_differ_inside_one_comparison_and_nothing_else_new():
    assert {"adv_variant", "group_filter", "kl_beta"} <= A.ALLOWED_CONFIG_DIFF
    for frozen in ("task", "backward_per_group", "max_new_tokens", "lr", "steps"):
        assert frozen not in A.ALLOWED_CONFIG_DIFF


# ---------------------------------------------------------------- arm names


def test_default_flags_keep_every_existing_arm_name():
    for task, specs in (("speed", A.ARM_SPECS), ("capacity", A.CAPACITY_ARM_SPECS)):
        for name, (arm, w, t) in specs.items():
            cfg = dict(A.LEGACY_CONFIG_DEFAULTS, task=task, arm=arm, w=w, w_task=t)
            assert A.arm_name(cfg) == name
    assert A.arm_name({"arm": "binary", "w": 5.0}) == "binary"


@pytest.mark.parametrize("flags,suffix", [
    ({"adv_variant": "drgrpo"}, "_drgrpo"),
    ({"adv_variant": "ngrpo_feasible"}, "_ngrpo"),
    ({"adv_variant": "ngrpo_sup"}, "_ngrposup"),
    ({"group_filter": "verdict_mixed"}, "_vfilter"),
    ({"group_filter": "reward_spread"}, "_sfilter"),
    ({"kl_beta": 0.04}, "_kl"),
    ({"adv_variant": "ngrpo_feasible", "group_filter": "verdict_mixed"}, "_ngrpo_vfilter"),
])
def test_each_remedy_flag_adds_one_suffix(flags, suffix):
    assert A.arm_name(dict({"arm": "binary", "w": 5.0, "task": "speed"}, **flags)) == "binary" + suffix
    assert A.arm_name(dict({"arm": "graded", "w": 5.0, "task": "capacity"}, **flags)) == "graded" + suffix


def test_the_verdict_at_five_times_its_weight_is_named_without_joining_the_arm_specs():
    assert A.BINARY_ABOVE_W == 5 * A.ARM_SPECS["binary"][1]
    assert "binary_5x" not in A.ARM_SPECS and "binary_5x" not in A.CAPACITY_ARM_SPECS
    assert A.arm_name({"arm": "binary", "w": A.BINARY_ABOVE_W}) == "binary_5x"
    assert A.arm_name({"arm": "binary", "w": A.BINARY_ABOVE_W,
                       "group_filter": "verdict_mixed"}) == "binary_5x_vfilter"
    assert A.arm_name({"arm": "graded", "w": A.BINARY_ABOVE_W}) == f"graded_w{A.BINARY_ABOVE_W}_t1"


def test_the_weight_above_the_speed_crossover_really_is_above_it():
    """The crossover is the largest task value a violator reaches minus the smallest a feasible
    answer reaches, taken from the reward code over the enumerated speeds."""
    for regime in ("A", "B"):
        import yaml
        vmax = float(yaml.safe_load(open(os.path.join(HERE, "configs", "regimes.yaml")))[regime]["v_max"])
        t = RW.speed_output_table(vmax)
        crossover = float(t["task"][~t["verdict"]].max() - t["task"][t["verdict"]].min())
        assert A.ARM_SPECS["binary"][1] < crossover < A.BINARY_ABOVE_W


def test_an_unknown_flag_value_is_refused_rather_than_named():
    with pytest.raises(ValueError):
        A.arm_name({"arm": "binary", "w": 5.0, "adv_variant": "ppo"})
    with pytest.raises(ValueError):
        A.arm_name({"arm": "binary", "w": 5.0, "group_filter": "dapo"})


def test_remedy_labels_are_composed_and_legacy_labels_are_untouched(tmp_path):
    runs = [A._load_run(write_run(tmp_path / "a", adv_variant="drgrpo")),
            A._load_run(write_run(tmp_path / "b", w=A.BINARY_ABOVE_W, group_filter="verdict_mixed"))]
    labels = A.arm_labels(runs)
    assert labels["binary_drgrpo"] == A.ARM_STYLE["binary"]["label"] + ", Dr. GRPO"
    assert labels["binary_5x_vfilter"].endswith("verifier filter")
    assert A.arm_labels([A._load_run(write_run(tmp_path / "c", group_filter="reward_spread"))])[
        "binary_sfilter"].endswith("identical-reward filter")
    assert A.arm_labels([]) == {k: v["label"] for k, v in A.ARM_STYLE.items()}


# ---------------------------------------------------------------- self-consistency


@pytest.mark.parametrize("variant", RW.ADV_VARIANTS)
@pytest.mark.parametrize("group_filter", RW.GROUP_FILTERS)
def test_honest_logs_are_self_consistent_under_every_variant_and_filter(tmp_path, variant, group_filter):
    run = A._load_run(write_run(tmp_path / "r", adv_variant=variant, group_filter=group_filter,
                                steps=4, groups=6))
    stats, worst = A.counterfactual_checks([run])
    assert worst["self"] == 0.0
    for k in ("kept_mismatch", "anchor_mismatch", "prompt_index_mismatch"):
        assert k not in stats


def test_capacity_logs_are_self_consistent_with_per_instance_anchors(tmp_path):
    run = A._load_run(write_run(tmp_path / "c", task="capacity", arm="none", w=0.0, vmax=0.03,
                                regime="H", adv_variant="ngrpo_sup", groups=6))
    stats, worst = A.counterfactual_checks([run])
    assert worst["self"] == 0.0 and "anchor_mismatch" not in stats
    anchors = {g["anchor"] for g in run["train"][0]["groups"]}
    assert len(anchors) > 1


def _tampered(tmp_path, variant, mutate, **kw):
    run = A._load_run(write_run(tmp_path / "t", adv_variant=variant, steps=4, groups=6, **kw))
    mutate(run)
    return A.counterfactual_checks([run])


def test_a_wrong_logged_anchor_is_caught(tmp_path):
    def m(run):
        run["train"][0]["groups"][0]["anchor"] = 0.123
    stats, _ = _tampered(tmp_path, "ngrpo_feasible", m)
    assert stats["anchor_mismatch"] == 1


def test_an_anchor_logged_for_a_variant_that_has_none_is_caught(tmp_path):
    def m(run):
        run["train"][1]["groups"][2]["anchor"] = 0.5
    stats, _ = _tampered(tmp_path, "grpo", m)
    assert stats["anchor_mismatch"] == 1


def test_a_flipped_kept_flag_is_caught(tmp_path):
    def m(run):
        g = run["train"][0]["groups"][0]
        g["kept"] = not g["kept"]
    stats, _ = _tampered(tmp_path, "grpo", m, group_filter="verdict_mixed")
    assert stats["kept_mismatch"] == 1


def test_logs_from_one_variant_under_a_config_naming_another_are_caught(tmp_path):
    def m(run):
        run["cfg"]["adv_variant"] = "grpo"
        for rec in run["train"]:
            for g in rec["groups"]:
                g["anchor"] = None
    _, worst = _tampered(tmp_path, "drgrpo", m)
    assert worst["self"] > 1e-6


def test_a_wrong_prompt_index_is_caught(tmp_path):
    def m(run):
        run["train"][2]["groups"][1]["prompt_index"] = 0
    stats, _ = _tampered(tmp_path, "grpo", m)
    assert stats["prompt_index_mismatch"] == 1


def test_a_legacy_log_passes_without_new_keys(tmp_path):
    d = write_run(tmp_path / "legacy", steps=3, groups=6)
    run = A._load_run(d)
    for rec in run["train"]:
        for g in rec["groups"]:
            for k in ("kept", "anchor", "prompt_index", "n_tokens"):
                g.pop(k)
    stats, worst = A.counterfactual_checks([run])
    assert worst["self"] == 0.0
    assert not {"kept_mismatch", "anchor_mismatch", "prompt_index_mismatch"} & set(stats)


def test_a_tampered_run_fails_the_self_consistency_flag_in_checks_json(tmp_path):
    root = tmp_path / "runs"
    d = write_run(root / "r", adv_variant="ngrpo_sup", steps=4, groups=6)
    lines = (d / "metrics.jsonl").read_text().splitlines()
    rec = json.loads(lines[1])
    rec["groups"][0]["anchor"] = -1.0
    lines[1] = json.dumps(rec)
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    r = _analyze(root, tmp_path / "an")
    assert r.returncode == 0, r.stderr[-1500:]
    checks = json.loads((tmp_path / "an" / "checks.json").read_text())
    assert checks["trainer_self_consistent"] is False


# ---------------------------------------------------------------- variant algebra


def test_the_variant_algebra_sorts_every_all_violating_group(tmp_path):
    spread = [40.0, 250.0, 900.0, None]              # task values differ, all violate
    capped = [300.0, 400.0, 500.0, 600.0]            # all at the task cap, so no spread
    run = A._load_run(write_run(tmp_path / "alg", steps=2, groups=4,
                                items_fn=lambda rng, gi: spread if gi % 2 == 0 else capped))
    alg = A.variant_algebra([run])
    n, nz = alg["all_violating_groups"], alg["all_violating_zero_spread"]
    assert (n, nz) == (8, 4)
    cell = alg["variants"]
    for v in ("grpo", "drgrpo", "ngrpo_sup"):
        assert cell[v]["binary"]["cancels"] == n, v
    # the light verdict leaves every capped reward above the feasible anchor
    assert cell["ngrpo_feasible"]["binary"]["acts_with_spread"] == n - nz
    assert cell["ngrpo_feasible"]["binary"]["eps_only_zero_spread"] == nz
    assert cell["ngrpo_feasible"]["binary"]["acts_zero_spread"] == 0
    # five times the verdict pushes the capped rewards below it, which flips the sign
    assert cell["ngrpo_feasible"]["binary_5x"]["acts_zero_spread"] == nz
    assert cell["ngrpo_sup"]["binary_5x"]["acts"] == n
    for v in RW.ADV_VARIANTS:
        for d, c in cell[v].items():
            assert (c["cancels"] + c["acts_with_spread"] + c["acts_zero_spread"]
                    + c["eps_only_zero_spread"]) == n, (v, d)
            assert c["acts"] == c["acts_with_spread"] + c["acts_zero_spread"]
    keep = alg["filters_keep_all_violating"]
    assert keep == {"none": n, "verdict_mixed": 0, "reward_spread": n - nz}
    assert alg["design_weights"]["binary_5x"] == A.BINARY_ABOVE_W
    assert list(alg["by_regime"]) == ["A"]
    assert alg["by_regime"]["A"]["variants"] == cell


def test_the_variant_algebra_reports_each_regime_and_their_total(tmp_path):
    items = lambda rng, gi: [40.0, 250.0, 900.0, None]  # noqa: E731
    a = A._load_run(write_run(tmp_path / "a", steps=2, groups=2, items_fn=items))
    b = A._load_run(write_run(tmp_path / "b", steps=3, groups=2, vmax=30.0, regime="B", items_fn=items))
    alg = A.variant_algebra([a, b])
    assert list(alg["by_regime"]) == ["A", "B"]
    assert alg["by_regime"]["A"]["all_violating_groups"] == 4
    assert alg["by_regime"]["B"]["all_violating_groups"] == 6
    assert alg["all_violating_groups"] == 10
    for v in RW.ADV_VARIANTS:
        for d in A.ALGEBRA_DESIGNS:
            for k in ("cancels", "acts", "acts_with_spread", "acts_zero_spread", "eps_only_zero_spread"):
                assert alg["variants"][v][d][k] == sum(alg["by_regime"][r]["variants"][v][d][k]
                                                       for r in ("A", "B"))


def test_the_literal_anchor_is_reported_by_instance_and_task_weight(tmp_path):
    spec = tasks.get("capacity")
    with_task = A._load_run(write_run(tmp_path / "t1", task="capacity", vmax=0.03, regime="H",
                                      steps=3, groups=6, rng_seed=3))
    no_task = A._load_run(write_run(tmp_path / "t0", task="capacity", vmax=0.03, regime="H",
                                    steps=3, groups=6, w_task=0.0, rng_seed=4))
    alg = A.variant_algebra([with_task, no_task])
    lit = alg["literal_anchor_by_instance"]["H"]
    assert set(lit) == {"0", "1"}
    groups = cancels = 0
    for wt, cells in lit.items():
        assert list(cells) == sorted(cells, key=int)
        for inst, c in cells.items():
            for design, arm, w in (("anchor_verdict", "binary", A.ALGEBRA_DESIGNS["binary"]),
                                   ("anchor_none", "none", 0.0)):
                assert c[design] == pytest.approx(RW.reward_anchor(
                    "ngrpo_sup", arm, w, 0.03, spec=spec, w_task=float(wt), instance=int(inst)))
            assert c["cancels"] + c["acts"] == c["all_violating_groups"]
            assert c["worst_diff"] >= 0.0
            groups += c["all_violating_groups"]
            cancels += c["cancels"]
    reg = alg["by_regime"]["H"]
    assert set(reg["by_task_weight"]) == {"0", "1"}
    assert reg["all_violating_groups"] == sum(
        reg["by_task_weight"][wt]["all_violating_groups"] for wt in ("0", "1"))
    for v in RW.ADV_VARIANTS:
        for d in A.ALGEBRA_DESIGNS:
            for k in ("cancels", "acts", "acts_with_spread", "acts_zero_spread", "eps_only_zero_spread"):
                assert reg["variants"][v][d][k] == sum(
                    reg["by_task_weight"][wt]["variants"][v][d][k] for wt in ("0", "1")), (v, d, k)
    sup = alg["by_regime"]["H"]["variants"]["ngrpo_sup"]["binary"]
    assert groups == alg["by_regime"]["H"]["all_violating_groups"]
    assert cancels == sup["cancels"]


def test_final_answers_count_empty_and_unparseable_answers(tmp_path):
    cap = A._load_run(write_run(tmp_path / "c", task="capacity", vmax=0.03, regime="H", steps=2,
                                groups=3, final_feasible=True))
    mixed = A._load_run(write_run(tmp_path / "m", task="capacity", vmax=0.03, regime="E", steps=2,
                                  groups=3))
    speed = A._load_run(write_run(tmp_path / "s", steps=2, groups=3))
    rows = {r["regime"]: r for r in A.final_answers([cap, mixed, speed])}
    assert rows["H"]["eval_n_final"] == 12 and rows["H"]["eval_step_final"] == 2
    assert rows["H"]["eval_empty_selections_final"] == 12 and rows["H"]["eval_unparseable_final"] == 0
    final = mixed["evals"][-1][tasks.get("capacity").EVAL_ITEMS_KEY]
    assert rows["E"]["eval_unparseable_final"] == sum(x is None for x in final)
    assert rows["E"]["eval_empty_selections_final"] == sum(
        1 for x in final if x is not None and len(x) == 1)
    assert rows["A"]["eval_empty_selections_final"] is None
    assert rows["A"]["eval_unparseable_final"] == sum(
        x is None for x in speed["evals"][-1][tasks.get("speed").EVAL_ITEMS_KEY])


# ---------------------------------------------------------------- counters and timing


def _run_dict(train, cfg=None):
    base = {"regime": "A", "arm": "binary", "w": 5.0, "seed": 0}
    base.update(cfg or {})
    return {"cfg": base, "train": train, "evals": [], "dir": type("D", (), {"name": "x"})()}


def test_mixed_and_all_feasible_counts_come_from_the_group_flags():
    groups = [
        {"rewards": [0, 0, 0, 0], "all_violating": True, "has_violator": True, "adv": [0] * 4},
        {"rewards": [0, 1, 0, 0], "all_violating": False, "has_violator": True, "adv": [1] * 4},
        {"rewards": [0, 1, 0, 0], "all_violating": False, "has_violator": True, "adv": [1] * 4},
        {"rewards": [1, 1, 1, 1], "all_violating": False, "has_violator": False, "adv": [0] * 4},
    ]
    c = A.group_counters([_run_dict([{"step": 1, "groups": groups}])])[0]
    assert c["mixed_groups"] == 2 and c["all_feasible_groups"] == 1
    assert c["mixed_groups"] + c["all_feasible_groups"] + c["all_violating"] == c["groups"]
    assert list(c)[-5:] == ["mixed_groups", "all_feasible_groups", "kept_groups",
                            "opt_steps_skipped", "momentum_only_steps"]


def test_a_log_without_the_violator_flag_reports_no_group_type_counts():
    groups = [{"rewards": [1.0] * 4, "all_violating": True}]
    c = A.group_counters([_run_dict([{"step": 1, "groups": groups}])])[0]
    assert c["mixed_groups"] is None and c["all_feasible_groups"] is None


def test_update_counts_separate_skipped_momentum_and_nonzero_steps():
    z = {"rewards": [1.0] * 4, "adv": [0.0] * 4, "all_violating": True, "has_violator": True}
    nz = {"rewards": [1.0, 2.0, 3.0, 4.0], "adv": [1.0, 0.0, 0.0, -1.0], "all_violating": True,
          "has_violator": True}
    dropped = dict(nz, kept=False)
    train = [
        {"step": 1, "groups": [z]},                                   # zero before any push
        {"step": 2, "groups": [nz]},                                  # nonzero, legacy
        {"step": 3, "groups": [z]},                                   # momentum only
        {"step": 4, "groups": [dropped], "opt_step_taken": False},    # skipped
        {"step": 5, "groups": [nz], "opt_step_taken": True, "grad_norm_preclip": 0.0},  # momentum
    ]
    counts = A._update_counts(_run_dict(train))
    assert counts == {"mixed_groups": 0, "all_feasible_groups": 0, "kept_groups": 4,
                      "opt_steps_skipped": 1, "momentum_only_steps": 2}
    row = A.remedy_timing([_run_dict(train)])[0]
    assert (row["opt_steps_taken"], row["opt_steps_skipped"], row["momentum_only_steps"]) == (4, 1, 2)
    assert row["first_skipped_step"] == 4 and row["last_nonzero_gradient_step"] == 2
    assert row["kept_groups"] == 4 and row["oversampling_factor"] == pytest.approx(5 / 4)


def test_a_run_that_kept_nothing_has_no_oversampling_factor_and_is_checked_for_no_change(tmp_path):
    d = write_run(tmp_path / "none_kept", group_filter="verdict_mixed", steps=2, groups=2,
                  items_fn=lambda rng, gi: [40.0, 250.0, None, 900.0])
    run = A._load_run(d)
    row = A.remedy_timing([run])[0]
    assert row["kept_groups"] == 0 and row["oversampling_factor"] is None
    assert row["first_kept_step"] is None and row["opt_steps_skipped"] == 2
    unchanged = A.no_update_runs([run])
    assert len(unchanged) == 1 and unchanged[0]["eval_unchanged"] is True
    # an evaluation that moved without an update is exactly what this check exists to catch
    run["evals"][-1]["eval_speeds"] = [1.0] * len(run["evals"][-1]["eval_speeds"])
    assert A.no_update_runs([run])[0]["eval_unchanged"] is False
    # a run that took even one step is not a no-update run
    run["train"][0]["opt_step_taken"] = True
    assert A.no_update_runs([run]) == []


# ---------------------------------------------------------------- fingerprint waiver


def test_the_committed_equivalence_file_loads_and_is_not_a_training_source():
    assert A.load_equivalence() == [] or isinstance(A.load_equivalence(), list)
    assert not any(s.endswith("equivalence.json") for s in A.TRAINING_SOURCES)
    assert not any(s.endswith("equivalence.json") for s in T.TRAINING_SOURCES)


def _equivalence_doc(**fp_over):
    fp = {"name": "F1", "reference": "runs/ref", "new": "runs/new", "rerun_at_base": "runs/base",
          "steps": 3, "outcome": "pass"}
    fp.update(fp_over)
    return {"schema": 1, "waivers": [{"base_sha": "base", "new_sha": "new",
                                      "new_training_sources_digest": "D", "fingerprints": [fp]}]}


@pytest.mark.parametrize("over,needle", [
    ({"new": "/abs/runs/new"}, "absolute"),
    ({"reference": "runs/../../elsewhere"}, "repo-relative"),
    ({"rerun_at_base": "~/runs"}, "absolute"),
    ({"outcome": "maybe"}, "outcome"),
    ({"steps": 0}, "steps"),
])
def test_a_malformed_equivalence_file_is_refused(tmp_path, over, needle):
    p = tmp_path / "equivalence.json"
    p.write_text(json.dumps(_equivalence_doc(**over)))
    with pytest.raises(A.EquivalenceError) as e:
        A.load_equivalence(p)
    assert needle in str(e.value)


def test_a_wrong_schema_is_refused(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"schema": 2, "waivers": []}))
    with pytest.raises(A.EquivalenceError):
        A.load_equivalence(p)


def _three(tmp_path, change_new=False, change_base=False, steps=3):
    ref = write_run(tmp_path / "runs" / "ref", steps=steps, rng_seed=5)
    new = tmp_path / "runs" / "new"
    base = tmp_path / "runs" / "base"
    shutil.copytree(ref, new)
    shutil.copytree(ref, base)
    for d, change in ((new, change_new), (base, change_base)):
        if change:
            lines = (d / "metrics.jsonl").read_text().splitlines()
            rec = json.loads(lines[1])
            rec["groups"][0]["rewards"][0] += 1.0
            lines[1] = json.dumps(rec)
            (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    return [A._load_run(x) for x in (ref, new, base)]


def test_fingerprint_outcomes(tmp_path):
    assert A.fingerprint_outcome(*_three(tmp_path / "a"), 3) == "pass"
    assert A.fingerprint_outcome(*_three(tmp_path / "b", change_new=True, change_base=True), 3) \
        == "conditional_pass"
    assert A.fingerprint_outcome(*_three(tmp_path / "c", change_new=True), 3) == "stop"
    ref, new, base = _three(tmp_path / "d")
    new["prov"]["final_param_checksum"] = "other"
    assert A.fingerprint_outcome(ref, new, base, 3) == "stop"
    # over fewer steps than the runs logged the checksum is not compared
    assert A.fingerprint_outcome(ref, new, base, 2) == "pass"
    with pytest.raises(A.EquivalenceError):
        A.fingerprint_outcome(ref, new, base, 4)


def _drift_three(tmp_path, bump_reward_new=False, base_equals_ref=False):
    """Three runs where the new run and the old-commit rerun sample different answers from the
    reference at step 1, the way floating-point drift in the weights shows up in the logs."""
    ref = write_run(tmp_path / "runs" / "ref", steps=3, rng_seed=5)
    new, base = tmp_path / "runs" / "new", tmp_path / "runs" / "base"
    shutil.copytree(ref, new)
    shutil.copytree(ref, base)

    def edit(d, fn):
        lines = (d / "metrics.jsonl").read_text().splitlines()
        rec = json.loads(lines[1])
        fn(rec)
        lines[1] = json.dumps(rec)
        (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")

    def answer(value):
        def fn(rec):
            rec["groups"][0]["speeds"][0] = value
        return fn

    def reward(rec):
        rec["groups"][0]["rewards"][0] += 1.0

    edit(new, reward if bump_reward_new else answer(997.0))
    if not base_equals_ref:
        edit(base, answer(998.0))
    return [A._load_run(x) for x in (ref, new, base)]


def test_drift_passes_only_when_allowed_and_both_runs_leave_the_reference(tmp_path):
    runs = _drift_three(tmp_path / "a")
    assert A.fingerprint_outcome(*runs, 3) == "stop"
    assert A.fingerprint_outcome(*runs, 3, allow_drift=True) == "drift_pass"


def test_drift_stops_when_identical_answers_carry_different_rewards(tmp_path):
    runs = _drift_three(tmp_path / "b", bump_reward_new=True)
    assert A.fingerprint_outcome(*runs, 3, allow_drift=True) == "stop"


def test_drift_stops_when_the_old_commit_rerun_reproduces_the_reference(tmp_path):
    runs = _drift_three(tmp_path / "c", base_equals_ref=True)
    assert A.fingerprint_outcome(*runs, 3, allow_drift=True) == "stop"


def test_a_recorded_drift_pass_reverifies_with_its_own_flag(tmp_path, monkeypatch):
    _drift_three(tmp_path)
    monkeypatch.setattr(A, "_REPO_ROOT", tmp_path)

    def between(a, b):
        if a == b or {a, b} <= {"old", "base"}:
            return []
        return ["experiments/cancellation/rewards.py"]
    monkeypatch.setattr(A, "training_sources_between", between)
    old = A._load_run(write_run(tmp_path / "cmp" / "old", seed=0, git_sha="old"))
    new = A._load_run(write_run(tmp_path / "cmp" / "new", seed=1, git_sha="new", digest="D",
                                adv_variant="drgrpo"))
    allowed = _equivalence_doc(outcome="drift_pass", allow_drift=True)["waivers"]
    assert A.matched_budget_check([old, new], notes=[], equivalence=allowed) == []
    not_allowed = _equivalence_doc(outcome="drift_pass")["waivers"]
    problems = A.matched_budget_check([old, new], notes=[], equivalence=not_allowed)
    assert problems and any("the logs give stop" in p for p in problems), problems


def _waiver_setup(tmp_path, monkeypatch, outcome="pass", digest="D", change_new=False,
                  change_base=False, drop=None):
    _three(tmp_path, change_new=change_new, change_base=change_base)
    if drop:
        shutil.rmtree(tmp_path / "runs" / drop)
    monkeypatch.setattr(A, "_REPO_ROOT", tmp_path)

    def between(a, b):
        if a == b or {a, b} <= {"old", "base"}:
            return []
        return ["experiments/cancellation/rewards.py"]
    monkeypatch.setattr(A, "training_sources_between", between)
    old = A._load_run(write_run(tmp_path / "cmp" / "old", seed=0, git_sha="old"))
    new = A._load_run(write_run(tmp_path / "cmp" / "new", seed=1, git_sha="new", digest=digest,
                                adv_variant="drgrpo"))
    equivalence = _equivalence_doc(outcome=outcome)["waivers"]
    return [old, new], equivalence


def test_a_reverified_waiver_admits_the_pair_and_is_reported(tmp_path, monkeypatch):
    runs, eq = _waiver_setup(tmp_path, monkeypatch)
    notes = []
    assert A.matched_budget_check(runs, notes=notes, equivalence=eq) == []
    assert notes and notes[0]["fingerprint_waiver"]["outcomes"] == {"F1": "pass"}
    assert notes[0]["training_sources_changed"] == ["experiments/cancellation/rewards.py"]


def test_a_conditional_pass_that_reverifies_also_admits(tmp_path, monkeypatch):
    runs, eq = _waiver_setup(tmp_path, monkeypatch, outcome="conditional_pass",
                             change_new=True, change_base=True)
    assert A.matched_budget_check(runs, notes=[], equivalence=eq) == []


@pytest.mark.parametrize("kw,needle", [
    ({"outcome": "stop", "change_new": True}, "recorded stop"),
    ({"outcome": "pass", "change_new": True, "change_base": True}, "the logs give conditional_pass"),
    ({"drop": "base"}, "cannot be re-verified"),
    ({"digest": "not-D"}, "training_sources_digest"),
])
def test_a_waiver_that_does_not_hold_is_refused(tmp_path, monkeypatch, kw, needle):
    runs, eq = _waiver_setup(tmp_path, monkeypatch, **kw)
    notes = []
    problems = A.matched_budget_check(runs, notes=notes, equivalence=eq)
    assert problems and any(needle in p for p in problems), problems


def test_no_waiver_keeps_the_strict_refusal(tmp_path, monkeypatch):
    runs, _ = _waiver_setup(tmp_path, monkeypatch)
    problems = A.matched_budget_check(runs, notes=[], equivalence=[])
    assert len(problems) == 1 and "rewards.py" in problems[0]


def test_an_invalid_equivalence_file_is_a_refusal_not_a_crash(tmp_path, monkeypatch):
    runs, _ = _waiver_setup(tmp_path, monkeypatch)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setattr(A, "EQUIVALENCE_PATH", bad)
    problems = A.matched_budget_check(runs, notes=[])
    assert problems and "equivalence.json is invalid" in problems[0]


def test_the_fingerprint_command_records_repo_relative_paths(tmp_path, monkeypatch):
    _three(tmp_path)
    monkeypatch.setattr(A, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(A, "training_sources_between", lambda a, b: [])
    for name, sha in (("ref", "old"), ("new", "new"), ("base", "base")):
        p = tmp_path / "runs" / name / "provenance.json"
        prov = json.loads(p.read_text())
        prov["git_sha"] = sha
        if name == "new":
            prov["training_sources_digest"] = "D"
        p.write_text(json.dumps(prov))
    eq = tmp_path / "equivalence.json"
    argv = ["fingerprint", "--name", "F1", "--reference", "runs/ref", "--new", "runs/new",
            "--rerun-at-base", "runs/base", "--steps", "3", "--base-sha", "base", "--new-sha", "new",
            "--equivalence", str(eq)]
    assert A.main(argv) == 0
    doc = json.loads(eq.read_text())
    assert A.load_equivalence(eq)[0]["fingerprints"][0]["outcome"] == "pass"
    assert doc["waivers"][0]["new_training_sources_digest"] == "D"
    assert A.main(argv[:6] + [str(tmp_path / "runs" / "new")] + argv[7:]) == 2


# ---------------------------------------------------------------- assemble_runs


def _two_arrays(tmp_path):
    a, b = tmp_path / "111", tmp_path / "222"
    write_run(a / "1_A_binary_s0", seed=0)
    write_run(a / "2_A_none_s0", arm="none", w=0.0, seed=0)
    write_run(a / "3_B_binary_s0", vmax=30.0, regime="B", seed=0)
    (a / "analysis").mkdir()
    write_run(b / "1_A_binary_drgrpo_s0", seed=0, adv_variant="drgrpo")
    return a, b


def test_assemble_links_runs_relatively_and_the_analysis_reads_them(tmp_path):
    a, b = _two_arrays(tmp_path)
    out = tmp_path / "combined"
    assert assemble_runs.main(["--out", str(out), "--src", f"{a}:*_A_*", "--src", str(b)]) == 0
    links = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert links == ["111_1_A_binary_s0", "111_2_A_none_s0", "222_1_A_binary_drgrpo_s0"]
    for p in out.iterdir():
        if p.is_symlink():
            assert not os.path.isabs(os.readlink(p))
    record = json.loads((out / "assembled.json").read_text())
    assert all(not os.path.isabs(r["target"]) for r in record["runs"])
    assert {A.arm_name(r["cfg"]) for r in A.load_runs(out)} == {"binary", "none", "binary_drgrpo"}


def test_assemble_refuses_a_duplicate_run(tmp_path):
    a, b = _two_arrays(tmp_path)
    write_run(b / "9_A_binary_s0", seed=0)
    out = tmp_path / "combined"
    assert assemble_runs.main(["--out", str(out), "--src", f"{a}:*_A_*", "--src", str(b)]) == 3
    assert not out.exists()


def test_assemble_refuses_a_run_without_done(tmp_path):
    a, b = _two_arrays(tmp_path)
    (b / "1_A_binary_drgrpo_s0" / "DONE").unlink()
    out = tmp_path / "combined"
    assert assemble_runs.main(["--out", str(out), "--src", str(a), "--src", str(b)]) == 3
    assert not out.exists()


def test_assemble_refuses_a_non_empty_output_and_an_empty_match(tmp_path):
    a, b = _two_arrays(tmp_path)
    out = tmp_path / "combined"
    out.mkdir()
    (out / "x").write_text("")
    assert assemble_runs.main(["--out", str(out), "--src", str(b)]) == 3
    assert assemble_runs.main(["--out", str(tmp_path / "c2"), "--src", f"{a}:*_Z_*"]) == 3


# ---------------------------------------------------------------- end to end through the table


REMEDY_ARMS = {
    "binary": {}, "none": {"arm": "none", "w": 0.0},
    "binary_drgrpo": {"adv_variant": "drgrpo"},
    "binary_ngrpo": {"adv_variant": "ngrpo_feasible"},
    "binary_ngrposup": {"adv_variant": "ngrpo_sup"},
    "binary_vfilter": {"group_filter": "verdict_mixed"},
    "binary_sfilter": {"group_filter": "reward_spread"},
    "binary_5x": {"w": A.BINARY_ABOVE_W},
    "binary_5x_ngrpo": {"w": A.BINARY_ABOVE_W, "adv_variant": "ngrpo_feasible"},
    "binary_5x_vfilter": {"w": A.BINARY_ABOVE_W, "group_filter": "verdict_mixed"},
}


@pytest.fixture(scope="module")
def remedy_analysis(tmp_path_factory):
    root = tmp_path_factory.mktemp("remedy")
    base, new = root / "base", root / "new"
    for name, kw in REMEDY_ARMS.items():
        for seed in (0, 1, 2):
            parent = base if name in ("binary", "none") else new
            write_run(parent / f"{name}_s{seed}", seed=seed, steps=3, groups=6,
                      final_feasible=(name == "binary_ngrpo"), **kw)
    combined = root / "combined"
    assert assemble_runs.main(["--out", str(combined), "--src", str(base), "--src", str(new)]) == 0
    an = root / "analysis"
    proc = _analyze(combined, an)
    return {"root": root, "analysis": an, "proc": proc}


def test_the_analysis_accepts_the_remedy_run_set(remedy_analysis):
    r = remedy_analysis["proc"]
    assert r.returncode == 0, r.stderr[-2000:]
    an = remedy_analysis["analysis"]
    checks = json.loads((an / "checks.json").read_text())
    assert checks["n_runs"] == 3 * len(REMEDY_ARMS)
    assert not checks["matched_budget_problems"] and not checks["vacuous_checks"]
    for k in ("P1_pass", "P2_pass", "P2_boundary_pass", "trainer_self_consistent", "no_update_eval_pass"):
        assert checks[k] is True, k
    for row in checks["group_counters"]:
        assert {"mixed_groups", "all_feasible_groups", "kept_groups", "opt_steps_skipped",
                "momentum_only_steps"} <= set(row)
    timing = list(csv.DictReader((an / "remedy_timing.csv").open()))
    assert {t["arm"] for t in timing} == set(REMEDY_ARMS) and len(timing) == 3 * len(REMEDY_ARMS)
    labels = json.loads((an / "arm_labels.json").read_text())
    assert set(REMEDY_ARMS) <= set(labels)


# fallback_group_counts adds eight fields beyond the regime, arm and seed triple every
# per-run table in this file already carries.
FALLBACK_FIELDS = ("penalty", "w_task", "stand_in_reach", "groups_with_unparseable",
                   "all_feasible_with_unparseable", "all_feasible_with_unparseable_zero_spread",
                   "unparseable_outranks_parsed_violator", "unparseable_beside_residual_above_reach")


def test_the_fallback_counts_are_written_beside_the_group_counters(remedy_analysis):
    an = remedy_analysis["analysis"]
    checks = json.loads((an / "checks.json").read_text())
    fallback = checks["fallback_group_counts"]
    assert len(fallback) == 3 * len(REMEDY_ARMS)
    for row in fallback:
        assert set(FALLBACK_FIELDS) <= set(row)
    csv_rows = list(csv.DictReader((an / "fallback_group_counts.csv").open()))
    assert len(csv_rows) == len(fallback)
    assert set(FALLBACK_FIELDS) <= set(csv_rows[0])
    key = lambda r: (r["regime"], r["arm"], int(r["seed"]))  # noqa: E731
    assert {key(r) for r in csv_rows} == {key(r) for r in fallback}
    by_key = {key(r): r for r in fallback}
    for r in csv_rows:
        want = by_key[key(r)]
        for field in ("groups_with_unparseable", "all_feasible_with_unparseable",
                      "all_feasible_with_unparseable_zero_spread",
                      "unparseable_outranks_parsed_violator",
                      "unparseable_beside_residual_above_reach"):
            assert int(r[field]) == want[field]


@requires_paper
def test_the_remedy_table_has_every_arm_and_its_macros(remedy_analysis, tmp_path):
    an = remedy_analysis["analysis"]
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path), "--table", "remedy",
                        "--macro-prefix", "rem", "--macros-file", "rem_macros.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    body = (tmp_path / "remedy_regimeA.tex").read_text()
    labels = json.loads((an / "arm_labels.json").read_text())
    for arm in REMEDY_ARMS:
        assert labels[arm] in body, arm
    row = next(line for line in body.splitlines() if line.startswith(labels["binary_ngrpo"] + " &"))
    assert "3 of 3" in row
    macros = (tmp_path / "rem_macros.tex").read_text()
    assert "\\newcommand{\\remFeasABinaryNgrpo}{3}" in macros
    for word in ("ABinaryDrGrpo", "ABinaryNgrpoSup", "ABinaryVFilter", "ABinarySFilter",
                 "ABinaryFive", "ABinaryFiveNgrpo", "ABinaryFiveVFilter", "ANone"):
        assert f"\\remFeas{word}}}" in macros, word
    assert "\\remAlgANgrpoFeasibleBinaryCancels" in macros
    assert "\\remAlgAKeepVerdictMixed}{0}" in macros
    assert "\\remAlgAAllViol" in macros
    names = [m for m in __import__("re").findall(r"\\newcommand\{\\(\w+)\}", macros)]
    assert len(names) == len(set(names)) and all(n.isalpha() for n in names)


@requires_paper
def test_the_remedy_table_stacks_both_new_headers_and_drops_the_old_one(remedy_analysis, tmp_path):
    """The table once set "kept groups" and "skipped steps" as single-line headers. The
    second one alone pushed a twelve-arm table past the page edge."""
    an = remedy_analysis["analysis"]
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path), "--table", "remedy",
                        "--macro-prefix", "rem", "--macros-file", "rem_macros.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    body = (tmp_path / "remedy_regimeA.tex").read_text()
    assert r"\shortstack{\textbf{kept}\\\textbf{groups}}" in body
    assert r"\shortstack{\textbf{skipped}\\\textbf{steps}}" in body
    assert r"\textbf{kept groups}" not in body


@requires_paper
def test_the_long_remedy_table_lists_every_seed_and_keeps_the_macros(remedy_analysis, tmp_path):
    an = remedy_analysis["analysis"]
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path), "--table", "remedy",
                        "--macro-prefix", "rem", "--macros-file", "rem_macros.tex", "--long"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    body = (tmp_path / "remedy_regimeA.tex").read_text()
    rows = [line for line in body.splitlines() if line.endswith("\\\\") and "textbf" not in line]
    assert len(rows) == 3 * len(REMEDY_ARMS)
    labels = json.loads((an / "arm_labels.json").read_text())
    for arm in REMEDY_ARMS:
        assert sum(1 for line in rows if line.startswith(labels[arm] + " &")) == 1, arm
    assert "of 3" not in body
    assert "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{3.6cm}" in body
    macros = (tmp_path / "rem_macros.tex").read_text()
    assert "\\newcommand{\\remFeasABinaryNgrpo}{3}" in macros
    # The per-seed counts the prose quotes come from the analysis files, seed by seed in order.
    timing = sorted((t for t in csv.DictReader((an / "remedy_timing.csv").open())
                     if t["arm"] == "binary_vfilter"), key=lambda t: int(t["seed"]))
    counters = sorted((c for c in json.loads((an / "checks.json").read_text())["group_counters"]
                       if c["arm"] == "binary_vfilter"), key=lambda c: int(c["seed"]))
    for stem, src, key in (("StepsTaken", timing, "opt_steps_taken"),
                           ("AllViol", timing, "all_violating_groups"),
                           ("AllFeas", counters, "all_feasible_groups")):
        want = ", ".join(f"{int(float(x[key])):,}".replace(",", "{,}") for x in src)
        assert f"\\newcommand{{\\rem{stem}ABinaryVFilterSeeds}}{{{want}}}" in macros, stem


@requires_paper
def test_the_remedy_table_refuses_an_analysis_without_group_counters(remedy_analysis, tmp_path):
    an = tmp_path / "an"
    shutil.copytree(remedy_analysis["analysis"], an)
    checks = json.loads((an / "checks.json").read_text())
    checks["group_counters"] = [c for c in checks["group_counters"] if c["arm"] != "binary_vfilter"]
    (an / "checks.json").write_text(json.dumps(checks))
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path / "t"), "--table", "remedy",
                        "--macro-prefix", "rem", "--macros-file", "m.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 3 and "group_counters lacks" in r.stderr


@requires_paper
def test_the_remedy_table_refuses_an_arm_it_has_no_row_for(tmp_path):
    root = tmp_path / "runs"
    write_run(root / "odd", w=7.0)
    write_run(root / "plain", seed=1)
    an = tmp_path / "an"
    assert _analyze(root, an).returncode == 0
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path / "t"), "--table", "remedy",
                        "--macro-prefix", "rem", "--macros-file", "m.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 3 and "no row for binary_w7.0_t1" in r.stderr


@requires_paper
def test_the_remedy_table_refuses_an_analysis_with_a_failed_check(remedy_analysis, tmp_path):
    an = tmp_path / "an"
    shutil.copytree(remedy_analysis["analysis"], an)
    checks = json.loads((an / "checks.json").read_text())
    checks["no_update_eval_pass"] = False
    (an / "checks.json").write_text(json.dumps(checks))
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path / "t"), "--table", "remedy",
                        "--macro-prefix", "rem", "--macros-file", "m.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 3 and "no_update_eval_pass" in r.stderr


@requires_paper
def test_the_kl_table_lists_each_design_with_and_without_the_penalty(tmp_path):
    root = tmp_path / "runs"
    for arm, w in (("graded", 5.0), ("binary", 5.0), ("none", 0.0)):
        for kl in (0.0, 0.04):
            for seed in (0, 1, 2):
                write_run(root / f"{arm}_{kl}_s{seed}", task="capacity", arm=arm, w=w, vmax=0.03,
                          regime="H", seed=seed, kl_beta=kl, groups=6)
    an = tmp_path / "an"
    proc = _analyze(root, an)
    assert proc.returncode == 0, proc.stderr[-2000:]
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path / "t"), "--table", "kl",
                        "--macro-prefix", "capkl", "--macros-file", "capkl_macros.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    body = (tmp_path / "t" / "kl_regimeH.tex").read_text()
    labels = json.loads((an / "arm_labels.json").read_text())
    for arm in ("graded", "graded_kl", "binary", "binary_kl", "none", "none_kl"):
        assert labels[arm] in body, arm
    macros = (tmp_path / "t" / "capkl_macros.tex").read_text()
    assert "\\capklKLHBinaryKlSeeds" in macros
    # Each penalized run against the same design and seed without the penalty, counted per level.
    per_run = {(r["arm"], r["seed"]): r for r in csv.DictReader((an / "final_per_run.csv").open())}
    seeds = ("0", "1", "2")
    lower = sum(1 for x in ("graded", "binary", "none") for s in seeds
                if float(per_run[(x + "_kl", s)]["kl_seq_mean_final"]) < float(per_run[(x, s)]["kl_seq_mean_final"]))
    assert f"\\newcommand{{\\capklKlLowerH}}{{{lower}}}" in macros
    assert "\\newcommand{\\capklKlPairsH}{9}" in macros
    assert "\\newcommand{\\capklRunsKlH}{9}" in macros

    def unparsed(arms):
        return sum(1 for x in arms for s in seeds if float(per_run[(x, s)]["eval_parse_fail_rate_final"]) == 1.0)

    assert f"\\newcommand{{\\capklUnparsedKlH}}{{{unparsed(('graded_kl', 'binary_kl', 'none_kl'))}}}" in macros
    assert f"\\newcommand{{\\capklUnparsedPlainH}}{{{unparsed(('graded', 'binary', 'none'))}}}" in macros
    # The worst parse-failure rate over every evaluation, not only the last one.
    peaks = {}
    for c in csv.DictReader((an / "curves_eval.csv").open()):
        k = (c["arm"], c["seed"])
        peaks[k] = max(peaks.get(k, 0.0), float(c["eval_parse_fail_rate"]))
    want = ", ".join(f'{peaks[("binary_kl", s)]:.2f}' for s in seeds)
    assert f"\\newcommand{{\\capklPeakUnparseableHBinaryKlSeeds}}{{{want}}}" in macros


@requires_paper
def test_the_kl_table_refuses_a_penalized_run_without_its_twin(tmp_path):
    root = tmp_path / "runs"
    for seed in (0, 1):
        write_run(root / f"graded_s{seed}", task="capacity", arm="graded", w=5.0, vmax=0.03,
                  regime="H", seed=seed, kl_beta=0.0, groups=6)
    for seed in (0, 1, 2):
        write_run(root / f"graded_kl_s{seed}", task="capacity", arm="graded", w=5.0, vmax=0.03,
                  regime="H", seed=seed, kl_beta=0.04, groups=6)
    an = tmp_path / "an"
    proc = _analyze(root, an)
    assert proc.returncode == 0, proc.stderr[-2000:]
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path / "t"), "--table", "kl",
                        "--macro-prefix", "capkl", "--macros-file", "capkl_macros.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "no unpenalized twin" in r.stderr and "graded_kl seed 2" in r.stderr


@requires_paper
def test_the_below_table_serves_the_capacity_level_from_a_real_analysis(tmp_path):
    """Five seeds of the residual at the capacity below-threshold weight beside the residual and the
    light verdict at the trained weight, all at the capacity task's hard level. analyze.py names the
    new arm graded_below_cap, and the below table lists it under regime H with the capbt prefix
    against the committed capacity manifest."""
    root = tmp_path / "runs"
    forced = {1, 3}
    for arm, w in (("graded", A.GRADED_BELOW_CAP_W), ("graded", 5.0), ("binary", 5.0)):
        for seed in range(5):
            write_run(root / f"{arm}_{w}_s{seed}", task="capacity", arm=arm, w=w, vmax=0.03,
                      regime="H", seed=seed, groups=6,
                      final_feasible=(w == A.GRADED_BELOW_CAP_W and seed in forced))
    an = tmp_path / "an"
    proc = _analyze(root, an)
    assert proc.returncode == 0, proc.stderr[-2000:]
    labels = json.loads((an / "arm_labels.json").read_text())
    assert labels["graded_below_cap"] == A.NAMED_LABELS["graded_below_cap"]
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path / "t"), "--table", "below",
                        "--macro-prefix", "capbt", "--macros-file", "capbt_macros.tex", "--long"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    per_run = list(csv.DictReader((an / "final_per_run.csv").open()))
    below = sorted((x for x in per_run if x["arm"] == "graded_below_cap"), key=lambda x: int(x["seed"]))
    assert [int(x["seed"]) for x in below] == [0, 1, 2, 3, 4]
    feasible = {int(x["seed"]) for x in below if float(x["eval_violation_rate_final"]) == 0.0}
    assert forced <= feasible
    macros = (tmp_path / "t" / "capbt_macros.tex").read_text()
    assert f"\\newcommand{{\\capbtFeasHGradedBelowCap}}{{{len(feasible)}}}" in macros
    assert "\\newcommand{\\capbtSeedCountHGradedBelowCap}{5}" in macros
    want = ", ".join(f"{float(x['eval_parse_fail_rate_final']):.2f}" for x in below)
    assert f"\\newcommand{{\\capbtUnparseableHGradedBelowCapSeeds}}{{{want}}}" in macros
    assert "\\newcommand{\\capbtSeedCountHGraded}{5}" in macros
    assert "\\newcommand{\\capbtSeedCountHBinary}{5}" in macros
    body = (tmp_path / "t" / "below_regimeH.tex").read_text()
    assert labels["graded_below_cap"] in body
    rows = [line for line in body.splitlines() if line.endswith("\\\\") and "textbf" not in line]
    assert len(rows) == 15
    assert not (tmp_path / "t" / "below_regimeA.tex").exists()
    names = re.findall(r"\\newcommand\{\\(\w+)\}", macros)
    assert len(names) == len(set(names)) and all(n.isalpha() for n in names)


@requires_paper
def test_the_capacity_remedy_macros_count_empty_answers_and_the_literal_anchor(tmp_path):
    root = tmp_path / "runs"
    for name, kw in (("binary", {}), ("none", {"arm": "none", "w": 0.0}),
                     ("binary_drgrpo", {"adv_variant": "drgrpo"}),
                     ("binary_vfilter", {"group_filter": "verdict_mixed"})):
        for seed in (0, 1):
            write_run(root / f"{name}_s{seed}", task="capacity", vmax=0.03, regime="H", seed=seed,
                      steps=2, groups=6, final_feasible=(name in ("binary_drgrpo", "binary") and seed == 1),
                      **kw)
    an = tmp_path / "an"
    proc = _analyze(root, an)
    assert proc.returncode == 0, proc.stderr[-2000:]
    answers = {(r["arm"], int(r["seed"])): r for r in csv.DictReader((an / "final_answers.csv").open())}
    assert answers[("binary_drgrpo", 1)]["eval_empty_selections_final"] == "24"
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path / "t"), "--table", "remedy",
                        "--macro-prefix", "caprem", "--macros-file", "caprem_macros.tex", "--long"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    macros = (tmp_path / "t" / "caprem_macros.tex").read_text()
    want = ", ".join(answers[("binary_drgrpo", s)]["eval_empty_selections_final"] for s in (0, 1))
    assert f"\\newcommand{{\\capremEmptyHBinaryDrGrpoSeeds}}{{{want}}}" in macros
    assert "\\newcommand{\\capremEvalAnswers}{24}" in macros
    counters = {(c["arm"], int(c["seed"])): c
                for c in json.loads((an / "checks.json").read_text())["group_counters"]}
    mixed = ", ".join(f'{counters[("binary_drgrpo", s)]["mixed_groups"]:,}' for s in (0, 1))
    assert f"\\newcommand{{\\capremMixedHBinaryDrGrpoSeeds}}{{{mixed}}}" in macros
    per_run = {(r["arm"], int(r["seed"])): r for r in csv.DictReader((an / "final_per_run.csv").open())}
    assert float(per_run[("binary_drgrpo", 1)]["eval_task_scalar_final"]) == 0.0
    assert "\\capremEvalScalarHBinaryDrGrpoSeeds}" in macros
    lit = json.loads((an / "checks.json").read_text())["variant_algebra"]["literal_anchor_by_instance"]
    cells = lit["H"]["1"].values()
    total = f'{sum(c["cancels"] for c in cells):,}'.replace(",", "{,}")
    assert f"\\newcommand{{\\capremAlgHLitTaskOneCancels}}{{{total}}}" in macros
    anchors = ", ".join(f'{c["anchor_verdict"]:.2f}' for c in cells)
    assert f"\\newcommand{{\\capremAlgHLitTaskOneAnchorVerdict}}{{{anchors}}}" in macros
    # Counts restricted to the seeds and runs that reached feasibility.
    feas = [k for k, r in per_run.items() if r["eval_violation_rate_final"] not in ("", None)
            and float(r["eval_violation_rate_final"]) == 0.0]
    assert f"\\newcommand{{\\capremRunsFeasH}}{{{len(feas)}}}" in macros
    all_empty = sum(1 for k in feas
                    if answers[k]["eval_empty_selections_final"] == answers[k]["eval_n_final"])
    assert f"\\newcommand{{\\capremRunsFeasAllEmptyH}}{{{all_empty}}}" in macros
    dr_feas = [s for s in (0, 1) if ("binary_drgrpo", s) in feas]
    want = (", ".join(answers[("binary_drgrpo", s)]["eval_empty_selections_final"] for s in dr_feas)
            if dr_feas else "\\mbox{--}")
    assert f"\\newcommand{{\\capremEmptyFeasHBinaryDrGrpoSeeds}}{{{want}}}" in macros
    assert "\\newcommand{\\capremAlgTol}{10^{-9}}" in macros
    assert "\\newcommand{\\capremAlgWFive}{25}" in macros
    one = json.loads((an / "checks.json").read_text())["variant_algebra"]["by_regime"]["H"]["by_task_weight"]["1"]
    total_one = f'{one["all_violating_groups"]:,}'.replace(",", "{,}")
    assert f"\\newcommand{{\\capremAlgHTaskOneAllViol}}{{{total_one}}}" in macros
    # Runs that reached zero violation at an evaluation and ended infeasible, counted per arm.
    timing = {(r["arm"], int(r["seed"])): r for r in csv.DictReader((an / "remedy_timing.csv").open())}
    for arm, word in (("binary", "Binary"), ("none", "None"), ("binary_drgrpo", "BinaryDrGrpo")):
        left = sum(1 for s in (0, 1)
                   if timing[(arm, s)]["arrival_step"] not in ("", None) and (arm, s) not in feas)
        assert f"\\newcommand{{\\capremLeftFeasH{word}}}{{{left}}}" in macros
    # The all-violating gap counts only the arms trained without a remedy.
    plain = [(x, s) for x in ("binary", "none") for s in (0, 1)]
    succ = [int(counters[k]["all_violating"]) for k in plain if k in feas]
    fail = [int(counters[k]["all_violating"]) for k in plain if k not in feas]
    assert succ and fail
    gap = min(fail) // max(1, max(succ))
    assert f"\\newcommand{{\\capremAllViolGapNoRemedyH}}{{{gap}}}" in macros


@requires_paper
def test_the_remedy_table_refuses_an_analysis_without_final_answers(remedy_analysis, tmp_path):
    an = tmp_path / "an"
    shutil.copytree(remedy_analysis["analysis"], an)
    (an / "final_answers.csv").unlink()
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path / "t"), "--table", "remedy",
                        "--macro-prefix", "rem", "--macros-file", "m.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 3 and "final_answers.csv is missing" in r.stderr


@requires_paper
def test_the_remedy_table_refuses_runs_evaluated_on_different_numbers_of_answers(remedy_analysis,
                                                                                 tmp_path):
    an = tmp_path / "an"
    shutil.copytree(remedy_analysis["analysis"], an)
    rows = list(csv.DictReader((an / "final_answers.csv").open()))
    rows[0]["eval_n_final"] = str(int(rows[0]["eval_n_final"]) + 1)
    with (an / "final_answers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_remedy_tables.py"),
                        "--analysis", str(an), "--out", str(tmp_path / "t"), "--table", "remedy",
                        "--macro-prefix", "rem", "--macros-file", "m.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 3 and "different numbers of answers" in r.stderr


# ---------------------------------------------------------------- frozen capacity rows


def _csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_the_snapshot_holds_the_36_capacity_rows():
    if not os.path.exists(SNAPSHOT):
        pytest.skip("the capacity snapshot is not present in this copy of the code")
    rows = _csv_rows(SNAPSHOT)
    assert len(rows) == 36
    c = Counter((r["regime"], r["arm"]) for r in rows)
    assert set(c.values()) == {3} and {k[0] for k in c} == {"H", "E"}
    assert {k[1] for k in c} == set(A.CAPACITY_ARM_SPECS)
    assert {int(r["seed"]) for r in rows} == {0, 1, 2}


# The compute ceiling cut seeds 3 and 4 of two easy-level arms before block C2 was submitted
# (LEDGER.md, 2026-09-15), so the five-seed set holds four fewer runs than twelve arms times five.
CAPACITY5_CUT = {("E", "verdict_only"), ("E", "graded_no_task")}


@pytest.mark.parametrize("directory,seeds,cut", [("572991", 3, set()), ("capacity5", 5, CAPACITY5_CUT)])
def test_a_capacity_analysis_reproduces_the_snapshot_for_seeds_0_to_2(directory, seeds, cut):
    path = os.path.join(HERE, "runs", directory, "analysis", "final_per_run.csv")
    if not os.path.exists(SNAPSHOT) or not os.path.exists(path):
        pytest.skip(f"no analysis of runs/{directory} in this checkout")
    got = _csv_rows(path)
    assert {int(r["seed"]) for r in got} == set(range(seeds))
    assert len(got) == 12 * seeds - len(cut) * (seeds - 3)
    later = {(r["regime"], r["arm"]) for r in got if int(r["seed"]) > 2}
    if seeds > 3:
        all_arms = {(reg, arm) for reg in ("H", "E") for arm in A.CAPACITY_ARM_SPECS}
        assert all_arms - later == cut
    key = lambda r: (r["regime"], r["arm"], int(r["seed"]))  # noqa: E731
    first = sorted((r for r in got if int(r["seed"]) <= 2), key=key)
    assert first == sorted(_csv_rows(SNAPSHOT), key=key), \
        "seeds 0-2 of the capacity table moved when more seeds were added"


# ---------------------------------------------------------------- yoked filter controls


def _yoked(d, schedule, sid, schedules_path, **kw):
    """Write a synthetic yoked control. It starts from a plain log and keeps exactly the scheduled
    prompts, as the trainer does, and it writes the schedule entry the analysis will read."""
    write_run(d, **kw)
    cfg = json.loads((d / "config.json").read_text())
    cfg["keep_schedule"] = sid
    (d / "config.json").write_text(json.dumps(cfg))
    lines = []
    for line in (d / "metrics.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["phase"] == "train":
            want = schedule[rec["step"] - 1]
            for g in rec["groups"]:
                g["kept"] = g["prompt_index"] in want
            rec["n_kept_groups"] = len(want)
            rec["opt_step_taken"] = bool(want)
            rec["keep_schedule_n"] = len(want)
            if not want:
                rec["grad_norm_preclip"] = rec["pg_loss"] = None
        lines.append(json.dumps(rec))
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    doc = json.loads(schedules_path.read_text()) if schedules_path.exists() else {"schedules": {}}
    doc["schedules"][sid] = {"kept_sets": [sorted(s) for s in schedule]}
    schedules_path.write_text(json.dumps(doc))
    return d


@pytest.mark.parametrize("cfg,name", [
    ({"arm": "binary", "w": 5.0, "task": "speed", "seed": 0, "keep_schedule": "speed_w5_s0_d3"},
     "binary_yoked_d3"),
    ({"arm": "binary", "w": A.BINARY_ABOVE_W, "task": "speed", "seed": 1,
      "keep_schedule": "speed_w25_s1_d5"}, "binary_5x_yoked_d5"),
    ({"arm": "binary", "w": 5.0, "task": "capacity", "seed": 2, "keep_schedule": "capacity_w5_s2_d0"},
     "binary_yoked_d0"),
    ({"arm": "binary", "w": 5.0, "task": "speed", "seed": 0, "keep_schedule": None}, "binary"),
    ({"arm": "binary", "w": 5.0, "task": "speed", "seed": 0}, "binary"),
])
def test_a_yoked_control_is_named_by_its_draw_and_a_default_run_keeps_its_name(cfg, name):
    assert A.arm_name(cfg) == name


@pytest.mark.parametrize("sid", ["speed_w5_s0_d", "speed_w5_s0_d03", "speed_w05_s0_d1", "fast_w5_s0_d1",
                                 "speed_w5_s0_d1 ", "speed_w25_s0_d1", "speed_w5_s1_d1",
                                 "capacity_w5_s0_d1", 7])
def test_a_malformed_or_foreign_keep_schedule_id_is_refused(sid):
    with pytest.raises(ValueError):
        A.arm_name({"arm": "binary", "w": 5.0, "task": "speed", "seed": 0, "keep_schedule": sid})


def test_the_yoked_label_names_its_design_and_its_draw():
    cfg = {"arm": "binary", "w": A.BINARY_ABOVE_W, "task": "speed", "seed": 1,
           "keep_schedule": "speed_w25_s1_d5"}
    assert A.compose_arm_label(cfg) == A.NAMED_LABELS["binary_5x"] + ", yoked control, draw 5"
    cfg = {"arm": "binary", "w": 5.0, "task": "capacity", "seed": 0, "keep_schedule": "capacity_w5_s0_d1"}
    assert A.compose_arm_label(cfg) == A.ARM_STYLE["binary"]["label"] + ", yoked control, draw 1"


def test_the_keep_schedule_may_differ_inside_one_comparison_without_changing_the_paper_count():
    """The paper counts ALLOWED_CONFIG_DIFF and names its ten fields, so the yoked field has its own set."""
    assert A.YOKED_CONFIG_DIFF == {"keep_schedule"}
    assert "keep_schedule" not in A.ALLOWED_CONFIG_DIFF and len(A.ALLOWED_CONFIG_DIFF) == 10
    assert A.LEGACY_CONFIG_DEFAULTS["keep_schedule"] is None
    plain = {"dir": type("D", (), {"name": "p"})(), "cfg": {"arm": "binary", "w": 5.0, "steps": 3,
                                                           "keep_schedule": None},
             "prov": {"steps_completed": 3}}
    yoked = {"dir": type("D", (), {"name": "y"})(), "cfg": dict(plain["cfg"], keep_schedule="speed_w5_s0_d0"),
             "prov": {"steps_completed": 3}}
    assert A.matched_budget_check([plain, yoked], notes=[]) == []
    other = {"dir": type("D", (), {"name": "o"})(), "cfg": dict(plain["cfg"], lr=1e-4),
             "prov": {"steps_completed": 3}}
    assert A.matched_budget_check([plain, other], notes=[]), "an unrelated field must still refuse"


def test_a_scheduled_log_is_self_consistent_and_a_flipped_kept_flag_is_caught(tmp_path, monkeypatch):
    path = tmp_path / "keep_schedules.json"
    monkeypatch.setattr(A, "KEEP_SCHEDULES_PATH", path)
    schedule = [[4], [], [0, 2, 5], []]
    d = _yoked(tmp_path / "c", schedule, "speed_w5_s0_d2", path, steps=4, groups=6)
    run = A._load_run(d)
    stats, worst = A.counterfactual_checks([run])
    assert worst["self"] == 0.0
    assert not {"kept_mismatch", "keep_schedule_n_mismatch"} & set(stats)
    g = run["train"][1]["groups"][3]
    g["kept"] = not g["kept"]
    stats, _ = A.counterfactual_checks([run])
    assert stats["kept_mismatch"] == 1
    run["train"][1]["groups"][3]["kept"] = False
    run["train"][2]["keep_schedule_n"] = 2
    stats, _ = A.counterfactual_checks([run])
    assert stats["keep_schedule_n_mismatch"] == 1 and "kept_mismatch" not in stats


def test_a_control_without_its_schedule_entry_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "keep_schedules.json"
    monkeypatch.setattr(A, "KEEP_SCHEDULES_PATH", path)
    d = _yoked(tmp_path / "c", [[1], [], []], "speed_w5_s0_d1", path, steps=3, groups=6)
    path.write_text(json.dumps({"schedules": {}}))
    with pytest.raises(ValueError):
        A.counterfactual_checks([A._load_run(d)])


def test_the_analysis_admits_a_yoked_control_beside_its_plain_arm(tmp_path, monkeypatch):
    """This test runs the end-to-end path, which assembles the runs and then runs analyze.py in a
    subprocess. The subprocess reads the committed schedule file, so the synthetic control uses one
    of its ids."""
    committed = json.loads(A.KEEP_SCHEDULES_PATH.read_text())["schedules"]
    sid = "speed_w5_s1_d3"
    schedule = committed[sid]["kept_sets"][:3]
    scratch = tmp_path / "scratch.json"
    base, new = tmp_path / "base", tmp_path / "new"
    for seed in (0, 1):
        write_run(base / f"binary_s{seed}", seed=seed, steps=3, groups=6)
    _yoked(new / "binary_yoked_d3_s1", schedule, sid, scratch, seed=1, steps=3, groups=6)
    # The subprocess reads the committed file, whose entry lists 300 steps. Only the first three
    # are logged here, and the check reads the entry by step index.
    combined = tmp_path / "combined"
    assert assemble_runs.main(["--out", str(combined), "--src", str(base), "--src", str(new)]) == 0
    r = _analyze(combined, tmp_path / "an")
    assert r.returncode == 0, r.stderr[-2000:]
    checks = json.loads((tmp_path / "an" / "checks.json").read_text())
    assert checks["matched_budget_problems"] == []
    assert checks["trainer_self_consistent"] is True
    labels = json.loads((tmp_path / "an" / "arm_labels.json").read_text())
    assert labels["binary_yoked_d3"] == A.ARM_STYLE["binary"]["label"] + ", yoked control, draw 3"


def test_a_wrong_logged_schedule_size_fails_the_trainer_self_consistency(tmp_path):
    """A control whose logged keep_schedule_n leaves its schedule at one step must fail the trainer
    self-consistency check of analyze.py, even though every kept flag is right."""
    committed = json.loads(A.KEEP_SCHEDULES_PATH.read_text())["schedules"]
    sid = "speed_w5_s1_d3"
    schedule = committed[sid]["kept_sets"][:3]
    base, new = tmp_path / "base", tmp_path / "new"
    for seed in (0, 1):
        write_run(base / f"binary_s{seed}", seed=seed, steps=3, groups=6)
    d = _yoked(new / "binary_yoked_d3_s1", schedule, sid, tmp_path / "scratch.json", seed=1, steps=3,
               groups=6)
    lines = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines()]
    for rec in lines:
        if rec["phase"] == "train" and rec["step"] == 2:
            rec["keep_schedule_n"] += 1
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    combined = tmp_path / "combined"
    assert assemble_runs.main(["--out", str(combined), "--src", str(base), "--src", str(new)]) == 0
    r = _analyze(combined, tmp_path / "an")
    checks = json.loads((tmp_path / "an" / "checks.json").read_text())
    assert checks["counterfactual"].get("keep_schedule_n_mismatch") == 1, r.stderr[-2000:]
    assert checks["trainer_self_consistent"] is False
