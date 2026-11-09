"""These tests cover compare_reruns.py, which pairs the single-commit capacity reruns with their originals.

The script guards a fixed reading order. Its file check prints no outcome, its gate reads only the
logged training groups, and its full mode reports labels only after the gate holds. Most tests build
their fixture runs in memory or under tmp_path, and none of them opens a network connection. Three
tests read run directories of this repository, and they skip when those directories are absent, as
they are in the anonymous supplement. One test reads the tracked capacity manifest, and one scans
the two new source files with the supplement's banned-term check.
"""
import csv
import itertools
import json
import os
import random
import socket
import sys
from fractions import Fraction

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402
import compare_reruns as C  # noqa: E402
import tasks  # noqa: E402

RERUN_SHA = "COMMIT_4"
DIGEST = "d5ea80637093bf77d98730fcbfe33043c57a6dbb987fbc253d2cb989037d1751"
SPEC = tasks.get("capacity")
EMPTY = [0]
# Item 0 of instance 0 fits the capacity at both levels, so this selection is feasible and not empty.
ONE_ITEM = [0, 0]
EVERY_ITEM = [0] + list(range(16))
VMAX = {"H": 0.03, "E": 0.20}
# The published cohort is found through this variable, so the test holds no absolute path.
COHORT = os.environ.get("WOH_REFERENCE_COHORT", os.path.join(HERE, "runs", "capacity5"))
SINGLE = os.path.join(HERE, "runs", "capacity5_single")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a compare_reruns test tried to open a network connection")
    monkeypatch.setattr(socket.socket, "connect", refuse)


# ---------------------------------------------------------------- fixture runs


def _group(sels, vmax):
    """The rewards and advantages of a fixture group depend only on its answers, as the trainer's do."""
    rewards = [-100.0 if s is None else -float(len(s)) for s in sels]
    mean = sum(rewards) / len(rewards)
    sd = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    violating = [not SPEC.verdict(s, vmax) for s in sels]
    return {"selections": sels, "rewards": rewards, "adv": [(r - mean) / (sd + 1e-6) for r in rewards],
            "all_violating": all(violating), "has_violator": any(violating)}


def base_answers(steps, salt=0):
    """Every step holds six groups of four selections, and a different salt changes every selection."""
    return [[[[g % 6, (t * 7 + g * 3 + i + salt) % 16] for i in range(4)] for g in range(6)]
            for t in range(1, steps + 1)]


def _eval(step, sels, vmax):
    n = len(sels)
    return {"step": step, "phase": "eval", "eval_n": n,
            "eval_violation_rate": sum(not SPEC.verdict(s, vmax) for s in sels) / n,
            "eval_parse_fail_rate": sum(s is None for s in sels) / n, "eval_selections": sels}


def make_run(name="run", *, regime="H", arm_name="binary", seed=0, sha=RERUN_SHA, digest=DIGEST,
             steps=5, eval_every=5, answers=None, final=None, checksum="aa", cfg=None, prov=None,
             parent="array"):
    """This function builds one run in memory, with the fields that analyze._load_run returns."""
    arm, w, w_task = A.CAPACITY_ARM_SPECS[arm_name]
    vmax = VMAX[regime]
    config = dict(C.EXPECTED_CONFIG, arm=arm, w=w, w_task=w_task, regime=regime, vmax=vmax,
                  seed=seed, steps=steps, eval_every=eval_every, out_dir=f"out/{name}")
    config.update(cfg or {})
    provenance = dict(C.EXPECTED_ENVIRONMENT, git_sha=sha, git_dirty=True, steps_completed=steps,
                      final_param_checksum=checksum, slurm_job_id="1", hostname="node",
                      config_hash="h", wall_seconds=1.0)
    if digest is not None:
        provenance["training_sources_digest"] = digest
    provenance.update(prov or {})
    answers = answers if answers is not None else base_answers(steps)
    final = final if final is not None else [EMPTY] * 4
    train = [{"step": t, "phase": "train", "groups": [_group(g, vmax) for g in answers[t - 1]]}
             for t in range(1, steps + 1)]
    evals = [_eval(s, final if s == steps else [ONE_ITEM] * 4, vmax)
             for s in range(0, steps + 1, eval_every)]

    class _Dir:
        pass
    d = _Dir()
    d.name = name
    d.parent = _Dir()
    d.parent.name = parent
    return {"dir": d, "cfg": config, "prov": provenance, "train": train, "evals": evals}


def write_run(root, name, done=True, **kw):
    """This function writes one fixture run to disk in the layout the trainer writes."""
    run = make_run(name, **kw)
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(run["cfg"]))
    (d / "provenance.json").write_text(json.dumps(run["prov"]))
    with open(d / "metrics.jsonl", "w") as f:
        for rec in sorted(run["train"] + run["evals"], key=lambda r: (r["step"], r["phase"] != "eval")):
            f.write(json.dumps(rec) + "\n")
    if done:
        (d / "DONE").write_text("ok\n")
    return d


def _pair_dirs(tmp_path, cells=(("H", "binary"),), seeds=(0, 1, 2), rerun_kw=None, orig_kw=None):
    """This function writes originals and reruns for the given cells and returns the two roots."""
    orig, rer = tmp_path / "orig", tmp_path / "rerun"
    for n, ((regime, arm), seed) in enumerate(itertools.product(cells, seeds), start=1):
        name = f"{n}_{regime}_{arm}_s{seed}"
        write_run(orig, name, regime=regime, arm_name=arm, seed=seed, sha=C.ORIGINAL_SHA[regime],
                  digest=None, **(orig_kw or {}))
        write_run(rer, name, regime=regime, arm_name=arm, seed=seed, **(rerun_kw or {}))
    return orig, rer


def _gate_args(orig, rer, *extra):
    return ["--gate-only", "--original", str(orig), "--rerun", str(rer), "--expect-sha", RERUN_SHA,
            "--expect-digest", DIGEST, *extra]


# ---------------------------------------------------------------- trajectories and gate SC0


def test_an_identical_pair_has_step_one_identity_no_break_no_divergence_and_equal_checksums():
    a, b = make_run("a"), make_run("b")
    t = C.trajectory(a, b)
    assert t == {"step1_identical": True, "logic_break_step": None,
                 "first_answer_difference_step": None, "identical_answer_steps": 5,
                 "checksums_equal": True}
    assert C.gate([(("capacity", "H", "binary", 0), a, b)])["sc0_holds"] is True


@pytest.mark.parametrize("k", [2, 3, 5])
def test_a_pair_that_diverges_at_step_k_reports_k_and_k_minus_one_identical_steps(k):
    steps = 5
    orig = base_answers(steps)
    other = base_answers(steps, salt=1)
    rerun = orig[:k - 1] + other[k - 1:]
    t = C.trajectory(make_run("a", answers=orig), make_run("b", answers=rerun, checksum="bb"))
    assert t["first_answer_difference_step"] == k
    assert t["identical_answer_steps"] == k - 1
    assert t["step1_identical"] is True and t["logic_break_step"] is None
    assert t["checksums_equal"] is False


def test_one_reward_changed_under_identical_answers_is_a_logic_break_and_fails_the_gate(tmp_path, capsys):
    orig, rer = _pair_dirs(tmp_path, seeds=(0,))
    path = rer / "1_H_binary_s0" / "metrics.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    for rec in recs:
        if rec["phase"] == "train" and rec["step"] == 3:
            rec["groups"][2]["rewards"][1] += 0.5
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    run_a = A._load_run(orig / "1_H_binary_s0")
    run_b = A._load_run(rer / "1_H_binary_s0")
    assert C.trajectory(run_a, run_b)["logic_break_step"] == 3
    assert C.main(_gate_args(orig, rer)) == 1
    assert "logic_break_step=3" in capsys.readouterr().out


def test_a_pair_that_differs_at_step_one_fails_the_gate(tmp_path):
    steps = 5
    orig, rer = tmp_path / "orig", tmp_path / "rerun"
    write_run(orig, "1_H_binary_s0", sha=C.ORIGINAL_SHA["H"], digest=None)
    write_run(rer, "1_H_binary_s0", answers=base_answers(steps, salt=3))
    assert C.main(_gate_args(orig, rer)) == 1


def test_a_step_one_advantage_difference_fails_the_gate_even_with_identical_answers():
    a = make_run("a")
    b = make_run("b")
    b["train"][0]["groups"][0]["adv"] = [x + 0.5 for x in b["train"][0]["groups"][0]["adv"]]
    result = C.gate([(("capacity", "H", "binary", 0), a, b)])
    assert result["sc0_holds"] is False and result["failing_pairs"][0]["step1_identical"] is False


def test_a_step_one_reward_difference_under_identical_answers_is_not_step_one_identity():
    a = make_run("a")
    b = make_run("b")
    b["train"][0]["groups"][0]["rewards"] = [x + 0.5 for x in b["train"][0]["groups"][0]["rewards"]]
    t = C.trajectory(a, b)
    assert t["step1_identical"] is False and t["logic_break_step"] == 1


def test_a_step_with_a_different_number_of_groups_is_not_an_identical_step():
    a = make_run("a")
    b = make_run("b")
    b["train"][2]["groups"] = b["train"][2]["groups"][:5]
    t = C.trajectory(a, b)
    assert t["first_answer_difference_step"] == 3 and t["identical_answer_steps"] == 4


def test_a_missing_checksum_never_counts_as_equal():
    t = C.trajectory(make_run("a", checksum=None), make_run("b", checksum=None))
    assert t["checksums_equal"] is False


def test_no_finished_pair_never_passes_the_gate():
    assert C.gate([])["sc0_holds"] is False


def test_a_seeded_loop_of_random_pairs_finds_no_logic_break_and_the_right_divergence_step():
    """The rewards and advantages of every fixture group are a function of its answers, so no pair can
    log a logic break. Each reported divergence step must equal the first step with a differing answer."""
    rng = random.Random(20260922)

    def selection():
        return [rng.randrange(6)] + sorted(rng.sample(range(16), rng.randrange(0, 4)))

    for _ in range(200):
        steps = rng.randint(1, 8)
        k = rng.randint(1, steps + 1)
        orig = [[[selection() for _ in range(4)] for _ in range(6)] for _ in range(steps)]
        rerun = [[[list(s) for s in g] for g in step] for step in orig]
        for t in range(k - 1, steps):
            rerun[t] = [[selection() for _ in range(4)] for _ in range(6)]
        if k <= steps and rerun[k - 1] == orig[k - 1]:
            # The random draw repeated the original step, so one instance index moves to force a difference.
            g, i = rng.randrange(6), rng.randrange(4)
            sel = rerun[k - 1][g][i]
            rerun[k - 1][g][i] = [(sel[0] + 1) % 6] + sel[1:]
        same = [a == b for a, b in zip(orig, rerun)]
        want_first = next((t + 1 for t, s in enumerate(same) if not s), None)
        t = C.trajectory(make_run("a", steps=steps, eval_every=steps, answers=orig),
                         make_run("b", steps=steps, eval_every=steps, answers=rerun))
        assert t["logic_break_step"] is None
        assert t["first_answer_difference_step"] == want_first
        assert t["first_answer_difference_step"] == (k if k <= steps else None)
        assert t["identical_answer_steps"] == sum(same)


# ---------------------------------------------------------------- labels


@pytest.mark.parametrize("final,want", [
    ([EMPTY] * 4, "FE"),
    ([EMPTY, ONE_ITEM, EMPTY, EMPTY], "FI"),
    ([None] * 4, "U"),
    ([EMPTY, EVERY_ITEM, EMPTY, EMPTY], "V"),
    # An unparseable completion counts as a violation, so a run with one is neither FE nor feasible.
    ([EMPTY, None, EMPTY, EMPTY], "V"),
])
def test_final_evaluations_give_the_four_outcome_classes(final, want):
    labels = C.run_labels(make_run(final=final))
    assert labels["outcome_class"] == want
    assert labels["feasibility"] is (want in ("FE", "FI"))
    assert labels["unparseable"] is (want == "U")


def test_a_run_without_an_evaluation_at_its_last_step_is_refused():
    run = make_run()
    run["evals"] = run["evals"][:-1]
    with pytest.raises(C.Refusal):
        C.run_labels(run)


# ---------------------------------------------------------------- reference values and bands


def _toy_cohort():
    """Cell H binary has feasibility labels T, T, F, T, T, and cell H graded has three FE seeds."""
    feasible = {0: True, 1: True, 2: False, 3: True, 4: True}
    cohort = {}
    for s, ok in feasible.items():
        cohort[("capacity", "H", "binary", s)] = make_run(
            f"b{s}", arm_name="binary", seed=s, steps=2, eval_every=2,
            final=[EMPTY] * 4 if ok else [EVERY_ITEM] * 4)
    for s in (0, 1, 2):
        cohort[("capacity", "H", "graded", s)] = make_run(f"g{s}", arm_name="graded", seed=s,
                                                          steps=2, eval_every=2)
    return cohort


def test_the_reference_values_on_a_toy_cohort_equal_hand_computed_values():
    cohort = _toy_cohort()
    originals = {i: r for i, r in cohort.items() if i[3] in (0, 1, 2)}
    ref = C.reference_values(originals, cohort)
    ps = [ref["per_pair"][i]["feasibility"] for i in sorted(ref["per_pair"])]
    # Sorted identities put the binary cell first. Seeds 0 and 1 match three of their four other
    # seeds, seed 2 matches none of its four, and every graded seed matches both of its others.
    assert ps == [Fraction(3, 4), Fraction(3, 4), Fraction(0), Fraction(1), Fraction(1), Fraction(1)]
    feas = ref["labels"]["feasibility"]
    assert feas["expected"] == 4.5
    assert feas["sd"] == pytest.approx((6 / 16) ** 0.5)
    assert feas["pairs_at_probability_one"] == 3
    # The distribution is 1/16, 6/16 and 9/16 on 3, 4 and 5 agreements, so the range is 3 to 5.
    assert (feas["range_low"], feas["range_high"]) == (3, 5)
    assert C.poisson_binomial(ps)[3:] == [Fraction(1, 16), Fraction(6, 16), Fraction(9, 16), 0]


def test_the_exact_distribution_equals_brute_force_enumeration():
    rng = random.Random(7)
    for _ in range(30):
        ps = [Fraction(rng.randrange(0, 5), 4) for _ in range(rng.randint(1, 7))]
        want = [Fraction(0)] * (len(ps) + 1)
        for outcome in itertools.product((0, 1), repeat=len(ps)):
            prob = Fraction(1)
            for p, o in zip(ps, outcome):
                prob *= p if o else 1 - p
            want[sum(outcome)] += prob
        assert C.poisson_binomial(ps) == want


def test_a_count_whose_cumulative_probability_equals_a_bound_is_inside_the_range():
    # One draw at 39/40 puts exactly 1/40 on zero agreements, so the rule "at least 0.025" starts
    # the range at zero. One draw at 1/40 puts exactly 39/40 on zero, so the range also ends there.
    assert C.central_range([Fraction(39, 40)]) == (0, 1)
    assert C.central_range([Fraction(1, 40)]) == (0, 0)


def test_the_bands_follow_the_range_and_band_r_takes_precedence():
    assert [C.band(a, 36, 22, 29) for a in (36, 35, 30, 29, 22, 21, 0)] == \
        ["R", "M", "M", "X", "X", "B", "B"]
    # With few pairs the range can reach the number of finished pairs, and band R then takes precedence.
    assert C.band(3, 3, 1, 3) == "R"
    assert C.band(2, 3, 1, 3) == "X"


def test_an_arm_with_no_other_seed_in_the_cohort_is_refused():
    run = make_run("only", arm_name="none", seed=0, steps=2, eval_every=2)
    cohort = {("capacity", "H", "none", 0): run}
    with pytest.raises(C.Refusal):
        C.reference_values(cohort, cohort)


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize("rerun_kw,needle", [
    ({"sha": "0" * 40}, "git_sha"),
    ({"digest": "f" * 64}, "training_sources_digest"),
    ({"cfg": {"max_new_tokens": 64}}, "config.max_new_tokens"),
    ({"prov": {"torch": "2.4.0"}}, "provenance.torch"),
    ({"prov": {"steps_completed": 4}}, "steps_completed"),
])
def test_each_refusal_fires_on_its_own_fixture(tmp_path, capsys, rerun_kw, needle):
    orig, rer = _pair_dirs(tmp_path, seeds=(0,), rerun_kw=rerun_kw)
    out = tmp_path / "gate"
    assert C.main(_gate_args(orig, rer, "--out", str(out))) == 2
    assert needle in capsys.readouterr().err
    assert not out.exists(), "a refusal must write nothing"


def test_a_rerun_without_done_is_refused(tmp_path, capsys):
    orig, rer = _pair_dirs(tmp_path, seeds=(0,))
    (rer / "1_H_binary_s0" / "DONE").unlink()
    assert C.main(_gate_args(orig, rer)) == 2
    assert "no DONE marker" in capsys.readouterr().err


def test_a_missing_rerun_is_refused_unless_a_not_run_entry_names_it(tmp_path, capsys):
    orig, rer = _pair_dirs(tmp_path, seeds=(0, 1))
    write_run(orig, "9_H_binary_s2", seed=2, sha=C.ORIGINAL_SHA["H"], digest=None)
    assert C.main(_gate_args(orig, rer)) == 2
    assert "H:binary:2" in capsys.readouterr().err
    out = tmp_path / "gate"
    assert C.main(_gate_args(orig, rer, "--not-run", "H:binary:2", "--out", str(out))) == 0
    record = json.loads((out / "gate.json").read_text())
    assert record["not_run"] == ["H:binary:2"] and record["pairs"] == 2


def test_a_not_run_entry_that_names_a_finished_rerun_or_no_original_is_refused(tmp_path):
    orig, rer = _pair_dirs(tmp_path, seeds=(0,))
    assert C.main(_gate_args(orig, rer, "--not-run", "H:binary:0")) == 2
    assert C.main(_gate_args(orig, rer, "--not-run", "E:binary:0")) == 2
    assert C.main(_gate_args(orig, rer, "--not-run", "H-binary-0")) == 2
    assert C.main(_gate_args(orig, rer, "--not-run", "H:binary:x")) == 2


def test_two_reruns_with_one_identity_are_refused(tmp_path, capsys):
    orig, rer = _pair_dirs(tmp_path, seeds=(0,))
    write_run(rer, "7_H_binary_s0", seed=0)
    assert C.main(_gate_args(orig, rer)) == 2
    assert "share the identity" in capsys.readouterr().err


def test_originals_at_their_level_commit_pass_and_a_third_commit_is_refused(tmp_path, capsys):
    orig, rer = _pair_dirs(tmp_path, cells=(("H", "binary"), ("E", "binary")), seeds=(0,))
    assert C.main(_gate_args(orig, rer)) == 0
    capsys.readouterr()
    third = tmp_path / "third"
    write_run(third, "1_H_binary_s0", sha="COMMIT_5" + "0" * 32, digest=None)
    assert C.main(_gate_args(third, rer)) == 2
    assert "is not the commit of level H" in capsys.readouterr().err


def test_an_easy_level_original_at_the_hard_level_commit_is_refused(tmp_path):
    orig, rer = tmp_path / "orig", tmp_path / "rerun"
    write_run(orig, "22_E_binary_s0", regime="E", sha=C.ORIGINAL_SHA["H"], digest=None)
    write_run(rer, "22_E_binary_s0", regime="E")
    assert C.main(_gate_args(orig, rer)) == 2


def test_a_pattern_leaves_out_a_failed_attempt_and_the_bare_directory_is_refused(tmp_path):
    orig, rer = _pair_dirs(tmp_path, seeds=(0, 1))
    retry = tmp_path / "retry"
    (rer / "2_H_binary_s1" / "DONE").unlink()
    write_run(retry, "2_H_binary_s1", seed=1)
    assert C.main(_gate_args(orig, rer, "--rerun", str(retry))) == 2
    args = ["--gate-only", "--original", str(orig), "--rerun", f"{rer}:1_*", "--rerun", str(retry),
            "--expect-sha", RERUN_SHA, "--expect-digest", DIGEST]
    assert C.main(args) == 0


def test_an_output_directory_that_already_holds_files_is_refused(tmp_path):
    orig, rer = _pair_dirs(tmp_path, seeds=(0,))
    out = tmp_path / "out"
    out.mkdir()
    (out / "old.json").write_text("{}")
    assert C.main(_gate_args(orig, rer, "--out", str(out))) == 2


def test_a_rerun_without_an_original_is_listed_as_a_new_seed(tmp_path):
    orig, rer = _pair_dirs(tmp_path, seeds=(0,))
    write_run(rer, "57_H_binary_s3", seed=3)
    out = tmp_path / "gate"
    assert C.main(_gate_args(orig, rer, "--out", str(out))) == 0
    assert json.loads((out / "gate.json").read_text())["new_seeds"] == ["H:binary:3"]
    write_run(rer, "58_H_binary_s4", seed=4, sha="0" * 40)
    assert C.main(_gate_args(orig, rer)) == 2


# ---------------------------------------------------------------- output order


def test_the_gate_output_holds_no_label_key_and_no_divergence_key(tmp_path, capsys):
    orig, rer = _pair_dirs(tmp_path, cells=(("H", "binary"), ("E", "graded")), seeds=(0, 1))
    out = tmp_path / "gate"
    assert C.main(_gate_args(orig, rer, "--out", str(out))) == 0
    printed = capsys.readouterr().out
    record = json.loads((out / "gate.json").read_text())
    assert sorted(p.name for p in out.iterdir()) == ["gate.json"]
    assert C.outcome_keys_in(record) == []
    for word in ("feasib", "unparse", "class", "label", "divergence", "difference",
                 "identical_answer", "checksum", "agree", "band"):
        assert word not in printed
        assert word not in json.dumps(record)
    assert record["sc0_holds"] is True and record["pairs"] == 4


def test_the_outcome_key_scan_finds_a_nested_label():
    assert C.outcome_keys_in({"a": [{"rerun_feasible": True}]}) == ["$.a[0].rerun_feasible"]


def test_the_full_mode_writes_pairs_and_summary_only_after_the_gate_holds(tmp_path):
    orig, rer = tmp_path / "orig", tmp_path / "rerun"
    cohort = tmp_path / "cohort"
    finals = {0: [EMPTY] * 4, 1: [EMPTY] * 4, 2: [EVERY_ITEM] * 4, 3: [EMPTY] * 4, 4: [EMPTY] * 4}
    rerun_finals = {0: [EMPTY] * 4, 1: [EVERY_ITEM] * 4, 2: [EVERY_ITEM] * 4}
    for s, fin in finals.items():
        write_run(cohort, f"c_{s}", seed=s, sha=C.ORIGINAL_SHA["H"] if s < 3 else RERUN_SHA,
                  digest=None if s < 3 else DIGEST, final=fin, checksum=f"o{s}")
        if s < 3:
            write_run(orig, f"{4 + s}_H_binary_s{s}", seed=s, sha=C.ORIGINAL_SHA["H"], digest=None,
                      final=fin, checksum=f"o{s}")
            write_run(rer, f"{4 + s}_H_binary_s{s}", seed=s, final=rerun_finals[s],
                      checksum=f"o{s}" if s == 2 else f"r{s}")
    out = tmp_path / "compare"
    args = ["--original", str(orig), "--rerun", str(rer), "--reference-cohort", str(cohort),
            "--expect-sha", RERUN_SHA, "--expect-digest", DIGEST, "--out", str(out)]
    assert C.main(args) == 0
    rows = list(csv.DictReader(open(out / "pairs.csv")))
    assert [(r["seed"], r["original_class"], r["rerun_class"]) for r in rows] == \
        [("0", "FE", "FE"), ("1", "FE", "V"), ("2", "V", "V")]
    assert [float(r["p_feasibility"]) for r in rows] == [0.75, 0.75, 0.0]
    summary = json.loads((out / "summary.json").read_text())
    feas = summary["labels"]["feasibility"]
    assert (feas["agree"], feas["pairs"]) == (2, 3)
    assert feas["original_true_rerun_false"] == 1 and feas["original_false_rerun_true"] == 0
    assert summary["pairs_with_different_outcome_class"] == 1
    assert summary["pairs_with_equal_checksums"] == 1
    # The same inputs with a logic break at step 2 write the gate record and no outcome file.
    path = rer / "4_H_binary_s0" / "metrics.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    for rec in recs:
        if rec["phase"] == "train" and rec["step"] == 2:
            rec["groups"][0]["adv"][0] += 1.0
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out2 = tmp_path / "compare2"
    args[-1] = str(out2)
    assert C.main(args) == 1
    assert sorted(p.name for p in out2.iterdir()) == ["gate.json"]


# ---------------------------------------------------------------- files-only mode


def _manifest(tmp_path, rows):
    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "regime", "arm_name", "arm", "w", "w_task", "vmax", "seed"])
        for task_id, regime, arm_name, seed in rows:
            arm, weight, w_task = A.CAPACITY_ARM_SPECS[arm_name]
            w.writerow([task_id, regime, arm_name, arm, weight, w_task, f"{VMAX[regime]:.2f}", seed])
    return path


def _files_args(src, manifest, rows):
    return ["--files-only", "--rerun", str(src), "--manifest", str(manifest), "--expect-rows", rows,
            "--expect-sha", RERUN_SHA, "--expect-digest", DIGEST]


def _full_length(**kw):
    return dict(steps=300, eval_every=10, answers=base_answers(300), **kw)


def test_the_file_check_passes_complete_runs_and_prints_no_outcome_value(tmp_path, capsys):
    manifest = _manifest(tmp_path, [(1, "H", "graded", 0), (2, "E", "verdict_only", 1)])
    src = tmp_path / "array"
    write_run(src, "1_H_graded_s0", arm_name="graded", seed=0, **_full_length())
    write_run(src, "2_E_verdict_only_s1", regime="E", arm_name="verdict_only", seed=1,
              final=[None] * 4, **_full_length())
    assert C.main(_files_args(src, manifest, "1-2")) == 0
    out = capsys.readouterr().out
    assert out.count(" OK") == 2
    for word in ("eval_", "violation", "parse_fail", "FE", "FI", "feasib", "unparse"):
        assert word not in out


def test_the_file_check_names_each_problem_and_each_missing_or_unexpected_row(tmp_path, capsys):
    manifest = _manifest(tmp_path, [(1, "H", "graded", 0), (2, "H", "graded", 1), (3, "H", "graded", 2),
                                    (4, "H", "binary", 0)])
    src = tmp_path / "array"
    write_run(src, "1_H_graded_s0", arm_name="graded", seed=0, done=False, **_full_length())
    write_run(src, "2_H_graded_s1", arm_name="graded", seed=1,
              prov={"platform": "Linux-other", "git_dirty": False}, cfg={"max_new_tokens": 64},
              **_full_length())
    write_run(src, "3_H_graded_s2", arm_name="graded", seed=2, steps=300, eval_every=10,
              answers=base_answers(300))
    lines = (src / "3_H_graded_s2" / "metrics.jsonl").read_text().splitlines()
    (src / "3_H_graded_s2" / "metrics.jsonl").write_text("\n".join(lines[:-3]) + "\n")
    write_run(src, "9_H_graded_s9", arm_name="graded", seed=9, **_full_length())
    assert C.main(_files_args(src, manifest, "1-4")) == 1
    out = capsys.readouterr().out
    assert "row 1 array/1_H_graded_s0 git_dirty=True PROBLEMS DONE" in out
    assert "provenance.platform='Linux-other'" in out and "config.max_new_tokens=64" in out
    assert "git_dirty=False" in out
    assert "records train=" in out
    assert "row 4 4_H_binary_s0 git_dirty=None PROBLEMS missing" in out
    assert "unexpected run directory 9_H_graded_s9" in out


def test_the_file_check_flags_a_row_whose_config_disagrees_with_the_manifest(tmp_path, capsys):
    manifest = _manifest(tmp_path, [(1, "H", "graded", 0)])
    src = tmp_path / "array"
    write_run(src, "1_H_graded_s0", arm_name="graded", seed=0, cfg={"w": 0.25}, **_full_length())
    assert C.main(_files_args(src, manifest, "1")) == 1
    assert "config.w=0.25" in capsys.readouterr().out


def test_the_file_check_fails_when_its_only_problem_is_an_unexpected_directory(tmp_path, capsys):
    manifest = _manifest(tmp_path, [(1, "H", "graded", 0)])
    src = tmp_path / "array"
    write_run(src, "1_H_graded_s0", arm_name="graded", seed=0, **_full_length())
    write_run(src, "57_H_graded_s3", arm_name="graded", seed=3, **_full_length())
    assert C.main(_files_args(src, manifest, "1")) == 1
    out = capsys.readouterr().out
    assert "row 1 array/1_H_graded_s0 git_dirty=True OK" in out
    assert "unexpected run directory 57_H_graded_s3" in out


def test_the_file_check_flags_a_row_that_two_sources_both_hold(tmp_path, capsys):
    manifest = _manifest(tmp_path, [(1, "H", "graded", 0)])
    for root in ("first", "retry"):
        write_run(tmp_path / root, "1_H_graded_s0", arm_name="graded", seed=0, **_full_length())
    args = ["--files-only", "--rerun", str(tmp_path / "first"), "--rerun", str(tmp_path / "retry"),
            "--manifest", str(manifest), "--expect-rows", "1", "--expect-sha", RERUN_SHA,
            "--expect-digest", DIGEST]
    assert C.main(args) == 1
    assert "duplicate directory in the sources" in capsys.readouterr().out


def test_the_file_check_refuses_rows_that_the_manifest_lacks(tmp_path):
    manifest = _manifest(tmp_path, [(1, "H", "graded", 0)])
    src = tmp_path / "array"
    write_run(src, "1_H_graded_s0", arm_name="graded", seed=0, **_full_length())
    assert C.main(_files_args(src, manifest, "1-2")) == 2


@pytest.mark.parametrize("prov,needle", [
    ({"git_sha": "0" * 40}, "git_sha="),
    ({"training_sources_digest": "f" * 64}, "training_sources_digest="),
    ({"steps_completed": 299}, "steps_completed=299"),
])
def test_the_file_check_names_a_wrong_commit_digest_or_step_count(tmp_path, capsys, prov, needle):
    manifest = _manifest(tmp_path, [(1, "H", "graded", 0)])
    src = tmp_path / "array"
    write_run(src, "1_H_graded_s0", arm_name="graded", seed=0, prov=prov, **_full_length())
    assert C.main(_files_args(src, manifest, "1")) == 1
    assert needle in capsys.readouterr().out


def test_the_file_check_names_a_row_whose_directory_holds_no_config(tmp_path, capsys):
    manifest = _manifest(tmp_path, [(1, "H", "graded", 0), (2, "H", "graded", 1)])
    src = tmp_path / "array"
    write_run(src, "1_H_graded_s0", arm_name="graded", seed=0, **_full_length())
    (src / "2_H_graded_s1").mkdir()
    assert C.main(_files_args(src, manifest, "1-2")) == 1
    assert "row 2 2_H_graded_s1 git_dirty=None PROBLEMS config.json" in capsys.readouterr().out


@pytest.mark.parametrize("args", [
    ["--files-only", "--rerun", "x", "--expect-sha", RERUN_SHA, "--expect-digest", DIGEST],
    ["--gate-only", "--rerun", "x", "--expect-sha", RERUN_SHA, "--expect-digest", DIGEST],
    ["--original", "x", "--rerun", "x", "--expect-sha", RERUN_SHA, "--expect-digest", DIGEST],
])
def test_a_mode_without_its_required_inputs_is_refused(args):
    assert C.main(args) == 2


def test_a_rerun_that_logged_fewer_steps_than_its_config_is_refused(tmp_path, capsys):
    orig, rer = _pair_dirs(tmp_path, seeds=(0,))
    path = rer / "1_H_binary_s0" / "metrics.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(r) + "\n" for r in recs
                            if not (r["phase"] == "train" and r["step"] == 4)))
    assert C.main(_gate_args(orig, rer)) == 2
    assert "logged 4 of the 5 steps" in capsys.readouterr().err


def test_a_source_that_is_not_a_directory_or_matches_nothing_is_refused(tmp_path):
    orig, rer = _pair_dirs(tmp_path, seeds=(0,))
    assert C.main(_gate_args(orig, tmp_path / "absent")) == 2
    assert C.main(_gate_args(orig, f"{rer}:99_*")) == 2


@pytest.mark.parametrize("text,want", [("1-3", [1, 2, 3]),
                                       ("1-36,57-60", list(range(1, 37)) + [57, 58, 59, 60]),
                                       ("5", [5]), ("2,1,2", [1, 2])])
def test_row_lists_parse(text, want):
    assert C.parse_rows(text) == want


@pytest.mark.parametrize("text", ["", "3-1", "0-2", "a-b", "1-"])
def test_bad_row_lists_are_refused(text):
    with pytest.raises(C.Refusal):
        C.parse_rows(text)


def test_the_tracked_manifest_names_rows_one_to_thirty_six_as_seeds_zero_to_two_of_twelve_cells():
    rows = C.read_manifest(C.DEFAULT_MANIFEST)
    cells = {(rows[r]["regime"], rows[r]["arm_name"]) for r in range(1, 37)}
    assert len(cells) == 12
    assert sorted(int(rows[r]["seed"]) for r in range(1, 37)) == sorted([0, 1, 2] * 12)
    for cell in cells:
        assert sorted(int(rows[r]["seed"]) for r in range(1, 37)
                      if (rows[r]["regime"], rows[r]["arm_name"]) == cell) == [0, 1, 2]


# ---------------------------------------------------------------- the published cohort

# This table holds the labels of the published reward-design cohort, which the plan fixed before any rerun.
PUBLISHED_LABELS = {
    ("H", "binary"): "V V U V V", ("H", "binary_1000x"): "FE FE FE V FE",
    ("H", "graded"): "FE FE U FE FE", ("H", "graded_no_task"): "FE FE FE FE FE",
    ("H", "none"): "V U V V V", ("H", "verdict_only"): "FE FE FE FE U",
    ("E", "binary"): "FI FI U FI FI", ("E", "binary_1000x"): "FI FI U FI FI",
    ("E", "graded"): "FI FI FI U FI", ("E", "graded_no_task"): "U FI FE",
    ("E", "none"): "V FE U U V", ("E", "verdict_only"): "FI FE FI",
}


@pytest.mark.skipif(not os.path.isdir(COHORT), reason="the published capacity cohort is not present")
def test_the_published_cohort_gives_the_reference_values_fixed_before_any_rerun():
    cohort = C.load_side([COHORT], "reference")
    assert len(cohort) == 56
    labels = {}
    for (task, regime, arm, seed), run in sorted(cohort.items()):
        labels.setdefault((regime, arm), []).append(C.run_labels(run)["outcome_class"])
    assert {k: " ".join(v) for k, v in labels.items()} == PUBLISHED_LABELS
    originals = {i: r for i, r in cohort.items() if i[3] in (0, 1, 2)}
    assert len(originals) == 36
    ref = C.reference_values(originals, cohort)["labels"]
    got = {lab: (v["expected"], round(v["sd"], 2), v["range_low"], v["range_high"],
                 v["pairs_at_probability_one"]) for lab, v in ref.items()}
    assert got == {"feasibility": (25.75, 1.92, 22, 29, 12),
                   "unparseable": (23.25, 2.05, 19, 27, 9),
                   "outcome_class": (18.75, 2.11, 14, 23, 3)}
    bands = {"feasibility": (30, 22), "unparseable": (28, 19), "outcome_class": (24, 14)}
    for lab, (m_low, x_low) in bands.items():
        lo, hi = ref[lab]["range_low"], ref[lab]["range_high"]
        assert [C.band(a, 36, lo, hi) for a in (36, 35, m_low, m_low - 1, x_low, x_low - 1)] == \
            ["R", "M", "M", "X", "X", "B"]
    for (task, regime, arm, seed), run in cohort.items():
        assert run["prov"]["git_sha"] == (C.ORIGINAL_SHA[regime] if seed < 3 else RERUN_SHA)
        assert {k: run["cfg"].get(k) for k in C.EXPECTED_CONFIG} == C.EXPECTED_CONFIG
        assert {k: run["prov"].get(k) for k in C.EXPECTED_ENVIRONMENT} == C.EXPECTED_ENVIRONMENT


FINGERPRINT_RUNS = {name: os.path.join(HERE, "runs", array, "4_H_binary_s0")
                    for name, array in (("reference", "572991"), ("new", "758246"), ("base", "758270"))}


@pytest.mark.skipif(not all(os.path.isdir(p) for p in FINGERPRINT_RUNS.values()),
                    reason="the fingerprint runs of 14 September are not present")
def test_the_fingerprint_runs_give_the_divergence_steps_the_ledger_recorded():
    """The ledger of 14 September records that both 30-step reruns left the reference at step 18 and
    left each other at step 21, with identical step-1 groups and no logic break in any comparison."""
    from pathlib import Path
    runs = {k: A._load_run(Path(p)) for k, p in FINGERPRINT_RUNS.items()}
    for a, b, first in (("reference", "new", 18), ("reference", "base", 18), ("base", "new", 21)):
        t = C.trajectory(runs[a], runs[b])
        assert t["step1_identical"] is True and t["logic_break_step"] is None
        assert t["first_answer_difference_step"] == first
        assert t["checksums_equal"] is False


# Under the scope chosen for this block, the two easy-level penalty-only cells keep three seeds.
SINGLE_SEEDS = {cell: (0, 1, 2, 3, 4) for cell in PUBLISHED_LABELS}
SINGLE_SEEDS[("E", "verdict_only")] = SINGLE_SEEDS[("E", "graded_no_task")] = (0, 1, 2)


@pytest.mark.skipif(not os.path.isdir(SINGLE), reason="runs/capacity5_single has not been assembled")
def test_the_single_commit_cohort_is_complete_and_at_one_commit():
    record = json.load(open(os.path.join(SINGLE, "assembled.json")))
    finished = [(e["regime"], e["arm"], int(e["seed"])) for e in record["runs"]]
    declared = [(x["regime"], x["arm"], int(x["seed"])) for x in record.get("not_run", [])]
    expected = {(regime, arm, s) for (regime, arm), seeds in SINGLE_SEEDS.items() for s in seeds}
    assert len(set(finished)) == len(finished), "a run appears twice"
    assert not set(finished) & set(declared), "a seed is both finished and listed as not run"
    assert set(finished) | set(declared) == expected
    for e in record["runs"]:
        prov = json.load(open(os.path.join(SINGLE, e["link"], "provenance.json")))
        assert prov["git_sha"] == RERUN_SHA and prov["training_sources_digest"] == DIGEST


def test_the_new_files_hold_no_banned_term():
    """The comparison script and this test ship in a later supplement, so they must pass its scan."""
    paper = os.path.join(HERE, "paper")
    if not os.path.isfile(os.path.join(paper, "make_supplement.py")):
        pytest.skip("the supplement builder is not present in this layout")
    sys.path.insert(0, paper)
    try:
        import make_supplement
    finally:
        sys.path.remove(paper)
    for name in ("compare_reruns.py", os.path.join("tests", "test_compare_reruns.py")):
        text = open(os.path.join(HERE, name)).read()
        assert make_supplement.scan(name, text) == []
        assert chr(0x2014) not in text
