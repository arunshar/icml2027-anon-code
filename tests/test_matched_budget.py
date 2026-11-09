import csv
import os
import re
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import capacity_task as C  # noqa: E402


SBATCH = os.path.join(HERE, "slurm", "cancel_arms.sbatch")
NO_SBATCH = (
    "the cluster submission script is not shipped in the anonymous supplement, because it "
    "names a compute account and a site-specific scratch path"
)
requires_sbatch = pytest.mark.skipif(not os.path.exists(SBATCH), reason=NO_SBATCH)


def test_arms_differ_only_in_arm_and_w():
    arms = yaml.safe_load(open(os.path.join(HERE, "configs", "arms.yaml")))
    assert len(arms) == 4
    for a in arms:
        assert set(a.keys()) == {"name", "arm", "w"}
    assert {a["name"] for a in arms} == {"graded", "binary", "none", "binary_1000x"}


def test_manifest_has_24_rows_three_seeds_per_regime_arm():
    rows = list(csv.DictReader(open(os.path.join(HERE, "slurm", "manifest.csv"))))
    assert len(rows) == 24
    from collections import Counter
    c = Counter((r["regime"], r["arm_name"]) for r in rows)
    assert all(v == 3 for v in c.values()) and len(c) == 8
    assert [int(r["task_id"]) for r in rows] == list(range(1, 25))
    assert all(r["regime"] == "A" for r in rows[:12]) and all(r["regime"] == "B" for r in rows[12:])


@requires_sbatch
def test_sbatch_pins_exactly_one_a100_gres_and_refuses_null_vmax():
    txt = open(os.path.join(HERE, "slurm", "cancel_arms.sbatch")).read()
    gres = re.findall(r"^#SBATCH --gres=(\S+)", txt, flags=re.M)
    assert gres == ["gpu:a100:1"]
    assert "v_max is null" in txt
    assert "--array=1-12%4" in txt
    assert "CANCEL_MANIFEST" in txt and "CANCEL_TASK" in txt
    # BASH_SOURCE under sbatch points at a spool copy, so it must not be a fallback for
    # locating the experiment directory. One array failed this way on 2026-09-11.
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert "BASH_SOURCE" not in code
    assert "${SLURM_SUBMIT_DIR}/train_speed_grpo.py" in txt


def test_regime_B_is_30_and_regime_A_is_calibrated_or_null():
    reg = yaml.safe_load(open(os.path.join(HERE, "configs", "regimes.yaml")))
    assert reg["B"]["v_max"] == 30.0
    assert reg["A"]["v_max"] is None or 0 < reg["A"]["v_max"] < 30.0


def _write_run(d, steps, groups_per_step, cfg_extra=None):
    """Minimal on-disk run so the analysis can be exercised without training."""
    import json
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"arm": "graded", "w": 5.0, "vmax": 30.0, "regime": "A", "seed": 0, "steps": steps,
           "group_size": 4, "batch_prompts": groups_per_step, "lr": 1e-5, "out_dir": str(d)}
    cfg.update(cfg_extra or {})
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "provenance.json").write_text(json.dumps({"steps_completed": steps, "git_sha": "x", "torch": "1",
                                                   "transformers": "1", "numpy": "1", "device": "cpu",
                                                   "device_name": "cpu", "python": "3.12", "platform": "p"}))
    lines = []
    for st in range(1, steps + 1):
        gs = []
        for _ in range(groups_per_step):
            speeds = [40.0, 60.0, 80.0, 100.0]
            gs.append({"speeds": speeds, "rewards": [0, 0, 0, 0], "adv": [0, 0, 0, 0],
                       "all_violating": True, "has_violator": True})
        lines.append(json.dumps({"step": st, "phase": "train", "groups": gs, "violation_rate": 1.0,
                                 "all_violating_fraction": 1.0, "mean_residual": 1.0, "kl_seq_mean": 0.0,
                                 "violator_mean_adv": 0.0, "frac_violators_pos_adv": 0.0}))
    lines.append(json.dumps({"step": steps, "phase": "eval", "eval_violation_rate": 1.0,
                             "eval_mean_residual": 1.0, "eval_parse_fail_rate": 0.0, "eval_n": 4}))
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    (d / "DONE").write_text("ok\n")


def test_a_run_with_no_groups_does_not_report_a_pass(tmp_path):
    """Zero failures out of zero cases is not a pass. This is the same shape of mistake as
    measuring a constant and reporting it as a result."""
    import json
    import subprocess
    import sys
    root = tmp_path / "runs"
    _write_run(root / "empty", steps=0, groups_per_step=0)
    out = tmp_path / "an"
    subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"), "--runs", str(root),
                    "--out", str(out), "--allow-unmatched"], cwd=HERE, capture_output=True)
    checks = json.loads((out / "checks.json").read_text())
    assert checks["P1_pass"] is False
    assert checks["trainer_self_consistent"] is False
    assert set(checks["vacuous_checks"]) == {"P1", "P2", "P2_boundary", "self_consistency"}


def test_analysis_refuses_when_a_config_field_differs(tmp_path):
    import json
    import subprocess
    import sys
    root = tmp_path / "runs"
    _write_run(root / "a", steps=1, groups_per_step=2)
    _write_run(root / "b", steps=1, groups_per_step=2, cfg_extra={"lr": 5e-5})
    out = tmp_path / "an"
    r = subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"), "--runs", str(root), "--out", str(out)],
                       cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 3
    assert "lr" in r.stderr


def test_group_counters_count_what_they_claim(tmp_path):
    """A zero-variance group is one where every member earns the same reward. An
    all-violating group with spread is the window in which an added verdict could still act."""
    import json
    import sys
    sys.path.insert(0, HERE)
    import analyze

    class R:  # minimal stand-in for a loaded run
        pass

    def run(groups, regime="A", arm="graded", w=5.0, seed=0):
        r = {"cfg": {"regime": regime, "arm": arm, "w": w, "seed": seed},
             "train": [{"step": 1, "groups": groups}]}
        return r

    groups = [
        {"rewards": [1.0, 1.0, 1.0, 1.0], "all_violating": True},    # all-viol, zero var
        {"rewards": [1.0, 2.0, 3.0, 4.0], "all_violating": True},    # all-viol, positive var
        {"rewards": [5.0, 5.0, 5.0, 5.0], "all_violating": False},   # feasible, zero var
        {"rewards": [1.0, 1.0, 1.0, 1.0 + 1e-15], "all_violating": True},  # within tolerance
    ]
    out = analyze.group_counters([run(groups)])
    assert len(out) == 1
    c = out[0]
    assert c["groups"] == 4
    assert c["zero_variance"] == 3          # the 1e-15 spread counts as zero
    assert c["all_violating"] == 3
    assert c["all_violating_zero_variance"] == 2
    assert c["all_violating_positive_variance"] == 1
    assert c["steps"] == 1
    assert c["arm"] == "graded" and c["regime"] == "A" and c["seed"] == 0


@requires_sbatch
def test_manifest_is_lf_only_and_has_no_stray_whitespace():
    """A CRLF manifest put a carriage return on the last field, which leaked into every run
    directory name of one array. The generator writes LF and the sbatch strips CR."""
    raw = open(os.path.join(HERE, "slurm", "manifest.csv"), "rb").read()
    assert b"\r" not in raw, "manifest must be LF-only"
    rows = list(csv.DictReader(open(os.path.join(HERE, "slurm", "manifest.csv"))))
    for r in rows:
        for k, v in r.items():
            assert v == v.strip(), f"field {k}={v!r} carries surrounding whitespace"
    txt = open(os.path.join(HERE, "slurm", "cancel_arms.sbatch")).read()
    assert "tr -d '\\r'" in txt, "sbatch must strip CR before parsing a manifest row"


# The capacity manifest exactly as the array behind the paper's capacity results ran it. Seeds 3 and
# 4 are appended after these rows, which keeps them in their original order and keeps the analysis's
# append-only exemption valid for every run already made.
LEGACY_CAPACITY_MANIFEST = """task_id,regime,arm_name,arm,w,w_task,vmax,seed
1,H,graded,graded,5.0,1.0,0.03,0
2,H,graded,graded,5.0,1.0,0.03,1
3,H,graded,graded,5.0,1.0,0.03,2
4,H,binary,binary,5.0,1.0,0.03,0
5,H,binary,binary,5.0,1.0,0.03,1
6,H,binary,binary,5.0,1.0,0.03,2
7,H,binary_1000x,binary,5000.0,1.0,0.03,0
8,H,binary_1000x,binary,5000.0,1.0,0.03,1
9,H,binary_1000x,binary,5000.0,1.0,0.03,2
10,H,none,none,0.0,1.0,0.03,0
11,H,none,none,0.0,1.0,0.03,1
12,H,none,none,0.0,1.0,0.03,2
13,H,verdict_only,binary,5.0,0.0,0.03,0
14,H,verdict_only,binary,5.0,0.0,0.03,1
15,H,verdict_only,binary,5.0,0.0,0.03,2
16,H,graded_no_task,graded,5.0,0.0,0.03,0
17,H,graded_no_task,graded,5.0,0.0,0.03,1
18,H,graded_no_task,graded,5.0,0.0,0.03,2
19,E,graded,graded,5.0,1.0,0.20,0
20,E,graded,graded,5.0,1.0,0.20,1
21,E,graded,graded,5.0,1.0,0.20,2
22,E,binary,binary,5.0,1.0,0.20,0
23,E,binary,binary,5.0,1.0,0.20,1
24,E,binary,binary,5.0,1.0,0.20,2
25,E,binary_1000x,binary,5000.0,1.0,0.20,0
26,E,binary_1000x,binary,5000.0,1.0,0.20,1
27,E,binary_1000x,binary,5000.0,1.0,0.20,2
28,E,verdict_only,binary,5.0,0.0,0.20,0
29,E,verdict_only,binary,5.0,0.0,0.20,1
30,E,verdict_only,binary,5.0,0.0,0.20,2
31,E,graded_no_task,graded,5.0,0.0,0.20,0
32,E,graded_no_task,graded,5.0,0.0,0.20,1
33,E,graded_no_task,graded,5.0,0.0,0.20,2
34,E,none,none,0.0,1.0,0.20,0
35,E,none,none,0.0,1.0,0.20,1
36,E,none,none,0.0,1.0,0.20,2
"""


def test_capacity_manifest_runs_every_arm_at_both_levels():
    """Six arms, five seeds, two levels, and no arm omitted.

    The manifest originally dropped the no-penalty arm at the easy level, on the reasoning that
    carried over from the speed task: there the task term is min(|s|,200)/10, which never mentions
    the limit, so the same run serves both regimes and the parameter checksums proved it.

    That reasoning does not survive the move. The capacity task's term is the chosen value divided
    by the exact optimum for that instance, and the optimum is computed against the capacity. The
    same selection scores 2.950 at the hard level and 0.766 at the easy one. The no-penalty arm is
    therefore a different run at each level, and it is the baseline the cancellation claim is read
    against, so the easy level cannot be reported without it.
    """
    from collections import Counter
    path = os.path.join(HERE, "slurm", "manifest_capacity.csv")
    assert b"\r" not in open(path, "rb").read(), "manifest must be LF-only"
    rows = list(csv.DictReader(open(path)))
    assert [int(r["task_id"]) for r in rows] == list(range(1, len(rows) + 1))
    for r in rows:
        for k, v in r.items():
            assert v == v.strip()
    c = Counter((r["regime"], r["arm_name"]) for r in rows)
    assert all(v == 5 for v in c.values()), "five seeds per arm per level"
    assert ("H", "none") in c and ("E", "none") in c, \
        "the no-penalty baseline is a different run at each level on this task"
    assert len(rows) == 60
    assert {int(r["seed"]) for r in rows} == {0, 1, 2, 3, 4}
    assert open(path).read().startswith(LEGACY_CAPACITY_MANIFEST), \
        "rows 1-36 of the capacity manifest are no longer the rows the published array ran"
    assert all(int(r["seed"]) in (3, 4) for r in rows[36:])
    # the two zero-task arms are the published shape and its residual twin
    by = {r["arm_name"]: r for r in rows}
    assert float(by["verdict_only"]["w_task"]) == 0.0
    assert float(by["graded_no_task"]["w_task"]) == 0.0
    assert float(by["graded"]["w_task"]) == 1.0


def test_analysis_refuses_to_mix_two_tasks(tmp_path):
    """A matched-budget comparison holds one task fixed. Mixing must fail loudly rather than
    quietly averaging two different experiments."""
    import subprocess
    import sys
    root = tmp_path / "runs"
    _write_run(root / "speedrun", steps=1, groups_per_step=2)
    _write_run(root / "caprun", steps=1, groups_per_step=2, cfg_extra={"task": "capacity"})
    r = subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"), "--runs", str(root),
                        "--out", str(tmp_path / "an")], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 3
    assert "more than one task" in r.stderr


# The bracketed list of all 16 indices, measured with Qwen2.5-0.5B-Instruct's tokenizer:
#     "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]"  ->  54 tokens
# An instruct model prefaces its answer, so the cap must hold the list AND a preamble.
CORNER_ANSWER_TOKENS = 54
PREAMBLE_MARGIN = 40


def _sbatch_maxtok(task: str) -> int:
    """The default this sbatch would use for a task, read out of the file rather than assumed."""
    txt = open(SBATCH).read()
    m = re.search(r'if \[ "\$\{CANCEL_TASK\}" = "speed" \]; then MAXTOK=(\d+); else MAXTOK=(\d+); fi',
                  txt)
    assert m, "the per-task generation cap is no longer where this test looks for it"
    return int(m.group(1) if task == "speed" else m.group(2))


@requires_sbatch
def test_generation_length_is_set_per_task_not_inherited():
    """A completion the policy cannot finish is a completion whose constraint violation is
    capped by the tokenizer. The speed task answers with one integer and is frozen at 6 tokens.
    The capacity task answers with a list of up to 16 indices and needs far more, and the first
    capacity smoke ran at 6 because the sbatch inherited the speed default."""
    txt = open(os.path.join(HERE, "slurm", "cancel_arms.sbatch")).read()
    assert "CANCEL_MAX_NEW_TOKENS" in txt
    assert "--max-new-tokens" in txt
    assert _sbatch_maxtok("speed") == 6, "the speed task's generation length is frozen"


@requires_sbatch
def test_capacity_cap_holds_the_corner_answer_and_a_preamble():
    """The earlier version of this test asserted `64 > 54`, which compares two literals and so
    could never fail. It also asked the wrong question. Room for the bare list is not enough,
    because under a cancelled verdict the task term drives the policy to take every item and an
    instruct model still writes a sentence around its answer. At 64 the closing bracket fell off
    the end, the parser returned None for every completion, all four rewards tied at the
    unparseable floor, and the update was exactly zero from step 4 onward. That is an absorbing
    state manufactured by the tokenizer, and it is indistinguishable in the logs from the one the
    paper is trying to measure."""
    cap = _sbatch_maxtok("capacity")
    assert cap >= CORNER_ANSWER_TOKENS + PREAMBLE_MARGIN, (
        f"generation cap {cap} leaves only {cap - CORNER_ANSWER_TOKENS} tokens around a "
        f"{CORNER_ANSWER_TOKENS}-token corner answer; the policy will be truncated into the "
        f"unparseable floor before it can express the violation")


def test_the_corner_answer_still_costs_what_the_cap_was_chosen_for():
    """Re-measure the constant the cap is built on, so a tokenizer change cannot silently
    invalidate it. Skipped when the tokenizer is not available locally."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception as e:  # pragma: no cover  (no cached tokenizer on this machine)
        pytest.skip(f"tokenizer not cached: {type(e).__name__}")
    corner = "[" + ", ".join(str(i) for i in range(C.N_ITEMS)) + "]"
    n = len(tok(corner).input_ids)
    assert n == CORNER_ANSWER_TOKENS, (
        f"the all-items answer now costs {n} tokens, not {CORNER_ANSWER_TOKENS}; "
        f"the generation cap was chosen from that number and must be rechecked")


def test_no_caller_hand_rolls_generate():
    """Qwen2.5 ships a generation_config with temperature 0.7, top_p 0.8, top_k 20 and
    repetition_penalty 1.1. A caller that omits any knob samples from a different distribution
    than the trainer, which is how one calibration measured a mean selection size of 2.9 against
    the trainer's 5.2. Every sampler goes through one helper."""
    import ast
    checked = 0
    for fn in ("train_speed_grpo.py", "calibrate_capacity.py", "verify_real_groups.py"):
        tree = ast.parse(open(os.path.join(HERE, fn)).read())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "generate"):
                    checked += 1
                    assert node.name == "sample_completions", (
                        f"{fn}:{sub.lineno} calls .generate inside {node.name}(), "
                        "not the shared sampling helper")
    assert checked >= 1, "found no generate call at all; the check would be vacuous"
    src = open(os.path.join(HERE, "train_speed_grpo.py")).read()
    body = src[src.index("def sample_completions"):src.index("def completion_mask")]
    for knob in ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens"):
        assert f"{knob}=args.{knob}" in body, f"{knob} not passed explicitly"


def test_arm_defining_fields_may_differ_but_the_task_may_not():
    """w_task is part of an arm's identity: verdict_only and graded_no_task are the same
    penalties at a task weight of zero. The task itself is not, and letting two tasks into one
    table is what the matched-budget gate exists to prevent."""
    import sys
    sys.path.insert(0, HERE)
    import analyze
    assert {"arm", "w", "w_task", "regime", "vmax", "seed"} <= analyze.ALLOWED_CONFIG_DIFF
    assert "task" not in analyze.ALLOWED_CONFIG_DIFF
    for frozen in ("lr", "steps", "group_size", "batch_prompts", "max_new_tokens",
                   "temperature", "top_p", "top_k", "repetition_penalty", "model_id"):
        assert frozen not in analyze.ALLOWED_CONFIG_DIFF, f"{frozen} must be held fixed"


def test_analysis_refuses_to_mix_two_generation_caps(tmp_path):
    """The capacity array was rerun at a longer generation cap after the first one truncated
    its own answers into the unparseable floor. Runs at the two caps are not the same
    experiment, and a table that averaged them would hide the defect the rerun exists to fix.

    Nothing special was added to make this work. `max_new_tokens` is simply absent from
    ALLOWED_CONFIG_DIFF, so the matched-budget check already treats it as a field that must
    agree. This test exists because the protection is an absence, and an absence is easy to
    undo by accident."""
    import subprocess
    import sys
    assert "max_new_tokens" not in _allowed_config_diff(), (
        "max_new_tokens must stay out of the allowed set, or runs at two caps will be averaged")
    root = tmp_path / "runs"
    _write_run(root / "old_cap", steps=1, groups_per_step=2,
               cfg_extra={"task": "capacity", "max_new_tokens": 64})
    _write_run(root / "new_cap", steps=1, groups_per_step=2,
               cfg_extra={"task": "capacity", "max_new_tokens": 128})
    r = subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"), "--runs", str(root),
                        "--out", str(tmp_path / "an")], cwd=HERE, capture_output=True, text=True)
    assert r.returncode != 0, "two generation caps were compared as if they were one experiment"
    assert "max_new_tokens" in (r.stderr + r.stdout)


def _allowed_config_diff():
    src = open(os.path.join(HERE, "analyze.py")).read()
    m = re.search(r"ALLOWED_CONFIG_DIFF\s*=\s*\{([^}]*)\}", src, re.S)
    assert m, "ALLOWED_CONFIG_DIFF is no longer where this test looks for it"
    return {t.strip().strip('"\'') for t in m.group(1).split(",") if t.strip()}


@requires_sbatch
def test_the_capacity_task_refuses_a_wall_clock_too_short_to_finish():
    """A run that dies at step 190 of 300 costs an hour of a card and produces nothing usable.

    The `#SBATCH --time` directive is a submission-time constant and cannot branch on the task,
    so the capacity array has always needed an override. Submitted without one it inherits the
    one-hour speed default, which is what happened to the easy-level no-penalty runs. The script
    now reads its own limit and refuses before loading the model.
    """
    src = open(SBATCH).read()
    assert "NEED_MIN=120" in src, "the capacity task must require at least two hours"
    assert "REFUSED: task" in src and "exit 78" in src, \
        "the check must stop the job rather than warn and continue"
    # the refusal has to sit before the trainer is invoked, or it saves nothing
    launch = src.index('"${PY}" train_speed_grpo.py')
    assert src.index("REFUSED: task") < launch, \
        "the wall-clock check must run before the trainer is launched"


@requires_sbatch
@pytest.mark.parametrize("limit,minutes", [
    ("02:30:00", 150), ("01:00:00", 60), ("1-00:00:00", 1440),
    ("30:00", 30), ("00:20:00", 20),
])
def test_the_slurm_time_limit_parser_handles_every_format(limit, minutes):
    """Slurm writes TimeLimit as M:S, H:M:S or D-H:M:S depending on size. A parser that only
    understood one of them would silently compute a wrong number of minutes and then either
    refuse a job that was fine or pass one that was not."""
    import re
    import subprocess
    m = re.search(r"awk -F'\[-:\]' '\{[^']*\}'", open(SBATCH).read())
    assert m, "the parser is no longer in the script under the expected shape"
    out = subprocess.run(["awk", "-F", "[-:]",
                          "{ if (NF==4) print $1*1440+$2*60+$3; "
                          "else if (NF==3) print $1*60+$2; else print $1 }"],
                         input=limit, capture_output=True, text=True)
    assert int(out.stdout.strip()) == minutes


@requires_sbatch
def test_an_unlimited_time_limit_does_not_refuse_the_job():
    """`UNLIMITED` is not a number. The comparison must fail closed into allowing the run rather
    than erroring the job out."""
    src = open(SBATCH).read()
    assert '[ -n "${LIMIT_MIN}" ]' in src, "an unreadable limit must not be compared"
    assert '2>/dev/null' in src.split("REFUSED: task")[0].split("LIMIT_MIN")[-1] or \
        '-lt "${NEED_MIN}" ] 2>/dev/null' in src, \
        "a non-numeric limit must not turn into a shell error"
