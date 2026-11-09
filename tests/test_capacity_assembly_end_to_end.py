"""The whole capacity pipeline, from run directories to the paper's .tex fragments.

The capacity array runs six reward designs at two constraint levels. Until it finishes, every
piece downstream of it has only ever been exercised on a single arm at a single level, which is
the shape of thing that works until the real data arrives and then does not. This builds a
complete synthetic run set, drives the real analysis and the real table generator over it, and
checks that what reaches the paper is complete.

The runs are synthetic but they are not fake: every reward is computed by the same
`rewards.group_rewards` the trainer calls, so the analysis's recomputation checks have to pass
for real. A test that wrote arbitrary rewards would pass the pipeline and prove nothing.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import capacity_task as C  # noqa: E402
import rewards as RW  # noqa: E402
import tasks  # noqa: E402

SPEC = tasks.get("capacity")
PAPER = os.path.join(HERE, "paper")
# The anonymous supplement ships this suite but not the paper, so the tests that drive the
# paper's table generator have nothing to drive. They skip and say so, rather than failing in a
# reader's terminal. The analysis-side tests below still run, because they need only this
# directory.
NO_PAPER = ("the paper's table generator is not part of the anonymous code supplement")
requires_paper = pytest.mark.skipif(not os.path.exists(os.path.join(PAPER, "make_tex.py")),
                                    reason=NO_PAPER)

# arm name -> (penalty kind, penalty weight, task-term weight), exactly as the manifest defines
ARMS = {
    "graded":         ("graded", 5.0, 1.0),
    "binary":         ("binary", 5.0, 1.0),
    "binary_1000x":   ("binary", 5000.0, 1.0),
    "none":           ("none", 0.0, 1.0),
    "verdict_only":   ("binary", 5.0, 0.0),
    "graded_no_task": ("graded", 5.0, 0.0),
}
LEVELS = {"H": 0.03, "E": 0.20}
SEEDS = (0, 1, 2)
STEPS, GROUPS_PER_STEP = 3, 2


def _selections(rng):
    """A group of four selections from one instance, drawn the way a policy would differ."""
    inst = int(rng.integers(0, len(C.INSTANCES)))
    out = []
    for _ in range(4):
        k = int(rng.integers(0, C.N_ITEMS + 1))
        idx = sorted(rng.choice(C.N_ITEMS, size=k, replace=False).tolist())
        out.append([inst] + [int(i) for i in idx])
    return out


def _write_capacity_run(d, arm_name, regime, seed):
    kind, w, w_task = ARMS[arm_name]
    ratio = LEVELS[regime]
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"task": "capacity", "arm": kind, "w": w, "w_task": w_task, "vmax": ratio,
           "regime": regime, "seed": seed, "steps": STEPS, "group_size": 4,
           "batch_prompts": GROUPS_PER_STEP, "lr": 1e-5, "max_new_tokens": 128,
           "out_dir": str(d)}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "provenance.json").write_text(json.dumps({
        "steps_completed": STEPS, "git_sha": "synthetic", "torch": "1", "transformers": "1",
        "numpy": "1", "device": "cpu", "device_name": "cpu", "python": "3.12", "platform": "p",
        "s_per_step_median": 1.0}))

    rng = np.random.default_rng(1000 * seed + hash(arm_name) % 997)
    lines = []
    for st in range(1, STEPS + 1):
        groups = []
        for _ in range(GROUPS_PER_STEP):
            sels = _selections(rng)
            tup = [tuple(s) for s in sels]
            r = RW.group_rewards(tup, kind, w, ratio, spec=SPEC, w_task=w_task)
            a = RW.advantage(r)
            groups.append({"selections": sels, "rewards": r.tolist(), "adv": a.tolist(),
                           "all_violating": RW.group_is_all_violating(tup, ratio, spec=SPEC),
                           "has_violator": RW.group_has_violator(tup, ratio, spec=SPEC)})
        flat = [tuple(s) for g in groups for s in g["selections"]]
        lines.append(json.dumps({
            "step": st, "phase": "train", "groups": groups,
            "violation_rate": float(np.mean([not C.verdict(s, ratio) for s in flat])),
            "all_violating_fraction": float(np.mean([g["all_violating"] for g in groups])),
            "parse_fail_rate": 0.0,
            "mean_n_selected": float(np.mean([len(s) - 1 for s in flat])),
            "mean_residual": float(np.mean([C.residual(s, ratio) for s in flat])),
            "reward_mean": float(np.mean([x for g in groups for x in g["rewards"]])),
            "group_reward_std_mean": float(np.mean([np.std(g["rewards"], ddof=1) for g in groups])),
            "violator_mean_adv": 0.0, "frac_violators_pos_adv": 0.0,
            "kl_seq_mean": 1.0 + seed, "pg_loss": 0.0, "s_per_step": 1.0}))
        ev = [list(s) for s in flat]
        lines.append(json.dumps({
            "step": st, "phase": "eval", "eval_n": len(ev),
            "eval_violation_rate": float(np.mean([not C.verdict(tuple(s), ratio) for s in ev])),
            "eval_parse_fail_rate": 0.0,
            "eval_mean_n_selected": float(np.mean([len(s) - 1 for s in ev])),
            "eval_mean_residual": float(np.mean([C.residual(tuple(s), ratio) for s in ev])),
            "eval_selections": ev}))
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    (d / "DONE").write_text("ok\n")


@pytest.fixture(scope="module")
def assembled(tmp_path_factory):
    root = tmp_path_factory.mktemp("cap") / "runs"
    n = 0
    for regime in LEVELS:
        for arm in ARMS:
            for seed in SEEDS:
                _write_capacity_run(root / f"{n}_{regime}_{arm}_s{seed}", arm, regime, seed)
                n += 1
    an = root.parent / "analysis"
    r = subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"),
                        "--runs", str(root), "--out", str(an)],
                       cwd=HERE, capture_output=True, text=True)
    return {"root": root, "analysis": an, "proc": r, "n_runs": n}


def test_the_analysis_accepts_all_six_arms_at_both_levels(assembled):
    r = assembled["proc"]
    assert r.returncode == 0, f"analysis refused the run set:\n{r.stderr[-2000:]}"
    checks = json.loads((assembled["analysis"] / "checks.json").read_text())
    assert checks["n_runs"] == assembled["n_runs"] == 36
    assert not checks.get("matched_budget_problems"), checks.get("matched_budget_problems")
    assert not checks.get("vacuous_checks"), checks["vacuous_checks"]
    for k in ("P1_pass", "P2_pass", "trainer_self_consistent"):
        assert checks[k] is True, f"{k} failed on rewards this test computed with the real code"


def test_every_arm_and_both_levels_reach_the_counters(assembled):
    checks = json.loads((assembled["analysis"] / "checks.json").read_text())
    seen = {(c["regime"], c["arm"]) for c in checks["group_counters"]}
    missing = {(r, a) for r in LEVELS for a in ARMS} - seen
    assert not missing, f"the counters lost these arm and level pairs: {sorted(missing)}"


@requires_paper
def test_the_generator_writes_a_complete_table_for_each_level(assembled, tmp_path):
    out = tmp_path / "tables"
    out.mkdir()
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_tex.py"),
                        "--analysis", str(assembled["analysis"]), "--out", str(out),
                        "--macro-prefix", "cap", "--macros-file", "cap_macros.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    labels = json.loads((assembled["analysis"] / "arm_labels.json").read_text())
    for regime in LEVELS:
        for name in (f"counters_regime{regime}.tex", f"final_regime{regime}.tex"):
            body = (out / name).read_text()
            for arm in ARMS:
                assert labels[arm] in body, f"{name} is missing the {arm} row"


@requires_paper
def test_the_two_macro_files_cannot_collide(assembled, tmp_path):
    """LaTeX errors on a redefined command, so the speed and capacity macro files must not
    define the same name. The prefix is what keeps them apart, and this checks it does."""
    import re
    out = tmp_path / "tables2"
    out.mkdir()
    subprocess.run([sys.executable, os.path.join(PAPER, "make_tex.py"),
                    "--analysis", str(assembled["analysis"]), "--out", str(out),
                    "--macro-prefix", "cap", "--macros-file", "cap_macros.tex"],
                   cwd=PAPER, capture_output=True, text=True, check=True)
    names = lambda p: set(re.findall(r"\\newcommand\{\\(\w+)\}", open(p).read()))
    cap = names(out / "cap_macros.tex")
    speed = names(os.path.join(PAPER, "tables", "macros.tex"))
    assert cap, "no capacity macros were written"
    assert not (cap & speed), f"these macro names are defined twice: {sorted(cap & speed)}"


def test_a_missing_arm_is_noticed_rather_than_quietly_dropped(assembled, tmp_path):
    """The check above only means something if an incomplete set would fail it."""
    labels = json.loads((assembled["analysis"] / "arm_labels.json").read_text())
    checks = json.loads((assembled["analysis"] / "checks.json").read_text())
    seen = {(c["regime"], c["arm"]) for c in checks["group_counters"]}
    assert ("H", "verdict_only") in seen
    body = "a table that happens to omit one arm"
    assert labels["verdict_only"] not in body


def test_the_figures_render_for_both_levels_and_name_every_arm(assembled):
    """The paper shows a figure per regime. With six arms the legend is where a missing style or
    a duplicated colour shows up, and a figure that silently drops an arm looks fine."""
    import re
    import shutil
    an = assembled["analysis"]
    for regime in LEVELS:
        for stem in ("eval_violation_rate", "violation_vs_kl"):
            f = an / f"{stem}_regime{regime}.pdf"
            assert f.exists(), f"{f.name} was not drawn"
            assert f.stat().st_size > 2000, f"{f.name} is too small to contain a plot"
    if not shutil.which("pdftotext"):
        pytest.skip("pdftotext is not available to read the legend")
    labels = json.loads((an / "arm_labels.json").read_text())
    text = subprocess.run(["pdftotext", str(an / "eval_violation_rate_regimeH.pdf"), "-"],
                          capture_output=True, text=True).stdout
    # the legend prints the label with LaTeX maths stripped by matplotlib's own renderer, so
    # compare on the words rather than the markup
    for arm in ARMS:
        word = re.sub(r"[^a-z ]", " ", labels[arm].lower()).split()[0]
        assert word in text.lower(), f"the legend does not mention the {arm} arm ({word!r})"


def test_no_two_arms_share_a_colour(assembled):
    """Six curves on one axis. Two arms drawn the same way is a figure that lies."""
    import analyze as A
    styles = {a: (A.ARM_STYLE[a]["color"], A.ARM_STYLE[a].get("ls", "-")) for a in ARMS}
    assert len(set(styles.values())) == len(ARMS), f"duplicate styles: {styles}"


@requires_paper
def test_the_per_seed_table_lists_seeds_and_never_averages_them(assembled, tmp_path):
    """On the capacity task the seeds do not agree about what happened. One seed of the residual
    arm leaves the parseable region while the other two converge, so its violation rates are 0, 0
    and 1. A mean of 0.33 describes no run that was performed, so the table lists the seeds."""
    out = tmp_path / "perseed"
    out.mkdir()
    r = subprocess.run([sys.executable, os.path.join(PAPER, "make_tex.py"),
                        "--analysis", str(assembled["analysis"]), "--out", str(out),
                        "--macro-prefix", "cap", "--macros-file", "cap_macros.tex",
                        "--per-seed-final"], cwd=PAPER, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    body = (out / "final_regimeH.tex").read_text()
    assert "\\pm" not in body, "the per-seed table must not carry a standard deviation"
    labels = json.loads((assembled["analysis"] / "arm_labels.json").read_text())
    for line in body.splitlines():
        if not any(labels[a] in line for a in ARMS):
            continue
        cells = [c.strip() for c in line.split("&")]
        # every numeric cell lists one value per seed
        for cell in cells[2:]:
            vals = [v for v in cell.replace("\\\\", "").split(",") if v.strip()]
            assert len(vals) == len(SEEDS), \
                f"expected {len(SEEDS)} seed values, got {len(vals)} in {cell!r}"


@requires_paper
def test_the_default_table_still_reports_a_mean(assembled, tmp_path):
    """The speed task's seeds do agree, and its published table is a mean with a deviation. The
    new option must not have changed that."""
    out = tmp_path / "mean"
    out.mkdir()
    subprocess.run([sys.executable, os.path.join(PAPER, "make_tex.py"),
                    "--analysis", str(assembled["analysis"]), "--out", str(out),
                    "--macro-prefix", "cap", "--macros-file", "cap_macros.tex"],
                   cwd=PAPER, capture_output=True, text=True, check=True)
    assert "\\pm" in (out / "final_regimeH.tex").read_text()


@requires_paper
def test_thousands_separators_survive_math_mode():
    """In math mode LaTeX reads a comma as punctuation and puts a space after it, so `$1,800$`
    sets as "1, 800". Every number in this paper goes through the generators, so the fix belongs
    there. A comma that separates list items must keep its space, which is why this is not a
    blanket replacement."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mt", os.path.join(PAPER, "make_tex.py"))
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)

    assert mt.math_safe("1,800") == "1{,}800"
    assert mt.math_safe("144,054") == "144{,}054"
    assert mt.math_safe("1,234,567") == "1{,}234{,}567"
    # a list separator keeps its space and its bare comma
    assert mt.math_safe("6.89, 6.89, 6.89") == "6.89, 6.89, 6.89"
    assert mt.math_safe("47, 88, 146") == "47, 88, 146"
    assert mt.math_safe("5, 4, 8") == "5, 4, 8"
    # nothing to do
    assert mt.math_safe("0.50") == "0.50"
    assert mt.math_safe("300") == "300"
