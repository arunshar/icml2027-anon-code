"""The figures' own text and titles, which are not the tables' LaTeX.

The arm labels are LaTeX because the table generators typeset them. A figure draws its legend,
axis text and titles itself, so it takes a plain form of the same words. The title of a regime's
panel names the limit the runs at that regime were trained under, read from their configs, so
the title cannot disagree with the runs it describes. None of this needs matplotlib.
"""
import analyze


def _run(regime, vmax, task="capacity"):
    cfg = {"regime": regime, "vmax": vmax}
    if task is not None:
        cfg["task"] = task
    return {"cfg": cfg}


def test_plain_labels_carry_no_math_while_the_arm_labels_keep_theirs():
    latex = dict(analyze.ARM_STYLE)
    latex_labels = {arm: st["label"] for arm, st in latex.items()}
    latex_labels.update(analyze.NAMED_LABELS)
    latex_labels["kl"] = analyze.compose_arm_label(
        {"arm": "graded", "w": 5.0, "w_task": 1.0, "task": "capacity", "kl_beta": 0.04})
    for arm, label in latex_labels.items():
        plain = analyze._plain_label(label)
        assert "$" not in plain, (arm, plain)
        assert "\\" not in plain, (arm, plain)
    assert analyze._plain_label(latex_labels["graded"]) == "residual + task"
    assert analyze._plain_label(latex_labels["binary_1000x"]) == "verdict × 1000 + task"
    assert analyze._plain_label(latex_labels["kl"]).endswith("KL β = 0.04")
    # The published labels are unchanged, because the table generators read them.
    labels = analyze.arm_labels()
    assert labels["graded"] == "residual $+$ task"
    assert labels["binary_1000x"] == r"verdict $\times 1000$ $+$ task"


def test_the_regime_title_names_the_limit_the_runs_share():
    runs = [_run("H", 0.03), _run("H", 0.03), _run("E", 0.05)]
    assert analyze.regime_title("H", runs) == "Hard level, capacity fraction = 0.03"
    assert analyze.regime_title("E", runs) == "Easy level, capacity fraction = 0.05"
    speed = [_run("A", 5.0, task="speed"), _run("B", 30.0, task=None)]
    assert analyze.regime_title("A", speed) == r"Hard limit, $v_{\max} = 5$"
    # A config without a task name is a speed run.
    assert analyze.regime_title("B", speed) == r"Easy limit, $v_{\max} = 30$"


def test_a_regime_whose_runs_disagree_on_the_limit_gets_the_phrase_alone():
    runs = [_run("E", 0.03), _run("E", 0.05), _run("H", 0.03)]
    assert analyze.regime_title("E", runs) == "Easy level"
    # The other regime's limit does not count as a disagreement.
    assert analyze.regime_title("H", runs) == "Hard level, capacity fraction = 0.03"
    assert analyze.regime_title("H", []) == "Hard level"
