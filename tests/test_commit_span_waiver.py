"""The matched-budget rule, and the one field it is allowed to reason about.

The capacity array ran across two commits, because the queue drained over hours while ledger and
paper work was committed. The strict rule refused the whole comparison on a `git_sha` difference,
which asks the wrong question. What the comparison needs is that the trainer, the reward, the task
and the frozen constants were identical. A commit that touches none of those cannot have changed
a run.

The waiver is therefore narrow and it must stay narrow. These tests pin the three ways it could
go wrong: waiving a commit that did touch training code, waiving when git cannot answer at all,
and waiving quietly.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze as A  # noqa: E402


class _Dir:
    def __init__(self, name):
        self.name = name


def _run(name, sha, **cfg):
    base = {"arm": "graded", "w": 5.0, "w_task": 1.0, "regime": "H", "vmax": 0.03,
            "seed": 0, "out_dir": name, "steps": 300, "task": "capacity"}
    base.update(cfg)
    prov = {"git_sha": sha, "steps_completed": 300, "slurm_job_id": name,
            "device_name": "A100", "torch_version": "2.4.0"}
    return {"dir": _Dir(name), "cfg": base, "prov": prov, "train": [], "evals": []}


def test_the_same_commit_needs_no_waiver_and_records_none():
    notes = []
    runs = [_run("a", "abc123", seed=0), _run("b", "abc123", seed=1)]
    assert A.matched_budget_check(runs, notes=notes) == []
    assert notes == []


def test_a_commit_that_touches_no_training_source_is_waived_and_recorded(monkeypatch):
    monkeypatch.setattr(A, "training_sources_between", lambda a, b: [])
    notes = []
    runs = [_run("a", "aaa", seed=0), _run("b", "bbb", seed=1)]
    assert A.matched_budget_check(runs, notes=notes) == []
    assert len(notes) == 1, "a waived difference must be recorded, not silently dropped"
    assert notes[0]["training_sources_changed"] == []


def test_a_commit_that_touches_a_training_source_is_still_refused(monkeypatch):
    monkeypatch.setattr(
        A, "training_sources_between",
        lambda a, b: ["experiments/cancellation/rewards.py"])
    notes = []
    problems = A.matched_budget_check([_run("a", "aaa"), _run("b", "bbb", seed=1)], notes=notes)
    assert len(problems) == 1
    assert "rewards.py" in problems[0], "the refusal must name the file that changed"
    assert notes == []


def test_git_failing_to_answer_is_a_refusal_and_never_a_pass(monkeypatch):
    """An unanswerable question is not a passed check. Returning None must keep the strict rule,
    because the alternative is that a missing git directory silently waives every difference."""
    monkeypatch.setattr(A, "training_sources_between", lambda a, b: None)
    notes = []
    problems = A.matched_budget_check([_run("a", "aaa"), _run("b", "bbb", seed=1)], notes=notes)
    assert len(problems) == 1
    assert "could not be asked" in problems[0]
    assert notes == []


def test_every_other_provenance_difference_is_untouched_by_the_waiver(monkeypatch):
    monkeypatch.setattr(A, "training_sources_between", lambda a, b: [])
    a, b = _run("a", "aaa"), _run("b", "aaa", seed=1)
    b["prov"]["device_name"] = "H100"
    problems = A.matched_budget_check([a, b], notes=[])
    assert len(problems) == 1 and "device_name" in problems[0]


def test_the_training_source_list_names_files_that_exist():
    """A path typo would make the diff match nothing, so every commit would look harmless.

    The paths are repo-relative because git needs them that way. They are checked here against
    the directory the trainer actually imports from, so this passes both in the repo and in the
    flattened code/ layout the supplement ships.
    """
    prefix = "experiments/cancellation/"
    missing = [f for f in A.TRAINING_SOURCES
               if not os.path.exists(os.path.join(HERE, f[len(prefix):]
                                                  if f.startswith(prefix) else f))]
    assert missing == [], f"declared training sources that do not exist: {missing}"


def test_every_training_source_path_is_repo_relative():
    """git diff resolves these against the repo root, so an absolute or bare path would silently
    match nothing and waive every commit."""
    bad = [f for f in A.TRAINING_SOURCES
           if f.startswith("/") or not f.startswith("experiments/cancellation/")]
    assert bad == [], f"paths git cannot resolve from the repo root: {bad}"


def test_the_training_source_list_covers_the_modules_the_trainer_imports():
    """If a new module joins the training path and nobody adds it here, a commit changing it
    would be waived. This fails loudly when that happens."""
    src = open(os.path.join(HERE, "train_speed_grpo.py")).read()
    declared = {os.path.basename(f)[:-3] for f in A.TRAINING_SOURCES if f.endswith(".py")}
    imported = set()
    for line in src.splitlines():
        line = line.strip()
        for mod in ("rewards", "tasks", "speed_task", "capacity_task"):
            if line.startswith(f"import {mod}") or line.startswith(f"from {mod} "):
                imported.add(mod)
    assert imported <= declared, f"trainer imports {imported - declared}, which is not declared"


@pytest.mark.parametrize("sha", ["abc", ""])
def test_identical_shas_short_circuit_without_calling_git(sha, monkeypatch):
    def boom(a, b):
        raise AssertionError("git must not be consulted when the commits are identical")
    monkeypatch.setattr(A, "training_sources_between", boom)
    assert A.matched_budget_check([_run("a", sha), _run("b", sha, seed=1)], notes=[]) == []


def test_the_manifests_are_training_sources():
    """A manifest row sets a run's arm, weights, limit and seed. Leaving the manifests out let a
    commit that edited a row count as touching no training source at all."""
    assert "experiments/cancellation/slurm/manifest.csv" in A.TRAINING_SOURCES
    assert "experiments/cancellation/slurm/manifest_capacity.csv" in A.TRAINING_SOURCES


def test_a_manifest_that_only_gained_rows_does_not_refuse_the_comparison(monkeypatch):
    """Appending the easy-level baseline cannot change a run built from an earlier row. Anything
    other than an append must still refuse."""
    monkeypatch.setattr(A, "_appended_only", lambda a, b, path: True)
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: type(
        "R", (), {"returncode": 0, "stdout": "experiments/cancellation/slurm/manifest_capacity.csv\n"})())
    assert A.training_sources_between("aaa", "bbb") == []


def test_a_manifest_row_that_was_edited_still_refuses(monkeypatch):
    monkeypatch.setattr(A, "_appended_only", lambda a, b, path: False)
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: type(
        "R", (), {"returncode": 0, "stdout": "experiments/cancellation/slurm/manifest_capacity.csv\n"})())
    assert A.training_sources_between("aaa", "bbb") == [
        "experiments/cancellation/slurm/manifest_capacity.csv"]


def test_the_append_only_exemption_applies_to_manifests_and_not_to_code(monkeypatch):
    """A .py file is never exempt, however its diff happens to look."""
    monkeypatch.setattr(A, "_appended_only", lambda a, b, path: True)
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: type(
        "R", (), {"returncode": 0, "stdout": "experiments/cancellation/rewards.py\n"})())
    assert A.training_sources_between("aaa", "bbb") == ["experiments/cancellation/rewards.py"]


def test_append_only_says_no_when_git_cannot_answer(monkeypatch):
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: type(
        "R", (), {"returncode": 128, "stdout": ""})())
    assert A._appended_only("aaa", "bbb", "any/path.csv") is False


# ---------------------------------------------------------------- the yoked-control commit

YOKED_FILES = ["experiments/cancellation/configs/keep_schedules.json",
               "experiments/cancellation/slurm/manifest_speed_yoked.csv",
               "experiments/cancellation/slurm/manifest_capacity_yoked.csv"]


def test_the_yoked_control_files_are_training_sources():
    """The schedule file and the two manifests decide what a control run does, so a commit that
    edits any of them must count as a training-source change."""
    import train_speed_grpo as T
    for f in YOKED_FILES:
        assert f in A.TRAINING_SOURCES
    assert tuple(A.TRAINING_SOURCES) == tuple(T.TRAINING_SOURCES)


def _fingerprint_dir(d, answer=10.0):
    import json
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"task": "speed", "arm": "binary", "w": 5.0, "vmax": 5.0,
                                               "seed": 0, "steps": 2}))
    (d / "provenance.json").write_text(json.dumps({"final_param_checksum": "c", "steps_completed": 2}))
    group = {"speeds": [answer, 3.0, None, 900.0], "rewards": [1.0, 2.0, 0.0, -4.0],
             "adv": [0.1, 0.9, -0.2, -0.8], "all_violating": False, "has_violator": True}
    lines = [{"step": 0, "phase": "eval", "eval_speeds": [1.0, 2.0]}]
    lines += [{"step": t, "phase": "train", "groups": [group]} for t in (1, 2)]
    lines += [{"step": 2, "phase": "eval", "eval_speeds": [1.0, 2.0]}]
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    (d / "DONE").write_text("ok\n")


def _yoked_pair(monkeypatch):
    changed = ["experiments/cancellation/train_speed_grpo.py"] + YOKED_FILES

    def between(a, b):
        if a == b:
            return []
        return changed
    monkeypatch.setattr(A, "training_sources_between", between)
    new = _run("control", "head_g", seed=0, keep_schedule="capacity_w5_s0_d0")
    new["prov"]["training_sources_digest"] = "NEW"
    old = _run("filter", "COMMIT_4", seed=0, group_filter="verdict_mixed")
    old["prov"]["training_sources_digest"] = "d5ea8063"
    return [new, old]


def _waiver(outcome="pass"):
    return [{"base_sha": "COMMIT_4", "new_sha": "head_g", "new_training_sources_digest": "NEW",
             "fingerprints": [{"name": "G1b", "reference": "runs/ref", "new": "runs/new",
                               "rerun_at_base": "runs/base", "steps": 2, "outcome": outcome}]}]


def test_a_run_at_the_yoked_commit_is_refused_beside_an_old_run_without_a_waiver(monkeypatch):
    problems = A.matched_budget_check(_yoked_pair(monkeypatch), notes=[], equivalence=[])
    assert len(problems) == 1
    assert all(f.rsplit("/", 1)[1] in problems[0] for f in YOKED_FILES)


def test_a_run_at_the_yoked_commit_is_admitted_by_a_waiver_that_reverifies(tmp_path, monkeypatch):
    for name in ("ref", "new", "base"):
        _fingerprint_dir(tmp_path / "runs" / name)
    monkeypatch.setattr(A, "_REPO_ROOT", tmp_path)
    runs = _yoked_pair(monkeypatch)
    notes = []
    assert A.matched_budget_check(runs, notes=notes, equivalence=_waiver()) == []
    assert notes[0]["fingerprint_waiver"]["outcomes"] == {"G1b": "pass"}
    # A control whose digest is not the waiver's is refused, and so is a waiver whose recorded
    # outcome the logs no longer give.
    runs[0]["prov"]["training_sources_digest"] = "d5ea8063"
    problems = A.matched_budget_check(runs, notes=[], equivalence=_waiver())
    assert problems and "training_sources_digest" in problems[0]
    runs[0]["prov"]["training_sources_digest"] = "NEW"
    (tmp_path / "runs" / "new" / "metrics.jsonl").write_text(
        (tmp_path / "runs" / "new" / "metrics.jsonl").read_text().replace("10.0", "11.0"))
    problems = A.matched_budget_check(runs, notes=[], equivalence=_waiver())
    assert problems and "the logs give" in problems[0]
