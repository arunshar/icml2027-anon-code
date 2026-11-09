"""The version-2 pilot admission, the D0 score and the D-choice of analyze_reasoning.py (pilot-d0 mode).

Every run here is synthetic and lives in a temporary directory. A run holds the files that a version-2
pilot run of the trainer writes (config.json, provenance.json, metrics.jsonl, DONE, optionally
completions.jsonl), built from the real pilot manifest rows. The tests cover the pre-registered rules
of PREDICTIONS.md (Block D' pilot): D0, the D-choice order and D-early. A last test checks that the
version-1 analysis writes the same bytes as it did before the pilot mode existed.
"""
import contextlib
import hashlib
import io
import json
import random
from pathlib import Path

import pytest

import analyze_reasoning as A
import make_reasoning_manifest as MM
import reasoning_task as RT
from reasoning_fixtures import fake_run

HERE = Path(__file__).resolve().parent.parent
MANIFEST = HERE / "slurm" / "manifest_reasoning_pilot.csv"
FROZEN = {"design_version": 2, "reasoning_sources_digest": "8" * 64, "schema": "wo_i_frozen_v2",
          "training_sources_digest": "6" * 64}
N_RANDOM, K = 200, RT.EVAL_SAMPLES_RANDOM
N_ITEMS = N_RANDOM * K
PIDS = [f"te{i}" for i in range(N_RANDOM)]
JOB = "12345"


def manifest_rows():
    return {int(r["task_id"]): r for r in A.read_manifest(MANIFEST)}


def eval_record(step, n_correct, n_parsed, name="random", k=K):
    """Return a random held-out record. The first n_correct items are correct, the next ones up to
    n_parsed parse but are wrong, and the rest do not parse."""
    assert n_correct <= n_parsed <= N_ITEMS
    items = []
    for i in range(N_ITEMS):
        pid, j = PIDS[i // K], i % K
        if i < n_correct:
            items.append([pid, j, 100, True, 0, "5", "5", 1])
        elif i < n_parsed:
            items.append([pid, j, 100, True, 1, "6", "6", 0])
        else:
            items.append([pid, j, 100, True, 1, None, None, 0])
    got = A._items_summary(items)
    return {"step": step, "phase": "eval", "set": name, "samples": k, "n": N_ITEMS, "accuracy": got["accuracy"],
            "lenient_accuracy": got["accuracy"], "strict_parse_fail_rate": got["strict_parse_fail_rate"],
            "mean_completion_tokens": 100.0, "items": items}


def completion_lines(step, n_correct, n_parsed):
    lines = []
    for i in range(N_ITEMS):
        text = "so \\boxed{5}" if i < n_correct else ("so \\boxed{6}" if i < n_parsed else "no answer")
        lines.append({"step": step, "phase": "eval", "set": "random", "pid": PIDS[i // K], "sample": i % K,
                      "token_ids": [], "text": text})
    return lines


def make_run(root, row, *, first=(174, 360), last=(174, 360), steps_completed=None, stopped_early=False,
             done=True, prov_patch=None, cfg_patch=None, with_completions=False, completions_last=None):
    """Write a pilot run directory as the trainer writes it. first and last are (n_correct, n_parsed)."""
    typed = RT.validate_arm_row(row)
    steps = int(row["steps"])
    sc = steps if steps_completed is None else steps_completed
    d = root / f"{row['task_id']}_pilot_0.5b_verdict_only_s0"
    d.mkdir(parents=True)
    cfg = {"mode": "manifest", "manifest_row": {k: str(v) for k, v in typed.items()}, "task_id": typed["task_id"],
           "kind": "pilot", "stage": typed["stage"], "size": typed["size"], "arm": typed["arm"],
           "seed": typed["seed"], "steps": typed["steps"], "L": typed["L"], "lr": typed["lr"],
           "kl_beta": typed["kl_beta"], "skip_zero_adv": bool(typed["skip_zero_adv"]),
           "fixedref_c": typed["fixedref_c"], "design_version": 2, "tiny": False, "eval_limit": None,
           "device": "cuda"}
    cfg.update(cfg_patch or {})
    prov = {"design_version": 2, "reasoning_sources_digest": FROZEN["reasoning_sources_digest"],
            "training_sources_digest": FROZEN["training_sources_digest"], "frozen_v2": FROZEN,
            "dataset_sha256": row["dataset_sha256"], "selection_sha256": row["selection_sha256"],
            "model_revision": row["model_revision"], "model_snapshot": row["model_revision"],
            "slurm_array_job_id": JOB, "slurm_array_task_id": row["task_id"], "steps_completed": sc,
            "stopped_early": stopped_early}
    prov.update(prov_patch or {})
    lines = [eval_record(0, *first)]
    lines += [{"step": s, "phase": "train", "accuracy": 0.4} for s in range(1, sc + 1)]
    lines.append(eval_record(sc, *last))
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "provenance.json").write_text(json.dumps(prov))
    (d / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    if with_completions:
        comp = completion_lines(0, *first) + completion_lines(sc, *(completions_last or last))
        (d / "completions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in comp))
    if done:
        (d / "DONE").write_text("ok\n")
    return d


def write_logs(logs, task_ids, tracebacks=()):
    logs.mkdir(exist_ok=True)
    for t in task_ids:
        (logs / f"wo-i-arms-v2-{JOB}_{t}.out").write_text('{"event": "done"}\n')
        (logs / f"wo-i-arms-v2-{JOB}_{t}.err").write_text("Traceback (most recent call last):\n" if t in tracebacks else "")


@pytest.fixture()
def pilot(tmp_path):
    """Three passing pilot runs (rows 1 to 3), their logs and a frozen file."""
    rows = manifest_rows()
    runs = tmp_path / "runs" / JOB
    runs.mkdir(parents=True)
    dirs = {t: make_run(runs, rows[t]) for t in (1, 2, 3)}
    write_logs(runs / "logs", (1, 2, 3))
    (tmp_path / "frozen.json").write_text(json.dumps(FROZEN))
    return tmp_path, rows, runs, dirs


def report(tmp, runs, **kw):
    return A.pilot_report(MANIFEST, runs, FROZEN, runs / "logs", **kw)


def row_of(rep, tid):
    return next(e for e in rep["rows"] if e["task_id"] == tid)


def edit_json(path, patch):
    data = json.loads(path.read_text())
    data.update(patch)
    path.write_text(json.dumps(data))


# ------------------------------------------------------------------ the manifest rows


def test_the_pilot_rows_are_named_by_the_one_change_they_make():
    rows = manifest_rows()
    assert [A.pilot_row_kind(rows[t]) for t in range(1, 7)] == ["lr", "skip", "kl", "lr", "kl", "combination"]
    assert [A.pilot_is_first_wave(rows[t]) for t in range(1, 7)] == [True, True, True, False, False, False]


# ------------------------------------------------------------------ D0


def test_a_passing_row_is_admitted_and_passes_d0(pilot):
    tmp, rows, runs, _ = pilot
    rep = report(tmp, runs)
    assert [e["task_id"] for e in rep["rows"]] == [1, 2, 3]
    for e in rep["rows"]:
        assert e["problems"] == [] and e["admitted"] and e["eligible"] and e["passes"]
        assert e["stopped_early"] is False and e["steps_completed"] == 100 and e["complete"]
        d = e["d0"]
        assert d["accuracy_first"] == pytest.approx(0.435) and d["accuracy_last"] == pytest.approx(0.435)
        assert d["signed_difference"] == 0.0 and d["strict_parse_rate_last"] == pytest.approx(0.9)
        assert d["n_first"] == 400 and d["n_last"] == 400 and d["last_step"] == 100
    assert rep["any_rise_above_0_05"] is False


def test_a_row_whose_accuracy_falls_by_more_than_0_05_fails_d0(pilot):
    tmp, rows, runs, dirs = pilot
    shutil_rm(dirs[2])
    make_run(runs, rows[2], first=(174, 360), last=(100, 360))
    e = row_of(report(tmp, runs), 2)
    assert e["problems"] == [] and e["eligible"] and not e["passes"]
    assert e["d0"]["signed_difference"] == pytest.approx(-0.185) and e["d0"]["within_0_05"] is False
    assert e["d0"]["parse_at_least_0_80"] is True and e["d0"]["rise_above_0_05"] is False


def test_a_row_whose_parse_rate_is_below_0_80_fails_d0_and_exactly_0_80_passes(pilot):
    tmp, rows, runs, dirs = pilot
    shutil_rm(dirs[1])
    shutil_rm(dirs[3])
    make_run(runs, rows[1], last=(174, 319))
    make_run(runs, rows[3], last=(174, 320))
    rep = report(tmp, runs)
    low, edge = row_of(rep, 1), row_of(rep, 3)
    assert low["d0"]["strict_parse_rate_last"] == pytest.approx(0.7975) and not low["d0"]["parse_at_least_0_80"]
    assert low["d0"]["within_0_05"] is True and not low["passes"]
    assert edge["d0"]["strict_parse_rate_last"] == pytest.approx(0.8) and edge["passes"]


@pytest.mark.parametrize("correct_last,passes,rise", [
    (194, True, False),   # +0.05 exactly, which is 20 of 400
    (154, True, False),   # -0.05 exactly
    (195, False, True),   # +0.0525
    (153, False, False),  # -0.0525
])
def test_the_0_05_boundary_is_inclusive_and_a_rise_above_it_is_flagged(pilot, correct_last, passes, rise):
    tmp, rows, runs, dirs = pilot
    shutil_rm(dirs[2])
    make_run(runs, rows[2], first=(174, 360), last=(correct_last, 360))
    rep = report(tmp, runs)
    e = row_of(rep, 2)
    assert e["d0"]["absolute_difference"] == pytest.approx(abs(correct_last - 174) / 400)
    assert e["passes"] is passes and e["d0"]["within_0_05"] is passes
    assert e["d0"]["rise_above_0_05"] is rise and rep["any_rise_above_0_05"] is rise


def test_the_boundary_needs_the_float_tolerance():
    # 0.55 - 0.50 is 0.05000000000000004 as a float, so the plain comparison would fail an exact 0.05.
    assert 0.55 - 0.5 > 0.05
    recs = [{"set": "random", "step": 0, "n": 400, "accuracy": 0.5, "strict_parse_fail_rate": 0.0},
            {"set": "random", "step": 100, "n": 400, "accuracy": 0.55, "strict_parse_fail_rate": 0.0}]
    d = A.pilot_d0(recs)
    assert d["within_0_05"] is True and d["rise_above_0_05"] is False and d["passes"] is True
    recs[1]["accuracy"] = 0.5500001
    assert A.pilot_d0(recs)["within_0_05"] is False


def test_a_rise_above_0_05_in_any_row_is_reported_at_the_top(pilot):
    tmp, rows, runs, dirs = pilot
    shutil_rm(dirs[3])
    make_run(runs, rows[3], last=(250, 360))
    rep = report(tmp, runs)
    assert rep["any_rise_above_0_05"] is True and row_of(rep, 3)["d0"]["rise_above_0_05"] is True
    assert not row_of(rep, 3)["passes"] and row_of(rep, 1)["d0"]["rise_above_0_05"] is False


def test_d0_reads_only_the_random_set_at_step_0_and_the_last_step():
    hard = {"set": "hard", "step": 100, "n": 800, "accuracy": 0.0, "strict_parse_fail_rate": 1.0}
    mid = {"set": "random", "step": 50, "n": 400, "accuracy": 0.0, "strict_parse_fail_rate": 1.0}
    recs = [hard, mid, {"set": "random", "step": 0, "n": 400, "accuracy": 0.4, "strict_parse_fail_rate": 0.1},
            {"set": "random", "step": 100, "n": 400, "accuracy": 0.42, "strict_parse_fail_rate": 0.1}]
    d = A.pilot_d0(recs)
    assert d["last_step"] == 100 and d["accuracy_first"] == 0.4 and d["passes"]
    with pytest.raises(ValueError):
        A.pilot_d0(recs[:2])
    with pytest.raises(ValueError):
        A.pilot_d0([recs[2]])


# ------------------------------------------------------------------ D-early and admission


def test_a_stopped_early_row_is_flagged_scored_and_not_called_complete(pilot):
    tmp, rows, runs, dirs = pilot
    shutil_rm(dirs[2])
    make_run(runs, rows[2], steps_completed=40, stopped_early=True)
    rep = report(tmp, runs)
    e = row_of(rep, 2)
    assert e["problems"] == [] and e["stopped_early"] is True and e["steps_completed"] == 40
    assert e["complete"] is False and e["admitted"] is False and e["eligible"] is True
    assert e["d0"]["last_step"] == 40 and e["passes"] is True
    # The row still enters the D-choice, and the report also gives the choice without it.
    assert rep["d_choice"]["chosen_task_id"] == 2
    assert rep["d_choice_complete_rows_only"]["chosen_task_id"] == 3


def test_a_short_run_that_did_not_flag_stopped_early_is_refused(pilot):
    tmp, rows, runs, dirs = pilot
    shutil_rm(dirs[2])
    make_run(runs, rows[2], steps_completed=40, stopped_early=False)
    e = row_of(report(tmp, runs), 2)
    assert any("steps_completed is 40" in p for p in e["problems"]) and not e["eligible"]


def test_a_flag_without_a_short_run_and_a_missing_flag_are_refused(pilot):
    tmp, rows, runs, dirs = pilot
    shutil_rm(dirs[2])
    make_run(runs, rows[2], stopped_early=True)
    assert any("steps_completed is 100" in p for p in row_of(report(tmp, runs), 2)["problems"])
    shutil_rm(dirs[3])
    d = make_run(runs, rows[3])
    prov = json.loads((d / "provenance.json").read_text())
    del prov["stopped_early"]
    (d / "provenance.json").write_text(json.dumps(prov))
    assert any("no stopped_early flag" in p for p in row_of(report(tmp, runs), 3)["problems"])


@pytest.mark.parametrize("patch,words", [
    ({"reasoning_sources_digest": "0" * 64}, "reasoning source digest"),
    ({"training_sources_digest": "0" * 64}, "training source digest"),
    ({"frozen_v2": dict(FROZEN, training_sources_digest="0" * 64)}, "frozen record"),
    ({"design_version": 1}, "design_version"),
    ({"dataset_sha256": "0" * 64}, "dataset or selection"),
    ({"model_snapshot": "main"}, "model revision"),
])
def test_a_provenance_mismatch_refuses_the_row(pilot, patch, words):
    tmp, rows, runs, dirs = pilot
    edit_json(dirs[1] / "provenance.json", patch)
    rep = report(tmp, runs)
    e = row_of(rep, 1)
    assert any(words in p for p in e["problems"]) and not e["admitted"] and not e["eligible"]
    assert not e["passes"] and 1 not in rep["d_choice"]["passing_task_ids"]
    assert row_of(rep, 2)["admitted"]


def test_a_missing_done_marker_refuses_the_row(pilot):
    tmp, rows, runs, dirs = pilot
    (dirs[3] / "DONE").unlink()
    e = row_of(report(tmp, runs), 3)
    assert "no DONE marker" in e["problems"] and not e["admitted"] and not e["eligible"] and not e["passes"]


@pytest.mark.parametrize("patch", [
    {"lr": 1e-5}, {"kl_beta": 0.02}, {"skip_zero_adv": True}, {"seed": 1}, {"steps": 50}, {"design_version": 1},
    {"arm": "plain"}, {"L": 384}, {"tiny": True}, {"eval_limit": 3}, {"kind": "study"},
])
def test_a_configuration_that_differs_from_its_manifest_row_refuses_the_row(pilot, patch):
    tmp, rows, runs, dirs = pilot
    # Row 1 moves the learning rate alone, so the edits above each move one field off its row.
    edit_json(dirs[1] / "config.json", patch)
    rep = report(tmp, runs)
    if patch == {"kind": "study"}:
        # The run no longer identifies itself as a pilot, so no run directory matches row 1.
        assert rep["missing_first_wave_task_ids"] == [1]
        return
    e = row_of(rep, 1)
    assert e["problems"] and all("config.json" in p for p in e["problems"]) and not e["eligible"]


def test_the_stored_manifest_row_must_equal_the_manifest_row(pilot):
    tmp, rows, runs, dirs = pilot
    cfg = json.loads((dirs[2] / "config.json").read_text())
    cfg["manifest_row"]["L"] = "384"
    (dirs[2] / "config.json").write_text(json.dumps(cfg))
    assert "the configuration differs from its manifest row in L" in row_of(report(tmp, runs), 2)["problems"]


def test_skip_zero_adv_may_be_stored_as_a_bool_or_as_an_integer(pilot):
    tmp, rows, runs, dirs = pilot
    edit_json(dirs[2] / "config.json", {"skip_zero_adv": 1})
    assert row_of(report(tmp, runs), 2)["problems"] == []
    edit_json(dirs[2] / "config.json", {"skip_zero_adv": "1"})
    assert row_of(report(tmp, runs), 2)["problems"]


def test_logs_with_refused_or_traceback_or_no_log_refuse_the_row(pilot):
    tmp, rows, runs, dirs = pilot
    logs = runs / "logs"
    (logs / f"wo-i-arms-v2-{JOB}_1.err").write_text("Traceback (most recent call last):\n")
    (logs / f"wo-i-arms-v2-{JOB}_2.err").write_text("REFUSED: the run directory already holds files.\n")
    for p in logs.glob(f"*-{JOB}_3.*"):
        p.unlink()
    rep = report(tmp, runs)
    assert "the logs hold Traceback" in row_of(rep, 1)["problems"]
    assert "the logs hold REFUSED" in row_of(rep, 2)["problems"]
    assert "no Slurm log was found for this run" in row_of(rep, 3)["problems"]
    assert rep["d_choice"]["chosen_task_id"] is None


def test_a_log_hit_that_no_row_owns_is_listed_and_does_not_refuse_a_row(pilot):
    tmp, rows, runs, dirs = pilot
    (runs / "logs" / "wo-i-arms-v2-5555555_9.err").write_text("REFUSED: earlier attempt.\n")
    rep = report(tmp, runs)
    assert rep["log_files_with_REFUSED_or_Traceback_not_matched_to_a_row"] == ["wo-i-arms-v2-5555555_9.err"]
    assert all(e["admitted"] for e in rep["rows"])


def test_a_log_with_other_words_such_as_a_time_limit_is_not_a_refused_or_traceback_hit(pilot):
    tmp, rows, runs, dirs = pilot
    (runs / "logs" / f"wo-i-arms-v2-{JOB}_1.err").write_text("slurmstepd: DUE TO TIME LIMIT\n")
    assert row_of(report(tmp, runs), 1)["problems"] == []


def test_a_missing_evaluation_or_a_record_that_disagrees_with_its_items_refuses_the_row(pilot):
    tmp, rows, runs, dirs = pilot
    lines = (dirs[1] / "metrics.jsonl").read_text().splitlines()
    (dirs[1] / "metrics.jsonl").write_text("".join(x + "\n" for x in lines[:-1]))
    assert any("not exactly at steps 0 and 100" in p for p in row_of(report(tmp, runs), 1)["problems"])
    lines = [json.loads(x) for x in (dirs[2] / "metrics.jsonl").read_text().splitlines()]
    lines[-1]["accuracy"] = 0.9
    (dirs[2] / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert any("accuracy that its items do not give" in p for p in row_of(report(tmp, runs), 2)["problems"])
    lines = [json.loads(x) for x in (dirs[3] / "metrics.jsonl").read_text().splitlines()]
    lines = [x for x in lines if not (x["phase"] == "train" and x["step"] == 7)]
    (dirs[3] / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert any("every completed step" in p for p in row_of(report(tmp, runs), 3)["problems"])


# ------------------------------------------------------------------ the recheck from completions


def tiny_dataset():
    return {"train_candidates": [], "test_candidates": [{"pid": p, "reference": "5"} for p in PIDS]}


def test_the_recheck_from_completions_agrees_with_the_record_and_catches_a_difference(tmp_path):
    rows = manifest_rows()
    runs = tmp_path / "runs"
    runs.mkdir()
    make_run(runs, rows[2], first=(174, 360), last=(180, 330), with_completions=True)
    write_logs(runs / "logs", (2,))
    sel = {"random": PIDS}
    rep = A.pilot_report(MANIFEST, runs, FROZEN, runs / "logs", tiny_dataset(), sel)
    e = row_of(rep, 2)
    assert e["problems"] == [] and e["recheck"]["agrees"] is True
    assert e["recheck"]["steps"][100]["accuracy"] == pytest.approx(0.45)
    shutil_rm(next(runs.glob("2_pilot_*")))
    make_run(runs, rows[2], first=(174, 360), last=(180, 330), with_completions=True, completions_last=(100, 330))
    e = row_of(A.pilot_report(MANIFEST, runs, FROZEN, runs / "logs", tiny_dataset(), sel), 2)
    assert e["recheck"]["agrees"] is False and not e["eligible"] and any("completions" in p for p in e["problems"])


# ------------------------------------------------------------------ the D-choice


def cand(tid, lr, kl, skip, passes=True):
    return {"task_id": tid, "lr": lr, "kl_beta": kl, "skip_zero_adv": skip, "passes": passes}


SKIP, KL, LR = cand(2, 1e-5, 0.0, 1), cand(3, 1e-5, 0.02, 0), cand(1, 1e-6, 0.0, 0)


def test_the_d_choice_takes_the_skip_then_the_kl_then_the_learning_rate():
    assert A.pilot_d_choice([LR, KL, SKIP])["chosen_task_id"] == 2
    assert A.pilot_d_choice([LR, KL, SKIP])["chosen_change"] == "skip"
    assert A.pilot_d_choice([LR, KL])["chosen_task_id"] == 3 and A.pilot_d_choice([LR, KL])["chosen_change"] == "kl"
    assert A.pilot_d_choice([LR])["chosen_task_id"] == 1 and A.pilot_d_choice([SKIP, LR])["chosen_task_id"] == 2
    assert A.pilot_d_choice([KL, SKIP])["passing_task_ids"] == [2, 3]


def test_the_d_choice_ignores_rows_that_do_not_pass_and_chooses_nothing_when_none_does():
    out = A.pilot_d_choice([dict(SKIP, passes=False), KL, dict(LR, passes=False)])
    assert out["chosen_task_id"] == 3 and out["passing_task_ids"] == [3]
    none = A.pilot_d_choice([dict(SKIP, passes=False), dict(KL, passes=False), dict(LR, passes=False)])
    assert none["chosen_task_id"] is None and none["chosen_change"] is None and none["passing_task_ids"] == []
    assert A.pilot_d_choice([])["chosen_task_id"] is None


def test_the_d_choice_lists_the_combination_and_ties_without_choosing_the_combination():
    combo = cand(6, 1e-5, 0.02, 1)
    out = A.pilot_d_choice([combo])
    assert out["chosen_task_id"] is None and out["passing_task_ids"] == [6] and out["passing_outside_the_order"] == [6]
    assert A.pilot_d_choice([combo, LR])["chosen_task_id"] == 1
    lr2 = cand(4, 5e-7, 0.0, 0)
    tie = A.pilot_d_choice([lr2, LR])
    assert tie["chosen_task_id"] == 1 and tie["tied_task_ids"] == [1, 4]


def test_the_report_applies_the_order_to_the_rows_that_pass_d0(pilot):
    tmp, rows, runs, dirs = pilot
    rep = report(tmp, runs)
    assert rep["d_choice"]["chosen_task_id"] == 2 and rep["d_choice"]["passing_task_ids"] == [1, 2, 3]
    shutil_rm(dirs[2])
    make_run(runs, rows[2], last=(100, 360))
    assert report(tmp, runs)["d_choice"]["chosen_task_id"] == 3
    shutil_rm(dirs[3])
    make_run(runs, rows[3], last=(100, 360))
    assert report(tmp, runs)["d_choice"]["chosen_task_id"] == 1
    shutil_rm(dirs[1])
    make_run(runs, rows[1], last=(100, 360))
    none = report(tmp, runs)["d_choice"]
    assert none["chosen_task_id"] is None and none["passing_task_ids"] == []


def test_a_row_that_fails_admission_never_enters_the_d_choice_even_if_its_d0_would_pass(pilot):
    tmp, rows, runs, dirs = pilot
    (dirs[2] / "DONE").unlink()
    e = row_of(report(tmp, runs), 2)
    assert e["d0"]["passes"] is True and e["passes"] is False
    assert report(tmp, runs)["d_choice"]["chosen_task_id"] == 3


# ------------------------------------------------------------------ rows that were not run, and the command line


def test_a_retry_row_without_a_run_is_left_out_and_a_first_wave_row_without_one_is_named(pilot):
    tmp, rows, runs, dirs = pilot
    rep = report(tmp, runs)
    assert [e["task_id"] for e in rep["rows"]] == [1, 2, 3] and rep["missing_first_wave_task_ids"] == []
    text = A.pilot_table(rep)
    assert "retry" not in text.lower() and not any(line.startswith(("4\t", "5\t", "6\t")) for line in text.splitlines())
    assert {e["lr"] for e in rep["rows"]} == {1e-6, 1e-5} and {e["kl_beta"] for e in rep["rows"]} == {0.0, 0.02}
    shutil_rm(dirs[3])
    assert report(tmp, runs)["missing_first_wave_task_ids"] == [3]


def test_a_retry_row_that_was_run_is_scored_like_any_other(pilot):
    tmp, rows, runs, dirs = pilot
    make_run(runs, rows[4])
    write_logs(runs / "logs", (4,))
    rep = report(tmp, runs)
    assert row_of(rep, 4)["change"] == "lr" and row_of(rep, 4)["passes"]


def test_the_command_line_mode_writes_the_json_and_the_table(pilot, capsys):
    tmp, rows, runs, dirs = pilot
    out = tmp / "out"
    rc = A.main(["pilot-d0", "--manifest", str(MANIFEST), "--frozen", str(tmp / "frozen.json"), "--runs", str(runs),
                 "--logs", str(runs / "logs"), "--out-dir", str(out)])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["d_choice_task_id"] == 2
    rep = json.loads((out / "pilot_d0.json").read_text())
    assert [e["task_id"] for e in rep["rows"]] == [1, 2, 3] and rep["rule"]["accuracy_tolerance"] == 0.05
    table = (out / "pilot_d0.txt").read_text().splitlines()
    assert table[0].startswith("task\tchange\tadmitted") and len(table) >= 5
    assert any(line.startswith("D-choice: task 2 (skip row)") for line in table)
    assert table[1].split("\t")[-1] == "PASS"


def test_the_command_line_refuses_a_frozen_file_that_is_not_a_version_2_file(pilot, capsys):
    tmp, rows, runs, dirs = pilot
    (tmp / "bad.json").write_text(json.dumps({"reasoning_sources_digest": "8" * 64}))
    rc = A.main(["pilot-d0", "--manifest", str(MANIFEST), "--frozen", str(tmp / "bad.json"), "--runs", str(runs),
                 "--logs", str(runs / "logs"), "--out-dir", str(tmp / "out")])
    assert rc == 3 and "REFUSED" in capsys.readouterr().err and not (tmp / "out").exists()


def test_the_command_line_mode_with_a_dataset_rechecks_from_completions(tmp_path, capsys):
    rows = manifest_rows()
    runs = tmp_path / "runs"
    runs.mkdir()
    make_run(runs, rows[2], with_completions=True)
    write_logs(runs / "logs", (2,))
    (tmp_path / "frozen.json").write_text(json.dumps(FROZEN))
    (tmp_path / "ds.json").write_text(json.dumps(tiny_dataset()))
    (tmp_path / "sel.json").write_text(json.dumps({"random": PIDS}))
    rc = A.main(["pilot-d0", "--manifest", str(MANIFEST), "--frozen", str(tmp_path / "frozen.json"), "--runs", str(runs),
                 "--logs", str(runs / "logs"), "--out-dir", str(tmp_path / "out"), "--dataset", str(tmp_path / "ds.json"),
                 "--selection", str(tmp_path / "sel.json")])
    assert rc == 0
    assert json.loads((tmp_path / "out" / "pilot_d0.json").read_text())["rows"][0]["recheck"]["agrees"] is True
    with pytest.raises(SystemExit):
        A.main(["pilot-d0", "--manifest", str(MANIFEST), "--frozen", "f", "--runs", "r", "--logs", "l",
                "--out-dir", "o", "--dataset", "d"])


def shutil_rm(path):
    import shutil
    shutil.rmtree(path)


# ------------------------------------------------------------------ the version-1 analysis is unchanged

# These digests were taken from the analysis before the pilot mode existed, on the study fixture below.
V1_REPORT_SHA = "77bc36b19176e0ee6eab8563df0a919cc425f9fe0d31f18c087cb26a67a36f4f"
V1_CURVES_SHA = "c1c217447b120a10d930614d94d5304d4c385bbbc925dd902403e43c048ada67"
V1_STDOUT_SHA = "dd2555f9a89dedfe896259c11945d17062f4e4de511e13b2e085388b53495bb4"


def test_the_version_1_analysis_writes_the_same_bytes_as_before_the_pilot_mode(tmp_path):
    ds, sel_sha, src = "d" * 64, "e" * 64, "f" * 64
    selection = {"candidates_sha256": ds, "L": 384,
                 "sizes": {"0.5b": {"status": "ok", "model_revision": RT.MODELS["0.5b"]["revision"], "short_threshold": 50.0},
                           "1.5b": {"status": "ok", "model_revision": RT.MODELS["1.5b"]["revision"], "short_threshold": 60.0}}}
    rows = MM.study_rows(selection, {"selection_sha256": sel_sha, "S": 150, "sizes_kept": ["0.5b", "1.5b"]}, ds, sel_sha)
    (tmp_path / "m.csv").write_bytes(MM.to_csv(RT.ARM_HEADER, rows))
    rng = random.Random(0)
    runs = tmp_path / "runs" / "900"
    runs.mkdir(parents=True)
    logs = tmp_path / "logs"
    logs.mkdir()
    for row in rows:
        fake_run(runs, row, rng, src, acc_shift=0.2 if row["arm"] == "gate" else 0.0)
        (logs / f"wo-i-s1-900_{row['task_id']}.out").write_text("ok\n")
    (tmp_path / "frozen.json").write_text(json.dumps({"reasoning_sources_digest": src}))
    (tmp_path / "sel.json").write_text(json.dumps(selection))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = A.main(["--manifest", str(tmp_path / "m.csv"), "--runs", str(tmp_path / "runs"),
                     "--frozen", str(tmp_path / "frozen.json"), "--selection", str(tmp_path / "sel.json"),
                     "--logs", str(logs), "--out", str(tmp_path / "o.json"), "--curves", str(tmp_path / "c.json")])
    assert rc == 0
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    assert sha((tmp_path / "o.json").read_bytes()) == V1_REPORT_SHA
    assert sha((tmp_path / "c.json").read_text().replace(str(tmp_path), "TMP").encode()) == V1_CURVES_SHA
    assert sha(out.getvalue().replace(str(tmp_path), "TMP").encode()) == V1_STDOUT_SHA


def test_the_version_1_help_does_not_mention_the_pilot_mode(capsys):
    with pytest.raises(SystemExit):
        A.main(["--help"])
    assert "pilot" not in capsys.readouterr().out
