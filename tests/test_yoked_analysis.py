"""These tests check the gates and the report of analyze_yoked.py on synthetic pairs of filter and control runs.

Y1 asks whether a control run stepped and kept exactly as its schedule says. Y2 asks whether it equals
its filter run through the step at which their kept sets first differ. These tests build pairs where
each gate should hold or fail, including a control that deviates from its schedule at one step and a
speed control whose answers leave the filter run's answers inside the shared window.
"""
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyze_yoked as Y  # noqa: E402

STEPS = 5
# No gate output may print this value, because it is an evaluation result.
SENTINEL_RATE = 0.123456789


def _items(task, step, pi, salt):
    if task == "speed":
        return [float((step * 7 + pi * 3 + k + salt) % 50) for k in range(4)]
    return [[pi % 6, (step + k + salt) % 16] if k < 3 else None for k in range(4)]


def _log(d, *, task="speed", seed=0, kept_sets, control, salt_from=None, salt=13,
         checksum="c", reward_bump_step=None, keep_schedule=None):
    """Write one run. Steps at or after `salt_from` sample other answers, the way a run does once
    its weights differ from another run's."""
    d.mkdir(parents=True)
    cfg = {"task": task, "arm": "binary", "w": 5.0, "w_task": 1.0, "seed": seed, "steps": STEPS,
           "regime": "A" if task == "speed" else "H", "vmax": 5.0 if task == "speed" else 0.03,
           "adv_variant": "grpo", "group_filter": "none" if control else "verdict_mixed",
           "kl_beta": 0.0, "batch_prompts": 6}
    if keep_schedule:
        cfg["keep_schedule"] = keep_schedule
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "provenance.json").write_text(json.dumps({"final_param_checksum": checksum,
                                                   "steps_completed": STEPS}))
    key = "speeds" if task == "speed" else "selections"
    eval_key = "eval_speeds" if task == "speed" else "eval_selections"
    lines = [{"step": 0, "phase": "eval", "eval_violation_rate": SENTINEL_RATE,
              "eval_parse_fail_rate": 0.0, eval_key: [1.0, 2.0]}]
    for t in range(1, STEPS + 1):
        s = salt if salt_from is not None and t >= salt_from else 0
        want = kept_sets[t - 1]
        groups = []
        for pi in range(6):
            items = _items(task, t, pi, s)
            rewards = [float(i) for i in range(4)]
            if reward_bump_step == t and pi == 0:
                rewards[0] += 1.0
            groups.append({key: items, "rewards": rewards, "adv": [-1.2, -0.4, 0.4, 1.2],
                           "all_violating": False, "has_violator": pi % 2 == 0,
                           "kept": pi in want, "prompt_index": pi, "n_tokens": [2, 3, 2, 3]})
        rec = {"step": t, "phase": "train", "groups": groups, "opt_step_taken": bool(want),
               "n_kept_groups": len(want), "grad_norm_preclip": 5.0 if want else None}
        if control:
            rec["keep_schedule_n"] = len(want)
        lines.append(rec)
    lines.append({"step": STEPS, "phase": "eval", "eval_violation_rate": SENTINEL_RATE,
                  "eval_parse_fail_rate": 0.0, eval_key: [1.0, 2.0]})
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    (d / "DONE").write_text("ok\n")
    return d


def _pair(tmp_path, monkeypatch, *, source_sets, control_sets, logged_sets=None, task="speed",
          salt_from=None, checksum="c", reward_bump_step=None):
    """Write a source run, a control run and the schedule file that ties them in a temporary repo."""
    sid = f"{task}_w5_s0_d1"
    src = _log(tmp_path / "runs" / "src", task=task, kept_sets=source_sets, control=False)
    ctrl = _log(tmp_path / "assembled" / "ctrl", task=task, kept_sets=logged_sets or control_sets,
                control=True, salt_from=salt_from, checksum=checksum,
                reward_bump_step=reward_bump_step, keep_schedule=sid)
    entries = {sid: {"kept_sets": control_sets, "source_kept_sets": source_sets,
                     "source_run": "runs/src",
                     "source_metrics_sha256": hashlib.sha256((src / "metrics.jsonl").read_bytes()).hexdigest()}}
    path = tmp_path / "keep_schedules.json"
    path.write_text(json.dumps({"schedules": entries}))
    monkeypatch.setattr(Y, "REPO_ROOT", tmp_path)
    return ctrl, entries, path


def _gate(tmp_path, monkeypatch, **kw):
    ctrl, entries, _ = _pair(tmp_path, monkeypatch, **kw)
    control = Y.analyze._load_run(ctrl)
    rows, pairs, problems = Y.gates([control], entries)
    assert problems == []
    return rows[0]


SRC = [[4], [2], [], [0, 1], []]


def test_a_full_copy_passes_both_gates(tmp_path, monkeypatch):
    row = _gate(tmp_path, monkeypatch, source_sets=SRC, control_sets=SRC)
    assert row["y1_pass"] and row["y2_kind"] == "full_copy" and row["y2_result"] == "pass"
    assert row["y2_compared_through_step"] == STEPS and row["gates_hold"]


def test_a_full_copy_with_another_checksum_fails_on_the_speed_task(tmp_path, monkeypatch):
    row = _gate(tmp_path, monkeypatch, source_sets=SRC, control_sets=SRC, checksum="other")
    assert row["y1_pass"] and row["y2_result"] == "fail" and not row["gates_hold"]


def test_a_distinct_first_update_is_compared_through_that_step_only(tmp_path, monkeypatch):
    ctrl = [[3], [2], [], [0, 5], []]
    row = _gate(tmp_path, monkeypatch, source_sets=SRC, control_sets=ctrl, salt_from=2)
    assert row["y2_kind"] == "distinct_first_update" and row["y2_compared_through_step"] == 1
    assert row["y2_result"] == "pass" and row["gates_hold"]


def test_a_first_update_copy_is_compared_through_its_second_update_step(tmp_path, monkeypatch):
    ctrl = [[4], [5], [], [0, 1], []]
    row = _gate(tmp_path, monkeypatch, source_sets=SRC, control_sets=ctrl, salt_from=3)
    assert row["y2_kind"] == "first_update_copy" and row["y2_compared_through_step"] == 2
    assert row["y2_result"] == "pass"
    # The same control fails when its answers leave the source's at the second update step, which
    # both runs sample from the same weights.
    row = _gate(tmp_path / "b", monkeypatch, source_sets=SRC, control_sets=ctrl, salt_from=2)
    assert row["y2_result"] == "fail" and row["y2_first_difference"] == 2


def test_a_control_that_deviates_from_its_schedule_at_one_step_fails_y1(tmp_path, monkeypatch):
    ctrl = [[3], [2], [], [0, 5], []]
    logged = [[3], [2], [1], [0, 5], []]
    row = _gate(tmp_path, monkeypatch, source_sets=SRC, control_sets=ctrl, logged_sets=logged,
                salt_from=2)
    assert not row["y1_pass"] and row["y1_first_failure_step"] == 3 and not row["gates_hold"]
    assert "kept" in row["y1_reason"] and "opt_step_taken" in row["y1_reason"]


def test_a_speed_control_whose_shared_window_answers_differ_fails_y2(tmp_path, monkeypatch):
    ctrl = [[3], [2], [], [0, 5], []]
    row = _gate(tmp_path, monkeypatch, source_sets=SRC, control_sets=ctrl, salt_from=1)
    assert row["y1_pass"] and row["y2_result"] == "fail" and row["y2_first_difference"] == 1
    assert not row["gates_hold"]


def test_a_capacity_difference_is_drift_only_without_a_logic_break(tmp_path, monkeypatch):
    ctrl = [[3], [2], [], [0, 5], []]
    row = _gate(tmp_path, monkeypatch, task="capacity", source_sets=SRC, control_sets=ctrl, salt_from=1)
    assert row["y2_result"] == "drift" and row["gates_hold"]
    row = _gate(tmp_path / "b", monkeypatch, task="capacity", source_sets=SRC, control_sets=ctrl,
                reward_bump_step=4)
    assert row["y2_result"] == "pass", "the bump lies after the shared window"
    row = _gate(tmp_path / "c", monkeypatch, task="capacity", source_sets=SRC, control_sets=ctrl,
                salt_from=1, reward_bump_step=1)
    assert row["y2_result"] == "drift", "the bumped step samples other answers, so it is not a break"
    row = _gate(tmp_path / "d", monkeypatch, task="capacity", source_sets=SRC, control_sets=SRC,
                reward_bump_step=4)
    assert row["y2_result"] == "fail", "equal answers with other rewards are a logic break"


def test_gates_only_prints_the_gates_and_no_evaluation_value(tmp_path, monkeypatch, capsys):
    _, _, path = _pair(tmp_path, monkeypatch, source_sets=SRC, control_sets=SRC)
    rc = Y.main(["--gates-only", "--runs", str(tmp_path / "assembled"), "--schedules", str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"y1_pass": true' in out and '"gates_hold": true' in out
    assert str(SENTINEL_RATE) not in out and "eval_violation_rate" not in out


def test_the_full_report_is_written_only_when_both_gates_hold(tmp_path, monkeypatch):
    ctrl = [[3], [2], [], [0, 5], []]
    _, _, path = _pair(tmp_path, monkeypatch, source_sets=SRC, control_sets=ctrl, salt_from=2)
    out = tmp_path / "report"
    assert Y.main(["--runs", str(tmp_path / "assembled"), "--schedules", str(path), "--out", str(out)]) == 0
    rows = {r["role"]: r for r in __import__("csv").DictReader((out / "per_run.csv").open())}
    assert set(rows) == {"control", "source"}
    c = rows["control"]
    assert c["update_steps"] == "1 2 4" and c["retained_groups"] == "4"
    assert c["retained_completions"] == "16" and c["retained_tokens"] == "40"
    assert c["kept_per_prompt"] == "0:1 1:0 2:1 3:1 4:0 5:1"
    assert c["first_update_keeps_mixed_group"] == "False"
    assert rows["source"]["first_update_keeps_mixed_group"] == "True"
    assert json.loads((out / "kept_sets.json").read_text())["ctrl"] == {"1": [3], "2": [2], "4": [0, 5]}

    bad = tmp_path / "bad"
    _, _, path = _pair(bad, monkeypatch, source_sets=SRC, control_sets=ctrl, salt_from=1)
    out = bad / "report"
    rc = Y.main(["--runs", str(bad / "assembled"), "--schedules", str(path), "--out", str(out)])
    assert rc == Y.GATE_FAILED
    assert sorted(p.name for p in out.iterdir()) == ["gates.json"]


def test_a_changed_source_log_is_refused(tmp_path, monkeypatch):
    _, _, path = _pair(tmp_path, monkeypatch, source_sets=SRC, control_sets=SRC)
    log = tmp_path / "runs" / "src" / "metrics.jsonl"
    log.write_text(log.read_text() + "\n")
    assert Y.main(["--gates-only", "--runs", str(tmp_path / "assembled"), "--schedules", str(path)]) == 2


def test_a_source_whose_logged_kept_sets_differ_from_the_schedule_is_refused(tmp_path, monkeypatch):
    ctrl, entries, _ = _pair(tmp_path, monkeypatch, source_sets=SRC, control_sets=SRC)
    entries[next(iter(entries))]["source_kept_sets"] = [[4], [3], [], [0, 1], []]
    with pytest.raises(Y.PairingError):
        Y.pair(Y.analyze._load_run(ctrl), entries)


@pytest.mark.parametrize("field", ["kept", "n_kept_groups", "keep_schedule_n"])
def test_each_y1_field_fails_the_gate_on_its_own(tmp_path, monkeypatch, field):
    """Y1 must catch a control whose kept prompts, kept count or logged schedule size alone leaves
    its schedule at one step, while every other logged field still agrees."""
    ctrl, entries, _ = _pair(tmp_path, monkeypatch, source_sets=SRC, control_sets=SRC)
    control = Y.analyze._load_run(ctrl)
    rec = control["train"][3]
    assert rec["step"] == 4 and SRC[3] == [0, 1]
    if field == "kept":
        for g in rec["groups"]:
            g["kept"] = g["prompt_index"] in (0, 2)
    else:
        rec[field] += 1
    rows, _, problems = Y.gates([control], entries)
    assert problems == []
    row = rows[0]
    assert not row["y1_pass"] and row["y1_first_failure_step"] == 4
    assert row["y1_reason"].split(", ") == [field]
    assert not row["gates_hold"]
