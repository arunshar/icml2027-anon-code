"""These tests run the WO-I analysis on synthetic logs and check admission, W0, the counters and the labels."""
import json
import random
from pathlib import Path

import pytest

import analyze_reasoning as A
import make_reasoning_manifest as MM
import reasoning_task as RT
from reasoning_fixtures import fake_run

DS, SEL_SHA, SRC = "d" * 64, "e" * 64, "f" * 64
SELECTION = {"candidates_sha256": DS, "L": 384,
             "sizes": {"0.5b": {"status": "ok", "model_revision": RT.MODELS["0.5b"]["revision"], "short_threshold": 50.0},
                       "1.5b": {"status": "ok", "model_revision": RT.MODELS["1.5b"]["revision"], "short_threshold": 60.0}}}


def write_manifest(path: Path, kept=("0.5b", "1.5b")) -> list:
    rows = MM.study_rows(SELECTION, {"selection_sha256": SEL_SHA, "S": 150, "sizes_kept": list(kept)}, DS, SEL_SHA)
    path.write_bytes(MM.to_csv(RT.ARM_HEADER, rows))
    return rows


@pytest.fixture()
def study(tmp_path):
    rows = write_manifest(tmp_path / "m.csv")
    rng = random.Random(0)
    runs = tmp_path / "runs" / "900"
    runs.mkdir(parents=True)
    for row in rows:
        fake_run(runs, row, rng, SRC, acc_shift=0.2 if row["arm"] == "gate" else 0.0)
    return tmp_path, rows


def analyze(tmp, **kw):
    return A.analyze(tmp / "m.csv", [tmp / "runs"], {"reasoning_sources_digest": SRC}, SELECTION,
                     kw.pop("not_run", {}), **kw)


def test_admitted_runs_pass_w0_and_build_every_table_row(study):
    tmp, rows = study
    report = analyze(tmp)
    assert report["W0"]["holds"] and report["W0"]["worst"]["a"] <= 1e-12
    assert report["admitted"] == list(range(1, 28)) and not report["stopped_after_W0"]
    assert len(report["table"]) == 27
    preds = report["predictions"]
    assert preds["W1"] is True
    assert set(preds["labels"]) == {"0.5b", "1.5b"} and all(v["n"] == 3 for v in preds["labels"].values())
    # The verifier filter never keeps an all-fail or all-pass group, so it takes fewer than S steps.
    assert preds["W3"] == {"0.5b": True, "1.5b": True}


def test_counters_match_an_independent_count_of_the_logs(study):
    tmp, rows = study
    report = analyze(tmp)
    for row in rows[:5]:
        run = A.load_run(next((tmp / "runs" / "900").glob(f"{row['task_id']}_study_*")))
        groups = [g for rec in run["train"] for g in rec["groups"]]
        c = report["counters"][row["task_id"]]
        assert c["groups"] == 150 * 6
        assert c["all_fail"] == sum(all(g["F"]) for g in groups)
        assert c["all_pass"] == sum(not any(g["F"]) for g in groups)
        assert c["mixed"] == len(groups) - c["all_fail"] - c["all_pass"]
        spread = [max(g["rewards"]) - min(g["rewards"]) > 1e-12 for g in groups]
        assert c["identical_reward_filter_drops"] == spread.count(False)
        assert c["kept_groups"] == sum(bool(g["kept"]) for g in groups)
        if row["arm"] == "gate":
            assert c["all_fail_positive_variance"] == 0 and c["window_last_step"] == 0


def test_a_tampered_advantage_fails_w0_and_stops_before_any_outcome(tmp_path):
    rows = write_manifest(tmp_path / "m.csv", kept=("0.5b",))
    runs = tmp_path / "runs" / "900"
    runs.mkdir(parents=True)
    rng = random.Random(1)
    for row in rows:
        fake_run(runs, row, rng, SRC, tamper=row["task_id"] == "4")
    report = analyze(tmp_path)
    assert not report["W0"]["holds"] and report["W0"]["failures"]["a"] == 1
    assert report["stopped_after_W0"] and "table" not in report and "predictions" not in report


def test_a_missing_row_is_refused_unless_it_is_named_not_run(study):
    tmp, rows = study
    victim = next((tmp / "runs" / "900").glob("7_study_*"))
    (victim / "DONE").unlink()
    with pytest.raises(A.Refusal, match=r"\[7\]"):
        analyze(tmp)
    report = analyze(tmp, not_run={"7": "timed out twice"})
    assert 7 not in report["admitted"] and report["not_run"] == {"7": "timed out twice"}
    assert all(r["task_id"] != 7 for r in report["table"])


@pytest.mark.parametrize("field,value,reason", [
    ("device_name", "NVIDIA A100-SXM4-80GB", "device"),
    ("reasoning_sources_digest", "0" * 64, "source digest"),
    ("model_snapshot", "main", "model revision"),
])
def test_admission_refuses_a_run_from_another_source(study, field, value, reason):
    tmp, rows = study
    run = next((tmp / "runs" / "900").glob("3_study_*"))
    prov = json.loads((run / "provenance.json").read_text())
    prov[field] = value
    (run / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(A.Refusal, match=reason):
        analyze(tmp)


def clean_logs(tmp, rows) -> Path:
    logs = tmp / "logs"
    logs.mkdir(exist_ok=True)
    for row in rows:
        (logs / f"wo-i-s1-900_{row['task_id']}.out").write_text('{"event": "done"}\n')
        (logs / f"wo-i-s1-900_{row['task_id']}.err").write_text("")
    return logs


def test_log_text_with_a_traceback_refuses_the_run(study):
    tmp, rows = study
    logs = clean_logs(tmp, rows)
    assert analyze(tmp, logs_dir=logs)["W0"]["holds"]
    (logs / "wo-i-s1-900_2.err").write_text("Traceback (most recent call last):\n")
    with pytest.raises(A.Refusal, match=r"\[2\].*traceback"):
        analyze(tmp, logs_dir=logs)


def test_a_run_without_a_slurm_log_is_refused_when_logs_are_given(study):
    tmp, rows = study
    logs = clean_logs(tmp, rows)
    for p in logs.glob("wo-i-s1-900_5.*"):
        p.unlink()
    with pytest.raises(A.Refusal, match=r"\[5\].*no Slurm log"):
        analyze(tmp, logs_dir=logs)


@pytest.mark.parametrize("field,value", [("tiny", True), ("eval_limit", 3), ("lr", 2e-5), ("device", "cpu"),
                                         ("mid_eval_every", 10), ("grad_clip", 0.5), ("w_task", 0.0),
                                         ("sampler", {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                                                      "repetition_penalty": 1.0})])
def test_admission_refuses_a_run_whose_held_equal_settings_differ(study, field, value):
    tmp, rows = study
    run = next((tmp / "runs" / "900").glob("4_study_*"))
    cfg = json.loads((run / "config.json").read_text())
    cfg[field] = value
    (run / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(A.Refusal, match=r"\[4\].*" + field):
        analyze(tmp)


def test_the_design_settings_equal_the_trainer_defaults():
    import train_reasoning_grpo as TR
    args = TR.build_arg_parser().parse_args(["--dataset", "d", "--selection", "s"])
    for key in ("lr", "weight_decay", "clip_coef", "grad_clip", "eval_seed", "eval_chunk_sequences",
                "micro_batch", "eval_limit", "tiny"):
        assert getattr(args, key) == A.DESIGN_CONFIG[key], key
    assert vars(TR.Sampler()) == A.DESIGN_CONFIG["sampler"]


def test_the_command_line_needs_the_logs(tmp_path, capsys):
    with pytest.raises(SystemExit):
        A.main(["--manifest", "m", "--runs", "r", "--frozen", "f", "--selection", "s", "--out", "o"])
    assert "--logs" in capsys.readouterr().err


def test_w0_only_mode_reports_w0_and_nothing_else(study):
    tmp, rows = study
    report = analyze(tmp, w0_only=True)
    assert report["W0"]["holds"] and report["stopped_after_W0"] and "table" not in report


def row(size, arm, seed, acc, short=0.1, length=100.0):
    return {"task_id": hash((size, arm, seed)) % 1000, "size": size, "arm": arm, "seed": seed,
            "final_hard_accuracy": acc, "final_heldout_short_incorrect_rate": short,
            "final_heldout_mean_length": length}


def test_labels_and_the_w4_to_w7_rules():
    assert A.label(3, 3, True) == "A" and A.label(3, 3, False) == "N"
    assert A.label(2, 0, True) == "Z" and A.label(1, 0, True) == "N" and A.label(3, 2, True) == "N"
    table = []
    for s in RT.SEEDS:
        table += [row("0.5b", "plain", s, 0.3, short=0.2, length=90), row("0.5b", "gate", s, 0.4, short=0.1),
                  row("0.5b", "verdict_only", s, 0.3, length=120), row("0.5b", "vfilter", s, 0.35),
                  row("0.5b", "drgrpo", s, 0.2),
                  row("1.5b", "plain", s, 0.5), row("1.5b", "gate", s, 0.6)]
    counters = {r["task_id"]: {"mixed_order_failures": 0, "all_fail_positive_variance_share": 1.0,
                               "opt_steps_taken": 10} for r in table}
    out = A.score_predictions(table, counters, True, 150)
    assert out["W2"] is True and out["W2_runs_without_all_fail_groups"] == []
    assert out["labels"]["0.5b"] == {"n": 3, "k": 3, "label": "A"} and out["W4"]["0.5b"]
    assert out["W5"]["0.5b"] is True and out["W6"] is True and out["W7"] is True
    assert out["V_and_D"]["0.5b"] == {"V": 3, "D": 0}
    # When the larger size flips, W7 fails. When the smaller size is N, W7 is not scored.
    flipped = [dict(r, final_hard_accuracy=0.4) if (r["size"], r["arm"]) == ("1.5b", "gate") else r for r in table]
    flipped = [dict(r, final_hard_accuracy=0.45) if (r["size"], r["arm"]) == ("1.5b", "plain") else r for r in flipped]
    assert A.score_predictions(flipped, counters, True, 150)["W7"] is False
    mixed = [dict(r, final_hard_accuracy=0.3 if r["seed"] == 0 else r["final_hard_accuracy"])
             if (r["size"], r["arm"]) == ("0.5b", "gate") else r for r in table]
    assert A.score_predictions(mixed, counters, True, 150)["W7"] == "not scored"
    # The 1.5B size has no verifier-filter run here, so W3 is not scored there and does not hold vacuously.
    assert out["W3"] == {"0.5b": True, "1.5b": "not scored"}
    counters[next(r["task_id"] for r in table if r["arm"] == "vfilter")]["opt_steps_taken"] = 150
    assert A.score_predictions(table, counters, True, 150)["W3"]["0.5b"] is False


def test_w2_counts_the_share_per_plain_run_and_lists_runs_without_all_fail_groups():
    table = [row("0.5b", "plain", s, 0.3) for s in RT.SEEDS]
    base = {"mixed_order_failures": 0, "opt_steps_taken": 10}
    counters = {table[0]["task_id"]: dict(base, all_fail_positive_variance_share=0.995),
                table[1]["task_id"]: dict(base, all_fail_positive_variance_share=None),
                table[2]["task_id"]: dict(base, all_fail_positive_variance_share=0.99)}
    out = A.score_predictions(table, counters, True, 150)
    assert out["W2"] is True and out["W2_runs_without_all_fail_groups"] == [table[1]["task_id"]]
    counters[table[2]["task_id"]]["all_fail_positive_variance_share"] = 0.989
    assert A.score_predictions(table, counters, True, 150)["W2"] is False


# ------------------------------------------------------------------ review additions


def _run_for(tmp, rows, arm, seed=0):
    row = next(r for r in rows if r["arm"] == arm and r["seed"] == str(seed) and r["size"] == "0.5b")
    return row, next((tmp / "runs" / "900").glob(f"{row['task_id']}_study_*"))


def _edit_first_all_fail_group(run_dir, edit):
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    for rec in lines:
        if rec["phase"] != "train":
            continue
        for g in rec["groups"]:
            if all(g["F"]):
                edit(g)
                (run_dir / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
                return
    raise AssertionError("the fixture run holds no all-fail group")


@pytest.mark.parametrize("arm,edit,counter", [
    ("plain", lambda g: g.update(adv=[a + 1e-3 for a in g["adv"]]), "b"),
    ("drgrpo", lambda g: g.update(adv=[a + 1e-3 for a in g["adv"]]), "b"),
    ("gate", lambda g: g.update(adv=[1e-3, -1e-3, 0.0, 0.0]), "c"),
    ("verdict_only", lambda g: g.update(adv=[1e-3, -1e-3, 0.0, 0.0]), "c"),
    ("vfilter", lambda g: g.update(kept=True), "c"),
    ("plain", lambda g: g.update(kept=False), "kept"),
    ("plain", lambda g: g.update(rewards=[x + 1e-3 for x in g["rewards"]]), "reward"),
])
def test_each_part_of_w0_fails_on_its_own_tampering(study, arm, edit, counter):
    tmp, rows = study
    _, run_dir = _run_for(tmp, rows, arm)
    _edit_first_all_fail_group(run_dir, edit)
    report = analyze(tmp)
    assert not report["W0"]["holds"] and report["W0"]["failures"][counter] >= 1
    assert report["stopped_after_W0"] and "table" not in report


def test_admission_refuses_a_missing_evaluation_a_changed_row_and_another_dataset(study):
    tmp, rows = study
    _, run_dir = _run_for(tmp, rows, "plain", 1)
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    drop = next(i for i, x in enumerate(lines) if json.loads(x)["phase"] == "eval" and json.loads(x)["step"] == 50)
    (run_dir / "metrics.jsonl").write_text("".join(x + "\n" for i, x in enumerate(lines) if i != drop))
    with pytest.raises(A.Refusal, match="evaluation record"):
        analyze(tmp)
    _, run_dir = _run_for(tmp, rows, "gate", 1)
    cfg = json.loads((run_dir / "config.json").read_text())
    cfg["manifest_row"] = dict(cfg["manifest_row"], L="512")
    (run_dir / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(A.Refusal, match="manifest row"):
        analyze(tmp, not_run={str(r["task_id"]): "x" for r in rows if r["arm"] == "plain" and r["seed"] == "1"
                              and r["size"] == "0.5b"})
    _, run_dir = _run_for(tmp, rows, "drgrpo", 2)
    prov = json.loads((run_dir / "provenance.json").read_text())
    prov["dataset_sha256"] = "0" * 64
    (run_dir / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(A.Refusal, match="dataset or selection"):
        analyze(tmp, not_run={str(r["task_id"]): "x" for r in rows if r["size"] == "0.5b" and r["seed"] == "1"
                              and r["arm"] in ("plain", "gate")})


def test_the_final_endpoints_pool_both_final_evaluations():
    hard = [["te0", 0, 10, True, 1, None, None, 0], ["te0", 1, 90, True, 0, "3", "3", 1]]
    rand = [["te5", 0, 4, False, 1, None, None, 0], ["te6", 0, 60, True, 1, "2", "2", 0]]
    run = {"eval": [{"step": 150, "set": "hard", "accuracy": 0.5, "lenient_accuracy": 0.5, "items": hard},
                    {"step": 150, "set": "random", "accuracy": 0.0, "lenient_accuracy": 0.0, "items": rand},
                    {"step": 0, "set": "hard", "accuracy": 1.0, "lenient_accuracy": 1.0, "items": hard[1:]}]}
    out = A.final_endpoints(run, 150, 20.0)
    assert out["final_hard_accuracy"] == 0.5 and out["final_random_accuracy"] == 0.0
    # Two of the four pooled completions are incorrect and shorter than 20 tokens.
    assert out["final_heldout_short_incorrect_rate"] == 0.5
    assert out["final_heldout_mean_length"] == (10 + 90 + 4 + 60) / 4
    assert out["final_heldout_truncation_rate"] == 0.25 and out["final_heldout_strict_parse_fail_rate"] == 0.5
