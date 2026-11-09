"""These tests run the trainer's --keep-schedule flag on the random tiny model.

A yoked control keeps exactly the groups its schedule lists and takes an optimizer step only when it
keeps one. These tests check five things.
  * The default path never reads a schedule file and logs nothing new in a step record.
  * A schedule copied from a filter run's kept flags reproduces that filter run exactly.
  * An empty schedule never moves a weight.
  * The same seed and the same schedule give the same run.
  * Every use of the flag that the experiment does not define is refused before anything is written.

The in-process runs point the trainer at a temporary schedule file through the module constant.
"""
import json
import os
import sys
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

torch = pytest.importorskip("torch")

import train_speed_grpo as T  # noqa: E402

# These flags are the golden speed case's setting. At seed 3 the random model parses a feasible
# speed in one group at step 2, so the verifier filter keeps that group and drops the other five.
FILTER_ARGV = ["--arm", "binary", "--w", "5", "--vmax", "5", "--steps", "3", "--tiny",
               "--device", "cpu", "--eval-every", "2", "--eval-samples", "2",
               "--batch-prompts", "6", "--seed", "3", "--max-new-tokens", "24"]
SMALL_ARGV = ["--arm", "binary", "--w", "5", "--vmax", "5", "--steps", "2", "--tiny",
              "--device", "cpu", "--eval-every", "2", "--eval-samples", "2",
              "--batch-prompts", "2", "--seed", "0"]
VOLATILE = {"s_per_step"}


def _read(out):
    recs = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    cfg = json.load(open(os.path.join(out, "config.json")))
    prov = json.load(open(os.path.join(out, "provenance.json")))
    return (cfg, prov, [r for r in recs if r["phase"] == "train"],
            [r for r in recs if r["phase"] == "eval"])


def _entry(argv, sets, **over):
    """Return a schedule entry for the run that `argv` makes, with `over` replacing any field."""
    args = T.build_arg_parser().parse_args(list(argv) + ["--out-dir", "unused"])
    entry = {k: getattr(args, k) for k in T.KEEP_SCHEDULE_FIELDS}
    entry.update(draw=0, kept_sets=[list(s) for s in sets])
    entry.update(over)
    return entry


def _schedule_file(tmp_path, monkeypatch, entries):
    path = tmp_path / "keep_schedules.json"
    path.write_text(json.dumps({"schema": 1, "schedules": entries}))
    monkeypatch.setattr(T, "KEEP_SCHEDULES_PATH", path)
    return path


def _run(tmp_path, name, argv):
    out = tmp_path / name
    assert T.main(list(argv) + ["--out-dir", str(out)]) == 0
    return _read(out)


def _strip(train):
    return [{k: v for k, v in r.items() if k not in VOLATILE} for r in train]


def test_the_default_path_never_reads_a_schedule(tmp_path, monkeypatch):
    def boom(path):
        raise AssertionError("a default run opened the schedule file")
    monkeypatch.setattr(T, "_read_keep_schedules", boom)
    cfg, _, train, _ = _run(tmp_path, "plain", SMALL_ARGV + ["--steps", "3"])
    assert cfg["keep_schedule"] is None
    assert [r["step"] for r in train] == [1, 2, 3]
    assert not any("keep_schedule_n" in r for r in train)
    assert all(g["kept"] is True for r in train for g in r["groups"])


def test_a_schedule_copied_from_a_filter_run_reproduces_it(tmp_path, monkeypatch):
    fcfg, fprov, ftrain, fevals = _run(tmp_path, "filter", FILTER_ARGV + ["--group-filter", "verdict_mixed"])
    kept = [sorted(g["prompt_index"] for g in r["groups"] if g["kept"]) for r in ftrain]
    flags = [g["kept"] for r in ftrain for g in r["groups"]]
    assert any(flags) and not all(flags), \
        "precondition: the filter run must keep some groups and drop others"
    _schedule_file(tmp_path, monkeypatch, {"copy": _entry(FILTER_ARGV, kept)})
    ccfg, cprov, ctrain, cevals = _run(tmp_path, "control", FILTER_ARGV + ["--keep-schedule", "copy"])
    assert (fcfg["group_filter"], ccfg["group_filter"]) == ("verdict_mixed", "none")
    assert ccfg["keep_schedule"] == "copy"
    assert [r["keep_schedule_n"] for r in ctrain] == [len(s) for s in kept]
    for f, c in zip(_strip(ftrain), _strip(ctrain)):
        c = dict(c)
        c.pop("keep_schedule_n")
        assert c == f, f"step {f['step']} differs between the filter run and its copy"
    assert cevals == fevals
    assert cprov["final_param_checksum"] == fprov["final_param_checksum"]


def _initial_checksum(argv):
    args = T.build_arg_parser().parse_args(list(argv) + ["--out-dir", "unused"])
    T.seed_everything(args.seed)
    _, model, _ = T.build_models(args, "cpu")
    return T.param_checksum(model)


def test_an_empty_schedule_never_moves_a_weight(tmp_path, monkeypatch):
    _schedule_file(tmp_path, monkeypatch, {"empty": _entry(SMALL_ARGV, [[], []])})
    _, prov, train, evals = _run(tmp_path, "empty", SMALL_ARGV + ["--keep-schedule", "empty"])
    assert all(r["opt_step_taken"] is False and r["n_kept_groups"] == 0 for r in train)
    assert all(g["kept"] is False for r in train for g in r["groups"])
    assert [e["step"] for e in evals] == [0, 2]
    assert evals[0]["eval_speeds"] == evals[1]["eval_speeds"]
    assert prov["final_param_checksum"] == _initial_checksum(SMALL_ARGV)


def test_a_scheduled_group_with_zero_advantages_still_takes_an_optimizer_step(tmp_path, monkeypatch):
    """The schedule, not the rewards, decides what is kept. A kept group whose rewards tie has
    advantages of exactly zero, so the step calls the optimizer on an exactly zero gradient."""
    _schedule_file(tmp_path, monkeypatch, {"tie": _entry(SMALL_ARGV, [[0], [0, 1]])})
    _, _, train, _ = _run(tmp_path, "tie", SMALL_ARGV + ["--keep-schedule", "tie"])
    first = train[0]
    assert all(x == 0.0 for x in first["groups"][0]["adv"]), \
        "precondition: the random model's first group at seed 0 has tied rewards"
    assert first["opt_step_taken"] is True and first["grad_norm_preclip"] == 0.0
    assert [g["kept"] for g in first["groups"]] == [True, False]
    assert [g["kept"] for g in train[1]["groups"]] == [True, True]


def test_same_seed_same_run_under_a_schedule(tmp_path, monkeypatch):
    kept = [[4], [1, 3], []]
    _schedule_file(tmp_path, monkeypatch, {"same": _entry(FILTER_ARGV, kept)})
    _, pa, ta, ea = _run(tmp_path, "a", FILTER_ARGV + ["--keep-schedule", "same"])
    _, pb, tb, eb = _run(tmp_path, "b", FILTER_ARGV + ["--keep-schedule", "same"])
    assert _strip(ta) == _strip(tb) and ea == eb
    assert pa["final_param_checksum"] == pb["final_param_checksum"]
    assert [sorted(g["prompt_index"] for g in r["groups"] if g["kept"]) for r in ta] == kept


REFUSALS = {
    "unknown id": ([], {}, "missing"),
    "another task": ([], {"task": "capacity"}, "bad"),
    "another regime": ([], {"regime": "B"}, "bad"),
    "another arm": ([], {"arm": "graded"}, "bad"),
    "another weight": ([], {"w": 25.0}, "bad"),
    "another task weight": ([], {"w_task": 0.0}, "bad"),
    "another limit": ([], {"vmax": 6.0}, "bad"),
    "another seed": ([], {"seed": 1}, "bad"),
    "another step count": ([], {"steps": 3}, "bad"),
    "another batch size": ([], {"batch_prompts": 6}, "bad"),
    "too few sets": ([], {"kept_sets": [[0]]}, "bad"),
    "too many sets": ([], {"kept_sets": [[0], [], []]}, "bad"),
    "an unsorted set": ([], {"kept_sets": [[1, 0], []]}, "bad"),
    "a repeated index": ([], {"kept_sets": [[0, 0], []]}, "bad"),
    "an index out of range": ([], {"kept_sets": [[2], []]}, "bad"),
    "a negative index": ([], {"kept_sets": [[-1], []]}, "bad"),
    "a boolean index": ([], {"kept_sets": [[True], []]}, "bad"),
    "the verifier filter": (["--group-filter", "verdict_mixed"], {}, "good"),
    "another estimator": (["--adv-variant", "drgrpo"], {}, "good"),
    "a KL term": (["--kl-beta", "0.04"], {}, "good"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_undefined_schedule_uses_are_refused_before_anything_is_written(tmp_path, monkeypatch, case):
    extra, over, sid = REFUSALS[case]
    _schedule_file(tmp_path, monkeypatch, {"good": _entry(SMALL_ARGV, [[0], []]),
                                           "bad": _entry(SMALL_ARGV, [[0], []], **over)})
    out = tmp_path / "refused"
    with pytest.raises(SystemExit) as e:
        T.main(SMALL_ARGV + extra + ["--keep-schedule", sid, "--out-dir", str(out)])
    assert e.value.code == 2
    assert not out.exists()


def test_a_missing_schedule_file_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "KEEP_SCHEDULES_PATH", tmp_path / "absent.json")
    out = tmp_path / "refused"
    with pytest.raises(SystemExit) as e:
        T.main(SMALL_ARGV + ["--keep-schedule", "good", "--out-dir", str(out)])
    assert e.value.code == 2
    assert not out.exists()


def test_a_valid_entry_passes_the_check_and_the_default_returns_none(tmp_path, monkeypatch):
    _schedule_file(tmp_path, monkeypatch, {"good": _entry(SMALL_ARGV, [[1], [0, 1]])})
    ap = T.build_arg_parser()
    got = T.check_args(ap, ap.parse_args(SMALL_ARGV + ["--keep-schedule", "good", "--out-dir", "x"]))
    assert got == [frozenset({1}), frozenset({0, 1})]
    assert T.check_args(ap, ap.parse_args(SMALL_ARGV + ["--out-dir", "x"])) is None


def test_the_committed_schedule_file_and_manifests_are_training_sources():
    for rel in ("configs/keep_schedules.json", "slurm/manifest_speed_yoked.csv",
                "slurm/manifest_capacity_yoked.csv"):
        assert f"experiments/cancellation/{rel}" in T.TRAINING_SOURCES
    assert T.KEEP_SCHEDULES_PATH == Path(HERE).resolve() / "configs" / "keep_schedules.json"
