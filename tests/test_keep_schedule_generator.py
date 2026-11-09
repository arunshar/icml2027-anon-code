"""These tests cover the rule that picks each yoked control's kept groups and the committed schedule file.

The rule may read only how many groups the filter run kept at each step. These tests check that it
reads nothing else, that it gives the draws of one seed distinct first-step sets, that it matches the
draw table the plan fixed before any control run, and that the committed file is its output.

Two checks need the filter runs' own logs, and they skip when the run directories are absent.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import make_keep_schedules as K  # noqa: E402
import train_speed_grpo as T  # noqa: E402

COMMITTED = Path(HERE) / K.OUT_REL
REPO = Path(HERE)
RUNS_PRESENT = all((REPO / s["run"] / "metrics.jsonl").exists() for s in K.SOURCES)
requires_runs = pytest.mark.skipif(not RUNS_PRESENT, reason="the filter runs' logs are not in this copy")


def _committed():
    return json.loads(COMMITTED.read_text())


def _same_numpy():
    return _committed()["numpy_version"] == np.__version__


requires_same_numpy = pytest.mark.skipif(
    not COMMITTED.exists() or not _same_numpy(),
    reason="numpy does not promise the same permutations across versions, and the committed file "
           "was written with another numpy")


class _Tripwire(dict):
    """This record raises when code reads any key it was not allowed to read."""

    def __init__(self, allowed, data):
        super().__init__(data)
        self._allowed = allowed

    def __getitem__(self, k):
        if k not in self._allowed:
            raise AssertionError(f"the rule read {k!r}")
        return super().__getitem__(k)

    def get(self, k, default=None):
        if k not in self._allowed:
            raise AssertionError(f"the rule read {k!r}")
        return super().get(k, default)


def _records(kept_by_step, rng, batch_prompts=6, tripwire=False):
    """Return synthetic training records with random rewards, advantages and verdicts."""
    out = [{"step": 0, "phase": "eval", "eval_violation_rate": float(rng.random())}]
    for t, kept in enumerate(kept_by_step, start=1):
        groups = []
        for pi in range(batch_prompts):
            g = {"prompt_index": pi, "kept": pi in kept, "rewards": rng.normal(size=4).tolist(),
                 "adv": rng.normal(size=4).tolist(), "speeds": rng.integers(0, 900, size=4).tolist(),
                 "all_violating": bool(rng.random() < 0.5), "has_violator": bool(rng.random() < 0.5)}
            groups.append(_Tripwire({"prompt_index", "kept"}, g) if tripwire else g)
        rec = {"step": t, "phase": "train", "groups": groups, "reward_mean": float(rng.random()),
               "violation_rate": float(rng.random())}
        out.append(_Tripwire({"step", "phase", "groups"}, rec) if tripwire else rec)
    return out


def test_the_rule_reads_only_the_kept_counts():
    """Two filter logs with equal kept counts and different rewards, verdicts and kept prompts give
    the same control schedules."""
    a = [[4], [], [1, 2], [], [0, 3, 5]]
    b = [[0], [], [3, 5], [], [1, 2, 4]]
    rec_a = _records(a, np.random.default_rng(1))
    rec_b = _records(b, np.random.default_rng(2))
    sets_a = K.source_kept_sets(rec_a, 5, 6)
    sets_b = K.source_kept_sets(rec_b, 5, 6)
    assert sets_a == a and sets_b == b
    counts_a, counts_b = [len(s) for s in sets_a], [len(s) for s in sets_b]
    assert counts_a == counts_b
    assert K.control_kept_sets("speed", 5.0, 0, counts_a, 6, 6) == \
        K.control_kept_sets("speed", 5.0, 0, counts_b, 6, 6)


def test_reading_the_source_touches_no_reward_advantage_or_verdict():
    recs = _records([[2], [], [0, 5]], np.random.default_rng(3), tripwire=True)
    assert K.source_kept_sets(recs, 3, 6) == [[2], [], [0, 5]]


def test_the_rule_never_receives_anything_but_counts():
    """The rule's signature takes the counts and the run's identity, and nothing from the log."""
    import inspect
    params = list(inspect.signature(K.control_kept_sets).parameters)
    assert params == ["task", "w", "seed", "counts", "n_draws", "batch_prompts"]


@pytest.mark.parametrize("task,w,seed,counts_at", [
    ("speed", 5.0, 0, {1: 1, 2: 1}),
    ("speed", 25.0, 1, {1: 2, 2: 5}),
    ("capacity", 5.0, 1, {19: 1, 120: 1, 122: 2, 126: 5}),
])
def test_first_step_sets_are_distinct_and_later_sets_have_the_scheduled_size(task, w, seed, counts_at):
    counts = [counts_at.get(t, 0) for t in range(1, 301)]
    draws = K.control_kept_sets(task, w, seed, counts, K.DRAWS[task], 6)
    t1 = min(counts_at)
    first = [tuple(d[t1 - 1]) for d in draws]
    assert len(set(first)) == len(first)
    for d in draws:
        assert len(d) == 300
        for t, s in enumerate(d, start=1):
            assert len(s) == counts[t - 1]
            assert s == sorted(set(s)) and all(0 <= i < 6 for i in s)


@requires_same_numpy
def test_the_rule_reproduces_the_draw_table_fixed_before_any_control_run():
    """This test pins the kept sets at the filter's update steps for three of the nine source runs.
    The values are the ones the plan's Section 2.4 table and its scratch output list."""
    cases = [
        ("speed", 5.0, 0, {1: 1, 2: 1},
         [[[0], [2]], [[4], [1]], [[5], [3]], [[2], [4]], [[1], [4]], [[3], [3]]]),
        ("speed", 25.0, 1, {1: 2, 2: 5},
         [[[3, 5], [0, 1, 2, 4, 5]], [[1, 4], [0, 2, 3, 4, 5]], [[1, 2], [0, 1, 2, 4, 5]],
          [[1, 5], [0, 1, 3, 4, 5]], [[1, 3], [1, 2, 3, 4, 5]], [[4, 5], [0, 1, 2, 4, 5]]]),
        ("capacity", 5.0, 2, {15: 1, 16: 1, 17: 5, 18: 4, 20: 1, 50: 1},
         [[[1], [2], [0, 1, 3, 4, 5], [0, 2, 3, 5], [3], [0]],
          [[2], [3], [0, 1, 2, 4, 5], [0, 1, 4, 5], [3], [2]]]),
    ]
    for task, w, seed, counts_at, want in cases:
        counts = [counts_at.get(t, 0) for t in range(1, 301)]
        draws = K.control_kept_sets(task, w, seed, counts, K.DRAWS[task], 6)
        steps = sorted(counts_at)
        assert [[d[t - 1] for t in steps] for d in draws] == want, (task, w, seed)


def test_schedule_ids_have_one_form():
    assert K.schedule_id("speed", 25.0, 1, 5) == "speed_w25_s1_d5"
    assert K.schedule_id("capacity", 5.0, 0, 0) == "capacity_w5_s0_d0"
    with pytest.raises(ValueError):
        K.schedule_id("speed", 5.5, 0, 0)
    for bad in ("speed_w05_s0_d0", "speed_w5_s0_d", "cap_w5_s0_d0", "speed_w5_s0_d1x"):
        assert K.SCHEDULE_ID.fullmatch(bad) is None


def test_a_run_without_an_update_or_with_too_many_draws_is_refused():
    with pytest.raises(ValueError):
        K.control_kept_sets("speed", 5.0, 0, [0] * 10, 2, 6)
    with pytest.raises(ValueError):
        K.control_kept_sets("speed", 5.0, 0, [6] + [0] * 9, 2, 6)   # one set of six prompts exists


def _fake_source(root: Path, rel: str, kept, seed=0, **cfg_over):
    d = root / rel
    d.mkdir(parents=True)
    cfg = {"task": "speed", "arm": "binary", "w": 5.0, "seed": seed, "adv_variant": "grpo",
           "group_filter": "verdict_mixed", "kl_beta": 0.0, "regime": "A", "w_task": 1.0,
           "vmax": 5.0, "steps": len(kept), "batch_prompts": 6}
    cfg.update(cfg_over)
    (d / "config.json").write_text(json.dumps(cfg))
    lines = [json.dumps(r) for r in _records(kept, np.random.default_rng(seed))]
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    (d / "DONE").write_text("ok\n")
    return {"task": "speed", "w": 5.0, "seed": seed, "manifest": "unused", "task_id": 1, "run": rel,
            "metrics_sha256": hashlib.sha256((d / "metrics.jsonl").read_bytes()).hexdigest()}


def test_a_document_is_built_from_synthetic_sources(tmp_path):
    src = _fake_source(tmp_path, "runs/1/a", [[3], [], [1, 4]])
    doc = K.build_document([src], tmp_path)
    assert sorted(doc["schedules"]) == [f"speed_w5_s0_d{d}" for d in range(6)]
    e = doc["schedules"]["speed_w5_s0_d0"]
    assert e["source_kept_sets"] == [[3], [], [1, 4]]
    assert [len(s) for s in e["kept_sets"]] == [1, 0, 2]
    assert json.loads(K.render(doc)) == doc


@pytest.mark.parametrize("damage", ["done", "hash", "filter"])
def test_a_changed_or_unfinished_source_is_refused(tmp_path, damage):
    src = _fake_source(tmp_path, "runs/1/a", [[3], [], [1, 4]],
                       **({"group_filter": "none"} if damage == "filter" else {}))
    d = tmp_path / src["run"]
    if damage == "done":
        (d / "DONE").unlink()
    if damage == "hash":
        (d / "metrics.jsonl").write_text((d / "metrics.jsonl").read_text() + "\n")
    with pytest.raises(K.SourceError):
        K.build_document([src], tmp_path)


def test_the_committed_file_holds_42_schedules_that_copy_their_sources_counts():
    doc = _committed()
    entries = doc["schedules"]
    assert len(entries) == 42
    assert sum(k.startswith("speed_") for k in entries) == 36
    assert sum(k.startswith("capacity_") for k in entries) == 6
    assert doc["rule"]["constant"] == K.THINNING_CONSTANT
    by_source = {}
    for sid, e in entries.items():
        assert K.SCHEDULE_ID.fullmatch(sid)
        assert sid == K.schedule_id(e["task"], e["w"], e["seed"], e["draw"])
        assert len(e["kept_sets"]) == e["steps"] == 300
        assert [len(s) for s in e["kept_sets"]] == [len(s) for s in e["source_kept_sets"]]
        by_source.setdefault(e["source_run"], []).append(e)
    assert len(by_source) == 9
    for run, group in by_source.items():
        t1 = next(t for t, s in enumerate(group[0]["source_kept_sets"]) if s)
        first = [tuple(e["kept_sets"][t1]) for e in group]
        assert len(set(first)) == len(first), run


def test_every_committed_entry_passes_the_trainers_own_check(monkeypatch):
    monkeypatch.setattr(T, "KEEP_SCHEDULES_PATH", COMMITTED)
    for sid, e in _committed()["schedules"].items():
        argv = ["--task", e["task"], "--regime", e["regime"], "--arm", e["arm"], "--w", str(e["w"]),
                "--w-task", str(e["w_task"]), "--vmax", str(e["vmax"]), "--seed", str(e["seed"]),
                "--keep-schedule", sid, "--out-dir", "unused"]
        ap = T.build_arg_parser()
        sched = T.check_args(ap, ap.parse_args(argv))
        assert [sorted(s) for s in sched] == e["kept_sets"]


@requires_runs
def test_the_committed_file_records_the_local_source_hashes():
    entries = _committed()["schedules"]
    for src in K.SOURCES:
        got = hashlib.sha256((REPO / src["run"] / "metrics.jsonl").read_bytes()).hexdigest()
        assert got == src["metrics_sha256"]
        for e in entries.values():
            if e["source_run"] == src["run"]:
                assert e["source_metrics_sha256"] == got


@requires_runs
@requires_same_numpy
def test_the_committed_file_is_the_generator_output():
    assert COMMITTED.read_text() == K.render(K.build_document())
    assert K.main(["--check"]) == 0
