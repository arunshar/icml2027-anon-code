"""These tests check the frozen GSM8K candidates and the selection rules of Section 2.3 on synthetic records."""
import hashlib
import json
import math
from pathlib import Path

import pytest

import make_reasoning_dataset as M
import reasoning_task as RT
import select_reasoning_pool as SEL
from reasoning_fixtures import calibration_set, synthetic_candidates

HERE = Path(__file__).resolve().parent.parent
FROZEN_FILE = HERE / "configs" / "reasoning_dataset.json"
REAL = HERE / "data" / "gsm8k_main"


def rows(n, tag):
    return [(f"{tag} question {i}?", f"work\n#### {i * 7}") for i in range(n)]


# ------------------------------------------------------------------ the builder


def test_candidate_draw_is_deterministic_and_seeded():
    a = M.dumps(M.build(rows(50, "train"), rows(30, "test"), seed=5, n_train=10, n_test=6))
    b = M.dumps(M.build(rows(50, "train"), rows(30, "test"), seed=5, n_train=10, n_test=6))
    c = M.dumps(M.build(rows(50, "train"), rows(30, "test"), seed=6, n_train=10, n_test=6))
    assert a == b and a != c
    data = json.loads(a)
    assert [x["cand"] for x in data["train_candidates"]] == list(range(10))
    assert len({x["row"] for x in data["test_candidates"]}) == 6


def test_builder_refuses_shared_text_duplicates_and_bad_answers():
    train = rows(20, "train")
    with pytest.raises(ValueError, match="share"):
        M.build(train, [train[3]] + rows(5, "test"), n_train=5, n_test=3)
    with pytest.raises(ValueError, match="twice"):
        M.build(train + [train[0]], rows(5, "test"), n_train=5, n_test=3)
    with pytest.raises(ValueError):
        M.build(train, rows(5, "test") + [("odd?", "#### twelve")], n_train=5, n_test=3)


def test_parquet_digest_and_size_checks_refuse_a_changed_file(tmp_path):
    p = tmp_path / "f.parquet"
    p.write_bytes(b"x" * 10)
    spec = {"bytes": 10, "sha256": hashlib.sha256(b"x" * 10).hexdigest()}
    M.check_file(p, spec)
    with pytest.raises(ValueError, match="bytes"):
        M.check_file(p, dict(spec, bytes=11))
    with pytest.raises(ValueError, match="SHA-256"):
        M.check_file(p, dict(spec, sha256="0" * 64))


def test_frozen_file_records_the_source_and_holds_disjoint_splits():
    data = M.load_dataset(FROZEN_FILE)
    assert data["source"] == M.SOURCE
    assert data["candidate_seed"] == RT.CANDIDATE_SEED
    assert data["checks"]["train_test_shared_problem_texts"] == 0
    norm = lambda s: " ".join(s.split())  # noqa: E731
    train = {norm(c["question"]) for c in data["train_candidates"]}
    assert not train & {norm(c["question"]) for c in data["test_candidates"]}
    pids = [c["pid"] for c in data["train_candidates"] + data["test_candidates"]]
    assert len(set(pids)) == len(pids)


@pytest.mark.skipif(not (REAL / "train-00000-of-00001.parquet").exists(),
                    reason="the GSM8K parquet files are not on this machine")
def test_frozen_file_is_exactly_what_the_builder_writes_from_the_parquet_files(tmp_path):
    out = tmp_path / "d.json"
    assert M.main(["--out", str(out)]) == 0
    assert out.read_bytes() == FROZEN_FILE.read_bytes()


# ------------------------------------------------------------------ selection


def c_of(train_c, test_c):
    return lambda cand: (train_c if cand["pid"].startswith("tr") else test_c)(cand["cand"])


@pytest.fixture(scope="module")
def cands():
    return synthetic_candidates()


def test_pool_hard_set_random_set_and_L_follow_the_rules(tmp_path, cands):
    # 400 train problems at c = 0, 300 at c = 2 and 300 at c = 8. The prefix mean stays at or
    # above 0.6 through all 700 of the first two groups and 73 of the third, so the pool holds 773.
    train_c = lambda i: 0 if i < 400 else (2 if i < 700 else 8)  # noqa: E731
    test_c = lambda i: 0 if i < 150 else 8                        # noqa: E731
    # One completion in eight is 300 tokens long, so 12.5 percent exceed 256 and none exceed 384.
    lengths = lambda cand, j: (300 if j == 0 else 100, True)      # noqa: E731
    cal = calibration_set(tmp_path, cands, c_of(train_c, test_c), lengths_of=lengths)
    sel = SEL.select(cands, "d" * 64, cal)
    assert sel["L"] == 384
    assert sel["exceed_share_by_L"] == {"256": 0.125, "384": 0.0, "512": 0.0}
    small = sel["sizes"]["0.5b"]
    assert small["status"] == "ok" and small["pool_size"] == 773 and small["pool_max_c"] == 8
    assert small["pool"][:400] == [f"tr{i}" for i in range(400)]
    assert small["pool_mean_expected_all_fail"] >= RT.POOL_TARGET
    # Every test problem has c <= 8, so the seeded draw keeps 200 of the 400.
    assert small["hard_size"] == 200 and len(set(small["hard"])) == 200
    assert sel["random"] == [f"te{i}" for i in range(200)]
    assert not set(small["pool"]) & set(small["hard"] + sel["random"])


def test_schedule_is_seeded_six_distinct_problems_per_step_and_cycles_the_pool():
    pool = [f"tr{i}" for i in range(310)]
    a, b = SEL.build_schedule(pool, "0.5b"), SEL.build_schedule(pool, "0.5b")
    assert a == b and a != SEL.build_schedule(pool, "1.5b")
    assert len(a) == 300 and all(len(set(step)) == 6 for step in a)
    flat = [p for step in a for p in step]
    assert sorted(flat[:310]) == sorted(pool)             # each problem appears once per pass
    assert flat[310:620] == flat[:310]                    # the same shuffle is cycled


def test_hard_set_uses_the_largest_c_in_the_pool(cands):
    # 500 problems at c = 0 have an expected all-fail fraction of 1, and 500 at c = 5 have 0, so
    # the prefix mean stays at or above 0.6 through 333 of the second group.
    c_train = {f"tr{i}": (0 if i < 500 else 5) for i in range(1000)}
    pool, mean = SEL.build_pool(cands["train_candidates"], c_train)
    assert len(pool) == 833 and mean == pytest.approx(500 / 833)
    assert max(c_train[p] for p in pool) == 5
    c_test = {f"te{i}": (1 if i < 120 else 3) for i in range(400)}
    hard = SEL.build_hard(cands["test_candidates"], c_test, 1, "0.5b")
    assert hard == [f"te{i}" for i in range(120)]


def test_stop_rules_fire(tmp_path, cands):
    ok_train = lambda i: 0 if i < 600 else 4       # noqa: E731
    ok_test = lambda i: 0                          # noqa: E731
    # No L candidate fits, because every completion runs to 512 without an end token.
    cal = calibration_set(tmp_path / "a", cands, c_of(ok_train, ok_test), lengths_of=lambda c, j: (512, False))
    with pytest.raises(SEL.Stop, match="no L candidate"):
        SEL.select(cands, "d" * 64, cal)
    # The strict parser finds no answer in most completions.
    cal = calibration_set(tmp_path / "b", cands, c_of(ok_train, ok_test), parsed=False)
    with pytest.raises(SEL.Stop, match="strict-parse rate"):
        SEL.select(cands, "d" * 64, cal)
    # Every train problem is easy, so the 0.5B pool is empty and the block stops.
    cal = calibration_set(tmp_path / "c", cands, c_of(lambda i: 8, ok_test))
    with pytest.raises(SEL.Stop, match="0.5B"):
        SEL.select(cands, "d" * 64, cal)


def test_a_1p5b_floor_miss_drops_only_that_size(tmp_path, cands):
    cal = calibration_set(tmp_path, cands, c_of(lambda i: 0 if i < 600 else 4, lambda i: 0))
    easy = calibration_set(tmp_path / "easy", cands, c_of(lambda i: 0 if i < 100 else 8, lambda i: 0))
    cal["1.5b"] = easy["1.5b"]
    sel = SEL.select(cands, "d" * 64, cal)
    assert sel["sizes"]["0.5b"]["status"] == "ok"
    assert sel["sizes"]["1.5b"]["status"] == "dropped"
    assert "pool holds" in sel["sizes"]["1.5b"]["reasons"][0]
    # A hard held-out set under 100 also drops the size.
    few = calibration_set(tmp_path / "few", cands, c_of(lambda i: 0 if i < 600 else 4,
                                                       lambda i: 0 if i < 50 else 8))
    cal["1.5b"] = few["1.5b"]
    sel = SEL.select(cands, "d" * 64, cal)
    assert sel["sizes"]["1.5b"]["status"] == "dropped" and sel["sizes"]["1.5b"]["hard_size"] == 50


def test_priors_follow_their_definitions(tmp_path, cands):
    cal = calibration_set(tmp_path, cands, c_of(lambda i: 0 if i < 600 else 4, lambda i: 0),
                          lengths_of=lambda c, j: (100 + j, True))
    sel = SEL.select(cands, "d" * 64, cal)
    small = sel["sizes"]["0.5b"]
    # Every all-fail base group has four different lengths, so every one keeps reward spread.
    assert small["w2_prior"] == 1.0
    # c = 0 gives no mixed group, and c = 4 gives a mixed group with probability 1 - 2/70.
    c = small["c_train"]
    manual = sum(1 - math.prod(
        1 - (1 - RT.expected_all_fail(c[p]) - RT.expected_all_pass(c[p])) for p in step)
        for step in small["schedule"][:150])
    assert small["w3_prior_by_S"]["150"] == pytest.approx(manual)
    # The correct completions are samples 0 to 3 of the c = 4 problems, at 100 to 103 tokens.
    assert small["short_threshold"] == 100.0


# ------------------------------------------------------------------ smoke rules


def timing(size, peak, step_s=10.0, L=384, sel_sha="s" * 64):
    return {"kind": "smoke", "size": size, "arm": "plain", "L": L, "micro_batch": 2, "steps": 20,
            "selection_sha256": sel_sha, "device_name": RT.REQUIRED_GPU_NAME, "model_load_seconds": 60.0,
            "step_seconds": [30.0] + [step_s] * 19, "generation_seconds": [], "update_seconds": [],
            "eval_seconds": [{"step": 0, "set": "hard", "samples": 4, "seconds": 100.0},
                             {"step": 0, "set": "random", "samples": 2, "seconds": 50.0},
                             {"step": 10, "set": "hard", "samples": 1, "seconds": 30.0},
                             {"step": 20, "set": "hard", "samples": 4, "seconds": 100.0},
                             {"step": 20, "set": "random", "samples": 2, "seconds": 50.0}],
            "peak_memory_reserved_bytes": peak, "peak_memory_allocated_bytes": peak,
            "steps_completed": 20}


SELECTION = {"L": 384, "sizes": {"0.5b": {"status": "ok"}, "1.5b": {"status": "ok"}}}


def test_smoke_rules_pick_S_from_the_slower_size():
    out = SEL.smoke_rules(SELECTION, "s" * 64, {"0.5b": timing("0.5b", 10**10, 12.0),
                                                "1.5b": timing("1.5b", 30 * 10**9, 25.0)})
    # At 25 s per step the 1.5B run at S = 300 needs (60 + 7500 + 300 + 5 x 30) / 60 = 133.5
    # minutes, which exceeds 120, and at S = 200 it needs 90.8 minutes.
    assert out["S"] == 200 and out["sizes_kept"] == ["0.5b", "1.5b"]
    assert out["slowest_projected_minutes_by_S"]["300"] > 120 >= out["slowest_projected_minutes_by_S"]["200"]


def test_a_smoke_above_36_gb_stops_its_size_and_a_0p5b_stop_ends_the_block():
    out = SEL.smoke_rules(SELECTION, "s" * 64, {"0.5b": timing("0.5b", 10**10),
                                                "1.5b": timing("1.5b", 36 * 10**9 + 1)})
    assert out["sizes_kept"] == ["0.5b"] and out["sizes_stopped"] == ["1.5b"]
    with pytest.raises(SEL.Stop, match="0.5B"):
        SEL.smoke_rules(SELECTION, "s" * 64, {"0.5b": timing("0.5b", 37 * 10**9),
                                              "1.5b": timing("1.5b", 10**10)})
    with pytest.raises(SEL.Stop, match="S = 150"):
        SEL.smoke_rules(SELECTION, "s" * 64, {"0.5b": timing("0.5b", 10**10, 60.0),
                                              "1.5b": timing("1.5b", 10**10)})


def test_smoke_rules_read_timing_fields_only():
    t = timing("0.5b", 10**10)
    t["accuracy"] = 0.5
    with pytest.raises(SEL.Stop, match="unexpected fields"):
        SEL.smoke_rules({"L": 384, "sizes": {"0.5b": {"status": "ok"}, "1.5b": {"status": "dropped"}}},
                        "s" * 64, {"0.5b": t})
    with pytest.raises(SEL.Stop, match="not a smoke"):
        SEL.smoke_rules({"L": 384, "sizes": {"0.5b": {"status": "ok"}, "1.5b": {"status": "dropped"}}},
                        "s" * 64, {"0.5b": dict(timing("0.5b", 10**10), L=256)})


def test_calibration_from_another_snapshot_or_mixed_sources_is_refused(tmp_path, cands):
    from reasoning_fixtures import calibration_dir
    d = calibration_dir(tmp_path, cands, "train", "0.5b", lambda c: 0)
    prov = json.loads((d / "provenance.json").read_text())
    prov["model_snapshot"] = "0" * 40
    (d / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(SEL.Stop, match="pinned revision"):
        SEL.check_calibration(cands, "0.5b", "train", d)
    cal = calibration_set(tmp_path / "ok", cands, c_of(lambda i: 0 if i < 600 else 4, lambda i: 0))
    cal["1.5b"]["test"]["reasoning_sources_digest"] = "other"
    with pytest.raises(SEL.Stop, match="source digests"):
        SEL.select(cands, "d" * 64, cal)


# ------------------------------------------------------------------ review additions


def test_the_limits_hold_at_their_boundaries():
    # The S rule accepts a projection of exactly 120 minutes, and the L rule accepts exactly 5 percent.
    assert RT.choose_steps({300: 120.0, 200: 100.0, 150: 90.0}) == 300
    assert RT.choose_steps({300: 120.0001, 200: 120.0, 150: 90.0}) == 200
    samples = [{"finished": True, "n_tokens": 300 if i < 5 else 100} for i in range(100)]
    assert SEL.choose_L(samples)[0] == 256
    samples[5]["n_tokens"] = 300
    assert SEL.choose_L(samples)[0] == 384


def test_the_parse_rate_is_read_at_the_chosen_L(tmp_path, cands):
    # Every completion parses at 512 tokens, but at 256 tokens only three of every eight do, so the rule stops.
    cal = calibration_set(tmp_path, cands, c_of(lambda i: 0 if i < 600 else 4, lambda i: 0))
    for size in RT.SIZES:
        for split in ("train", "test"):
            for r in cal[size][split]["records"]:
                for s in r["samples"]:
                    s["by_L"]["256"]["strict"] = "1" if s["sample"] % 3 == 0 else None
    with pytest.raises(SEL.Stop, match="strict-parse rate is 0.3750"):
        SEL.select(cands, "d" * 64, cal)


def test_the_short_threshold_is_the_fifth_percentile_of_correct_lengths(tmp_path, cands):
    # Correct completions are samples 0 to 3 of the train problems 600 to 999, whose length is 50 plus
    # the candidate index modulo 100. Their fifth and tenth percentiles differ, so the test tells them apart.
    import numpy as np
    lengths = lambda c, j: (50 + c["cand"] % 100, True)  # noqa: E731
    cal = calibration_set(tmp_path, cands, c_of(lambda i: 0 if i < 600 else 4, lambda i: 0), lengths_of=lengths)
    sel = SEL.select(cands, "d" * 64, cal)
    correct = [50 + i % 100 for i in range(600, 1000) for _ in range(4)]
    assert np.percentile(correct, 5) != np.percentile(correct, 10)
    assert sel["sizes"]["0.5b"]["short_threshold"] == float(np.percentile(correct, 5))


def test_the_w2_prior_counts_only_all_fail_groups_with_spread(tmp_path, cands):
    # Samples 0 to 3 share one length and samples 4 to 7 differ. The pool holds all 1,000 problems. The
    # 600 problems at c = 0 give 1,200 all-fail groups, and 600 of them have no spread. The 400 problems
    # at c = 4 give 400 all-fail groups from samples 4 to 7, which all have spread. The share is 1,000 of 1,600.
    lengths = lambda c, j: ((100 if j < 4 else 100 + j), True)  # noqa: E731
    cal = calibration_set(tmp_path, cands, c_of(lambda i: 0 if i < 600 else 4, lambda i: 0), lengths_of=lengths)
    sel = SEL.select(cands, "d" * 64, cal)
    assert sel["sizes"]["0.5b"]["pool_size"] == 1000
    assert sel["sizes"]["0.5b"]["w2_prior"] == 1000 / 1600


def test_smoke_rules_refuse_a_smoke_on_another_card():
    t = dict(timing("0.5b", 10**10), device_name="NVIDIA A100-SXM4-80GB")
    with pytest.raises(SEL.Stop, match="ran on"):
        SEL.smoke_rules({"L": 384, "sizes": {"0.5b": {"status": "ok"}, "1.5b": {"status": "dropped"}}},
                        "s" * 64, {"0.5b": t})


@pytest.mark.parametrize("field,value,reason", [("tiny", True, "tiny test model"),
                                                ("device_name", "cpu", "ran on 'cpu'")])
def test_a_calibration_from_the_tiny_model_or_another_card_is_refused(tmp_path, cands, field, value, reason):
    from reasoning_fixtures import calibration_dir
    d = calibration_dir(tmp_path, cands, "test", "1.5b", lambda c: 0)
    prov = json.loads((d / "provenance.json").read_text())
    prov[field] = value
    (d / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(SEL.Stop, match=reason):
        SEL.check_calibration(cands, "1.5b", "test", d)


# ------------------------------------------------------------------ fixes approved on 23 September


def _add_retry(d: Path, **changes) -> Path:
    """Append a second attempt to a calibration directory, with the given fields changed."""
    from reasoning_fixtures import calibration_attempt_fields
    prov = json.loads((d / "provenance.json").read_text())
    retry = {"slurm_array_job_id": "2", **calibration_attempt_fields(prov["design"]["model_revision"], **changes)}
    prov["attempts"].append(retry)
    (d / "provenance.json").write_text(json.dumps(prov))
    return d


def test_the_selection_accepts_a_retry_from_the_same_bundle(tmp_path, cands):
    from reasoning_fixtures import FAKE_BUNDLE_DIGEST, FAKE_SOURCE_DIGEST, calibration_dir
    c_fn = c_of(lambda i: 0 if i < 600 else 4, lambda i: 0)
    cal = {size: {split: SEL.check_calibration(cands, size, split,
                                               _add_retry(calibration_dir(tmp_path, cands, split, size, c_fn)))
                  for split in ("train", "test")} for size in RT.SIZES}
    sel = SEL.select(cands, "d" * 64, cal)
    source = sel["calibration_sources"]["1.5b"]["test"]
    assert source["attempts"] == ["1", "2"]
    assert (source["bundle_digest"], source["reasoning_sources_digest"]) == (FAKE_BUNDLE_DIGEST, FAKE_SOURCE_DIGEST)


@pytest.mark.parametrize("field,value", [("bundle_digest", "c" * 64), ("bundle_digest", None),
                                         ("reasoning_sources_digest", "6" * 64),
                                         ("training_sources_digest", "8" * 64), ("dataset_sha256", "e" * 64),
                                         ("model_snapshot", "1" * 40), ("device_name", "cpu"), ("tiny", True),
                                         ("design", {"model_revision": RT.MODELS["1.5b"]["revision"], "seed": 7})])
def test_the_selection_refuses_a_retry_that_differs_from_the_first_attempt(tmp_path, cands, field, value):
    # A retry from another bundle changes the bundle digest and may change the source digest. The top
    # of provenance.json still holds the first attempt's values, so only the attempts show the change.
    from reasoning_fixtures import calibration_dir
    d = _add_retry(calibration_dir(tmp_path, cands, "train", "1.5b", lambda c: 0), **{field: value})
    with pytest.raises(SEL.Stop, match=f"attempts of the 1.5b train calibration carry different values of {field}"):
        SEL.check_calibration(cands, "1.5b", "train", d)


def test_the_selection_refuses_attempts_without_their_own_digests(tmp_path, cands):
    from reasoning_fixtures import calibration_dir
    d = calibration_dir(tmp_path, cands, "test", "0.5b", lambda c: 0)
    prov = json.loads((d / "provenance.json").read_text())
    # A record written before this fix holds a single attempt with no digest in it.
    old = dict(prov, attempts=[{"slurm_array_job_id": "1"}])
    (d / "provenance.json").write_text(json.dumps(old))
    with pytest.raises(SEL.Stop, match="attempt 1 of the 0.5b test calibration does not record reasoning_sources"):
        SEL.check_calibration(cands, "0.5b", "test", d)
    (d / "provenance.json").write_text(json.dumps(dict(prov, attempts=[])))
    with pytest.raises(SEL.Stop, match="records no attempt"):
        SEL.check_calibration(cands, "0.5b", "test", d)
    empty = dict(prov, reasoning_sources_digest=None)
    empty["attempts"] = [dict(prov["attempts"][0], reasoning_sources_digest=None)]
    (d / "provenance.json").write_text(json.dumps(empty))
    with pytest.raises(SEL.Stop, match="carries no source digest"):
        SEL.check_calibration(cands, "0.5b", "test", d)
    top = {k: v for k, v in prov.items() if k != "bundle_digest"}
    (d / "provenance.json").write_text(json.dumps(top))
    with pytest.raises(SEL.Stop, match="does not record the first values of bundle_digest"):
        SEL.check_calibration(cands, "0.5b", "test", d)


def test_the_selection_checks_every_attempt_against_the_first_values(tmp_path, cands):
    # The attempts agree with each other, but the top of the file names another bundle, so the
    # selection would report a bundle that no attempt ran from.
    from reasoning_fixtures import calibration_dir
    d = _add_retry(calibration_dir(tmp_path / "top", cands, "train", "1.5b", lambda c: 0))
    prov = json.loads((d / "provenance.json").read_text())
    (d / "provenance.json").write_text(json.dumps(dict(prov, bundle_digest="c" * 64)))
    with pytest.raises(SEL.Stop, match="different values of bundle_digest"):
        SEL.check_calibration(cands, "1.5b", "train", d)
    # With three attempts, a middle attempt from another bundle is refused as well.
    d = _add_retry(_add_retry(calibration_dir(tmp_path / "mid", cands, "train", "1.5b", lambda c: 0),
                              bundle_digest="c" * 64))
    assert len(json.loads((d / "provenance.json").read_text())["attempts"]) == 3
    with pytest.raises(SEL.Stop, match="different values of bundle_digest"):
        SEL.check_calibration(cands, "1.5b", "train", d)


def test_the_selection_refuses_calibrations_of_another_dataset_file(tmp_path, cands):
    # Every calibration graded its completions against the references of the file it read, so the
    # selection must be made from that same file.
    cal = calibration_set(tmp_path, cands, c_of(lambda i: 0 if i < 600 else 4, lambda i: 0))
    assert SEL.select(cands, "d" * 64, cal)["candidates_sha256"] == "d" * 64
    with pytest.raises(SEL.Stop, match="0.5b train calibration read a dataset file other than"):
        SEL.select(cands, "e" * 64, cal)
    cal["1.5b"]["test"]["dataset_sha256"] = "e" * 64
    with pytest.raises(SEL.Stop, match="1.5b test calibration read a dataset file other than"):
        SEL.select(cands, "d" * 64, cal)
