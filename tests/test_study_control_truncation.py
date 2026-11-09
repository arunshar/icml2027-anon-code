"""Tests for the measurement that splits the control's parse failures into cut and finished completions.

A miniature study stands in for the real one: three control folders whose metrics.jsonl holds the two random held-out
evaluations. Each test checks one count against a value worked out here by hand. The script must refuse when a control folder
or an evaluation is missing or appears twice. On the real study, when its records exist, the output must agree with the
logged truncation rate and the logged mean completion length of every evaluation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
import study_control_truncation as S  # noqa: E402

FIELDS = ["pid", "sample", "n_tokens", "finished", "F", "strict", "lenient", "lenient_correct"]


def item(pid, tokens, finished, strict):
    return [pid, 0, tokens, finished, 0 if strict is not None else 1, strict, strict, 0]


# Eight completions. Four finish and parse, one finishes and does not parse, two are cut and do not parse, one is cut and
# parses, and one ends exactly at the limit with an end token (it counts as finished).
ITEMS_300 = [item("a", 100, True, "1"), item("b", 200, True, "2"), item("c", 300, True, "3"), item("d", 400, True, "4"),
             item("e", 250, True, None), item("f", 512, False, None), item("g", 512, False, None), item("h", 512, True, "5")]
ITEMS_0 = [item(p, 100, True, "1") for p in "abcdefgh"]


def write_row(base, seed, items300=ITEMS_300, items0=ITEMS_0, extra=()):
    d = base / f"{seed * 4 + 1}_study_0.5b_verdict_only_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    records = [{"phase": "train", "step": 1},
               {"phase": "eval", "set": "hard", "step": 300, "item_fields": FIELDS, "items": items0},
               {"phase": "eval", "set": "random", "step": 0, "item_fields": FIELDS, "items": items0},
               {"phase": "eval", "set": "random", "step": 300, "item_fields": FIELDS, "items": items300}, *extra]
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return d


@pytest.fixture
def runs(tmp_path):
    for seed in S.SEEDS:
        write_row(tmp_path, seed)
        # the other three arms must not be picked up
        for arm in ("plain", "restored", "vfilter"):
            (tmp_path / f"{seed * 4 + 2}_study_0.5b_{arm}_s{seed}").mkdir(exist_ok=True)
    return tmp_path


def test_the_counts_of_a_miniature_evaluation_are_the_ones_worked_out_by_hand():
    out = S.split(FIELDS, ITEMS_300, 512)
    assert out["completions"] == 8 and out["parsed"] == 5 and out["unparsed"] == 3
    assert out["cut"] == 2 and out["unparsed_cut"] == 2 and out["finished"] == 6 and out["parsed_of_finished"] == 5
    assert out["at_token_limit"] == 3          # the completion that ends exactly at the limit counts here and not as cut
    assert out["mean_tokens"] == (100 + 200 + 300 + 400 + 250 + 512 * 3) / 8
    assert out["cut_share"] == 0.25 and out["parsed_share_of_finished"] == 5 / 6


def test_a_cut_completion_that_parses_is_counted_as_parsed_and_cut_but_not_as_unparsed_cut():
    items = [item("a", 512, False, "1"), item("b", 100, True, None)]
    out = S.split(FIELDS, items, 512)
    assert (out["parsed"], out["cut"], out["unparsed_cut"], out["finished"], out["parsed_of_finished"]) == (1, 1, 0, 1, 0)


def test_no_finished_completion_gives_a_missing_share_and_not_a_division_error():
    out = S.split(FIELDS, [item("a", 512, False, None)], 512)
    assert out["finished"] == 0 and out["parsed_of_finished"] == 0 and out["parsed_share_of_finished"] != out["parsed_share_of_finished"]


def test_the_table_lists_three_seeds_at_two_steps_and_leaves_the_other_arms_out(runs):
    rows = S.table(runs, (0, 300), 512)
    assert [(r["seed"], r["step"]) for r in rows] == [(0, 0), (0, 300), (1, 0), (1, 300), (2, 0), (2, 300)]
    assert all(r["completions"] == 8 for r in rows)
    assert rows[0]["cut"] == 0 and rows[1]["cut"] == 2


def test_the_text_has_a_header_and_one_line_for_each_row(runs):
    text = S.render(S.table(runs, (0, 300), 512))
    lines = text.splitlines()
    assert lines[0] == " ".join(S.COLUMNS) and len(lines) == 7
    assert lines[2].split()[:9] == ["0", "300", "8", "5", "3", "2", "2", "6", "5"]
    assert lines[2].split()[-2:] == ["0.2500", "0.8333"]


def test_the_command_prints_the_table_and_exits_with_zero(runs, capsys):
    assert S.main(["--runs", str(runs)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == " ".join(S.COLUMNS)


def test_the_limit_changes_only_the_token_count(runs, capsys):
    assert S.main(["--runs", str(runs), "--limit", "300"]) == 0
    out = capsys.readouterr().out.splitlines()[2].split()
    # tokens of 300 or more: the completions of 300, 400, 512, 512 and 512 tokens
    assert out[S.COLUMNS.index("at_token_limit")] == "5" and out[S.COLUMNS.index("cut")] == "2"


@pytest.mark.parametrize("what", ["missing_folder", "two_folders", "missing_evaluation", "twice_evaluation", "no_strict_field"])
def test_a_missing_or_doubled_folder_or_evaluation_makes_the_script_refuse(tmp_path, capsys, what):
    for seed in S.SEEDS:
        write_row(tmp_path, seed)
    if what == "missing_folder":
        next(tmp_path.glob("5_study_*")).rename(tmp_path / "elsewhere")
    elif what == "two_folders":
        (tmp_path / "99_study_0.5b_verdict_only_s1").mkdir()
    elif what == "missing_evaluation":
        write_row(tmp_path, 2, items300=ITEMS_300)
        path = next(tmp_path.glob("9_study_*")) / "metrics.jsonl"
        path.write_text("\n".join(l for l in path.read_text().splitlines() if '"step": 300' not in l or '"hard"' in l) + "\n")
    elif what == "twice_evaluation":
        duplicate = {"phase": "eval", "set": "random", "step": 0, "item_fields": FIELDS, "items": ITEMS_0}
        with (next(tmp_path.glob("1_study_*")) / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(duplicate) + "\n")
    else:
        path = next(tmp_path.glob("1_study_*")) / "metrics.jsonl"
        path.write_text(path.read_text().replace('"strict"', '"other"'))
    assert S.main(["--runs", str(tmp_path)]) == 2
    assert capsys.readouterr().err.startswith("REFUSED:")


# ------------------------------------------------------------------ the real study

REAL = HERE / "runs" / "reasoning_icml_v2" / "runs" / "6322778"
needs_real = pytest.mark.skipif(not REAL.exists(), reason="the study records are withheld from the anonymous archive")


@needs_real
def test_the_cut_share_equals_the_logged_truncation_rate_and_the_mean_equals_the_logged_mean_for_every_evaluation():
    for row in S.table(REAL, (0, 300), 512):
        for line in (S.control_dir(REAL, row["seed"]) / "metrics.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("phase") == "eval" and r.get("set") == "random" and r.get("step") == row["step"]:
                assert row["cut_share"] == pytest.approx(r["truncation_rate"], abs=1e-12)
                assert row["mean_tokens"] == pytest.approx(r["mean_completion_tokens"], abs=1e-9)
                assert row["parsed"] == round((1 - r["strict_parse_fail_rate"]) * r["n"])


@needs_real
def test_the_real_table_has_the_counts_that_the_correction_entry_gives():
    rows = {(r["seed"], r["step"]): r for r in S.table(REAL, (0, 300), 512)}
    step0 = rows[(0, 0)]
    assert (step0["parsed"], step0["unparsed"], step0["cut"], step0["unparsed_cut"], step0["finished"],
            step0["parsed_of_finished"]) == (338, 62, 26, 25, 374, 337)
    for seed, expected in ((0, (296, 104, 91, 75, 309, 280)), (1, (263, 137, 140, 134, 260, 257)),
                           (2, (344, 56, 54, 53, 346, 343))):
        r = rows[(seed, 300)]
        assert (r["parsed"], r["unparsed"], r["cut"], r["unparsed_cut"], r["finished"], r["parsed_of_finished"]) == expected
