"""The WO-I manifests come from the design and differ only in the fields the design lets vary."""
import csv
import io
import json
from pathlib import Path

import pytest

import make_reasoning_manifest as MM
import reasoning_task as RT

HERE = Path(__file__).resolve().parent.parent
DS = "d" * 64
SEL_SHA = "e" * 64


def selection(one_five="ok"):
    return {"candidates_sha256": DS, "L": 384,
            "sizes": {"0.5b": {"status": "ok", "model_revision": RT.MODELS["0.5b"]["revision"]},
                      "1.5b": {"status": one_five, "model_revision": RT.MODELS["1.5b"]["revision"]}}}


def rules(kept=("0.5b", "1.5b"), S=200):
    return {"selection_sha256": SEL_SHA, "S": S, "sizes_kept": list(kept)}


def parse(data: bytes):
    return list(csv.DictReader(io.StringIO(data.decode())))


def test_calibration_manifest_has_four_rows_at_the_pinned_revisions():
    rows = MM.calibration_rows(DS)
    assert [(r["size"], r["split"]) for r in rows] == [("0.5b", "train"), ("0.5b", "test"),
                                                       ("1.5b", "train"), ("1.5b", "test")]
    assert [r["model_revision"] for r in rows] == [RT.MODELS["0.5b"]["revision"]] * 2 + [RT.MODELS["1.5b"]["revision"]] * 2
    data = MM.to_csv(RT.CALIBRATION_HEADER, rows)
    assert b"\r" not in data and data.endswith(b"\n")
    with pytest.raises(ValueError):
        MM.calibration_rows(DS, {"0.5b": "main", "1.5b": RT.MODELS["1.5b"]["revision"]})


def test_study_manifest_has_27_rows_and_rows_differ_only_in_arm_fields_and_seed():
    rows = MM.study_rows(selection(), rules(), DS, SEL_SHA)
    assert len(rows) == 27
    assert [r["task_id"] for r in rows] == [str(i) for i in range(1, 28)]
    assert {r["stage"] for r in rows[:15]} == {"1"} and {r["size"] for r in rows[:15]} == {"0.5b"}
    assert {r["stage"] for r in rows[15:]} == {"2"} and {r["size"] for r in rows[15:]} == {"1.5b"}
    arm_fields = {"task_id", "arm", "form", "w_task", "adv_variant", "group_filter", "seed"}
    for size in RT.SIZES:
        group = [r for r in rows if r["size"] == size]
        for key in set(RT.ARM_HEADER) - arm_fields:
            assert len({r[key] for r in group}) == 1, key
        assert sorted({(r["arm"], r["seed"]) for r in group}) == sorted(
            (a, str(s)) for a in RT.STUDY_ARMS[size] for s in RT.SEEDS)
    # The rows are seed-major, so a stage cut short keeps whole seeds.
    assert [r["seed"] for r in rows[:5]] == ["0"] * 5
    for r in rows:
        RT.validate_arm_row(r)


def test_a_dropped_size_leaves_fifteen_rows_and_a_wrong_digest_is_refused():
    rows = MM.study_rows(selection(), rules(kept=("0.5b",)), DS, SEL_SHA)
    assert len(rows) == 15 and {r["size"] for r in rows} == {"0.5b"}
    with pytest.raises(ValueError):
        MM.study_rows(selection(), rules(), "f" * 64, SEL_SHA)
    with pytest.raises(ValueError):
        MM.study_rows(selection(one_five="dropped"), rules(), DS, SEL_SHA)
    with pytest.raises(ValueError):
        MM.study_rows(selection(), rules(S=250), DS, SEL_SHA)


def test_smoke_manifest_holds_one_plain_row_per_selected_size():
    rows = MM.smoke_rows(selection(), DS, SEL_SHA)
    assert [(r["size"], r["arm"], r["seed"], r["steps"], r["stage"]) for r in rows] == [
        ("0.5b", "plain", "100", "20", "0"), ("1.5b", "plain", "100", "20", "0")]
    assert len(MM.smoke_rows(selection(one_five="dropped"), DS, SEL_SHA)) == 1


def test_the_committed_calibration_manifest_matches_the_generator():
    path = HERE / "slurm" / "manifest_reasoning_calibration.csv"
    ds_sha = RT.file_sha256(HERE / "configs" / "reasoning_dataset.json")
    rows = [RT.read_manifest_row(path, i) for i in range(1, 5)]
    for row in rows:
        RT.validate_calibration_row(row)
    # The 1.5B revision is the one the download job pins. The 0.5B revision is the laptop's snapshot
    # unless U0 finds another snapshot on the cluster, and READY Section 3.1 then regenerates this file with it.
    assert {r["model_revision"] for r in rows if r["size"] == "1.5b"} == {RT.MODELS["1.5b"]["revision"]}
    revisions = {size: next(r["model_revision"] for r in rows if r["size"] == size) for size in RT.SIZES}
    assert path.read_bytes() == MM.to_csv(RT.CALIBRATION_HEADER, MM.calibration_rows(ds_sha, revisions))


def test_smoke_and_study_rows_take_the_revision_that_the_calibration_sampled():
    other = "0123456789abcdef0123456789abcdef01234567"
    sel = selection()
    sel["sizes"]["0.5b"]["model_revision"] = other
    assert MM.smoke_rows(sel, DS, SEL_SHA)[0]["model_revision"] == other
    assert {r["model_revision"] for r in MM.study_rows(sel, rules(), DS, SEL_SHA) if r["size"] == "0.5b"} == {other}


# ------------------------------------------------------------------ design version 2


def selection_v2(L=512, status="ok"):
    return {"candidates_sha256": DS, "L": L,
            "sizes": {"0.5b": {"status": status, "model_revision": RT.MODELS["0.5b"]["revision"]},
                      "1.5b": {"status": "dropped"}}}


def test_pilot_manifest_has_three_one_change_rows_then_three_retry_rows():
    rows = MM.pilot_rows(selection_v2(), DS, SEL_SHA)
    assert [r["task_id"] for r in rows] == [str(i) for i in range(1, 7)]
    assert [(r["lr"], r["kl_beta"], r["skip_zero_adv"]) for r in rows] == [
        ("1e-06", "0.0", "0"), ("1e-05", "0.0", "1"), ("1e-05", "0.02", "0"),
        ("5e-07", "0.0", "0"), ("1e-05", "0.05", "0"), ("1e-05", "0.02", "1")]
    for r in rows:
        assert (r["kind"], r["stage"], r["size"], r["arm"], r["seed"], r["steps"], r["L"], r["fixedref_c"]) == (
            "pilot", "3", "0.5b", "verdict_only", "0", "100", "512", "0.0")
        assert tuple(r.keys()) == RT.ARM_HEADER_V2
        assert RT.validate_arm_row(r)["design_version"] == 2
    changes = [RT.pilot_changes(float(r["lr"]), float(r["kl_beta"]), int(r["skip_zero_adv"])) for r in rows]
    assert [len(c) for c in changes] == [1, 1, 1, 1, 1, 2]
    assert changes[5] == ["kl_beta", "skip_zero_adv"], "row 6 is the documented retry combination"
    data = MM.to_csv(RT.ARM_HEADER_V2, rows)
    assert b"\r" not in data and len(parse(data)) == 6


def test_pilot_rows_need_the_selected_size_at_L_512_and_the_bundled_dataset():
    with pytest.raises(ValueError, match="L = 512"):
        MM.pilot_rows(selection_v2(L=384), DS, SEL_SHA)
    with pytest.raises(ValueError, match="0.5B"):
        MM.pilot_rows(selection_v2(status="dropped"), DS, SEL_SHA)
    with pytest.raises(ValueError, match="not made from this dataset"):
        MM.pilot_rows(selection_v2(), "f" * 64, SEL_SHA)


@pytest.mark.parametrize("lr,kl,skip", [(1e-6, 0.0, 0), (1e-5, 0.02, 1), (5e-7, 0.05, 1), (1e-5, 0.0, 1)])
def test_study2_manifest_has_12_rows_sharing_the_chosen_levers(lr, kl, skip):
    rows = MM.study2_rows(selection_v2(), lr, kl, skip, DS, SEL_SHA)
    assert len(rows) == 12 and [r["task_id"] for r in rows] == [str(i) for i in range(1, 13)]
    assert {(r["lr"], r["kl_beta"], r["skip_zero_adv"]) for r in rows} == {(repr(lr), repr(kl), str(skip))}
    assert [r["seed"] for r in rows[:4]] == ["0"] * 4, "seed-major, so a stage cut short keeps whole seeds"
    assert sorted((r["arm"], r["seed"]) for r in rows) == sorted((a, str(s)) for a in RT.STUDY_ARMS_V2 for s in RT.SEEDS)
    for r in rows:
        assert (r["kind"], r["stage"], r["steps"], r["L"], r["size"]) == ("study", "4", "300", "512", "0.5b")
        assert r["fixedref_c"] == ("2.0" if r["arm"] == "restored" else "0.0")
        assert r["adv_variant"] == ("grpo_fixedref" if r["arm"] == "restored" else RT.ARMS[r["arm"]]["adv_variant"])
        assert RT.validate_arm_row(r)["design_version"] == 2
    MM.check_study2_shared(rows)


def test_study2_rows_that_do_not_share_the_levers_are_refused():
    rows = MM.study2_rows(selection_v2(), 1e-6, 0.0, 0, DS, SEL_SHA)
    rows[3] = dict(rows[3], lr="5e-07")
    with pytest.raises(ValueError, match="share one lr"):
        MM.check_study2_shared(rows)
    rows = MM.study2_rows(selection_v2(), 1e-6, 0.0, 0, DS, SEL_SHA)
    with pytest.raises(ValueError, match="every arm at every seed"):
        MM.check_study2_shared(rows[:-1])
    with pytest.raises(ValueError):
        MM.study2_rows(selection_v2(), 2e-6, 0.0, 0, DS, SEL_SHA)
    with pytest.raises(ValueError, match="L = 512"):
        MM.study2_rows(selection_v2(L=384), 1e-6, 0.0, 0, DS, SEL_SHA)


def test_the_committed_pilot_manifest_matches_the_generator_and_the_real_selection():
    path = HERE / "slurm" / "manifest_reasoning_pilot.csv"
    sel_path = HERE / "configs" / "reasoning_selection.json"
    selection = json.loads(sel_path.read_text())
    ds_sha, sel_sha = RT.file_sha256(HERE / "configs" / "reasoning_dataset.json"), RT.file_sha256(sel_path)
    assert path.read_bytes() == MM.to_csv(RT.ARM_HEADER_V2, MM.pilot_rows(selection, ds_sha, sel_sha))
    rows = [RT.read_manifest_row(path, i) for i in range(1, 7)]
    for row in rows:
        assert RT.validate_arm_row(row)["design_version"] == 2
        assert row["model_revision"] == selection["sizes"]["0.5b"]["model_revision"]
    # The study2 manifest is written after the pilot chooses the levers, so the generator has no default
    # for them and the command line requires all three.
    with pytest.raises(SystemExit):
        MM.main(["study2", "--out", "x.csv"])


def test_the_pilot_and_study2_command_lines_write_lf_only_files(tmp_path):
    ds = tmp_path / "ds.json"
    ds.write_text("{}")
    sel = tmp_path / "sel.json"
    sel_data = selection_v2()
    sel_data["candidates_sha256"] = RT.file_sha256(ds)
    sel.write_text(json.dumps(sel_data))
    out = tmp_path / "pilot.csv"
    assert MM.main(["pilot", "--dataset", str(ds), "--selection", str(sel), "--out", str(out)]) == 0
    assert b"\r" not in out.read_bytes() and len(parse(out.read_bytes())) == 6
    out2 = tmp_path / "study2.csv"
    assert MM.main(["study2", "--dataset", str(ds), "--selection", str(sel), "--out", str(out2),
                    "--lr", "1e-6", "--kl-beta", "0.02", "--skip-zero-adv", "1"]) == 0
    rows = parse(out2.read_bytes())
    assert len(rows) == 12 and {r["lr"] for r in rows} == {"1e-06"}
    assert MM.main(["study2", "--dataset", str(ds), "--selection", str(sel), "--out", str(tmp_path / "bad.csv"),
                    "--lr", "3e-6", "--kl-beta", "0.0", "--skip-zero-adv", "0"]) == 2
    assert not (tmp_path / "bad.csv").exists()


def test_the_command_line_writes_lf_only_files(tmp_path):
    out = tmp_path / "cal.csv"
    assert MM.main(["calibration", "--out", str(out)]) == 0
    assert b"\r" not in out.read_bytes() and len(parse(out.read_bytes())) == 4
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(selection()))
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules()))
    assert MM.main(["study", "--selection", str(sel), "--smoke-rules", str(rules_path),
                    "--out", str(tmp_path / "s.csv")]) == 2
