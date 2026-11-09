#!/usr/bin/env python3
"""Generate the WO-I manifests from the design, and refuse any field outside it.

calibration  slurm/manifest_reasoning_calibration.csv, the four stage 0 calibration rows.
smoke        slurm/manifest_reasoning_smoke.csv, one stage 0 smoke row per selected size.
study        slurm/manifest_reasoning.csv, 15 rows at 0.5B (stage 1) and 12 at 1.5B (stage 2),
             or the 15 rows alone when the 1.5B size was dropped.
pilot        slurm/manifest_reasoning_pilot.csv, the six stage 3 pilot rows of design version 2
             (ICML 2027, November 2026): three one-change rows and three retry rows.
study2       slurm/manifest_reasoning_study2.csv, the 12 stage 4 rows of the version-2 study, with
             lr, kl_beta and skip_zero_adv given on the command line after the pilot has chosen them.

The pilot rows, in task order, each change one lever of the version-1 loop at 0.5B, verdict_only,
seed 0, 100 steps and L 512: (1) lr 1e-6; (2) the zero-advantage skip; (3) kl_beta 0.02. The retry
rows are (4) lr 5e-7; (5) kl_beta 0.05; (6) the skip together with kl_beta 0.02. Row 6 is the one
two-change row, the retry combination of the plan, and reasoning_task.PILOT_RETRY_COMBINATION names it.

Every row goes through reasoning_task.validate_calibration_row or validate_arm_row before it is
written, which is the same check the cluster code applies to the row it runs. The files are
LF-only. The model revisions come from the calibration provenance recorded in the selection, so
a smoke or study row always names the snapshot that the calibration sampled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import reasoning_task as RT

HERE = Path(__file__).resolve().parent


def to_csv(header, rows: List[Dict[str, str]]) -> bytes:
    lines = [",".join(header)] + [",".join(str(r[k]) for k in header) for r in rows]
    text = "\n".join(lines) + "\n"
    if "\r" in text or any("," in str(v) for r in rows for v in r.values()):
        raise ValueError("a manifest field holds a comma or a carriage return")
    return text.encode("utf-8")


def calibration_rows(dataset_sha256: str, revisions: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    revisions = revisions or {s: RT.MODELS[s]["revision"] for s in RT.SIZES}
    rows, task = [], 1
    for size in RT.SIZES:
        for split in ("train", "test"):
            row = {"task_id": str(task), "kind": "calibration", "stage": "0", "size": size,
                   "model_id": RT.MODELS[size]["model_id"], "model_revision": revisions[size], "split": split,
                   "samples": str(RT.CAL_SAMPLES), "max_new_tokens": str(RT.CAL_MAX_NEW_TOKENS),
                   "chunk_problems": str(RT.CAL_CHUNK_PROBLEMS), "seed": "0", "dataset_sha256": dataset_sha256}
            RT.validate_calibration_row(row)
            rows.append(row)
            task += 1
    return rows


def _arm_row_fields(task: int, kind: str, stage: int, size: str, revision: str, arm: str, seed: int, steps: int,
                    L: int, dataset_sha256: str, selection_sha256: str) -> Dict[str, str]:
    spec = RT.ARMS[arm]
    return {"task_id": str(task), "kind": kind, "stage": str(stage), "size": size,
            "model_id": RT.MODELS[size]["model_id"], "model_revision": revision, "arm": arm,
            "form": spec["form"], "w": repr(RT.W), "w_task": repr(float(spec["w_task"])),
            "adv_variant": spec["adv_variant"], "group_filter": spec["group_filter"], "seed": str(seed),
            "steps": str(steps), "L": str(L), "micro_batch": str(RT.MICRO_BATCH),
            "dataset_sha256": dataset_sha256, "selection_sha256": selection_sha256}


def _arm_row(task: int, kind: str, stage: int, size: str, revision: str, arm: str, seed: int, steps: int,
             L: int, dataset_sha256: str, selection_sha256: str) -> Dict[str, str]:
    row = _arm_row_fields(task, kind, stage, size, revision, arm, seed, steps, L, dataset_sha256, selection_sha256)
    RT.validate_arm_row(row)
    return row


def selected_sizes(selection: dict) -> List[str]:
    return [s for s in RT.SIZES if selection["sizes"][s]["status"] == "ok"]


# ------------------------------------------------------------------ design version 2


def _arm_row_v2(task: int, kind: str, stage: int, revision: str, arm: str, seed: int, steps: int, L: int,
                lr: float, kl_beta: float, skip_zero_adv: int, dataset_sha256: str,
                selection_sha256: str) -> Dict[str, str]:
    row = _arm_row_fields(task, kind, stage, RT.STUDY_SIZE_V2, revision, arm, seed, steps, L, dataset_sha256,
                          selection_sha256)
    c = RT.FIXEDREF_C if arm == "restored" else 0.0
    row.update({"lr": repr(float(lr)), "kl_beta": repr(float(kl_beta)), "skip_zero_adv": str(int(skip_zero_adv)),
                "fixedref_c": repr(float(c))})
    RT.validate_arm_row(row)
    return row


PILOT_LEVERS = (
    # (lr, kl_beta, skip_zero_adv), one change each, then the three retry rows.
    (1e-6, 0.0, 0),
    (RT.LR_DEFAULT, 0.0, 1),
    (RT.LR_DEFAULT, 0.02, 0),
    (5e-7, 0.0, 0),
    (RT.LR_DEFAULT, 0.05, 0),
    (RT.PILOT_RETRY_COMBINATION["lr"], RT.PILOT_RETRY_COMBINATION["kl_beta"], RT.PILOT_RETRY_COMBINATION["skip_zero_adv"]),
)


def _check_v2_selection(selection: dict, dataset_sha256: str) -> str:
    if selection["candidates_sha256"] != dataset_sha256:
        raise ValueError("the selection was not made from this dataset file")
    size = selection["sizes"].get(RT.STUDY_SIZE_V2, {})
    if size.get("status") != "ok":
        raise ValueError("the version-2 block needs the selected 0.5B size")
    if int(selection["L"]) != RT.PILOT_L:
        raise ValueError(f"the version-2 block runs at L = {RT.PILOT_L}, and the selection has L = {selection['L']}")
    return size["model_revision"]


def pilot_rows(selection: dict, dataset_sha256: str, selection_sha256: str) -> List[Dict[str, str]]:
    """Return the six pilot rows: three one-change rows, then the three retry rows."""
    revision = _check_v2_selection(selection, dataset_sha256)
    return [_arm_row_v2(i, "pilot", RT.PILOT_STAGE, revision, RT.PILOT_ARM, RT.PILOT_SEED, RT.PILOT_STEPS,
                        RT.PILOT_L, lr, kl, skip, dataset_sha256, selection_sha256)
            for i, (lr, kl, skip) in enumerate(PILOT_LEVERS, start=1)]


def study2_rows(selection: dict, lr: float, kl_beta: float, skip_zero_adv: int, dataset_sha256: str,
                selection_sha256: str) -> List[Dict[str, str]]:
    """Return the 12 version-2 study rows in seed-major order, every row sharing one lr, kl_beta and skip."""
    revision = _check_v2_selection(selection, dataset_sha256)
    rows, task = [], 1
    for seed in RT.SEEDS:
        for arm in RT.STUDY_ARMS_V2:
            rows.append(_arm_row_v2(task, "study", RT.STUDY_STAGE_V2, revision, arm, seed, RT.STUDY_STEPS_V2,
                                    RT.PILOT_L, lr, kl_beta, skip_zero_adv, dataset_sha256, selection_sha256))
            task += 1
    check_study2_shared(rows)
    return rows


def check_study2_shared(rows: List[Dict[str, str]]) -> None:
    """Refuse a version-2 study manifest whose rows do not share lr, kl_beta and skip_zero_adv.

    validate_arm_row checks each row alone, so this is the one place the shared-lever rule is enforced."""
    for key in ("lr", "kl_beta", "skip_zero_adv"):
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f"every version-2 study row must share one {key}")
    if sorted((r["arm"], r["seed"]) for r in rows) != sorted((a, str(s)) for a in RT.STUDY_ARMS_V2 for s in RT.SEEDS):
        raise ValueError("the version-2 study is every arm at every seed, once")


def smoke_rows(selection: dict, dataset_sha256: str, selection_sha256: str) -> List[Dict[str, str]]:
    if selection["candidates_sha256"] != dataset_sha256:
        raise ValueError("the selection was not made from this dataset file")
    return [_arm_row(i, "smoke", 0, size, selection["sizes"][size]["model_revision"], "plain", RT.SMOKE_SEED,
                     RT.SMOKE_STEPS, selection["L"], dataset_sha256, selection_sha256)
            for i, size in enumerate(selected_sizes(selection), start=1)]


def study_rows(selection: dict, rules: dict, dataset_sha256: str, selection_sha256: str) -> List[Dict[str, str]]:
    """Return the study rows in seed-major order, so a stage cut short leaves whole seeds of every arm."""
    if selection["candidates_sha256"] != dataset_sha256 or rules["selection_sha256"] != selection_sha256:
        raise ValueError("the smoke rules, the selection and the dataset do not belong together")
    rows, task = [], 1
    for size in rules["sizes_kept"]:
        if selection["sizes"][size]["status"] != "ok":
            raise ValueError(f"the {size} size is kept by the smoke rules but not selected")
        for seed in RT.SEEDS:
            for arm in RT.STUDY_ARMS[size]:
                rows.append(_arm_row(task, "study", RT.STAGE_OF_SIZE[size], size,
                                     selection["sizes"][size]["model_revision"], arm, seed, rules["S"],
                                     selection["L"], dataset_sha256, selection_sha256))
                task += 1
    if "0.5b" not in rules["sizes_kept"]:
        raise ValueError("the study needs the 0.5B size")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibration", "smoke", "study", "pilot", "study2"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", type=Path, default=HERE / "configs" / "reasoning_dataset.json")
        if name == "calibration":
            p.add_argument("--revision-0p5b", default=RT.MODELS["0.5b"]["revision"])
            p.add_argument("--revision-1p5b", default=RT.MODELS["1.5b"]["revision"])
        else:
            p.add_argument("--selection", type=Path, default=HERE / "configs" / "reasoning_selection.json")
        if name == "study":
            p.add_argument("--smoke-rules", type=Path, default=HERE / "configs" / "reasoning_smoke_rules.json")
        if name == "study2":
            # These three are chosen after the pilot. There is no default, so the file cannot be written
            # from a loop nobody chose.
            p.add_argument("--lr", type=float, required=True)
            p.add_argument("--kl-beta", type=float, required=True)
            p.add_argument("--skip-zero-adv", type=int, choices=(0, 1), required=True)
        default_out = {"calibration": "manifest_reasoning_calibration.csv", "smoke": "manifest_reasoning_smoke.csv",
                       "study": "manifest_reasoning.csv", "pilot": "manifest_reasoning_pilot.csv",
                       "study2": "manifest_reasoning_study2.csv"}[name]
        p.add_argument("--out", type=Path, default=HERE / "slurm" / default_out)
    args = ap.parse_args(argv)
    dataset_sha = RT.file_sha256(args.dataset)
    try:
        if args.cmd == "calibration":
            data = to_csv(RT.CALIBRATION_HEADER, calibration_rows(
                dataset_sha, {"0.5b": args.revision_0p5b, "1.5b": args.revision_1p5b}))
        else:
            selection = json.loads(args.selection.read_text())
            sel_sha = RT.file_sha256(args.selection)
            if args.cmd == "smoke":
                data = to_csv(RT.ARM_HEADER, smoke_rows(selection, dataset_sha, sel_sha))
            elif args.cmd == "study":
                rules = json.loads(args.smoke_rules.read_text())
                data = to_csv(RT.ARM_HEADER, study_rows(selection, rules, dataset_sha, sel_sha))
            elif args.cmd == "pilot":
                data = to_csv(RT.ARM_HEADER_V2, pilot_rows(selection, dataset_sha, sel_sha))
            else:
                data = to_csv(RT.ARM_HEADER_V2, study2_rows(selection, args.lr, args.kl_beta, args.skip_zero_adv,
                                                            dataset_sha, sel_sha))
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    args.out.write_bytes(data)
    print(json.dumps({"out": str(args.out), "sha256": hashlib.sha256(data).hexdigest(),
                      "rows": data.count(b"\n") - 1}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
