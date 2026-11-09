#!/usr/bin/env python3
"""Write slurm/manifest_speed_bounded.csv, the Block A' manifest of the ICML 2027 plan (November 2026).

The manifest uses the v4 layout. It is the v2 layout followed by three fields: `fixedref_c`, the shift
of the fixed-reference variant; `skip_zero_adv`, 1 when a zero-advantage group stays out of the update;
and `group_size`, the number of completions per prompt. slurm/cancel_arms.sbatch passes those three
only for this layout, and it passes the skip only when the field is 1.

Every row runs the task `speed_bounded` with the zero-advantage skip on and no KL term. The limits come
from configs/regimes.yaml, so regime A is the hard limit and regime B is the easy one.

  regime A, K = 4, seeds 0 to 4   none (no penalty), binary_w5, binary_w5000, binary_w5_fixedref
  regime A, K = 8, seeds 0 to 4   binary_w5_k8, binary_w5_fixedref_k8 (the exploration control)
  regime B, K = 4, seeds 0 to 2   none, binary_w5, binary_w5_fixedref

That is 39 rows. The two fixed-reference arms differ from binary_w5 and binary_w5_k8 only in the
advantage variant and the shift, so each pair is identical on every mixed group by construction. The
arm names are the plan's and are not the names analyze.arm_name would give, because that function does
not yet know the fixed-reference variant.

Each row is validated before it is written: the arm is one of rewards.ARMS, the variant is one the
trainer accepts, the shift is 0.0 unless the variant is grpo_fixedref and positive when it is, the
group size is 4 or 8, the skip is 0 or 1, the KL weight is 0.0, and the regime's limit is the one
configs/regimes.yaml gives. The file is written with LF line endings only.

Usage: python make_bounded_manifest.py            write the manifest
       python make_bounded_manifest.py --check    exit 1 when the committed file differs from this output
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

import rewards as RW

HERE = Path(__file__).resolve().parent
OUT_REL = "slurm/manifest_speed_bounded.csv"

V2_HEADER = ("task_id", "task", "regime", "arm_name", "arm", "w", "w_task", "vmax", "seed",
             "adv_variant", "group_filter", "kl_beta")
V4_HEADER = V2_HEADER + ("fixedref_c", "skip_zero_adv", "group_size")

TASK = "speed_bounded"
W_TASK = 1.0
KL_BETA = 0.0
SKIP_ZERO_ADV = 1
GROUP_FILTER = "none"
FIXEDREF_VARIANT = "grpo_fixedref"
FIXEDREF_C = 2.0
GROUP_SIZES = (4, 8)
W_LIGHT = 5.0
W_HEAVY = 5000.0
SEEDS_A = (0, 1, 2, 3, 4)
SEEDS_B = (0, 1, 2)

# (arm_name, arm, w, adv_variant, fixedref_c, group_size), in the order the plan lists them.
CELLS_A_K4 = (
    ("none", "none", 0.0, "grpo", 0.0, 4),
    ("binary_w5", "binary", W_LIGHT, "grpo", 0.0, 4),
    ("binary_w5000", "binary", W_HEAVY, "grpo", 0.0, 4),
    ("binary_w5_fixedref", "binary", W_LIGHT, FIXEDREF_VARIANT, FIXEDREF_C, 4),
)
CELLS_A_K8 = (
    ("binary_w5_k8", "binary", W_LIGHT, "grpo", 0.0, 8),
    ("binary_w5_fixedref_k8", "binary", W_LIGHT, FIXEDREF_VARIANT, FIXEDREF_C, 8),
)
CELLS_B_K4 = (
    ("none", "none", 0.0, "grpo", 0.0, 4),
    ("binary_w5", "binary", W_LIGHT, "grpo", 0.0, 4),
    ("binary_w5_fixedref", "binary", W_LIGHT, FIXEDREF_VARIANT, FIXEDREF_C, 4),
)
BLOCKS = (("A", CELLS_A_K4, SEEDS_A), ("A", CELLS_A_K8, SEEDS_A), ("B", CELLS_B_K4, SEEDS_B))


def regime_limits(here: Path = HERE) -> dict:
    """The speed limit of each regime, read from configs/regimes.yaml."""
    doc = yaml.safe_load((here / "configs" / "regimes.yaml").read_text())
    limits = {}
    for name, entry in doc.items():
        v = entry.get("v_max") if isinstance(entry, dict) else None
        if v is None:
            raise ValueError(f"regime {name!r} has no v_max in configs/regimes.yaml")
        limits[str(name)] = float(v)
    return limits


def validate_row(row: dict, limits: dict) -> None:
    """Raise ValueError when a row is not one the trainer and the submission script define."""
    if tuple(row) != V4_HEADER:
        raise ValueError(f"row {row.get('task_id')!r} has fields {tuple(row)}, not the v4 header")
    if row["task"] != TASK:
        raise ValueError(f"row {row['task_id']} runs task {row['task']!r}, not {TASK!r}")
    if row["arm"] not in RW.ARMS:
        raise ValueError(f"row {row['task_id']} has arm {row['arm']!r}, not one of {RW.ARMS}")
    if row["adv_variant"] not in RW.TRAINER_ADV_VARIANTS:
        raise ValueError(f"row {row['task_id']} has variant {row['adv_variant']!r}, not one of "
                         f"{RW.TRAINER_ADV_VARIANTS}")
    if row["group_filter"] not in RW.GROUP_FILTERS:
        raise ValueError(f"row {row['task_id']} has filter {row['group_filter']!r}")
    c = float(row["fixedref_c"])
    if row["adv_variant"] == FIXEDREF_VARIANT:
        if not c > 0.0:
            raise ValueError(f"row {row['task_id']} runs {FIXEDREF_VARIANT} with a shift of {c!r}")
    elif c != 0.0:
        raise ValueError(f"row {row['task_id']} has a shift of {c!r} under variant {row['adv_variant']!r}")
    if int(row["group_size"]) not in GROUP_SIZES:
        raise ValueError(f"row {row['task_id']} has group size {row['group_size']!r}, not one of {GROUP_SIZES}")
    if row["skip_zero_adv"] not in ("0", "1"):
        raise ValueError(f"row {row['task_id']} has skip_zero_adv {row['skip_zero_adv']!r}, not 0 or 1")
    if float(row["kl_beta"]) != KL_BETA:
        raise ValueError(f"row {row['task_id']} has a KL weight of {row['kl_beta']!r}")
    if row["arm"] == "none" and float(row["w"]) != 0.0:
        raise ValueError(f"row {row['task_id']} has the no-penalty arm at a nonzero weight")
    if row["arm"] != "none" and not float(row["w"]) > 0.0:
        raise ValueError(f"row {row['task_id']} has a penalty arm at weight {row['w']!r}")
    if row["regime"] not in limits:
        raise ValueError(f"row {row['task_id']} names regime {row['regime']!r}, which configs/regimes.yaml lacks")
    if float(row["vmax"]) != limits[row["regime"]]:
        raise ValueError(f"row {row['task_id']} has vmax {row['vmax']!r} for regime {row['regime']!r}, "
                         f"and configs/regimes.yaml gives {limits[row['regime']]!r}")
    for k, v in row.items():
        if v != v.strip() or v == "":
            raise ValueError(f"row {row['task_id']} has an empty or untrimmed field {k!r}")


def rows(here: Path = HERE) -> list:
    """The 39 rows, by block, then cell, then seed, each validated."""
    limits = regime_limits(here)
    out = []
    for regime, cells, seeds in BLOCKS:
        for arm_name, arm, w, variant, c, k in cells:
            for seed in seeds:
                row = {"task_id": str(len(out) + 1), "task": TASK, "regime": regime, "arm_name": arm_name,
                       "arm": arm, "w": repr(float(w)), "w_task": repr(float(W_TASK)),
                       "vmax": repr(limits[regime]), "seed": str(seed), "adv_variant": variant,
                       "group_filter": GROUP_FILTER, "kl_beta": repr(float(KL_BETA)),
                       "fixedref_c": repr(float(c)), "skip_zero_adv": str(SKIP_ZERO_ADV),
                       "group_size": str(k)}
                validate_row(row, limits)
                out.append(row)
    return out


def text(here: Path = HERE) -> str:
    """The manifest as one string with LF line endings and a trailing newline."""
    lines = [",".join(V4_HEADER)] + [",".join(r[k] for k in V4_HEADER) for r in rows(here)]
    return "\n".join(lines) + "\n"


def read_rows(path: Path) -> list:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    built = text()
    target = HERE / OUT_REL
    if a.check:
        if not target.exists() or target.read_bytes() != built.encode("ascii"):
            print(f"STALE: {OUT_REL} differs from make_bounded_manifest.py output", file=sys.stderr)
            return 1
        return 0
    target.write_bytes(built.encode("ascii"))
    print(f"wrote {OUT_REL}: {built.count(chr(10)) - 1} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
