#!/usr/bin/env python3
"""Score the GSM8K study at a learning rate of 1e-6 against the D-gate and D1 to D4 of PREDICTIONS.md.

The rules come from the entry of PREDICTIONS.md headed "Mixed-reward GRPO on grade-school math at a
learning rate of 1e-6, study recorded 7 November 2026, before any study row". The pilot entries of 6 and
7 November 2026 supply the meaning of "stopped early", "parse strictly" and "admitted". This module only
reads. It writes three new files into a new output directory and never edits PREDICTIONS.md, LEDGER.md,
a run directory, a manifest or a configuration file.

Admission comes first. A row is scorable when its run directory passes every check of this module. A
scorable row is admitted when it also finished all 300 steps and did not stop early. A row that the
trainer's time guard stopped early is scorable and not admitted. It is scored on the evaluations it
wrote, with its stopped_early flag and steps_completed beside its verdict, and it never enters a
comparison between two arms, because a pair of arms on one seed is compared only if both rows are
admitted. The scorer refuses, with exit status 3 and a message that starts with REFUSED, when any of the
12 manifest rows is not scorable and is not named in the not-run file. Nothing is scored before then.

The rules in short:
  D-gate: on at least 2 of 3 seeds the verdict-only random held-out accuracy at step 300 is within 0.05 of
          its own step-0 value and at least 0.80 of its step-300 random held-out completions parse strictly.
  D1:     on at least 2 of 3 seeds plain's mean training completion length over steps 281 to 300 is below
          half of the same seed's verdict-only value, and its step-300 accuracy is at least 0.10 below it.
  D2:     on at least 2 of 3 seeds restored's step-300 accuracy is at least 0.10 below verdict only.
  D3:     on at least 2 of 3 seeds the verifier filter's step-300 accuracy is within 0.05 of verdict only.
  D4:     D4 is a measurement with no prediction, and the scorer records it for every row.

The entry says that D4 is recorded "For every row", and it ties admission only to the comparison of a pair of arms. The
scorer therefore records D4, the random held-out values at step 0 and at the last step, and the per-step series for every
row whose run directory holds readable records, whether or not the row is scorable. A row that is not scorable keeps its
problems beside these values and is marked measured_only. No gate, D1, D2, D3, clause or branch reads a measured_only value.
A row with no run directory, no usable steps_completed, or training records that do not cover every completed step has
no D4, and its problems say why. A logged random held-out value that the record of the row contradicts is left out of the
row, and the mismatch block holds both readings of it (see the reading d4_every_row).

Every comparison uses integer counts out of 400 completions. Every random held-out accuracy is a multiple
of 1/400, so a count comparison gives the same verdict as a float comparison with the entry's tolerance of
1e-12 on every possible input. The strict bounds (below half) take no tolerance.

The command line has one subcommand:
  python analyze_study2.py study --manifest M --frozen F --runs R --logs L --out-dir O
         [--dataset D --selection S [--recheck]] [--not-run N]
The same subcommand is reached through "python analyze_reasoning.py study ...", which imports this module on
demand. The scorer writes study_d.json, study_d.md and study_d4_series.json into the new output directory and
never overwrites them. It exits with status 0 when it writes a report, including a report of a failed control,
and with status 3 and a message that starts with REFUSED when it refuses. Status 3 is the refusal status of the
pilot-d0 mode and of the version-1 mode of analyze_reasoning.py, and status 2 stays reserved for a wrong command
line, which argparse reports. The three files appear together or not at all.

The entry leaves some points open: how a seed without an admitted pair counts, whether "within 0.10" includes a gap of
exactly 0.10, and whether a row whose log disagrees with its own items or completions is admitted. The scorer reads each
point in the cautious way, and the report also prints the verdicts and the branch under every other reading that the
specification lists (ALTERNATIVE_READINGS). A verdict or a branch that depends on the choice is marked VERDICT DEPENDS ON
READING.

The run directories can sit in any directory. The array job id of a row comes from its provenance.json, and the Slurm
logs are matched on that id and on the task id, so a retry under a new array id needs no special handling. The report
names every array job id that it saw, in the run directories and in the names of the log files, and it lists the log
files of an array that has no run directory (for example an array that failed at launch). The report carries no
absolute path, and no NaN or infinity, in any text that it copies from the run files.

This module imports analyze_reasoning only when a function needs it, so importing this module never loads
analyze_reasoning and the two modules cannot import each other in a cycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import reasoning_task as RT

# ------------------------------------------------------------------ the numbers fixed by the entry

# The study entry names the manifest checksum and the two source digests, so they are constants here.
# The scorer never takes an expected value from the file it checks.
MANIFEST_SHA256 = "13d32585c66d853ae05a2416895f7e406b9cd6af97e86ad9136f2d7aef494aaa"
FROZEN_REASONING_DIGEST = "80d48c701f241c1d41e1a4680029dd7cee31c1454a9f9b28f6b661b89bce4858"
FROZEN_TRAINING_DIGEST = "65d67374f425580b8d4ac7e3d21e4d5fa08f705f6816d7ee8239c8d297d95690"
# These two checksums come from the provenance of the three pilot rows (the pilot array), not from the entry.
# Every study row loads the same model snapshot, so a different value means a different model cache.
EXPECTED_MODEL_FILE_SHA256 = {"config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
                              "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"}
# The pilot outcomes entry records an accuracy of 0.435 and a strict parse rate of 0.845 at step 0 on 400
# completions. The study rows are expected to match (UNVERIFIED), and a difference is reported and never gates.
PILOT_STEP0_COUNTS = (174, 338)

SEEDS = (0, 1, 2)
ARMS = tuple(RT.STUDY_ARMS_V2)
CONTROL, PLAIN, RESTORED, FILTER = "verdict_only", "plain", "restored", "vfilter"
STEPS = RT.STUDY_STEPS_V2
N_RANDOM = RT.RANDOM_SET_SIZE * RT.EVAL_SAMPLES_RANDOM
GROUPS = RT.GROUPS_PER_STEP
GROUP_SIZE = RT.GROUP_SIZE
PER_STEP = GROUPS * GROUP_SIZE
WINDOW = (281, 300)
WINDOW_STEPS = WINDOW[1] - WINDOW[0] + 1
WINDOW_COMPLETIONS = WINDOW_STEPS * PER_STEP

ACC_BAND = 0.05
GAP = 0.10
PARSE_MIN = 0.80
LOOSE_BAND = 0.10
FLOAT_TOL = 1e-12
ITEM_TOL = 1e-9
SEEDS_NEEDED = 2
# The same bounds as counts of the 400 completions of a random held-out evaluation.
ACC_BAND_COUNTS = 20
GAP_COUNTS = 40
PARSE_MIN_COUNT = 320
LOOSE_BAND_COUNTS = 40
# A training step has 24 completions. A parse rate strictly below 0.5 means at most 11 of them parse.
PARSE_BELOW_HALF_MAX = PER_STEP // 2 - 1

assert ACC_BAND_COUNTS == round(ACC_BAND * N_RANDOM) and GAP_COUNTS == round(GAP * N_RANDOM)
assert PARSE_MIN_COUNT == round(PARSE_MIN * N_RANDOM) and LOOSE_BAND_COUNTS == round(LOOSE_BAND * N_RANDOM)

STATUSES = ("admitted", "stopped_early", "refused", "no_run_directory", "ambiguous", "not_run")
HOLDS, FAILS, UNDETERMINED = "HOLDS", "FAILS", "UNDETERMINED"
# REFUSED_STATUS is the exit status of a refusal. The pilot-d0 mode and the version-1 mode of analyze_reasoning.py
# return 3 as well, and argparse keeps 2 for a wrong command line, so a caller can tell a refusal from a typing mistake.
REFUSED_STATUS = 3

# The held-equal settings of config.json. The learning rate is not here because the manifest row fixes it.
HELD_EQUAL = {"device": "cuda", "weight_decay": 0.0, "clip_coef": 0.2, "grad_clip": 1.0, "eval_seed": RT.EVAL_SEED,
              "eval_chunk_sequences": RT.EVAL_CHUNK_SEQUENCES, "mid_eval_every": RT.EVAL_EVERY,
              "group_size": RT.GROUP_SIZE, "groups_per_step": RT.GROUPS_PER_STEP, "w": RT.W}
SAMPLER = {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.0}

# ------------------------------------------------------------------ the sentences of the decision rule

# The three conditions of the entry's decision rule use the entry's words. NO_BRANCH_A, NO_BRANCH_B and PENDING
# are this scorer's own descriptions of the cases that the entry's three sentences do not cover.
BRANCH_LINES = {
    "MAIN": "the gate passes and D1 and D3 hold",
    "NULL": "the gate passes and D1 fails, so that plain stays within 0.10 of verdict only on at least 2 of 3 seeds",
    "FAILED_CONTROL": "the gate fails",
    "NO_BRANCH_A": "the gate passes, D1 holds and D3 fails",
    "NO_BRANCH_B": "the gate passes, D1 fails, and plain stays within 0.10 of verdict only on fewer than 2 seeds",
    "PENDING": "the gate, D1, D3 or the within-0.10 clause is UNDETERMINED",
}
# Each sentence is the entry's own sentence under that branch, word for word. Only the first letter is capitalized.
# A test compares every one of them with the text of PREDICTIONS.md.
PAPER_STATEMENTS = {
    "MAIN": ("The paper reports that on grade-school math at a learning rate of 1e-6 the plain mixed reward "
             "loses accuracy while the verifier filter does not, and it reports D2 as it falls out."),
    "NULL": ("The paper reports that cancellation has no measurable consequence in this loop at this learning "
             "rate, and it claims no benchmark effect."),
    "FAILED_CONTROL": ("The block is a failed control and the paper keeps the recorded collapse "
                       "as its statement on this benchmark."),
    "NO_BRANCH_A": None,
    "NO_BRANCH_B": None,
    "PENDING": None,
}
NO_BRANCH_LINE = "no branch of the decision rule applies exactly"
PENDING_LINE = ("no branch is selected until the missing or non-admitted rows are admitted "
                "or the authors record a decision")

# ------------------------------------------------------------------ the readings chosen for each ambiguity

def _normalize_readings(raw) -> tuple:
    """Give every entry of the readings an "alternatives" list, which names the ids of ALTERNATIVE_READINGS printed for it.

    An entry with alternative_printed True and no list of its own prints one alternative with the id of the entry."""
    out = []
    for entry in raw:
        entry = dict(entry)
        entry.setdefault("alternatives", [entry["id"]] if entry["alternative_printed"] else [])
        if entry["alternative_printed"] != bool(entry["alternatives"]):
            raise ValueError(f"the reading {entry['id']} has alternative_printed {entry['alternative_printed']} and the alternatives "
                             f"{entry['alternatives']}")
        out.append(entry)
    return tuple(out)


# Each entry records one reading that the specification lists as ambiguous, and the reading this scorer uses.
# "alternative_printed" is True when the report also prints the verdicts under another reading, and "alternatives" lists
# the ids of those readings in ALTERNATIVE_READINGS.
_READINGS_RAW = (
    {"id": "conjunction_per_seed", "alternative_printed": True,
     "ambiguity": "Whether the word and inside the gate and inside D1 binds within one seed.",
     "chosen": ("Both conditions must hold on the same seed, and the claim needs at least 2 such seeds. "
                "The verdict that counts each condition over seeds on its own is printed beside it.")},
    {"id": "missing_seed_count", "alternative_printed": True,
     "ambiguity": "What at least 2 of 3 seeds means when a seed has a row that is not admitted.",
     "chosen": ("Three-valued. With c comparable seeds and k seeds that satisfy the claim, the claim HOLDS "
                "when k is at least 2, FAILS when k plus (3 minus c) is below 2, and is otherwise UNDETERMINED. "
                "The verdicts with a denominator that stays at 3 are printed beside them. In that reading a seed "
                "without an admitted pair counts as not satisfying, so the claim FAILS whenever k is below 2. "
                "The reading that rescales the threshold to the admitted seeds is not printed, because it would let "
                "one seed carry a claim that the entry sets at 2 seeds.")},
    {"id": "held_out_mismatch", "alternative_printed": True,
     "alternatives": ["mismatch_logged_value", "mismatch_recomputed_value"],
     "ambiguity": ("What the scorer does with a row whose logged random held-out accuracy or strict parse failure rate "
                   "disagrees with the items or the completions of the same record."),
     "chosen": ("The entry does not define admitted, and the pilot list does not name this case. The scorer refuses "
                "the row, so the study is not scored until the authors name the row in the not-run file. When the "
                "author does, the report also scores the study with the logged value of the row and with the "
                "recomputed value, and it prints the verdicts and the branch under each, because the branch can "
                "depend on the choice.")},
    {"id": "admitted_definition", "alternative_printed": False,
     "ambiguity": "What admitted means for a study row and how a stopped-early row enters a pair.",
     "chosen": ("Admitted means scorable with 300 completed steps and stopped_early false. A stopped-early row "
                "is scorable, is scored on the evaluations it wrote, and never enters a pair.")},
    {"id": "gate_stopped_early", "alternative_printed": True,
     "ambiguity": "How a stopped-early verdict-only row is treated inside the gate.",
     "chosen": ("It is scored on its last written random evaluation, flagged, and counted. A second gate that "
                "treats stopped-early rows as not comparable is printed beside the first.")},
    {"id": "float_tolerance", "alternative_printed": True,
     "ambiguity": "The scope and direction of the 1e-12 float tolerance.",
     "chosen": ("Comparisons use integer counts out of 400, which gives the verdicts of a 1e-12 slack on every "
                "non-strict bound. Strict bounds take no tolerance. The verdicts of a plain float comparison "
                "without slack, apart from the entry's own tolerance on within 0.05, are printed beside them.")},
    {"id": "within_two_sided", "alternative_printed": False,
     "ambiguity": "Whether within 0.05 is one-sided or two-sided.",
     "chosen": ("Two-sided, as the entry defines it. Signed differences and a flag for a rise of more than "
                "0.05 are printed, and no one-sided verdict is computed.")},
    {"id": "length_mean_and_window", "alternative_printed": False,
     "ambiguity": "What the mean training completion length averages and which steps are 281 to 300.",
     "chosen": ("The integer sum T of n_tokens over the 480 completions of the training records whose step "
                "field is 281 to 300. The scorer refuses a row whose window holds another number of completions.")},
    {"id": "below_half_strict", "alternative_printed": False,
     "ambiguity": "Whether below half is strict.",
     "chosen": "Strict. The test is 2 * T(plain) < T(verdict only), so an exact tie fails."},
    {"id": "ten_points_absolute", "alternative_printed": False,
     "ambiguity": "Whether 0.10 below is in accuracy points or relative to the control.",
     "chosen": "Accuracy points, which is a difference of at least 40 of 400 completions."},
    {"id": "own_step_0", "alternative_printed": False,
     "ambiguity": "Which step-0 value the gate uses.",
     "chosen": ("Each row's own step-0 random evaluation. Whether all 12 step-0 records agree, and whether "
                "they match the pilot's 174 of 400, is reported and does not gate.")},
    {"id": "control_health_per_seed", "alternative_printed": True,
     "ambiguity": "Whether seeds on which the control fails its own health test count in D1 to D3.",
     "chosen": ("Every seed with an admitted pair counts. The verdicts that count only seeds on which the "
                "verdict-only row meets the per-seed gate conditions are printed beside them.")},
    {"id": "same_seed_definition", "alternative_printed": False,
     "ambiguity": "What counts as the same seed.",
     "chosen": ("Rows are paired by the arm and seed fields of config.json. Equality of the step-1 training "
                "completions across the four arms of a seed is reported and does not gate.")},
    {"id": "held_out_definitions", "alternative_printed": False,
     "ambiguity": "Which quantities random held-out accuracy and parse strictly name.",
     "chosen": ("The logged accuracy of the set-random record, and the share of the 400 completions for which "
                "the strict parser returns a value.")},
    {"id": "crossing_step", "alternative_printed": False,
     "ambiguity": "How the first step with a training strict parse rate below 0.5 is computed.",
     "chosen": ("The smallest training step with at most 11 of its 24 completions parsing. Null with "
                "steps_observed when none qualifies. The number of such steps and the step from which every "
                "later step stays below are printed as supplements that are not pre-registered.")},
    {"id": "training_groups", "alternative_printed": False,
     "ambiguity": "Which groups count as the training groups in D4.",
     "chosen": "All six groups of a step, kept or dropped."},
    {"id": "d4_parse_rate", "alternative_printed": False,
     "ambiguity": "Whether the D4 strict parse rate is the training rate or the held-out rate.",
     "chosen": "Both are recorded."},
    {"id": "d4_summary", "alternative_printed": False,
     "ambiguity": "How D4 summarizes a value over the training steps.",
     "chosen": ("The full series go to the series file. The report holds the mean over all completed steps and "
                "over steps 281 to 300. Entropy is reported as a pooled mean weighted by completion length and "
                "as an unweighted mean. A null entropy stays null. A NaN or infinite entropy is not a reason to "
                "refuse a row, because it can be an outcome of the run. It is left out of the means, never "
                "imputed, and its steps are listed.")},
    {"id": "d4_every_row", "alternative_printed": False,
     "ambiguity": ("The question is whether D4 covers every row of the manifest or only the rows that are scorable. The "
                   "entry says that D4 is recorded for every row, and it ties admission only to the comparison of a pair of "
                   "arms. The REPORT-ALL line of the specification says all scorable rows, and its D4-LENGTH line says every "
                   "row that has training records."),
     "chosen": ("The scorer records D4 for every row whose run directory holds readable records, as the entry says. A row "
                "that is not scorable keeps its D4 values, its random held-out values at step 0 and at the last step, and "
                "its series. The row is marked measured_only, its problems stay beside it, and no gate, D1, D2, D3, clause "
                "or branch reads these values. A row with no run directory, with no usable steps_completed, or with "
                "training records that do not cover every completed step has no D4, and its problems say why. A logged "
                "held-out accuracy or strict parse failure rate that the record of the row contradicts is left out of the "
                "row, and the mismatch block holds both readings of it. The rule holds for every kind of problem, including "
                "a configuration or a provenance that differs from the manifest row, so the authors read the problems "
                "beside such a row before using its values. If the authors decide that D4 covers scorable rows only, the "
                "rows marked measured_only are the ones to leave out of the D4 table. No verdict changes either way.")},
    {"id": "branch_two_trigger", "alternative_printed": False,
     "ambiguity": "What triggers the second branch of the decision rule.",
     "chosen": ("The gate holds, D1 fails, and plain stays within 0.10 of verdict only on at least 2 of 3 seeds. "
                "When D1 fails and the clause is false, no branch applies exactly.")},
    {"id": "within_ten_two_sided", "alternative_printed": True,
     "alternatives": ["within_ten_one_sided", "within_ten_strict_complement"],
     "ambiguity": "What within 0.10 means in the second branch.",
     "chosen": ("An absolute difference of at most 0.10, which is at most 40 of 400 completions. Two other readings "
                "are printed beside it. The one-sided reading counts plain as within 0.10 when the control minus plain "
                "is at most 0.10. The strict reading counts plain as within 0.10 when the control minus plain is below "
                "0.10, which is the complement of D1's accuracy test. The three readings differ when the control minus "
                "plain is exactly 40 completions, and when plain ends more than 0.10 above the control.")},
    {"id": "d1_holds_d3_fails", "alternative_printed": False,
     "ambiguity": "The case where the gate holds, D1 holds and D3 fails.",
     "chosen": "No branch applies exactly. Every measurement is printed and no summary sentence is chosen."},
    {"id": "gate_failed_then_d1_to_d3", "alternative_printed": False,
     "ambiguity": "What the scorer does with D1 to D3 after the gate fails.",
     "chosen": "It computes and prints D1 to D4 and marks D1 to D3 as not supported by a failed control."},
    {"id": "loses_accuracy", "alternative_printed": False,
     "ambiguity": "What loses accuracy refers to in the main-branch sentence.",
     "chosen": ("Accuracy below the verdict-only control on the same seed at step 300. Each arm's change from "
                "its own step 0 is printed beside the gap.")},
    {"id": "d2_unconditional", "alternative_printed": False,
     "ambiguity": "Whether D2 is conditional on plain collapsing.",
     "chosen": "D2 is the unconditional accuracy test as written."},
    {"id": "unscored_comparisons", "alternative_printed": False,
     "ambiguity": "Comparisons that the arm descriptions suggest and no prediction scores.",
     "chosen": ("Restored minus plain and filter minus plain are printed as descriptive differences with no "
                "verdict.")},
    {"id": "run_matching", "alternative_printed": False,
     "ambiguity": "How runs are matched to manifest rows and what happens with a duplicate or retried run.",
     "chosen": ("A run is keyed by mode, kind, design version and task id in config.json and checked against "
                "its manifest row. More than one run directory for a row makes the row not scorable. The "
                "scorer never picks one copy.")},
    {"id": "log_words", "alternative_printed": False,
     "ambiguity": "Which logs and which words the test for no refusal or traceback uses.",
     "chosen": ("The Slurm .out and .err files matched by array job id and array task id are searched for the "
                "words REFUSED and Traceback. The first matching line of each file is printed. Log files with "
                "a match that no row owns are listed.")},
    {"id": "boundary_flags", "alternative_printed": False,
     "ambiguity": "Seeds that land exactly on a threshold or within one standard error of it.",
     "chosen": ("The threshold rule decides. A margin of zero completions is flagged as an exact boundary, and "
                "a margin within one binomial standard error of the comparator accuracy is flagged as well.")},
    {"id": "retry_combination", "alternative_printed": False,
     "ambiguity": "The pilot D-choice for the two-change retry row.",
     "chosen": "Moot. The study scorer does not apply the pilot D-choice."},
    {"id": "array_directory_name", "alternative_printed": False,
     "ambiguity": "Whether a run directory must sit under a directory that is named for its Slurm array job id.",
     "chosen": ("No. The array job id comes from provenance.json, and the Slurm logs are matched on it and on the task "
                "id, so the name of the directory that holds the runs is not checked. The report lists the array job "
                "ids it saw in the run directories and in the names of the log files, and it lists a digit-only "
                "directory name that differs from the id of its runs.")},
)
READINGS = _normalize_readings(_READINGS_RAW)


# ------------------------------------------------------------------ small helpers


def _ar():
    """Return the analyze_reasoning module, imported on first use."""
    import analyze_reasoning
    return analyze_reasoning


def _refuse(message: str):
    """Raise analyze_reasoning.Refusal, which every command line mode turns into a REFUSED message."""
    raise _ar().Refusal(message)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    """Return True for a finite number that is not a bool."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_real(x) -> bool:
    """Return True for an int or a float that is not a bool. A float may be NaN or infinite."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def count_of(accuracy: float, n: int = N_RANDOM) -> int:
    """Return the number of correct completions, n times the accuracy, and raise when it is not an integer."""
    count = round(n * accuracy)
    if abs(n * accuracy - count) > ITEM_TOL:
        raise ValueError(f"the accuracy {accuracy!r} is not a multiple of 1/{n}")
    return count


def parse_count_of(fail_rate: float, n: int = N_RANDOM) -> int:
    """Return the number of completions that parse strictly, n minus n times the failure rate."""
    return n - count_of(fail_rate, n)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True).encode())


# An absolute path has a leading slash that follows no word character, dot, tilde, dash or slash, and it has at least
# two components. A fraction such as 1/400 and an address such as https://host/page do not match. A path after a
# colon, as in host:/path/x/y, does match, because a copy command quotes it that way.
_ABSOLUTE_PATH = re.compile(r"(?<![\w./~-])/(?:[^\s/'\"()\[\]{},;:]+/)+[^\s/'\"()\[\]{},;:]*")


def scrub_paths(text: str) -> str:
    """Replace every absolute path in a message with its last component.

    A message that quotes an exception or a log line can hold the path of a file on this machine or on the cluster.
    The report must not carry such a path, so only the name of the file or directory stays."""
    return _ABSOLUTE_PATH.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], text)


def report_safe(value):
    """Return a copy of a value read from JSON that is safe to write into the report.

    The report is strict JSON, which has no NaN and no infinity, so each such float becomes its name as text. The
    report carries no absolute path, so each text keeps the last component of any path it holds. A field that the
    scorer only copies into the report, such as the early_finish record, must not end the whole scorer."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, str):
        return scrub_paths(value)
    if isinstance(value, dict):
        return {k: report_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [report_safe(v) for v in value]
    return value


def spans(numbers) -> str:
    """Return whole numbers as text in which a run of consecutive numbers is joined, such as 1 to 4, 6, 8 and 9."""
    xs = sorted(set(numbers))
    parts, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        parts.append(str(xs[i]) if j == i else (f"{xs[i]} and {xs[j]}" if j == i + 1 else f"{xs[i]} to {xs[j]}"))
        i = j + 1
    return ", ".join(parts) if parts else "none"


# ------------------------------------------------------------------ the seed-count convention


def seed_claim(flags: Dict[int, Optional[bool]], planned: Tuple[int, ...] = SEEDS, need: int = SEEDS_NEEDED,
               fixed_denominator: bool = False) -> dict:
    """Apply the convention for at least 2 of 3 seeds.

    flags maps a seed to True or False when the seed is comparable and to None when it is not. With c
    comparable seeds and k of them satisfying the claim, the claim HOLDS when k is at least need, FAILS when
    k plus the number of missing seeds is still below need, and is otherwise UNDETERMINED.

    The entry does not say how a seed without an admitted pair counts. With fixed_denominator the denominator stays
    at the planned number of seeds, a seed that is not comparable counts as not satisfying, and the claim FAILS
    whenever k is below need. The report prints this reading beside the three-valued one."""
    comparable = sorted(s for s in planned if flags.get(s) is not None)
    satisfying = sorted(s for s in comparable if flags[s])
    k, c = len(satisfying), len(comparable)
    missing = len(planned) - c
    if k >= need:
        verdict = HOLDS
    elif fixed_denominator or k + missing < need:
        verdict = FAILS
    else:
        verdict = UNDETERMINED
    return {"verdict": verdict, "k": k, "c": c, "needed": need, "planned_seeds": list(planned),
            "satisfying_seeds": satisfying, "comparable_seeds": comparable,
            "non_comparable_seeds": sorted(s for s in planned if flags.get(s) is None)}


def and3(*verdicts: str) -> str:
    """Return the three-valued AND of claim verdicts. A failed claim fails the whole, and a pending one leaves it pending."""
    if FAILS in verdicts:
        return FAILS
    if all(v == HOLDS for v in verdicts):
        return HOLDS
    return UNDETERMINED


def decision_branch(gate: str, d1: str, d3: str, w: str) -> str:
    """Return the branch of the decision rule for the verdicts of the gate, D1, D3 and the within-0.10 clause.

    The gate is tested first. D1 and D3 follow. The within-0.10 clause w matters only when D1 fails."""
    if gate == FAILS:
        return "FAILED_CONTROL"
    if gate == UNDETERMINED or d1 == UNDETERMINED:
        return "PENDING"
    if d1 == HOLDS:
        return {HOLDS: "MAIN", FAILS: "NO_BRANCH_A"}.get(d3, "PENDING")
    return {HOLDS: "NULL", FAILS: "NO_BRANCH_B"}.get(w, "PENDING")


def margin_flag(margin: int, p: float, n: int = N_RANDOM) -> Optional[str]:
    """Flag a margin to a bound. The margin is a count of completions, positive on the passing side and zero on the bound.

    A margin of zero is an exact boundary, where one completion decides the verdict. A margin within one
    binomial standard error of the accuracy p is flagged as well."""
    if margin == 0:
        return "exact_boundary"
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    if abs(margin) / n <= se:
        return "within_one_standard_error"
    return None


# ------------------------------------------------------------------ the training record and D4


def train_parse_count(rec: dict) -> int:
    """Return how many of the 24 completions of a training record parse strictly."""
    return sum(1 for g in rec["groups"] for s in g["strict"] if s is not None)


def first_step_below_half(parse_counts: List[int], steps: List[int]) -> Optional[int]:
    """Return the first training step whose strict parse rate is below 0.5, which means at most 11 of 24 parse.

    A rate of exactly 0.5 (12 of 24) is not below 0.5. The result is None when no step qualifies."""
    for step, count in zip(steps, parse_counts):
        if count <= PARSE_BELOW_HALF_MAX:
            return step
    return None


def sustained_below_half_from(parse_counts: List[int], steps: List[int]) -> Optional[int]:
    """Return the step from which every later step stays below a parse rate of 0.5, or None."""
    start = None
    for step, count in zip(steps, parse_counts):
        if count <= PARSE_BELOW_HALF_MAX:
            if start is None:
                start = step
        else:
            start = None
    return start


def _entropy_means(entropy: List[Optional[float]], length: List[float], idx: List[int]) -> dict:
    """Return the pooled mean weighted by completion length and the unweighted mean of the logged entropy.

    A step whose entropy is null is skipped and never imputed."""
    pairs = [(entropy[i], length[i]) for i in idx if entropy[i] is not None]
    if not pairs:
        return {"steps_with_value": 0, "pooled_mean_weighted_by_length": None, "unweighted_mean": None}
    weight = sum(w for _, w in pairs)
    return {"steps_with_value": len(pairs),
            "pooled_mean_weighted_by_length": sum(e * w for e, w in pairs) / weight,
            "unweighted_mean": sum(e for e, _ in pairs) / len(pairs)}


def d4_numbers(train: List[dict], held_out: Optional[dict], window: Tuple[int, int] = WINDOW) -> Tuple[dict, dict]:
    """Return the D4 measurements of one row and the per-step series they come from.

    D4 has no prediction and no threshold. The summary holds the mean completion length, the strict parse
    rate of the training groups and of the held-out set, the entropy, and the first step whose training
    strict parse rate is below 0.5. The window of steps 281 to 300 is reported only when the row holds all
    20 of its steps. T is the exact integer sum of n_tokens over the 480 completions of the window. The window
    argument exists so that the same code can be checked against the pilot's steps 81 to 100.

    held_out is None for a row whose random held-out values are not recorded, and the three held-out entries of
    the summary are then None. The series and every training measurement do not depend on it."""
    steps = [r["step"] for r in train]
    length = [float(r["completion_tokens_mean"]) for r in train]
    tokens = [sum(sum(g["n_tokens"]) for g in r["groups"]) for r in train]
    parse = [train_parse_count(r) for r in train]
    raw_entropy = [r.get("entropy_tok_mean") for r in train]
    # A NaN or infinite entropy stays unavailable (null) and is never imputed. The steps that hold one are listed.
    entropy = [e if e is not None and math.isfinite(e) else None for e in raw_entropy]
    non_finite_entropy = [s for s, e in zip(steps, raw_entropy) if e is not None and not math.isfinite(e)]
    n_all = PER_STEP * len(train)
    idx_all = list(range(len(train)))
    idx_win = [i for i, s in enumerate(steps) if window[0] <= s <= window[1]]
    window_steps = window[1] - window[0] + 1
    window_completions = window_steps * PER_STEP
    full = len(idx_win) == window_steps
    total = sum(tokens[i] for i in idx_win) if full else None
    summary = {
        "steps_observed": len(train),
        "window_complete": full,
        "mean_completion_length_all_steps": sum(tokens) / n_all if n_all else None,
        "mean_completion_length_steps_281_300": total / window_completions if full else None,
        "window_total_tokens": total,
        "window_completions": window_completions if full else None,
        "training_strict_parse_rate_mean_all_steps": sum(parse) / n_all if n_all else None,
        "training_strict_parse_rate_mean_steps_281_300": sum(parse[i] for i in idx_win) / window_completions if full else None,
        "held_out_strict_parse_count_first": held_out["strict_parse_count_first"] if held_out else None,
        "held_out_strict_parse_count_last": held_out["strict_parse_count_last"] if held_out else None,
        "held_out_completions": held_out["n"] if held_out else None,
        "entropy_all_steps": _entropy_means(entropy, length, idx_all),
        "entropy_steps_281_300": _entropy_means(entropy, length, idx_win) if full else None,
        "entropy_non_finite_steps": non_finite_entropy,
        "first_step_with_training_strict_parse_rate_below_0_5": first_step_below_half(parse, steps),
        "first_step_steps_observed": len(train),
        "supplement_not_preregistered": {
            "steps_below_0_5": sum(1 for c in parse if c <= PARSE_BELOW_HALF_MAX),
            "below_0_5_from_step_onward": sustained_below_half_from(parse, steps),
            "minimum_strict_parse_count_of_24": min(parse) if parse else None},
    }
    series = {"step": steps, "completion_tokens_mean": length, "n_tokens_total": tokens,
              "strict_parse_count": parse, "strict_parse_rate": [c / PER_STEP for c in parse],
              "entropy_tok_mean": entropy}
    return summary, series


# ------------------------------------------------------------------ the manifest, the frozen file, the runs


def check_frozen(frozen) -> None:
    """Refuse a frozen file that is not the version-2 file with the digests the study entry records."""
    if (not isinstance(frozen, dict) or frozen.get("schema") != "wo_i_frozen_v2"
            or frozen.get("design_version") != RT.DESIGN_VERSION_V2
            or not all(isinstance(frozen.get(k), str) and len(frozen[k]) == 64
                       for k in ("reasoning_sources_digest", "training_sources_digest"))):
        _refuse("the frozen file is not a version-2 frozen file")
    if (frozen["reasoning_sources_digest"] != FROZEN_REASONING_DIGEST
            or frozen["training_sources_digest"] != FROZEN_TRAINING_DIGEST):
        _refuse("the frozen file holds source digests that differ from the study entry")


def read_study_manifest(path, expected_sha256: str = MANIFEST_SHA256) -> Tuple[List[Dict[str, str]], str]:
    """Read the study manifest and refuse anything but the 12-row version-2 design of the entry.

    The expected checksum is a constant of this module and is never read from the file."""
    raw = Path(path).read_bytes()
    digest = sha256_bytes(raw)
    if digest != expected_sha256:
        _refuse(f"the manifest hashes to {digest}, and the study entry records {expected_sha256}")
    rows = _ar().read_manifest(path)
    if sorted(int(r["task_id"]) for r in rows) != list(range(1, len(ARMS) * len(SEEDS) + 1)):
        _refuse("the manifest does not hold task ids 1 to 12 once each")
    if any(r["kind"] != "study" for r in rows):
        _refuse("every manifest row must have kind study")
    import make_reasoning_manifest as MM
    try:
        MM.check_study2_shared(rows)
    except ValueError as exc:
        _refuse(str(exc))
    if {(float(r["lr"]), float(r["kl_beta"]), int(r["skip_zero_adv"])) for r in rows} != {(1e-6, 0.0, 0)}:
        _refuse("every study row must carry lr 1e-06, kl_beta 0.0 and skip_zero_adv 0")
    return rows, digest


def load_study_run(run_dir) -> dict:
    """Read one run directory like analyze_reasoning.load_pilot_run, and turn any other failure into a load problem.

    load_pilot_run lists a missing or unparsable file as a load problem. A metrics.jsonl line that is valid JSON
    and not an object (a number, null or a list) makes it raise AttributeError instead. That would end the whole
    scorer, and no not-run file could get past it, so the run is returned with a load problem and its row refuses."""
    try:
        return _ar().load_pilot_run(run_dir)
    except Exception as exc:  # noqa: BLE001
        run_dir = Path(run_dir)
        text = ("metrics.jsonl holds a line that is not a JSON object" if isinstance(exc, (AttributeError, TypeError))
                else f"the run files could not be read ({type(exc).__name__}: {exc})")
        run = {"dir": run_dir, "cfg": {}, "prov": {}, "train": [], "eval": [], "done": (run_dir / "DONE").exists(),
               "load_problems": [text]}
        for key, name in (("cfg", "config.json"), ("prov", "provenance.json")):
            try:
                data = json.loads((run_dir / name).read_text())
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                run[key] = data
        return run


def find_study_runs(runs_root) -> Tuple[Dict[int, List[dict]], List[dict]]:
    """Group the run directories under a root by task id, for version-2 study runs only.

    A directory whose config.json is not a version-2 study run (a pilot row, a version-1 study run, an
    unreadable file) is returned in the second list with its reason, and never attached to a row. Only the
    configuration is read to decide this, so a large tree of other runs costs little."""
    root = Path(runs_root)
    by_task: Dict[int, List[dict]] = {}
    other: List[dict] = []
    for cfg_path in sorted(root.rglob("config.json")):
        try:
            cfg = json.loads(cfg_path.read_text())
        except (OSError, ValueError):
            cfg = None
        reason = None
        if not isinstance(cfg, dict) or not cfg:
            reason = "config.json is missing, empty or not valid JSON"
        elif cfg.get("mode") != "manifest":
            reason = "mode is not manifest"
        elif cfg.get("kind") != "study":
            reason = f"kind is {cfg.get('kind')!r}, and a study row needs 'study'"
        elif cfg.get("design_version") != RT.DESIGN_VERSION_V2:
            reason = "design_version is not 2"
        elif not _is_int(cfg.get("task_id")):
            reason = "task_id is not an integer"
        if reason is None:
            by_task.setdefault(cfg["task_id"], []).append(load_study_run(cfg_path.parent))
        else:
            other.append({"dir": os.path.relpath(cfg_path.parent, root), "reason": reason})
    return by_task, other


def study_log_files(logs_dir, prov: dict) -> List[Path]:
    """Return the Slurm .out and .err files of one run, matched by array job id and array task id."""
    job, task = prov.get("slurm_array_job_id"), prov.get("slurm_array_task_id")
    if not job or not task:
        return []
    pattern = re.compile(r"^.+-" + re.escape(str(job)) + "_" + re.escape(str(task)) + r"\.(out|err)$")
    return sorted(p for p in Path(logs_dir).iterdir() if p.is_file() and pattern.match(p.name))


def log_problems(logs_dir, prov: dict) -> List[str]:
    """Return the log problems of one run. A log must exist, and it must hold neither REFUSED nor Traceback."""
    files = study_log_files(logs_dir, prov)
    if not files:
        return ["no Slurm log was found for this run"]
    problems = []
    for word in ("REFUSED", "Traceback"):
        hits = []
        for path in files:
            for line in path.read_text(errors="replace").splitlines():
                if word in line:
                    hits.append(f"{path.name}: {line.strip()[:160]}")
                    break
        if hits:
            problems.append(f"the logs hold {word} ({', '.join(hits)})")
    return problems


# ------------------------------------------------------------------ admission of one row


def study_config_problems(cfg: dict, row: Dict[str, str]) -> List[str]:
    """Return how config.json differs from its manifest row and from the design. An empty list means it matches."""
    ar = _ar()
    typed = RT.validate_arm_row(row)
    spec = RT.ARMS[typed["arm"]]
    problems: List[str] = []
    want = {"mode": "manifest", "kind": "study", "task_id": typed["task_id"], "arm": typed["arm"],
            "seed": typed["seed"], "steps": typed["steps"], "L": typed["L"], "lr": typed["lr"],
            "kl_beta": typed["kl_beta"], "design_version": RT.DESIGN_VERSION_V2, "size": typed["size"],
            "stage": typed["stage"], "fixedref_c": typed["fixedref_c"], "micro_batch": typed["micro_batch"],
            "model_id": typed["model_id"], "model_revision": typed["model_revision"],
            "form": spec["form"], "adv_variant": spec["adv_variant"], "group_filter": spec["group_filter"],
            "w_task": spec["w_task"], **HELD_EQUAL}
    for key, value in want.items():
        have = cfg.get(key)
        same = (have == value) if isinstance(value, str) else ar._number_is(have, value)
        if not same:
            problems.append(f"config.json has {key} = {have!r}, and the study row needs {value!r}")
    skip = cfg.get("skip_zero_adv")
    if not (isinstance(skip, (bool, int)) and int(skip) == typed["skip_zero_adv"]):
        problems.append(f"config.json has skip_zero_adv = {skip!r}, and the study row needs {typed['skip_zero_adv']!r}")
    for key, value in (("tiny", False), ("eval_limit", None), ("reference_loaded", False)):
        if cfg.get(key) is not value:
            problems.append(f"config.json has {key} = {cfg.get(key)!r}, and a study row needs {value!r}")
    if cfg.get("sampler") != SAMPLER:
        problems.append(f"config.json has sampler = {cfg.get('sampler')!r}, and the study needs {SAMPLER!r}")
    # The trainer reads --time-limit-minutes as an integer, so config.json holds a whole number or null. An absent
    # limit is not a fault, and any other value is, because the early-finish guard would have read it.
    limit = cfg.get("time_limit_minutes")
    if limit is not None and not _is_int(limit):
        problems.append(f"config.json has time_limit_minutes = {limit!r}, and the trainer writes null or a whole number of minutes")
    elif limit is not None and limit < RT.MIN_MINUTES["study"]:
        problems.append(f"config.json has time_limit_minutes = {limit!r}, and a study row needs at least {RT.MIN_MINUTES['study']}")
    stored = cfg.get("manifest_row")
    if not isinstance(stored, dict):
        problems.append("the configuration holds no manifest row")
    else:
        differs = [k for k in row if not ar._text_same(stored.get(k), row[k])]
        extra = sorted(set(stored) - set(row) - {"design_version"})
        if differs or extra or not ar._text_same(stored.get("design_version"), str(RT.DESIGN_VERSION_V2)):
            problems.append("the configuration differs from its manifest row"
                            + (f" in {', '.join(differs + extra)}" if differs or extra else ""))
    return problems


def study_provenance_problems(prov: dict, row: Dict[str, str], frozen: dict, manifest_sha256: str) -> List[str]:
    """Return how provenance.json differs from the frozen digests, the manifest row, the manifest and the design."""
    problems = []
    if prov.get("design_version") != RT.DESIGN_VERSION_V2:
        problems.append(f"provenance.json has design_version = {prov.get('design_version')!r}, and the study needs 2")
    for key, name in (("reasoning_sources_digest", "reasoning"), ("training_sources_digest", "training")):
        if prov.get(key) != frozen.get(key):
            problems.append(f"the {name} source digest differs from the frozen value")
    if prov.get("frozen_v2") != frozen:
        problems.append("the frozen record inside provenance.json is missing or differs from the frozen file")
    if prov.get("dataset_sha256") != row["dataset_sha256"] or prov.get("selection_sha256") != row["selection_sha256"]:
        problems.append("the dataset or selection digest differs from the manifest")
    if (prov.get("model_id") != row["model_id"] or prov.get("model_revision") != row["model_revision"]
            or prov.get("model_snapshot") != row["model_revision"]):
        problems.append("the model or its revision differs from the manifest")
    if prov.get("model_file_sha256") != EXPECTED_MODEL_FILE_SHA256:
        problems.append("the checksums of the model files differ from the pilot's")
    if prov.get("manifest_sha256") != manifest_sha256:
        problems.append("the manifest checksum in provenance.json differs from the study entry")
    if prov.get("device_name") != RT.REQUIRED_GPU_NAME or prov.get("device") != "cuda":
        problems.append(f"the run was not made on {RT.REQUIRED_GPU_NAME}")
    if str(prov.get("slurm_array_task_id")) != row["task_id"]:
        problems.append("the Slurm array task id differs from the task id of the row")
    if not prov.get("slurm_array_job_id"):
        problems.append("provenance.json holds no Slurm array job id")
    return problems


def stop_problems(prov: dict, steps: int) -> List[str]:
    """Return the problems of steps_completed, stopped_early and early_finish taken together."""
    problems = []
    sc, stopped, early = prov.get("steps_completed"), prov.get("stopped_early"), prov.get("early_finish")
    if not isinstance(stopped, bool):
        problems.append("provenance.json has no stopped_early flag")
    if not _is_int(sc):
        problems.append("provenance.json has no steps_completed")
        return problems
    if sc > steps or (sc < steps and stopped is not True) or (sc == steps and stopped is True):
        problems.append(f"steps_completed is {sc}, stopped_early is {stopped!r}, and the row needs {steps} steps")
    elif sc == 0:
        problems.append("the run completed no training step")
    elif stopped is True and not (isinstance(early, dict) and early.get("before_step") == sc + 1):
        problems.append("provenance.json has no early_finish record that stops before the step after steps_completed")
    elif stopped is False and early is not None:
        problems.append("provenance.json holds an early_finish record although the run did not stop early")
    return problems


def _valid_tokens(x) -> bool:
    return _is_int(x) and 1 <= x <= RT.L_V2


def _list_of(xs, size: int, valid) -> bool:
    return isinstance(xs, list) and len(xs) == size and all(valid(x) for x in xs)


def train_problems(train: List[dict], sc: int, schedule: Optional[List[List[str]]] = None) -> List[str]:
    """Return the problems of the training records, reporting the first bad step of each kind.

    When the registered problem schedule is given, the six problems of step s must be its row s - 1, in order."""
    seen = [r.get("step") for r in train]
    # A step of 1.0 or True equals 1 in a list comparison, so the type is tested as well.
    if not all(_is_int(s) for s in seen) or seen != list(range(1, sc + 1)):
        return ["the training records do not cover every completed step once"]
    first_bad: Dict[str, int] = {}
    for r in train:
        step, groups = r["step"], r.get("groups")
        if not (isinstance(groups, list) and len(groups) == GROUPS
                and all(isinstance(g, dict) and _list_of(g.get("n_tokens"), GROUP_SIZE, _valid_tokens)
                        and _list_of(g.get("strict"), GROUP_SIZE, lambda s: s is None or isinstance(s, str))
                        for g in groups)):
            first_bad.setdefault("shape", step)
            continue
        if schedule is not None and [g.get("pid") for g in groups] != schedule[step - 1]:
            first_bad.setdefault("schedule", step)
        n_tok = [n for g in groups for n in g["n_tokens"]]
        mean = r.get("completion_tokens_mean")
        if not _is_num(mean) or abs(mean - sum(n_tok) / PER_STEP) > ITEM_TOL:
            first_bad.setdefault("length", step)
        fail = r.get("strict_parse_fail_rate")
        if not _is_num(fail) or abs(fail - (PER_STEP - train_parse_count(r)) / PER_STEP) > ITEM_TOL:
            first_bad.setdefault("parse", step)
        # A NaN or infinite entropy is a logged value and not a malformed record. It can only come from a policy that
        # has broken down, which is an outcome, and "inclusion does not depend on the outcome". d4_numbers reports
        # such a step as unavailable and lists it. Only a value that is not a number at all refuses the row.
        ent = r.get("entropy_tok_mean")
        if ent is not None and not _is_real(ent):
            first_bad.setdefault("entropy", step)
    text = {"shape": "does not hold 6 groups of 4 completions with integer lengths and strict answers",
            "length": "has a completion_tokens_mean that its group lengths do not give",
            "parse": "has a strict_parse_fail_rate that its strict answers do not give",
            "entropy": "has an entropy_tok_mean that is neither null nor a number",
            "schedule": "does not hold the problems of the registered schedule"}
    return [f"the training record at step {first_bad[k]} {text[k]}"
            for k in ("shape", "schedule", "length", "parse", "entropy") if k in first_bad]


# Three kinds of problem blame the logged value of a random held-out record. A row whose only problems are of these kinds
# is a mismatch row. The pattern below matches the messages of eval_problems and of recheck_problems.
_LOGGED_NAME = {"accuracy": "an accuracy", "strict_parse_fail_rate": "a strict parse failure rate"}
_BLAMES_LOGGED_VALUE = re.compile(
    r"^(the random evaluation at step \d+ logs (an accuracy|a strict parse failure rate) of .+ while its items give .+"
    r"|the random evaluation at step \d+ has an accuracy that is not a multiple of 1/\d+"
    r"|the completions at step \d+ give (accuracy|strict parse rate) .+)$", re.DOTALL)


def blames_logged_value(problem: str) -> bool:
    """Return True when a problem says that the logged value of a random held-out record differs from its recomputation."""
    return _BLAMES_LOGGED_VALUE.match(problem) is not None


def eval_problems(run: dict, sc: int, random_pids: Optional[List[str]]) -> List[str]:
    """Return the problems of the random held-out records, which must sit exactly at steps 0 and sc.

    A logged accuracy or strict parse failure rate that its own items do not give is reported with both values, as in
    "the random evaluation at step 300 logs an accuracy of 0.42 while its items give 0.075"."""
    ar = _ar()
    rand = [r for r in run["eval"] if r.get("set") == "random"]
    if any(not _is_int(r.get("step")) for r in rand):
        return ["a random held-out record has no integer step"]
    if sorted(r["step"] for r in rand) != [0, sc]:
        return [f"the random held-out evaluations are not exactly at steps 0 and {sc}"]
    problems = []
    for r in sorted(rand, key=lambda x: x["step"]):
        tag = f"the random evaluation at step {r['step']}"
        items = r.get("items")
        if r.get("samples") != RT.EVAL_SAMPLES_RANDOM or r.get("n") != N_RANDOM or not isinstance(items, list) \
                or len(items) != N_RANDOM:
            problems.append(f"{tag} has a wrong sample or item count")
            continue
        try:
            got = ar._items_summary(items)
            pids = Counter(it[0] for it in items)
            verdicts_ok = all(it[4] in (0, 1) for it in items)
        except (IndexError, TypeError, KeyError):
            problems.append(f"{tag} has items that cannot be read")
            continue
        if not verdicts_ok:
            problems.append(f"{tag} has an item whose verdict is neither 0 nor 1")
        for key in ("accuracy", "strict_parse_fail_rate"):
            logged = r.get(key)
            if not _is_num(logged) or got[key] is None or abs(got[key] - logged) > ITEM_TOL:
                problems.append(f"{tag} logs {_LOGGED_NAME[key]} of {logged!r} while its items give {got[key]!r}")
        acc = r.get("accuracy")
        if _is_num(acc) and abs(N_RANDOM * acc - round(N_RANDOM * acc)) > ITEM_TOL:
            problems.append(f"{tag} has an accuracy that is not a multiple of 1/{N_RANDOM}")
        want = Counter({pid: RT.EVAL_SAMPLES_RANDOM for pid in random_pids}) if random_pids is not None else None
        if (want is not None and pids != want) or (want is None and set(pids.values()) != {RT.EVAL_SAMPLES_RANDOM}):
            problems.append(f"{tag} does not cover the random held-out problems twice each")
    return problems


def held_out_from_values(accuracy_first, accuracy_last, fail_first, fail_last, sc: int) -> dict:
    """Return the random held-out numbers at step 0 and at step sc as accuracies and as integer counts, from four values.

    The four values are the accuracy and the strict parse failure rate at step 0 and at step sc. A value that is not a
    multiple of 1/400 makes count_of raise ValueError."""
    c0, cl = count_of(accuracy_first), count_of(accuracy_last)
    p0, pl = parse_count_of(fail_first), parse_count_of(fail_last)
    return {"n": N_RANDOM, "step_first": 0, "step_last": sc,
            "accuracy_first": accuracy_first, "accuracy_last": accuracy_last,
            "count_first": c0, "count_last": cl, "signed_change": (cl - c0) / N_RANDOM,
            "signed_change_counts": cl - c0,
            "strict_parse_fail_rate_first": fail_first, "strict_parse_fail_rate_last": fail_last,
            "strict_parse_count_first": p0, "strict_parse_count_last": pl,
            "strict_parse_rate_first": p0 / N_RANDOM, "strict_parse_rate_last": pl / N_RANDOM}


def held_out_numbers(run: dict, sc: int) -> dict:
    """Return the random held-out numbers at step 0 and at step sc as the record logs them."""
    by_step = {r["step"]: r for r in run["eval"] if r.get("set") == "random"}
    first, last = by_step[0], by_step[sc]
    return held_out_from_values(first["accuracy"], last["accuracy"], first["strict_parse_fail_rate"],
                                last["strict_parse_fail_rate"], sc)


def recheck_problems(run: dict, sc: int, dataset: dict, selection: dict) -> Tuple[List[str], dict]:
    """Recompute the random held-out accuracy and strict parse rate at steps 0 and sc from completions.jsonl.

    Each completion is parsed with reasoning_task.strict_parse and judged with reasoning_task.verdict_fail,
    as recheck_pilot_from_completions does. A recomputed value that differs from the logged one is a problem."""
    ar = _ar()
    try:
        rechecked = ar.recheck_pilot_from_completions(run["dir"], dataset, list(selection["random"]), [0, sc])
    except (KeyError, ValueError, TypeError) as exc:
        return [f"the completions could not be rechecked: {type(exc).__name__}: {exc}"], {"agrees": False, "steps": {}}
    problems = list(rechecked["problems"])
    logged = {r["step"]: r for r in run["eval"] if r.get("set") == "random"}
    for step in (0, sc):
        got = rechecked["steps"].get(step)
        if got is None:
            continue
        if abs(got["accuracy"] - logged[step]["accuracy"]) > ITEM_TOL:
            problems.append(f"the completions at step {step} give accuracy {got['accuracy']}, "
                            f"and the record has {logged[step]['accuracy']}")
        if abs((1.0 - got["strict_parse_rate"]) - logged[step]["strict_parse_fail_rate"]) > ITEM_TOL:
            problems.append(f"the completions at step {step} give strict parse rate {got['strict_parse_rate']}, "
                            f"and the record has failure rate {logged[step]['strict_parse_fail_rate']}")
    steps = {str(k): v for k, v in sorted(rechecked["steps"].items())}
    return problems, {"agrees": not problems, "steps": steps}


def step1_completions_digest(run_dir) -> Optional[str]:
    """Return a checksum of the step-1 training completions (pid, sample, token ids) of completions.jsonl, or None."""
    path = Path(run_dir) / "completions.jsonl"
    if not path.is_file():
        return None
    found = []
    # This line of the report never gates, so a line that cannot be read makes the comparison unavailable and
    # does not refuse the row.
    try:
        with path.open() as fh:
            for line in fh:
                if '"train"' not in line:
                    continue
                rec = json.loads(line)
                if not isinstance(rec, dict) or rec.get("phase") != "train":
                    continue
                step = rec.get("step")
                if step == 1:
                    found.append([rec.get("pid"), rec.get("sample"), rec.get("token_ids")])
                elif _is_int(step) and step > 1:
                    break
    except (OSError, ValueError):
        return None
    return sha256_json(found) if found else None


def mismatch_readings(run: dict, sc: int, source: str, recheck) -> Optional[dict]:
    """Return the two readings of a row whose logged random held-out value disagrees with its own record.

    The "logged" reading takes the accuracy and the strict parse failure rate of steps 0 and sc as the record logs them.
    The "recomputed" reading takes the values that the items of the record give (source "items") or that completions.jsonl
    gives (source "completions", taken from the recheck). Each reading holds the held-out numbers and the D4 numbers that
    a scorable row would hold. A reading is marked unusable when one of its values is not a number that is a count of the
    400 completions, and the report then leaves the row out under that reading. The result is None when the recomputed
    values of steps 0 and sc are not both available."""
    ar = _ar()
    rand = {r["step"]: r for r in run["eval"] if r.get("set") == "random"}
    logged = {s: (rand[s].get("accuracy"), rand[s].get("strict_parse_fail_rate")) for s in (0, sc)}
    recomputed = {}
    if source == "items":
        for s in (0, sc):
            got = ar._items_summary(rand[s]["items"])
            recomputed[s] = (got["accuracy"], got["strict_parse_fail_rate"])
    elif isinstance(recheck, dict):
        recomputed = {int(k): (v["accuracy"], 1.0 - v["strict_parse_rate"]) for k, v in recheck["steps"].items()}
    if set(recomputed) != {0, sc}:
        return None
    readings = {}
    for mode, values in (("logged", logged), ("recomputed", recomputed)):
        try:
            if not all(_is_num(x) for pair in values.values() for x in pair):
                raise ValueError("a value is not a finite number")
            held = held_out_from_values(values[0][0], values[sc][0], values[0][1], values[sc][1], sc)
        except (ValueError, TypeError, OverflowError):
            readings[mode] = {"usable": False, "held_out": None, "d4": None}
            continue
        readings[mode] = {"usable": True, "held_out": held, "d4": d4_numbers(run["train"], held)[0]}
    evidence = [{"step": s, "field": name, "logged": report_safe(lg), "recomputed": report_safe(rc)}
                for s in (0, sc)
                for name, lg, rc in (("accuracy", logged[s][0], recomputed[s][0]),
                                     ("strict_parse_fail_rate", logged[s][1], recomputed[s][1]))
                if not (_is_num(lg) and _is_num(rc) and abs(lg - rc) <= ITEM_TOL)]
    return {"recomputed_from": source, "evidence": evidence, "readings": readings}


def agreed_d4(d4: Optional[dict], mismatch: Optional[dict]) -> Optional[dict]:
    """Return the D4 summary that a mismatch row keeps.

    The two readings of a mismatch row differ in a logged random held-out value, and the entry does not say which one
    counts. The training measurements are the same under both readings, and so are the held-out strict parse counts when the
    mismatch is in the accuracy only. The row therefore keeps the D4 summary of its readings when they give the same summary,
    and it keeps the summary without held-out entries otherwise, or when a reading is unusable."""
    readings = (mismatch or {}).get("readings") or {}
    logged, recomputed = readings.get("logged"), readings.get("recomputed")
    if d4 is not None and logged and recomputed and logged["usable"] and recomputed["usable"] \
            and logged["d4"] == recomputed["d4"]:
        return logged["d4"]
    return d4


def _new_entry(row: Dict[str, str]) -> dict:
    return {"task_id": int(row["task_id"]), "arm": row["arm"], "seed": int(row["seed"]), "run_dir": None,
            "status": None, "problems": [], "scorable": False, "admitted": False, "measured_only": False,
            "stopped_early": None,
            "steps_completed": None, "steps_planned": int(row["steps"]), "early_finish": None,
            "time_limit_minutes": None, "array_job_id": None, "not_run_reason": None, "held_out": None,
            "pilot_d0": None, "gate_float_reading_agrees": None, "recheck": "not run", "d4": None,
            "integrity": None, "mismatch": None}


def _admit_run(entry: dict, run: dict, row: Dict[str, str], ctx: dict) -> Optional[dict]:
    """Fill the entry of one run directory and return its D4 series, or None when the run has no usable series."""
    ar = _ar()
    cfg, prov, problems = run["cfg"], run["prov"], entry["problems"]
    problems.extend(run["load_problems"])
    done = Path(run["dir"]) / "DONE"
    if not done.is_file():
        problems.append("no DONE marker")
    elif done.read_text().strip() != "ok":
        problems.append("the DONE marker does not say ok")
    problems.extend(study_config_problems(cfg, row))
    want_name = f"{row['task_id']}_study_{row['size']}_{row['arm']}_s{row['seed']}"
    if Path(run["dir"]).name != want_name:
        problems.append(f"the run directory is named {Path(run['dir']).name}, and its row needs {want_name}")
    if not prov and not any(m.startswith("provenance.json") for m in run["load_problems"]):
        problems.append("provenance.json holds no fields")
    # The name of the directory that holds a run directory is not checked. The array job id comes from provenance.json,
    # and the Slurm logs are matched on it, so a pull into a directory with any name is read the same way.
    if prov:
        problems.extend(study_provenance_problems(prov, row, ctx["frozen"], ctx["manifest_sha256"]))
        problems.extend(stop_problems(prov, entry["steps_planned"]))
    problems.extend(log_problems(ctx["logs_dir"], prov))
    sc = prov.get("steps_completed")
    stopped = prov.get("stopped_early")
    # These three values are copied into the report as they are. A NaN or an infinity in one of them would stop the
    # strict JSON writer, so each is copied with the name of the constant.
    entry.update({"stopped_early": stopped if isinstance(stopped, bool) else None,
                  "steps_completed": sc if _is_int(sc) else None, "early_finish": report_safe(prov.get("early_finish")),
                  "time_limit_minutes": report_safe(cfg.get("time_limit_minutes")),
                  "array_job_id": report_safe(prov.get("slurm_array_job_id"))})
    series = None
    held_out = None
    d4 = None
    # blame lists the problems that blame a logged value, and source says where the recomputation of that value comes from.
    blame, source = [], None
    if _is_int(sc) and 0 < sc <= entry["steps_planned"]:
        bad_train = train_problems(run["train"], sc, ctx["schedule"])
        bad_eval = eval_problems(run, sc, ctx["random_pids"])
        problems.extend(bad_train + bad_eval)
        blame = [p for p in bad_eval if blames_logged_value(p)]
        source = "items" if blame else None
        if not bad_eval:
            held_out = held_out_numbers(run, sc)
            entry["pilot_d0"] = ar.pilot_d0(run["eval"], last_step=sc)
            if ctx["recheck"]:
                more, entry["recheck"] = recheck_problems(run, sc, ctx["dataset"], ctx["selection"])
                problems.extend(more)
                blame = [p for p in more if blames_logged_value(p)]
                source = "completions" if blame else None
        # A logged random held-out value that the record of the row contradicts is not recorded for the row. The mismatch
        # block holds both readings of it, and the entry does not say which one counts.
        if blame:
            held_out = None
        # D4 is recorded for every row that has readable training records, whether or not the row is scorable.
        if not bad_train:
            d4, series = d4_numbers(run["train"], held_out)
    if problems:
        # A row that is not scorable keeps the measurements that its own records support, and nothing reads them in a
        # verdict. The pilot D0 verdict and the integrity lines belong to scorable rows only.
        entry["pilot_d0"] = None
        # A row whose only problems blame a logged value is a mismatch row. The report scores the study once more with
        # the logged value of such a row and once with the recomputed value, because the entry does not say which counts.
        if blame and set(problems) == set(blame):
            entry["mismatch"] = mismatch_readings(run, sc, source, entry["recheck"])
            d4 = agreed_d4(d4, entry["mismatch"])
        entry["held_out"], entry["d4"] = held_out, d4
        return series
    entry["held_out"], entry["d4"] = held_out, d4
    if entry["pilot_d0"] is not None:
        counts_pass = (abs(held_out["signed_change_counts"]) <= ACC_BAND_COUNTS
                       and held_out["strict_parse_count_last"] >= PARSE_MIN_COUNT)
        entry["gate_float_reading_agrees"] = entry["pilot_d0"]["passes"] == counts_pass
    first = next(r for r in run["eval"] if r.get("set") == "random" and r["step"] == 0)
    entry["integrity"] = {
        "step_0_items_sha256": sha256_json(first["items"]),
        "step_1_groups_sha256": sha256_json([[g.get("pid"), g["n_tokens"], g.get("F"), g.get("finished"), g["strict"]]
                                             for g in run["train"][0]["groups"]]),
        "step_1_completions_sha256": step1_completions_digest(run["dir"]),
        "stack": {k: report_safe(prov.get(k)) for k in ("python", "torch", "transformers", "numpy", "platform", "device_name")}}
    return series


def admit_row(row: Dict[str, str], runs: List[dict], ctx: dict) -> Tuple[dict, Optional[dict]]:
    """Return the report entry of one manifest row and the D4 series of its run, or None for the series.

    A row with no run directory, or with more than one, is not scorable. Any exception raised while one
    run is read becomes a problem of that row, so a malformed file refuses its row and nothing else. A row that is not
    scorable keeps the held-out and D4 values that its records support, with measured_only set, because the entry records
    D4 for every row. A row whose reading raised keeps none."""
    entry = _new_entry(row)
    tid = entry["task_id"]
    if not runs:
        hints = [f"{u['dir']}: {u['reason']}" for u in ctx["unmatched"]
                 if os.path.basename(u["dir"]).startswith(f"{tid}_study_")]
        entry["status"] = "no_run_directory"
        entry["problems"] = [scrub_paths("no run directory for this row"
                                         + (f" (unmatched directories {', '.join(hints)})" if hints else ""))]
        return entry, None
    if len(runs) > 1:
        entry["status"] = "ambiguous"
        entry["problems"] = ["more than one run directory for this row: "
                             + ", ".join(sorted(os.path.relpath(r["dir"], ctx["runs_root"]) for r in runs))]
        return entry, None
    run = runs[0]
    entry["run_dir"] = os.path.relpath(run["dir"], ctx["runs_root"])
    series = None
    try:
        series = _admit_run(entry, run, row, ctx)
    # A malformed record must refuse its row and must not end the scorer in a traceback. A record that raised while it was
    # read supports no measurement, so the row keeps none.
    except Exception as exc:  # noqa: BLE001
        entry["problems"].append(f"admission raised {type(exc).__name__}: {exc}")
        entry["mismatch"] = entry["held_out"] = entry["d4"] = entry["pilot_d0"] = entry["integrity"] = None
    # An exception or a log line can quote the path of a file, and the report carries no path of this machine or of the cluster.
    entry["problems"] = [scrub_paths(p) for p in entry["problems"]]
    entry["scorable"] = not entry["problems"]
    # The pilot D0 verdict and the integrity lines belong to scorable rows, and _admit_run fills them for those rows only. A row
    # that is not scorable keeps the held-out and D4 values that its records support, and it is marked measured_only when it keeps any.
    entry["measured_only"] = not entry["scorable"] and (entry["held_out"] is not None or entry["d4"] is not None)
    entry["admitted"] = entry["scorable"] and entry["steps_completed"] == entry["steps_planned"] \
        and entry["stopped_early"] is False
    entry["status"] = "refused" if entry["problems"] else ("admitted" if entry["admitted"] else "stopped_early")
    return entry, series


# ------------------------------------------------------------------ the rules, one seed at a time


def _flags(*pairs: Tuple[str, Optional[str]]) -> List[str]:
    return [f"{name}: {flag}" for name, flag in pairs if flag]


def _not_comparable(seed: int, reason: str) -> dict:
    return {"seed": seed, "comparable": False, "satisfies": None, "conditions": {}, "reason": reason}


def _unavailable(arm: str, x: Optional[dict], v: Optional[dict]) -> Optional[str]:
    """Return why a pair is not compared, or None when both rows are admitted."""
    parts = []
    for name, e in ((arm, x), (CONTROL, v)):
        if e is None:
            parts.append(f"the {name} row is not scorable")
        elif not e["admitted"]:
            parts.append(f"the {name} row of task {e['task_id']} is {e['status']} and not admitted")
    return " and ".join(parts) if parts else None


def gate_seed(seed: int, v: Optional[dict], rows: str = "scorable", compare: str = "counts") -> dict:
    """Score the D-gate on one seed from the verdict-only row.

    The row is comparable when it is scorable, or when it is admitted if rows is "admitted". The accuracy
    condition is an absolute difference of at most 20 of 400 completions from the row's own step 0, and the
    parse condition is at least 320 of 400 strict parses at the last step."""
    if v is None or not v["scorable"]:
        return _not_comparable(seed, "the verdict-only row is not scorable")
    if rows == "admitted" and not v["admitted"]:
        return _not_comparable(seed, f"the verdict-only row of task {v['task_id']} stopped early and this gate leaves it out")
    h = v["held_out"]
    diff = h["signed_change_counts"]
    if compare == "counts":
        within = abs(diff) <= ACC_BAND_COUNTS
        parse_ok = h["strict_parse_count_last"] >= PARSE_MIN_COUNT
    else:
        within = abs(h["accuracy_last"] - h["accuracy_first"]) <= ACC_BAND + FLOAT_TOL
        parse_ok = (1.0 - h["strict_parse_fail_rate_last"]) >= PARSE_MIN
    margins = {"within_0_05": ACC_BAND_COUNTS - abs(diff), "parse_at_least_0_80": h["strict_parse_count_last"] - PARSE_MIN_COUNT}
    return {"seed": seed, "comparable": True, "task_id": v["task_id"], "admitted": v["admitted"],
            "stopped_early": v["stopped_early"], "steps_completed": v["steps_completed"],
            "step_last": h["step_last"], "n": h["n"],
            "accuracy_first": h["accuracy_first"], "accuracy_last": h["accuracy_last"],
            "count_first": h["count_first"], "count_last": h["count_last"],
            "signed_difference": h["signed_change"], "signed_difference_counts": diff,
            "absolute_difference_counts": abs(diff), "rise_above_0_05": diff > ACC_BAND_COUNTS,
            "strict_parse_count_first": h["strict_parse_count_first"],
            "strict_parse_count_last": h["strict_parse_count_last"],
            "strict_parse_rate_last": h["strict_parse_rate_last"],
            "conditions": {"within_0_05": within, "parse_at_least_0_80": parse_ok},
            "satisfies": within and parse_ok, "margins": margins,
            "boundary_flags": _flags(("within_0_05", margin_flag(margins["within_0_05"], h["accuracy_first"])),
                                     ("parse_at_least_0_80", margin_flag(margins["parse_at_least_0_80"], h["strict_parse_rate_last"])))}


def _pair_base(seed: int, arm: str, x: dict, v: dict) -> dict:
    hx, hv = x["held_out"], v["held_out"]
    return {"seed": seed, "comparable": True, "arm": arm, "task_id_arm": x["task_id"], "task_id_control": v["task_id"],
            "step_last": hx["step_last"], "n": hx["n"],
            "accuracy_arm": hx["accuracy_last"], "accuracy_control": hv["accuracy_last"],
            "count_arm": hx["count_last"], "count_control": hv["count_last"],
            "arm_minus_control": (hx["count_last"] - hv["count_last"]) / N_RANDOM,
            "arm_minus_control_counts": hx["count_last"] - hv["count_last"],
            "absolute_difference_counts": abs(hx["count_last"] - hv["count_last"]),
            "arm_change_from_step_0": hx["signed_change"], "control_change_from_step_0": hv["signed_change"]}


def d1_seed(seed: int, p: Optional[dict], v: Optional[dict], compare: str = "counts") -> dict:
    """Score D1 on one seed. Plain's window total T must be below half the control's, and its accuracy 40 of 400 or more below."""
    reason = _unavailable(PLAIN, p, v)
    if reason:
        return _not_comparable(seed, reason)
    out = _pair_base(seed, PLAIN, p, v)
    tp, tv = p["d4"]["window_total_tokens"], v["d4"]["window_total_tokens"]
    gap = -out["arm_minus_control_counts"]
    length_ok = 2 * tp < tv
    acc_ok = gap >= GAP_COUNTS if compare == "counts" else (v["held_out"]["accuracy_last"] - p["held_out"]["accuracy_last"]) >= GAP
    margins = {"length_below_half": tv - 2 * tp, "accuracy_gap_at_least_0_10": gap - GAP_COUNTS}
    out.update({"window_total_tokens_arm": tp, "window_total_tokens_control": tv,
                "mean_length_arm": tp / WINDOW_COMPLETIONS, "mean_length_control": tv / WINDOW_COMPLETIONS,
                "control_minus_arm_counts": gap,
                "conditions": {"length_below_half": length_ok, "accuracy_gap_at_least_0_10": acc_ok},
                "satisfies": length_ok and acc_ok, "margins": margins,
                "boundary_flags": _flags(("length_below_half", "exact_boundary" if margins["length_below_half"] == 0 else None),
                                         ("accuracy_gap_at_least_0_10", margin_flag(margins["accuracy_gap_at_least_0_10"], v["held_out"]["accuracy_last"])))})
    return out


def d2_seed(seed: int, r: Optional[dict], v: Optional[dict], compare: str = "counts") -> dict:
    """Score D2 on one seed. Restored's accuracy at step 300 must be 40 of 400 or more below the control's."""
    reason = _unavailable(RESTORED, r, v)
    if reason:
        return _not_comparable(seed, reason)
    out = _pair_base(seed, RESTORED, r, v)
    gap = -out["arm_minus_control_counts"]
    acc_ok = gap >= GAP_COUNTS if compare == "counts" else (v["held_out"]["accuracy_last"] - r["held_out"]["accuracy_last"]) >= GAP
    margins = {"accuracy_gap_at_least_0_10": gap - GAP_COUNTS}
    out.update({"control_minus_arm_counts": gap, "conditions": {"accuracy_gap_at_least_0_10": acc_ok},
                "satisfies": acc_ok, "margins": margins,
                "boundary_flags": _flags(("accuracy_gap_at_least_0_10", margin_flag(margins["accuracy_gap_at_least_0_10"], v["held_out"]["accuracy_last"])))})
    return out


def _within_seed(seed: int, arm: str, x: Optional[dict], v: Optional[dict], bound_counts: int, bound: float,
                 tolerance: float, name: str, compare: str) -> dict:
    reason = _unavailable(arm, x, v)
    if reason:
        return _not_comparable(seed, reason)
    out = _pair_base(seed, arm, x, v)
    if compare == "counts":
        ok = out["absolute_difference_counts"] <= bound_counts
    else:
        ok = abs(x["held_out"]["accuracy_last"] - v["held_out"]["accuracy_last"]) <= bound + tolerance
    margins = {name: bound_counts - out["absolute_difference_counts"]}
    out.update({"conditions": {name: ok}, "satisfies": ok, "margins": margins,
                "boundary_flags": _flags((name, margin_flag(margins[name], v["held_out"]["accuracy_last"])))})
    return out


def d3_seed(seed: int, f: Optional[dict], v: Optional[dict], compare: str = "counts") -> dict:
    """Score D3 on one seed. The filter's accuracy at step 300 must be within 20 of 400 completions of the control's, in both directions."""
    return _within_seed(seed, FILTER, f, v, ACC_BAND_COUNTS, ACC_BAND, FLOAT_TOL, "within_0_05", compare)


WITHIN_READINGS = ("two_sided", "one_sided", "strict_complement")


def w_seed(seed: int, p: Optional[dict], v: Optional[dict], compare: str = "counts", reading: str = "two_sided") -> dict:
    """Score the within-0.10 clause of the second branch on one seed.

    The primary reading is "two_sided": plain must be within 40 of 400 completions of the control, in both directions.
    Two other readings are scored beside it. In "one_sided", plain is within when the control minus plain is at most 40
    completions, so a plain that ends above the control always counts. In "strict_complement", plain is within when the
    control minus plain is below 40 completions, which is exactly the case in which D1's accuracy test fails for it."""
    if reading not in WITHIN_READINGS:
        raise ValueError(f"unknown reading {reading!r} of the within-0.10 clause")
    if reading == "two_sided":
        return _within_seed(seed, PLAIN, p, v, LOOSE_BAND_COUNTS, LOOSE_BAND, 0.0, "within_0_10", compare)
    reason = _unavailable(PLAIN, p, v)
    if reason:
        return _not_comparable(seed, reason)
    out = _pair_base(seed, PLAIN, p, v)
    gap = -out["arm_minus_control_counts"]
    limit = LOOSE_BAND_COUNTS if reading == "one_sided" else LOOSE_BAND_COUNTS - 1
    margins = {"within_0_10": limit - gap}
    out.update({"conditions": {"within_0_10": gap <= limit}, "satisfies": gap <= limit, "margins": margins,
                "boundary_flags": _flags(("within_0_10", margin_flag(margins["within_0_10"], v["held_out"]["accuracy_last"])))})
    return out


def _block(per_seed: List[dict], names: Tuple[str, ...], conj: str, health: Optional[Dict[int, bool]],
           fixed: bool = False) -> dict:
    """Combine the per-seed results of one rule into a claim with the seed-count convention.

    With fixed the claim keeps a denominator of 3, and a seed that is not comparable counts as not satisfying."""
    if health is not None:
        for d in per_seed:
            if d["comparable"]:
                d["satisfies_before_control_health"] = d["satisfies"]
                d["satisfies"] = bool(d["satisfies"] and health.get(d["seed"], False))
    claim = seed_claim({d["seed"]: (d["satisfies"] if d["comparable"] else None) for d in per_seed},
                       fixed_denominator=fixed)
    separate = {n: seed_claim({d["seed"]: (d["conditions"][n] if d["comparable"] else None) for d in per_seed},
                              fixed_denominator=fixed)
                for n in names}
    other = and3(*[separate[n]["verdict"] for n in names])
    block = dict(claim)
    block["verdict"] = claim["verdict"] if conj == "same_seed" else other
    block["per_seed"] = per_seed
    block["conditions_counted_separately"] = {
        "verdict": other, "claims": {n: {key: separate[n][key] for key in ("verdict", "k", "c", "satisfying_seeds")}
                                     for n in names}}
    block["boundary_seeds"] = sorted(d["seed"] for d in per_seed if d["comparable"] and d["boundary_flags"])
    return block


def score_claims(entries: Dict[Tuple[str, int], dict], *, conj: str = "same_seed", gate_rows: str = "scorable",
                 health: str = "none", compare: str = "counts", missing: str = "three_valued",
                 within: str = "two_sided") -> dict:
    """Score the gate, D1, D2, D3 and the within-0.10 clause and pick the branch of the decision rule.

    The defaults are the primary reading. The other values select the alternative readings that the report
    prints beside it: conj "separate" counts each condition of the gate and of D1 over seeds on its own,
    gate_rows "admitted" leaves stopped-early rows out of the gate, health "control_gate" counts only seeds on
    which the control meets the gate conditions, compare "float" uses plain float comparisons, missing "fixed"
    keeps the denominator at 3 so that a seed without an admitted pair counts as not satisfying, and within
    "one_sided" or "strict_complement" reads the within-0.10 clause in another way (see w_seed)."""
    fixed = missing == "fixed"
    def at(arm: str, seed: int) -> Optional[dict]:
        return entries.get((arm, seed))

    gate_ps = [gate_seed(s, at(CONTROL, s), gate_rows, compare) for s in SEEDS]
    health_map = ({d["seed"]: bool(d["satisfies"]) for d in gate_ps if d["comparable"]}
                  if health == "control_gate" else None)
    gate = _block(gate_ps, ("within_0_05", "parse_at_least_0_80"), conj, None, fixed)
    d1 = _block([d1_seed(s, at(PLAIN, s), at(CONTROL, s), compare) for s in SEEDS],
                ("length_below_half", "accuracy_gap_at_least_0_10"), conj, health_map, fixed)
    d2 = _block([d2_seed(s, at(RESTORED, s), at(CONTROL, s), compare) for s in SEEDS],
                ("accuracy_gap_at_least_0_10",), conj, health_map, fixed)
    d3 = _block([d3_seed(s, at(FILTER, s), at(CONTROL, s), compare) for s in SEEDS],
                ("within_0_05",), conj, health_map, fixed)
    w = _block([w_seed(s, at(PLAIN, s), at(CONTROL, s), compare, within) for s in SEEDS],
               ("within_0_10",), conj, health_map, fixed)
    verdicts = {"gate": gate["verdict"], "d1": d1["verdict"], "d2": d2["verdict"], "d3": d3["verdict"],
                "within_0_10_clause": w["verdict"]}
    branch = decision_branch(verdicts["gate"], verdicts["d1"], verdicts["d3"], verdicts["within_0_10_clause"])
    return {"gate": gate, "d1": d1, "d2": d2, "d3": d3, "within_0_10_clause": w, "verdicts": verdicts, "branch": branch}


# Each alternative reading changes one option of score_claims. The report prints the verdicts under each one. The two
# mismatch readings change the rows instead: they admit every mismatch row with its logged value or with its recomputed
# value (see with_mismatch_rows), and they change no option of score_claims.
ALTERNATIVE_READINGS = (
    ("conjunction_per_seed", {"conj": "separate"}),
    ("gate_stopped_early", {"gate_rows": "admitted"}),
    ("control_health_per_seed", {"health": "control_gate"}),
    ("float_tolerance", {"compare": "float"}),
    ("missing_seed_count", {"missing": "fixed"}),
    ("within_ten_one_sided", {"within": "one_sided"}),
    ("within_ten_strict_complement", {"within": "strict_complement"}),
    ("mismatch_logged_value", {"mismatch": "logged"}),
    ("mismatch_recomputed_value", {"mismatch": "recomputed"}),
)


def with_mismatch_rows(scorable: Dict[Tuple[str, int], dict], entries: List[dict], mode: str) -> Dict[Tuple[str, int], dict]:
    """Return the scorable rows plus every mismatch row, admitted under its logged or its recomputed value.

    mode is "logged" or "recomputed". A mismatch row is a row whose only problems blame a logged random held-out value.
    It is admitted when it finished all its steps and did not stop early, and it is scorable and not admitted when it
    stopped early, like any other row. A mismatch row whose values of this mode are not counts of the 400 completions
    stays out. The input rows are not changed."""
    pool = dict(scorable)
    for e in entries:
        reading = ((e.get("mismatch") or {}).get("readings") or {}).get(mode)
        if reading and reading["usable"]:
            admitted = e["steps_completed"] == e["steps_planned"] and e["stopped_early"] is False
            pool[(e["arm"], e["seed"])] = {**e, "held_out": reading["held_out"], "d4": reading["d4"], "scorable": True,
                                           "measured_only": False, "admitted": admitted,
                                           "status": "admitted" if admitted else "stopped_early", "problems": []}
    return pool


def paper_statement(branch: str, d2: dict) -> Optional[str]:
    """Return the sentence the paper reports for a branch, with D2 as it falls out for the main branch."""
    base = PAPER_STATEMENTS[branch]
    if branch != "MAIN":
        return base
    gaps = []
    for d in d2["per_seed"]:
        gaps.append(f"seed {d['seed']}: {d['control_minus_arm_counts'] / N_RANDOM:+.4f}" if d["comparable"]
                    else f"seed {d['seed']}: not compared")
    return (f"{base} D2 {d2['verdict']}. Restored ends at least 0.10 below verdict only on {d2['k']} of "
            f"{d2['c']} compared seeds, and the gaps of verdict only minus restored are {', '.join(gaps)}.")


def descriptive_differences(entries: Dict[Tuple[str, int], dict]) -> List[dict]:
    """Return restored minus plain and filter minus plain per seed. These have no prediction and no verdict."""
    out = []
    for s in SEEDS:
        p = entries.get((PLAIN, s))
        rec = {"seed": s, "label": "descriptive, no prediction"}
        for name, arm in (("restored_minus_plain", RESTORED), ("vfilter_minus_plain", FILTER)):
            x = entries.get((arm, s))
            if p and x and p["admitted"] and x["admitted"]:
                diff = x["held_out"]["count_last"] - p["held_out"]["count_last"]
                rec[name] = {"accuracy_step_300": diff / N_RANDOM, "count_step_300": diff,
                             "window_total_tokens": x["d4"]["window_total_tokens"] - p["d4"]["window_total_tokens"],
                             "mean_length_steps_281_300": (x["d4"]["window_total_tokens"] - p["d4"]["window_total_tokens"]) / WINDOW_COMPLETIONS}
            else:
                rec[name] = None
        out.append(rec)
    return out


# ------------------------------------------------------------------ the integrity lines (they never gate)


def integrity_block(entries: List[dict]) -> dict:
    """Return the checks that the study entry states as facts of the design. None of them refuses a row."""
    scorable = [e for e in entries if e["scorable"]]

    def identical(key: str, es: List[dict]) -> dict:
        digests = [e["integrity"][key] for e in es]
        if len(es) < 2 or any(d is None for d in digests):
            return {"identical": None, "rows_compared": [e["task_id"] for e in es]}
        return {"identical": len(set(digests)) == 1, "rows_compared": [e["task_id"] for e in es]}

    by_seed = {s: [e for e in scorable if e["seed"] == s] for s in SEEDS}
    pairs = sorted({(e["held_out"]["count_first"], e["held_out"]["strict_parse_count_first"]) for e in scorable})
    stack = {}
    for key in ("python", "torch", "transformers", "numpy", "platform", "device_name"):
        stack[key] = sorted({str(e["integrity"]["stack"][key]) for e in scorable})
    return {
        "step_0_random_records_identical_across_scorable_rows": (
            len({e["integrity"]["step_0_items_sha256"] for e in scorable}) <= 1),
        "step_0_count_pairs_correct_and_strict_parse": [list(p) for p in pairs],
        "step_0_equals_pilot_values_174_and_338_UNVERIFIED_for_the_study": pairs == [PILOT_STEP0_COUNTS],
        "step_1_training_groups_identical_across_the_arms_of_a_seed": {
            str(s): identical("step_1_groups_sha256", by_seed[s]) for s in SEEDS},
        "step_1_training_completions_identical_across_the_arms_of_a_seed": {
            str(s): identical("step_1_completions_sha256", by_seed[s]) for s in SEEDS},
        "software_and_hardware_values_seen": stack,
        "array_job_ids": sorted({str(e["array_job_id"]) for e in scorable}),
        "time_limit_minutes": sorted({str(e["time_limit_minutes"]) for e in scorable}),
        "gate_float_reading_agrees_with_counts_for_every_scorable_row": all(
            e["gate_float_reading_agrees"] is True for e in scorable),
    }


# ------------------------------------------------------------------ the report


def _check_not_run(not_run, rows: List[Dict[str, str]], entries: List[dict]) -> Dict[int, str]:
    """Validate the not-run mapping and return it as task id to reason. A contradiction refuses the study."""
    if not_run is None:
        return {}
    if not isinstance(not_run, dict):
        _refuse("the not-run file must be a JSON object from task ids to reasons")
    ids = {int(r["task_id"]) for r in rows}
    named: Dict[int, str] = {}
    for key, reason in not_run.items():
        if not (isinstance(key, str) and key.isdigit() and int(key) in ids):
            _refuse(f"the not-run file names {key!r}, which is not a manifest task id")
        if not isinstance(reason, str) or not reason.strip():
            _refuse(f"the not-run file gives no reason for task {key}")
        named[int(key)] = reason.strip()
    fine = sorted(e["task_id"] for e in entries if e["task_id"] in named and e["scorable"])
    if fine:
        _refuse(f"the not-run file names tasks {fine}, which have a scorable run")
    return named


# A Slurm array task log is named <job name>-<array job id>_<array task id>.out or .err.
LOG_FILE_NAME = re.compile(r"^.+-(?P<array>[0-9]+)_(?P<task>[0-9]+)\.(?:out|err)$")


def scan_logs(logs_dir: Path, owned: set) -> Tuple[List[str], Dict[str, dict], int, List[str]]:
    """Read every file under the log directory once, and return what the report says about the files no row owns.

    The first result lists the names of the files that hold REFUSED or Traceback and that no row owns. The second
    groups the files whose names look like array task logs by array job id, with the task ids, the number of files,
    the number that hold REFUSED or Traceback, and the number that a row owns. The third counts the other files.
    The fourth lists the files that could not be read. An unreadable file that a row owns has already refused that
    row, so it must not end the whole scorer here, and it counts as a file without a hit."""
    pattern = _ar().PILOT_LOG_FAILURE
    unowned: List[str] = []
    unreadable: List[str] = []
    groups: Dict[str, dict] = {}
    other = 0
    for path in sorted(logs_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            hit = bool(pattern.search(path.read_text(errors="replace")))
        except OSError:
            hit = False
            unreadable.append(path.name)
        mine = path in owned
        if hit and not mine:
            unowned.append(path.name)
        found = LOG_FILE_NAME.match(path.name)
        if found is None:
            other += 1
            continue
        g = groups.setdefault(found.group("array"), {"tasks": set(), "files": 0, "hits": 0, "owned": 0})
        g["tasks"].add(int(found.group("task")))
        g["files"] += 1
        g["hits"] += hit
        g["owned"] += mine
    arrays = {job: {"task_ids": sorted(g["tasks"]), "files": g["files"], "files_with_REFUSED_or_Traceback": g["hits"],
                    "files_owned_by_a_row": g["owned"]} for job, g in sorted(groups.items())}
    return unowned, arrays, other, unreadable


def arrays_block(by_task: Dict[int, List[dict]], log_arrays: Dict[str, dict], other_logs: int) -> dict:
    """Return the array job ids the scorer saw, in the run directories and in the names of the log files.

    A retry runs under a new array job id, and the log directory can still hold the logs of the array that failed at
    launch. This block names both, so the reader sees which array each row came from and which logs belong to no row."""
    runs_by_array: Dict[str, set] = {}
    renamed = []
    for tid in sorted(by_task):
        for r in by_task[tid]:
            job = r["prov"].get("slurm_array_job_id")
            key = str(report_safe(job)) if job else "none recorded"
            runs_by_array.setdefault(key, set()).add(tid)
            parent = Path(r["dir"]).parent.name
            if parent.isdigit() and job and parent != str(job):
                renamed.append({"task_id": tid, "directory_name": parent, "array_job_id": key})
    return {
        "run_directories_by_array_job_id": {k: sorted(v) for k, v in sorted(runs_by_array.items())},
        "log_files_by_array_job_id": log_arrays,
        "log_files_not_named_like_array_task_logs": other_logs,
        "array_job_ids_with_log_files_and_no_run_directory": sorted(set(log_arrays) - set(runs_by_array)),
        "rows_come_from_more_than_one_array": len(runs_by_array) > 1,
        "run_directory_parents_with_a_digit_name_that_differs_from_the_array_job_id": renamed,
    }


def _blocked_message(entries: List[dict]) -> str:
    parts = []
    for e in entries:
        parts.append(f"task {e['task_id']} ({e['arm']}, seed {e['seed']}): " + ", ".join(e["problems"]) + ".")
    return ("the set of rows is not admissible, so nothing was scored. "
            "Name a row in the not-run file only when it will not be rerun. " + " ".join(parts))


def study_report(manifest, runs_root, frozen: dict, logs_dir, *, dataset_path=None, selection_path=None,
                 recheck: bool = False, not_run: Optional[dict] = None,
                 expected_manifest_sha256: str = MANIFEST_SHA256) -> Tuple[dict, dict]:
    """Admit the 12 study rows, score the gate and D1 to D4, and return the report and the D4 series.

    The function refuses, by raising analyze_reasoning.Refusal, when the manifest, the frozen file, the log
    directory or the optional dataset and selection do not match the study entry, when a version-2 study
    run has a task id outside the manifest, or when any row is not scorable and is not named in not_run."""
    ar = _ar()
    check_frozen(frozen)
    rows, manifest_digest = read_study_manifest(manifest, expected_manifest_sha256)
    logs = Path(logs_dir)
    if not logs.is_dir() or not any(p.is_file() for p in logs.iterdir()):
        _refuse("the log directory is missing or holds no file, and an absent log is never read as a clean one")
    if (dataset_path is None) != (selection_path is None):
        raise ValueError("the dataset and the selection go together")
    if recheck and dataset_path is None:
        raise ValueError("the recheck needs the dataset and the selection")
    dataset = selection = random_pids = schedule = None
    inputs_extra: Dict[str, object] = {}
    if dataset_path is not None:
        ds_sha, sel_sha = RT.file_sha256(dataset_path), RT.file_sha256(selection_path)
        if any(r["dataset_sha256"] != ds_sha or r["selection_sha256"] != sel_sha for r in rows):
            _refuse("the dataset or the selection file differs from the digests in the manifest")
        dataset, selection = json.loads(Path(dataset_path).read_text()), json.loads(Path(selection_path).read_text())
        random_pids = selection.get("random")
        if not (isinstance(random_pids, list) and len(random_pids) == RT.RANDOM_SET_SIZE
                and all(isinstance(p, str) for p in random_pids)):
            _refuse("the selection does not list the 200 random held-out problems")
        schedule = selection.get("sizes", {}).get("0.5b", {}).get("schedule")
        if not (isinstance(schedule, list) and len(schedule) >= STEPS and all(isinstance(x, list) for x in schedule)):
            _refuse("the selection holds no problem schedule of 300 steps for the 0.5B size")
        inputs_extra = {"dataset_sha256": ds_sha, "selection_sha256": sel_sha}
    if not Path(runs_root).is_dir():
        _refuse("the runs directory does not exist")
    by_task, unmatched = find_study_runs(runs_root)
    stray = sorted(t for t in by_task if t not in {int(r["task_id"]) for r in rows})
    if stray:
        _refuse(f"the runs root holds version-2 study runs for task ids {stray}, which the manifest does not list")
    root = Path(runs_root)
    ctx = {"frozen": frozen, "manifest_sha256": manifest_digest, "logs_dir": logs, "dataset": dataset,
           "selection": selection, "random_pids": random_pids, "schedule": schedule, "recheck": recheck, "runs_root": root,
           "unmatched": unmatched}
    entries, series_by_task = [], {}
    for row in rows:
        entry, series = admit_row(row, by_task.get(int(row["task_id"]), []), ctx)
        entries.append(entry)
        if series is not None:
            series_by_task[str(entry["task_id"])] = {"arm": entry["arm"], "seed": entry["seed"],
                                                     "steps_completed": entry["steps_completed"],
                                                     "measured_only": entry["measured_only"], **series}
    named = _check_not_run(not_run, rows, entries)
    for e in entries:
        if e["task_id"] in named:
            e["status"], e["not_run_reason"] = "not_run", named[e["task_id"]]
    matched = {p for runs in by_task.values() for r in runs for p in study_log_files(logs, r["prov"])}
    unowned, log_arrays, other_logs, unreadable_logs = scan_logs(logs, matched)
    arrays = arrays_block(by_task, log_arrays, other_logs)
    blocked = [e for e in entries if not e["scorable"] and e["task_id"] not in named]
    if blocked:
        # The last sentence names the array job ids the scorer saw, because a row without a run directory often
        # means that the pull holds another array than the one that the logs belong to.
        _refuse(_blocked_message(blocked) + " " + arrays_headline(arrays))
    if not any(e["scorable"] for e in entries):
        _refuse("no row is scorable, so there is nothing to score. " + arrays_headline(arrays))
    scorable = {(e["arm"], e["seed"]): e for e in entries if e["scorable"]}
    primary = score_claims(scorable)
    alternatives = []
    for rid, opts in ALTERNATIVE_READINGS:
        mode = opts.get("mismatch")
        pool = scorable if mode is None else with_mismatch_rows(scorable, entries, mode)
        alt = score_claims(pool, **{k: v for k, v in opts.items() if k != "mismatch"})
        alternatives.append({"reading": rid, "options": opts, "verdicts": alt["verdicts"], "branch": alt["branch"]})
        if rid == "gate_stopped_early":
            # The second gate is always printed, whether or not it differs from the first.
            primary["gate"]["complete_rows_only"] = {key: alt["gate"][key] for key in (
                "verdict", "k", "c", "satisfying_seeds", "comparable_seeds", "non_comparable_seeds")}
    for name in ("gate", "d1", "d2", "d3", "within_0_10_clause"):
        differ = {a["reading"]: a["verdicts"][name] for a in alternatives if a["verdicts"][name] != primary["verdicts"][name]}
        primary[name]["verdict_depends_on_reading"] = bool(differ)
        primary[name]["alternative_verdicts_that_differ"] = differ
        if name != "gate" and primary["verdicts"]["gate"] == FAILS:
            primary[name]["note"] = "not supported by a failed control"
    branch = primary["branch"]
    branch_differs = {a["reading"]: a["branch"] for a in alternatives if a["branch"] != branch}
    not_admitted = [{"task_id": e["task_id"], "arm": e["arm"], "seed": e["seed"], "status": e["status"]}
                    for e in entries if not e["admitted"]]
    decision = {
        "branch": branch, "condition": BRANCH_LINES[branch], "verdicts": primary["verdicts"],
        "paper_statement": paper_statement(branch, primary["d2"]),
        "line": (NO_BRANCH_LINE if branch in ("NO_BRANCH_A", "NO_BRANCH_B") else PENDING_LINE if branch == "PENDING" else None),
        "rows_not_admitted": not_admitted if branch == "PENDING" else [],
        "d2_as_it_falls_out": {"verdict": primary["d2"]["verdict"], "k": primary["d2"]["k"], "c": primary["d2"]["c"]},
        "alternative_branches_that_differ": branch_differs, "verdict_depends_on_reading": bool(branch_differs)}
    report = {
        "schema": "study_v2_scoring_1",
        "rule": {"accuracy_band": ACC_BAND, "collapse_gap": GAP, "parse_minimum": PARSE_MIN, "within_ten_band": LOOSE_BAND,
                 "float_tolerance": FLOAT_TOL, "item_tolerance": ITEM_TOL, "random_evaluation_completions": N_RANDOM,
                 "accuracy_band_counts": ACC_BAND_COUNTS, "collapse_gap_counts": GAP_COUNTS,
                 "parse_minimum_count": PARSE_MIN_COUNT, "within_ten_band_counts": LOOSE_BAND_COUNTS,
                 "training_completions_per_step": PER_STEP, "parse_below_half_max_count": PARSE_BELOW_HALF_MAX,
                 "window_steps": list(WINDOW), "window_completions": WINDOW_COMPLETIONS,
                 "seeds_planned": list(SEEDS), "seeds_needed": SEEDS_NEEDED},
        "inputs": {"manifest_file": Path(manifest).name, "manifest_sha256": manifest_digest, "manifest_rows": len(rows),
                   "frozen_reasoning_sources_digest": frozen["reasoning_sources_digest"],
                   "frozen_training_sources_digest": frozen["training_sources_digest"],
                   "recheck_from_completions": bool(recheck), "not_run": {str(k): v for k, v in sorted(named.items())},
                   **inputs_extra},
        "scorer": {Path(__file__).name: sha256_bytes(Path(__file__).read_bytes()),
                   Path(ar.__file__).name: sha256_bytes(Path(ar.__file__).read_bytes())},
        "design_complete": all(e["admitted"] for e in entries),
        "rows_not_admitted": not_admitted,
        "rows": entries,
        "d_gate": primary["gate"], "d1": primary["d1"], "d2": primary["d2"], "d3": primary["d3"],
        "within_0_10_clause": primary["within_0_10_clause"],
        "alternative_readings": alternatives,
        "decision_rule": decision,
        "descriptive_no_prediction": descriptive_differences(scorable),
        "d4_series_file": "study_d4_series.json",
        "integrity": integrity_block(entries),
        "readings": [dict(r) for r in READINGS],
        "arrays": arrays,
        "log_files_with_REFUSED_or_Traceback_not_matched_to_a_row": unowned,
        "log_files_that_could_not_be_read": unreadable_logs,
        "run_directories_not_matched_to_a_study_row": unmatched,
    }
    return report, {"schema": "study_v2_d4_series_1", "rows": series_by_task}


def render_json(obj) -> str:
    """Return the deterministic JSON text of a report. NaN and Infinity are refused."""
    return json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n"


# ------------------------------------------------------------------ the Markdown table


def _cell(x, spec: str = "") -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if spec and isinstance(x, (int, float)):
        return format(x, spec)
    return str(x)


def _md(header: List[str], body: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return lines


def _seeds_text(seeds: List[int]) -> str:
    return ", ".join(str(s) for s in seeds) if seeds else "none"


def _list_text(items) -> str:
    """Join words with commas and the word and, as in 6240177, 6240999 and 6241000."""
    items = [str(x) for x in items]
    return " and ".join(items) if len(items) <= 2 else ", ".join(items[:-1]) + " and " + items[-1]


def _count_text(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _tasks_text(tasks) -> str:
    """Return task ids as text with the word task or tasks, as in task 3 or tasks 1 to 4."""
    return ("task " if len(set(tasks)) == 1 else "tasks ") + spans(tasks)


def arrays_headline(arrays: dict) -> str:
    """Return the sentence under the title that names the arrays the rows come from."""
    by_array = arrays["run_directories_by_array_job_id"]
    ids = [k for k in by_array if k != "none recorded"]
    parts = []
    if len(ids) == 1:
        parts.append(f"The run directories come from array job id {ids[0]}.")
    elif ids:
        parts.append(f"The run directories come from {len(ids)} arrays, with array job ids {_list_text(ids)}.")
    if "none recorded" in by_array:
        parts.append("Some run directories record no array job id.")
    only_logs = arrays["array_job_ids_with_log_files_and_no_run_directory"]
    if only_logs:
        many = len(only_logs) > 1
        parts.append(f"The log directory also holds log files of array job id{'s' if many else ''} {_list_text(only_logs)}, "
                     f"and no run directory belongs to {'them' if many else 'it'}.")
    return " ".join(parts) if parts else "No run directory records an array job id."


def inputs_lines(report: dict) -> List[str]:
    """Return the lines of the Markdown section that lists the arrays, the logs and the directories the scorer saw.

    The section also says which optional checks did not run, so that a reader of the table alone can tell."""
    arrays = report["arrays"]
    lines = [f"Array job id {job} holds the run directories of {_tasks_text(tasks)}."
             for job, tasks in arrays["run_directories_by_array_job_id"].items()]
    for job, g in arrays["log_files_by_array_job_id"].items():
        lines.append(f"The log directory holds {_count_text(g['files'], 'log file', 'log files')} of array job id {job} "
                     f"for {_tasks_text(g['task_ids'])}, with {_count_text(g['files_with_REFUSED_or_Traceback'], 'file', 'files')} "
                     f"holding REFUSED or Traceback and {_count_text(g['files_owned_by_a_row'], 'file', 'files')} "
                     "belonging to a row.")
    other = arrays["log_files_not_named_like_array_task_logs"]
    if other:
        lines.append(f"In the log directory, {_count_text(other, 'file is', 'files are')} not named like an array task log.")
    lines.append("The rows come from more than one array." if arrays["rows_come_from_more_than_one_array"]
                 else "The rows come from one array.")
    renamed: Dict[Tuple[str, str], List[int]] = {}
    for r in arrays["run_directory_parents_with_a_digit_name_that_differs_from_the_array_job_id"]:
        renamed.setdefault((r["directory_name"], r["array_job_id"]), []).append(r["task_id"])
    for (name, job), tasks in sorted(renamed.items()):
        one = len(tasks) == 1
        lines.append(f"The run {'directory' if one else 'directories'} of {_tasks_text(tasks)} {'sits' if one else 'sit'} under a "
                     f"directory named {name}, and {'its' if one else 'their'} provenance.json records array job id {job}. "
                     "This does not refuse a row.")
    left_out = len(report["run_directories_not_matched_to_a_study_row"])
    if left_out == 0:
        lines.append("The runs root holds no run directory that is not a version-2 study row.")
    else:
        lines.append(f"The scorer left out {_count_text(left_out, 'run directory, because it is', 'run directories, because they are')} "
                     f"not {'a version-2 study row' if left_out == 1 else 'version-2 study rows'}.")
    for row in report["rows"]:
        m = row["mismatch"]
        if m:
            lines.append(f"The only problem of task {row['task_id']} is a logged random held-out value that disagrees with its record. "
                         "The readings mismatch_logged_value and mismatch_recomputed_value score the study with the logged value "
                         f"of this row and with the value that its {m['recomputed_from']} give.")
            unusable = sorted(k for k, v in m["readings"].items() if not v["usable"])
            if unusable:
                lines.append(f"For task {row['task_id']}, the {_list_text(unusable)} value is not a count of the 400 completions, "
                             "so the row stays out under that reading.")
    lines.append(f"The log directory holds {_count_text(len(report['log_files_with_REFUSED_or_Traceback_not_matched_to_a_row']), 'file', 'files')} "
                 "with REFUSED or Traceback that no row owns.")
    unreadable = report["log_files_that_could_not_be_read"]
    if unreadable:
        lines.append(f"The scorer could not read {_count_text(len(unreadable), 'log file', 'log files')}: {', '.join(unreadable)}.")
    if "dataset_sha256" not in report["inputs"]:
        lines.append("The training groups and the held-out problems were not compared with the selection file, "
                     "because no dataset and selection were given.")
    if not report["inputs"]["recheck_from_completions"]:
        lines.append("The held-out accuracies and strict parse rates were not recomputed from completions.jsonl, "
                     "because the recheck was not requested.")
    return lines


def held_out_parse_rates(held_out: Optional[dict], d4: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
    """Return the strict parse rates of the random held-out set at step 0 and at the last step, or None for each.

    A row keeps them in its held-out values. A mismatch row has no held-out values, and it keeps the two parse counts in its
    D4 summary when both readings of its mismatch give the same counts."""
    if held_out:
        return held_out["strict_parse_rate_first"], held_out["strict_parse_rate_last"]
    if d4 and d4["held_out_completions"]:
        n = d4["held_out_completions"]
        return d4["held_out_strict_parse_count_first"] / n, d4["held_out_strict_parse_count_last"] / n
    return None, None


def measured_only_line(e: dict) -> str:
    """Return the line that tells a reader of the table that a row which is not scorable holds measurements only.

    The line names the kinds of value that the table shows for the row, because a mismatch row keeps its D4 values and no
    held-out values."""
    kinds = " and ".join(name for name, value in (("held-out", e["held_out"]), ("D4", e["d4"])) if value is not None)
    return (f"task {e['task_id']} measured only: the {kinds} values in the table come from the files of this row. "
            "The row is not scorable, so no verdict reads them, and the problems listed above say why.")


def study_table(report: dict) -> str:
    """Return the Markdown report: one table of the 12 rows, one table per rule, the branch and the readings."""
    out: List[str] = []
    if not report["design_complete"]:
        names = ", ".join(f"{r['task_id']} ({r['status']})" for r in report["rows_not_admitted"])
        out += [f"INCOMPLETE: these rows are not admitted: {names}. Nothing is imputed.", ""]
    out += ["# GSM8K study at a learning rate of 1e-6: scoring report", "", arrays_headline(report["arrays"]), "",
            "## Rows", ""]
    body = []
    for e in report["rows"]:
        h, d = e["held_out"], e["d4"]
        parse_first, parse_last = held_out_parse_rates(h, d)
        body.append([_cell(e["task_id"]), e["arm"], _cell(e["seed"]), e["status"], _cell(e["stopped_early"]),
                     _cell(e["steps_completed"]),
                     _cell(h and h["accuracy_first"], ".4f"), _cell(h and h["accuracy_last"], ".4f"),
                     _cell(h and h["signed_change"], "+.4f"),
                     _cell(parse_first, ".4f"), _cell(parse_last, ".4f"),
                     _cell(d and d["mean_completion_length_all_steps"], ".2f"),
                     _cell(d and d["mean_completion_length_steps_281_300"], ".2f"),
                     _cell(d and d["window_total_tokens"]),
                     _cell(d and d["training_strict_parse_rate_mean_all_steps"], ".4f"),
                     _cell(d and d["entropy_all_steps"]["pooled_mean_weighted_by_length"], ".4f"),
                     _cell(d and d["first_step_with_training_strict_parse_rate_below_0_5"])])
    out += _md(["task", "arm", "seed", "status", "stopped_early", "steps_completed", "acc_step_0", "acc_last",
                "change", "strict_parse_step_0", "strict_parse_last", "mean_length_all_steps",
                "mean_length_281_300", "T_281_300", "train_strict_parse_mean", "entropy_pooled",
                "first_step_parse_below_0.5"], body)
    out += ["", "A first step of - means that no training step fell below a strict parse rate of 0.5 within the "
            "steps the row completed.", ""]
    for e in report["rows"]:
        for problem in e["problems"]:
            out.append(f"task {e['task_id']} problem: {problem}")
        if e["measured_only"]:
            out.append(measured_only_line(e))
    out += ["", "## Rules", ""]
    claims = [("D-gate", report["d_gate"]), ("D1", report["d1"]), ("D2", report["d2"]), ("D3", report["d3"]),
              ("within-0.10 clause", report["within_0_10_clause"])]
    out += _md(["rule", "verdict", "satisfying seeds", "comparable seeds", "depends on reading"],
               [[name, c["verdict"], _seeds_text(c["satisfying_seeds"]), _seeds_text(c["comparable_seeds"]),
                 "VERDICT DEPENDS ON READING" if c["verdict_depends_on_reading"] else "no"] for name, c in claims])
    out.append("")
    for name, c in claims:
        if c["verdict_depends_on_reading"]:
            alts = ", ".join(f"{rid}: {v}" for rid, v in sorted(c["alternative_verdicts_that_differ"].items()))
            out.append(f"VERDICT DEPENDS ON READING: {name} is {c['verdict']} under the primary reading and {alts}.")
        if c.get("note"):
            out.append(f"{name}: {c['note']}.")
    gate = report["d_gate"]
    out += ["", "## D-gate by seed", ""]
    out += _md(["seed", "task", "stopped_early", "steps_completed", "count_step_0", "count_last", "signed_difference",
                "rise_above_0.05", "within_0.05", "strict_parse_count_last", "parse>=0.80", "satisfies", "flags"],
               [[_cell(d["seed"]), _cell(d.get("task_id")), _cell(d.get("stopped_early")), _cell(d.get("steps_completed")),
                 _cell(d.get("count_first")), _cell(d.get("count_last")), _cell(d.get("signed_difference"), "+.4f"),
                 _cell(d.get("rise_above_0_05")), _cell(d["conditions"].get("within_0_05")),
                 _cell(d.get("strict_parse_count_last")), _cell(d["conditions"].get("parse_at_least_0_80")),
                 _cell(d["satisfies"]), ", ".join(d.get("boundary_flags", [])) or d.get("reason", "-")]
                for d in gate["per_seed"]])
    second = gate["complete_rows_only"]
    out += ["", f"Counted over seeds on their own, the gate is {gate['conditions_counted_separately']['verdict']}.",
            f"With stopped-early rows left out (complete rows only), the gate is {second['verdict']} "
            f"(k = {second['k']}, c = {second['c']}).", ""]
    for title, key, extra in (("D1", "d1", "length"), ("D2", "d2", ""), ("D3", "d3", ""), ("within-0.10 clause", "within_0_10_clause", "")):
        c = report[key]
        out += [f"## {title} by seed", ""]
        header = ["seed", "task_arm", "task_control", "count_arm", "count_control", "arm_minus_control_counts"]
        if extra:
            header += ["T_arm", "T_control"]
        header += ["satisfies", "flags"]
        body = []
        for d in c["per_seed"]:
            row = [_cell(d["seed"]), _cell(d.get("task_id_arm")), _cell(d.get("task_id_control")), _cell(d.get("count_arm")),
                   _cell(d.get("count_control")), _cell(d.get("arm_minus_control_counts"), "+d")]
            if extra:
                row += [_cell(d.get("window_total_tokens_arm")), _cell(d.get("window_total_tokens_control"))]
            row += [_cell(d["satisfies"]), ", ".join(d.get("boundary_flags", [])) or d.get("reason", "-")]
            body.append(row)
        out += _md(header, body)
        if key == "d1":
            out += ["", f"Counted over seeds on their own, D1 is {c['conditions_counted_separately']['verdict']}."]
        out.append("")
    out += ["## Decision rule", ""]
    dr = report["decision_rule"]
    out += [f"Branch: {dr['branch']}", f"Condition: {dr['condition']}"]
    if dr["paper_statement"]:
        out.append(f"Paper statement: {dr['paper_statement']}")
    if dr["line"]:
        out.append(f"Line: {dr['line']}")
    if dr["verdict_depends_on_reading"]:
        alts = ", ".join(f"{rid}: {b}" for rid, b in sorted(dr["alternative_branches_that_differ"].items()))
        out.append(f"VERDICT DEPENDS ON READING: the branch is {dr['branch']} under the primary reading and {alts}.")
    out += ["", "## Descriptive differences, no prediction", ""]
    desc = []
    for rec in report["descriptive_no_prediction"]:
        for name in ("restored_minus_plain", "vfilter_minus_plain"):
            v = rec[name]
            desc.append([_cell(rec["seed"]), name, _cell(v and v["count_step_300"], "+d"),
                         _cell(v and v["mean_length_steps_281_300"], "+.2f")])
    out += _md(["seed", "difference", "count_step_300", "mean_length_281_300"], desc)
    integ = report["integrity"]
    out += ["", "## Integrity lines, which never gate", ""]
    out.append("step-0 random records identical across rows: " + _cell(integ["step_0_random_records_identical_across_scorable_rows"]))
    out.append("step-0 counts (correct, strict parse): " + str(integ["step_0_count_pairs_correct_and_strict_parse"]))
    for s in SEEDS:
        g = integ["step_1_training_groups_identical_across_the_arms_of_a_seed"][str(s)]["identical"]
        c = integ["step_1_training_completions_identical_across_the_arms_of_a_seed"][str(s)]["identical"]
        out.append(f"seed {s}: step-1 training groups identical across arms: {_cell(g)}, completions identical: {_cell(c)}")
    out.append("array job ids: " + ", ".join(integ["array_job_ids"]))
    out += ["", "## Inputs seen", ""] + inputs_lines(report)
    out += ["", "## Readings chosen", ""]
    out += _md(["id", "chosen reading"], [[r["id"], r["chosen"]] for r in report["readings"]])
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out) + "\n")


# ------------------------------------------------------------------ the command line


def build_parser(prog: str = "analyze_study2.py") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Score the GSM8K study at a learning rate of 1e-6.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("study", help="admit the 12 study rows, score the D-gate and D1 to D4, apply the decision rule",
                        description="Admit the 12 study rows, score the D-gate and D1 to D4 and apply the decision rule.")
    st.add_argument("--manifest", type=Path, required=True, help="slurm/manifest_reasoning_study2.csv")
    st.add_argument("--frozen", type=Path, required=True, help="configs/reasoning_frozen_v2.json")
    st.add_argument("--runs", type=Path, required=True, help="the directory that holds the study run directories")
    st.add_argument("--logs", type=Path, required=True, help="the directory of the pulled Slurm logs")
    st.add_argument("--out-dir", type=Path, required=True, help="a new directory for the report files")
    st.add_argument("--dataset", type=Path, default=None, help="configs/reasoning_dataset.json, with --selection")
    st.add_argument("--selection", type=Path, default=None, help="configs/reasoning_selection.json, with --dataset")
    st.add_argument("--recheck", action="store_true",
                    help="recompute the random held-out accuracy and strict parse rate at step 0 and the last step "
                         "from completions.jsonl and refuse when a value differs from the logged one")
    st.add_argument("--not-run", type=Path, default=None,
                    help="a JSON object from task ids to the reason a row will not be rerun")
    return ap


def write_report_files(out_dir: Path, files: Dict[str, str]) -> None:
    """Create each report file in the output directory, so that either all of them exist or none does.

    A file is created with the exclusive mode, so an existing file is never overwritten. When any write fails, the
    files that this call created are removed again, because a table with a missing part must not be left behind."""
    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    try:
        for name, text in files.items():
            path = out_dir / name
            with path.open("x", encoding="utf-8", newline="\n") as fh:
                created.append(path)
                fh.write(text)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def study_main(args: argparse.Namespace) -> int:
    out_files = ("study_d.json", "study_d.md", "study_d4_series.json")
    try:
        frozen = json.loads(args.frozen.read_text())
        not_run = json.loads(args.not_run.read_text()) if args.not_run else None
        report, series = study_report(args.manifest, args.runs, frozen, args.logs, dataset_path=args.dataset,
                                      selection_path=args.selection, recheck=args.recheck, not_run=not_run)
        texts = (render_json(report), study_table(report), render_json(series))
        held = [n for n in out_files if (args.out_dir / n).exists()]
        if held:
            _refuse(f"the output directory already holds {', '.join(held)}, and a report is never overwritten")
        write_report_files(args.out_dir, dict(zip(out_files, texts)))
    # Every failure ends as one REFUSED line. The report is built in memory first, so a refusal writes no file.
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED: {exc}" if isinstance(exc, (_ar().Refusal, ValueError, OSError))
              else f"REFUSED: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return REFUSED_STATUS
    print(json.dumps({"out_dir": str(args.out_dir), "design_complete": report["design_complete"],
                      "array_job_ids": list(report["arrays"]["run_directories_by_array_job_id"]),
                      "branch": report["decision_rule"]["branch"], "verdicts": report["decision_rule"]["verdicts"]}))
    return 0


def main(argv=None, prog: str = "analyze_study2.py") -> int:
    """The study subcommand. It returns 0 when a report is written and REFUSED_STATUS (3) when the scorer refuses."""
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = build_parser(prog)
    args = ap.parse_args(argv)
    if args.cmd == "study":
        if (args.dataset is None) != (args.selection is None):
            ap.error("--dataset and --selection go together")
        if args.recheck and args.dataset is None:
            ap.error("--recheck needs --dataset and --selection")
        return study_main(args)
    # The parser accepts no other subcommand, so this line cannot run.
    return 2  # pragma: no cover  (2 is the status of a wrong command line, as in argparse)


if __name__ == "__main__":
    sys.exit(main())
