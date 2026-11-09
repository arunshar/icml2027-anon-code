"""This module defines the WO-I reasoning task on GSM8K, with its parsers, rewards and design.

Every rule here is fixed before any calibration output exists. The written plan states the rules that
the constants below encode. A completion earns R = w_task T - w F under the linear form and
R = w_task T (1 - F) - w F under the gated form. F is the strict verdict, which is 1 when the
strict parser finds no answer or a wrong one. T is the brevity term 1 - n / L.

This module runs on the cluster, so it is listed in REASONING_SOURCES and its content is part of
the WO-I source digest. It imports numpy and the frozen training modules only.

Design version 2 (ICML 2027, Block D' of the plan of November 2026) adds the restored arm, the
version-2 manifest header with lr, kl_beta, skip_zero_adv and fixedref_c, and the pilot and study2
row rules. Every version-1 row still validates exactly as it did.
"""
from __future__ import annotations

import hashlib
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

import train_speed_grpo as TSG

# ------------------------------------------------------------------ prompt

PROMPT_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def prompt_text(question: str) -> str:
    """Return the user message, which is the problem text, a newline and the fixed instruction."""
    return question.strip() + "\n" + PROMPT_SUFFIX


# ------------------------------------------------------------------ parsers

_BOX = "\\boxed{"
# The LaTeX spacing commands are spaces, so the strict parser removes them with the spaces.
_LATEX_SPACES = ("\\,", "\\;", "\\:", "\\!", "\\ ", "~")
_NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+")
_RATIO = re.compile(r"([+-]?\d+)/(\d+)")
_LENIENT = re.compile(r"(?<![\d.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def last_boxed(text: str) -> Optional[str]:
    """Return the content of the last \\boxed{ in the text, read up to its balancing brace.

    The last occurrence decides. When its braces never close, as in a completion cut at the
    generation limit, there is no answer and the function returns None."""
    start = text.rfind(_BOX)
    if start < 0:
        return None
    depth, i = 1, start + len(_BOX)
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(_BOX):i]
        i += 1
    return None


def parse_rational(s: str) -> Optional[Fraction]:
    """Parse a plain decimal or an integer ratio as an exact rational number, or return None.

    Commas are accepted only as thousands separators in groups of three digits."""
    if _NUMBER.fullmatch(s):
        return Fraction(s.replace(",", ""))
    m = _RATIO.fullmatch(s)
    if m and int(m.group(2)) != 0:
        return Fraction(int(m.group(1)), int(m.group(2)))
    return None


def strict_parse(text: str) -> Optional[Fraction]:
    """Apply the strict parser. It reads the last \\boxed{} with balanced braces, removes spaces, LaTeX
    spacing commands and dollar signs, reads {,} as a thousands separator, and parses an exact
    rational number. Anything else in the box, such as a unit or a LaTeX fraction, fails."""
    content = last_boxed(text)
    if content is None:
        return None
    s = content
    for token in _LATEX_SPACES:
        s = s.replace(token, "")
    s = s.replace("\\$", "").replace("$", "").replace("{,}", ",")
    s = "".join(s.split())
    return parse_rational(s)


def lenient_parse(text: str) -> Optional[Fraction]:
    """Return the last number in the completion. It is logged only and never enters a reward."""
    found = _LENIENT.findall(text)
    if not found:
        return None
    return Fraction(found[-1].replace(",", ""))


def reference_answer(solution: str) -> Fraction:
    """Return the number after the last #### of a GSM8K solution, and raise when it does not parse."""
    if "####" not in solution:
        raise ValueError("the reference solution has no #### marker")
    tail = "".join(solution.rsplit("####", 1)[1].split())
    value = parse_rational(tail)
    if value is None:
        raise ValueError(f"the reference answer {tail!r} does not parse")
    return value


def fraction_str(x: Optional[Fraction]) -> Optional[str]:
    """Return a canonical string for a parsed value, so logs compare exactly. None stays None."""
    if x is None:
        return None
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def parse_fraction_str(s: Optional[str]) -> Optional[Fraction]:
    return None if s is None else Fraction(s)


def verdict_fail(text: str, reference: Fraction) -> int:
    """Return F, which is 0 only when the strict parser finds a number equal to the reference answer."""
    parsed = strict_parse(text)
    return 0 if (parsed is not None and parsed == reference) else 1


# ------------------------------------------------------------------ rewards

FORMS = ("linear", "gated")


def brevity(n_tokens: Sequence[int], L: int) -> np.ndarray:
    """Return T = 1 - n / L for completions of n tokens through the first end token, with 1 <= n <= L."""
    n = np.asarray(n_tokens, dtype=np.int64)
    if L < 1:
        raise ValueError("the generation limit L must be positive")
    if n.size and (n.min() < 1 or n.max() > L):
        raise ValueError(f"every completion length must lie between 1 and L = {L}")
    return 1.0 - n.astype(np.float64) / float(L)


def reasoning_rewards(fails: Sequence[int], n_tokens: Sequence[int], form: str, w: float,
                      w_task: float, L: int) -> np.ndarray:
    """Return one group's rewards. The linear form is w_task T - w F, and the gated one is w_task T (1 - F) - w F."""
    f = np.asarray(fails, dtype=np.float64)
    if f.shape != np.asarray(n_tokens).shape:
        raise ValueError("fails and n_tokens must have one entry per completion")
    if not np.isin(f, (0.0, 1.0)).all():
        raise ValueError("every verdict F must be 0 or 1")
    t = brevity(n_tokens, L)
    if form == "linear":
        return w_task * t - w * f
    if form == "gated":
        return w_task * t * (1.0 - f) - w * f
    raise ValueError(f"unknown reward form {form!r}, expected one of {FORMS}")


def task_only_rewards(n_tokens: Sequence[int], w_task: float, L: int) -> np.ndarray:
    """Return w_task T alone, which W0 (b) compares against in all-fail groups."""
    return w_task * brevity(n_tokens, L)


# ------------------------------------------------------------------ design

W = 2.0
GROUP_SIZE = 4
GROUPS_PER_STEP = 6
SIZES = ("0.5b", "1.5b")
MODELS = {
    "0.5b": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct",
             # This is a locally cached snapshot, which equaled the Hub's main revision on 22 September.
             # U0 reads the cluster cache, and a different cluster snapshot is pinned in the manifest.
             "revision": "7ae557604adf67be50417f59c2c2f167def9a775"},
    "1.5b": {"model_id": "Qwen/Qwen2.5-1.5B-Instruct",
             "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"},
}
ARMS: Dict[str, Dict[str, object]] = {
    "plain": {"form": "linear", "w_task": 1.0, "adv_variant": "grpo", "group_filter": "none"},
    "vfilter": {"form": "linear", "w_task": 1.0, "adv_variant": "grpo", "group_filter": "verdict_mixed"},
    "drgrpo": {"form": "linear", "w_task": 1.0, "adv_variant": "drgrpo", "group_filter": "none"},
    "gate": {"form": "gated", "w_task": 1.0, "adv_variant": "grpo", "group_filter": "none"},
    "verdict_only": {"form": "linear", "w_task": 0.0, "adv_variant": "grpo", "group_filter": "none"},
    # ICML 2027 (November 2026), design version 2 only. The restored arm is the plain reward with the
    # fixed-reference verdict of rewards.advantage_fixedref in homogeneous-verdict groups, so it
    # agrees with plain on every mixed group and differs only where centering cancels the verdict.
    "restored": {"form": "linear", "w_task": 1.0, "adv_variant": "grpo_fixedref", "group_filter": "none"},
}
STUDY_ARMS = {"0.5b": ("plain", "vfilter", "drgrpo", "gate", "verdict_only"),
              "1.5b": ("plain", "vfilter", "drgrpo", "gate")}
STAGE_OF_SIZE = {"0.5b": 1, "1.5b": 2}

# ------------------------------------------------------------------ design version 2
# Version 2 of the GSM8K block (ICML 2027, Block D' of the plan of November 2026) sits beside version 1.
# Every version-1 row validates exactly as before. A version-2 row carries four more fields, which
# the WO-I collapse investigation of 25 September 2026 named as the untested levers: the learning
# rate, a k3 KL penalty to a frozen reference, the zero-advantage optimizer skip, and the
# fixed-reference shift c of the restored arm.
DESIGN_VERSION_V1 = 1
DESIGN_VERSION_V2 = 2
DESIGN_VERSIONS = (DESIGN_VERSION_V1, DESIGN_VERSION_V2)
LR_DEFAULT = 1e-5
PILOT_LR_CHOICES = (1e-6, 5e-7)
PILOT_KL_CHOICES = (0.02, 0.05)
LR_CHOICES_V2 = (LR_DEFAULT,) + PILOT_LR_CHOICES
KL_CHOICES_V2 = (0.0,) + PILOT_KL_CHOICES
FIXEDREF_C = 2.0
PILOT_STAGE = 3
PILOT_ARM = "verdict_only"
PILOT_SEED = 0
PILOT_STEPS = 100
# Every version-2 row runs at the selection's L of 512, the value the version-1 study ran at.
L_V2 = 512
PILOT_L = L_V2
# The one pilot row that changes two things at once. It is the retry combination of the plan: the
# zero-advantage skip together with the smaller KL penalty, at the default learning rate.
PILOT_RETRY_COMBINATION = {"lr": LR_DEFAULT, "kl_beta": 0.02, "skip_zero_adv": 1}
STUDY_STAGE_V2 = 4
STUDY_STEPS_V2 = 300
STUDY_ARMS_V2 = ("verdict_only", "plain", "restored", "vfilter")
STUDY_SIZE_V2 = "0.5b"
SEEDS = (0, 1, 2)
SMOKE_SEED = 100
SMOKE_STEPS = 20
# The smoke evaluates the hard set with one sample at step 10, so the S rule reads a measured
# middle-evaluation time. No study run evaluates at a step that is not a multiple of 50.
SMOKE_MID_EVAL_EVERY = 10
EVAL_SEED = 12345

# These constants encode the calibration and selection rules of Section 2.3.
N_TRAIN_CANDIDATES = 1000
N_TEST_CANDIDATES = 400
CANDIDATE_SEED = 20260922
CAL_SAMPLES = 8
CAL_MAX_NEW_TOKENS = 512
CAL_CHUNK_PROBLEMS = 8
# Every calibration attempt records these fields for itself, and the selection refuses a task whose
# attempts differ on any of them. The bundle digest is the SHA-256 of the bundle's SHA256SUMS file.
# The design holds the task's seed, sample count, token cap and chunk size, so a retry that runs
# another design is refused too.
CAL_ATTEMPT_FIELDS = ("reasoning_sources_digest", "training_sources_digest", "bundle_digest", "dataset_sha256",
                      "model_snapshot", "device_name", "tiny", "design")
L_CANDIDATES = (256, 384, 512)
L_MAX_EXCEED_SHARE = 0.05
POOL_TARGET = 0.6
POOL_FLOOR = 300
HARD_CAP = 200
HARD_FLOOR = 100
RANDOM_SET_SIZE = 200
MIN_PARSE_RATE = 0.5
S_CANDIDATES = (300, 200, 150)
PLANNING_MINUTES = 120.0
MICRO_BATCH = 2
# The memory rule compares torch.cuda.max_memory_reserved(), in bytes, with 36 x 10^9.
MEMORY_LIMIT_BYTES = 36 * 10**9
EVAL_EVERY = 50
EVAL_SAMPLES_HARD_FULL = 4
EVAL_SAMPLES_RANDOM = 2
EVAL_SAMPLES_HARD_MID = 1
EVAL_CHUNK_SEQUENCES = 64
REQUIRED_GPU_NAME = "NVIDIA A100-SXM4-40GB"
MIN_MINUTES = {"calibration": 120, "study": 120, "smoke": 60, "pilot": 60}


def expected_all_fail(c: int, samples: int = CAL_SAMPLES, k: int = GROUP_SIZE) -> float:
    """Return C(samples - c, k) / C(samples, k), the unbiased estimate of (1 - p) ** k."""
    if not 0 <= c <= samples:
        raise ValueError("c must lie between 0 and the number of samples")
    return math.comb(samples - c, k) / math.comb(samples, k)


def expected_all_pass(c: int, samples: int = CAL_SAMPLES, k: int = GROUP_SIZE) -> float:
    if not 0 <= c <= samples:
        raise ValueError("c must lie between 0 and the number of samples")
    return math.comb(c, k) / math.comb(samples, k)


def eval_plan(step: int, steps: int, mid_every: int = EVAL_EVERY) -> List[tuple]:
    """Return the held-out evaluations at one step as (set name, samples per problem) pairs.

    Step 0 and the last step evaluate the hard set with 4 samples and the random set with 2.
    Every mid_every-th step between them evaluates the hard set with 1 sample."""
    if step == 0 or step == steps:
        return [("hard", EVAL_SAMPLES_HARD_FULL), ("random", EVAL_SAMPLES_RANDOM)]
    if mid_every and step % mid_every == 0:
        return [("hard", EVAL_SAMPLES_HARD_MID)]
    return []


def project_minutes(load_seconds: float, step_seconds: Sequence[float], full_eval_seconds: float,
                    mid_eval_seconds: float, steps: int) -> float:
    """Project the wall minutes of one run from smoke timing fields only.

    The step time is the mean of every smoke step after the first, because the first step
    carries one-time warm-up. The run pays two full evaluations and one middle evaluation at
    every multiple of 50 strictly between 0 and S."""
    if len(step_seconds) < 2:
        raise ValueError("the projection needs at least two smoke steps")
    per_step = float(np.mean(step_seconds[1:]))
    n_mid = len([s for s in range(1, steps) if s % EVAL_EVERY == 0])
    total = load_seconds + steps * per_step + 2 * full_eval_seconds + n_mid * mid_eval_seconds
    return total / 60.0


def choose_steps(projected: Dict[int, float]) -> Optional[int]:
    """Return the largest S whose projection fits PLANNING_MINUTES, or None when none fits."""
    for s in S_CANDIDATES:
        if projected[s] <= PLANNING_MINUTES:
            return s
    return None


# ------------------------------------------------------------------ manifest rows

CALIBRATION_HEADER = ("task_id", "kind", "stage", "size", "model_id", "model_revision", "split",
                      "samples", "max_new_tokens", "chunk_problems", "seed", "dataset_sha256")
ARM_HEADER = ("task_id", "kind", "stage", "size", "model_id", "model_revision", "arm", "form",
              "w", "w_task", "adv_variant", "group_filter", "seed", "steps", "L", "micro_batch",
              "dataset_sha256", "selection_sha256")
# A version-2 row adds the four levers after the version-1 fields. skip_zero_adv is 0 or 1.
ARM_HEADER_V2 = ARM_HEADER + ("lr", "kl_beta", "skip_zero_adv", "fixedref_c")
_SHA = re.compile(r"[0-9a-f]{64}")
_REV = re.compile(r"[0-9a-f]{40}")


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def validate_calibration_row(row: Dict[str, str]) -> Dict[str, object]:
    """Refuse any calibration row outside the design, and return it with typed fields."""
    _require(tuple(row.keys()) == CALIBRATION_HEADER, "the calibration row has the wrong fields")
    _require(row["kind"] == "calibration" and row["stage"] == "0", "a calibration row is stage 0")
    _require(row["size"] in SIZES, "unknown size")
    _require(row["model_id"] == MODELS[row["size"]]["model_id"], "the model does not match the size")
    _require(bool(_REV.fullmatch(row["model_revision"])), "the model revision must be a commit hash")
    _require(row["split"] in ("train", "test"), "the split must be train or test")
    _require(row["samples"] == str(CAL_SAMPLES), "calibration samples 8 completions per problem")
    _require(row["max_new_tokens"] == str(CAL_MAX_NEW_TOKENS), "calibration runs at 512 tokens")
    _require(row["chunk_problems"] == str(CAL_CHUNK_PROBLEMS), "calibration batches 8 problems")
    _require(row["seed"].isdigit(), "the seed must be a non-negative integer")
    _require(bool(_SHA.fullmatch(row["dataset_sha256"])), "the dataset digest must be a SHA-256")
    _require(row["task_id"].isdigit() and int(row["task_id"]) >= 1, "task ids start at 1")
    out: Dict[str, object] = dict(row)
    for key in ("task_id", "stage", "samples", "max_new_tokens", "chunk_problems", "seed"):
        out[key] = int(row[key])
    return out


def design_version_of(row: Dict[str, str]) -> int:
    """Return 1 for a row with the version-1 fields and 2 for one with the version-2 fields."""
    keys = tuple(row.keys())
    if keys == ARM_HEADER:
        return DESIGN_VERSION_V1
    if keys == ARM_HEADER_V2:
        return DESIGN_VERSION_V2
    raise ValueError("the arm row has the wrong fields")


def pilot_changes(lr: float, kl_beta: float, skip_zero_adv: int) -> List[str]:
    """Return the names of the levers a pilot row moves away from the version-1 loop."""
    changed = []
    if lr != LR_DEFAULT:
        changed.append("lr")
    if kl_beta > 0:
        changed.append("kl_beta")
    if skip_zero_adv == 1:
        changed.append("skip_zero_adv")
    return changed


def _validate_v2_fields(row: Dict[str, str]) -> None:
    """Refuse a version-2 row whose levers lie outside the design. The restored arm carries c = 2."""
    lr, kl, c = float(row["lr"]), float(row["kl_beta"]), float(row["fixedref_c"])
    _require(row["skip_zero_adv"] in ("0", "1"), "skip_zero_adv must be 0 or 1")
    skip = int(row["skip_zero_adv"])
    _require(lr in LR_CHOICES_V2, "lr must be one of 1e-5, 1e-6 and 5e-7")
    _require(kl in KL_CHOICES_V2, "kl_beta must be one of 0, 0.02 and 0.05")
    want_c = FIXEDREF_C if row["arm"] == "restored" else 0.0
    _require(c == want_c, f"fixedref_c must be {want_c} for the {row['arm']} arm")
    if row["kind"] == "pilot":
        _require(row["stage"] == str(PILOT_STAGE), "a pilot row is stage 3")
        _require(row["arm"] == PILOT_ARM, "a pilot row runs the verdict_only arm")
        _require(int(row["seed"]) == PILOT_SEED and int(row["steps"]) == PILOT_STEPS,
                 "a pilot row runs 100 steps at seed 0")
        _require(int(row["L"]) == PILOT_L, "a pilot row runs at L = 512")
        changed = pilot_changes(lr, kl, skip)
        is_retry = {"lr": lr, "kl_beta": kl, "skip_zero_adv": skip} == PILOT_RETRY_COMBINATION
        _require(len(changed) == 1 or is_retry,
                 "a pilot row changes exactly one lever, or it is the documented retry combination")
    else:
        _require(row["stage"] == str(STUDY_STAGE_V2), "a version-2 study row is stage 4")
        _require(row["size"] == STUDY_SIZE_V2, "the version-2 study runs at 0.5B only")
        _require(row["arm"] in STUDY_ARMS_V2, "this arm is not in the version-2 study")
        _require(int(row["seed"]) in SEEDS, "study seeds are 0, 1 and 2")
        _require(int(row["steps"]) == STUDY_STEPS_V2, "a version-2 study row runs 300 steps")
        _require(int(row["L"]) == L_V2, "a version-2 study row runs at L = 512")


def validate_arm_row(row: Dict[str, str]) -> Dict[str, object]:
    """Refuse any smoke, study or pilot row outside the design, and return it with typed fields.

    A version-1 row (ARM_HEADER) validates exactly as it did before version 2 existed. A version-2 row
    (ARM_HEADER_V2) is a pilot or a study row, and the manifest generator, not this function, checks
    that every version-2 study row shares one lr, kl_beta and skip_zero_adv."""
    version = design_version_of(row)
    if version == DESIGN_VERSION_V1:
        _require(row["kind"] in ("smoke", "study"), "an arm row is a smoke or a study row")
    else:
        _require(row["kind"] in ("pilot", "study"), "a version-2 arm row is a pilot or a study row")
    size = row["size"]
    _require(size in SIZES, "unknown size")
    _require(row["model_id"] == MODELS[size]["model_id"], "the model does not match the size")
    _require(bool(_REV.fullmatch(row["model_revision"])), "the model revision must be a commit hash")
    _require(row["arm"] in ARMS, "unknown arm")
    spec = ARMS[row["arm"]]
    _require(row["form"] == spec["form"] and row["adv_variant"] == spec["adv_variant"]
             and row["group_filter"] == spec["group_filter"], "the arm fields differ from the design")
    _require(float(row["w"]) == W and float(row["w_task"]) == spec["w_task"], "the weights differ from the design")
    _require(int(row["L"]) in L_CANDIDATES, "L must be one of 256, 384 and 512")
    _require(int(row["micro_batch"]) == MICRO_BATCH, "the scoring micro-batch is 2")
    _require(bool(_SHA.fullmatch(row["dataset_sha256"])) and bool(_SHA.fullmatch(row["selection_sha256"])),
             "the dataset and selection digests must be SHA-256 values")
    if version == DESIGN_VERSION_V2:
        _validate_v2_fields(row)
    elif row["kind"] == "smoke":
        _require(row["stage"] == "0" and row["arm"] == "plain", "a smoke is the stage 0 plain arm")
        _require(int(row["seed"]) == SMOKE_SEED and int(row["steps"]) == SMOKE_STEPS,
                 "a smoke runs 20 steps at seed 100")
    else:
        _require(row["arm"] in STUDY_ARMS[size], "this arm is not run at this size")
        _require(int(row["stage"]) == STAGE_OF_SIZE[size], "the stage does not match the size")
        _require(int(row["seed"]) in SEEDS, "study seeds are 0, 1 and 2")
        _require(int(row["steps"]) in S_CANDIDATES, "S must be one of 300, 200 and 150")
    _require(row["task_id"].isdigit() and int(row["task_id"]) >= 1, "task ids start at 1")
    out: Dict[str, object] = dict(row)
    for key in ("task_id", "stage", "seed", "steps", "L", "micro_batch"):
        out[key] = int(row[key])
    for key in ("w", "w_task"):
        out[key] = float(row[key])
    if version == DESIGN_VERSION_V2:
        for key in ("lr", "kl_beta", "fixedref_c"):
            out[key] = float(row[key])
        out["skip_zero_adv"] = int(row["skip_zero_adv"])
    out["design_version"] = version
    return out


def read_manifest_row(path, task_id: int) -> Dict[str, str]:
    """Return one manifest row by task id, read as strings. The file must be LF-only."""
    raw = Path(path).read_bytes()
    _require(b"\r" not in raw, "the manifest must be LF-only")
    lines = raw.decode("utf-8").split("\n")
    header = tuple(lines[0].split(","))
    rows = [dict(zip(header, line.split(","))) for line in lines[1:] if line]
    _require(all(len(r) == len(header) for r in rows), "a manifest row has the wrong number of fields")
    hits = [r for r in rows if r["task_id"] == str(task_id)]
    _require(len(hits) == 1, f"the manifest has {len(hits)} rows for task {task_id}")
    return hits[0]


# ------------------------------------------------------------------ frozen sources

_PREFIX = "experiments/cancellation/"
# This tuple names the files that run on the cluster, in this order and nothing else. It starts with the
# training sources of train_speed_grpo.py, which WO-I imports and never edits. The rest
# are new with WO-I. Data files (the dataset, the selection and the manifests) carry their own
# digests and are not part of this list.
REASONING_SOURCES = tuple(TSG.TRAINING_SOURCES) + (
    _PREFIX + "reasoning_task.py",
    _PREFIX + "calibrate_reasoning.py",
    _PREFIX + "train_reasoning_grpo.py",
    _PREFIX + "slurm/download_reasoning.sbatch",
    _PREFIX + "slurm/reasoning_calibrate.sbatch",
    _PREFIX + "slurm/reasoning_arms.sbatch",
    # Added for design version 2 (ICML 2027, November 2026). It is the launcher of the pilot and
    # study2 rows, so it is part of the source digest exactly as the version-1 launcher is.
    _PREFIX + "slurm/reasoning_arms_v2.sbatch",
)


def reasoning_sources_digest(exp_dir=None, sources: Iterable[str] = REASONING_SOURCES) -> str:
    """Return the SHA-256 over each source's path and content hash, in the declared order.

    It is the same construction as training_sources_digest(), so it gives one value in the
    repository and in a flattened bundle. A missing file contributes a fixed marker."""
    base = Path(exp_dir) if exp_dir is not None else Path(__file__).resolve().parent
    h = hashlib.sha256()
    for rel in sources:
        p = base / rel[len(_PREFIX):]
        h.update(rel.encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode() if p.is_file() else b"MISSING")
        h.update(b"\0")
    return h.hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
