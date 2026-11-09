#!/usr/bin/env python3
"""Freeze the WO-I GSM8K candidates into configs/reasoning_dataset.json.

The script reads the two GSM8K `main` parquet files that were downloaded on
23 September 2026. It refuses to read a file whose byte size or SHA-256 differs from the values
recorded below. Every reference answer must parse, and no problem text may appear in both splits.
A fixed seed then draws 1,000 train candidates and 400 test candidates, and the file records the
seed, the draw order, the prompt text and the reference answer of every candidate.

The file is written once in U1 and never edited. The selection of Section 2.3 writes a separate
file that names this file's SHA-256, so every later record can be traced to these candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import parquet_lite
import reasoning_task as RT

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "gsm8k_main"
DEFAULT_OUT = HERE / "configs" / "reasoning_dataset.json"
SCHEMA = "wo_i_reasoning_dataset_v1"

# These values were recorded from the Hub on 22 September 2026, in the listing of openai/gsm8k at
# the revision below. The files downloaded on 23 September 2026 (UTC) match them.
SOURCE = {
    "repo": "openai/gsm8k",
    "config": "main",
    "revision": "740312add88f781978c0658806c59bc2815b9866",
    "license": "mit",
    "files": {
        "train": {"path_in_repo": "main/train-00000-of-00001.parquet", "bytes": 2306545,
                  "sha256": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
                  "rows": 7473},
        "test": {"path_in_repo": "main/test-00000-of-00001.parquet", "bytes": 419088,
                 "sha256": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
                 "rows": 1319},
    },
}


def _normalize(text: str) -> str:
    return " ".join(text.split())


def check_file(path: Path, spec: dict) -> None:
    """Refuse a parquet file whose size or SHA-256 differs from the recorded values."""
    size = path.stat().st_size
    if size != spec["bytes"]:
        raise ValueError(f"{path.name} has {size} bytes, and {spec['bytes']} were recorded")
    digest = RT.file_sha256(path)
    if digest != spec["sha256"]:
        raise ValueError(f"{path.name} has SHA-256 {digest}, and {spec['sha256']} was recorded")


def load_split(path: Path, spec: dict) -> list:
    check_file(path, spec)
    table = parquet_lite.read_string_table(path)
    if set(table) != {"question", "answer"}:
        raise ValueError(f"{path.name} has columns {sorted(table)}, expected question and answer")
    rows = list(zip(table["question"], table["answer"]))
    if len(rows) != spec["rows"]:
        raise ValueError(f"{path.name} has {len(rows)} rows, and {spec['rows']} were recorded")
    if any(q is None or a is None for q, a in rows):
        raise ValueError(f"{path.name} has a missing question or answer")
    return rows


def build(train_rows: list, test_rows: list, seed: int = RT.CANDIDATE_SEED,
          n_train: int = RT.N_TRAIN_CANDIDATES, n_test: int = RT.N_TEST_CANDIDATES,
          source: dict = SOURCE) -> dict:
    """Return the frozen candidate file as a dict, and raise on any failed check."""
    refs = {"train": [RT.reference_answer(a) for _, a in train_rows],
            "test": [RT.reference_answer(a) for _, a in test_rows]}
    train_text = {_normalize(q) for q, _ in train_rows}
    shared = sum(1 for q, _ in test_rows if _normalize(q) in train_text)
    if shared:
        raise ValueError(f"{shared} test problems share their text with a train problem")
    if len(train_text) != len(train_rows) or len({_normalize(q) for q, _ in test_rows}) != len(test_rows):
        raise ValueError("a split holds the same problem text twice")
    rng = random.Random(seed)
    train_idx = rng.sample(range(len(train_rows)), n_train)
    test_idx = rng.sample(range(len(test_rows)), n_test)

    def cands(split, rows, idx, tag):
        return [{"cand": c, "pid": f"{tag}{row}", "row": row, "split": split,
                 "question": rows[row][0], "prompt": RT.prompt_text(rows[row][0]),
                 "reference": RT.fraction_str(refs[split][row])}
                for c, row in enumerate(idx)]

    all_refs = refs["train"] + refs["test"]
    return {
        "schema": SCHEMA,
        "source": source,
        "prompt_suffix": RT.PROMPT_SUFFIX,
        "candidate_seed": seed,
        "draw": "random.Random(candidate_seed).sample over train row indices, then over test row indices",
        "checks": {"reference_answers_parsed": len(all_refs),
                   "integer_reference_answers": sum(1 for r in all_refs if r.denominator == 1),
                   "train_test_shared_problem_texts": shared},
        "train_candidates": cands("train", train_rows, train_idx, "tr"),
        "test_candidates": cands("test", test_rows, test_idx, "te"),
    }


def dumps(obj: dict) -> bytes:
    """Return the canonical file bytes, with sorted keys, a one-space indent, UTF-8 and a final newline."""
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def load_dataset(path, expected_sha256: str = None) -> dict:
    """Read the frozen file, checking its SHA-256 first when a value is given."""
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("the dataset file's SHA-256 differs from the expected value")
    data = json.loads(raw.decode("utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError("the dataset file has the wrong schema")
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-parquet", type=Path, default=DATA_DIR / "train-00000-of-00001.parquet")
    ap.add_argument("--test-parquet", type=Path, default=DATA_DIR / "test-00000-of-00001.parquet")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="overwrite an existing file with different bytes")
    args = ap.parse_args(argv)
    try:
        train = load_split(args.train_parquet, SOURCE["files"]["train"])
        test = load_split(args.test_parquet, SOURCE["files"]["test"])
        payload = dumps(build(train, test))
    except (ValueError, OSError, parquet_lite.ParquetError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if args.out.exists() and args.out.read_bytes() != payload and not args.force:
        print("REFUSED: the output exists with different bytes, and the frozen file is never rewritten",
              file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    print(json.dumps({"out": str(args.out), "sha256": hashlib.sha256(payload).hexdigest(),
                      "bytes": len(payload), "train_candidates": RT.N_TRAIN_CANDIDATES,
                      "test_candidates": RT.N_TEST_CANDIDATES}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
