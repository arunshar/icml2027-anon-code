"""Golden CPU fixtures for the trainer's default path, recorded before the remedy flags existed.

The remedy arms add flags to the trainer, the reward module and the task modules. Every one of
those flags has a default, and the default must leave a run exactly what it was. The fixtures in
`fixtures/golden_tiny.json` were written by this file's `--record` mode at the commit before any
of that code, on the random two-layer model the other tiny tests use. Two runs are pinned, each for three steps with a small batch:

  speed     the added-verdict arm on the hard speed limit, one backward call per step
  capacity  the added-verdict arm on the hard capacity level, one backward call per group

This test reruns both with the same flags and compares every key the fixture holds. Keys a later
commit adds to the logs are ignored, because adding a key is allowed and changing a value is not.
Wall-clock fields are never compared.

Floating-point results from a CPU forward pass depend on the library versions and on the machine
architecture, so the comparison skips, and says why, when either differs from the recording.

To re-record (only ever at a commit whose trainer is known to be the reference):
    python tests/test_golden_tiny.py --record
"""
from __future__ import annotations

import json
import os
import platform
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

FIXTURE = os.path.join(HERE, "tests", "fixtures", "golden_tiny.json")

# Fields that measure time or name the machine, so they differ between two identical runs.
VOLATILE_RECORD_KEYS = {"s_per_step"}
VOLATILE_CONFIG_KEYS = {"out_dir"}

CASES = {
    # The random model rarely emits a digit, so this case uses every prompt, a longer generation
    # and a seed at which one group parses. Without that the pinned update would be zero.
    "speed": ["--arm", "binary", "--w", "5", "--vmax", "5", "--steps", "3", "--tiny",
              "--device", "cpu", "--eval-every", "2", "--eval-samples", "2",
              "--batch-prompts", "6", "--seed", "3", "--max-new-tokens", "24"],
    "capacity": ["--task", "capacity", "--arm", "binary", "--w", "5", "--vmax", "0.03",
                 "--steps", "3", "--tiny", "--device", "cpu", "--eval-every", "2",
                 "--eval-samples", "2", "--batch-prompts", "2", "--seed", "0",
                 "--backward-per-group", "--max-new-tokens", "24"],
}


def _versions():
    import numpy
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": numpy.__version__, "python": platform.python_version(),
            "machine": platform.machine()}


def _run(argv, out_dir):
    import train_speed_grpo as T
    rc = T.main(list(argv) + ["--out-dir", str(out_dir)])
    assert rc == 0
    recs = [json.loads(line) for line in open(os.path.join(out_dir, "metrics.jsonl"))]
    for r in recs:
        for k in VOLATILE_RECORD_KEYS:
            r.pop(k, None)
    cfg = json.load(open(os.path.join(out_dir, "config.json")))
    for k in VOLATILE_CONFIG_KEYS:
        cfg.pop(k, None)
    prov = json.load(open(os.path.join(out_dir, "provenance.json")))
    return {"config": cfg, "records": recs, "final_param_checksum": prov["final_param_checksum"],
            "git_sha": prov["git_sha"]}


def _subset_equal(want, got, path="$"):
    """Every key in `want` must be present in `got` with an equal value. Extra keys are allowed."""
    if isinstance(want, dict):
        assert isinstance(got, dict), f"{path}: expected an object, got {type(got).__name__}"
        for k, v in want.items():
            assert k in got, f"{path}.{k} is missing from the rerun"
            _subset_equal(v, got[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), \
            f"{path}: length {len(got) if isinstance(got, list) else '?'} != {len(want)}"
        for i, (a, b) in enumerate(zip(want, got)):
            _subset_equal(a, b, f"{path}[{i}]")
    else:
        assert want == got, f"{path}: {got!r} != recorded {want!r}"


def _load():
    if not os.path.exists(FIXTURE):
        pytest.skip("the golden fixture is not present in this copy of the code")
    fx = json.load(open(FIXTURE))
    now = _versions()
    for k in ("torch", "transformers", "machine"):
        if fx["versions"][k] != now[k]:
            pytest.skip(f"recorded with {k} {fx['versions'][k]}, running {now[k]}; a CPU forward "
                        f"pass is only bit-reproducible on the recorded stack")
    return fx


@pytest.mark.parametrize("case", sorted(CASES))
def test_default_flags_reproduce_the_golden_run(case, tmp_path):
    fx = _load()
    rec = fx["cases"][case]
    assert rec["argv"] == CASES[case], "the fixture was recorded with different flags"
    got = _run(CASES[case], tmp_path / case)
    _subset_equal(rec["config"], got["config"], "$.config")
    _subset_equal(rec["records"], got["records"], "$.records")
    assert got["final_param_checksum"] == rec["final_param_checksum"], \
        "the parameters after three default steps differ from the recording"


def test_the_fixture_is_not_vacuous():
    """A fixture whose runs never moved a parameter, or never produced a reward spread, would pass
    against a trainer that ignores the advantage entirely."""
    if not os.path.exists(FIXTURE):
        pytest.skip("the golden fixture is not present in this copy of the code")
    fx = json.load(open(FIXTURE))
    for case, rec in fx["cases"].items():
        train = [r for r in rec["records"] if r["phase"] == "train"]
        assert [r["step"] for r in train] == [1, 2, 3], case
        spread = [g for r in train for g in r["groups"] if max(g["rewards"]) > min(g["rewards"])]
        assert spread, f"{case}: no logged group has reward spread, so no update was pinned"


def record():  # pragma: no cover  (run by hand at the reference commit)
    import subprocess
    import tempfile
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    out = {"recorded_at_commit": sha, "versions": _versions(), "cases": {}}
    with tempfile.TemporaryDirectory() as td:
        for case, argv in CASES.items():
            got = _run(argv, os.path.join(td, case))
            got.pop("git_sha")
            out["cases"][case] = {"argv": argv, **got}
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {FIXTURE} at {sha[:8]}")


if __name__ == "__main__":  # pragma: no cover
    if "--record" in sys.argv:
        record()
    else:
        print(__doc__)
