#!/usr/bin/env python3
"""Build one analysis directory out of runs that live in several array directories.

The remedy tables put new arms beside baselines from an earlier array, and the five-seed capacity
table joins two arrays. analyze.py reads one directory of runs, so this makes that directory out of
relative symbolic links and copies nothing.

It refuses, and writes nothing, when:
  * a matched run directory has a config.json and no DONE marker, because an unfinished run must
    not enter a table by accident;
  * two runs share a task, level, arm and seed, because a table would then silently average or
    silently pick one of them;
  * two links would have the same name, or a source pattern matches no run;
  * the output directory already holds anything.

Usage:
  python assemble_runs.py --out runs/capacity5 --src runs/572991 --src runs/<array>
  python assemble_runs.py --out runs/speed_remedy --src 'runs/430542:*_A_*' --src runs/<array>

A source is a directory, optionally followed by a colon and a shell pattern on run directory names.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

import analyze


def parse_source(text: str):
    root, sep, pattern = text.rpartition(":")
    if not sep or "/" in pattern:
        return Path(text), "*"
    return Path(root), (pattern or "*")


def run_identity(cfg: dict):
    cfg = dict(cfg)
    for k, v in analyze.LEGACY_CONFIG_DEFAULTS.items():
        cfg.setdefault(k, v)
    return (cfg["task"], cfg.get("regime", ""), analyze.arm_name(cfg), int(cfg["seed"]))


def collect(sources):
    """The runs to link, and every reason to refuse. Reads only."""
    entries, problems = [], []
    by_identity, by_link = {}, {}
    for root, pattern in sources:
        if not root.is_dir():
            problems.append(f"{root}: not a directory")
            continue
        matched = 0
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if not fnmatch.fnmatchcase(d.name, pattern) or not (d / "config.json").exists():
                continue            # analysis/, logs/ and anything else that is not a run
            matched += 1
            where = f"{root.name}/{d.name}"
            if not (d / "DONE").exists():
                problems.append(f"{where}: no DONE marker")
                continue
            ident = run_identity(json.loads((d / "config.json").read_text()))
            link = f"{root.name}_{d.name}"
            if ident in by_identity:
                problems.append(f"duplicate run {ident}: {by_identity[ident]} and {where}")
                continue
            if link in by_link:
                problems.append(f"duplicate link name {link}: {by_link[link]} and {where}")
                continue
            by_identity[ident], by_link[link] = where, where
            entries.append({"link": link, "target": d, "task": ident[0], "regime": ident[1],
                            "arm": ident[2], "seed": ident[3]})
        if matched == 0:
            problems.append(f"{root}:{pattern} matched no run directory")
    return entries, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print(f"REFUSED: {out} already exists and is not an empty directory", file=sys.stderr)
        return 3
    sources = [parse_source(s) for s in a.src]
    entries, problems = collect(sources)
    if problems:
        print("REFUSED: nothing was written.", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 3
    out.mkdir(parents=True, exist_ok=True)
    real_out = out.resolve()
    for e in entries:
        os.symlink(os.path.relpath(e["target"].resolve(), real_out), out / e["link"],
                   target_is_directory=True)
    record = {
        "sources": [{"root": os.path.relpath(r.resolve(), real_out), "pattern": p} for r, p in sources],
        "runs": [{"link": e["link"], "target": os.path.relpath(e["target"].resolve(), real_out),
                  "task": e["task"], "regime": e["regime"], "arm": e["arm"], "seed": e["seed"]}
                 for e in entries],
    }
    (out / "assembled.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"assembled": str(out.name), "runs": len(entries)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
