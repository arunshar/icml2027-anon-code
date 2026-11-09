# Anonymous code release

This archive holds the code behind the experiments of the paper under review. The paper studies how
reward design changes training on constrained generation tasks. The code covers the task definitions,
the reward terms and the group advantage, the trainer, the run analysis, the calibration scripts and the
task-quality audit. The archive holds the files of one anonymized git commit. The zip carries those files
and no git folder, so the commit itself is not inside the zip.

## What the archive says it contains

The paper states what ships in this appendix text, and this archive follows it.

> The supplement contains the run-analysis code. The archive includes the task-quality audit generator and summary records, but not the full set of table generators or raw training logs.

The paper also names the files below, and each one is in the archive.

- The `PROMPTS` lists of the speed task and the capacity task are in `speed_task.py` and `capacity_task.py`.
- The seeded script that generated the fixed capacity instances is `make_capacity_instances.py`.
- The calibration scripts that sample the base policy with no gradient are `calibrate_vmax.py` and
  `calibrate_capacity.py`. `calibrate_reasoning.py` does the same for the grade-school-math study.
- The task-quality audit generator is `make_revision_audit_tables.py`. It reads `audit_revision_checks.py`
  and writes its summary records into `revision_audit/`, where the shipped records are.
- The dated predictions file is `predictions_dated.md`.
- The commit and the training-source digest of the single-commit rerun of the capacity seeds are in
  `rerun_commit.json`. The commit appears as a label, as it does everywhere else in the archive.

## What is included

Everything sits at the top level of the archive.

- Task, reward and trainer code: `rewards.py`, `tasks.py`, `speed_task.py`, `capacity_task.py`,
  `speed_bounded_task.py`, `train_speed_grpo.py`, `make_capacity_instances.py`, `verify_real_groups.py`.
- Run-analysis code: `analyze.py`, `analyze_yoked.py`, `assemble_runs.py`, `compare_reruns.py`.
  `python analyze.py --help` lists its options.
- Manifest and replay code: `make_manifests.py`, `make_keep_schedules.py`, `make_bounded_manifest.py`,
  `replay_bounded_scale.py`.
- The strict-protocol extension: `train_revision.py`, `revision_task.py`, `revision_ppo.py`,
  `run_revision_cell.py`, `make_revision_manifest.py`, `analyze_revision.py`, `audit_revision_checks.py`,
  `audit_revision_tokens.py`.
- The grade-school-math study: `reasoning_task.py`, `train_reasoning_grpo.py`, `calibrate_reasoning.py`,
  `select_reasoning_pool.py`, `make_reasoning_dataset.py`, `parquet_lite.py`, `make_reasoning_manifest.py`,
  `analyze_reasoning.py`, `analyze_study2.py`, `study_control_truncation.py`.
- `configs/`: the shared configuration, the frozen datasets and the frozen manifests. The file
  `configs/reasoning_dataset.json` holds the text of public benchmark problems.
- `slurm/`: the run matrices as `.csv` files. The folder holds no job scripts.
- `tests/`: the test suite described below.
- `revision_audit/`: the sanitized summary records of the task-quality audit.
- `equivalence.json`: the fingerprint waivers that `analyze.py` reads.
- `rerun_commit.json`, `requirements.txt` and `predictions_dated.md`.

## What is not included

- Raw training logs and run folders. The analysis code reads a `runs/` folder that this archive does not
  contain.
- The full set of table generators. The generators for tables, macros and figures of the manuscript are
  not part of the archive, and neither is the manuscript. The matched-weight, below-threshold and fallback
  replays, the gate scorers that read the logged training groups, and the macro generators that turn
  their counts into the numbers of the paper belong to this group. `predictions_dated.md` names some of
  these scripts. A reader cannot rerun those replays from this archive.
- The cluster launch chain. The job scripts, the submission and transfer helpers, the compute budget
  gates and the bundle builders are withheld.
- The code of the second scorer of the grade-school-math study.
- The private work log and the raw predictions record. `predictions_dated.md` is the edited copy that
  replaces the raw record.
- Model weights and the downloaded benchmark files.

Some comments in the code name two files that are not here, `PREDICTIONS.md` and `LEDGER.md`. The first is
the raw record behind `predictions_dated.md`, and the second is the work log. A few files in the archive
also hold comments that cite a private provenance folder, which is not here. Those comments sit in files
whose digest the frozen configuration pins, so they were left as they are. Where the authors' commits
are named in fixtures and in `equivalence.json`, the archive writes the labels `COMMIT_1` to `COMMIT_6`
and keeps every other character of the file. A few files also name scheduler job numbers, which
are queue numbers of one cluster and name no person.

## Install

The tests ran on Python 3.12.13 with torch 2.14.0, transformers 5.17.0, numpy 2.5.3, pyyaml 6.0.3,
pytest 9.1.1, matplotlib 3.11.2 and huggingface_hub 1.31.0. `requirements.txt` lists these packages
without versions. The source files parse under the Python 3.9 grammar (checked with the `ast` module), but
no Python older than 3.12 was used to run the tests. No test is marked to need a GPU. The test run used a
laptop with an Apple GPU, and a machine without any GPU was not tried.

    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt

## Run the tests offline

Run the suite from the top level of the archive. The environment variable keeps the Hugging Face
libraries from reaching the network.

    HF_HUB_OFFLINE=1 python -m pytest -q tests

Add `-rs` to see why each skipped case skipped.

The tests load no model weights. They do load the tokenizer of one public model,
`Qwen/Qwen2.5-0.5B-Instruct`, and the trainers ask for it even in the tiny mode. When its files are not in
the Hugging Face cache and the network is off, `tests/conftest.py` turns the failed load into a skip with a
stated reason. To run those cases, fetch the tokenizer files once. Only this fetch needs the network, it downloads
about 11 MB, and it runs with `HF_HUB_OFFLINE` unset.

    python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-0.5B-Instruct', allow_patterns=['*.json', 'merges.txt'])"

## Tests that skip

A skip does not report a passing result. The cases that skip, with a stated reason, fall into these
groups.

- Cases that read the withheld `runs/` folders or the study records.
- Cases that need the manuscript directory, the table generators, the job scripts or the bundle builder.
- Cases that need the downloaded benchmark files, or a commit of the private history.
- Cases that need the tokenizer files described above.
- The golden-run cases in `tests/test_golden_tiny.py`. They compare against a run recorded on one stack,
  and they skip when the torch version, the transformers version or the machine differs from the recorded
  stack. The recorded values are in `tests/fixtures/golden_tiny.json`.

A reader cannot rerun the reported tables from this archive alone, because the raw logs are withheld. The
one analysis that runs end to end here is the task-quality audit. Point its outputs at scratch folders so
that the shipped records stay as they are.

    python make_revision_audit_tables.py --records revision_audit --out /tmp/tables --audit-out /tmp/audit

`analyze_study2.py` has no test in this archive. Its test reads the private predictions record and
names the cluster many times, so it is withheld. `python analyze_study2.py --help` still runs.

## Limits

The dates in `predictions_dated.md` cannot be checked. The archive holds the files of a single commit, so
it has no earlier history, and the file records the dates only as text. The file also names run records,
score reports, scripts and hashes that are not in the archive.

The edits made for anonymity reworded comments, docstrings and messages, replaced commit identifiers with
labels, changed a few constants and paths in tests, and added the skip guards and the fixture in
`tests/conftest.py`. They did not change the logic of any module. Two defaults changed place. The root that
`analyze.py` and `make_keep_schedules.py` use for repository-relative paths is now the folder of the archive,
and the tests look for the manuscript folder inside the archive. Because the commit identifiers are labels,
`compare_reruns.py` and the fingerprint waivers in `equivalence.json` cannot validate the original run logs,
whose provenance records hold the real commit identifiers. A release that includes those logs would need
the identifiers back.

`analyze.py` exempts the ten fields of `ALLOWED_CONFIG_DIFF` when it compares the configurations of runs in
one cohort. It also exempts `keep_schedule` in the filter-control cohorts and `fixedref_c` in the bounded
task cohorts. Each of these two fields sits in its own set in the code.

The digest over the training sources equals the value that the frozen configuration records, because
those files are unchanged. The digest over the grade-school-math sources differs from the recorded
value, because job scripts that the digest covers are withheld and comments in `reasoning_task.py`
were reworded. The grade-school-math trainer therefore refuses every launch of design version 2 from this
archive, by its own check. The shipped code is the latest state of the trainers. Runs made earlier used
earlier states of the training sources, which the archive does not hold.
