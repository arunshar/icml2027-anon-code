#!/usr/bin/env python3
"""Run one frozen extension cell and preserve its launch record.

This wrapper never submits jobs. A scheduler or a local validation caller supplies
one manifest row. It refuses an existing output and validates the frozen source
snapshot before it starts the historical or strict trainer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


LEGACY_KEYS = {
    'task', 'arm', 'w', 'w_task', 'vmax', 'regime', 'seed', 'adv_variant',
    'group_filter', 'kl_beta', 'steps', 'group_size', 'batch_prompts', 'eval_every',
    'eval_samples', 'eval_seed', 'lr', 'max_new_tokens', 'temperature', 'top_p',
    'top_k', 'repetition_penalty', 'weight_decay', 'clip_coef', 'grad_clip',
    'model_id', 'backward_per_group', 'no_ref', 'tiny',
}
STRICT_KEYS = {
    'method', 'reward', 'weight', 'reward_scale', 'seed', 'group_size', 'steps',
    'epochs', 'eval_every', 'eval_samples', 'eval_seed', 'score_microbatch',
    'max_new_tokens', 'lr', 'critic_lr', 'critic_steps', 'grad_clip',
    'weight_decay', 'clip_coef', 'model_id', 'no_ref', 'tiny', 'skip_heldout',
}


REQUIRED_SOURCES = {
    'run_revision_cell.py', 'train_speed_grpo.py', 'train_revision.py',
    'revision_task.py', 'revision_ppo.py', 'make_revision_manifest.py',
    'rewards.py', 'tasks.py', 'capacity_task.py', 'speed_task.py',
    'make_capacity_instances.py', 'configs/revision_manifest.json',
    'configs/revision_dataset.json',
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cell(manifest: dict, row: int) -> dict:
    cells = manifest['new_runs']
    rows = [c['row'] for c in cells]
    ids = [c['run_id'] for c in cells]
    if len(set(rows)) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError('The manifest contains duplicate rows or run IDs.')
    matches = [c for c in cells if c['row'] == row]
    if len(matches) != 1:
        raise ValueError('The requested manifest row does not exist exactly once.')
    cell = matches[0]
    if not re.fullmatch(r'[A-Za-z0-9_]+', cell['run_id']):
        raise ValueError('The run ID is not a safe directory name.')
    if cell['protocol'] not in {'legacy', 'strict'}:
        raise ValueError('The experiment protocol is unknown.')
    cfg = cell['config']
    if cfg.get('skip_heldout', False):
        raise ValueError('Final study cells cannot omit held-out evaluation.')
    if cfg.get('tiny', False):
        raise ValueError('Final study cells cannot use a tiny test model.')
    if cfg['steps'] * cfg['group_size'] * 6 != cell['training_completions']:
        raise ValueError('The declared training sample budget does not match the cell.')
    if cell['training_completions'] != 7200:
        raise ValueError('A final study cell must use the frozen 7200-sample budget.')
    return cell


def make_command(cell: dict, exp: Path, dataset: Path, out: Path,
                 validation_steps: int | None = None) -> list[str]:
    cfg = dict(cell['config'])
    protocol = cell['protocol']
    allowed = LEGACY_KEYS if protocol == 'legacy' else STRICT_KEYS
    if set(cfg) - allowed:
        raise ValueError(f'Unsupported configuration keys: {sorted(set(cfg) - allowed)}')
    if validation_steps is not None:
        if validation_steps < 1 or validation_steps >= cfg['steps']:
            raise ValueError('Validation must be shorter than the declared final run.')
        cfg['steps'] = validation_steps
        cfg['eval_every'] = validation_steps
    trainer = 'train_speed_grpo.py' if protocol == 'legacy' else 'train_revision.py'
    command = [sys.executable, str(exp / trainer)]
    for key, value in cfg.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    command.extend(['--device', 'cuda'])
    if protocol == 'legacy':
        command.extend(['--out-dir', str(out)])
    else:
        command.extend(['--dataset', str(dataset), '--out', str(out)])
        if validation_steps is not None:
            command.append('--skip-heldout')
    return command


def verify_snapshot(exp: Path, snapshot_path: Path, *, required=None) -> dict:
    snapshot = json.loads(snapshot_path.read_text())
    entries = snapshot['files']
    seen = set()
    for item in entries:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('The snapshot contains an unsafe file path.')
        if item['path'] in seen:
            raise ValueError('The snapshot contains duplicate file paths.')
        seen.add(item['path'])
    required = REQUIRED_SOURCES if required is None else set(required)
    if required - seen:
        raise ValueError(f'The snapshot omits required sources: {sorted(required - seen)}')
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True,
                                       separators=(',', ':')).encode()).hexdigest()
    if digest != snapshot['source_sha256']:
        raise ValueError('The snapshot aggregate digest does not match its file list.')
    for item in entries:
        rel = Path(item['path'])
        path = exp / rel
        if not path.is_file() or path.is_symlink() or file_hash(path) != item['sha256']:
            raise ValueError(f'The frozen snapshot does not match {rel}.')
    return snapshot


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--row', type=int, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--validation-steps', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    exp = Path(__file__).resolve().parent
    snapshot = verify_snapshot(exp, args.snapshot)
    frozen = {item['path']: item['sha256'] for item in snapshot['files']}
    for path, name in ((args.manifest, 'configs/revision_manifest.json'),
                       (args.dataset, 'configs/revision_dataset.json')):
        if name not in frozen or file_hash(path) != frozen[name]:
            raise ValueError(f'The supplied {name} is outside the frozen snapshot.')
    from revision_task import load_manifest
    from make_revision_manifest import validate_manifest
    dataset = load_manifest(args.dataset)
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, dataset)
    cell = select_cell(manifest, args.row)
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    if not re.fullmatch(r'[A-Za-z0-9_]+', job_id):
        raise ValueError('The scheduler job ID is not a safe directory component.')
    phase = 'validation' if args.validation_steps is not None else 'study'
    out = args.out_root.resolve() / phase / cell['run_id'] / ('job_' + job_id)
    command = make_command(cell, exp, args.dataset.resolve(), out, args.validation_steps)
    if args.dry_run:
        print(json.dumps({'phase': phase, 'cell': cell, 'out': str(out), 'command': command}, indent=2))
        return 0
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Real model training must run inside a scheduled compute allocation.')
    if out.exists():
        raise FileExistsError(f'The wrapper will not overwrite an earlier attempt at {out}.')
    import torch
    if not torch.cuda.is_available() or 'a100' not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError('The study requires the matched A100 GPU type.')
    out.parent.mkdir(parents=True, exist_ok=True)
    record_path = out.parent / ('launch_' + job_id + '.json')
    record = {
        'run_id': cell['run_id'], 'row': args.row, 'phase': phase,
        'slurm_job_id': job_id, 'command': command,
        'manifest_file_sha256': file_hash(args.manifest),
        'dataset_file_sha256': file_hash(args.dataset),
        'snapshot_file_sha256': file_hash(args.snapshot),
        'snapshot': snapshot, 'cell': cell,
        'started_unix': time.time(),
    }
    with record_path.open('x') as handle:
        json.dump(record, handle, sort_keys=True, indent=2)
    result = subprocess.run(command, cwd=exp, check=False)
    record['exit_code'] = result.returncode
    record['ended_unix'] = time.time()
    record['done_marker_exists'] = (out / 'DONE').is_file()
    temp = record_path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    temp.replace(record_path)
    if result.returncode == 0 and not record['done_marker_exists']:
        raise RuntimeError('The process returned zero but did not finish its recorded run.')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
