import hashlib
import json
from pathlib import Path

import pytest
from run_revision_cell import make_command, select_cell, verify_snapshot


def cell(protocol='strict'):
    return {'row': 1, 'run_id': 'strict_grpo_binary_w25_k4_s0', 'protocol': protocol,
            'training_completions': 7200,
            'config': {'steps': 300, 'group_size': 4, 'seed': 0}}


def test_duplicate_and_unmatched_sample_budgets_are_refused():
    one = cell()
    assert select_cell({'new_runs': [one]}, 1) == one
    with pytest.raises(ValueError, match='duplicate'):
        select_cell({'new_runs': [one, one]}, 1)
    one['config']['steps'] = 299
    with pytest.raises(ValueError, match='sample budget'):
        select_cell({'new_runs': [one]}, 1)


def test_validation_is_separate_and_never_samples_heldout():
    one = cell()
    command = make_command(one, Path('/exp'), Path('/dataset'), Path('/out'), 20)
    assert command[command.index('--steps') + 1] == '20'
    assert '--skip-heldout' in command
    assert one['config']['steps'] == 300
    assert '--skip-heldout' not in make_command(one, Path('/exp'), Path('/dataset'), Path('/out'))
    with pytest.raises(ValueError, match='shorter'):
        make_command(one, Path('/exp'), Path('/dataset'), Path('/out'), 300)


def test_legacy_trainer_keeps_its_original_cli():
    command = make_command(cell('legacy'), Path('/exp'), Path('/dataset'), Path('/out'), 20)
    assert command[1] == '/exp/train_speed_grpo.py'
    assert '--out-dir' in command
    assert '--dataset' not in command
    assert '--skip-heldout' not in command


def test_changed_sources_and_unsafe_paths_are_refused(tmp_path):
    source = tmp_path / 'train.py'
    source.write_text('original')
    snap = {'files': [{'path': 'train.py', 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]}
    snap['source_sha256'] = hashlib.sha256(json.dumps(snap['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path = tmp_path / 'snapshot.json'
    path.write_text(json.dumps(snap))
    assert verify_snapshot(tmp_path, path, required={'train.py'}) == snap
    source.write_text('changed')
    with pytest.raises(ValueError, match='does not match'):
        verify_snapshot(tmp_path, path, required={'train.py'})
    snap['files'][0]['path'] = '../train.py'
    path.write_text(json.dumps(snap))
    with pytest.raises(ValueError, match='unsafe'):
        verify_snapshot(tmp_path, path, required={'train.py'})


def test_unknown_cli_and_unsafe_run_ids_are_refused():
    one = cell()
    one['config']['shell_command'] = 'unused'
    with pytest.raises(ValueError, match='Unsupported'):
        make_command(one, Path('/exp'), Path('/dataset'), Path('/out'))
    one['run_id'] = '../other-session'
    with pytest.raises(ValueError, match='safe directory'):
        select_cell({'new_runs': [one]}, 1)


def test_tiny_models_cannot_enter_the_final_study():
    one = cell()
    one['config']['tiny'] = True
    with pytest.raises(ValueError, match='tiny test model'):
        select_cell({'new_runs': [one]}, 1)


def test_final_study_cannot_skip_heldout_evaluation():
    one = cell()
    one['config']['skip_heldout'] = True
    with pytest.raises(ValueError, match='omit held-out'):
        select_cell({'new_runs': [one]}, 1)


def test_incomplete_duplicate_and_false_aggregate_snapshots_are_refused(tmp_path):
    source = tmp_path / 'train.py'
    source.write_text('original')
    item = {'path': 'train.py', 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    snap = {'files': [item]}
    snap['source_sha256'] = hashlib.sha256(json.dumps(snap['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path = tmp_path / 'snapshot.json'
    path.write_text(json.dumps(snap))
    with pytest.raises(ValueError, match='omits required'):
        verify_snapshot(tmp_path, path)
    snap['files'].append(item)
    path.write_text(json.dumps(snap))
    with pytest.raises(ValueError, match='duplicate'):
        verify_snapshot(tmp_path, path, required={'train.py'})
    snap['files'] = [item]
    snap['source_sha256'] = '0' * 64
    path.write_text(json.dumps(snap))
    with pytest.raises(ValueError, match='aggregate digest'):
        verify_snapshot(tmp_path, path, required={'train.py'})
