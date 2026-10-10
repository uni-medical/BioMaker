"""Explicit packages retain task/data identity across roots without a provider."""
import errno
import shutil
from pathlib import Path

import pytest

from test_task_split_constraints import split_package, TASK, write_json
from biomaker.utils.agentic_data_planner import _resolve_path
from biomaker.utils.split_protocol import prepare_split_protocol
from biomaker.utils.verified_split_package import seal_split_package, verify_package_identity


def seal_fixture(source, package):
    import json
    for name in ('private_gt/private_gt.json', 'split_manifest.json'):
        path = package / name
        data = json.loads(path.read_text())
        hidden = data['hidden_answers'] if name == 'split_manifest.json' else data
        for entry in hidden.values():
            entry['file_reference_columns'] = []
        write_json(path, data)
    seal_split_package(package, source, task_text=TASK)


def test_explicit_package_moves_between_roots_without_generation(split_package, tmp_path, monkeypatch):
    source, package = split_package
    seal_fixture(source, package)
    relocated = tmp_path / 'other-machine' / 'input'
    shutil.copytree(source, relocated)
    private = tmp_path / 'other-machine/private_gt'
    monkeypatch.setattr('biomaker.utils.split_protocol.run_agentic_split_builder_probe',
                        lambda *a, **k: pytest.fail('Verified inputs must not invoke a provider'))
    result = prepare_split_protocol(relocated, private, task_desc=TASK, package_dir=package)
    assert result.split_plan['strategy'] == 'verified_package'
    assert result.split_plan['row_counts']['agent_train'] == 2
    assert (private / 'score_val.csv').is_file()
    assert not (relocated / 'private_gt').exists()
    assert (relocated / 'train/a.bin').read_text() == 'train/a.bin'
    assert verify_package_identity(package, source, task_text=TASK)['version'] == 1


@pytest.mark.parametrize('change', ['task', 'source', 'package', 'extra_file', 'missing_file'])
def test_changed_identity_fails_before_output_mutation(split_package, tmp_path, change):
    source, package = split_package
    seal_fixture(source, package)
    task = TASK
    if change == 'task':
        task += '\nDifferent task'
    elif change == 'source':
        (source / 'train/a.bin').write_text('changed')
    elif change == 'package':
        (package / 'private_gt/test.csv').write_text('changed')
    elif change == 'extra_file':
        (package / 'extra.csv').write_text('added')
    else:
        (source / 'train/a.bin').unlink()
    private = tmp_path / 'existing-private'
    private.mkdir()
    (private / 'marker').write_text('preserved')
    with pytest.raises(ValueError, match='identity|inventory/hash'):
        prepare_split_protocol(source, private, task_desc=task, package_dir=package)
    assert (private / 'marker').read_text() == 'preserved'


def test_current_verifier_is_required_after_hash_verification(split_package, tmp_path, monkeypatch):
    source, package = split_package
    seal_fixture(source, package)
    monkeypatch.setattr('biomaker.utils.split_protocol.verify_split_package',
                        lambda *a, **k: {'ok': False, 'errors': ['new contract restriction']})
    with pytest.raises(ValueError, match='new contract restriction'):
        prepare_split_protocol(source, tmp_path / 'run/private', task_desc=TASK, package_dir=package)


def test_absolute_declared_file_paths_are_not_portable(split_package):
    source, package = split_package
    seal_fixture(source, package)
    path = package / 'biomaker_splits/score_val.csv'
    path.write_text(path.read_text().replace('input/train/c.bin', str(source / 'train/c.bin')))
    with pytest.raises(ValueError, match='logical input'):
        seal_split_package(package, source, task_text=TASK)


def test_inline_box_payload_is_not_a_path(tmp_path):
    assert _resolve_path('[' + '1,2,3,4,' * 1000 + ']', tmp_path, tmp_path) is None


def test_path_permission_errors_are_not_hidden(tmp_path, monkeypatch):
    def denied(_):
        raise OSError(errno.EACCES, 'permission denied')
    monkeypatch.setattr(Path, 'exists', denied)
    with pytest.raises(OSError, match='permission denied'):
        _resolve_path('image.png', tmp_path, tmp_path)
