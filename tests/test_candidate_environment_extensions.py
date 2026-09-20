from unittest.mock import patch

import pytest

from biomaker.utils.environment import validate_code_environment


@pytest.mark.parametrize('module', ['einops', 'tqdm', 'networkx', 'h5py', 'tensorboard', 'cc3d', 'torchio'])
def test_extended_candidate_imports_pass_when_installed(module):
    with patch('biomaker.utils.environment.importlib.util.find_spec', return_value=object()):
        assert validate_code_environment(f'import {module}').ok


@pytest.mark.parametrize('module', ['einops', 'cc3d', 'torchio'])
def test_missing_extension_is_reported_as_missing_not_prohibited(module):
    with patch('biomaker.utils.environment.importlib.util.find_spec', return_value=None):
        result = validate_code_environment(f'import {module}')
    assert not result.ok
    assert result.exc_info['reason'] == 'allowed package not installed'
