"""The durable queue runs from an installed core package without an apps tree."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_installed_cli_dispatches_queue_without_apps_tree(tmp_path):
    package_root = tmp_path / "site-packages"
    source = Path(__file__).resolve().parents[1] / "biomaker"
    shutil.copytree(
        source,
        package_root / "biomaker",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    script = """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import biomaker.run as entrypoint
from biomaker.utils import queue_client
assert Path(entrypoint.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert not (Path(sys.argv[1]) / 'apps').exists()
def submit():
    assert sys.argv == ['biomaker', 'goal=Train a model', 'data_dir=dataset']
    print('installed queue dispatch completed')
    return 17
def direct():
    raise AssertionError('queued invocation reached the direct engine')
queue_client.submit_cli_and_wait = submit
entrypoint._run_direct_entry = direct
sys.argv = ['biomaker', '--queue', 'goal=Train a model', 'data_dir=dataset']
entrypoint.run()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(package_root)],
        cwd=unrelated,
        env={key: value for key, value in os.environ.items() if not key.startswith("BIOMAKER_")},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    assert "installed queue dispatch completed" in result.stdout


def test_installed_core_owns_detached_queue_lifecycle(tmp_path):
    package_root = tmp_path / "site-packages"
    source = Path(__file__).resolve().parents[1] / "biomaker"
    shutil.copytree(
        source,
        package_root / "biomaker",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    data_home = tmp_path / "data"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("BIOMAKER_", "UV_PROJECT_"))
    }
    environment.update(PYTHONPATH=str(package_root), XDG_DATA_HOME=str(data_home))
    worker = """import json,os,sys
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True)
(root/'run_manifest.json').write_text(json.dumps({'launch_token':os.environ['BIOMAKER_LAUNCH_TOKEN'],'data_dir':str(root.parent/'synthetic')}))
(root/'events.jsonl').write_text(json.dumps({'event':'run_finished'})+'\\n')
(root/'results.tsv').write_text('step\\tmetric\\tmaximize\\tstatus\\n1\\t0.75\\ttrue\\tsucceeded\\n')
print('isolated queue completed')
"""
    script = """import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from biomaker.application import RunService
from biomaker.runtime.settings import Settings
from biomaker.runtime.runner import RunManager, build_command
settings=Settings.default()
assert settings.method_root == Path(sys.argv[1])
assert settings.workspace_root == Path(sys.argv[2])/'biomaker'
assert settings.env_file == settings.workspace_root/'.env'
assert not (settings.method_root/'apps').exists()
assert build_command(settings,dataset_path='/synthetic',task_file=Path('/task'),requested_exp_name=None,options={})[:3] == [sys.executable,'-m','biomaker.run']
manager=RunManager(settings)
service=RunService(manager,settings)
try:
    manager.start()
    record=service.submit_command([sys.executable,'-c',sys.argv[3],str(settings.logs_root/'isolated')],cwd=Path.cwd())
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        current=service.get(record['id'])
        if current['status'] in {'succeeded','failed','interrupted'}:break
        time.sleep(.05)
    assert current['status']=='succeeded', current
    assert current['return_code']==0
    assert 'isolated queue completed' in Path(current['console_log']).read_text()
    assert current['log_dir']==str(settings.logs_root/'isolated')
    view=service.status(record['id'])
    assert view['status']=='complete'
    assert view['detailed_status']=='succeeded'
    assert view['allowed_actions']==['requeue']
    assert view['terminal_event']=='run_finished'
    assert service.list_runs()==[view]
    detail=service.detail(record['id'])
    assert {key:value for key,value in detail.items() if key not in {'observed','artifacts'}}==view
    assert detail['observed']['last_event']['event']=='run_finished'
    assert service.results(record['id'])==detail['observed']['results']
    assert service.results(record['id'])['best_metric']==0.75
    assert any(artifact['name']=='results.tsv' for artifact in detail['artifacts'])
    assert service.get(record['id'])==current
    while manager.store.get_queue_control().get('runner_pid') and time.monotonic()<deadline:time.sleep(.05)
    assert not manager.store.get_queue_control().get('runner_pid')
    assert not any(name=='webui' or name.startswith('webui.') for name in sys.modules)
finally:
    manager.shutdown()
    manager.store.close()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(package_root),
            str(data_home),
            worker,
        ],
        cwd=unrelated,
        env=environment,
        text=True,
        capture_output=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
