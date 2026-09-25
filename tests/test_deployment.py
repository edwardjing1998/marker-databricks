from copy import deepcopy
from pathlib import Path
import ast
import json
import pytest
import yaml
from scripts.deploy_databricks import job_settings, verify_cpu_job
from worker.gpu_launch import gpu_submit_body, command_text

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def deploy_env(monkeypatch):
    for key,val in {'SOURCE_VOLUME_PATH':'/Volumes/c/s/source','OUTPUT_VOLUME_PATH':'/Volumes/c/s/generated',
                    'STATE_VOLUME_PATH':'/Volumes/c/s/state','DATABRICKS_HOST':'https://a.azuredatabricks.net',
                    'DATABRICKS_CLIENT_ID':'00000000-0000-0000-0000-000000000001'}.items():
        monkeypatch.setenv(key,val)


def test_parent_is_cpu_python_file(deploy_env):
    settings=job_settings('/Workspace/Shared/marker-databricks/releases/test')
    task=settings['tasks'][0]
    assert 'spark_python_task' in task and 'notebook_task' not in task and 'compute' not in task
    assert task['environment_key']=='marker_cpu'
    assert settings['max_concurrent_runs']==1
    verify_cpu_job(settings,settings)


def test_gpu_setting_on_parent_rejected(deploy_env):
    settings=job_settings('/Workspace/Shared/marker-databricks/releases/test')
    altered=deepcopy(settings);altered['tasks'][0]['compute']={'hardware_accelerator':'GPU_1xA10'}
    with pytest.raises(RuntimeError):verify_cpu_job(altered,settings)


def test_gpu_child_uses_ai_runtime_command(cfg):
    body=gpu_submit_body(cfg,'/Workspace/Shared/request/run.sh')
    task=body['tasks'][0]
    assert 'ai_runtime_task' in task and 'notebook_task' not in task
    assert task['ai_runtime_task']['deployments'][0]['compute']['accelerator_count']==1
    assert len(body['idempotency_token'])==64
    assert 'code_source_path' not in task['ai_runtime_task']


def test_handoff_does_not_use_notebook_task_values(cfg):
    body=json.dumps(gpu_submit_body(cfg,'/Workspace/Shared/run.sh'))
    assert 'taskValues' not in body and '.values.' not in body


def test_gpu_manifest_quoted_and_archive_verified():
    text=command_text('/Volumes/c/s/state/runs/a/gpu-manifest.json','/Volumes/c/s/state/a.tgz','a'*64)
    assert 'hashlib.sha256' in text and 'jobs/gpu_worker.py --manifest' in text
    assert 'CUDA' not in text # device is selected in the Python engine, not guessed here


def test_no_notebook_files_shipped():
    assert not list(ROOT.rglob('*.ipynb'))
    for path in (ROOT/'jobs').glob('*.py'):
        assert '# Databricks notebook source' not in path.read_text()


def test_all_code_compiles():
    for folder in ('api','worker','shared','jobs','scripts'):
        for path in (ROOT/folder).glob('*.py'):ast.parse(path.read_text())


def test_workflows_parse():
    for path in (ROOT/'.github/workflows').glob('*.yml'):
        assert 'jobs' in yaml.safe_load(path.read_text())


def test_runtime_does_not_install_pyspark():
    for name in ('requirements-cpu.txt','requirements-gpu.txt'):
        lines=[s for s in (ROOT/name).read_text().splitlines() if s and not s.startswith('#')]
        assert not any(s.lower().startswith('pyspark') for s in lines)


def test_gpu_schema_with_real_sdk_when_installed(cfg):
    jobs=pytest.importorskip('databricks.sdk.service.jobs')
    if not hasattr(jobs,'AiRuntimeTask'):pytest.skip('Installed SDK predates AI Runtime task')
    body=gpu_submit_body(cfg,'/Workspace/Shared/run.sh')
    assert jobs.SubmitTask.from_dict(body['tasks'][0]).as_dict()==body['tasks'][0]
