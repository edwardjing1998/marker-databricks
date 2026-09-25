"""OAuth-only authentication and migration regression tests. No live credentials."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from api.databricks_client import DatabricksAPI, RemoteError
from shared.oauth import oauth_credentials
from scripts.configure_gateway_secret import secret_manifest
from scripts.deploy_databricks import job_settings, verify_cpu_job, fixed_config
from worker.gpu_launch import runtime_api

ROOT = Path(__file__).resolve().parents[1]
CLIENT_ID = '00000000-0000-0000-0000-000000000001'
SECRET = 'unit-test-secret-not-a-real-credential'


@pytest.fixture
def oauth_env(monkeypatch):
    monkeypatch.setenv('DATABRICKS_HOST', 'https://example.azuredatabricks.net')
    monkeypatch.setenv('DATABRICKS_CLIENT_ID', CLIENT_ID)
    monkeypatch.setenv('DATABRICKS_CLIENT_SECRET', SECRET)
    monkeypatch.setenv('DATABRICKS_AUTH_TYPE', 'oauth-m2m')


@pytest.fixture
def sdk_stub(monkeypatch):
    module = ModuleType('databricks.sdk.core')
    config = Mock()
    config.authenticate.return_value = {'Authorization': 'Bearer test-access-token'}
    module.Config = Mock(return_value=config)
    monkeypatch.setitem(sys.modules, 'databricks.sdk.core', module)
    return module, config


def test_explicit_oauth_only_config(oauth_env, sdk_stub):
    module, _ = sdk_stub
    DatabricksAPI()
    assert module.Config.call_args.kwargs == {
        'host': 'https://example.azuredatabricks.net', 'client_id': CLIENT_ID,
        'client_secret': SECRET, 'auth_type': 'oauth-m2m', 'http_timeout_seconds': 30}


@pytest.mark.parametrize('name', ['DATABRICKS_HOST', 'DATABRICKS_CLIENT_ID', 'DATABRICKS_CLIENT_SECRET'])
def test_missing_config_fails_without_printing_secret(oauth_env, monkeypatch, name):
    monkeypatch.delenv(name)
    with pytest.raises(ValueError) as exc:
        oauth_credentials()
    assert name in str(exc.value) and SECRET not in str(exc.value)


def test_whitespace_secret_rejected(oauth_env, monkeypatch):
    monkeypatch.setenv('DATABRICKS_CLIENT_SECRET', '  ')
    with pytest.raises(ValueError, match='DATABRICKS_CLIENT_SECRET'):
        oauth_credentials()


def test_secret_not_in_repr(oauth_env):
    assert SECRET not in repr(oauth_credentials())


def test_stale_pat_cannot_override_oauth(oauth_env, monkeypatch):
    monkeypatch.setenv('DATABRICKS_TOKEN', 'unit-test-old-token')
    with pytest.raises(ValueError, match='unset DATABRICKS_TOKEN'):
        oauth_credentials()


def test_other_auth_mode_rejected(oauth_env, monkeypatch):
    monkeypatch.setenv('DATABRICKS_AUTH_TYPE', 'pat')
    with pytest.raises(ValueError, match='oauth-m2m'):
        oauth_credentials()


@pytest.mark.parametrize('host', [
    'http://example.azuredatabricks.net', 'https://example.azuredatabricks.net/api',
    'https://example.azuredatabricks.net?x=y', 'https://example.azuredatabricks.net#x',
    'https://user:secret@example.azuredatabricks.net', 'https://example.azuredatabricks.net:bad'])
def test_invalid_workspace_origins(oauth_env, host):
    with pytest.raises(ValueError):
        oauth_credentials(host=host)


def test_trailing_slash_normalized(oauth_env):
    assert oauth_credentials(host='https://example.azuredatabricks.net/').host.endswith('.net')


def test_sdk_initialization_error_is_sanitized(oauth_env, sdk_stub):
    module, _ = sdk_stub
    module.Config.side_effect = RuntimeError('secret=' + SECRET)
    with pytest.raises(RemoteError) as exc:
        DatabricksAPI()
    assert SECRET not in str(exc.value)


def test_auth_refresh_failure_sanitized(oauth_env, sdk_stub):
    _, cfg = sdk_stub
    api = DatabricksAPI()
    cfg.authenticate.side_effect = RuntimeError('secret=' + SECRET)
    with pytest.raises(RemoteError) as exc:
        api.call('GET', '/api/2.2/jobs/list')
    assert SECRET not in str(exc.value) and 'OAuth token acquisition' in str(exc.value)


def test_sdk_authenticate_called_on_each_request(oauth_env, sdk_stub, monkeypatch):
    _, cfg = sdk_stub
    cfg.authenticate.side_effect = [{'Authorization': 'Bearer first'}, {'Authorization': 'Bearer refreshed'}]
    response = Mock(status_code=200, ok=True)
    response.json.return_value = {'jobs': []}
    send = Mock(return_value=response)
    monkeypatch.setattr('api.databricks_client.requests.request', send)
    api = DatabricksAPI()
    api.call('GET', '/api/2.2/jobs/list')
    api.call('GET', '/api/2.2/jobs/list')
    assert send.call_args_list[0].kwargs['headers']['Authorization'] == 'Bearer first'
    assert send.call_args_list[1].kwargs['headers']['Authorization'] == 'Bearer refreshed'
    assert cfg.authenticate.call_count == 2


def test_error_body_not_exposed(oauth_env, sdk_stub, monkeypatch):
    response = Mock(status_code=401, ok=False, text=SECRET)
    monkeypatch.setattr('api.databricks_client.requests.request', Mock(return_value=response))
    with pytest.raises(RemoteError) as exc:
        DatabricksAPI().call('GET', '/api/2.2/jobs/list')
    assert SECRET not in str(exc.value) and 'HTTP 401' in str(exc.value)


def test_gateway_secret_contains_only_pair(oauth_env):
    manifest = secret_manifest()
    assert manifest['stringData'] == {'DATABRICKS_CLIENT_ID': CLIENT_ID, 'DATABRICKS_CLIENT_SECRET': SECRET}
    assert manifest['metadata']['name'] == 'marker-databricks-secret'


def test_secret_sent_over_stdin_not_command_line(oauth_env, monkeypatch, capsys):
    import scripts.configure_gateway_secret as module
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(module.subprocess, 'run', run)
    module.main()
    assert run.call_args.args[0] == ['oc', 'apply', '-f', '-']
    assert SECRET not in repr(run.call_args.args[0])
    assert SECRET in run.call_args.kwargs['input']
    assert SECRET not in capsys.readouterr().out


def test_secret_failure_does_not_echo_payload(oauth_env, monkeypatch):
    import scripts.configure_gateway_secret as module
    monkeypatch.setattr(module.subprocess, 'run', Mock(return_value=SimpleNamespace(returncode=1, stderr=SECRET)))
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert SECRET not in str(exc.value)


def test_gpu_launcher_reads_oauth_pair_only(cfg, monkeypatch):
    spark = Mock()
    def sql(query):
        return Mock(first=lambda: {'value': CLIENT_ID if 'gpu-launch-client-id' in query else SECRET})
    spark.sql.side_effect = sql
    factory = Mock()
    monkeypatch.setattr('worker.gpu_launch.DatabricksAPI', factory)
    runtime_api(spark, cfg)
    assert factory.call_args.kwargs == {'host': cfg.workspace_host, 'client_id': CLIENT_ID, 'client_secret': SECRET}
    assert spark.sql.call_count == 2
    assert all('gpu-launch-token' not in c.args[0] for c in spark.sql.call_args_list)


def test_gpu_launcher_pat_configuration_rejected(cfg):
    cfg.gpu_launch_auth = 'pat'
    with pytest.raises(ValueError, match='oauth-m2m'):
        cfg.validate(require_volumes=False)


@pytest.fixture
def job_env(oauth_env, monkeypatch):
    for key, value in {'SOURCE_VOLUME_PATH':'/Volumes/c/s/source',
                       'OUTPUT_VOLUME_PATH':'/Volumes/c/s/output',
                       'STATE_VOLUME_PATH':'/Volumes/c/s/state'}.items():
        monkeypatch.setenv(key, value)


def test_job_defaults_to_client_identity(job_env):
    settings = job_settings('/Workspace/Shared/marker-databricks/releases/test')
    assert settings['run_as'] == {'service_principal_name': CLIENT_ID}
    assert SECRET not in json.dumps(settings)


def test_run_as_override_requires_no_secret_in_job_config(job_env, monkeypatch):
    monkeypatch.setenv('DATABRICKS_RUN_AS_SERVICE_PRINCIPAL', 'another-principal-id')
    settings = job_settings('/Workspace/Shared/marker-databricks/releases/test')
    assert settings['run_as']['service_principal_name'] == 'another-principal-id'


def test_incorrect_saved_run_identity_rejected(job_env):
    settings = job_settings('/Workspace/Shared/marker-databricks/releases/test')
    actual = deepcopy(settings)
    actual['run_as']['service_principal_name'] = 'unexpected'
    with pytest.raises(RuntimeError, match='Run as'):
        verify_cpu_job(actual, settings)


def test_release_configuration_contains_no_credentials(job_env):
    cfg = fixed_config('/Workspace/Shared/marker-databricks/releases/test')
    assert cfg['GPU_LAUNCH_AUTH'] == 'oauth-m2m'
    assert SECRET not in json.dumps(cfg)
    assert not any(k in cfg for k in ['DATABRICKS_TOKEN', 'DATABRICKS_CLIENT_SECRET', 'DATABRICKS_CLIENT_ID'])


def test_workflow_never_references_pat_secret():
    text = (ROOT/'.github/workflows/deploy.yml').read_text()
    workflow = yaml.safe_load(text)['jobs']['deploy']
    assert 'secrets.DATABRICKS_TOKEN' not in text
    assert 'DATABRICKS_TOKEN' not in workflow['env']
    assert workflow['env']['DATABRICKS_AUTH_TYPE'] == 'oauth-m2m'
    assert workflow['env']['GPU_LAUNCH_AUTH'] == 'oauth-m2m'
    assert '"DATABRICKS_TOKEN":null' in text
    assert 'scripts/configure_gateway_secret.py' in text


def test_openshift_only_injects_explicit_oauth_secret_fields():
    docs = list(yaml.safe_load_all((ROOT/'openshift/app.yml').read_text()))
    dep = next(x for x in docs if x['kind'] == 'Deployment')
    container = dep['spec']['template']['spec']['containers'][0]
    assert not any('secretRef' in e for e in container['envFrom'])
    assert {e['name'] for e in container['env']} == {'DATABRICKS_CLIENT_ID', 'DATABRICKS_CLIENT_SECRET', 'DATABRICKS_AUTH_TYPE'}


@pytest.mark.parametrize('missing', ['DATABRICKS_CLIENT_ID', 'DATABRICKS_CLIENT_SECRET'])
def test_workflow_validation_requires_both_fields(missing):
    workflow = yaml.safe_load((ROOT/'.github/workflows/deploy.yml').read_text())
    step = next(x for x in workflow['jobs']['deploy']['steps'] if x.get('name') == 'Validate deployment configuration')
    env = {'PATH': os.environ['PATH']}
    for name in ('OPENSHIFT_SERVER','OPENSHIFT_NAMESPACE','OPENSHIFT_TOKEN','DATABRICKS_HOST',
                 'DATABRICKS_CLIENT_ID','DATABRICKS_CLIENT_SECRET','AZURE_STORAGE_ENDPOINT',
                 'SOURCE_VOLUME_PATH','OUTPUT_VOLUME_PATH','STATE_VOLUME_PATH','GHCR_USERNAME','GHCR_PULL_TOKEN'):
        env[name] = 'test-placeholder'
    del env[missing]
    result = subprocess.run(['bash','-c',step['run']], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and missing in result.stdout


def test_oauth_preflight_is_read_only(oauth_env, monkeypatch, capsys):
    import scripts.check_oauth as module
    api = Mock()
    monkeypatch.setattr(module, 'DatabricksAPI', lambda: api)
    module.main()
    assert api.call.call_args.args == ('GET', '/api/2.2/jobs/list')
    assert SECRET not in capsys.readouterr().out
