"""Regression checks for the requested removal and private deployment default."""
from pathlib import Path
import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from scripts.check_private_exposure import exposures, read_list, main

ROOT = Path(__file__).resolve().parents[1]


def workflow():
    return yaml.safe_load((ROOT/'.github/workflows/deploy.yml').read_text())


def test_no_client_key_lookup_in_runtime():
    for path in (ROOT/'api').glob('*.py'):
        assert 'MARKER_API_KEY' not in path.read_text()
    assert 'MARKER_API_KEY' not in (ROOT/'.env.example').read_text()


def test_workflow_has_no_api_key_input():
    w = workflow()
    env = w['jobs']['deploy']['env']
    assert 'MARKER_API_KEY' not in env
    text = (ROOT/'.github/workflows/deploy.yml').read_text()
    assert 'secrets.MARKER_API_KEY' not in text
    assert '--from-literal=MARKER_API_KEY' not in text
    assert '"MARKER_API_KEY":null' in text  # Deletes stale deployment data only.


def test_development_environment_and_required_secret_set():
    job = workflow()['jobs']['deploy']
    assert job['environment'] == 'development'
    for name in ('OPENSHIFT_TOKEN', 'DATABRICKS_CLIENT_ID', 'DATABRICKS_CLIENT_SECRET', 'GHCR_PULL_TOKEN'):
        assert 'secrets.'+name in job['env'][name]
    assert 'vars.GHCR_USERNAME' in job['env']['GHCR_USERNAME']


def test_no_route_or_public_service_manifest():
    items = list(yaml.safe_load_all((ROOT/'openshift/app.yml').read_text()))
    assert not any(x['kind'] in ('Route', 'Ingress') for x in items)
    svc = next(x for x in items if x['kind'] == 'Service')
    assert svc['spec']['type'] == 'ClusterIP'
    assert svc['spec']['externalIPs'] == []


def test_ingress_isolation_scoped_to_gateway():
    policy = yaml.safe_load((ROOT/'openshift/network-policy.yml').read_text())
    assert policy['kind'] == 'NetworkPolicy'
    assert policy['spec']['podSelector']['matchLabels'] == {'app': 'marker-databricks'}
    assert policy['spec']['policyTypes'] == ['Ingress']
    assert policy['spec']['ingress'] == []
    assert 'egress' not in policy['spec']


def test_old_route_removed_before_keyless_rollout():
    text = (ROOT/'.github/workflows/deploy.yml').read_text()
    assert text.index('oc delete route marker-databricks') < text.index('oc apply -f openshift/rendered.yml')
    assert text.index('oc apply -f openshift/network-policy.yml') < text.index('oc apply -f openshift/rendered.yml')
    assert 'oc get route marker-databricks' not in text


def test_health_uses_local_port_forward():
    text = (ROOT/'scripts/check_gateway_health.sh').read_text()
    assert '--address=127.0.0.1' in text
    assert 'trap cleanup EXIT' in text
    assert '/api/storage-documents/process' not in text
    assert '/health/ready' in text


def test_health_script_shell_syntax():
    subprocess.run(['bash', '-n', str(ROOT/'scripts/check_gateway_health.sh')], check=True)


def test_inline_workflow_shell_syntax():
    for step in workflow()['jobs']['deploy']['steps']:
        if 'run' in step:
            subprocess.run(['bash', '-n'], input=step['run'], text=True, check=True)


def test_no_matching_exposure():
    assert exposures({'items': []}, {'items': []}) == []


def test_owned_service_routes_found():
    routes={'items':[{'metadata':{'name':'alias'},'spec':{'to':{'kind':'Service','name':'marker-databricks'}}}]}
    assert exposures(routes, {}) == ['Route/alias']


def test_alternate_backend_found():
    routes={'items':[{'metadata':{'name':'alias'},'spec':{'to':{'name':'other'},'alternateBackends':[{'name':'marker-databricks'}]}}]}
    assert exposures(routes, {}) == ['Route/alias']


def test_unrelated_route_not_blocked():
    routes={'items':[{'metadata':{'name':'other'},'spec':{'to':{'name':'other'}}}]}
    assert exposures(routes, {}) == []


def test_ingress_rule_found():
    ingress={'items':[{'metadata':{'name':'public'},'spec':{'rules':[{'http':{'paths':[{'backend':{'service':{'name':'marker-databricks'}}}]}}]}}]}
    assert exposures({}, ingress) == ['Ingress/public']


def test_ingress_default_backend_found():
    ingress={'items':[{'metadata':{'name':'default'},'spec':{'defaultBackend':{'service':{'name':'marker-databricks'}}}}]}
    assert exposures({}, ingress) == ['Ingress/default']


def test_legacy_ingress_backend_found():
    ingress={'items':[{'metadata':{'name':'legacy'},'spec':{'backend':{'serviceName':'marker-databricks'}}}]}
    assert exposures({}, ingress) == ['Ingress/legacy']


def test_private_check_failure_blocks_deployment(monkeypatch):
    monkeypatch.setattr('scripts.check_private_exposure.read_list', lambda kind: {
        'items':[{'metadata':{'name':'public'},'spec':{'to':{'name':'marker-databricks'}}}]
    } if kind=='routes' else {'items':[]})
    with pytest.raises(SystemExit, match='Refusing keyless deployment'):
        main()


def test_no_read_permission_fails_closed(monkeypatch):
    monkeypatch.setattr('scripts.check_private_exposure.subprocess.run', lambda *a,**k: SimpleNamespace(returncode=1, stdout='', stderr='denied'))
    with pytest.raises(RuntimeError, match='requires permission'):
        read_list('routes')


def test_read_list_uses_safe_argument_array(monkeypatch):
    stub=Mock(return_value=SimpleNamespace(returncode=0, stdout='{"items":[]}'))
    monkeypatch.setattr('scripts.check_private_exposure.subprocess.run', stub)
    assert read_list('ingresses') == {'items':[]}
    assert stub.call_args.args[0] == ['oc','get','ingresses','-o','json']


def test_workflow_validation_accepts_missing_api_key(tmp_path):
    """Execute the real workflow validation block with dummy, non-secret values."""
    step=next(s for s in workflow()['jobs']['deploy']['steps'] if s.get('name')=='Validate deployment configuration')
    env={k:v for k,v in os.environ.items() if not k.startswith(('MARKER_', 'DATABRICKS_'))}
    env.update({
        'OPENSHIFT_SERVER':'https://cluster.example:6443', 'OPENSHIFT_NAMESPACE':'dev',
        'OPENSHIFT_TOKEN':'test-only', 'DATABRICKS_HOST':'https://test.azuredatabricks.net',
        'DATABRICKS_CLIENT_ID':'test-client', 'DATABRICKS_CLIENT_SECRET':'test-secret',
        'AZURE_STORAGE_ENDPOINT':'https://test.blob.core.windows.net',
        'SOURCE_VOLUME_PATH':'/Volumes/c/s/source', 'OUTPUT_VOLUME_PATH':'/Volumes/c/s/output',
        'STATE_VOLUME_PATH':'/Volumes/c/s/state', 'GHCR_USERNAME':'test-user', 'GHCR_PULL_TOKEN':'test-only',
    })
    result=subprocess.run(['bash','-c',step['run']], env=env, text=True, capture_output=True)
    assert result.returncode==0, result.stdout+result.stderr
