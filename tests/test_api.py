from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
import api.main as main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv('MARKER_API_KEY', raising=False)
    monkeypatch.setenv('GPU_FALLBACK_ENABLED', 'false')
    monkeypatch.setenv('DATABRICKS_JOB_ID', '123')
    monkeypatch.setenv('AZURE_STORAGE_SOURCE_PREFIX', 'source/')
    monkeypatch.setenv('AZURE_STORAGE_OUTPUT_PREFIX', 'generated/')
    remote = Mock()
    remote.start.return_value = {'run_id': 456}
    remote.run.return_value = {'job_id': 123, 'state': {'life_cycle_state': 'RUNNING'}}
    monkeypatch.setattr(main, 'DatabricksAPI', lambda: remote)
    with TestClient(main.app) as test_client:
        yield test_client, remote


def test_health_no_remote_call(client):
    c, remote = client
    assert c.get('/health/ready').status_code == 200
    remote.start.assert_not_called()
    remote.run.assert_not_called()


def test_processing_without_client_key(client):
    c, remote = client
    assert c.post('/api/storage-documents/process?dryRun=true').status_code == 202
    remote.start.assert_called_once()


def test_accept_and_parameters(client):
    c, remote = client
    response = c.post('/api/storage-documents/process?sourcePrefix=source/book/&dryRun=true')
    assert response.status_code == 202
    assert response.json()['runId'] == 456
    assert remote.start.call_args.args[1]['source_prefix'] == 'source/book/'
    assert remote.start.call_args.args[1]['dry_run'] == 'true'


def test_same_key_payload_same_token(client):
    c, remote = client
    headers = {'Idempotency-Key': 'same-request'}
    one = c.post('/api/storage-documents/process', headers=headers).json()
    two = c.post('/api/storage-documents/process', headers=headers).json()
    assert one['requestId'] == two['requestId']
    assert len(remote.start.call_args.args[2]) == 64


def test_prefix_restriction(client):
    c, remote = client
    assert c.post('/api/storage-documents/process?sourcePrefix=private/').status_code == 400
    assert c.post('/api/storage-documents/process?outputPrefix=source/').status_code == 400
    remote.start.assert_not_called()


def test_replace_needs_overwrite(client):
    c, remote = client
    assert c.post('/api/storage-documents/process?replaceExisting=true').status_code == 400


def test_run_must_belong_to_app(client):
    c, remote = client
    remote.run.return_value = {'job_id': 999, 'state': {}}
    assert c.get('/api/storage-documents/runs/456').status_code == 404


def test_completed_run_summary(client):
    c, remote = client
    remote.run.return_value = {'job_id': 123, 'state': {'life_cycle_state': 'TERMINATED', 'result_state': 'SUCCESS'},
                              'tasks': [{'task_key': 'process_documents_cpu', 'run_id': 457, 'state': {'result_state': 'SUCCESS'}}]}
    remote.output.return_value = {'logs': 'MARKER_SUMMARY={"succeeded": 1, "failed": 0}'}
    result = c.get('/api/storage-documents/runs/456').json()
    assert result['summary']['succeeded'] == 1
    remote.output.assert_called_once_with(457)


def test_local_gpu_parameters(client):
    c, remote = client
    response = c.post('/api/storage-documents/process?pagesPerChunk=2&forceOcr=true&dropHandwriting=true')
    assert response.status_code == 202
    params = remote.start.call_args.args[1]
    assert params['pages_per_chunk'] == '2' and params['force_ocr'] == 'true' and params['drop_handwriting'] == 'true'
    assert 'mode' not in params


def test_hosted_mode_rejected_not_silently_ignored(client):
    c, remote = client
    response = c.post('/api/storage-documents/process?mode=balanced')
    assert response.status_code == 400
    remote.start.assert_not_called()


def test_invalid_chunk_size_rejected(client):
    c, remote = client
    assert c.post('/api/storage-documents/process?pagesPerChunk=0').status_code == 422
    remote.start.assert_not_called()


def test_compute_mode_forwarded(client):
    c, remote=client
    response=c.post('/api/storage-documents/process?computeMode=cpu')
    assert response.status_code==202
    assert remote.start.call_args.args[1]['compute_mode']=='cpu'


def test_invalid_compute_mode(client):
    c,remote=client
    assert c.post('/api/storage-documents/process?computeMode=magic').status_code==422


def test_explicit_gpu_blocked_before_start_when_policy_disabled(client):
    c, remote = client
    main.app.state.gpu_enabled = False
    response = c.post('/api/storage-documents/process?computeMode=gpu')
    assert response.status_code == 400
    remote.start.assert_not_called()


def test_explicit_gpu_forwarded_when_enabled(client):
    c, remote = client
    main.app.state.gpu_enabled = True
    response = c.post('/api/storage-documents/process?computeMode=gpu')
    assert response.status_code == 202
    assert remote.start.call_args.args[1]['compute_mode'] == 'gpu'


def test_status_without_client_key(client):
    c, remote = client
    response = c.get('/api/storage-documents/runs/456')
    assert response.status_code == 200
    remote.run.assert_called_once_with(456)


def test_openapi_has_no_api_key_security(client):
    c, _ = client
    schema = c.get('/openapi.json').json()
    assert not schema.get('components', {}).get('securitySchemes')
    for path in schema['paths'].values():
        for operation in path.values():
            assert not operation.get('security')
            assert not any(p.get('name', '').lower() == 'x-api-key' for p in operation.get('parameters', []))


def test_cross_origin_browser_submission_rejected(client):
    c, remote = client
    response = c.post('/api/storage-documents/process', headers={'Origin': 'https://unrelated.example'})
    assert response.status_code == 403
    remote.start.assert_not_called()


def test_same_origin_swagger_submission_allowed(client):
    c, remote = client
    response = c.post('/api/storage-documents/process?dryRun=true', headers={'Origin': 'http://testserver'})
    assert response.status_code == 202


def test_null_origin_rejected(client):
    c, remote = client
    assert c.post('/api/storage-documents/process', headers={'Origin': 'null'}).status_code == 403
    remote.start.assert_not_called()


def test_browser_cross_site_metadata_rejected(client):
    c, remote = client
    assert c.post('/api/storage-documents/process', headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 403
    remote.start.assert_not_called()
