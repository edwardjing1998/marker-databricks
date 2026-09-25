from pathlib import Path
import pytest
from worker.config import Config

@pytest.fixture
def cfg(tmp_path):
    for name in ('input', 'output', 'state'):
        (tmp_path / name).mkdir()
    return Config(str(tmp_path/'input'), str(tmp_path/'output'), str(tmp_path/'state'),
        workspace_host='https://example.azuredatabricks.net',
        workspace_release='/Workspace/Shared/marker-databricks/releases/test',
        request_id='a'*32, dry_run=False).validate(require_volumes=False)


@pytest.fixture(autouse=True)
def isolate_auth_environment(monkeypatch):
    # CI has real deployment credentials. Unit tests must never use or log them.
    for key in ('DATABRICKS_TOKEN', 'DATABRICKS_CLIENT_ID', 'DATABRICKS_CLIENT_SECRET',
                'DATABRICKS_AUTH_TYPE', 'DATABRICKS_RUN_AS_SERVICE_PRINCIPAL'):
        monkeypatch.delenv(key, raising=False)
