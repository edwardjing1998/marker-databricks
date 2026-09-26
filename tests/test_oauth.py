"""OAuth-only authentication and migration regression tests. No live credentials."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from api.databricks_client import DatabricksAPI, RemoteError
from shared.oauth import oauth_credentials
from scripts.configure_gateway_secret import secret_manifest
from scripts.deploy_databricks import (
    fixed_config,
    job_settings,
    verify_cpu_job,
)
from worker.gpu_launch import runtime_api


ROOT = Path(__file__).resolve().parents[1]

CLIENT_ID = "00000000-0000-0000-0000-000000000001"
SECRET = "unit-test-secret-not-a-real-credential"

HOST = "https://example.azuredatabricks.net"

ACCESS_TOKEN = "unit-test-access-token"


@pytest.fixture
def oauth_env(monkeypatch):
    monkeypatch.setenv(
        "DATABRICKS_HOST",
        HOST,
    )

    monkeypatch.setenv(
        "DATABRICKS_CLIENT_ID",
        CLIENT_ID,
    )

    monkeypatch.setenv(
        "DATABRICKS_CLIENT_SECRET",
        SECRET,
    )

    monkeypatch.setenv(
        "DATABRICKS_AUTH_TYPE",
        "oauth-m2m",
    )


def token_response(
    *,
    access_token=ACCESS_TOKEN,
    expires_in=3600,
    status_code=200,
    ok=True,
):
    response = Mock(
        status_code=status_code,
        ok=ok,
    )

    response.json.return_value = {
        "access_token": access_token,
        "scope": "all-apis",
        "token_type": "Bearer",
        "expires_in": expires_in,
    }

    return response


def api_response(
    payload=None,
    *,
    status_code=200,
    ok=True,
):
    response = Mock(
        status_code=status_code,
        ok=ok,
    )

    response.json.return_value = (
        payload
        if payload is not None
        else {}
    )

    return response


def test_explicit_oauth_only_config(
    oauth_env,
):
    api = DatabricksAPI()

    assert api.host == HOST
    assert api.client_id == CLIENT_ID
    assert api.client_secret == SECRET

    assert api._access_token is None
    assert api._token_expires_at == 0.0


@pytest.mark.parametrize(
    "name",
    [
        "DATABRICKS_HOST",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
    ],
)
def test_missing_config_fails_without_printing_secret(
    oauth_env,
    monkeypatch,
    name,
):
    monkeypatch.delenv(name)

    with pytest.raises(
        ValueError
    ) as exc:
        oauth_credentials()

    assert name in str(exc.value)
    assert SECRET not in str(exc.value)


def test_whitespace_secret_rejected(
    oauth_env,
    monkeypatch,
):
    monkeypatch.setenv(
        "DATABRICKS_CLIENT_SECRET",
        "  ",
    )

    with pytest.raises(
        ValueError,
        match="DATABRICKS_CLIENT_SECRET",
    ):
        oauth_credentials()


def test_secret_not_in_repr(
    oauth_env,
):
    assert SECRET not in repr(
        oauth_credentials()
    )


def test_stale_pat_cannot_override_oauth(
    oauth_env,
    monkeypatch,
):
    monkeypatch.setenv(
        "DATABRICKS_TOKEN",
        "unit-test-old-token",
    )

    with pytest.raises(
        ValueError,
        match="unset DATABRICKS_TOKEN",
    ):
        oauth_credentials()


def test_other_auth_mode_rejected(
    oauth_env,
    monkeypatch,
):
    monkeypatch.setenv(
        "DATABRICKS_AUTH_TYPE",
        "pat",
    )

    with pytest.raises(
        ValueError,
        match="oauth-m2m",
    ):
        oauth_credentials()


@pytest.mark.parametrize(
    "host",
    [
        "http://example.azuredatabricks.net",
        "https://example.azuredatabricks.net/api",
        "https://example.azuredatabricks.net?x=y",
        "https://example.azuredatabricks.net#x",
        "https://user:secret@example.azuredatabricks.net",
        "https://example.azuredatabricks.net:bad",
    ],
)
def test_invalid_workspace_origins(
    oauth_env,
    host,
):
    with pytest.raises(
        ValueError
    ):
        oauth_credentials(
            host=host
        )


def test_trailing_slash_normalized(
    oauth_env,
):
    credentials = oauth_credentials(
        host=(
            "https://"
            "example.azuredatabricks.net/"
        )
    )

    assert (
        credentials.host
        == HOST
    )


def test_direct_oauth_token_request(
    oauth_env,
    monkeypatch,
):
    post = Mock(
        return_value=token_response()
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.post",
        post,
    )

    api = DatabricksAPI()

    token = api._fetch_access_token()

    assert token == ACCESS_TOKEN

    post.assert_called_once()

    args = post.call_args

    assert (
        args.args[0]
        == HOST + "/oidc/v1/token"
    )

    assert args.kwargs["auth"] == (
        CLIENT_ID,
        SECRET,
    )

    assert args.kwargs["data"] == {
        "grant_type": "client_credentials",
        "scope": "all-apis",
    }

    assert args.kwargs[
        "allow_redirects"
    ] is False


def test_token_endpoint_failure_is_sanitized(
    oauth_env,
    monkeypatch,
):
    response = Mock(
        status_code=401,
        ok=False,
        text=SECRET,
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.post",
        Mock(return_value=response),
    )

    api = DatabricksAPI()

    with pytest.raises(
        RemoteError
    ) as exc:
        api._fetch_access_token()

    message = str(exc.value)

    assert SECRET not in message
    assert "HTTP 401" in message


def test_token_request_network_failure_is_sanitized(
    oauth_env,
    monkeypatch,
):
    import requests

    post = Mock(
        side_effect=requests.ConnectionError(
            "secret=" + SECRET
        )
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.post",
        post,
    )

    api = DatabricksAPI()

    with pytest.raises(
        RemoteError
    ) as exc:
        api._fetch_access_token()

    assert SECRET not in str(
        exc.value
    )

    assert (
        "OAuth token request"
        in str(exc.value)
    )


def test_non_json_token_response_rejected(
    oauth_env,
    monkeypatch,
):
    response = Mock(
        status_code=200,
        ok=True,
    )

    response.json.side_effect = (
        ValueError("bad json")
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.post",
        Mock(return_value=response),
    )

    api = DatabricksAPI()

    with pytest.raises(
        RemoteError,
        match="non-JSON",
    ):
        api._fetch_access_token()


def test_missing_access_token_rejected(
    oauth_env,
    monkeypatch,
):
    response = Mock(
        status_code=200,
        ok=True,
    )

    response.json.return_value = {
        "token_type": "Bearer",
        "expires_in": 3600,
    }

    monkeypatch.setattr(
        "api.databricks_client.requests.post",
        Mock(return_value=response),
    )

    api = DatabricksAPI()

    with pytest.raises(
        RemoteError,
        match="no access_token",
    ):
        api._fetch_access_token()


def test_token_cached_between_requests(
    oauth_env,
    monkeypatch,
):
    post = Mock(
        return_value=token_response(
            access_token="cached-token"
        )
    )

    send = Mock(
        return_value=api_response(
            {"jobs": []}
        )
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.post",
        post,
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.request",
        send,
    )

    api = DatabricksAPI()

    api.call(
        "GET",
        "/api/2.2/jobs/list",
    )

    api.call(
        "GET",
        "/api/2.2/jobs/list",
    )

    assert post.call_count == 1
    assert send.call_count == 2

    assert (
        send.call_args_list[0]
        .kwargs["headers"]["Authorization"]
        == "Bearer cached-token"
    )

    assert (
        send.call_args_list[1]
        .kwargs["headers"]["Authorization"]
        == "Bearer cached-token"
    )


def test_expired_token_is_refreshed(
    oauth_env,
    monkeypatch,
):
    post = Mock(
        side_effect=[
            token_response(
                access_token="first-token"
            ),
            token_response(
                access_token="second-token"
            ),
        ]
    )

    send = Mock(
        return_value=api_response(
            {"jobs": []}
        )
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.post",
        post,
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.request",
        send,
    )

    api = DatabricksAPI()

    api.call(
        "GET",
        "/api/2.2/jobs/list",
    )

    api._token_expires_at = 0.0

    api.call(
        "GET",
        "/api/2.2/jobs/list",
    )

    assert post.call_count == 2

    assert (
        send.call_args_list[0]
        .kwargs["headers"]["Authorization"]
        == "Bearer first-token"
    )

    assert (
        send.call_args_list[1]
        .kwargs["headers"]["Authorization"]
        == "Bearer second-token"
    )


def test_401_safe_retry_refreshes_token(
    oauth_env,
    monkeypatch,
):
    post = Mock(
        side_effect=[
            token_response(
                access_token="first-token"
            ),
            token_response(
                access_token="second-token"
            ),
        ]
    )

    unauthorized = api_response(
        status_code=401,
        ok=False,
    )

    success = api_response(
        {"jobs": []},
        status_code=200,
        ok=True,
    )

    send = Mock(
        side_effect=[
            unauthorized,
            success,
        ]
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.post",
        post,
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.request",
        send,
    )

    monkeypatch.setattr(
        "api.databricks_client.time.sleep",
        Mock(),
    )

    api = DatabricksAPI()

    result = api.call(
        "GET",
        "/api/2.2/jobs/list",
        safe_retry=True,
    )

    assert result == {
        "jobs": []
    }

    assert post.call_count == 2
    assert send.call_count == 2

    assert (
        send.call_args_list[0]
        .kwargs["headers"]["Authorization"]
        == "Bearer first-token"
    )

    assert (
        send.call_args_list[1]
        .kwargs["headers"]["Authorization"]
        == "Bearer second-token"
    )


def test_error_body_not_exposed(
    oauth_env,
    monkeypatch,
):
    post = Mock(
        return_value=token_response()
    )

    response = Mock(
        status_code=403,
        ok=False,
        text=SECRET,
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.post",
        post,
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.request",
        Mock(return_value=response),
    )

    with pytest.raises(
        RemoteError
    ) as exc:
        DatabricksAPI().call(
            "GET",
            "/api/2.2/jobs/list",
        )

    message = str(exc.value)

    assert SECRET not in message
    assert "HTTP 403" in message


def test_api_request_uses_bearer_token(
    oauth_env,
    monkeypatch,
):
    post = Mock(
        return_value=token_response(
            access_token="request-token"
        )
    )

    send = Mock(
        return_value=api_response(
            {"jobs": []}
        )
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.post",
        post,
    )

    monkeypatch.setattr(
        "api.databricks_client.requests.request",
        send,
    )

    api = DatabricksAPI()

    result = api.call(
        "GET",
        "/api/2.2/jobs/list",
    )

    assert result == {
        "jobs": []
    }

    call = send.call_args

    assert call.args == (
        "GET",
        HOST + "/api/2.2/jobs/list",
    )

    assert (
        call.kwargs[
            "headers"
        ]["Authorization"]
        == "Bearer request-token"
    )


def test_gateway_secret_contains_only_pair(
    oauth_env,
):
    manifest = secret_manifest()

    assert manifest[
        "stringData"
    ] == {
        "DATABRICKS_CLIENT_ID": (
            CLIENT_ID
        ),
        "DATABRICKS_CLIENT_SECRET": (
            SECRET
        ),
    }

    assert (
        manifest["metadata"]["name"]
        == "marker-databricks-secret"
    )


def test_secret_sent_over_stdin_not_command_line(
    oauth_env,
    monkeypatch,
    capsys,
):
    import scripts.configure_gateway_secret as module

    run = Mock(
        return_value=SimpleNamespace(
            returncode=0
        )
    )

    monkeypatch.setattr(
        module.subprocess,
        "run",
        run,
    )

    module.main()

    assert (
        run.call_args.args[0]
        == [
            "oc",
            "apply",
            "-f",
            "-",
        ]
    )

    assert SECRET not in repr(
        run.call_args.args[0]
    )

    assert (
        SECRET
        in run.call_args.kwargs[
            "input"
        ]
    )

    assert (
        SECRET
        not in capsys.readouterr().out
    )


def test_secret_failure_does_not_echo_payload(
    oauth_env,
    monkeypatch,
):
    import scripts.configure_gateway_secret as module

    monkeypatch.setattr(
        module.subprocess,
        "run",
        Mock(
            return_value=SimpleNamespace(
                returncode=1,
                stderr=SECRET,
            )
        ),
    )

    with pytest.raises(
        SystemExit
    ) as exc:
        module.main()

    assert SECRET not in str(
        exc.value
    )


def test_gpu_launcher_reads_oauth_pair_only(
    cfg,
    monkeypatch,
):
    spark = Mock()

    def sql(query):
        return Mock(
            first=lambda: {
                "value": (
                    CLIENT_ID
                    if "gpu-launch-client-id"
                    in query
                    else SECRET
                )
            }
        )

    spark.sql.side_effect = sql

    factory = Mock()

    monkeypatch.setattr(
        "worker.gpu_launch.DatabricksAPI",
        factory,
    )

    runtime_api(
        spark,
        cfg,
    )

    assert factory.call_args.kwargs == {
        "host": cfg.workspace_host,
        "client_id": CLIENT_ID,
        "client_secret": SECRET,
    }

    assert spark.sql.call_count == 2

    assert all(
        "gpu-launch-token"
        not in call.args[0]
        for call
        in spark.sql.call_args_list
    )


def test_gpu_launcher_pat_configuration_rejected(
    cfg,
):
    cfg.gpu_launch_auth = "pat"

    with pytest.raises(
        ValueError,
        match="oauth-m2m",
    ):
        cfg.validate(
            require_volumes=False
        )


@pytest.fixture
def job_env(
    oauth_env,
    monkeypatch,
):
    values = {
        "SOURCE_VOLUME_PATH": (
            "/Volumes/c/s/source"
        ),
        "OUTPUT_VOLUME_PATH": (
            "/Volumes/c/s/output"
        ),
        "STATE_VOLUME_PATH": (
            "/Volumes/c/s/state"
        ),
    }

    for key, value in values.items():
        monkeypatch.setenv(
            key,
            value,
        )


def test_job_defaults_to_client_identity(
    job_env,
):
    settings = job_settings(
        "/Workspace/Shared/"
        "marker-databricks/releases/test"
    )

    assert settings["run_as"] == {
        "service_principal_name": (
            CLIENT_ID
        )
    }

    assert SECRET not in json.dumps(
        settings
    )


def test_run_as_override_requires_no_secret_in_job_config(
    job_env,
    monkeypatch,
):
    monkeypatch.setenv(
        "DATABRICKS_RUN_AS_SERVICE_PRINCIPAL",
        "another-principal-id",
    )

    settings = job_settings(
        "/Workspace/Shared/"
        "marker-databricks/releases/test"
    )

    assert (
        settings["run_as"][
            "service_principal_name"
        ]
        == "another-principal-id"
    )


def test_incorrect_saved_run_identity_rejected(
    job_env,
):
    settings = job_settings(
        "/Workspace/Shared/"
        "marker-databricks/releases/test"
    )

    actual = deepcopy(
        settings
    )

    actual["run_as"][
        "service_principal_name"
    ] = "unexpected"

    with pytest.raises(
        RuntimeError,
        match="Run as",
    ):
        verify_cpu_job(
            actual,
            settings,
        )


def test_release_configuration_contains_no_credentials(
    job_env,
):
    cfg = fixed_config(
        "/Workspace/Shared/"
        "marker-databricks/releases/test"
    )

    assert (
        cfg["GPU_LAUNCH_AUTH"]
        == "oauth-m2m"
    )

    assert SECRET not in json.dumps(
        cfg
    )

    assert not any(
        key in cfg
        for key in [
            "DATABRICKS_TOKEN",
            "DATABRICKS_CLIENT_SECRET",
            "DATABRICKS_CLIENT_ID",
        ]
    )


def test_workflow_never_references_pat_secret():
    text = (
        ROOT
        / ".github/workflows/deploy.yml"
    ).read_text()

    workflow = yaml.safe_load(
        text
    )["jobs"]["deploy"]

    assert (
        "secrets.DATABRICKS_TOKEN"
        not in text
    )

    assert (
        "DATABRICKS_TOKEN"
        not in workflow["env"]
    )

    assert (
        workflow["env"][
            "DATABRICKS_AUTH_TYPE"
        ]
        == "oauth-m2m"
    )

    assert (
        workflow["env"][
            "GPU_LAUNCH_AUTH"
        ]
        == "oauth-m2m"
    )

    assert (
        '"DATABRICKS_TOKEN":null'
        in text
    )

    assert (
        "scripts/configure_gateway_secret.py"
        in text
    )


def test_openshift_only_injects_explicit_oauth_secret_fields():
    docs = list(
        yaml.safe_load_all(
            (
                ROOT
                / "openshift/app.yml"
            ).read_text()
        )
    )

    deployment = next(
        item
        for item in docs
        if item["kind"]
        == "Deployment"
    )

    container = (
        deployment[
            "spec"
        ][
            "template"
        ][
            "spec"
        ][
            "containers"
        ][0]
    )

    assert not any(
        "secretRef" in entry
        for entry
        in container["envFrom"]
    )

    assert {
        entry["name"]
        for entry
        in container["env"]
    } == {
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_AUTH_TYPE",
    }


@pytest.mark.parametrize(
    "missing",
    [
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
    ],
)
def test_workflow_validation_requires_both_fields(
    missing,
):
    workflow = yaml.safe_load(
        (
            ROOT
            / ".github/workflows/deploy.yml"
        ).read_text()
    )

    step = next(
        item
        for item
        in workflow[
            "jobs"
        ][
            "deploy"
        ][
            "steps"
        ]
        if item.get("name")
        == "Validate deployment configuration"
    )

    env = {
        "PATH": os.environ["PATH"]
    }

    for name in (
        "OPENSHIFT_SERVER",
        "OPENSHIFT_NAMESPACE",
        "OPENSHIFT_TOKEN",
        "DATABRICKS_HOST",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "AZURE_STORAGE_ENDPOINT",
        "SOURCE_VOLUME_PATH",
        "OUTPUT_VOLUME_PATH",
        "STATE_VOLUME_PATH",
        "GHCR_USERNAME",
        "GHCR_PULL_TOKEN",
    ):
        env[name] = (
            "test-placeholder"
        )

    del env[missing]

    result = subprocess.run(
        [
            "bash",
            "-c",
            step["run"],
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert (
        result.returncode != 0
    )

    assert (
        missing
        in result.stdout
    )


def test_oauth_preflight_is_read_only(
    oauth_env,
    monkeypatch,
    capsys,
):
    import scripts.check_oauth as module

    api = Mock()

    monkeypatch.setattr(
        module,
        "DatabricksAPI",
        lambda: api,
    )

    module.main()

    assert (
        api.call.call_args.args
        == (
            "GET",
            "/api/2.2/jobs/list",
        )
    )

    assert (
        SECRET
        not in capsys.readouterr().out
    )