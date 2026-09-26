"""Explicit optional GPU child run; no task values, notebooks, or CPU-to-GPU mutation.

The CPU coordinator waits for the child, so parent CPU resources may remain
billable during this wait. A whole-parent kill may leave a child alive until its
independent timeout. A sticky lock prevents a new parent from racing that child.
"""

from dataclasses import asdict
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import tarfile
import tempfile
import time

from api.databricks_client import DatabricksAPI
from shared.errors import PipelineError
from shared.paths import digest
from worker.volumes import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]

TERMINAL = {
    "TERMINATED",
    "SKIPPED",
    "INTERNAL_ERROR",
}


def runtime_api(spark, config):
    def secret(key):
        if not re.fullmatch(
            "[A-Za-z0-9_.-]{1,128}",
            config.secret_scope,
        ):
            raise PipelineError(
                "Invalid launcher secret scope"
            )

        # Literal scope/key only.
        # Never print, persist, or return secret values.
        return spark.sql(
            f"SELECT secret("
            f"'{config.secret_scope}', "
            f"'{key}') AS value"
        ).first()["value"]

    if config.gpu_launch_auth != "oauth-m2m":
        raise PipelineError(
            "GPU launcher requires oauth-m2m"
        )

    client_id = secret(
        "gpu-launch-client-id"
    )

    client_secret = secret(
        "gpu-launch-client-secret"
    )

    # Safe diagnostics only.
    # Never print the actual client ID or client secret.
    print(
        "[MARKER][GPU][AUTH] "
        f"workspace_host={config.workspace_host}",
        flush=True,
    )

    print(
        "[MARKER][GPU][AUTH] "
        f"secret_scope={config.secret_scope}",
        flush=True,
    )

    print(
        "[MARKER][GPU][AUTH] "
        f"client_id_length={len(client_id)} "
        f"client_id_sha256="
        f"{hashlib.sha256(client_id.encode()).hexdigest()}",
        flush=True,
    )

    print(
        "[MARKER][GPU][AUTH] "
        f"client_secret_length={len(client_secret)} "
        f"client_secret_sha256="
        f"{hashlib.sha256(client_secret.encode()).hexdigest()}",
        flush=True,
    )

    return DatabricksAPI(
        host=config.workspace_host,
        client_id=client_id,
        client_secret=client_secret,
    )


def command_text(
    manifest_path,
    archive_path,
    archive_sha,
):
    """The command is in a workspace file; only trusted application code is unpacked."""

    unpack = (
        "import hashlib,sys,tarfile; "
        "p=sys.argv[1]; "
        "data=open(p,'rb').read(); "
        "assert hashlib.sha256(data).hexdigest()==sys.argv[3], "
        "'Code hash mismatch'; "
        "t=tarfile.open(p); "
        "t.extractall(sys.argv[2],filter='data'); "
        "t.close()"
    )

    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "CODE_DIR=$(mktemp -d /tmp/marker-code-XXXXXXXX)\n"
        "trap 'rm -rf \"$CODE_DIR\"' EXIT\n"
        f"python -c {shlex.quote(unpack)} "
        f"{shlex.quote(str(archive_path))} "
        f"\"$CODE_DIR\" "
        f"{shlex.quote(archive_sha)}\n"
        "cd \"$CODE_DIR\"\n"
        f"python jobs/gpu_worker.py "
        f"--manifest {shlex.quote(str(manifest_path))}\n"
    )


def gpu_submit_body(
    config,
    command_path,
):
    return {
        "run_name": (
            "marker-gpu-"
            + config.request_id
        ),
        "idempotency_token": digest(
            [
                "marker-gpu-v3",
                config.request_id,
                config.release_id,
            ]
        ),
        "timeout_seconds": (
            config.gpu_timeout_seconds
        ),
        "tasks": [
            {
                "task_key": "marker_gpu",
                "max_retries": 0,
                "retry_on_timeout": False,
                "timeout_seconds": (
                    config.gpu_timeout_seconds
                ),
                "environment_key": (
                    "marker_gpu"
                ),
                "ai_runtime_task": {
                    "experiment": (
                        "marker-cpu-first-gpu"
                    ),
                    "mlflow_experiment_directory": (
                        "/Workspace/Shared/"
                        "marker-databricks"
                    ),
                    "deployments": [
                        {
                            "command_path": (
                                command_path
                            ),
                            "compute": {
                                "accelerator_type": (
                                    config.accelerator
                                ),
                                "accelerator_count": 1,
                            },
                        }
                    ],
                },
            }
        ],
        "environments": [
            {
                "environment_key": (
                    "marker_gpu"
                ),
                "spec": {
                    "environment_version": (
                        config.environment_version
                    ),
                    "dependencies": [
                        "-r "
                        + config.workspace_release
                        + "/requirements-gpu.txt"
                    ],
                },
            }
        ],
    }


def pack_runtime(target):
    """Stage only trusted release files. No credentials, .git files, tests, or inputs."""

    with tempfile.TemporaryDirectory() as tmp:
        archive = (
            Path(tmp)
            / "runtime.tgz"
        )

        with tarfile.open(
            archive,
            "w:gz",
        ) as tar:
            for folder in (
                "jobs",
                "worker",
                "shared",
                "api",
            ):
                for path in sorted(
                    (
                        ROOT
                        / folder
                    ).glob("*.py")
                ):
                    tar.add(
                        path,
                        arcname=str(
                            path.relative_to(
                                ROOT
                            )
                        ),
                    )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copyfile(
            archive,
            target,
        )

    from worker.volumes import sha256_file

    return sha256_file(
        target
    )


def run_gpu(
    config,
    records,
    spark,
    lock,
    report,
    save_report,
    api=None,
):
    if (
        not config.gpu_enabled
        or not records
    ):
        raise PipelineError(
            "GPU launch requires enabled policy "
            "and nonempty work"
        )

    api = (
        api
        or runtime_api(
            spark,
            config,
        )
    )

    run_dir = (
        Path(config.state_volume)
        / "runs"
        / config.request_id
    )

    manifest = (
        run_dir
        / "gpu-manifest.json"
    )

    write_json(
        manifest,
        {
            "schema": (
                "marker-gpu-manifest/v3"
            ),
            "config": asdict(
                config
            ),
            "records": records,
        },
    )

    archive = (
        run_dir
        / "runtime.tgz"
    )

    archive_sha = pack_runtime(
        archive
    )

    command_path = (
        "/Workspace/Shared/"
        "marker-databricks/"
        "gpu-requests/"
        + config.request_id
        + "/run.sh"
    )

    api.call(
        "POST",
        "/api/2.0/workspace/mkdirs",
        body={
            "path": str(
                Path(
                    command_path
                ).parent
            ).removeprefix(
                "/Workspace"
            )
        },
    )

    api.call(
        "POST",
        "/api/2.0/workspace/import",
        body={
            "path": (
                command_path.removeprefix(
                    "/Workspace"
                )
            ),
            "format": "AUTO",
            "overwrite": True,
            "content": (
                base64.b64encode(
                    command_text(
                        manifest,
                        archive,
                        archive_sha,
                    ).encode()
                ).decode()
            ),
        },
    )

    # Write the hold BEFORE POST:
    # a response can be lost after the service accepts it.
    body = gpu_submit_body(
        config,
        command_path,
    )

    lock.hold(
        phase="GPU_SUBMITTING",
        idempotencyToken=(
            body[
                "idempotency_token"
            ]
        ),
    )

    write_json(
        run_dir
        / "gpu-submission.json",
        body,
    )

    response = api.call(
        "POST",
        "/api/2.2/jobs/runs/submit",
        body=body,
        safe_retry=True,
    )

    child = int(
        response["run_id"]
    )

    report["gpuRunId"] = (
        child
    )

    report["status"] = (
        "GPU_RUNNING"
    )

    save_report()

    lock.hold(
        phase="GPU_RUNNING",
        gpuRunId=child,
    )

    deadline = (
        time.monotonic()
        + config.gpu_wait_seconds
    )

    terminal = False

    try:
        while (
            time.monotonic()
            < deadline
        ):
            run = api.run(
                child
            )

            state = run.get(
                "state",
                {},
            )

            if (
                state.get(
                    "life_cycle_state"
                )
                in TERMINAL
            ):
                terminal = True

                lock.clear_child()

                report[
                    "gpuResultState"
                ] = state.get(
                    "result_state"
                )

                if not (
                    run_dir
                    / "gpu-report.json"
                ).exists():
                    raise PipelineError(
                        "GPU child produced no report; "
                        "inspect its Databricks run"
                    )

                child_report = read_json(
                    run_dir
                    / "gpu-report.json"
                )

                if (
                    child_report.get(
                        "requestId"
                    )
                    != config.request_id
                ):
                    raise PipelineError(
                        "GPU report identity mismatch"
                    )

                expected = {
                    (
                        config.source_root
                        + row["relative"]
                    )
                    for row in records
                }

                results = (
                    child_report.get(
                        "results",
                        [],
                    )
                )

                actual = [
                    row.get(
                        "sourceBlob"
                    )
                    for row in results
                ]

                if (
                    len(actual)
                    != len(
                        set(actual)
                    )
                    or set(actual)
                    != expected
                ):
                    raise PipelineError(
                        "GPU report does not match "
                        "the submitted documents"
                    )

                if any(
                    row.get(
                        "status"
                    )
                    not in (
                        "SUCCEEDED",
                        "FAILED",
                        "SKIPPED",
                    )
                    for row
                    in results
                ):
                    raise PipelineError(
                        "GPU report contains "
                        "an invalid result status"
                    )

                if (
                    state.get(
                        "result_state"
                    )
                    != "SUCCESS"
                    and all(
                        row[
                            "status"
                        ]
                        != "FAILED"
                        for row
                        in results
                    )
                ):
                    raise PipelineError(
                        "GPU run failed despite "
                        "successful file report; "
                        "inspect child logs"
                    )

                return results

            time.sleep(
                15
            )

        raise PipelineError(
            "GPU wait budget exceeded; "
            "cancellation requested, "
            "verify child state"
        )

    finally:
        if not terminal:
            try:
                api.call(
                    "POST",
                    "/api/2.2/jobs/runs/cancel",
                    body={
                        "run_id": child
                    },
                )

            except Exception:
                pass

            # Keep lock:
            # cancellation acceptance is not proof
            # the child has stopped.
            lock.hold(
                phase=(
                    "GPU_STATUS_UNCERTAIN"
                ),
                gpuRunId=child,
            )