"""CPU-first coordinator, called by a real PySpark Python-script task."""

from contextlib import nullcontext
from pathlib import Path

from shared.errors import PipelineError, CpuResourceLimit
from worker.config import route_document
from worker.cpu_runner import run_cpu
from worker.discovery import discover
from worker.document import (
    convert_and_publish,
    existing_output,
    inspect_source,
    utc_now,
)
from worker.gpu_launch import run_gpu
from worker.volumes import RunLock, probe_volume, write_json


def run_pipeline(
    config,
    spark,
    *,
    discover_fn=discover,
    inspect_fn=inspect_source,
    convert_fn=convert_and_publish,
    gpu_fn=run_gpu,
):
    report = {
        "requestId": config.request_id,
        "status": "RUNNING",
        "computeMode": config.compute_mode,
        "startedAt": utc_now(),
        "results": [],
        "gpuRunId": None,
        "succeeded": 0,
        "failed": 0,
        "needsGpu": 0,
        "skipped": 0,
        "reportBlob": (
            "_marker_jobs/runs/"
            + config.request_id
            + "/report.json"
        ),
    }

    report_path = (
        Path(config.state_volume)
        / "runs"
        / config.request_id
        / "report.json"
    )

    def save():
        if not config.dry_run:
            write_json(report_path, report)

    if not config.dry_run:
        probe_volume(config.output_volume)
        probe_volume(config.state_volume)

    context = (
        nullcontext(None)
        if config.dry_run
        else RunLock(
            config.state_volume,
            config.request_id,
        )
    )

    with context as lock:
        save()

        print(
            "[MARKER][pipeline] discovery started "
            f"source_prefix={config.source_prefix} "
            f"max_files={config.max_files}",
            flush=True,
        )

        records = discover_fn(spark, config)

        print(
            "[MARKER][pipeline] discovery completed "
            f"records={len(records)}",
            flush=True,
        )

        pending = []
        attempted = 0

        for item in records:
            if attempted >= config.max_files:
                break

            source_blob = (
                config.source_root
                + item["relative"]
            )

            try:
                attempted += 1

                print(
                    "[MARKER][pipeline] inspecting "
                    f"source={source_blob}",
                    flush=True,
                )

                if (
                    existing_output(
                        config,
                        item["relative"],
                    )
                    == "SKIP"
                ):
                    attempted -= 1
                    report["skipped"] += 1

                    print(
                        "[MARKER][pipeline] skipped existing "
                        f"source={source_blob}",
                        flush=True,
                    )

                    if len(report["results"]) < 200:
                        report["results"].append(
                            {
                                "sourceBlob": source_blob,
                                "status": "SKIPPED",
                            }
                        )

                    continue

                record = inspect_fn(
                    config,
                    item["relative"],
                )

                route, reason = route_document(
                    config,
                    pages=record["pages"],
                    size=record["bytes"],
                )

                print(
                    "[MARKER][pipeline] routing "
                    f"source={source_blob} "
                    f"route={route} "
                    f"reason={reason}",
                    flush=True,
                )

                if config.dry_run:
                    report["results"].append(
                        {
                            "sourceBlob": source_blob,
                            "status": "WOULD_PROCESS",
                            "route": route,
                            "reason": reason,
                            "gpuAllowed": (
                                config.gpu_enabled
                                and config.compute_mode != "cpu"
                            ),
                        }
                    )

                    continue

                if route == "cpu":
                    try:
                        print(
                            "[MARKER][CPU] conversion started "
                            f"source={source_blob}",
                            flush=True,
                        )

                        result = convert_fn(
                            config,
                            record,
                            "cpu",
                            runner=run_cpu,
                        )

                        print(
                            "[MARKER][CPU] conversion completed "
                            f"source={source_blob} "
                            f"status={result.get('status')}",
                            flush=True,
                        )

                        report["results"].append(result)

                        continue

                    except CpuResourceLimit as exc:
                        reason = str(exc)

                        print(
                            "[MARKER][CPU] resource limit "
                            f"source={source_blob} "
                            f"reason={reason}",
                            flush=True,
                        )

                pending.append(
                    {
                        **record,
                        "routingReason": reason,
                    }
                )

                print(
                    "[MARKER][GPU] queued "
                    f"source={source_blob} "
                    f"reason={reason}",
                    flush=True,
                )

            except Exception as exc:
                # Corruption, permissions, package/model download
                # errors do NOT trigger GPU fallback.
                message = (
                    str(exc)
                    if isinstance(
                        exc,
                        (PipelineError, ValueError),
                    )
                    else (
                        f"{type(exc).__name__}: "
                        f"{str(exc)}"
                    )
                )

                print(
                    "[MARKER][pipeline] document failed "
                    f"source={source_blob} "
                    f"error={message}",
                    flush=True,
                )

                report["results"].append(
                    {
                        "sourceBlob": source_blob,
                        "status": "FAILED",
                        "error": message,
                    }
                )

            finally:
                save()

        if pending:
            print(
                "[MARKER][GPU] pending documents "
                f"count={len(pending)} "
                f"compute_mode={config.compute_mode} "
                f"gpu_enabled={config.gpu_enabled}",
                flush=True,
            )

            if (
                config.compute_mode == "cpu"
                or not config.gpu_enabled
            ):
                print(
                    "[MARKER][GPU] GPU handoff not allowed "
                    f"compute_mode={config.compute_mode} "
                    f"gpu_enabled={config.gpu_enabled}",
                    flush=True,
                )

                report["results"].extend(
                    {
                        "sourceBlob": (
                            config.source_root
                            + r["relative"]
                        ),
                        "status": "NEEDS_GPU",
                        "reason": r["routingReason"],
                    }
                    for r in pending
                )

            else:
                try:
                    print(
                        "[MARKER][GPU] handoff started "
                        f"count={len(pending)}",
                        flush=True,
                    )

                    gpu_results = gpu_fn(
                        config,
                        pending,
                        spark,
                        lock,
                        report,
                        save,
                    )

                    print(
                        "[MARKER][GPU] handoff completed "
                        f"count={len(gpu_results)} "
                        f"gpuRunId={report.get('gpuRunId')}",
                        flush=True,
                    )

                    report["results"].extend(
                        gpu_results
                    )

                except Exception as exc:
                    error_message = (
                        f"{type(exc).__name__}: "
                        f"{str(exc)}"
                    )

                    print(
                        "[MARKER][GPU] handoff failed "
                        f"error={error_message}",
                        flush=True,
                    )

                    report["gpuError"] = (
                        error_message
                    )

                    report["results"].extend(
                        {
                            "sourceBlob": (
                                config.source_root
                                + r["relative"]
                            ),
                            "status": "FAILED",
                            "error": (
                                "GPU handoff failed; "
                                "see gpuError and child run"
                            ),
                        }
                        for r in pending
                    )

        report["succeeded"] = sum(
            r["status"] == "SUCCEEDED"
            for r in report["results"]
        )

        report["failed"] = sum(
            r["status"] == "FAILED"
            for r in report["results"]
        )

        report["needsGpu"] = sum(
            r["status"] == "NEEDS_GPU"
            for r in report["results"]
        )

        report["status"] = (
            "DRY_RUN_COMPLETE"
            if (
                config.dry_run
                and not report["failed"]
            )
            else "COMPLETED_WITH_ERRORS"
            if report["failed"]
            else "NEEDS_GPU"
            if report["needsGpu"]
            else "COMPLETED"
        )

        report["finishedAt"] = utc_now()

        print(
            "[MARKER][pipeline] completed "
            f"status={report['status']} "
            f"succeeded={report['succeeded']} "
            f"failed={report['failed']} "
            f"needsGpu={report['needsGpu']} "
            f"gpuRunId={report['gpuRunId']}",
            flush=True,
        )

        save()

    return report