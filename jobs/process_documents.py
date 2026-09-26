"""Databricks Python-script entry point. This is not a notebook source file."""

import argparse
import json
from pathlib import Path
import sys


print("[MARKER] process_documents.py STARTED", flush=True)


def _project_root() -> Path:
    script_file = globals().get("__file__")

    if script_file:
        root = Path(script_file).resolve().parents[1]
        print(f"[MARKER] project root from __file__: {root}", flush=True)
        return root

    # Databricks spark_python_task wrapper may expose the script path as `filename`
    databricks_filename = globals().get("filename")

    if databricks_filename:
        root = Path(databricks_filename).resolve().parents[1]
        print(
            f"[MARKER] project root from Databricks filename: {root}",
            flush=True,
        )
        return root

    root = Path.cwd()
    print(f"[MARKER] project root from cwd: {root}", flush=True)
    return root


PROJECT_ROOT = _project_root()

print(f"[MARKER] PROJECT_ROOT={PROJECT_ROOT}", flush=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print("[MARKER] added PROJECT_ROOT to sys.path", flush=True)

print("[MARKER] bootstrap completed", flush=True)


print("[MARKER] importing application modules", flush=True)

from shared.defaults import PARAM_DEFAULTS
from worker.config import Config
from worker.pipeline import run_pipeline

print("[MARKER] application modules imported", flush=True)


def main():
    print("[MARKER] main started", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)

    for name, default in PARAM_DEFAULTS.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            default=default,
        )

    args = vars(parser.parse_args())

    print(
        f"[MARKER] arguments parsed: "
        f"source_prefix={args.get('source_prefix')}, "
        f"output_prefix={args.get('output_prefix')}, "
        f"max_files={args.get('max_files')}, "
        f"dry_run={args.get('dry_run')}, "
        f"compute_mode={args.get('compute_mode')}",
        flush=True,
    )

    config_file = args.pop("config_file")

    print(
        f"[MARKER] loading config file: {config_file}",
        flush=True,
    )

    fixed = json.loads(Path(config_file).read_text())

    config = Config.from_values(fixed, args)

    print(
        f"[MARKER] config created: "
        f"request_id={config.request_id}, "
        f"source_volume={config.source_volume}, "
        f"output_volume={config.output_volume}, "
        f"state_volume={config.state_volume}, "
        f"dry_run={config.dry_run}",
        flush=True,
    )

    from pyspark.sql import SparkSession

    print("[MARKER] creating/getting SparkSession", flush=True)

    spark = SparkSession.builder.getOrCreate()

    print("[MARKER] SparkSession ready", flush=True)

    try:
        print("[MARKER] starting run_pipeline()", flush=True)

        report = run_pipeline(config, spark)

        print(
            "[MARKER] run_pipeline() completed",
            flush=True,
        )

    except Exception as exc:
        print(
            f"[MARKER] run_pipeline() FAILED: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        from shared.errors import PipelineError
        from worker.document import utc_now
        from worker.volumes import write_json

        report = {
            "requestId": config.request_id,
            "status": "FAILED",
            "failed": 1,
            "needsGpu": 0,
            "succeeded": 0,
            "finishedAt": utc_now(),
            "error": (
                str(exc)
                if isinstance(exc, (PipelineError, ValueError))
                else type(exc).__name__
            ),
        }

        if not config.dry_run:
            try:
                report_path = (
                    Path(config.state_volume)
                    / "runs"
                    / config.request_id
                    / "report.json"
                )

                print(
                    f"[MARKER] writing failure report: {report_path}",
                    flush=True,
                )

                write_json(report_path, report)

            except Exception as report_exc:
                print(
                    f"[MARKER] unable to write failure report: "
                    f"{type(report_exc).__name__}: {report_exc}",
                    flush=True,
                )

    print(
        "MARKER_SUMMARY="
        + json.dumps(report, separators=(",", ":")),
        flush=True,
    )

    print(
        f"[MARKER] main completed: "
        f"failed={report.get('failed')}, "
        f"needsGpu={report.get('needsGpu')}, "
        f"succeeded={report.get('succeeded')}",
        flush=True,
    )

    return 1 if report["failed"] or report["needsGpu"] else 0


if __name__ == "__main__":
    raise SystemExit(main())