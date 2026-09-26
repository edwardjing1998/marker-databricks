"""Databricks Python-script entry point. This is not a notebook source file."""

import argparse
import json
from pathlib import Path
import sys


print("[MARKER] process_documents.py STARTED", flush=True)


def _project_root() -> Path:
    """
    Resolve the project root.

    Normal Python execution provides __file__.
    The Databricks spark_python_task wrapper may instead expose
    the script path through the global variable `filename`.
    """

    script_file = globals().get("__file__")

    if script_file:
        root = Path(script_file).resolve().parents[1]

        print(
            f"[MARKER] project root from __file__: {root}",
            flush=True,
        )

        return root

    databricks_filename = globals().get("filename")

    if databricks_filename:
        root = Path(databricks_filename).resolve().parents[1]

        print(
            f"[MARKER] project root from Databricks filename: {root}",
            flush=True,
        )

        return root

    # Last-resort fallback
    root = Path.cwd()

    print(
        f"[MARKER] project root from cwd: {root}",
        flush=True,
    )

    return root


PROJECT_ROOT = _project_root()

print(
    f"[MARKER] PROJECT_ROOT={PROJECT_ROOT}",
    flush=True,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

    print(
        "[MARKER] added PROJECT_ROOT to sys.path",
        flush=True,
    )


print("[MARKER] bootstrap completed", flush=True)


# ------------------------------------------------------------------
# Application imports
# ------------------------------------------------------------------

print("[MARKER] importing application modules", flush=True)

from shared.defaults import PARAM_DEFAULTS
from worker.config import Config
from worker.pipeline import run_pipeline

print("[MARKER] application modules imported", flush=True)


def main():
    print("[MARKER] main started", flush=True)

    # --------------------------------------------------------------
    # Parse Databricks job parameters
    # --------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        required=True,
    )

    for name, default in PARAM_DEFAULTS.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            default=default,
        )

    args = vars(parser.parse_args())

    print(
        "[MARKER] arguments parsed: "
        f"source_prefix={args.get('source_prefix')}, "
        f"output_prefix={args.get('output_prefix')}, "
        f"max_files={args.get('max_files')}, "
        f"overwrite={args.get('overwrite')}, "
        f"replace_existing={args.get('replace_existing')}, "
        f"dry_run={args.get('dry_run')}, "
        f"pages_per_chunk={args.get('pages_per_chunk')}, "
        f"force_ocr={args.get('force_ocr')}, "
        f"drop_handwriting={args.get('drop_handwriting')}, "
        f"compute_mode={args.get('compute_mode')}, "
        f"request_id={args.get('request_id')}",
        flush=True,
    )

    # --------------------------------------------------------------
    # Load fixed deployment configuration
    # --------------------------------------------------------------

    config_file = args.pop("config_file")

    print(
        f"[MARKER] loading config file: {config_file}",
        flush=True,
    )

    fixed = json.loads(
        Path(config_file).read_text()
    )

    print(
        "[MARKER] deployment config loaded",
        flush=True,
    )

    # --------------------------------------------------------------
    # Create worker configuration
    # --------------------------------------------------------------

    config = Config.from_values(
        fixed,
        args,
    )

    print(
        "[MARKER] Config created: "
        f"request_id={config.request_id}, "
        f"dry_run={config.dry_run}",
        flush=True,
    )

    # --------------------------------------------------------------
    # Obtain runtime-provided Spark session
    # --------------------------------------------------------------

    print(
        "[MARKER] importing SparkSession",
        flush=True,
    )

    from pyspark.sql import SparkSession

    print(
        "[MARKER] getting SparkSession",
        flush=True,
    )

    # Runtime-provided Spark / Spark Connect.
    # Do not pip-install pyspark in this project.
    spark = SparkSession.builder.getOrCreate()

    print(
        "[MARKER] SparkSession ready",
        flush=True,
    )

    # --------------------------------------------------------------
    # Execute Marker processing pipeline
    # --------------------------------------------------------------

    try:
        print(
            "[MARKER] starting run_pipeline()",
            flush=True,
        )

        report = run_pipeline(
            config,
            spark,
        )

        print(
            "[MARKER] run_pipeline() completed",
            flush=True,
        )

    except Exception as exc:
        print(
            "[MARKER] run_pipeline() FAILED: "
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
                if isinstance(
                    exc,
                    (PipelineError, ValueError),
                )
                else type(exc).__name__
            ),
        }

        # ----------------------------------------------------------
        # Persist failure report when this is a real run
        # ----------------------------------------------------------

        if not config.dry_run:
            try:
                report_path = (
                    Path(config.state_volume)
                    / "runs"
                    / config.request_id
                    / "report.json"
                )

                print(
                    "[MARKER] writing failure report: "
                    f"{report_path}",
                    flush=True,
                )

                write_json(
                    report_path,
                    report,
                )

                print(
                    "[MARKER] failure report written",
                    flush=True,
                )

            except Exception as report_exc:
                # The original implementation intentionally does not
                # fail again here because the underlying problem might
                # itself be a storage/permissions issue.
                print(
                    "[MARKER] unable to write failure report: "
                    f"{type(report_exc).__name__}: "
                    f"{report_exc}",
                    flush=True,
                )

    # --------------------------------------------------------------
    # Print machine-readable job summary
    # --------------------------------------------------------------

    print(
        "MARKER_SUMMARY="
        + json.dumps(
            report,
            separators=(",", ":"),
        ),
        flush=True,
    )

    print(
        "[MARKER] main completed: "
        f"status={report.get('status')}, "
        f"succeeded={report.get('succeeded')}, "
        f"failed={report.get('failed')}, "
        f"needsGpu={report.get('needsGpu')}",
        flush=True,
    )

    # Preserve original behavior:
    # failed documents or GPU-required documents make the task fail.
    return (
        1
        if report["failed"] or report["needsGpu"]
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())