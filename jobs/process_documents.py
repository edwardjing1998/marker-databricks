"""Databricks Python-script entry point. This is not a notebook source file."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.defaults import PARAM_DEFAULTS
from worker.config import Config
from worker.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    for name, default in PARAM_DEFAULTS.items():
        parser.add_argument('--' + name.replace('_', '-'), default=default)
    args = vars(parser.parse_args())
    fixed = json.loads(Path(args.pop('config_file')).read_text())
    config = Config.from_values(fixed, args)
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()  # Runtime-provided Spark Connect; do not pip-install pyspark here.
    try:
        report = run_pipeline(config, spark)
    except Exception as exc:
        from shared.errors import PipelineError
        from worker.document import utc_now
        from worker.volumes import write_json
        report = {'requestId': config.request_id, 'status': 'FAILED', 'failed': 1,
                  'needsGpu': 0, 'succeeded': 0, 'finishedAt': utc_now(),
                  'error': str(exc) if isinstance(exc, (PipelineError, ValueError)) else type(exc).__name__}
        if not config.dry_run:
            try:
                write_json(Path(config.state_volume) / 'runs' / config.request_id / 'report.json', report)
            except Exception:
                pass  # Permissions may be the failure; stdout still provides the safe summary.
    print('MARKER_SUMMARY=' + json.dumps(report, separators=(',', ':')), flush=True)
    return 1 if report['failed'] or report['needsGpu'] else 0

if __name__ == '__main__':
    raise SystemExit(main())
