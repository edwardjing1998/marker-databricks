"""CPU-first coordinator, called by a real PySpark Python-script task."""
from contextlib import nullcontext
from pathlib import Path
from shared.errors import PipelineError, CpuResourceLimit
from worker.config import route_document
from worker.cpu_runner import run_cpu
from worker.discovery import discover
from worker.document import convert_and_publish, existing_output, inspect_source, utc_now
from worker.gpu_launch import run_gpu
from worker.volumes import RunLock, probe_volume, write_json


def run_pipeline(config, spark, *, discover_fn=discover, inspect_fn=inspect_source,
                 convert_fn=convert_and_publish, gpu_fn=run_gpu):
    report = {'requestId': config.request_id, 'status': 'RUNNING', 'computeMode': config.compute_mode,
              'startedAt': utc_now(), 'results': [], 'gpuRunId': None, 'succeeded': 0,
              'failed': 0, 'needsGpu': 0, 'skipped': 0,
              'reportBlob': '_marker_jobs/runs/' + config.request_id + '/report.json'}
    report_path = Path(config.state_volume) / 'runs' / config.request_id / 'report.json'
    def save():
        if not config.dry_run:
            write_json(report_path, report)
    if not config.dry_run:
        probe_volume(config.output_volume)
        probe_volume(config.state_volume)
    context = nullcontext(None) if config.dry_run else RunLock(config.state_volume, config.request_id)
    with context as lock:
        save()
        records = discover_fn(spark, config)
        pending = []
        attempted = 0
        for item in records:
            if attempted >= config.max_files:
                break
            source_blob = config.source_root + item['relative']
            try:
                attempted += 1
                if existing_output(config, item['relative']) == 'SKIP':
                    attempted -= 1
                    report['skipped'] += 1
                    if len(report['results']) < 200:
                        report['results'].append({'sourceBlob': source_blob, 'status': 'SKIPPED'})
                    continue
                record = inspect_fn(config, item['relative'])
                route, reason = route_document(config, pages=record['pages'], size=record['bytes'])
                if config.dry_run:
                    report['results'].append({'sourceBlob': source_blob, 'status': 'WOULD_PROCESS',
                                              'route': route, 'reason': reason,
                                              'gpuAllowed': config.gpu_enabled and config.compute_mode != 'cpu'})
                    continue
                if route == 'cpu':
                    try:
                        result = convert_fn(config, record, 'cpu', runner=run_cpu)
                        report['results'].append(result)
                        continue
                    except CpuResourceLimit as exc:
                        reason = str(exc)
                pending.append({**record, 'routingReason': reason})
            except Exception as exc:
                # Corruption, permissions, package/model download errors do NOT trigger GPU fallback.
                message = str(exc) if isinstance(exc, (PipelineError, ValueError)) else type(exc).__name__
                report['results'].append({'sourceBlob': source_blob, 'status': 'FAILED', 'error': message})
            finally:
                save()
        if pending:
            if config.compute_mode == 'cpu' or not config.gpu_enabled:
                report['results'].extend({'sourceBlob': config.source_root + r['relative'],
                    'status': 'NEEDS_GPU', 'reason': r['routingReason']} for r in pending)
            else:
                try:
                    report['results'].extend(gpu_fn(config, pending, spark, lock, report, save))
                except Exception as exc:
                    report['gpuError'] = str(exc) if isinstance(exc, PipelineError) else type(exc).__name__
                    report['results'].extend({'sourceBlob': config.source_root + r['relative'],
                        'status': 'FAILED', 'error': 'GPU handoff failed; see gpuError and child run'} for r in pending)
        report['succeeded'] = sum(r['status'] == 'SUCCEEDED' for r in report['results'])
        report['failed'] = sum(r['status'] == 'FAILED' for r in report['results'])
        report['needsGpu'] = sum(r['status'] == 'NEEDS_GPU' for r in report['results'])
        report['status'] = ('DRY_RUN_COMPLETE' if config.dry_run and not report['failed'] else
                            'COMPLETED_WITH_ERRORS' if report['failed'] else
                            'NEEDS_GPU' if report['needsGpu'] else 'COMPLETED')
        report['finishedAt'] = utc_now()
        save()
    return report
