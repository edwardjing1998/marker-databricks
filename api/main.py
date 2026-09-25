"""Keyless development gateway. Expose only through controlled private access."""
import logging
import json
import os
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from api.databricks_client import DatabricksAPI, RemoteError
from shared.paths import digest, roots, scoped_prefix


@asynccontextmanager
async def lifespan(app):
    logging.getLogger(__name__).warning(
        'Client authentication is disabled. Use the private OpenShift port-forward; do not expose an unprotected Route.')
    app.state.gpu_enabled = os.getenv('GPU_FALLBACK_ENABLED', 'false').lower() == 'true'
    app.state.job_id = int(os.environ['DATABRICKS_JOB_ID'])
    if app.state.job_id <= 0:
        raise RuntimeError('DATABRICKS_JOB_ID must be positive')
    app.state.source, app.state.output = roots(os.getenv('AZURE_STORAGE_SOURCE_PREFIX', 'source/'), os.getenv('AZURE_STORAGE_OUTPUT_PREFIX', 'generated/'))
    app.state.client = DatabricksAPI()
    yield


app = FastAPI(title='marker-databricks', version='3.2.0', lifespan=lifespan,
              description='OAuth-backed keyless private-development REST gateway to a CPU-first serverless PySpark script with optional GPU fallback. Originals are not modified.')


@app.middleware('http')
async def reject_cross_origin_mutations(request: Request, call_next):
    if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'} and request.url.path.startswith('/api/'):
        origin = request.headers.get('origin')
        expected_origin = f'{request.url.scheme}://{request.headers.get("host", "")}'
        if (origin and origin != expected_origin) or request.headers.get('sec-fetch-site') == 'cross-site':
            return JSONResponse(status_code=403, content={'detail': 'Cross-origin browser requests are disabled for this private development API'})
    return await call_next(request)


@app.exception_handler(RemoteError)
async def remote_error(request, exception):
    return JSONResponse(status_code=502, content={'detail': str(exception)})


@app.get('/health/live', include_in_schema=False)
@app.get('/health/ready', include_in_schema=False)
def health():
    # Readiness is local; it deliberately does not trigger a remote job or billable query.
    return {'status': 'UP', 'application': 'marker-databricks'}


@app.post('/api/storage-documents/process', status_code=202)
def process(request: Request,
            sourcePrefix: str | None = None, outputPrefix: str | None = None,
            maxFiles: int = Query(1, ge=1, le=100), overwrite: bool = False,
            replaceExisting: bool = False, dryRun: bool = False,
            pagesPerChunk: int = Query(2, ge=1, le=100),
            forceOcr: bool = False, dropHandwriting: bool = False,
            computeMode: str = Query('auto', pattern='^(auto|cpu|gpu)$'),
            mode: str | None = Query(None, deprecated=True),
            idempotency_key: str | None = Header(None, alias='Idempotency-Key')):
    if computeMode == 'gpu' and not request.app.state.gpu_enabled:
        raise HTTPException(400, 'GPU processing is disabled by deployment policy')
    if mode is not None:
        raise HTTPException(400, 'Hosted API mode was removed. Omit mode; use forceOcr, pagesPerChunk, and dropHandwriting')
    try:
        source = scoped_prefix(sourcePrefix, request.app.state.source)
        output = scoped_prefix(outputPrefix, request.app.state.output)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if replaceExisting and not overwrite:
        raise HTTPException(400, 'replaceExisting=true requires overwrite=true')
    key = idempotency_key or uuid.uuid4().hex
    if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,64}', key):
        raise HTTPException(400, 'Idempotency-Key must be 1-64 ASCII letters, digits, or ._:-')
    params = {'source_prefix': source, 'output_prefix': output, 'max_files': str(maxFiles),
              'overwrite': str(overwrite).lower(), 'replace_existing': str(replaceExisting).lower(),
              'dry_run': str(dryRun).lower(), 'pages_per_chunk': str(pagesPerChunk),
              'force_ocr': str(forceOcr).lower(), 'drop_handwriting': str(dropHandwriting).lower(), 'compute_mode': computeMode}
    token = digest([request.app.state.job_id, key, params])
    params['request_id'] = token[:32]
    result = request.app.state.client.start(request.app.state.job_id, params, token)
    run_id = result['run_id']
    return {'status': 'ACCEPTED', 'runId': run_id, 'requestId': params['request_id'],
            'idempotencyKey': key, 'dryRun': dryRun,
            'statusPath': f'/api/storage-documents/runs/{run_id}',
            'reportBlob': None if dryRun else f'_marker_jobs/runs/{params["request_id"]}/report.json',
            'note': 'Accepted is not conversion success. Poll statusPath. CPU-first PySpark coordination; GPU fallback is explicit and deployment-policy controlled. No notebook or hosted conversion API is used.'}


@app.get('/api/storage-documents/runs/{run_id}')
def run_status(request: Request, run_id: int):
    if run_id <= 0:
        raise HTTPException(400, 'Invalid run ID')
    client = request.app.state.client
    run = client.run(run_id)
    if int(run.get('job_id', -1)) != request.app.state.job_id:
        raise HTTPException(404, 'Run does not belong to this application job')
    state = run.get('state', {})
    out = {'runId': run_id, 'lifeCycleState': state.get('life_cycle_state'),
           'resultState': state.get('result_state'), 'runPageUrl': run.get('run_page_url'),
           'tasks': [{'taskKey': t.get('task_key'), 'state': t.get('state')} for t in run.get('tasks', [])]}
    parameters = {p.get('name'): p.get('value', p.get('default')) for p in run.get('job_parameters', [])}
    rid = parameters.get('request_id', '')
    if re.fullmatch(r'[a-f0-9]{32}', rid or '') and parameters.get('dry_run') != 'true':
        out['reportBlob'] = f'_marker_jobs/runs/{rid}/report.json'
    if state.get('life_cycle_state') in ('TERMINATED', 'SKIPPED', 'INTERNAL_ERROR'):
        for task in run.get('tasks', []):
            if task.get('task_key') == 'process_documents_cpu':
                result = client.output(task['run_id'])
                for line in reversed((result.get('logs') or '').splitlines()):
                    if line.startswith('MARKER_SUMMARY='):
                        try:
                            out['summary'] = json.loads(line.split('=', 1)[1])
                        except ValueError:
                            out['summaryUnavailable'] = True
                        break
        if 'summary' not in out:
            out['summaryUnavailable'] = True
            out['note'] = 'Script logs may be unavailable/truncated. Read reportBlob or the Databricks run logs.'
    return out
