"""Deploy plain Python FILE assets and a serverless CPU spark_python_task."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import sys
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.databricks_client import DatabricksAPI
from shared.defaults import FIXED_DEFAULTS, PARAM_DEFAULTS
from shared.paths import boolean
from worker.config import Config


def fixed_config(release_path):
    fixed = {key: os.getenv(key) or default for key, default in FIXED_DEFAULTS.items()}
    fixed['RELEASE_ID'] = os.getenv('GITHUB_SHA') or 'manual'
    fixed['WORKSPACE_RELEASE_PATH'] = release_path
    return fixed


def job_settings(release_path):
    fixed = fixed_config(release_path)
    defaults = {**PARAM_DEFAULTS, 'source_prefix': fixed['AZURE_STORAGE_SOURCE_PREFIX'],
                'output_prefix': fixed['AZURE_STORAGE_OUTPUT_PREFIX'],
                'max_files': os.getenv('JOB_MAX_FILES') or '1',
                'dry_run': str(boolean(os.getenv('JOB_DEFAULT_DRY_RUN') or 'true')).lower(),
                'compute_mode': os.getenv('JOB_COMPUTE_MODE') or 'auto'}
    config = Config.from_values(fixed, defaults)
    timeout = int(os.getenv('JOB_TIMEOUT_SECONDS') or '21600')
    if not 600 <= timeout <= 86400:
        raise ValueError('JOB_TIMEOUT_SECONDS must be 600..86400')
    if config.gpu_enabled and timeout <= config.gpu_wait_seconds + config.cpu_timeout_seconds:
        raise ValueError('Parent job timeout must leave time for CPU inference and GPU completion')
    parameters = ['--config-file', release_path + '/deployment-config.json']
    for name in defaults:
        parameters.extend(['--' + name.replace('_', '-'), '{{job.parameters.' + name + '}}'])
    task = {'task_key': 'process_documents_cpu', 'max_retries': 0, 'retry_on_timeout': False,
            'disable_auto_optimization': True, 'timeout_seconds': timeout,
            'spark_python_task': {'python_file': release_path + '/jobs/process_documents.py',
                                  'source': 'WORKSPACE', 'parameters': parameters},
            'environment_key': 'marker_cpu'}
    settings = {
        'name': os.getenv('MARKER_JOB_NAME') or 'marker-databricks',
        'description': 'CPU-first PySpark discovery + local Marker. Explicit optional GPU child; no notebooks.',
        'max_concurrent_runs': 1, 'queue': {'enabled': True}, 'timeout_seconds': timeout,
        'tags': {'managed_by': 'marker-databricks', 'github_repo': os.getenv('GITHUB_REPOSITORY') or 'local',
                 'inference': 'marker-cpu-first'},
        'parameters': [{'name': k, 'default': v} for k,v in defaults.items()],
        'tasks': [task],
        'environments': [{'environment_key': 'marker_cpu', 'spec': {
            'environment_version': config.environment_version,
            'dependencies': ['-r ' + release_path + '/requirements-cpu.txt'],
        }}],
    }
    if os.getenv('JOB_SCHEDULE_CRON'):
        pause = os.getenv('JOB_SCHEDULE_PAUSE_STATUS') or 'PAUSED'
        if pause not in ('PAUSED', 'UNPAUSED'):
            raise ValueError('Invalid schedule pause status')
        settings['schedule'] = {'quartz_cron_expression': os.environ['JOB_SCHEDULE_CRON'],
            'timezone_id': os.getenv('JOB_SCHEDULE_TIMEZONE') or 'America/Los_Angeles', 'pause_status': pause}
    run_as = (os.getenv('DATABRICKS_RUN_AS_SERVICE_PRINCIPAL') or os.getenv('DATABRICKS_CLIENT_ID') or '').strip()
    if not run_as:
        raise ValueError('DATABRICKS_CLIENT_ID is required to set the job Run as service principal')
    settings['run_as'] = {'service_principal_name': run_as}
    return settings


def verify_cpu_job(actual, expected):
    settings = actual.get('settings', actual)
    tasks = settings.get('tasks', [])
    if len(tasks) != 1 or tasks[0].get('spark_python_task') != expected['tasks'][0]['spark_python_task']:
        raise RuntimeError('Serverless Python-script task was not saved as requested')
    if any(key in tasks[0] for key in ('notebook_task', 'new_cluster', 'existing_cluster_id')):
        raise RuntimeError('Unexpected notebook or classic-cluster configuration')
    if tasks[0].get('compute', {}).get('hardware_accelerator'):
        raise RuntimeError('CPU parent must not allocate a GPU')
    saved_run_as = settings.get('run_as', {}).get('service_principal_name') or actual.get('run_as_user_name')
    if saved_run_as != expected['run_as']['service_principal_name']:
        raise RuntimeError('Job Run as identity was not saved as the requested service principal')
    env = next((e for e in settings.get('environments', []) if e.get('environment_key') == 'marker_cpu'), None)
    if not env or env.get('spec') != expected['environments'][0]['spec']:
        raise RuntimeError('CPU environment configuration mismatch')


def upload_file(api, path, content):
    api_path = path.removeprefix('/Workspace')
    api.call('POST', '/api/2.0/workspace/mkdirs', body={'path': str(Path(api_path).parent)})
    api.call('POST', '/api/2.0/workspace/import', body={
        'path': api_path, 'format': 'AUTO', 'overwrite': False,
        'content': base64.b64encode(content).decode(),
    })
    status = api.call('GET', '/api/2.0/workspace/get-status', params={'path': api_path}, safe_retry=True)
    if status.get('object_type') != 'FILE':
        raise RuntimeError('Code was not uploaded as a plain FILE; refusing notebook substitution')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    revision = os.getenv('GITHUB_SHA') or 'manual'
    if not re.fullmatch('[A-Za-z0-9_.-]{1,64}', revision):
        raise ValueError('Invalid release ID')
    release = '/Workspace/Shared/marker-databricks/releases/' + revision + '-' + uuid.uuid4().hex[:8]
    settings = job_settings(release)
    if args.render_only:
        print(json.dumps(settings, indent=2))
        return
    api = DatabricksAPI()
    found, token = [], None
    while True:
        params = {'name': settings['name'], 'limit': 100}
        if token:
            params['page_token'] = token
        result = api.call('GET', '/api/2.2/jobs/list', params=params, safe_retry=True)
        found.extend(j for j in result.get('jobs', []) if j.get('settings', {}).get('name') == settings['name'])
        token = result.get('next_page_token')
        if not token:
            break
    if len(found) > 1:
        raise RuntimeError('Multiple jobs share the configured name')
    job_id = None
    if found:
        job_id = found[0]['job_id']
        current = api.call('GET', '/api/2.2/jobs/get', params={'job_id': job_id}, safe_retry=True)
        tags = current.get('settings', {}).get('tags', {})
        if tags.get('managed_by') != 'marker-databricks' or tags.get('github_repo') != settings['tags']['github_repo']:
            raise RuntimeError('Refusing to replace a job owned by another repository')
        active = api.call('GET', '/api/2.2/jobs/runs/list', params={'job_id': job_id, 'active_only': 'true', 'limit': 1}, safe_retry=True)
        if active.get('runs'):
            raise RuntimeError('Pause schedules and complete active/queued runs before deployment')
    for folder in ('api', 'shared', 'worker', 'jobs'):
        for path in sorted((ROOT / folder).glob('*.py')):
            upload_file(api, release + '/' + str(path.relative_to(ROOT)), path.read_bytes())
    for name in ('requirements-cpu.txt', 'requirements-gpu.txt'):
        upload_file(api, release + '/' + name, (ROOT / name).read_bytes())
    upload_file(api, release + '/deployment-config.json', json.dumps(fixed_config(release), indent=2).encode())
    if job_id:
        api.call('POST', '/api/2.2/jobs/reset', body={'job_id': job_id, 'new_settings': settings})
    else:
        job_id = api.call('POST', '/api/2.2/jobs/create', body=settings)['job_id']
    verify_cpu_job(api.call('GET', '/api/2.2/jobs/get', params={'job_id': job_id}, safe_retry=True), settings)
    record = {'job_id': job_id, 'workspace_release': release, 'serverless': True,
              'task_type': 'spark_python_task', 'initial_device': 'cpu', 'auth_type': 'oauth-m2m',
              'run_as_service_principal': settings['run_as']['service_principal_name'],
              'gpu_fallback_enabled': boolean(fixed_config(release)['GPU_FALLBACK_ENABLED'])}
    print(json.dumps(record))
    (ROOT / 'databricks/deployed-job.json').write_text(json.dumps(record, indent=2))
    if os.getenv('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
            out.write(f'job_id={job_id}\n')

if __name__ == '__main__':
    main()
