"""Monitor an isolated CPU process; only resource/time limits can request a GPU.

RSS monitoring is a soft safeguard, not a cgroup limit. A whole task/host OOM or
platform kill cannot be recovered by this Python coordinator.
"""
from dataclasses import asdict
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil
from shared.errors import CpuResourceLimit, PipelineError
from worker.volumes import write_json, read_json

ROOT = Path(__file__).resolve().parents[1]


def run_cpu(local, work, checkpoint_dir, config, device, fingerprint):
    if device != 'cpu':
        raise ValueError('CPU runner cannot use a GPU')
    request = Path(work) / 'inference-request.json'
    write_json(request, {'local': str(local), 'work': str(work), 'checkpoint_dir': str(checkpoint_dir),
                         'config': asdict(config), 'device': device, 'fingerprint': fingerprint})
    env = dict(os.environ)
    env.update({'CUDA_VISIBLE_DEVICES': '', 'TORCH_DEVICE': 'cpu', 'TOKENIZERS_PARALLELISM': 'false'})
    # Do not inherit a task launcher token into the model subprocess.
    for key in ('DATABRICKS_TOKEN', 'DATABRICKS_CLIENT_SECRET'):
        env.pop(key, None)
    with (Path(work) / 'inference.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, str(ROOT / 'jobs/infer_one.py'), str(request)],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        deadline = time.monotonic() + config.cpu_timeout_seconds
        reason = None
        try:
            while process.poll() is None:
                if time.monotonic() > deadline:
                    reason = 'CPU timeout, including model setup'
                    break
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running())
                    if rss > config.cpu_max_rss_mb * 1024 * 1024:
                        reason = 'CPU RSS limit exceeded'
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if reason:
            raise CpuResourceLimit(reason)
        if process.returncode == 0:
            return
        error_path = Path(work) / 'error.json'
        if error_path.exists():
            error = read_json(error_path)
            if error.get('kind') == 'resource':
                raise CpuResourceLimit(error['message'])
            raise PipelineError('CPU Marker failed: ' + error.get('message', 'unknown error'))
        # A kill is ambiguous: do not automatically spend money on a GPU.
        raise PipelineError(f'CPU inference exited {process.returncode} without a classified error; inspect runtime logs')
