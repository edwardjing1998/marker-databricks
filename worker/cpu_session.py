"""Reusable isolated CPU inference process; per-document timeout and RSS safeguards."""
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


class CpuSession:
    def __init__(self, config):
        self.config = config
        self.process = None
        self.log = None
        self.completed = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        process = self.process
        self.process = None
        if process is not None:
            if process.poll() is None:
                try:
                    process.stdin.close()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        if self.log is not None:
            self.log.close()
            self.log = None

    def _start(self):
        env = dict(os.environ)
        env.update({'CUDA_VISIBLE_DEVICES': '', 'TORCH_DEVICE': 'cpu',
                    'TOKENIZERS_PARALLELISM': 'false',
                    'CPU_TORCH_THREADS': str(self.config.cpu_torch_threads)})
        for key in ('DATABRICKS_TOKEN', 'DATABRICKS_CLIENT_SECRET'):
            env.pop(key, None)
        self.log = open('/tmp/marker-cpu-batch-' + self.config.request_id + '.log', 'ab')
        self.process = subprocess.Popen(
            [sys.executable, str(ROOT / 'jobs/infer_batch.py')],
            stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
            env=env, start_new_session=True, text=True, bufsize=1,
        )
        self.completed = 0
        print('[MARKER][PERF] CPU batch worker started', flush=True)

    def run(self, local, work, checkpoint_dir, config, device, fingerprint):
        if device != 'cpu':
            raise ValueError('CPU session cannot run on GPU')
        if self.process is None or self.process.poll() is not None or self.completed >= config.cpu_batch_size:
            self.close()
            self._start()
        work = Path(work)
        request = work / 'inference-request.json'
        status = work / 'cpu-batch-status.json'
        write_json(request, {'local': str(local), 'work': str(work), 'checkpoint_dir': str(checkpoint_dir),
                             'config': asdict(config), 'device': 'cpu', 'fingerprint': fingerprint})
        started = time.monotonic()
        try:
            self.process.stdin.write(str(request) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise PipelineError('CPU batch worker exited before accepting request') from exc
        deadline = started + config.cpu_timeout_seconds
        try:
            while not status.exists():
                if self.process.poll() is not None:
                    raise PipelineError('CPU batch worker exited unexpectedly; inspect /tmp/marker-cpu-batch logs')
                if time.monotonic() > deadline:
                    raise CpuResourceLimit('CPU batch inference timeout')
                try:
                    parent = psutil.Process(self.process.pid)
                    rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)]
                              if p.is_running())
                    if rss > config.cpu_max_rss_mb * 1024 * 1024:
                        raise CpuResourceLimit('CPU batch RSS limit exceeded')
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.25)
            result = read_json(status)
            if result.get('status') == 'RESOURCE':
                raise CpuResourceLimit(result.get('message', 'CPU resource limit'))
            if result.get('status') != 'SUCCEEDED':
                raise PipelineError('CPU Marker failed: ' + result.get('message', 'unknown error'))
            self.completed += 1
            print(f'[MARKER][PERF] cpu_document_seconds={time.monotonic()-started:.2f} '
                  f'worker_document_count={self.completed}', flush=True)
        except BaseException:
            self.close()
            raise
