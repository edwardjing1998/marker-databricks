"""Long-lived isolated CPU process. One Marker model instance per worker lifetime."""
import os
import faulthandler
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker.config import Config
from worker.document import build_document
from worker.local_marker import LocalMarker
from worker.model_cache import restore, publish
from worker.volumes import read_json, write_json
from shared.errors import CpuResourceLimit, PipelineError


def main():
    engine = None
    engine_key = None
    for line in sys.stdin:
        request_path = line.strip()
        if not request_path:
            continue
        request = read_json(request_path)
        work = Path(request['work'])
        status = work / 'cpu-batch-status.json'
        started = time.monotonic()
        print('[MARKER][CPU-BATCH] request_started', flush=True)
        try:
            config = Config(**request['config'])
            key = (config.model_cache_dir, config.hf_cache_dir, config.cpu_torch_threads)
            if engine is None or engine_key != key:
                os.environ['CPU_TORCH_THREADS'] = str(config.cpu_torch_threads)
                restore(config)
                engine = LocalMarker(config.model_cache_dir, device='cpu',
                                     hf_cache_dir=config.hf_cache_dir)
                engine_key = key
                print('[MARKER][PERF] new_engine_created (models load lazily)', flush=True)
            else:
                print('[MARKER][PERF] reusing_existing_engine', flush=True)
            conversion_started = time.monotonic()
            print('[MARKER][PERF] build_document_started', flush=True)
            # Repeated stack samples expose which Python call is active during a stall.
            # Enabled only around conversion, to avoid noisy idle-process traces.
            faulthandler.enable(file=sys.stderr)
            faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
            try:
                build_document(request['local'], request['work'], request['checkpoint_dir'],
                               config, 'cpu', request['fingerprint'], engine=engine)
            finally:
                faulthandler.cancel_dump_traceback_later()
            print(f'[MARKER][PERF] build_document_seconds={time.monotonic()-conversion_started:.2f}', flush=True)
            if engine.models is not None:
                publish(config)
            write_json(status, {'status': 'SUCCEEDED'})
        except (CpuResourceLimit, MemoryError) as exc:
            write_json(status, {'status': 'RESOURCE', 'message': str(exc) or 'CPU memory limit'})
            engine = None
        except Exception as exc:
            message = str(exc) if isinstance(exc, PipelineError) else f'{type(exc).__name__}: {exc}'
            write_json(status, {'status': 'FAILED', 'message': message})
        print(f'[MARKER][CPU-BATCH] request_completed seconds={time.monotonic()-started:.2f}', flush=True)


if __name__ == '__main__':
    main()
