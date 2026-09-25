"""Isolated CPU inference process; never imports or starts Spark."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker.config import Config
from worker.document import build_document
from worker.volumes import read_json, write_json
from shared.errors import CpuResourceLimit, PipelineError


def main():
    request = read_json(sys.argv[1])
    config = Config(**request['config'])
    try:
        build_document(request['local'], request['work'], request['checkpoint_dir'],
                       config, request['device'], request['fingerprint'])
    except (CpuResourceLimit, MemoryError):
        write_json(Path(request['work']) / 'error.json', {'kind': 'resource', 'message': 'CPU memory limit'})
        return 75
    except Exception as exc:
        message = str(exc) if isinstance(exc, PipelineError) else type(exc).__name__
        write_json(Path(request['work']) / 'error.json', {'kind': 'error', 'message': message})
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
