"""Notebook-free AI Runtime entry point. PyTorch inference, NOT distributed Spark."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker.config import Config
from worker.document import convert_and_publish, inspect_source
from worker.local_marker import LocalMarker
from worker.volumes import read_json, write_json, safe_child
from shared.errors import PipelineError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    config = Config(**manifest['config']).validate()
    expected = Path(config.state_volume) / 'runs' / config.request_id / 'gpu-manifest.json'
    if Path(args.manifest) != expected or manifest.get('schema') != 'marker-gpu-manifest/v3':
        raise PipelineError('Unexpected GPU manifest')
    if not config.gpu_enabled or config.compute_mode == 'cpu' or config.dry_run:
        raise PipelineError('GPU inference is not authorized by this request')
    engine = LocalMarker(config.model_cache_dir, device='cuda')
    engine.preflight()  # Never silently run on a CPU in a GPU task.
    report = {'requestId': config.request_id, 'device': 'cuda', 'results': []}
    target = expected.with_name('gpu-report.json')
    for item in manifest['records']:
        try:
            source = safe_child(config.source_volume, item['relative'])
            relative_prefix = config.source_prefix[len(config.source_root):]
            if not item['relative'].startswith(relative_prefix):
                raise PipelineError('GPU manifest source is outside request prefix')
            record = inspect_source(config, item['relative'])
            result = convert_and_publish(config, record, 'cuda', engine=engine)
        except Exception as exc:
            result = {'sourceBlob': config.source_root + item['relative'], 'status': 'FAILED',
                      'error': str(exc) if isinstance(exc, PipelineError) else type(exc).__name__}
        report['results'].append(result)
        write_json(target, report)
    print('MARKER_GPU_SUMMARY=' + json.dumps(report, separators=(',', ':')), flush=True)
    return 1 if any(r['status'] == 'FAILED' for r in report['results']) else 0

if __name__ == '__main__':
    raise SystemExit(main())
