"""Real lightweight subprocess tests; no Marker, Spark, or cloud inference."""
from dataclasses import replace
from pathlib import Path
import pytest
from worker import cpu_runner
from worker.local_marker import LocalMarker
from shared.errors import CpuResourceLimit, PipelineError


def launch(monkeypatch, tmp_path, cfg, script, **changes):
    root = tmp_path / 'fake-runtime'
    (root / 'jobs').mkdir(parents=True)
    (root / 'jobs' / 'infer_one.py').write_text(script)
    work = tmp_path / 'work'; work.mkdir()
    monkeypatch.setattr(cpu_runner, 'ROOT', root)
    cpu_runner.run_cpu(tmp_path / 'source.pdf', work, tmp_path / 'checkpoints',
                      replace(cfg, **changes), 'cpu', 'fingerprint')
    return work


def test_process_is_forced_cpu_and_has_no_launcher_token(monkeypatch, tmp_path, cfg):
    monkeypatch.setenv('DATABRICKS_TOKEN', 'private-value')
    script = "import os; assert os.environ['TORCH_DEVICE']=='cpu'; assert os.environ['CUDA_VISIBLE_DEVICES']==''; assert 'DATABRICKS_TOKEN' not in os.environ"
    launch(monkeypatch, tmp_path, cfg, script)


def test_timeout_becomes_resource_error(monkeypatch, tmp_path, cfg):
    with pytest.raises(CpuResourceLimit, match='timeout'):
        launch(monkeypatch, tmp_path, cfg, 'import time; time.sleep(10)', cpu_timeout_seconds=0.1)


def test_unknown_exit_not_classified_as_gpu_retry(monkeypatch, tmp_path, cfg):
    with pytest.raises(PipelineError, match='without a classified error') as error:
        launch(monkeypatch, tmp_path, cfg, 'raise SystemExit(3)')
    assert not isinstance(error.value, CpuResourceLimit)


def test_device_default_is_cpu():
    assert LocalMarker().signature()['device'] == 'cpu'
    assert LocalMarker(device='cuda').signature()['device'] == 'cuda'
    with pytest.raises(ValueError):
        LocalMarker(device='auto')
