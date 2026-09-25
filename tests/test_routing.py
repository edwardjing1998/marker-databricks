from dataclasses import replace
import pytest
from worker.config import route_document, volume_path
from shared.errors import CpuResourceLimit, PipelineError
from worker.pipeline import run_pipeline


def test_small_uses_cpu(cfg):
    assert route_document(cfg, pages=1, size=3000) == ('cpu', 'cpu_first')


def test_large_uses_explicit_threshold(cfg):
    assert route_document(cfg, pages=21, size=3000) == ('gpu', 'page_threshold')


def test_size_threshold(cfg):
    assert route_document(cfg, pages=1, size=21*1024*1024)[0] == 'gpu'


def test_cpu_mode_never_reroutes_by_size(cfg):
    cfg.compute_mode='cpu'
    assert route_document(cfg,pages=999,size=50*1024*1024)[0]=='cpu'


def test_gpu_mode_rejected_when_disabled(cfg):
    cfg.compute_mode='gpu'
    with pytest.raises(ValueError): cfg.validate(require_volumes=False)

@pytest.mark.parametrize('value', ['source/', '/tmp/x', '/Volumes/a/b/../x', '/Volumes/a'])
def test_bad_volume_path(value):
    with pytest.raises(ValueError): volume_path(value)


def test_proper_volume_path():
    assert volume_path('/Volumes/c/s/v') == '/Volumes/c/s/v'


def test_overlapping_paths_rejected(cfg):
    cfg.output_volume=cfg.source_volume
    with pytest.raises(ValueError):cfg.validate(require_volumes=False)


def calls(cfg, convert, gpu=None):
    def default_gpu(*args):
        raise AssertionError('GPU should not run')
    return run_pipeline(cfg,None,
        discover_fn=lambda *_:[{'relative':'a.pdf','bytes':40}],
        inspect_fn=lambda *_:{'relative':'a.pdf','bytes':40,'pages':1},
        convert_fn=convert,gpu_fn=gpu or default_gpu)


def test_cpu_success_never_launches_gpu(cfg):
    cfg.gpu_enabled=True
    result=calls(cfg,lambda *_a,**_k:{'sourceBlob':'source/a.pdf','status':'SUCCEEDED','device':'cpu'})
    assert result['succeeded']==1 and result['gpuRunId'] is None


def test_cpu_timeout_can_launch_gpu(cfg):
    cfg.gpu_enabled=True
    seen=[]
    def convert(*a,**k): raise CpuResourceLimit('timeout')
    def gpu(config,records,*args):
        seen.extend(records)
        return [{'sourceBlob':'source/a.pdf','status':'SUCCEEDED','device':'cuda'}]
    result=calls(cfg,convert,gpu)
    assert seen[0]['routingReason']=='timeout'
    assert result['succeeded']==1


def test_disabled_gpu_leaves_needs_gpu(cfg):
    def convert(*a,**k):raise CpuResourceLimit('RSS')
    result=calls(cfg,convert)
    assert result['status']=='NEEDS_GPU' and result['needsGpu']==1


def test_cpu_mode_resource_failure_no_gpu(cfg):
    cfg.compute_mode='cpu';cfg.gpu_enabled=True
    def convert(*a,**k):raise CpuResourceLimit('RSS')
    assert calls(cfg,convert)['status']=='NEEDS_GPU'


def test_network_error_not_gpu_retry(cfg):
    cfg.gpu_enabled=True
    def convert(*a,**k):raise PipelineError('model network error')
    result=calls(cfg,convert)
    assert result['failed']==1 and result['needsGpu']==0


def test_corrupt_input_not_gpu_retry(cfg):
    cfg.gpu_enabled=True
    def bad(*a): raise PipelineError('corrupt')
    result=run_pipeline(cfg,None,discover_fn=lambda *_:[{'relative':'a.pdf'}],inspect_fn=bad)
    assert result['failed']==1


def test_dry_run_does_not_infer_or_write(cfg):
    cfg.dry_run=True
    def bad(*a,**k):raise AssertionError('unexpected inference')
    result=calls(cfg,bad)
    assert result['status']=='DRY_RUN_COMPLETE'
    from pathlib import Path
    assert not list(Path(cfg.state_volume).iterdir())


def test_gpu_without_files_never_runs(cfg):
    result=run_pipeline(cfg,None,discover_fn=lambda *_:[])
    assert result['succeeded']==0 and result['gpuRunId'] is None
