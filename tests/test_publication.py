import base64
from pathlib import Path
import pytest
from shared.errors import PipelineError, CpuResourceLimit
from worker.document import convert_and_publish, existing_output, destination
from worker.volumes import safe_child, RunLock, read_json

class Engine:
    def __init__(self,pages=3,fail=None):self.pages=pages;self.calls=[];self.fail=fail
    def signature(self):return {'fake':'test'}
    def page_count(self,p):return self.pages
    def convert_chunk(self,p,pages,**kw):
        self.calls.append(pages)
        if self.fail is not None and pages[0]>=self.fail:raise PipelineError('deliberate failure')
        return {'pages':pages,'markdown':'Question ![diagram](fig.png)',
            'images':{'fig.png':base64.b64encode(b'image-bytes').decode()},'removedBlocks':[]}


def source(cfg):
    p=Path(cfg.source_volume)/'book'/'a.pdf';p.parent.mkdir();p.write_bytes(b'%PDF-1.7 test snapshot')
    return {'relative':'book/a.pdf','bytes':p.stat().st_size,'pages':3}


def test_full_publication_and_relative_figures(cfg):
    result=convert_and_publish(cfg,source(cfg),'cpu',engine=Engine())
    target=Path(cfg.output_volume)/'book/a/content.md'
    assert result['status']=='SUCCEEDED'
    text=target.read_text()
    assert 'pageCount: 3' in text and 'source/book/a.pdf' in text
    assert 'figures/' in text and '(fig.png)' not in text
    assert len(list(target.parent.glob('figures/*/*/*.png')))==2


def test_failure_does_not_publish_partial_markdown(cfg):
    record=source(cfg)
    with pytest.raises(PipelineError):convert_and_publish(cfg,record,'cpu',engine=Engine(fail=2))
    assert not (Path(cfg.output_volume)/'book/a/content.md').exists()
    assert list(Path(cfg.state_volume).glob('checkpoints/*/*.json'))


def test_checkpoint_resume(cfg):
    record=source(cfg)
    with pytest.raises(PipelineError):convert_and_publish(cfg,record,'cpu',engine=Engine(fail=2))
    engine=Engine()
    result=convert_and_publish(cfg,record,'cpu',engine=engine)
    assert result['resumedChunks']==1 and engine.calls==[[2]]


def test_existing_skip(cfg):
    record=source(cfg);convert_and_publish(cfg,record,'cpu',engine=Engine())
    e=Engine();result=convert_and_publish(cfg,record,'cpu',engine=e)
    assert result['status']=='SKIPPED' and not e.calls


def test_other_provider_requires_explicit_replacement(cfg):
    record=source(cfg);target=destination(cfg,record['relative'])/'content.md'
    target.parent.mkdir(parents=True);target.write_text('Azure old output')
    cfg.overwrite=True
    with pytest.raises(PipelineError):convert_and_publish(cfg,record,'cpu',engine=Engine())
    assert target.read_text()=='Azure old output'


def test_explicit_replacement_keeps_backup(cfg):
    record=source(cfg);target=destination(cfg,record['relative'])/'content.md'
    target.parent.mkdir(parents=True);target.write_text('Azure old output')
    cfg.overwrite=True;cfg.replace_existing=True
    convert_and_publish(cfg,record,'cpu',engine=Engine())
    backups=list(target.parent.glob('marker/*/previous-content.md'))
    assert backups[0].read_text()=='Azure old output'


def test_changed_source_rejected(cfg):
    record=source(cfg)
    class Changing(Engine):
        def convert_chunk(self,*a,**kw):
            (Path(cfg.source_volume)/'book/a.pdf').write_bytes(b'%PDF-CHANGED')
            return super().convert_chunk(*a,**kw)
    with pytest.raises(PipelineError,match='Source changed'):
        convert_and_publish(cfg,record,'cpu',engine=Changing())


def test_selected_subprefix_keeps_book_path(cfg):
    cfg.source_prefix='source/book/'
    assert destination(cfg,'book/a.pdf')==Path(cfg.output_volume)/'book/a'

@pytest.mark.parametrize('name',['../x','/tmp/x','a/../../x','a\\x','a//x'])
def test_unsafe_paths(cfg,name):
    with pytest.raises(PipelineError):safe_child(cfg.output_volume,name)


def test_lock_cleanup(cfg):
    with RunLock(cfg.state_volume,cfg.request_id):pass
    assert not (Path(cfg.state_volume)/'locks/cpu-first.lock').exists()


def test_uncertain_gpu_submission_keeps_lock(cfg):
    with RunLock(cfg.state_volume,cfg.request_id) as lock:lock.hold(phase='GPU_SUBMITTING')
    with pytest.raises(PipelineError):
        with RunLock(cfg.state_volume,'b'*32):pass


def test_confirmed_child_terminal_releases_lock(cfg):
    with RunLock(cfg.state_volume,cfg.request_id) as lock:
        lock.hold(gpuRunId=123);lock.clear_child()
    assert not (Path(cfg.state_volume)/'locks/cpu-first.lock').exists()
