"""Real local Spark tests. Skipped when PySpark is unavailable; never use cloud APIs."""
from pathlib import Path
import pytest
pyspark=pytest.importorskip('pyspark')
from pyspark.sql import SparkSession
from worker.discovery import document_metadata, discover
from shared.errors import PipelineError

@pytest.fixture(scope='module')
def spark():
    session=(SparkSession.builder.master('local[1]').appName('marker-offline-tests')
             .config('spark.ui.enabled','false').config('spark.sql.shuffle.partitions','1').getOrCreate())
    yield session
    session.stop()


def test_projects_out_binary_content(spark,cfg):
    Path(cfg.source_volume,'a.pdf').write_bytes(b'%PDF test')
    Path(cfg.source_volume,'b.PNG').write_bytes(b'png bytes')
    Path(cfg.source_volume,'ignore.txt').write_text('not a document')
    frame=document_metadata(spark,cfg.source_volume)
    assert 'content' not in frame.columns
    assert frame.count()==2
    assert {r.extension for r in frame.collect()}=={'pdf','png'}


def test_discovers_relative_hierarchy(spark,cfg):
    root=Path(cfg.source_volume)/'book';root.mkdir();(root/'one.pdf').write_bytes(b'fake')
    records=discover(spark,cfg)
    assert records[0]['relative']=='book/one.pdf'


def test_spark_collision_fails(spark,cfg):
    Path(cfg.source_volume,'same.pdf').write_bytes(b'pdf')
    Path(cfg.source_volume,'same.jpg').write_bytes(b'jpg')
    with pytest.raises(PipelineError,match='collide'):discover(spark,cfg)


def test_scan_limit_fails(spark,cfg):
    cfg.max_scan=1
    for name in ['one.pdf','two.pdf']:Path(cfg.source_volume,name).write_bytes(b'pdf')
    with pytest.raises(PipelineError,match='MAX_SCAN_FILES'):discover(spark,cfg)
