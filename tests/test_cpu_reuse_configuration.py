"""Opt-in CPU reuse configuration; original subprocess mode stays default."""
from dataclasses import replace
from worker.config import Config
from shared.defaults import FIXED_DEFAULTS


def test_cpu_reuse_is_off_by_default(cfg):
    assert cfg.cpu_model_reuse is False
    assert cfg.cpu_batch_size == 5
    assert cfg.cpu_torch_threads == 4


def test_cpu_reuse_validates_limits(cfg):
    import pytest
    with pytest.raises(ValueError):
        replace(cfg, cpu_batch_size=0).validate(require_volumes=False)
    with pytest.raises(ValueError):
        replace(cfg, cpu_torch_threads=0).validate(require_volumes=False)


def test_defaults_provide_reuse_flags():
    assert FIXED_DEFAULTS['CPU_MODEL_REUSE'] == 'false'
    assert FIXED_DEFAULTS['CPU_BATCH_SIZE'] == '5'
