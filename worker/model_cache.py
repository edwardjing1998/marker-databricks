
"""Persistent, versioned model cache snapshot for Databricks Serverless CPU workers.

Keep live inference caches on local disk. Read/write immutable uncompressed tar
snapshots on a UC Volume; snapshots are optional and failures fall back to download.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import uuid


def _log(message):
    print(f"[MARKER][MODEL-CACHE] {message}", flush=True)


def _digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def snapshot_path(config):
    # Both fields are validated by Config.validate; use only a fixed subfolder.
    return (Path(config.state_volume) / 'marker-model-snapshots' /
            f'marker-1.10.2-{config.cache_revision}.tar')


def _paths(config):
    return {'marker': Path(config.model_cache_dir),
            'huggingface': Path(config.hf_cache_dir)}


def restore(config):
    """Restore only if an archive and a matching checksum are both present."""
    if not config.model_cache_snapshot_enabled:
        return False
    archive = snapshot_path(config)
    manifest = Path(str(archive) + '.json')
    if not (archive.is_file() and manifest.is_file()):
        _log('snapshot_missing; will download models')
        return False
    started = time.monotonic()
    try:
        metadata = json.loads(manifest.read_text())
        if (metadata.get('version') != 1 or metadata.get('bytes') != archive.stat().st_size
                or metadata.get('sha256') != _digest(archive)):
            raise ValueError('snapshot checksum/size mismatch')
        paths = _paths(config)
        # Keep extraction in local temp storage; do not execute/infer directly from Volume.
        with tempfile.TemporaryDirectory(prefix='marker-snapshot-') as temp:
            temp_dir = Path(temp)
            with tarfile.open(archive, 'r:') as tar:
                for member in tar:
                    p = PurePosixPath(member.name)
                    if (p.is_absolute() or '..' in p.parts or not p.parts
                            or p.parts[0] not in paths or
                            not (member.isfile() or member.isdir())):
                        raise ValueError('unsafe snapshot member')
                    target = temp_dir.joinpath(*p.parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = tar.extractfile(member)
                        if source is None:
                            raise ValueError('unreadable snapshot entry')
                        with source, open(target, 'wb') as dst:
                            shutil.copyfileobj(source, dst, 8 * 1024 * 1024)
            for key, destination in paths.items():
                staged = temp_dir / key
                if not staged.exists():
                    raise ValueError(f'missing {key} cache')
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copytree(staged, destination, dirs_exist_ok=True)
        _log(f'snapshot_restored seconds={time.monotonic()-started:.2f} bytes={archive.stat().st_size}')
        return True
    except (OSError, ValueError, tarfile.TarError, json.JSONDecodeError) as exc:
        _log(f'snapshot_restore_failed reason={type(exc).__name__}; falling back to downloads')
        return False


def publish(config):
    """Publish after successful model load. Concurrent publishers use immutable data."""
    if not config.model_cache_snapshot_enabled:
        return
    archive = snapshot_path(config)
    manifest = Path(str(archive) + '.json')
    if archive.exists() and manifest.exists():
        _log('snapshot_already_exists')
        return
    started = time.monotonic()
    tmp_archive = None
    tmp_manifest = None
    try:
        archive.parent.mkdir(parents=True, exist_ok=True)
        # Stage beside target for atomic final renames; never expose an incomplete manifest.
        suffix = '.' + uuid.uuid4().hex + '.tmp'
        tmp_archive = Path(str(archive) + suffix)
        tmp_manifest = Path(str(manifest) + suffix)
        with tarfile.open(tmp_archive, 'w:') as tar:
            for key, source in _paths(config).items():
                if source.exists():
                    tar.add(source, arcname=key, recursive=True)
        metadata = {'version': 1, 'sha256': _digest(tmp_archive),
                    'bytes': tmp_archive.stat().st_size}
        tmp_manifest.write_text(json.dumps(metadata))
        # Last writer wins only for the same model revision; checksum protects readers.
        os.replace(tmp_archive, archive)
        os.replace(tmp_manifest, manifest)
        _log(f'snapshot_published seconds={time.monotonic()-started:.2f} bytes={metadata["bytes"]}')
    except (OSError, tarfile.TarError) as exc:
        _log(f'snapshot_publish_failed reason={type(exc).__name__}; inference continues')
    finally:
        for tmp in (tmp_archive, tmp_manifest):
            if tmp is not None:
                tmp.unlink(missing_ok=True)
