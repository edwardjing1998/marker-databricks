"""Validated CPU-first routing and Unity Catalog configuration."""
from dataclasses import dataclass
from pathlib import Path
import re
import uuid
from urllib.parse import urlsplit
from shared.paths import roots, scoped_prefix, boolean


def volume_path(value):
    p = Path(value)
    if not value.startswith('/Volumes/') or len(p.parts) < 5 or any(x in ('..', '.') for x in value.split('/')):
        raise ValueError('Configure an absolute /Volumes/catalog/schema/volume path')
    if any(c in value for c in ('\n', '\r', '\x00')):
        raise ValueError('Invalid volume path')
    return str(p)


@dataclass
class Config:
    source_volume: str
    output_volume: str
    state_volume: str
    source_root: str = 'source/'
    output_root: str = 'generated/'
    source_prefix: str = 'source/'
    output_prefix: str = 'generated/'
    max_files: int = 1
    max_scan: int = 10000
    overwrite: bool = False
    replace_existing: bool = False
    dry_run: bool = True
    force_ocr: bool = False
    drop_handwriting: bool = False
    pages_per_chunk: int = 2
    compute_mode: str = 'auto'
    gpu_enabled: bool = False
    cpu_max_pages: int = 20
    cpu_max_file_mb: int = 20
    cpu_timeout_seconds: int = 900
    cpu_max_rss_mb: int = 6000
    max_file_mb: int = 200
    max_document_pages: int = 5000
    gpu_timeout_seconds: int = 7200
    gpu_wait_seconds: int = 7800
    accelerator: str = 'GPU_1xA10'
    environment_version: str = '6'
    workspace_host: str = ''
    workspace_release: str = ''
    secret_scope: str = 'marker-databricks'
    gpu_launch_auth: str = 'oauth-m2m'
    release_id: str = 'manual'
    cache_revision: str = 'cpu-first-v3'
    request_id: str = ''
    model_cache_dir: str = '/tmp/marker-models-v1.10.2'

    def validate(self, require_volumes=True):
        self.source_root, self.output_root = roots(self.source_root, self.output_root)
        self.source_prefix = scoped_prefix(self.source_prefix, self.source_root)
        self.output_prefix = scoped_prefix(self.output_prefix, self.output_root)
        self.request_id = self.request_id or uuid.uuid4().hex
        if not re.fullmatch('[a-f0-9]{32}', self.request_id):
            raise ValueError('Invalid request ID')
        if self.compute_mode not in ('auto', 'cpu', 'gpu'):
            raise ValueError('compute_mode must be auto, cpu, or gpu')
        if self.compute_mode == 'gpu' and not self.gpu_enabled:
            raise ValueError('GPU use is disabled by deployment policy')
        for value, lo, hi in ((self.max_files, 1, 100), (self.max_scan, 1, 100000),
                             (self.pages_per_chunk, 1, 100), (self.cpu_max_pages, 1, 10000),
                             (self.cpu_max_file_mb, 1, 1024), (self.max_file_mb, 1, 1024),
                             (self.cpu_timeout_seconds, 1, 21600), (self.cpu_max_rss_mb, 256, 64000),
                             (self.max_document_pages, 1, 10000), (self.gpu_timeout_seconds, 300, 86400)):
            if not lo <= value <= hi:
                raise ValueError('A processing safeguard is outside the permitted range')
        if self.gpu_wait_seconds <= self.gpu_timeout_seconds:
            raise ValueError('GPU_WAIT_SECONDS must exceed GPU_TIMEOUT_SECONDS')
        if self.replace_existing and not self.overwrite:
            raise ValueError('replace_existing requires overwrite')
        if self.accelerator not in ('GPU_1xA10', 'GPU_1xH100'):
            raise ValueError('Only one A10 or one H100 is supported')
        if self.environment_version not in ('5', '6'):
            raise ValueError('Choose Standard environment 5 or 6 and validate dependencies')
        if not re.fullmatch('[A-Za-z0-9_.-]{1,64}', self.cache_revision):
            raise ValueError('Invalid cache revision')
        if not re.fullmatch('[A-Za-z0-9_.-]{1,64}', self.release_id):
            raise ValueError('Invalid release ID')
        if self.gpu_launch_auth != 'oauth-m2m':
            raise ValueError('GPU_LAUNCH_AUTH must be oauth-m2m in this release')
        if self.gpu_enabled:
            host = urlsplit(self.workspace_host)
            if host.scheme != 'https' or not host.hostname or host.path not in ('', '/') or host.username or host.query:
                raise ValueError('Invalid Databricks workspace origin')
            if not self.workspace_release.startswith('/Workspace/Shared/marker-databricks/releases/'):
                raise ValueError('Invalid immutable workspace release path')
        if require_volumes:
            for name in ('source_volume', 'output_volume', 'state_volume'):
                setattr(self, name, volume_path(getattr(self, name)))
        paths = [Path(self.source_volume), Path(self.output_volume), Path(self.state_volume)]
        if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i+1:]):
            raise ValueError('Input, output, and state volume paths must not overlap')
        return self

    @classmethod
    def from_values(cls, fixed, params):
        names = {
            'source_volume': 'SOURCE_VOLUME_PATH', 'output_volume': 'OUTPUT_VOLUME_PATH',
            'state_volume': 'STATE_VOLUME_PATH', 'source_root': 'AZURE_STORAGE_SOURCE_PREFIX',
            'output_root': 'AZURE_STORAGE_OUTPUT_PREFIX', 'max_scan': 'MAX_SCAN_FILES',
            'cpu_max_pages': 'CPU_MAX_PAGES', 'cpu_max_file_mb': 'CPU_MAX_FILE_MB',
            'cpu_timeout_seconds': 'CPU_TIMEOUT_SECONDS', 'cpu_max_rss_mb': 'CPU_MAX_RSS_MB',
            'max_file_mb': 'MAX_FILE_MB', 'max_document_pages': 'MAX_DOCUMENT_PAGES',
            'gpu_timeout_seconds': 'GPU_TIMEOUT_SECONDS', 'gpu_wait_seconds': 'GPU_WAIT_SECONDS',
            'accelerator': 'DATABRICKS_GPU_ACCELERATOR', 'environment_version': 'DATABRICKS_ENVIRONMENT_VERSION',
            'workspace_host': 'DATABRICKS_HOST', 'workspace_release': 'WORKSPACE_RELEASE_PATH',
            'secret_scope': 'DATABRICKS_SECRET_SCOPE', 'gpu_launch_auth': 'GPU_LAUNCH_AUTH',
            'release_id': 'RELEASE_ID', 'cache_revision': 'MARKER_CACHE_REVISION',
            'model_cache_dir': 'MODEL_CACHE_DIR', 'gpu_enabled': 'GPU_FALLBACK_ENABLED',
        }
        values = {key: fixed[val] for key, val in names.items()}
        values.update(params)
        ints = {'max_scan', 'max_files', 'pages_per_chunk', 'cpu_max_pages', 'cpu_max_file_mb',
                'cpu_timeout_seconds', 'cpu_max_rss_mb', 'max_file_mb', 'max_document_pages',
                'gpu_timeout_seconds', 'gpu_wait_seconds'}
        bools = {'gpu_enabled', 'overwrite', 'replace_existing', 'dry_run', 'force_ocr', 'drop_handwriting'}
        return cls(**{k: int(v) if k in ints else boolean(v) if k in bools else v for k, v in values.items()}).validate()


def route_document(config, *, pages, size):
    """Explicit application policy, never a Databricks automatic upgrade."""
    if config.compute_mode == 'gpu':
        return 'gpu', 'requested_gpu'
    if config.compute_mode == 'auto' and pages > config.cpu_max_pages:
        return 'gpu', 'page_threshold'
    if config.compute_mode == 'auto' and size > config.cpu_max_file_mb * 1024 * 1024:
        return 'gpu', 'size_threshold'
    return 'cpu', 'cpu_first'
