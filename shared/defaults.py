"""Deployment configuration is fixed; request parameters are separately validated."""
FIXED_DEFAULTS = {
    'AZURE_STORAGE_ENDPOINT': 'https://chat8gpteducationaccount.blob.core.windows.net',
    'AZURE_STORAGE_CONTAINER': 'education',
    'AZURE_STORAGE_SOURCE_PREFIX': 'source/',
    'AZURE_STORAGE_OUTPUT_PREFIX': 'generated/',
    'SOURCE_VOLUME_PATH': '', 'OUTPUT_VOLUME_PATH': '', 'STATE_VOLUME_PATH': '',
    'DATABRICKS_HOST': '', 'DATABRICKS_SECRET_SCOPE': 'marker-databricks',
    'GPU_LAUNCH_AUTH': 'oauth-m2m', 'GPU_FALLBACK_ENABLED': 'false',
    'DATABRICKS_GPU_ACCELERATOR': 'GPU_1xA10',
    'DATABRICKS_ENVIRONMENT_VERSION': '6',
    'CPU_MAX_PAGES': '20', 'CPU_MAX_FILE_MB': '20',
    'CPU_TIMEOUT_SECONDS': '900', 'CPU_MAX_RSS_MB': '6000',
    'MAX_FILE_MB': '200', 'MAX_DOCUMENT_PAGES': '5000',
    'GPU_TIMEOUT_SECONDS': '7200', 'GPU_WAIT_SECONDS': '7800',
    'MAX_SCAN_FILES': '10000', 'MARKER_CACHE_REVISION': 'cpu-first-v3',
    'MODEL_CACHE_DIR': '/tmp/marker-models-v1.10.2',
    'RELEASE_ID': 'manual', 'WORKSPACE_RELEASE_PATH': '',
}
PARAM_DEFAULTS = {
    'source_prefix': 'source/', 'output_prefix': 'generated/', 'max_files': '1',
    'overwrite': 'false', 'replace_existing': 'false', 'dry_run': 'true',
    'pages_per_chunk': '2', 'force_ocr': 'false', 'drop_handwriting': 'false',
    'compute_mode': 'auto', 'request_id': '',
}
