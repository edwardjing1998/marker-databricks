from pathlib import Path
import sys


def _project_root() -> Path:
    script_file = globals().get("__file__")

    if script_file:
        return Path(script_file).resolve().parents[1]

    # Databricks spark_python_task wrapper exposes the script path as `filename`
    databricks_filename = globals().get("filename")
    if databricks_filename:
        return Path(databricks_filename).resolve().parents[1]

    # Last-resort fallback
    return Path.cwd()


PROJECT_ROOT = _project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))