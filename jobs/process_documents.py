from pathlib import Path
import sys


print("[MARKER] process_documents.py STARTED", flush=True)


def _project_root() -> Path:
    script_file = globals().get("__file__")

    if script_file:
        root = Path(script_file).resolve().parents[1]
        print(
            f"[MARKER] project root from __file__: {root}",
            flush=True,
        )
        return root

    databricks_filename = globals().get("filename")

    if databricks_filename:
        root = Path(databricks_filename).resolve().parents[1]
        print(
            f"[MARKER] project root from Databricks filename: {root}",
            flush=True,
        )
        return root

    root = Path.cwd()

    print(
        f"[MARKER] project root from cwd: {root}",
        flush=True,
    )

    return root


PROJECT_ROOT = _project_root()

print(
    f"[MARKER] PROJECT_ROOT={PROJECT_ROOT}",
    flush=True,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(
        "[MARKER] added PROJECT_ROOT to sys.path",
        flush=True,
    )

print("[MARKER] bootstrap completed", flush=True)


# ============================================================
# ACTUAL APPLICATION CODE MUST CONTINUE HERE
# ============================================================

print("[MARKER] importing application modules", flush=True)

from shared.defaults import PARAM_DEFAULTS
from worker.config import Config

print("[MARKER] application modules imported", flush=True)

# ... parameter parsing
# ... Config creation
# ... source file discovery
# ... Marker conversion
# ... output writing
# ... report/state writing