"""Check OAuth + Jobs read access only. Does not start compute or print tokens."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.databricks_client import DatabricksAPI


def main():
    try:
        api = DatabricksAPI()
        api.call('GET', '/api/2.2/jobs/list', params={'limit': 1}, safe_retry=True)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
    print('OAuth M2M and Jobs read access verified. No compute was started.')
    print('Workspace-file writes, Run as, volume grants, and model execution require subsequent deployment/smoke tests.')


if __name__ == '__main__':
    main()
