"""Send only OAuth credentials to this application's OpenShift Secret via stdin.

No secret values in command arguments, log messages, source files, or artifacts.
Run only after `oc project` selects the intended namespace.
"""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.oauth import oauth_credentials


def secret_manifest():
    credentials = oauth_credentials()
    return {'apiVersion': 'v1', 'kind': 'Secret', 'type': 'Opaque',
            'metadata': {'name': 'marker-databricks-secret'},
            'stringData': {'DATABRICKS_CLIENT_ID': credentials.client_id,
                           'DATABRICKS_CLIENT_SECRET': credentials.client_secret}}


def main():
    # Capture output and replace errors rather than risking an echoed payload.
    result = subprocess.run(['oc', 'apply', '-f', '-'],
                            input=json.dumps(secret_manifest()), text=True, capture_output=True)
    if result.returncode:
        raise SystemExit('Could not apply the gateway OAuth Secret; check namespace and Secret write permission')
    print('Gateway OAuth Secret configured; credential values were not logged.')


if __name__ == '__main__':
    main()
