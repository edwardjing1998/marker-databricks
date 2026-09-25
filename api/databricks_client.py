"""Databricks Jobs REST API with SDK-managed OAuth M2M token refresh."""
import time
import requests
from shared.oauth import oauth_credentials


class RemoteError(RuntimeError):
    pass


class DatabricksAPI:
    def __init__(self, *, host=None, client_id=None, client_secret=None):
        credentials = oauth_credentials(host=host, client_id=client_id, client_secret=client_secret)
        from databricks.sdk.core import Config
        try:
            self.config = Config(host=credentials.host, client_id=credentials.client_id,
                                 client_secret=credentials.client_secret,
                                 auth_type='oauth-m2m', http_timeout_seconds=30)
        except Exception:
            # SDK exception text can include configuration: do not echo it.
            raise RemoteError('Databricks OAuth initialization failed; verify workspace host, OAuth secret, SDK, and network access') from None
        self.host = credentials.host

    def _headers(self):
        try:
            # The SDK obtains/caches/refreshes access tokens. Never store a
            # one-hour access token in GitHub as a replacement for this secret.
            return self.config.authenticate()
        except Exception:
            raise RemoteError('Databricks OAuth token acquisition failed; check client ID, Databricks-issued secret, expiry, scopes, and workspace assignment') from None

    def call(self, method: str, path: str, *, body=None, params=None, safe_retry=False):
        # POST retries are enabled only for idempotent run-now / runs-submit requests.
        attempts = 3 if safe_retry else 1
        for attempt in range(attempts):
            try:
                response = requests.request(method, self.host + path, headers=self._headers(),
                                            json=body, params=params, timeout=(10, 25), allow_redirects=False)
            except requests.RequestException:
                if attempt + 1 == attempts:
                    raise RemoteError('Databricks request timed out or could not connect; retry with the same Idempotency-Key') from None
                time.sleep(1 + attempt)
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
                time.sleep(1 + attempt)
                continue
            if not response.ok or 300 <= response.status_code < 400:
                raise RemoteError(f'Databricks API returned HTTP {response.status_code}; check workspace credentials, permissions, and network access')
            try:
                result = response.json()
            except ValueError:
                raise RemoteError('Databricks returned non-JSON data') from None
            if not isinstance(result, dict):
                raise RemoteError('Unexpected Databricks response shape')
            return result
        raise RemoteError('Databricks request failed')

    def start(self, job_id: int, parameters: dict, token: str):
        return self.call('POST', '/api/2.2/jobs/run-now', body={
            'job_id': job_id, 'job_parameters': parameters, 'idempotency_token': token,
            'queue': {'enabled': True}}, safe_retry=True)

    def run(self, run_id: int):
        return self.call('GET', '/api/2.2/jobs/runs/get', params={'run_id': run_id}, safe_retry=True)

    def output(self, task_run_id: int):
        return self.call('GET', '/api/2.2/jobs/runs/get-output', params={'run_id': task_run_id}, safe_retry=True)
