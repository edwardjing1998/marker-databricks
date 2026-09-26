"""Databricks Jobs REST API with direct OAuth M2M token acquisition."""

import time

import requests

from shared.oauth import oauth_credentials


class RemoteError(RuntimeError):
    pass


class DatabricksAPI:
    def __init__(
        self,
        *,
        host=None,
        client_id=None,
        client_secret=None,
    ):
        credentials = oauth_credentials(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
        )

        self.host = credentials.host
        self.client_id = credentials.client_id
        self.client_secret = credentials.client_secret

        # Cached short-lived OAuth access token.
        # Never persist this token to files, logs, or job configuration.
        self._access_token = None
        self._token_expires_at = 0.0

    def _fetch_access_token(self):
        """
        Obtain a Databricks workspace OAuth M2M access token.

        Equivalent to:

        curl -X POST \
          https://<workspace>/oidc/v1/token \
          --user "<client_id>:<client_secret>" \
          --data "grant_type=client_credentials&scope=all-apis"
        """

        token_url = (
            self.host
            + "/oidc/v1/token"
        )

        try:
            response = requests.post(
                token_url,
                auth=(
                    self.client_id,
                    self.client_secret,
                ),
                data={
                    "grant_type": "client_credentials",
                    "scope": "all-apis",
                },
                timeout=(10, 30),
                allow_redirects=False,
            )

        except requests.RequestException:
            raise RemoteError(
                "Databricks OAuth token request "
                "timed out or could not connect"
            ) from None

        if (
            not response.ok
            or 300 <= response.status_code < 400
        ):
            raise RemoteError(
                "Databricks OAuth token endpoint "
                f"returned HTTP {response.status_code}; "
                "check client ID, Databricks-issued secret, "
                "expiry, and workspace assignment"
            )

        try:
            payload = response.json()

        except ValueError:
            raise RemoteError(
                "Databricks OAuth token endpoint "
                "returned non-JSON data"
            ) from None

        if not isinstance(payload, dict):
            raise RemoteError(
                "Unexpected Databricks OAuth token "
                "response shape"
            )

        token = payload.get(
            "access_token"
        )

        if (
            not isinstance(token, str)
            or not token.strip()
        ):
            raise RemoteError(
                "Databricks OAuth response "
                "contained no access_token"
            )

        try:
            expires_in = int(
                payload.get(
                    "expires_in",
                    3600,
                )
            )

        except (TypeError, ValueError):
            expires_in = 3600

        # Refresh before actual expiration.
        #
        # Normal Databricks OAuth tokens are typically
        # returned with expires_in=3600.
        refresh_margin = min(
            60,
            max(
                1,
                expires_in // 2,
            ),
        )

        self._access_token = (
            token.strip()
        )

        self._token_expires_at = (
            time.monotonic()
            + max(
                1,
                expires_in
                - refresh_margin,
            )
        )

        return self._access_token

    def _headers(self):
        """
        Return Authorization headers.

        Fetch a new OAuth token only when no token exists
        or the cached token is close to expiration.
        """

        if (
            not self._access_token
            or time.monotonic()
            >= self._token_expires_at
        ):
            self._fetch_access_token()

        return {
            "Authorization": (
                "Bearer "
                + self._access_token
            )
        }

    def _clear_token(self):
        """
        Clear cached OAuth token.

        Used when Databricks returns HTTP 401 so that
        a safe retry can acquire a fresh token.
        """

        self._access_token = None
        self._token_expires_at = 0.0

    def call(
        self,
        method: str,
        path: str,
        *,
        body=None,
        params=None,
        safe_retry=False,
    ):
        """
        Call a Databricks REST API endpoint.

        POST retries are enabled only when the caller has
        explicitly marked the request safe to retry.
        """

        attempts = (
            3
            if safe_retry
            else 1
        )

        for attempt in range(
            attempts
        ):
            try:
                response = requests.request(
                    method,
                    self.host + path,
                    headers=self._headers(),
                    json=body,
                    params=params,
                    timeout=(10, 25),
                    allow_redirects=False,
                )

            except requests.RequestException:
                if (
                    attempt + 1
                    == attempts
                ):
                    raise RemoteError(
                        "Databricks request timed out "
                        "or could not connect; retry "
                        "with the same Idempotency-Key"
                    ) from None

                time.sleep(
                    1 + attempt
                )

                continue

            # If a cached access token is rejected,
            # discard it and obtain a fresh one.
            #
            # Only do this automatically for requests
            # explicitly marked safe for retry.
            if (
                response.status_code == 401
                and safe_retry
                and attempt + 1
                < attempts
            ):
                self._clear_token()

                time.sleep(
                    1 + attempt
                )

                continue

            if (
                response.status_code
                in (
                    429,
                    500,
                    502,
                    503,
                    504,
                )
                and attempt + 1
                < attempts
            ):
                time.sleep(
                    1 + attempt
                )

                continue

            if (
                not response.ok
                or 300
                <= response.status_code
                < 400
            ):
                raise RemoteError(
                    "Databricks API returned "
                    f"HTTP {response.status_code}; "
                    "check workspace credentials, "
                    "permissions, and network access"
                )

            try:
                result = response.json()

            except ValueError:
                raise RemoteError(
                    "Databricks returned "
                    "non-JSON data"
                ) from None

            if not isinstance(
                result,
                dict,
            ):
                raise RemoteError(
                    "Unexpected Databricks "
                    "response shape"
                )

            return result

        raise RemoteError(
            "Databricks request failed"
        )

    def start(
        self,
        job_id: int,
        parameters: dict,
        token: str,
    ):
        return self.call(
            "POST",
            "/api/2.2/jobs/run-now",
            body={
                "job_id": job_id,
                "job_parameters": (
                    parameters
                ),
                "idempotency_token": (
                    token
                ),
                "queue": {
                    "enabled": True
                },
            },
            safe_retry=True,
        )

    def run(
        self,
        run_id: int,
    ):
        return self.call(
            "GET",
            "/api/2.2/jobs/runs/get",
            params={
                "run_id": run_id
            },
            safe_retry=True,
        )

    def output(
        self,
        task_run_id: int,
    ):
        return self.call(
            "GET",
            "/api/2.2/jobs/runs/get-output",
            params={
                "run_id": (
                    task_run_id
                )
            },
            safe_retry=True,
        )