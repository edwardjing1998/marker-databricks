"""Explicit Databricks OAuth M2M configuration; no PAT/profile fallback."""
from dataclasses import dataclass, field
import os
from urllib.parse import urlsplit


@dataclass(frozen=True)
class OAuthCredentials:
    host: str
    client_id: str
    client_secret: str = field(repr=False)


def oauth_credentials(*, host=None, client_id=None, client_secret=None):
    """Validate without network calls and without echoing sensitive values."""
    if os.getenv('DATABRICKS_TOKEN', '').strip():
        raise ValueError('OAuth-only release: unset DATABRICKS_TOKEN; use DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET')
    if os.getenv('DATABRICKS_AUTH_TYPE', 'oauth-m2m').strip() not in ('', 'oauth-m2m'):
        raise ValueError('DATABRICKS_AUTH_TYPE must be oauth-m2m')
    values = {
        'DATABRICKS_HOST': os.getenv('DATABRICKS_HOST', '') if host is None else host,
        'DATABRICKS_CLIENT_ID': os.getenv('DATABRICKS_CLIENT_ID', '') if client_id is None else client_id,
        'DATABRICKS_CLIENT_SECRET': os.getenv('DATABRICKS_CLIENT_SECRET', '') if client_secret is None else client_secret,
    }
    missing = [name for name, value in values.items() if not isinstance(value, str) or not value.strip()]
    if missing:
        raise ValueError('Missing OAuth configuration: ' + ', '.join(missing))
    values = {k: v.strip() for k, v in values.items()}
    if any(any(ord(c) < 32 or ord(c) == 127 for c in value) for value in values.values()):
        raise ValueError('OAuth configuration contains an invalid control character')
    origin = values['DATABRICKS_HOST'].rstrip('/')
    parsed = urlsplit(origin)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.path or parsed.query
            or parsed.fragment or parsed.username is not None or parsed.password is not None):
        raise ValueError('DATABRICKS_HOST must be an HTTPS workspace origin without credentials, paths, query, or fragment')
    try:
        parsed.port
    except ValueError:
        raise ValueError('Invalid DATABRICKS_HOST port') from None
    return OAuthCredentials(origin, values['DATABRICKS_CLIENT_ID'], values['DATABRICKS_CLIENT_SECRET'])
