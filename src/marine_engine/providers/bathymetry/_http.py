"""Tiny shared HTTP-GET-with-retry helper for the bathymetry source modules.

Retries only transient connection/timeout failures, never HTTP error
responses (a 403/404 will not fix itself by retrying).
"""

import time

import requests

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_S = 1.0


def _request_with_retries(
    method: str,
    url: str,
    *,
    timeout: float,
    max_retries: int,
    backoff_s: float,
    **kwargs,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_s * (attempt + 1))
                continue
            raise
        except requests.HTTPError:
            raise
    raise last_exc  # pragma: no cover -- loop always returns or raises


def get_with_retries(
    url: str,
    params: dict | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
) -> requests.Response:
    return _request_with_retries(
        "GET", url, params=params, timeout=timeout, max_retries=max_retries, backoff_s=backoff_s
    )


def post_with_retries(
    url: str,
    data: bytes | str,
    *,
    headers: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
) -> requests.Response:
    return _request_with_retries(
        "POST",
        url,
        data=data,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        backoff_s=backoff_s,
    )
