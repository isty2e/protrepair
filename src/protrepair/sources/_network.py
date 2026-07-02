"""Network-boundary policy for external source retrieval."""

from math import isfinite

DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS = 30.0


def normalize_source_retrieval_timeout(timeout_seconds: float) -> float:
    """Return a finite positive source-retrieval timeout."""

    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds,
        int | float,
    ):
        raise TypeError("timeout_seconds must be a real number")

    timeout = float(timeout_seconds)
    if not isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")

    return timeout
