"""Network-boundary policy for external source retrieval."""

from math import isfinite
from typing import Protocol

DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS = 30.0
DEFAULT_SOURCE_RETRIEVAL_MAX_BYTES = 32 * 1024 * 1024


class SourceResponseTooLargeError(ValueError):
    """Raised when a source response exceeds the configured byte budget."""


class _ReadableResponse(Protocol):
    """Minimal response protocol needed for bounded body reads."""

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes from the response body."""

        ...


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


def read_bounded_response_text(
    response: _ReadableResponse,
    *,
    source_name: str,
    max_bytes: int | None = None,
) -> str:
    """Read one response body as UTF-8 text without unbounded memory growth."""

    active_max_bytes = (
        DEFAULT_SOURCE_RETRIEVAL_MAX_BYTES if max_bytes is None else max_bytes
    )
    if isinstance(active_max_bytes, bool) or not isinstance(active_max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if active_max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    body = response.read(active_max_bytes + 1)
    if len(body) > active_max_bytes:
        raise SourceResponseTooLargeError(
            f"{source_name} response exceeded {active_max_bytes} bytes"
        )

    return body.decode("utf-8")
