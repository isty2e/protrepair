"""Package exception taxonomy tests."""

import inspect

import protrepair
import protrepair.errors as errors
from protrepair.transformer.packing.faspr.backend import PackingBackendError


def test_protrepair_error_is_canonical_public_exception_base() -> None:
    """The package should expose one protrepair-named exception base."""

    assert protrepair.ProtrepairError is errors.ProtrepairError
    assert "ProtrepairError" in protrepair.__all__
    assert not hasattr(errors, "PrasError")
    assert "PrasError" not in protrepair.__all__


def test_package_exceptions_inherit_from_protrepair_error() -> None:
    """Every package-specific exception should inherit from ProtrepairError."""

    package_exceptions = [
        exception_type
        for _, exception_type in inspect.getmembers(errors, inspect.isclass)
        if (
            exception_type.__module__ == errors.__name__
            and exception_type is not errors.ProtrepairError
        )
    ]

    assert package_exceptions
    assert all(
        issubclass(exception_type, errors.ProtrepairError)
        for exception_type in package_exceptions
    )


def test_backend_exceptions_inherit_from_protrepair_error() -> None:
    """Backend-local exception bases should use the package-level base."""

    assert issubclass(PackingBackendError, errors.ProtrepairError)
