"""Package exception taxonomy tests."""

from pathlib import Path

import pytest

import protrepair
from protrepair.chemistry import van_der_waals_radius_angstrom
from protrepair.io import read_structure
from protrepair.transformer.packing.faspr.backend import PackingBackendError


def test_radius_lookup_failure_is_catchable_through_package_error() -> None:
    """A real public-boundary failure must use the shared exception contract."""

    with pytest.raises(protrepair.ProtrepairError) as failure:
        van_der_waals_radius_angstrom("Xx")

    assert isinstance(failure.value, ValueError)


def test_missing_structure_is_catchable_through_package_error(tmp_path: Path) -> None:
    """I/O failures must remain catchable without importing an internal error type."""

    with pytest.raises(protrepair.ProtrepairError) as failure:
        read_structure(tmp_path / "missing.pdb")

    assert isinstance(failure.value.__cause__, FileNotFoundError)


def test_backend_exceptions_inherit_from_protrepair_error() -> None:
    """Backend-local exception bases should use the package-level base."""

    with pytest.raises(protrepair.ProtrepairError, match="backend failure"):
        raise PackingBackendError("backend failure")
