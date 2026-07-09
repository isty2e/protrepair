"""Shared policy for release-strict scientific regression gates."""

import os

STRICT_RDKIT_RELEASE_GATE_ENV = "PROTREPAIR_RELEASE_STRICT_RDKIT"


def strict_rdkit_release_gate_enabled() -> bool:
    """Return whether all RDKit-version-bound release gates must fail closed."""

    return os.environ.get(STRICT_RDKIT_RELEASE_GATE_ENV) == "1"
