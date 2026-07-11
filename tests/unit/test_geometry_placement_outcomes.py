"""Adversarial contracts for recoverable geometry-placement outcomes."""

from typing import cast

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    completion_payload,
)

from protrepair.geometry import Vec3
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.completion.atom.outcome import GeometryPlacementOutcome
from protrepair.transformer.completion.diagnostics import (
    skipped_geometry_placement_issue,
)


def _snapshot() -> ProteinStructureSnapshot:
    structure = build_structure(
        chains=(),
        source_format=FileFormat.PDB,
        source_name="geometry-placement-outcome",
    )
    return ProteinStructureSnapshot.from_structure(structure)


def _residue_payload():
    return completion_payload(
        component_id="SER",
        residue_id=ResidueId("A", 1),
        atoms=(atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),),
    )


def test_geometry_placement_outcome_requires_reason_for_skipped_atoms() -> None:
    """Skipped atom evidence without a failure cause is not canonical."""

    with pytest.raises(ValueError, match="requires a failure reason"):
        GeometryPlacementOutcome(_snapshot(), skipped_atom_names=("OG",))


def test_geometry_placement_outcome_normalizes_and_deduplicates_atom_names() -> None:
    """Placement outcomes should expose one canonical atom identity sequence."""

    outcome = GeometryPlacementOutcome(
        _snapshot(),
        skipped_atom_names=(" og ", "OG"),
        failure_reason=" degenerate frame ",
    )

    assert outcome.skipped_atom_names == ("OG",)
    assert outcome.failure_reason == "degenerate frame"
    assert outcome.has_skipped_atoms()


def test_geometry_placement_outcome_rejects_blank_atom_names() -> None:
    """Blank atom identities must not enter completion evidence."""

    with pytest.raises(ValueError, match="must not be blank"):
        GeometryPlacementOutcome(
            _snapshot(),
            skipped_atom_names=(" ",),
            failure_reason="degenerate frame",
        )


def test_geometry_placement_outcome_rejects_non_snapshot_carrier() -> None:
    """Outcome carriers must remain canonical structure snapshots."""

    with pytest.raises(TypeError, match="structure snapshot"):
        GeometryPlacementOutcome(cast(ProteinStructureSnapshot, object()))


def test_skipped_geometry_issue_rejects_missing_atom_evidence() -> None:
    """A completion warning without named skipped atoms is not actionable."""

    with pytest.raises(ValueError, match="requires named atoms"):
        skipped_geometry_placement_issue(
            _residue_payload(),
            atom_names=(),
            reason="degenerate frame",
        )


def test_skipped_geometry_issue_normalizes_atom_names_and_reason() -> None:
    """Completion diagnostics should expose stable atom-scoped evidence."""

    issue = skipped_geometry_placement_issue(
        _residue_payload(),
        atom_names=(" og ", "OG"),
        reason=" degenerate frame ",
    )

    assert issue.atom_names == ("OG",)
    assert issue.component_id == "SER"
    assert "(degenerate frame)" in issue.message
