"""Source graph and residue-slot boundaries for donor span insertion."""

from pathlib import Path

import pytest

from protrepair.api import process_structure
from protrepair.diagnostics import ValidationIssueKind
from protrepair.io import FileFormat, read_structure, read_structure_string
from protrepair.scope import AbsentResidueSpanScope
from protrepair.structure import AtomRef, ProteinStructure, ResidueId
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.span_reconstruction import (
    SpanReconstructionFailure,
    SpanReconstructionFailureKind,
    reconstruct_donor_span,
)
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionSpec,
    WorkflowTransformRequests,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/pdb/span-reconstruction/1ubq-short-gaps.pdb"
)


def _source_lines(*, suffix: bool = False) -> list[str]:
    return [
        line
        for line in FIXTURE.read_text().splitlines()
        if line.startswith("ATOM  ")
        and int(line[22:26]) not in ({4, 5} if suffix else {4})
    ]


def _read_lines(lines: list[str]) -> ProteinStructure:
    return read_structure_string("\n".join(lines) + "\nEND\n", FileFormat.PDB)


def _with_bonds(
    source: ProteinStructure, bonds: tuple[TopologyBond, ...]
) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=bonds,
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


def _assert_rejected(
    source: ProteinStructure,
    scope: AbsentResidueSpanScope,
    *,
    workflow: bool,
    reason: str,
) -> None:
    donor = read_structure(FIXTURE)
    if not workflow:
        outcome = reconstruct_donor_span(
            source,
            scope=scope,
            donor_structure=donor,
            donor_residue_ids=scope.absent_residue_ids,
            donor_preceding_residue_id=scope.preceding_residue_id,
            donor_following_residue_id=scope.following_residue_id,
        )
        assert isinstance(outcome, SpanReconstructionFailure)
        assert outcome.kind is SpanReconstructionFailureKind.INVALID_TARGET_STATE
        assert reason in outcome.message
        return

    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(scope, donor, scope.absent_residue_ids),
            ),
        ),
    )
    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    assert result.structure.topology == source.topology
    assert not result.repairs
    assert len(result.issues) == 1
    assert result.issues[0].kind is ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED
    assert reason in result.issues[0].message


@pytest.mark.parametrize("workflow", [False, True])
@pytest.mark.parametrize("insertion_number", [3, 4])
def test_span_rejects_intervening_source_insertion_code(
    workflow: bool, insertion_number: int
) -> None:
    lines = [
        line[:22] + f"{insertion_number:4d}A" + line[27:]
        if int(line[22:26]) == 4
        else line
        for line in FIXTURE.read_text().splitlines()
        if line.startswith("ATOM  ")
    ]
    source = _read_lines(lines)
    inserted_code = ResidueId("A", insertion_number, "A")
    # Isolate slot identity from pre-existing anchor valence or plane failures.
    source = _with_bonds(
        source,
        tuple(
            bond
            for bond in source.topology.bonds
            if (
                ids := {
                    source.constitution.atom_ref_at(index).residue_id
                    for index in bond.endpoint_pair()
                }
            )
            == {inserted_code}
            or inserted_code not in ids
        ),
    )
    _assert_rejected(
        source,
        AbsentResidueSpanScope(
            ResidueId("A", 3), ResidueId("A", 5), (ResidueId("A", 4),)
        ),
        workflow=workflow,
        reason="consecutive source chain slots",
    )


@pytest.mark.parametrize("workflow", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
def test_terminal_span_rejects_undeclared_available_flank(
    workflow: bool, prefix: bool
) -> None:
    _assert_rejected(
        _read_lines(_source_lines()),
        AbsentResidueSpanScope(
            None if prefix else ResidueId("A", 3),
            ResidueId("A", 5) if prefix else None,
            (ResidueId("A", 4),),
        ),
        workflow=workflow,
        reason="undeclared peptide junction",
    )


@pytest.mark.parametrize("workflow", [False, True])
def test_span_rejects_reordering_carried_source_slots(workflow: bool) -> None:
    lines = _source_lines()
    source = _read_lines(
        [line for line in lines if int(line[22:26]) != 3]
        + [line for line in lines if int(line[22:26]) == 3]
    )
    _assert_rejected(
        source,
        AbsentResidueSpanScope(
            ResidueId("A", 3), ResidueId("A", 5), (ResidueId("A", 4),)
        ),
        workflow=workflow,
        reason="reorder existing source chain slots",
    )


@pytest.mark.parametrize("workflow", [False, True])
@pytest.mark.parametrize("suffix", [False, True])
@pytest.mark.parametrize("partner", ["crosslink", "OXT", "H"])
def test_span_rejects_occupied_source_carbonyl(
    workflow: bool, suffix: bool, partner: str
) -> None:
    lines = _source_lines(suffix=suffix)
    anchor = ResidueId("A", 3)
    partner_ref = AtomRef(ResidueId("A", 27), "NZ")
    if partner != "crosslink":
        oxygen = next(
            line
            for line in lines
            if int(line[22:26]) == 3 and line[12:16].strip() == "O"
        )
        lines.insert(
            0,
            oxygen[:12]
            + f"{partner:>4}"
            + oxygen[16:76]
            + f"{('O' if partner == 'OXT' else 'H'):>2}"
            + oxygen[78:],
        )
        partner_ref = AtomRef(anchor, partner)

    source = _read_lines(lines)
    carbon = source.constitution.atom_index(AtomRef(anchor, "C"))
    other = source.constitution.atom_index(partner_ref)
    bond = source.topology.bond_between(carbon, other)
    if bond is None:
        source = _with_bonds(
            source,
            (
                *source.topology.bonds,
                TopologyBond(carbon, other, provenance=BondProvenance.SOURCE_EXPLICIT),
            ),
        )
    _assert_rejected(
        source,
        AbsentResidueSpanScope(
            anchor, None if suffix else ResidueId("A", 5), (ResidueId("A", 4),)
        ),
        workflow=workflow,
        reason="carbonyl",
    )


@pytest.mark.parametrize(
    "relationship",
    [BondRelationshipType.HYDROGEN_BOND, BondRelationshipType.METAL_COORDINATION],
)
def test_terminal_span_preserves_noncovalent_carbonyl_connection(
    relationship: BondRelationshipType,
) -> None:
    source = _read_lines(_source_lines(suffix=True))
    left = AtomRef(ResidueId("A", 3), "C")
    right = AtomRef(ResidueId("A", 27), "NZ")
    source = _with_bonds(
        source,
        (
            *source.topology.bonds,
            TopologyBond(
                source.constitution.atom_index(left),
                source.constitution.atom_index(right),
                relationship_type=relationship,
                provenance=BondProvenance.SOURCE_EXPLICIT,
            ),
        ),
    )
    gap = (ResidueId("A", 4),)
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(
                    AbsentResidueSpanScope(ResidueId("A", 3), None, gap),
                    read_structure(FIXTURE),
                    gap,
                ),
            ),
        ),
    )
    assert not result.issues
    preserved = result.structure.topology.bond_between(
        result.structure.constitution.atom_index(left),
        result.structure.constitution.atom_index(right),
    )
    assert preserved is not None
    assert preserved.relationship_type is relationship
    peptide = result.structure.topology.bond_between(
        result.structure.constitution.atom_index(left),
        result.structure.constitution.atom_index(AtomRef(gap[0], "N")),
    )
    assert peptide is not None
    assert peptide.provenance is BondProvenance.SEQUENCE_INFERRED


def test_span_inserts_a_declared_insertion_code_with_both_peptide_bonds() -> None:
    source = _read_lines(
        [
            line[:22] + "   4 " + line[27:] if int(line[22:26]) == 3 else line
            for line in _source_lines()
        ]
    )
    shortcut = source.topology.bond_between(
        source.constitution.atom_index(AtomRef(ResidueId("A", 4), "C")),
        source.constitution.atom_index(AtomRef(ResidueId("A", 5), "N")),
    )
    assert shortcut is not None
    source = _with_bonds(
        source, tuple(bond for bond in source.topology.bonds if bond != shortcut)
    )
    gap = (ResidueId("A", 4, "A"),)
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(
                    AbsentResidueSpanScope(ResidueId("A", 4), ResidueId("A", 5), gap),
                    read_structure(FIXTURE),
                    (ResidueId("A", 4),),
                ),
            ),
        ),
    )
    assert not result.issues
    observed = {
        frozenset(
            result.structure.constitution.atom_ref_at(i) for i in bond.endpoint_pair()
        )
        for bond in result.structure.topology.bonds
        if len(
            {
                result.structure.constitution.atom_ref_at(i).residue_id
                for i in bond.endpoint_pair()
            }
        )
        == 2
        and any(
            result.structure.constitution.atom_ref_at(i).residue_id == gap[0]
            for i in bond.endpoint_pair()
        )
    }
    assert observed == {
        frozenset((AtomRef(ResidueId("A", 4), "C"), AtomRef(gap[0], "N"))),
        frozenset((AtomRef(gap[0], "C"), AtomRef(ResidueId("A", 5), "N"))),
    }
