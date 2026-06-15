"""Topology diagnostics for cis peptides and disulfide assignments."""

from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalChainPayload,
    CanonicalResiduePayload,
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)

from protrepair.diagnostics import (
    EventScopeKind,
    ValidationIssueKind,
    detect_topology,
)
from protrepair.geometry import Vec3
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    ResidueId,
)
from protrepair.structure.provenance import FileFormat


def test_detect_topology_reports_cis_peptide() -> None:
    """Adjacent residues with cis-like omega should be reported explicitly."""

    structure = build_structure(
        (
            chain_payload(
                "A",
                (
                    gly_residue(
                        chain_id="A",
                        seq_num=1,
                        atoms=(
                            atom_payload("CA", "C", Vec3(0.0, 1.0, 0.0)),
                            atom_payload("C", "C", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                    gly_residue(
                        chain_id="A",
                        seq_num=2,
                        atoms=(
                            atom_payload("N", "N", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
        )
    )

    report = detect_topology(structure)
    issues = report.to_issues()

    assert len(report.cis_peptides) == 1
    assert report.cis_peptides[0].omega_degrees == 0.0
    assert report.likely_disulfides == ()
    assert report.ambiguous_disulfides == ()
    assert len(issues) == 1
    assert issues[0].kind is ValidationIssueKind.CIS_PEPTIDE


def test_detect_topology_ignores_collinear_trans_peptide() -> None:
    """Collinear trans omega should not fall back to a cis-peptide report."""

    structure = build_structure(
        (
            chain_payload(
                "A",
                (
                    gly_residue(
                        chain_id="A",
                        seq_num=1,
                        atoms=(
                            atom_payload("CA", "C", Vec3(-1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                    gly_residue(
                        chain_id="A",
                        seq_num=2,
                        atoms=(
                            atom_payload("N", "N", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(2.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        )
    )

    report = detect_topology(structure)

    assert report.cis_peptides == ()


def test_detect_topology_reports_likely_disulfide_without_warning_issue() -> None:
    """A unique close SG-SG pair should surface as a likely disulfide artifact."""

    structure = build_structure(
        (
            chain_payload(
                "A",
                (
                    cys_residue("A", 1, Vec3(0.0, 0.0, 0.0)),
                    cys_residue("A", 2, Vec3(2.2, 0.0, 0.0)),
                ),
            ),
        )
    )

    report = detect_topology(structure)

    assert len(report.likely_disulfides) == 1
    assert report.ambiguous_disulfides == ()
    assert report.to_issues() == ()


def test_detect_topology_reports_ambiguous_disulfide() -> None:
    """One sulfur with two nearby candidates should be flagged as ambiguous."""

    structure = build_structure(
        (
            chain_payload(
                "A",
                (
                    cys_residue("A", 1, Vec3(0.0, 0.0, 0.0)),
                    cys_residue("A", 2, Vec3(2.8, 0.0, 0.0)),
                ),
            ),
            chain_payload(
                "B",
                (cys_residue("B", 1, Vec3(0.0, 2.8, 0.0)),),
            ),
        )
    )

    report = detect_topology(structure)
    issues = report.to_issues()

    assert report.cis_peptides == ()
    assert report.likely_disulfides == ()
    assert len(report.ambiguous_disulfides) == 1
    assert report.ambiguous_disulfides[0].residue_id == ResidueId("A", 1)
    assert len(report.ambiguous_disulfides[0].candidates) == 2
    assert len(issues) == 1
    assert issues[0].kind is ValidationIssueKind.AMBIGUOUS_DISULFIDE
    assert issues[0].scope.kind is EventScopeKind.RESIDUE_SET
    assert issues[0].residue_id is None
    assert issues[0].scope.targets_residue(ResidueId("A", 1))
    assert issues[0].scope.targets_residue(ResidueId("A", 2))
    assert issues[0].scope.targets_residue(ResidueId("B", 1))


def test_detect_topology_projects_cis_peptide_as_pair_scope() -> None:
    """Cis-peptide issues should carry residue-pair provenance."""

    structure = build_structure(
        (
            chain_payload(
                "A",
                (
                    gly_residue(
                        chain_id="A",
                        seq_num=1,
                        atoms=(
                            atom_payload("CA", "C", Vec3(0.0, 1.0, 0.0)),
                            atom_payload("C", "C", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                    gly_residue(
                        chain_id="A",
                        seq_num=2,
                        atoms=(
                            atom_payload("N", "N", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
        )
    )

    issue = detect_topology(structure).to_issues()[0]

    assert issue.scope.kind is EventScopeKind.RESIDUE_PAIR
    assert issue.residue_id is None
    assert issue.scope.targets_residue(ResidueId("A", 1))
    assert issue.scope.targets_residue(ResidueId("A", 2))


def cys_residue(
    chain_id: str,
    seq_num: int,
    sg_position: Vec3,
) -> CanonicalResiduePayload:
    """Build one minimal cysteine residue with an SG atom for topology tests."""

    return residue_payload(
        component_id="CYS",
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=(
            atom_payload("SG", "S", sg_position),
        ),
    )


def gly_residue(
    *,
    chain_id: str,
    seq_num: int,
    atoms: tuple[CanonicalAtomPayload, ...],
) -> CanonicalResiduePayload:
    """Build one canonical glycine payload for topology tests."""

    return residue_payload(
        component_id="GLY",
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
    )


def build_structure(chains: tuple[CanonicalChainPayload, ...]) -> ProteinStructure:
    """Build one canonical structure for topology diagnostics tests."""

    return build_canonical_structure(
        chains=chains,
        source_format=FileFormat.PDB,
        source_name="topology-diagnostic-fixture",
    )
