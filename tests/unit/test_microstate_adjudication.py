"""Focused tests for source-microstate adjudication."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.request_builders import ingress_options
from tests.support.whole_structure_sources import WHOLE_STRUCTURE_CORPUS_SOURCES

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.inference import retained_non_polymer_fallback
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.diagnostics.source_microstate import (
    MicrostateApplicability,
    MicrostateChemistrySupportMode,
    MicrostateDecision,
    MicrostateDecisionReason,
    MicrostateStructuralRole,
    adjudicate_microstate_evidence,
    collect_microstate_evidence,
    validation_issue_from_microstate_decision,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.io import read_structure, write_structure_string
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import ResidueIndex
from protrepair.transformer.source_microstate_adjudication import (
    adjudicate_source_microstate_contradictions,
)
from protrepair.workflow import process_canonical_structure
from protrepair.workflow.contracts import LigandPolicy


def test_adjudicate_source_microstate_contradictions_demotes_double_negative_aspartate(
) -> None:
    """Double-negative ASP source charges should be demoted when geometry agrees."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ASP",
                        residue_id=ResidueId("A", 220),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 1.2, 0.0)),
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(1.5, -0.2, 0.0)),
                            atom_payload("O", "O", Vec3(2.5, 0.1, 0.0)),
                            atom_payload("CB", "C", Vec3(-0.7, -0.8, 1.2)),
                            atom_payload("CG", "C", Vec3(-1.7, -1.6, 1.8)),
                            atom_payload(
                                "OD1",
                                "O",
                                Vec3(-2.8, -1.5, 2.5),
                                formal_charge=-1,
                            ),
                            atom_payload(
                                "OD2",
                                "O",
                                Vec3(-1.4, -2.8, 1.5),
                                formal_charge=-1,
                            ),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )

    adjudicated_structure, issues = adjudicate_source_microstate_contradictions(
        structure
    )

    assert (
        adjudicated_structure.residue_formal_charge_by_atom_name(ResidueIndex(0)) == ()
    )
    assert len(issues) == 1
    assert issues[0].kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
    assert "OD1/OD2 were both annotated as -1" in issues[0].message


def test_adjudicate_source_microstate_contradictions_keeps_explicit_hydrogen_evidence(
) -> None:
    """Explicit hydrogen evidence should block this narrow acidic adjudication."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLU",
                        residue_id=ResidueId("A", 42),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 1.2, 0.0)),
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(1.5, -0.2, 0.0)),
                            atom_payload("O", "O", Vec3(2.5, 0.1, 0.0)),
                            atom_payload("CB", "C", Vec3(-0.7, -0.8, 1.2)),
                            atom_payload("CG", "C", Vec3(-1.7, -1.6, 1.8)),
                            atom_payload("CD", "C", Vec3(-2.7, -2.4, 2.4)),
                            atom_payload(
                                "OE1",
                                "O",
                                Vec3(-3.8, -2.3, 3.1),
                                formal_charge=-1,
                            ),
                            atom_payload(
                                "OE2",
                                "O",
                                Vec3(-2.4, -3.6, 2.1),
                                formal_charge=-1,
                            ),
                            atom_payload("HE2", "H", Vec3(-2.1, -4.1, 2.6)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )

    adjudicated_structure, issues = adjudicate_source_microstate_contradictions(
        structure
    )

    assert adjudicated_structure.residue_formal_charge_by_atom_name(
        ResidueIndex(0)
    ) == structure.residue_formal_charge_by_atom_name(ResidueIndex(0))
    assert issues == ()


def test_adjudicate_source_microstate_contradictions_requires_geometry_support() -> (
    None
):
    """Asymmetric acidic geometry should not be auto-adjudicated."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ASP",
                        residue_id=ResidueId("A", 17),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 1.2, 0.0)),
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(1.5, -0.2, 0.0)),
                            atom_payload("O", "O", Vec3(2.5, 0.1, 0.0)),
                            atom_payload("CB", "C", Vec3(-0.7, -0.8, 1.2)),
                            atom_payload("CG", "C", Vec3(-1.7, -1.6, 1.8)),
                            atom_payload(
                                "OD1",
                                "O",
                                Vec3(-2.8, -1.5, 2.5),
                                formal_charge=-1,
                            ),
                            atom_payload(
                                "OD2",
                                "O",
                                Vec3(-0.8, -3.2, 1.1),
                                formal_charge=-1,
                            ),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )

    adjudicated_structure, issues = adjudicate_source_microstate_contradictions(
        structure
    )

    assert adjudicated_structure.residue_formal_charge_by_atom_name(
        ResidueIndex(0)
    ) == structure.residue_formal_charge_by_atom_name(ResidueIndex(0))
    assert issues == ()


def test_adjudicate_source_microstate_contradictions_demotes_terminal_double_negative(
) -> None:
    """Terminal carboxylate-like motifs should use the same adjudication rule."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 9),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 1.2, 0.0)),
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(1.5, -0.2, 0.0)),
                            atom_payload(
                                "O",
                                "O",
                                Vec3(2.6, -0.1, 0.7),
                                formal_charge=-1,
                            ),
                            atom_payload(
                                "OXT",
                                "O",
                                Vec3(1.2, -1.4, -0.3),
                                formal_charge=-1,
                            ),
                            atom_payload("CB", "C", Vec3(-0.7, -0.8, 1.2)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )

    adjudicated_structure, issues = adjudicate_source_microstate_contradictions(
        structure
    )

    assert (
        adjudicated_structure.residue_formal_charge_by_atom_name(ResidueIndex(0)) == ()
    )
    assert len(issues) == 1
    assert "O/OXT were both annotated as -1" in issues[0].message


def test_adjudicate_source_microstate_contradictions_batches_multiple_demotions() -> (
    None
):
    """Independent source-charge demotions should be applied in one structure pass."""

    def acidic_residue(seq_num: int, x_offset: float):
        return residue_payload(
            component_id="ASP",
            residue_id=ResidueId("A", seq_num),
            atoms=(
                atom_payload("N", "N", Vec3(x_offset + 0.0, 1.2, 0.0)),
                atom_payload("CA", "C", Vec3(x_offset + 0.0, 0.0, 0.0)),
                atom_payload("C", "C", Vec3(x_offset + 1.5, -0.2, 0.0)),
                atom_payload("O", "O", Vec3(x_offset + 2.5, 0.1, 0.0)),
                atom_payload("CB", "C", Vec3(x_offset - 0.7, -0.8, 1.2)),
                atom_payload("CG", "C", Vec3(x_offset - 1.7, -1.6, 1.8)),
                atom_payload(
                    "OD1",
                    "O",
                    Vec3(x_offset - 2.8, -1.5, 2.5),
                    formal_charge=-1,
                ),
                atom_payload(
                    "OD2",
                    "O",
                    Vec3(x_offset - 1.4, -2.8, 1.5),
                    formal_charge=-1,
                ),
            ),
        )

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    acidic_residue(17, 0.0),
                    acidic_residue(18, 10.0),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )

    adjudicated_structure, issues = adjudicate_source_microstate_contradictions(
        structure
    )

    assert (
        adjudicated_structure.residue_formal_charge_by_atom_name(ResidueIndex(0)) == ()
    )
    assert (
        adjudicated_structure.residue_formal_charge_by_atom_name(ResidueIndex(1)) == ()
    )
    assert len(issues) == 2


def test_adjudicate_source_microstate_contradictions_on_3ja8_demotes_asp220() -> None:
    """3JA8 ASP 2:220 should trigger one explicit contradiction adjudication."""

    source = WHOLE_STRUCTURE_CORPUS_SOURCES["3ja8-whole-structure"]
    structure = read_structure(
        source.output_path,
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )

    adjudicated_structure, issues = adjudicate_source_microstate_contradictions(
        structure
    )
    contradiction_issues = tuple(
        issue
        for issue in issues
        if (
            issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
            and issue.residue_id == ResidueId("2", 220)
        )
    )

    assert len(contradiction_issues) == 1
    pdb_text = write_structure_string(adjudicated_structure, FileFormat.PDB)
    od1_line = next(line for line in pdb_text.splitlines() if " OD1 ASP 2 220" in line)
    od2_line = next(line for line in pdb_text.splitlines() if " OD2 ASP 2 220" in line)
    assert "O1-" not in od1_line
    assert "O1-" not in od2_line


def test_process_canonical_structure_propagates_initial_microstate_issues() -> None:
    """Canonical workflow results should retain adjudication issues from ingress."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ASP",
                        residue_id=ResidueId("A", 8),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 1.2, 0.0)),
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(1.5, -0.2, 0.0)),
                            atom_payload("O", "O", Vec3(2.5, 0.1, 0.0)),
                            atom_payload("CB", "C", Vec3(-0.7, -0.8, 1.2)),
                            atom_payload("CG", "C", Vec3(-1.7, -1.6, 1.8)),
                            atom_payload(
                                "OD1",
                                "O",
                                Vec3(-2.8, -1.5, 2.5),
                                formal_charge=-1,
                            ),
                            atom_payload(
                                "OD2",
                                "O",
                                Vec3(-1.4, -2.8, 1.5),
                                formal_charge=-1,
                            ),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )

    result = process_canonical_structure(structure)
    contradiction_issues = tuple(
        issue
        for issue in result.issues
        if issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
    )

    assert len(contradiction_issues) == 1
    assert result.structure.residue_formal_charge_by_atom_name(ResidueIndex(0)) == ()


def test_collect_microstate_evidence_classifies_curated_heme_family() -> None:
    """Known bundled cofactors should land in the curated retained-non-polymer
    family."""

    structure = build_structure(
        ligands=(
            residue_payload(
                component_id="HEM",
                residue_id=ResidueId("A", 501),
                atoms=(
                    atom_payload("CHA", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("C1A", "C", Vec3(1.4, 0.0, 0.0)),
                    atom_payload("C2A", "C", Vec3(2.1, 1.2, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        chains=(),
        source_format=FileFormat.PDB,
    )
    component_library = build_default_component_library()

    evidence = collect_microstate_evidence(
        structure.constitution.ligands[0],
        residue_geometry=structure.residue_geometry(ResidueIndex(0)),
        source_formal_charge_by_atom_name=dict(
            structure.residue_formal_charge_by_atom_name(ResidueIndex(0))
        ),
        standard_component_library=build_default_component_library(),
        component_library=component_library,
    )
    decision = adjudicate_microstate_evidence(evidence)

    assert (
        evidence.classification.structural_role
        is MicrostateStructuralRole.RETAINED_NON_POLYMER
    )
    assert (
        evidence.classification.chemistry_support_mode
        is MicrostateChemistrySupportMode.CURATED_COMPONENT_TEMPLATE
    )
    assert evidence.classification.applicability is MicrostateApplicability.APPLICABLE
    assert decision.decision is MicrostateDecision.PRESERVE_SOURCE
    assert decision.reasons == (MicrostateDecisionReason.NO_CONTRADICTION_DETECTED,)


def test_unknown_retained_non_polymer_contradiction_is_marked_ambiguous() -> None:
    """Unknown retained non-polymers should surface contradictions, not auto-fix."""

    structure = build_structure(
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 7),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload(
                        "O1",
                        "O",
                        Vec3(5.2, 0.0, 0.0),
                        formal_charge=-1,
                    ),
                    atom_payload("N1", "N", Vec3(4.0, 1.2, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        chains=(),
        source_format=FileFormat.PDB,
    )
    component_library = build_default_component_library()

    evidence = collect_microstate_evidence(
        structure.constitution.ligands[0],
        residue_geometry=structure.residue_geometry(ResidueIndex(0)),
        source_formal_charge_by_atom_name=dict(
            structure.residue_formal_charge_by_atom_name(ResidueIndex(0))
        ),
        standard_component_library=build_default_component_library(),
        component_library=component_library,
    )
    decision = adjudicate_microstate_evidence(evidence)
    issue = validation_issue_from_microstate_decision(decision)

    assert (
        evidence.classification.structural_role
        is MicrostateStructuralRole.RETAINED_NON_POLYMER
    )
    assert (
        evidence.classification.chemistry_support_mode
        is MicrostateChemistrySupportMode.TEMPLATELESS_RDKIT_FALLBACK
    )
    assert evidence.classification.applicability is MicrostateApplicability.APPLICABLE
    assert decision.decision is MicrostateDecision.AMBIGUOUS
    assert decision.reasons == (
        MicrostateDecisionReason.SOURCE_CHARGE_GEOMETRY_CONTRADICTION,
    )
    assert issue is not None
    assert "unknown retained non-polymer chemistry" in issue.message


def test_unknown_retained_microstate_propagates_no_rdkit_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Microstate diagnostics must not hide a broken required RDKit install."""

    monkeypatch.setattr(retained_non_polymer_fallback, "Chem", None)
    structure = build_structure(
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 7),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload(
                        "O1",
                        "O",
                        Vec3(5.2, 0.0, 0.0),
                        formal_charge=-1,
                    ),
                ),
                is_hetero=True,
            ),
        ),
        chains=(),
        source_format=FileFormat.PDB,
    )

    with pytest.raises(RdkitUnavailableError, match="required rdkit dependency"):
        collect_microstate_evidence(
            structure.constitution.ligands[0],
            residue_geometry=structure.residue_geometry(ResidueIndex(0)),
            source_formal_charge_by_atom_name=dict(
                structure.residue_formal_charge_by_atom_name(ResidueIndex(0))
            ),
            standard_component_library=build_default_component_library(),
            component_library=build_default_component_library(),
        )


def test_unknown_retained_non_polymer_without_contradiction_preserves_source() -> None:
    """Unknown retained non-polymers should stay preserved when evidence agrees."""

    structure = build_structure(
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 8),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        chains=(),
        source_format=FileFormat.PDB,
    )
    component_library = build_default_component_library()

    evidence = collect_microstate_evidence(
        structure.constitution.ligands[0],
        residue_geometry=structure.residue_geometry(ResidueIndex(0)),
        source_formal_charge_by_atom_name=dict(
            structure.residue_formal_charge_by_atom_name(ResidueIndex(0))
        ),
        standard_component_library=build_default_component_library(),
        component_library=component_library,
    )
    decision = adjudicate_microstate_evidence(evidence)

    assert (
        evidence.classification.chemistry_support_mode
        is MicrostateChemistrySupportMode.TEMPLATELESS_RDKIT_FALLBACK
    )
    assert decision.decision is MicrostateDecision.PRESERVE_SOURCE
    assert decision.reasons == (MicrostateDecisionReason.NO_CONTRADICTION_DETECTED,)


def test_collect_microstate_evidence_classifies_single_atom_iron_as_metal_or_ion() -> (
    None
):
    """Single-atom inorganic hetero species should use the metal/ion family."""

    structure = build_structure(
        ligands=(
            residue_payload(
                component_id="FE",
                residue_id=ResidueId("M", 1),
                atoms=(
                    atom_payload(
                        "FE",
                        "FE",
                        Vec3(0.0, 0.0, 0.0),
                        formal_charge=2,
                    ),
                ),
                is_hetero=True,
            ),
        ),
        chains=(),
        source_format=FileFormat.PDB,
    )
    component_library = build_default_component_library()

    evidence = collect_microstate_evidence(
        structure.constitution.ligands[0],
        residue_geometry=structure.residue_geometry(ResidueIndex(0)),
        source_formal_charge_by_atom_name=dict(
            structure.residue_formal_charge_by_atom_name(ResidueIndex(0))
        ),
        standard_component_library=build_default_component_library(),
        component_library=component_library,
    )
    decision = adjudicate_microstate_evidence(evidence)

    assert (
        evidence.classification.structural_role
        is MicrostateStructuralRole.SINGLE_ATOM_INORGANIC
    )
    assert (
        evidence.classification.chemistry_support_mode
        is MicrostateChemistrySupportMode.NONE
    )
    assert (
        evidence.classification.applicability
        is MicrostateApplicability.NOT_APPLICABLE
    )
    assert decision.decision is MicrostateDecision.PRESERVE_SOURCE
    assert decision.reasons == (MicrostateDecisionReason.NO_CONTRADICTION_DETECTED,)


def test_collect_microstate_evidence_keeps_multi_atom_sulfate_out_of_metal_family() -> (
    None
):
    """Multi-atom inorganic residues should not collapse into metal/ion family."""

    structure = build_structure(
        ligands=(
            residue_payload(
                component_id="SO4",
                residue_id=ResidueId("M", 2),
                atoms=(
                    atom_payload("S", "S", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                    atom_payload("O2", "O", Vec3(-1.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        chains=(),
        source_format=FileFormat.PDB,
    )
    component_library = build_default_component_library()

    evidence = collect_microstate_evidence(
        structure.constitution.ligands[0],
        residue_geometry=structure.residue_geometry(ResidueIndex(0)),
        source_formal_charge_by_atom_name=dict(
            structure.residue_formal_charge_by_atom_name(ResidueIndex(0))
        ),
        standard_component_library=build_default_component_library(),
        component_library=component_library,
    )

    assert (
        evidence.classification.structural_role
        is MicrostateStructuralRole.RETAINED_NON_POLYMER
    )
    assert (
        evidence.classification.chemistry_support_mode
        is MicrostateChemistrySupportMode.TEMPLATELESS_RDKIT_FALLBACK
    )
    assert evidence.classification.applicability is MicrostateApplicability.APPLICABLE
