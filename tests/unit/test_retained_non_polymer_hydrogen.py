"""Unit tests for retained non-polymer hydrogen completion."""

from collections import Counter

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    completion_payload,
    residue_payload,
)
from tests.support.refinement_benchmarks import resolve_fixture_path
from tests.support.refinement_cases import EXPLORATORY_REFINEMENT_FIXTURE_SOURCES
from tests.support.retained_non_polymer_components import (
    build_retained_non_polymer_component_library,
)
from tests.support.whole_structure_sources import WHOLE_STRUCTURE_CORPUS_SOURCES

from protrepair.chemistry import (
    ComponentLibrary,
    build_default_component_library,
)
from protrepair.chemistry.component.graph import ChemicalComponentDefinition
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.inference import (
    retained_non_polymer_fallback as fallback_inference,
)
from protrepair.diagnostics.chemistry_contradictions import (
    diagnose_retained_non_polymer_template_chemistry_contradictions,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure, write_structure_string
from protrepair.io.gemmi_writer import pdb_atom_serial_by_atom_ref
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import (
    RetainedNonPolymerChemistryReadinessFact,
    TopologyAvailabilityState,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen import (
    rdkit_evidence,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen import (
    repair as retained_non_polymer_repair,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - optional dependency
    Chem = None

RDKIT_AVAILABLE = Chem is not None


def test_template_backed_linked_glycan_hydrogenation_skips_absent_anchor_only() -> (
    None
):
    """Template-backed linked glycans should materialize remaining anchored Hs."""

    component_library = build_default_component_library()
    structure = read_structure(
        resolve_fixture_path(
            EXPLORATORY_REFINEMENT_FIXTURE_SOURCES[
                "2h6o-glycan-problematic"
            ].output_path
        )
    )
    before_facts = _retained_non_polymer_facts_by_residue_id(
        structure,
        component_library=component_library,
    )

    assert before_facts
    assert all(fact.requires_hydrogen_completion() for fact in before_facts.values())
    assert {
        fact.heavy_atom_topology_availability_state.value
        for fact in before_facts.values()
    } == {
        "absent"
    }

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=component_library,
    )

    assert result.issues == ()
    assert len(result.repairs) == len(before_facts)
    assert all("HO1" not in repair.atom_names for repair in result.repairs)
    for ligand in result.structure.constitution.ligands:
        assert any(atom_site.element == "H" for atom_site in ligand.atom_sites)
        assert not ligand.has_atom_site("HO1")

    after_facts = _retained_non_polymer_facts_by_residue_id(
        result.structure,
        component_library=component_library,
    )
    assert {
        fact.hydrogen_coverage_state.value for fact in after_facts.values()
    } == {"complete"}
    assert not any(
        fact.requires_hydrogen_completion() for fact in after_facts.values()
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit is required for fallback")
def test_mixed_glycan_template_and_rdkit_fallback_hydrogenation_resolves_2z62() -> (
    None
):
    """2Z62 glycans should resolve template and RDKit-fallback hydrogen needs."""

    component_library = build_default_component_library()
    structure = read_structure(
        WHOLE_STRUCTURE_CORPUS_SOURCES["2z62-whole-structure"].output_path
    )
    before_facts = _retained_non_polymer_facts_by_residue_id(
        structure,
        component_library=component_library,
        component_ids=frozenset({"NAG", "BMA", "FUL"}),
    )
    target_residue_ids = frozenset(before_facts)

    assert Counter(
        (
            fact.component_id,
            fact.hydrogen_expectation_source.value,
            fact.requires_hydrogen_completion(),
        )
        for fact in before_facts.values()
    ) == Counter(
        {
            ("NAG", "template", True): 6,
            ("BMA", "rdkit_fallback", True): 1,
            ("FUL", "rdkit_fallback", True): 2,
        }
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=component_library,
        target_residue_ids=target_residue_ids,
    )

    assert len(result.repairs) == len(target_residue_ids)
    fallback_issues = _fallback_used_issues(result.issues)
    assert Counter(issue.residue_id for issue in fallback_issues) == Counter(
        {
            residue_id: 1
            for residue_id, fact in before_facts.items()
            if fact.hydrogen_expectation_source.value == "rdkit_fallback"
        }
    )
    after_facts = _retained_non_polymer_facts_by_residue_id(
        result.structure,
        component_library=component_library,
        component_ids=frozenset({"NAG", "BMA", "FUL"}),
    )
    assert Counter(
        (
            fact.component_id,
            fact.hydrogen_expectation_source.value,
            fact.hydrogen_coverage_state.value,
            fact.requires_hydrogen_completion(),
        )
        for fact in after_facts.values()
    ) == Counter(
        {
            ("NAG", "template", "complete", False): 6,
            ("BMA", "rdkit_fallback", "complete", False): 1,
            ("FUL", "rdkit_fallback", "complete", False): 2,
        }
    )


def test_add_retained_non_polymer_hydrogens_hydrogenates_supported_ligands() -> None:
    """Supported retained non-polymers should receive template-backed hydrogens."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.2, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(4.0, 1.2, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-hydrogenation",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        allow_retained_non_polymer_rdkit_fallback=False,
    )

    hydrogenated_ligand = result.structure.constitution.ligands[0]
    assert hydrogenated_ligand.has_atom_site("H1")
    assert _has_topology_bond(
        result.structure,
        ResidueId("L", 1),
        "C1",
        "H1",
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )
    assert result.structure.constitution.chain("A").residues[0].atom_site_names() == (
        "N",
        "CA",
        "C",
        "O",
    )
    assert any(
        repair.kind is RepairEventKind.HYDROGENS_ADDED
        and repair.residue_id == ResidueId("L", 1)
        and repair.atom_names == ("H1",)
        for repair in result.repairs
    )
    assert not result.issues


def test_template_backed_retained_non_polymer_completion_preserves_source_h() -> None:
    """Template completion should not rename compatible source hydrogens."""

    residue_id = ResidueId("L", 1)
    source_h_position = Vec3(3.3, 0.8, 0.0)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.2, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(4.0, 1.2, 0.0)),
                    atom_payload("HSRC", "H", source_h_position),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-template-source-h-preservation",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        allow_retained_non_polymer_rdkit_fallback=False,
    )

    assert result.structure.constitution.ligands[0].atom_site_names() == (
        "C1",
        "O1",
        "N1",
        "HSRC",
    )
    assert result.structure.residue_geometry(
        result.structure.constitution.residue_index(residue_id)
    ).position("HSRC") == source_h_position
    assert result.repairs == ()
    assert _has_topology_bond(
        result.structure,
        residue_id,
        "C1",
        "HSRC",
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )
    assert not result.issues


def test_template_backed_retained_non_polymer_hydrogenation_reports_noop() -> None:
    """Degenerate template placement should not silently look successful."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(0.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-template-placement-noop",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        allow_retained_non_polymer_rdkit_fallback=False,
    )

    assert result.structure.constitution.ligands[0].atom_site_names() == (
        "C1",
        "O1",
        "N1",
    )
    assert result.repairs == ()
    assert not _has_topology_bond(
        result.structure,
        residue_id,
        "C1",
        "H1",
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )
    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        and issue.severity is IssueSeverity.WARNING
        and issue.residue_id == residue_id
        and "template-backed hydrogen placement failed" in issue.message
        and "leaving residue unchanged" in issue.message
        for issue in result.issues
    )
    assert not _fallback_used_issues(result.issues)


def test_template_backed_retained_non_polymer_noop_preserves_source_h() -> None:
    """Failed template placement should not erase source hydrogens."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("HSRC", "H", Vec3(0.5, 0.5, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-template-noop-source-h",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        allow_retained_non_polymer_rdkit_fallback=False,
    )

    assert result.structure.constitution.ligands[0].atom_site_names() == (
        "C1",
        "O1",
        "N1",
        "HSRC",
    )
    assert result.repairs == ()
    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        and issue.residue_id == residue_id
        and "template-backed hydrogen placement failed" in issue.message
        for issue in result.issues
    )


def test_add_retained_non_polymer_hydrogens_batches_supported_ligands() -> None:
    """Multiple retained non-polymers should be hydrogenated before one rebuild."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.2, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(4.0, 1.2, 0.0)),
                ),
                is_hetero=True,
            ),
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("M", 2),
                atoms=(
                    atom_payload("C1", "C", Vec3(14.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(15.2, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(14.0, 1.2, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-batched-hydrogenation",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    assert tuple(
        ligand.has_atom_site("H1") for ligand in result.structure.constitution.ligands
    ) == (True, True)
    assert tuple(repair.residue_id for repair in result.repairs) == (
        ResidueId("L", 1),
        ResidueId("M", 2),
    )


def test_add_retained_non_polymer_hydrogens_reports_charge_template_contradiction() -> (
    None
):
    """Template-backed ligands should surface source/template charge conflicts."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
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
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-charge-contradiction",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    contradiction_issues = tuple(
        issue
        for issue in result.issues
        if issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
    )
    assert len(contradiction_issues) == 1
    assert "O1 source -1 vs template +0" in contradiction_issues[0].message
    assert "template-backed hydrogenation" in contradiction_issues[0].message
    assert any(
        repair.kind is RepairEventKind.HYDROGENS_ADDED
        and repair.residue_id == ResidueId("L", 1)
        for repair in result.repairs
    )


def test_add_retained_non_polymer_hydrogens_updates_ligand_on_polymer_chain_id() -> (
    None
):
    """Retained non-polymers should still update as ligands on polymer chain IDs."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("A", 1058),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.2, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(4.0, 1.2, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-shared-chain-id",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    hydrogenated_ligand = result.structure.constitution.ligands[0]
    assert hydrogenated_ligand.residue_id == ResidueId("A", 1058)
    assert hydrogenated_ligand.has_atom_site("H1")
    assert result.structure.constitution.chain("A").residues[0].residue_id == ResidueId(
        "A", 1
    )


def test_retained_non_polymer_charge_contradiction_detector_is_quiet_when_charges_match(
) -> None:
    """Contradiction diagnostics should stay quiet when charges agree."""

    payload = completion_payload(
        component_id="LIG",
        residue_id=ResidueId("L", 1),
        is_hetero=True,
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
    )
    template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="LIG",
            atom_names=("C1", "O1", "N1"),
            formal_charges={"O1": -1},
        ),
    )

    assert (
        diagnose_retained_non_polymer_template_chemistry_contradictions(
            payload.residue_site,
            residue_geometry=payload.residue_geometry,
            source_formal_charge_by_atom_name=dict(payload.formal_charge_by_atom_name),
            template=template,
        )
        == ()
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_add_retained_non_polymer_hydrogens_uses_override_for_unsupported_component(
) -> None:
    """Unsupported retained non-polymers should hydrogenate from override chemistry."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-override-hydrogenation",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        chemistry_evidence=(
            RetainedNonPolymerChemistryOverride(
                residue_id=ResidueId("L", 1),
                smiles="CO",
                heavy_atom_names=("C1", "O1"),
            ).to_evidence(),
        ),
    )

    hydrogenated_ligand = result.structure.constitution.ligands[0]
    added_hydrogen_names = tuple(
        atom_site.name
        for atom_site in hydrogenated_ligand.atom_sites
        if atom_site.element == "H"
    )
    assert added_hydrogen_names == ("H001", "H002", "H003", "H004")
    assert _has_topology_bond(
        result.structure,
        ResidueId("L", 1),
        "C1",
        "O1",
        provenance=BondProvenance.EVIDENCE_RESOLVED,
    )
    after_facts = _retained_non_polymer_facts_by_residue_id(
        result.structure,
        component_library=build_retained_non_polymer_component_library(),
        component_ids=frozenset({"UNK"}),
    )
    assert (
        after_facts[ResidueId("L", 1)].heavy_atom_topology_availability_state
        is TopologyAvailabilityState.PRESENT
    )
    assert (
        after_facts[ResidueId("L", 1)].hydrogen_topology_availability_state
        is TopologyAvailabilityState.PRESENT
    )
    assert _has_topology_bond(
        result.structure,
        ResidueId("L", 1),
        "C1",
        "H001",
        provenance=BondProvenance.EVIDENCE_RESOLVED,
    )
    assert _has_topology_bond(
        result.structure,
        ResidueId("L", 1),
        "O1",
        "H004",
        provenance=BondProvenance.EVIDENCE_RESOLVED,
    )
    assert any(
        repair.kind is RepairEventKind.HYDROGENS_ADDED
        and repair.residue_id == ResidueId("L", 1)
        and repair.atom_names == added_hydrogen_names
        for repair in result.repairs
    )
    assert not _fallback_used_issues(result.issues)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_fallback_rejects_unsupported_motif_with_warning() -> (
    None
):
    """Fallback-only unsupported chemistry should leave the ligand unchanged."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(1.45, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(2.69, 0.0, 0.0)),
                    atom_payload("O2", "O", Vec3(1.45, 1.24, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-fallback-unsupported-nitro",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    assert result.structure.constitution.ligands[0].atom_site_names() == (
        "C1",
        "N1",
        "O1",
        "O2",
    )
    assert not result.repairs
    assert not _fallback_used_issues(result.issues)
    assert any(
        issue.kind is ValidationIssueKind.MISSING_COMPONENT_DEFINITION
        and issue.residue_id == residue_id
        and "RDKit fallback hydrogenation failed" in issue.message
        and "unsupported retained non-polymer fallback nitro or nitrate motif"
        in issue.message
        for issue in result.issues
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_evidence_allows_fallback_unsupported_motif() -> None:
    """Explicit evidence should remain authoritative for fallback-rejected motifs."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(1.45, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(2.69, 0.0, 0.0)),
                    atom_payload("O2", "O", Vec3(1.45, 1.24, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-evidence-nitro",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        chemistry_evidence=(
            RetainedNonPolymerChemistryOverride(
                residue_id=residue_id,
                smiles="C[N+](=O)[O-]",
                heavy_atom_names=("C1", "N1", "O1", "O2"),
            ).to_evidence(),
        ),
    )

    assert _hydrogen_atom_names(result.structure, residue_id) == (
        "H001",
        "H002",
        "H003",
    )
    assert _has_topology_bond(
        result.structure,
        residue_id,
        "N1",
        "O1",
        provenance=BondProvenance.EVIDENCE_RESOLVED,
    )
    assert _has_topology_bond(
        result.structure,
        residue_id,
        "N1",
        "O2",
        provenance=BondProvenance.EVIDENCE_RESOLVED,
    )
    assert not _fallback_used_issues(result.issues)
    assert not result.issues


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_evidence_preserves_partial_source_h_names() -> None:
    """Evidence completion should add missing Hs around preserved source Hs."""

    residue_id = ResidueId("L", 1)
    source_h_position = Vec3(3.3, 0.8, 0.0)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                    atom_payload("HSRC", "H", source_h_position),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-evidence-partial-source-h",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        chemistry_evidence=(
            RetainedNonPolymerChemistryOverride(
                residue_id=residue_id,
                smiles="CO",
                heavy_atom_names=("C1", "O1"),
            ).to_evidence(),
        ),
    )

    hydrogen_names = _hydrogen_atom_names(result.structure, residue_id)
    assert "HSRC" in hydrogen_names
    assert len(hydrogen_names) == 4
    assert result.structure.residue_geometry(
        result.structure.constitution.residue_index(residue_id)
    ).position("HSRC") == source_h_position
    assert _has_topology_bond(
        result.structure,
        residue_id,
        "C1",
        "HSRC",
        provenance=BondProvenance.EVIDENCE_RESOLVED,
    )
    assert any(
        repair.kind is RepairEventKind.HYDROGENS_ADDED
        and repair.residue_id == residue_id
        and "HSRC" not in repair.atom_names
        and len(repair.atom_names) == 3
        for repair in result.repairs
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_evidence_does_not_preserve_overpopulated_source_h(
) -> None:
    """Incompatible source H overpopulation should not be preserved."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                    atom_payload("HS1", "H", Vec3(3.3, 0.8, 0.0)),
                    atom_payload("HS2", "H", Vec3(3.3, -0.8, 0.0)),
                    atom_payload("HS3", "H", Vec3(4.0, 0.0, 1.0)),
                    atom_payload("HS4", "H", Vec3(6.0, 0.8, 0.0)),
                    atom_payload("HS5", "H", Vec3(6.0, -0.8, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-evidence-overpopulated-source-h",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        chemistry_evidence=(
            RetainedNonPolymerChemistryOverride(
                residue_id=residue_id,
                smiles="CO",
                heavy_atom_names=("C1", "O1"),
            ).to_evidence(),
        ),
    )

    hydrogen_names = _hydrogen_atom_names(result.structure, residue_id)
    assert hydrogen_names == ("H001", "H002", "H003", "H004")
    assert all(
        not _has_topology_bond(
            result.structure,
            residue_id,
            "C1",
            source_h_name,
            provenance=BondProvenance.EVIDENCE_RESOLVED,
        )
        for source_h_name in ("HS1", "HS2", "HS3", "HS4", "HS5")
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_override_detects_swapped_same_element_mapping() -> None:
    """Evidence heavy-name mapping must not trust same-element order alone."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("C2", "C", Vec3(1.50, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(2.90, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-swapped-evidence-mapping",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        chemistry_evidence=(
            RetainedNonPolymerChemistryOverride(
                residue_id=residue_id,
                smiles="CCO",
                heavy_atom_names=("C2", "C1", "O1"),
            ).to_evidence(),
        ),
    )

    assert result.structure == structure
    assert result.repairs == ()
    assert any(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        and issue.residue_id == residue_id
        and "evidence atom mapping" in issue.message
        and "C1-O1" in issue.message
        for issue in result.issues
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_override_topology_takes_precedence_over_template(
) -> None:
    """Explicit evidence should own topology even when a component template exists."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(4.0, 1.2, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-template-overridden-by-evidence",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        chemistry_evidence=(
            RetainedNonPolymerChemistryOverride(
                residue_id=ResidueId("L", 1),
                smiles="CON",
                heavy_atom_names=("C1", "O1", "N1"),
            ).to_evidence(),
        ),
    )

    ligand = result.structure.constitution.ligands[0]
    assert "H1" not in ligand.atom_site_names()
    assert _has_topology_bond(
        result.structure,
        ResidueId("L", 1),
        "C1",
        "H001",
        provenance=BondProvenance.EVIDENCE_RESOLVED,
    )
    assert not _has_topology_bond(
        result.structure,
        ResidueId("L", 1),
        "C1",
        "H1",
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_add_retained_non_polymer_hydrogens_uses_rdkit_fallback_without_support() -> (
    None
):
    """Unsupported retained non-polymers should fall back to RDKit hydrogenation."""

    from rdkit import rdBase

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-rdkit-fallback-hydrogenation",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    hydrogenated_ligand = result.structure.constitution.ligands[0]
    added_hydrogen_names = tuple(
        atom_site.name
        for atom_site in hydrogenated_ligand.atom_sites
        if atom_site.element == "H"
    )
    assert added_hydrogen_names == ("H001", "H002", "H003", "H004")
    assert _has_topology_bond(
        result.structure,
        ResidueId("L", 1),
        "C1",
        "O1",
        provenance=BondProvenance.REPAIR_INFERRED,
    )
    after_facts = _retained_non_polymer_facts_by_residue_id(
        result.structure,
        component_library=build_retained_non_polymer_component_library(),
        component_ids=frozenset({"UNK"}),
    )
    assert (
        after_facts[ResidueId("L", 1)].heavy_atom_topology_availability_state
        is TopologyAvailabilityState.PRESENT
    )
    assert (
        after_facts[ResidueId("L", 1)].hydrogen_topology_availability_state
        is TopologyAvailabilityState.PRESENT
    )
    assert _has_topology_bond(
        result.structure,
        ResidueId("L", 1),
        "C1",
        "H001",
        provenance=BondProvenance.REPAIR_INFERRED,
    )
    assert _has_topology_bond(
        result.structure,
        ResidueId("L", 1),
        "O1",
        "H004",
        provenance=BondProvenance.REPAIR_INFERRED,
    )
    assert any(
        repair.kind is RepairEventKind.HYDROGENS_ADDED
        and repair.residue_id == ResidueId("L", 1)
        and repair.atom_names == added_hydrogen_names
        and repair.details is not None
        and "RDKit coordinate/proximity fallback" in repair.details
        and rdBase.rdkitVersion in repair.details
        for repair in result.repairs
    )
    fallback_issues = _fallback_used_issues(result.issues)
    assert len(fallback_issues) == 1
    assert not _fallback_blocked_issues(result.issues)
    assert fallback_issues[0].residue_id == ResidueId("L", 1)
    assert fallback_issues[0].severity is IssueSeverity.WARNING
    assert "L:1" in fallback_issues[0].message
    assert "UNK" in fallback_issues[0].message
    assert "RDKit coordinate/proximity fallback" in fallback_issues[0].message
    assert rdBase.rdkitVersion in fallback_issues[0].message
    pdb_conect_lines = tuple(
        line
        for line in write_structure_string(
            result.structure,
            FileFormat.PDB,
        ).splitlines()
        if line.startswith("CONECT")
    )
    assert any(line.startswith("CONECT    1    2") for line in pdb_conect_lines)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_fallback_preserves_partial_source_h_names() -> None:
    """Fallback completion should add missing Hs around preserved source Hs."""

    residue_id = ResidueId("L", 1)
    source_h_position = Vec3(3.3, 0.8, 0.0)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                    atom_payload("HSRC", "H", source_h_position),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-fallback-partial-source-h",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    hydrogen_names = _hydrogen_atom_names(result.structure, residue_id)
    assert "HSRC" in hydrogen_names
    assert len(hydrogen_names) == 4
    assert result.structure.residue_geometry(
        result.structure.constitution.residue_index(residue_id)
    ).position("HSRC") == source_h_position
    assert _has_topology_bond(
        result.structure,
        residue_id,
        "C1",
        "HSRC",
        provenance=BondProvenance.REPAIR_INFERRED,
    )
    assert any(
        repair.kind is RepairEventKind.HYDROGENS_ADDED
        and repair.residue_id == residue_id
        and "HSRC" not in repair.atom_names
        and len(repair.atom_names) == 3
        for repair in result.repairs
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_fallback_does_not_preserve_far_source_h() -> None:
    """Stale source H coordinates should not replace generated fallback Hs."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                    atom_payload("HSRC", "H", Vec3(40.0, 40.0, 40.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-fallback-far-source-h",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    hydrogen_names = _hydrogen_atom_names(result.structure, residue_id)
    assert "HSRC" not in hydrogen_names
    assert hydrogen_names == ("H001", "H002", "H003", "H004")
    assert not _has_topology_bond(
        result.structure,
        residue_id,
        "C1",
        "HSRC",
        provenance=BondProvenance.REPAIR_INFERRED,
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_source_h_reconciliation_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-H reconciliation should skip preservation beyond its pair cap."""

    monkeypatch.setattr(
        retained_non_polymer_repair,
        "_SOURCE_HYDROGEN_RECONCILIATION_CANDIDATE_LIMIT",
        0,
    )
    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                    atom_payload("HSRC", "H", Vec3(3.3, 0.8, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-source-h-reconciliation-bounded",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    hydrogen_names = _hydrogen_atom_names(result.structure, residue_id)
    assert "HSRC" not in hydrogen_names
    assert hydrogen_names == ("H001", "H002", "H003", "H004")


def test_strict_retained_non_polymer_mode_blocks_rdkit_fallback_and_preserves_source_h(
) -> None:
    """Strict fallback policy should leave source ligand atoms untouched."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                    atom_payload("HSRC", "H", Vec3(3.3, 0.8, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-strict-fallback-disabled",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        allow_retained_non_polymer_rdkit_fallback=False,
    )

    assert result.structure.constitution.ligands[0].atom_site_names() == (
        "C1",
        "O1",
        "HSRC",
    )
    assert not result.repairs
    assert not _has_topology_bond(
        result.structure,
        residue_id,
        "O1",
        "H004",
        provenance=BondProvenance.REPAIR_INFERRED,
    )
    assert not _fallback_used_issues(result.issues)
    fallback_issues = _fallback_blocked_issues(result.issues)
    assert len(fallback_issues) == 1
    assert fallback_issues[0].residue_id == residue_id
    assert fallback_issues[0].severity is IssueSeverity.WARNING
    assert "RDKit coordinate/proximity fallback is disabled" in (
        fallback_issues[0].message
    )


def test_retained_non_polymer_rdkit_fallback_unavailable_reports_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable RDKit fallback should be visible without flaky skip behavior."""

    residue_id = ResidueId("L", 1)
    monkeypatch.setattr(fallback_inference, "Chem", None)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                    atom_payload("HSRC", "H", Vec3(3.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-rdkit-unavailable-fallback",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    assert result.structure.constitution.ligands[0].atom_site_names() == (
        "C1",
        "O1",
        "HSRC",
    )
    assert result.repairs == ()
    assert not _fallback_used_issues(result.issues)
    assert any(
        issue.kind is ValidationIssueKind.MISSING_COMPONENT_DEFINITION
        and issue.residue_id == residue_id
        and "RDKit optional backend is unavailable" in issue.message
        and "RdkitUnavailableError" not in issue.message
        and "leaving retained non-polymer unchanged" in issue.message
        for issue in result.issues
    )


def test_retained_non_polymer_evidence_rdkit_unavailable_reports_contradiction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable RDKit evidence projection should remain a chemistry issue."""

    residue_id = ResidueId("L", 1)
    monkeypatch.setattr(rdkit_evidence, "Chem", None)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                    atom_payload("HSRC", "H", Vec3(3.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-rdkit-unavailable-evidence",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
        chemistry_evidence=(
            RetainedNonPolymerChemistryOverride(
                residue_id=residue_id,
                smiles="CO",
                heavy_atom_names=("C1", "O1"),
            ).to_evidence(),
        ),
    )

    assert result.structure == structure
    assert result.repairs == ()
    assert not _fallback_used_issues(result.issues)
    assert any(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        and issue.residue_id == residue_id
        and "RDKit optional backend is unavailable" in issue.message
        and "RdkitUnavailableError" not in issue.message
        and "chemistry evidence could not be projected" in issue.message
        for issue in result.issues
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_fallback_failure_stays_unsupported_hydrogenation(
) -> None:
    """Failed fallback should not look like successful fallback provenance."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.21, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-rdkit-fallback-failure",
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    assert result.structure.constitution.ligands[0].atom_site_names() == ("C1", "O1")
    assert not result.repairs
    assert not _fallback_used_issues(result.issues)
    assert any(
        issue.kind is ValidationIssueKind.MISSING_COMPONENT_DEFINITION
        and issue.residue_id == ResidueId("L", 1)
        and "RDKit fallback hydrogenation failed" in issue.message
        for issue in result.issues
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_hydrogen_topology_preserves_source_h_bond() -> None:
    """Regenerated ligand H topology should not overwrite source H bonds."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                    atom_payload("H001", "H", Vec3(3.6, 0.9, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-source-h-topology",
    )
    structure = _with_topology_bonds(
        structure,
        (
            TopologyBond(
                atom_index_1=structure.constitution.atom_index(
                    AtomRef(residue_id, "C1")
                ),
                atom_index_2=structure.constitution.atom_index(
                    AtomRef(residue_id, "O1")
                ),
                relationship_type=BondRelationshipType.COVALENT,
                provenance=BondProvenance.SOURCE_EXPLICIT,
                source_metadata=SourceBondMetadata(
                    record_type=SourceBondRecordType.PDB_CONECT,
                    source_id="CONECT",
                ),
            ),
            TopologyBond(
                atom_index_1=structure.constitution.atom_index(
                    AtomRef(residue_id, "C1")
                ),
                atom_index_2=structure.constitution.atom_index(
                    AtomRef(residue_id, "H001")
                ),
                relationship_type=BondRelationshipType.COVALENT,
                provenance=BondProvenance.SOURCE_EXPLICIT,
                source_metadata=SourceBondMetadata(
                    record_type=SourceBondRecordType.PDB_CONECT,
                    source_id="CONECT",
                ),
            ),
        ),
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    source_bond = _topology_bond_between(
        result.structure,
        residue_id,
        "C1",
        "H001",
    )
    assert source_bond is not None
    assert source_bond.provenance is BondProvenance.SOURCE_EXPLICIT
    assert source_bond.source_metadata == SourceBondMetadata(
        record_type=SourceBondRecordType.PDB_CONECT,
        source_id="CONECT",
    )
    source_heavy_bond = _topology_bond_between(
        result.structure,
        residue_id,
        "C1",
        "O1",
    )
    assert source_heavy_bond is not None
    assert source_heavy_bond.provenance is BondProvenance.SOURCE_EXPLICIT
    assert source_heavy_bond.source_metadata == SourceBondMetadata(
        record_type=SourceBondRecordType.PDB_CONECT,
        source_id="CONECT",
    )
    assert _has_topology_bond(
        result.structure,
        residue_id,
        "O1",
        "H004",
        provenance=BondProvenance.REPAIR_INFERRED,
    )
    fallback_issues = _fallback_used_issues(result.issues)
    assert len(fallback_issues) == 1
    assert fallback_issues[0].residue_id == residue_id


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_topology_conflict_skips_source_h_preservation() -> None:
    """Source H topology and generated H anchor must agree before preservation."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(5.4, 0.0, 0.0)),
                    atom_payload("HSRC", "H", Vec3(5.8, 0.8, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-source-h-anchor-conflict",
    )
    structure = _with_topology_bonds(
        structure,
        (
            TopologyBond(
                atom_index_1=structure.constitution.atom_index(
                    AtomRef(residue_id, "C1")
                ),
                atom_index_2=structure.constitution.atom_index(
                    AtomRef(residue_id, "HSRC")
                ),
                relationship_type=BondRelationshipType.COVALENT,
                provenance=BondProvenance.SOURCE_EXPLICIT,
                source_metadata=SourceBondMetadata(
                    record_type=SourceBondRecordType.PDB_CONECT,
                    source_id="CONECT",
                ),
            ),
        ),
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    assert "HSRC" not in _hydrogen_atom_names(result.structure, residue_id)
    assert _topology_bond_between(result.structure, residue_id, "C1", "HSRC") is None
    assert not _has_topology_bond(
        result.structure,
        residue_id,
        "O1",
        "HSRC",
        provenance=BondProvenance.REPAIR_INFERRED,
    )
    assert _has_topology_bond(
        result.structure,
        residue_id,
        "O1",
        "H004",
        provenance=BondProvenance.REPAIR_INFERRED,
    )
    assert "HSRC" not in write_structure_string(result.structure, FileFormat.PDB)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_retained_non_polymer_fallback_projects_existing_h_names_by_geometry() -> None:
    """Existing H names should map by coordinates, not source atom order."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                    atom_payload("HO", "H", Vec3(1.95, 0.75, 0.0)),
                    atom_payload("HC1", "H", Vec3(-0.65, 0.80, 0.0)),
                    atom_payload("HC2", "H", Vec3(-0.65, -0.80, 0.0)),
                    atom_payload("HC3", "H", Vec3(0.0, 0.0, 1.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-existing-h-name-projection",
    )

    component_library = build_retained_non_polymer_component_library()

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=component_library,
    )

    assert set(result.structure.constitution.ligands[0].atom_site_names()) == {
        "C1",
        "O1",
        "HC1",
        "HC2",
        "HC3",
        "HO",
    }
    assert _has_topology_bond(
        result.structure,
        residue_id,
        "O1",
        "HO",
        provenance=BondProvenance.REPAIR_INFERRED,
    )
    assert not _has_topology_bond(
        result.structure,
        residue_id,
        "C1",
        "HO",
        provenance=BondProvenance.REPAIR_INFERRED,
    )
    assert all(
        _has_topology_bond(
            result.structure,
            residue_id,
            "C1",
            hydrogen_atom_name,
            provenance=BondProvenance.REPAIR_INFERRED,
        )
        for hydrogen_atom_name in ("HC1", "HC2", "HC3")
    )
    after_facts = _retained_non_polymer_facts_by_residue_id(
        result.structure,
        component_library=component_library,
        component_ids=frozenset({"UNK"}),
    )
    assert after_facts[residue_id].hydrogen_coverage_state.value == "complete"
    assert not after_facts[residue_id].requires_hydrogen_completion()


def _retained_non_polymer_facts_by_residue_id(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    component_ids: frozenset[str] | None = None,
) -> dict[ResidueId, RetainedNonPolymerChemistryReadinessFact]:
    """Return retained non-polymer facts keyed by residue id."""

    _, chemistry_facts = derive_structure_coverage_and_chemistry_readiness_facts(
        structure,
        component_library=component_library,
    )
    return {
        fact.residue_id: fact
        for fact in chemistry_facts.retained_non_polymer_facts
        if component_ids is None or fact.component_id in component_ids
    }


def _fallback_used_issues(
    issues: tuple[ValidationIssue, ...],
) -> tuple[ValidationIssue, ...]:
    """Return retained ligand fallback-used warning issues."""

    return tuple(
        issue
        for issue in issues
        if issue.kind is ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_USED
    )


def _fallback_blocked_issues(
    issues: tuple[ValidationIssue, ...],
) -> tuple[ValidationIssue, ...]:
    """Return strict-policy retained ligand fallback-blocked warning issues."""

    return tuple(
        issue
        for issue in issues
        if issue.kind is ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_BLOCKED
    )


def _hydrogen_atom_names(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> tuple[str, ...]:
    """Return retained ligand hydrogen atom names in residue order."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    return tuple(
        atom_site.name
        for atom_site in residue_site.atom_sites
        if atom_site.element == "H"
    )


def _with_topology_bonds(
    structure: ProteinStructure,
    bonds: tuple[TopologyBond, ...],
) -> ProteinStructure:
    """Return one test structure with replacement topology bonds."""

    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=bonds,
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def _topology_bond_between(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
) -> TopologyBond | None:
    """Return one residue-local topology bond when present."""

    atom_index_1 = structure.constitution.resolve_atom_index(
        AtomRef(residue_id, atom_name_1)
    )
    atom_index_2 = structure.constitution.resolve_atom_index(
        AtomRef(residue_id, atom_name_2)
    )
    if atom_index_1 is None or atom_index_2 is None:
        return None

    return structure.topology.bond_between(atom_index_1, atom_index_2)


def _has_topology_bond(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
    *,
    provenance: BondProvenance,
) -> bool:
    """Return whether one residue-local covalent topology bond exists."""

    bond = _topology_bond_between(
        structure,
        residue_id,
        atom_name_1,
        atom_name_2,
    )
    return (
        bond is not None
        and bond.relationship_type is BondRelationshipType.COVALENT
        and bond.provenance is provenance
    )


def _has_pdb_conect_between(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
) -> bool:
    """Return whether PDB egress emits one residue-local CONECT pair."""

    pdb_text = write_structure_string(structure, FileFormat.PDB)
    serial_by_atom_ref = pdb_atom_serial_by_atom_ref(pdb_text)
    serial_1 = serial_by_atom_ref[AtomRef(residue_id, atom_name_1)]
    serial_2 = serial_by_atom_ref[AtomRef(residue_id, atom_name_2)]
    return frozenset((serial_1, serial_2)) in _pdb_conect_serial_pairs(pdb_text)


def _pdb_conect_serial_pairs(pdb_text: str) -> frozenset[frozenset[int]]:
    """Return unordered serial pairs emitted in PDB CONECT records."""

    pairs: set[frozenset[int]] = set()
    for line in pdb_text.splitlines():
        if not line.startswith("CONECT"):
            continue

        source_serial = int(line[6:11])
        neighbor_serials = tuple(
            int(raw_serial)
            for start in range(11, len(line), 5)
            for raw_serial in (line[start : start + 5].strip(),)
            if raw_serial
        )
        pairs.update(
            frozenset((source_serial, neighbor_serial))
            for neighbor_serial in neighbor_serials
        )

    return frozenset(pairs)
