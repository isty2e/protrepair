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
from protrepair.diagnostics.chemistry_contradictions import (
    diagnose_retained_non_polymer_template_chemistry_contradictions,
)
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import (
    RetainedNonPolymerChemistryReadinessFact,
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
    assert result.issues == ()
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
        for repair in result.repairs
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
    assert _has_topology_bond(
        result.structure,
        residue_id,
        "O1",
        "H004",
        provenance=BondProvenance.REPAIR_INFERRED,
    )


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

    for bond in structure.topology.bonds:
        if {bond.atom_index_1, bond.atom_index_2} == {atom_index_1, atom_index_2}:
            return bond

    return None


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
