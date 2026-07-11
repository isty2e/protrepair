"""Unit tests for structure hydrogen expectation read-model policy."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.retained_non_polymer_components import (
    build_retained_non_polymer_component_library,
)

import protrepair.state.retained_non_polymer_chemistry as retained_non_polymer_chemistry
from protrepair.chemistry import (
    build_default_component_library,
)
from protrepair.chemistry.inference import (
    retained_non_polymer_evidence as evidence_inference,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import (
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.state.hydrogen_expectation import (
    derive_structure_hydrogen_expectation_model,
)
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - required dependency import guard
    Chem = None

RDKIT_AVAILABLE = Chem is not None


def test_hydrogen_expectation_model_adds_polymer_backbone_hydrogens() -> None:
    """Polymer expectation policy should include chain-aware backbone hydrogens."""

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
                            atom_payload("CA", "C", Vec3(1.5, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.5, 1.5, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(3.0, 1.3, 0.0)),
                            atom_payload("CA", "C", Vec3(4.3, 1.3, 0.0)),
                            atom_payload("C", "C", Vec3(5.6, 1.3, 0.0)),
                            atom_payload("O", "O", Vec3(6.6, 1.3, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="polymer-backbone-hydrogen-expectation",
    )

    model = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=build_default_component_library(),
    )

    first_residue_expectation = set(
        model.expected_hydrogen_atom_names_by_residue[ResidueId("A", 1)]
    )
    second_residue_expectation = set(
        model.expected_hydrogen_atom_names_by_residue[ResidueId("A", 2)]
    )

    assert {"H1", "H2", "H3"}.issubset(first_residue_expectation)
    assert "H" in second_residue_expectation


def test_hydrogen_expectation_model_suppresses_disulfide_hg() -> None:
    """Disulfide-bonded cysteines should not expect thiol HG hydrogens."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="CYS",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.5, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.5, 1.5, 0.0)),
                            atom_payload("SG", "S", Vec3(2.6, 2.5, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="CYS",
                        residue_id=ResidueId("A", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(3.0, 1.3, 0.0)),
                            atom_payload("CA", "C", Vec3(4.3, 1.3, 0.0)),
                            atom_payload("C", "C", Vec3(5.6, 1.3, 0.0)),
                            atom_payload("O", "O", Vec3(6.6, 1.3, 0.0)),
                            atom_payload("CB", "C", Vec3(4.3, 2.8, 0.0)),
                            atom_payload("SG", "S", Vec3(3.4, 2.5, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="disulfide-hydrogen-suppression",
    )
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(
                        AtomRef(ResidueId("A", 1), "SG")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(ResidueId("A", 2), "SG")
                    ),
                    relationship_type=BondRelationshipType.DISULFIDE,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )

    model = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=build_default_component_library(),
    )

    assert "HG" not in set(
        model.expected_hydrogen_atom_names_by_residue[ResidueId("A", 1)]
    )
    assert "HG" not in set(
        model.expected_hydrogen_atom_names_by_residue[ResidueId("A", 2)]
    )


def test_hydrogen_expectation_model_uses_override_for_unknown_ligand() -> None:
    """Unknown retained ligands should resolve hydrogen expectation from override."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="override-backed-hydrogen-expectation",
    )

    model = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=build_default_component_library(),
        retained_non_polymer_chemistry_evidence=(
            RetainedNonPolymerChemistryOverride(
                residue_id=ResidueId("L", 1),
                smiles="CO",
                heavy_atom_names=("C1", "O1"),
            ).to_evidence(),
        ),
    )
    resolution = model.resolution_for_retained_non_polymer(ResidueId("L", 1))

    assert resolution.source is (
        RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
    )
    assert len(resolution.expected_hydrogen_atom_names) > 0
    assert resolution.heavy_atom_elements == ("C", "O")
    assert {
        frozenset((bond.atom_name_1, bond.atom_name_2))
        for bond in resolution.heavy_bond_definitions
    } == {frozenset(("C1", "O1"))}
    assert resolution.hydrogen_bond_definitions
    assert not resolution.failure_reason


def test_hydrogen_expectation_model_propagates_no_rdkit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required RDKit capability failure should not become unresolved chemistry."""

    monkeypatch.setattr(evidence_inference, "Chem", None)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="no-rdkit-override-hydrogen-expectation",
    )

    with pytest.raises(RdkitUnavailableError, match="operational RDKit installation"):
        derive_structure_hydrogen_expectation_model(
            structure,
            component_library=build_default_component_library(),
            retained_non_polymer_chemistry_evidence=(
                RetainedNonPolymerChemistryOverride(
                    residue_id=ResidueId("L", 1),
                    smiles="CO",
                    heavy_atom_names=("C1", "O1"),
                ).to_evidence(),
            ),
        )


def test_hydrogen_expectation_model_propagates_no_rdkit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback capability failure should not become unresolved chemistry."""

    def fail_infer_fallback(*args, **kwargs):
        raise RdkitUnavailableError("operational RDKit installation is unavailable")

    monkeypatch.setattr(
        retained_non_polymer_chemistry,
        "infer_retained_non_polymer_rdkit_fallback",
        fail_infer_fallback,
    )
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="no-rdkit-fallback-hydrogen-expectation",
    )

    with pytest.raises(RdkitUnavailableError, match="operational RDKit installation"):
        derive_structure_hydrogen_expectation_model(
            structure,
            component_library=build_default_component_library(),
        )


def test_hydrogen_expectation_model_respects_disabled_rdkit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict retained-ligand policy should not precompute RDKit fallback facts."""

    def fail_infer_fallback(*args, **kwargs):
        raise AssertionError("RDKit fallback should not be called")

    monkeypatch.setattr(
        retained_non_polymer_chemistry,
        "infer_retained_non_polymer_rdkit_fallback",
        fail_infer_fallback,
    )
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="fallback-disabled-hydrogen-expectation",
    )

    model = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    resolution = model.resolution_for_retained_non_polymer(ResidueId("L", 1))

    assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    assert resolution.expected_hydrogen_atom_names == ()
    assert not resolution.heavy_bond_definitions
    assert not resolution.hydrogen_bond_definitions
    assert resolution.failure_reason == "RDKit fallback is disabled"


def test_hydrogen_expectation_model_marks_metal_context_not_applicable() -> None:
    """Single-center retained metals should resolve as fallback with no Hs."""

    component_library = build_default_component_library()
    for template in build_retained_non_polymer_component_library().templates.values():
        component_library = component_library.with_template(template)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="ION",
                residue_id=ResidueId("L", 1),
                atoms=(atom_payload("ZN", "Zn", Vec3(0.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="metal-not-applicable-hydrogen-expectation",
    )

    model = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=component_library,
    )
    resolution = model.resolution_for_retained_non_polymer(ResidueId("L", 1))

    assert resolution.source is (
        RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    )
    assert resolution.expected_hydrogen_atom_names == ()


def test_retained_fallback_resolution_owns_topology_and_projection_facts() -> None:
    """Fallback resolution should own coordinate-free topology facts."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="fallback-resolution-topology-facts",
    )

    model = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=build_default_component_library(),
    )
    resolution = model.resolution_for_retained_non_polymer(ResidueId("L", 1))

    assert resolution.source is (
        RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    )
    assert resolution.expected_hydrogen_atom_names == (
        "H001",
        "H002",
        "H003",
        "H004",
    )
    assert {
        frozenset((bond.atom_name_1, bond.atom_name_2))
        for bond in resolution.heavy_bond_definitions
    } == {frozenset(("C1", "O1"))}
    assert {
        frozenset((bond.atom_name_1, bond.atom_name_2))
        for bond in resolution.hydrogen_bond_definitions
    } == {
        frozenset(("C1", "H001")),
        frozenset(("C1", "H002")),
        frozenset(("C1", "H003")),
        frozenset(("O1", "H004")),
    }
    assert resolution.hydrogen_name_projection_candidate_count > 0
    assert resolution.hydrogen_name_projection_candidate_limit > 0
    assert not resolution.failure_reason


def test_retained_readiness_reuses_single_fallback_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness should not replay fallback inference after resolution."""

    calls = 0
    original_infer_fallback = (
        retained_non_polymer_chemistry.infer_retained_non_polymer_rdkit_fallback
    )

    def counting_infer_fallback(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_infer_fallback(*args, **kwargs)

    monkeypatch.setattr(
        retained_non_polymer_chemistry,
        "infer_retained_non_polymer_rdkit_fallback",
        counting_infer_fallback,
    )
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="fallback-resolution-readiness-reuse",
    )

    _, chemistry_readiness = derive_structure_coverage_and_chemistry_readiness_facts(
        structure,
        component_library=build_default_component_library(),
    )

    retained_fact = chemistry_readiness.retained_non_polymer_facts[0]
    assert retained_fact.heavy_topology_source is (
        RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    )
    assert retained_fact.hydrogen_expectation_source is (
        RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    )
    assert calls == 1
