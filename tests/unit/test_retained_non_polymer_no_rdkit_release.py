"""Release-gate tests for retained ligands without optional RDKit support."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import (
    ComponentSupportState,
    RetainedNonPolymerChemistryEvidenceSource,
    TopologyAvailabilityState,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.state.hydrogen_expectation import (
    derive_structure_hydrogen_expectation_model,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised in lean CI
    Chem = None

pytestmark = pytest.mark.skipif(
    Chem is not None,
    reason="exercises the actual no-RDKit release lane",
)

LIGAND_RESIDUE_ID = ResidueId("A", 99)


def test_no_rdkit_readiness_marks_template_less_retained_ligand_unresolved() -> None:
    """Readiness should not claim fallback support when RDKit is unavailable."""

    structure = _template_less_retained_ligand_structure()
    library = build_default_component_library()

    expectation_model = derive_structure_hydrogen_expectation_model(
        structure,
        component_library=library,
    )
    resolution = expectation_model.resolution_for_retained_non_polymer(
        LIGAND_RESIDUE_ID
    )
    _, chemistry_readiness = derive_structure_coverage_and_chemistry_readiness_facts(
        structure,
        component_library=library,
    )
    retained_fact = chemistry_readiness.retained_non_polymer_facts[0]

    assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    assert resolution.expected_hydrogen_atom_names == ()
    assert "optional rdkit dependency" in resolution.failure_reason
    assert (
        retained_fact.component_support_state
        is ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
    )
    assert (
        retained_fact.heavy_topology_source
        is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    )
    assert (
        retained_fact.hydrogen_expectation_source
        is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    )
    assert (
        retained_fact.heavy_atom_topology_availability_state
        is TopologyAvailabilityState.ABSENT
    )
    assert (
        retained_fact.hydrogen_topology_availability_state
        is TopologyAvailabilityState.ABSENT
    )


def test_no_rdkit_override_readiness_degrades_without_projection_crash() -> None:
    """Unresolved explicit evidence should not be projected as resolved topology."""

    _, chemistry_readiness = derive_structure_coverage_and_chemistry_readiness_facts(
        _template_less_retained_ligand_structure(),
        component_library=build_default_component_library(),
        retained_non_polymer_chemistry_evidence=(_override_evidence(),),
    )
    retained_fact = chemistry_readiness.retained_non_polymer_facts[0]

    assert (
        retained_fact.component_support_state
        is ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
    )
    assert (
        retained_fact.heavy_topology_source
        is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    )
    assert (
        retained_fact.hydrogen_expectation_source
        is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    )


def test_no_rdkit_retained_ligand_transformer_reports_stable_fallback_issues() -> None:
    """Direct retained-ligand completion should expose stable no-backend outcomes."""

    structure = _template_less_retained_ligand_structure()
    library = build_default_component_library()

    permissive_result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=library,
        allow_retained_non_polymer_rdkit_fallback=True,
    )
    strict_result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=library,
        allow_retained_non_polymer_rdkit_fallback=False,
    )

    assert _retained_ligand_hydrogen_names(permissive_result.structure) == ()
    assert _retained_ligand_hydrogen_names(strict_result.structure) == ()
    assert tuple(issue.kind for issue in permissive_result.issues) == (
        ValidationIssueKind.MISSING_COMPONENT_DEFINITION,
    )
    assert "RDKit optional backend is unavailable" in permissive_result.issues[
        0
    ].message
    assert tuple(issue.kind for issue in strict_result.issues) == (
        ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_BLOCKED,
    )
    assert not permissive_result.has_errors()
    assert not strict_result.has_errors()


def test_no_rdkit_public_override_validation_uses_stable_input_error() -> None:
    """Public explicit overrides should fail at ingress without private errors."""

    with pytest.raises(
        ValueError,
        match=(
            "retained non-polymer chemistry override validation requires "
            "optional RDKit support for A:99"
        ),
    ):
        process_structure(
            _template_less_retained_ligand_structure(),
            ingress=StructureIngressOptions(
                ligand_policy=LigandPolicy.KEEP,
                retained_non_polymer_chemistry_overrides=(
                    RetainedNonPolymerChemistryOverride(
                        residue_id=LIGAND_RESIDUE_ID,
                        smiles="CO",
                        heavy_atom_names=("C1", "O1"),
                    ),
                ),
            ),
        )


def _template_less_retained_ligand_structure() -> ProteinStructure:
    """Return one unknown retained ligand that needs RDKit for fallback support."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.4, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_RESIDUE_ID,
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="release-no-rdkit-retained-ligand",
    )


def _override_evidence() -> RetainedNonPolymerChemistryEvidence:
    """Return explicit CO chemistry evidence for the retained test ligand."""

    return RetainedNonPolymerChemistryOverride(
        residue_id=LIGAND_RESIDUE_ID,
        smiles="CO",
        heavy_atom_names=("C1", "O1"),
    ).to_evidence()


def _retained_ligand_hydrogen_names(structure: ProteinStructure) -> tuple[str, ...]:
    """Return retained-ligand hydrogen atom names from the test fixture."""

    return tuple(
        atom_site.name
        for atom_site in structure.constitution.ligands[0].atom_sites
        if atom_site.element == "H"
    )
