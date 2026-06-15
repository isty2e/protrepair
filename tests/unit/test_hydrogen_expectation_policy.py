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

from protrepair.chemistry import (
    build_default_component_library,
)
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state.hydrogen_expectation import (
    RetainedNonPolymerChemistryEvidenceSource,
    derive_structure_hydrogen_expectation_model,
)
from protrepair.structure.labels import ResidueId

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - optional dependency
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


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires RDKit fallback chemistry")
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
