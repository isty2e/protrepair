"""Readiness must use current evidence rather than a borrowed interpretation."""

from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state.hydrogen_expectation import (
    derive_structure_hydrogen_expectation_model,
)
from protrepair.state.structure_readiness import (
    derive_projection_coverage_and_chemistry_readiness_facts,
)
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)


def test_borrowed_hydrogen_expectations_do_not_hide_changed_ligand_evidence():
    residue_id = ResidueId("L", 1)
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )
    old_evidence, new_evidence = (
        (
            RetainedNonPolymerChemistryOverride(
                residue_id=residue_id,
                smiles=smiles,
                heavy_atom_names=("C1", "O1"),
            ).to_evidence(),
        )
        for smiles in ("CO", "CN")
    )
    library = build_default_component_library()
    prepared = add_retained_non_polymer_hydrogens(
        source, component_library=library, chemistry_evidence=old_evidence
    )
    assert not prepared.has_errors()
    source = prepared.structure
    borrowed = derive_structure_hydrogen_expectation_model(
        source,
        component_library=library,
        retained_non_polymer_chemistry_evidence=old_evidence,
    )
    _, expected = derive_projection_coverage_and_chemistry_readiness_facts(
        context_structure=source,
        residues=(),
        ligands=source.constitution.ligands,
        component_library=library,
        retained_non_polymer_chemistry_evidence=new_evidence,
    )
    _, observed = derive_projection_coverage_and_chemistry_readiness_facts(
        context_structure=source,
        residues=(),
        ligands=source.constitution.ligands,
        component_library=library,
        retained_non_polymer_chemistry_evidence=new_evidence,
        hydrogen_expectation_model=borrowed,
    )
    assert (
        expected.retained_non_polymer_facts[0].heavy_atom_topology_availability_state
        is TopologyAvailabilityState.ABSENT
    )
    assert observed == expected
