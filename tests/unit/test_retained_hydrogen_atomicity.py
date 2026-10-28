"""Atomic admission of retained-ligand hydrogen placement."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)
from tests.support.refinement_benchmarks import resolve_fixture_path
from tests.support.refinement_cases import EXPLORATORY_REFINEMENT_FIXTURE_SOURCES
from tests.support.retained_non_polymer_components import (
    build_retained_non_polymer_component_library,
)

from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure, write_structure_string
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen.cleanup import (
    cleanup_residual_hydrogen_clashes,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload


def _incomplete_cysteine(element: str = "H") -> ProteinStructure:
    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="CYS",
                residue_id=residue_id,
                is_hetero=True,
                atoms=(
                    atom_payload("N", "N", Vec3(0, 1, 0), formal_charge=1),
                    atom_payload("CA", "C", Vec3(0, 0, 0)),
                    atom_payload("CB", "C", Vec3(1, 0, 0)),
                    atom_payload("SG", "S", Vec3(1, 1, 1)),
                    *(
                        atom_payload(
                            name,
                            element,
                            Vec3(0, 2, index * 0.1),
                            occupancy=0.75,
                            b_factor=12.5,
                            formal_charge=0,
                            altloc="B",
                        )
                        for index, name in enumerate(("H1", "H2", "H3", "HB2", "HB3"))
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(
                        AtomRef(residue_id, "N")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(residue_id, "H1")
                    ),
                    order=1,
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


@pytest.mark.parametrize("element", ["H", "D", "T"])
def test_incomplete_template_does_not_replace_source_hydrogens(element: str) -> None:
    structure = _incomplete_cysteine(element)

    result = add_retained_non_polymer_hydrogens(structure)

    assert result.structure.constitution == structure.constitution
    assert result.structure.geometry == structure.geometry
    assert result.structure.topology == structure.topology
    assert result.repairs == ()
    assert any(
        issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        and issue.residue_id == ResidueId("L", 1)
        for issue in result.issues
    )
    for file_format in (FileFormat.PDB, FileFormat.MMCIF):
        assert write_structure_string(
            result.structure, file_format=file_format
        ) == write_structure_string(structure, file_format=file_format)

    repeated = add_retained_non_polymer_hydrogens(result.structure)
    assert repeated.structure == result.structure
    assert repeated.repairs == ()
    assert repeated.issues == result.issues


def test_missing_static_placement_anchor_reports_unchanged_residue() -> None:
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 0, 0)),
                    atom_payload("O1", "O", Vec3(1.2, 0, 0)),
                    atom_payload("HSRC", "H", Vec3(0, 1, 0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )

    result = add_retained_non_polymer_hydrogens(
        structure,
        component_library=build_retained_non_polymer_component_library(),
    )

    assert result.structure == structure
    assert result.repairs == ()
    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        for issue in result.issues
    )


def test_incomplete_template_can_add_anchored_h_without_replacing_source() -> None:
    structure = _incomplete_cysteine().without_hydrogens_in_residues(
        frozenset({ResidueId("L", 1)})
    )

    result = add_retained_non_polymer_hydrogens(
        structure, component_library=build_default_component_library()
    )

    assert result.structure.constitution.ligands[0].atom_site_names() == (
        "N",
        "CA",
        "CB",
        "SG",
        "HA",
        "HB1",
        "HB2",
        "HG",
    )
    assert result.repairs
    assert not any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        for issue in result.issues
    )


def test_incomplete_amino_acid_keeps_terminal_h_while_preparing_fixed_body() -> None:
    source = _incomplete_cysteine()
    residue_id = ResidueId("L", 1)
    index = source.constitution.residue_index(residue_id)
    residue = source.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    excluded = {"H2", "H3", "HB2", "HB3"}
    structure = source.with_updated_residue_facets_batch(
        (
            (
                residue.without_atom_sites(excluded),
                source.residue_geometry(index).without_atoms(excluded),
                tuple(
                    (name, charge)
                    for name, charge in source.residue_formal_charge_by_atom_name(index)
                    if name not in excluded
                ),
            ),
        )
    )

    result = add_retained_non_polymer_hydrogens(structure)

    repaired_index = result.structure.constitution.residue_index(residue_id)
    assert result.structure.constitution.residue_site_at(repaired_index).atom_site(
        "H1"
    ) == residue.atom_site("H1")
    assert result.structure.residue_geometry(repaired_index).atom_geometry(
        "H1"
    ) == source.residue_geometry(index).atom_geometry("H1")
    assert {"HA", "HB1", "HB2", "HG"} <= set(
        result.structure.constitution.residue_site_at(repaired_index).atom_site_names()
    )
    assert result.repairs
    assert any(
        issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        and issue.residue_id == residue_id
        for issue in result.issues
    )


@pytest.mark.parametrize("invalid_evidence", [False, True])
def test_failed_ligand_is_excluded_from_cleanup_in_mixed_batch(
    invalid_evidence: bool,
) -> None:
    failed_id = ResidueId("L", 2)
    library = build_retained_non_polymer_component_library().with_template(
        build_default_component_library().require("SER")
    )
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                is_hetero=True,
                atoms=(
                    atom_payload("C1", "C", Vec3(1, 2.1, 0)),
                    atom_payload("O1", "O", Vec3(2.2, 2.1, 0)),
                    atom_payload("N1", "N", Vec3(1, 3.3, 0)),
                ),
            ),
            residue_payload(
                component_id="SER",
                residue_id=failed_id,
                is_hetero=True,
                atoms=(
                    atom_payload("CA", "C", Vec3(0, 0, 0)),
                    atom_payload("CB", "C", Vec3(1, 0, 0)),
                    atom_payload("OG", "O", Vec3(1, 1, 0)),
                    atom_payload("HG", "H", Vec3(1, 2, 0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
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
                        AtomRef(failed_id, "OG")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(failed_id, "HG")
                    ),
                    order=1,
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    failed_index = structure.constitution.residue_index(failed_id)
    source_geometry = structure.residue_geometry(failed_index)
    unrestricted, _ = cleanup_residual_hydrogen_clashes(
        structure, component_library=library
    )
    assert unrestricted.residue_geometry(failed_index) != source_geometry

    evidence = (
        (
            RetainedNonPolymerChemistryOverride(
                residue_id=failed_id,
                smiles="CO",
                heavy_atom_names=("CB", "OG"),
            ).to_evidence(),
        )
        if invalid_evidence
        else ()
    )
    result = add_retained_non_polymer_hydrogens(
        structure, component_library=library, chemistry_evidence=evidence
    )

    assert result.structure.constitution.ligands[0].has_atom_site("H1")
    assert result.structure.constitution.ligands[1] == structure.constitution.ligands[1]
    assert result.structure.residue_geometry(failed_index) == source_geometry
    assert tuple(repair.residue_id for repair in result.repairs) == (ResidueId("L", 1),)
    source_bonds = tuple(
        bond
        for bond in result.structure.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
    )
    assert len(source_bonds) == 1
    assert tuple(
        result.structure.constitution.atom_ref_at(index)
        for index in source_bonds[0].endpoint_pair()
    ) == (AtomRef(failed_id, "OG"), AtomRef(failed_id, "HG"))
    assert any(
        issue.kind
        is (
            ValidationIssueKind.CHEMISTRY_CONTRADICTION
            if invalid_evidence
            else ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        )
        and issue.residue_id == failed_id
        for issue in result.issues
    )


@pytest.mark.parametrize("unmatched_source_h", [False, True])
def test_partial_rigid_template_admission_depends_on_source_preservation(
    unmatched_source_h: bool,
) -> None:
    structure = read_structure(
        resolve_fixture_path(
            EXPLORATORY_REFINEMENT_FIXTURE_SOURCES[
                "2h6o-glycan-problematic"
            ].output_path
        )
    )
    residue_id = structure.constitution.ligands[0].residue_id
    targets = frozenset({residue_id})
    source = add_retained_non_polymer_hydrogens(
        structure, target_residue_ids=targets
    ).structure
    index = source.constitution.residue_index(residue_id)
    residue = source.constitution.residue_site_at(index)
    assert not residue.has_atom_site("O1")
    if unmatched_source_h:
        payload = CompletionResiduePayload(
            residue_site=residue,
            residue_geometry=source.residue_geometry(index),
            formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(index),
        )
        atom, geometry, charge = atom_payload("HSRC", "D", Vec3(0, 0, 0))
        payload = payload.with_atom_payload(
            atom, atom_geometry=geometry, formal_charge=charge
        )
        source = source.with_updated_residue_facets_batch(
            (
                (
                    payload.residue_site,
                    payload.residue_geometry,
                    payload.formal_charge_by_atom_name,
                ),
            )
        )

    result = add_retained_non_polymer_hydrogens(source, target_residue_ids=targets)

    assert result.structure == source
    assert result.repairs == ()
    assert (
        any(
            issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
            and issue.residue_id == residue_id
            for issue in result.issues
        )
        is unmatched_source_h
    )
