"""Graph-only repair attribution in mixed retained-ligand batches."""

from math import cos, pi, sin

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.geometry import Vec3
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
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
    resolve_retained_non_polymer_topology,
)


def _benzene_batch(
    *, complete_first: bool, isotope: str
) -> tuple[ProteinStructure, ResidueId, ResidueId]:
    complete_id = ResidueId("L", 1 if complete_first else 2)
    incomplete_id = ResidueId("L", 2 if complete_first else 1)
    structure = build_structure(
        chains=(),
        ligands=tuple(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                is_hetero=True,
                atoms=(
                    *(
                        atom_payload(
                            f"C{i + 1}",
                            "C",
                            Vec3(x + 1.4 * cos(i * pi / 3), 1.4 * sin(i * pi / 3), 0),
                        )
                        for i in range(6)
                    ),
                    *(
                        atom_payload(
                            f"H{i + 1}",
                            isotope,
                            Vec3(x + 2.49 * cos(i * pi / 3), 2.49 * sin(i * pi / 3), 0),
                            occupancy=0.75,
                            b_factor=12.5,
                        )
                        for i in range(6)
                        if residue_id == complete_id
                    ),
                ),
            )
            for residue_id, x in ((ResidueId("L", 1), 0), (ResidueId("L", 2), 10))
        ),
        source_format=FileFormat.PDB,
    )
    bonds = []
    for residue in structure.constitution.ligands:
        pairs = [(f"C{i + 1}", f"C{(i + 1) % 6 + 1}") for i in range(6)]
        if residue.residue_id == complete_id:
            pairs.extend((f"C{i + 1}", f"H{i + 1}") for i in range(6))
        for left, right in pairs:
            bonds.append(
                TopologyBond(
                    structure.constitution.atom_index(
                        AtomRef(residue.residue_id, left)
                    ),
                    structure.constitution.atom_index(
                        AtomRef(residue.residue_id, right)
                    ),
                    order=None,
                    relationship_type=BondRelationshipType.UNKNOWN,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_CONECT,
                    ),
                )
            )
    return (
        ProteinStructure.from_payload(
            constitution=structure.constitution,
            geometry=structure.geometry,
            topology=StructureTopology(
                constitution=structure.constitution,
                atom_topologies=structure.topology.atom_topologies,
                bonds=tuple(bonds),
            ),
            polymer_blueprint=structure.polymer_blueprint,
            provenance=structure.provenance,
        ),
        complete_id,
        incomplete_id,
    )


@pytest.mark.parametrize("complete_first", (False, True))
@pytest.mark.parametrize("pre_resolved", (False, True))
@pytest.mark.parametrize("isotope", ("H", "D"))
def test_mixed_ligand_batch_attributes_only_actual_graph_only_repairs(
    complete_first: bool, pre_resolved: bool, isotope: str
) -> None:
    source, complete_id, incomplete_id = _benzene_batch(
        complete_first=complete_first, isotope=isotope
    )
    library = ComponentLibrary({})
    if pre_resolved:
        source = resolve_retained_non_polymer_topology(
            source, library, target_residue_ids=frozenset((complete_id,))
        ).structure
    index = source.constitution.residue_index(complete_id)
    result = add_retained_non_polymer_hydrogens(source, library)
    final_index = result.structure.constitution.residue_index(complete_id)
    assert result.structure.constitution.residue_site_at(final_index) == (
        source.constitution.residue_site_at(index)
    )
    assert result.structure.residue_geometry(final_index) == source.residue_geometry(
        index
    )
    graph_events = tuple(
        event
        for event in result.repairs
        if event.kind is RepairEventKind.RETAINED_NON_POLYMER_TOPOLOGY_RESOLVED
    )
    assert tuple(event.residue_id for event in graph_events) == (
        () if pre_resolved else (complete_id,)
    )
    if graph_events:
        assert set(graph_events[0].atom_names) == {
            *(f"C{i + 1}" for i in range(6)),
            *(f"H{i + 1}" for i in range(6)),
        }
    assert tuple(
        event.residue_id
        for event in result.repairs
        if event.kind is RepairEventKind.HYDROGENS_ADDED
    ) == (incomplete_id,)
    assert add_retained_non_polymer_hydrogens(result.structure, library).repairs == ()


@pytest.mark.parametrize("select_complete", (False, True))
def test_mixed_ligand_batch_reports_only_selected_repairs(
    select_complete: bool,
) -> None:
    source, complete_id, incomplete_id = _benzene_batch(
        complete_first=False, isotope="D"
    )
    selected_id = complete_id if select_complete else incomplete_id
    untouched_id = incomplete_id if select_complete else complete_id
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), target_residue_ids=frozenset((selected_id,))
    )
    assert tuple(event.residue_id for event in result.repairs) == (selected_id,)
    assert result.repairs[0].kind is (
        RepairEventKind.RETAINED_NON_POLYMER_TOPOLOGY_RESOLVED
        if select_complete
        else RepairEventKind.HYDROGENS_ADDED
    )
    original_index = source.constitution.residue_index(untouched_id)
    final_index = result.structure.constitution.residue_index(untouched_id)
    assert source.constitution.residue_site_at(original_index) == (
        result.structure.constitution.residue_site_at(final_index)
    )
    assert source.residue_geometry(original_index) == (
        result.structure.residue_geometry(final_index)
    )


@pytest.mark.parametrize("complete_first", (False, True))
@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_mixed_batch_reports_charge_only_repair_after_neighbor_insertion(
    complete_first: bool, isotope: str
) -> None:
    source, complete_id, incomplete_id = _benzene_batch(
        complete_first=complete_first, isotope=isotope
    )
    source = resolve_retained_non_polymer_topology(
        source, ComponentLibrary({}), target_residue_ids=frozenset((complete_id,))
    ).structure
    evidence = tuple(
        RetainedNonPolymerChemistryOverride(
            residue_id, "c1ccccc1", tuple(f"C{i}" for i in range(1, 7))
        ).to_evidence()
        for residue_id in (complete_id, incomplete_id)
    )
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=evidence
    )
    assert not result.issues
    events = tuple(
        event
        for event in result.repairs
        if event.kind is RepairEventKind.RETAINED_NON_POLYMER_TOPOLOGY_RESOLVED
    )
    assert len(events) == 1
    assert events[0].scope.residue_ids == (complete_id,)
    assert events[0].atom_names == tuple(f"C{i}" for i in range(1, 7))
    residue = source.constitution.residue_site_at(
        source.constitution.residue_index(complete_id)
    )
    for atom in residue.atom_sites:
        ref = AtomRef(complete_id, atom.name)
        old = source.constitution.atom_index(ref)
        new = result.structure.constitution.atom_index(ref)
        assert result.structure.geometry.atom_geometry(
            new
        ) == source.geometry.atom_geometry(old)
        assert result.structure.constitution.atom_site_at(
            new
        ) == source.constitution.atom_site_at(old)
        assert result.structure.topology.formal_charge(new) == (
            None if atom.is_hydrogen() else 0
        )
    repeated = add_retained_non_polymer_hydrogens(
        result.structure, ComponentLibrary({}), chemistry_evidence=evidence
    )
    assert repeated.structure is result.structure
    assert not repeated.repairs and not repeated.issues
