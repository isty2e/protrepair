"""Retained H identity needs compatible authority and complete correspondence."""

from dataclasses import replace
from itertools import product

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.chemistry import (
    ChemicalComponentDefinition,
    ComponentLibrary,
    IdealGeometryHydrogenSemantics,
    IdealizedComponent,
    IdealizedComponentAtom,
    ResidueTemplate,
)
from protrepair.chemistry.component.graph import BondDefinition
from protrepair.geometry import Vec3
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen import repair
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)

LIGAND_ID = ResidueId("L", 1)


def _hydroxyl_component() -> tuple[ProteinStructure, ComponentLibrary]:
    positions = (
        ("C1", "C", (1.0, 1.0, 0.0)),
        ("O1", "O", (0.0, 0.0, 0.0)),
        ("O2", "O", (2.0, 0.0, 0.0)),
        ("F1", "F", (1.0, 1.0, 1.4)),
        ("HO1", "H", (1.0, 0.0, 0.0)),
        ("HO2", "H", (2.0, 1.0, 0.0)),
        ("HC1", "H", (1.0, 1.0, -1.09)),
    )
    component = IdealizedComponent(
        "LIG",
        None,
        atoms=tuple(
            IdealizedComponentAtom(name, element, 0, ideal_position=position)
            for name, element, position in positions
        ),
        bonds=tuple(
            BondDefinition(first, second)
            for first, second in (
                ("C1", "O1"),
                ("C1", "O2"),
                ("C1", "F1"),
                ("O1", "HO1"),
                ("O2", "HO2"),
                ("C1", "HC1"),
            )
        ),
    )
    library = ComponentLibrary(
        {
            "LIG": ResidueTemplate(
                ChemicalComponentDefinition(
                    "LIG",
                    atom_names=component.heavy_atom_names(),
                    bonds=component.heavy_bonds(),
                ),
                hydrogen_semantics=IdealGeometryHydrogenSemantics(component),
            ),
        }
    )
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=LIGAND_ID,
                is_hetero=True,
                atoms=tuple(
                    atom_payload(name, element, Vec3(*position), formal_charge=0)
                    for name, element, position in positions
                    if element != "H"
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    return source, library


def _with_bonds(
    source: ProteinStructure, bonds: tuple[TopologyBond, ...]
) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=bonds,
        ),
        provenance=source.provenance,
    )


@pytest.mark.parametrize("provenance", tuple(BondProvenance))
@pytest.mark.parametrize("obsolete_parent", ("F1", "O2"))
@pytest.mark.parametrize("topology_only", (False, True))
@pytest.mark.parametrize("isotope", ("H", "D"))
def test_only_declared_stale_h_attachments_block_admitted_template(
    provenance: BondProvenance, obsolete_parent: str, topology_only: bool, isotope: str
) -> None:
    heavy, library = _hydroxyl_component()
    prepared = add_retained_non_polymer_hydrogens(heavy, library).structure
    index = prepared.constitution.residue_index(LIGAND_ID)
    residue = prepared.constitution.ligands[0]
    source = prepared.with_updated_residue_facets_batch(
        (
            (
                replace(
                    residue,
                    atom_sites=tuple(
                        replace(atom, element=isotope) if atom.name == "HO1" else atom
                        for atom in residue.atom_sites
                    ),
                ),
                prepared.residue_geometry(index).with_atom_geometry(
                    "HO1",
                    replace(
                        prepared.residue_geometry(index).atom_geometry("HO1"),
                        occupancy=0.75,
                        b_factor=12.5,
                    ),
                ),
                prepared.residue_formal_charge_by_atom_name(index),
            ),
        )
    )
    hydrogen_index = source.constitution.atom_index(AtomRef(LIGAND_ID, "HO1"))
    source = _with_bonds(
        source,
        tuple(
            bond for bond in source.topology.bonds if not bond.involves(hydrogen_index)
        )
        + (
            TopologyBond(
                hydrogen_index,
                source.constitution.atom_index(AtomRef(LIGAND_ID, obsolete_parent)),
                provenance=provenance,
            ),
        ),
    )
    transform = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = transform(source, library)
    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    if provenance is BondProvenance.SOURCE_EXPLICIT:
        assert result.structure is source
        assert not result.repairs
        assert result.issues
        return
    assert result.repairs
    parent = result.structure.constitution.atom_index(AtomRef(LIGAND_ID, "O1"))
    assert result.structure.topology.bond_between(hydrogen_index, parent) is not None
    assert (
        result.structure.topology.bond_between(
            hydrogen_index,
            result.structure.constitution.atom_index(
                AtomRef(LIGAND_ID, obsolete_parent)
            ),
        )
        is None
    )
    assert (
        not StructureChemistryObservation(result.structure, component_library=library)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
        .requires_topology_resolution()
    )
    assert transform(result.structure, library).repairs == ()


@pytest.mark.parametrize("topology_only", (False, True))
@pytest.mark.parametrize("impossible", (False, True))
def test_public_template_correspondence_does_not_lose_feasible_matching(
    topology_only: bool,
    impossible: bool,
) -> None:
    source, library = _hydroxyl_component()
    index = source.constitution.residue_index(LIGAND_ID)
    residue = source.constitution.ligands[0]
    geometry = source.residue_geometry(index)
    extras = (
        atom_payload("Q1", "D", Vec3(0, 1, 0) if impossible else Vec3(1, 0.05, 0)),
        atom_payload("Q2", "H", Vec3(0, -1, 0) if impossible else Vec3(0, 1, 0)),
        atom_payload("Q3", "H", Vec3(1, 1, -1.09)),
    )
    source = source.with_updated_residue_facets_batch(
        (
            (
                replace(
                    residue, atom_sites=residue.atom_sites + tuple(a[0] for a in extras)
                ),
                geometry.with_atom_geometries((a[0].name, a[1]) for a in extras),
                source.residue_formal_charge_by_atom_name(index)
                + tuple((atom[0].name, atom[2]) for atom in extras),
            ),
        )
    )
    source = _with_bonds(
        source,
        (
            TopologyBond(
                source.constitution.atom_index(AtomRef(LIGAND_ID, "Q3")),
                source.constitution.atom_index(AtomRef(LIGAND_ID, "C1")),
                provenance=BondProvenance.SOURCE_EXPLICIT,
            ),
        ),
    )
    transform = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = transform(source, library)
    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    if impossible:
        assert result.structure is source
        assert not result.repairs
        assert result.issues
        return
    assert result.repairs
    for hydrogen, parent in (("Q1", "O2"), ("Q2", "O1"), ("Q3", "C1")):
        assert (
            result.structure.topology.bond_between(
                result.structure.constitution.atom_index(AtomRef(LIGAND_ID, hydrogen)),
                result.structure.constitution.atom_index(AtomRef(LIGAND_ID, parent)),
            )
            is not None
        )


def test_matching_cardinality_agrees_with_exhaustive_small_graph_oracle() -> None:
    source_names = ("S1", "S2", "S3")
    generated_names = ("G1", "G2", "G3")
    pairs = tuple(product(source_names, generated_names))
    for mask in range(1 << len(pairs)):
        candidates = tuple(pair for bit, pair in enumerate(pairs) if mask & (1 << bit))
        edges = set(candidates)
        expected = max(
            sum(target is not None for target in targets)
            for targets in product((None, *generated_names), repeat=len(source_names))
            if len({target for target in targets if target is not None})
            == sum(target is not None for target in targets)
            and all(
                target is None or (source, target) in edges
                for source, target in zip(source_names, targets, strict=True)
            )
        )
        actual = repair._maximum_hydrogen_assignment(source_names, candidates)
        assert len(actual) == expected
        assert len(set(actual.values())) == len(actual)
        assert set(actual.items()).issubset(edges)
        assert repair._maximum_hydrogen_assignment(source_names, candidates) == actual


def test_matching_handles_long_alternating_path_without_recursion() -> None:
    size = 1500
    sources = tuple(f"S{i}" for i in range(size))
    candidates = tuple(
        pair
        for index, source in enumerate(sources[:-1])
        for pair in ((source, f"G{index}"), (source, f"G{index + 1}"))
    ) + ((sources[-1], "G0"),)
    result = repair._maximum_hydrogen_assignment(sources, candidates)
    assert len(result) == size
    assert len(set(result.values())) == size
    assert set(result.items()).issubset(set(candidates))


def test_parent_projection_filters_authority_before_detecting_ambiguity() -> None:
    heavy, library = _hydroxyl_component()
    prepared = add_retained_non_polymer_hydrogens(heavy, library).structure
    hydrogen = prepared.constitution.atom_index(AtomRef(LIGAND_ID, "HO1"))
    source = _with_bonds(
        prepared,
        prepared.topology.bonds
        + (
            TopologyBond(
                hydrogen,
                prepared.constitution.atom_index(AtomRef(LIGAND_ID, "F1")),
                provenance=BondProvenance.SOURCE_EXPLICIT,
            ),
        ),
    )
    residue_index = source.constitution.residue_index(LIGAND_ID)
    assert "HO1" not in source.topology.residue_hydrogen_anchor_by_name(
        constitution=source.constitution,
        residue_index=residue_index,
    )
    assert source.topology.residue_hydrogen_anchor_by_name(
        constitution=source.constitution,
        residue_index=residue_index,
        provenance=BondProvenance.SOURCE_EXPLICIT,
    ) == {"HO1": "F1"}
    assert (
        source.topology.residue_hydrogen_anchor_by_name(
            constitution=source.constitution,
            residue_index=residue_index,
            provenance=BondProvenance.TEMPLATE_RESOLVED,
        )["HO1"]
        == "O1"
    )
