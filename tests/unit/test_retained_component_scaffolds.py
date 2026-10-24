"""Retained component inventories and partial preparation are not polymer slots."""

from dataclasses import replace

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.chemistry.component.topology import (
    template_hydrogen_bond_definitions_for_names,
)
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)

LIGAND_ID = ResidueId("L", 1)


def component_structure(
    component_id: str,
    *,
    omitted: tuple[str, ...] = (),
    with_hydrogens: bool = False,
    charges: tuple[tuple[str, int], ...] = (),
    elements: tuple[tuple[str, str], ...] = (),
    extra: bool = False,
) -> ProteinStructure:
    record = build_bundled_retained_non_polymer_registry().get(component_id)
    assert record is not None
    template = build_default_component_library().require(component_id)
    expected_h = (
        set(template.expected_hydrogen_atom_names()) if with_hydrogens else set()
    )
    charge_map, element_map = dict(charges), dict(elements)
    atoms = []
    for atom in record.atoms:
        if atom.atom_name in omitted or (
            atom.is_hydrogen() and atom.atom_name not in expected_h
        ):
            continue
        assert atom.ideal_position is not None
        atoms.append(
            atom_payload(
                atom.atom_name,
                element_map.get(atom.atom_name, atom.element),
                Vec3.coerce(atom.ideal_position),
                formal_charge=charge_map.get(
                    atom.atom_name, atom.formal_charge if with_hydrogens else None
                ),
            )
        )
    if extra:
        atoms.append(atom_payload("EXTRA", "C", Vec3(30, 30, 30)))
    structure = build_structure(
        chains=(),
        source_format=FileFormat.PDB,
        ligands=(
            residue_payload(
                component_id=component_id,
                residue_id=LIGAND_ID,
                atoms=tuple(atoms),
                is_hetero=True,
            ),
        ),
    )
    if not with_hydrogens:
        return structure
    names = set(structure.constitution.ligands[0].atom_site_names())
    definitions = (
        template.definition.bonds
        + template_hydrogen_bond_definitions_for_names(
            template, hydrogen_atom_names=tuple(expected_h)
        )
    )
    bonds = tuple(
        TopologyBond(
            structure.constitution.atom_index(AtomRef(LIGAND_ID, b.atom_name_1)),
            structure.constitution.atom_index(AtomRef(LIGAND_ID, b.atom_name_2)),
            order=b.order,
            aromatic=b.aromatic,
            provenance=BondProvenance.TEMPLATE_RESOLVED,
        )
        for b in definitions
        if {b.atom_name_1, b.atom_name_2} <= names
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=bonds,
        ),
        provenance=structure.provenance,
    )


@pytest.mark.parametrize("component_id,count", (("SAH", 20), ("SAM", 22)))
def test_complete_retained_oxt_component_receives_dictionary_hydrogens(
    component_id: str, count: int
) -> None:
    source = component_structure(component_id)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    assert (
        sum(
            atom.is_hydrogen()
            for atom in result.structure.constitution.ligands[0].atom_sites
        )
        == count
    )
    for index, _ in source.geometry.iter_positions():
        target = result.structure.constitution.atom_index(
            source.constitution.atom_ref_at(index)
        )
        assert result.structure.geometry.atom_geometry(
            target
        ) == source.geometry.atom_geometry(index)


@pytest.mark.parametrize("component_id", ("SAH", "SAM"))
def test_complete_hydrogenated_retained_oxt_component_is_ready(
    component_id: str,
) -> None:
    source = component_structure(component_id, with_hydrogens=True)
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert (
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.PRESENT
    )
    assert (
        fact.hydrogen_topology_availability_state is TopologyAvailabilityState.PRESENT
    )
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert not result.issues
    assert result.structure.constitution is source.constitution
    assert result.structure.geometry is source.geometry


def test_sam_retains_its_component_oxt_charge_requirement() -> None:
    template = build_default_component_library().require("SAM")
    assert dict(template.heavy_atom_charge_requirements())["OXT"] == -1
    result = add_retained_non_polymer_hydrogens(component_structure("SAM"))
    index = result.structure.constitution.atom_index(AtomRef(LIGAND_ID, "OXT"))
    assert result.structure.topology.formal_charge(index) == -1


@pytest.mark.parametrize("topology_only", (False, True))
def test_sam_oxt_source_charge_conflict_is_atomic(topology_only: bool) -> None:
    source = component_structure("SAM", charges=(("OXT", 0),))
    execute = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = execute(source, build_default_component_library())
    assert result.structure is source and not result.repairs
    assert len(result.issues) == 1
    assert "OXT charge +0" in result.issues[0].message
    assert "source connectivity" not in result.issues[0].message


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("topology_only", (False, True))
def test_partial_template_preserves_source_hydrogen_and_charge_authority(
    isotope: str, topology_only: bool
) -> None:
    source = component_structure(
        "SAM",
        omitted=("CE", "HE1", "HE2", "HE3"),
        with_hydrogens=True,
        elements=(("HA", isotope),),
    )
    execute = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = execute(source, build_default_component_library())
    assert not result.issues
    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    for index, _ in source.geometry.iter_positions():
        target = result.structure.constitution.atom_index(
            source.constitution.atom_ref_at(index)
        )
        assert result.structure.topology.formal_charge(
            target
        ) == source.topology.formal_charge(index)
    fact = (
        StructureChemistryObservation(result.structure)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert (
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.ABSENT
    )


@pytest.mark.parametrize(
    "component_id", tuple(sorted(build_bundled_retained_non_polymer_registry().records))
)
def test_every_bundled_component_accepts_its_full_heavy_inventory(
    component_id: str,
) -> None:
    source = component_structure(component_id)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    fact = (
        StructureChemistryObservation(result.structure)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert (
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.PRESENT
    )


@pytest.mark.parametrize(
    "component_id,omitted,count",
    (
        ("NAD", ("C7N", "O7N", "N7N"), 25),
        ("NAD", ("N6A",), 25),
        ("NAP", ("C7N", "O7N", "N7N"), 26),
        ("SAM", ("CE",), 19),
        ("FMN", ("O3P",), 20),
    ),
)
def test_partial_components_place_h_on_available_parents(
    component_id: str, omitted: tuple[str, ...], count: int
) -> None:
    source = component_structure(component_id, omitted=omitted)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    assert (
        sum(
            atom.is_hydrogen()
            for atom in result.structure.constitution.ligands[0].atom_sites
        )
        == count
    )
    fact = (
        StructureChemistryObservation(result.structure)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert (
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.ABSENT
    )
    charged_name = "SD" if component_id == "SAM" else "N1N"
    if component_id != "FMN":
        index = result.structure.constitution.atom_index(
            AtomRef(LIGAND_ID, charged_name)
        )
        assert result.structure.topology.formal_charge(index) == (
            None if component_id == "SAM" else 1
        )


@pytest.mark.parametrize("reported", (None, 0, 1))
def test_incomplete_charge_neighborhood_never_overwrites_source_authority(
    reported: int | None,
) -> None:
    source = component_structure(
        "NAD",
        omitted=("C2N",),
        charges=() if reported is None else (("N1N", reported),),
    )
    result = add_retained_non_polymer_hydrogens(source)
    if reported == 0:
        assert result.structure is source and result.issues
    else:
        assert not result.issues
        index = result.structure.constitution.atom_index(AtomRef(LIGAND_ID, "N1N"))
        assert result.structure.topology.formal_charge(index) == reported
        assert any(
            a.is_hydrogen() for a in result.structure.constitution.ligands[0].atom_sites
        )


@pytest.mark.parametrize("extra", (False, True))
@pytest.mark.parametrize("topology_only", (False, True))
def test_undeclared_template_charges_do_not_bypass_scaffold_validation(
    extra: bool, topology_only: bool
) -> None:
    source = component_structure(
        "MAN", extra=extra, elements=() if extra else (("O1", "F"),)
    )
    stock = build_default_component_library().require("MAN")
    template = replace(stock, definition=replace(stock.definition, formal_charges={}))
    library = ComponentLibrary({"MAN": template})
    execute = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = execute(source, library)
    assert result.structure is source and not result.repairs
    assert result.issues and all(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        for issue in result.issues
    )
    fact = (
        StructureChemistryObservation(source, component_library=library)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert (
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.ABSENT
    )
