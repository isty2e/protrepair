"""Topology projections derived from component chemistry definitions."""

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.semantics import (
    IDEALIZED_BACKBONE_OR_TERMINAL_HYDROGEN_ANCHORS,
)
from protrepair.structure.constitution import ResidueSite, StructureConstitution
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    TopologyBond,
)


def template_resolved_topology_bonds(
    constitution: StructureConstitution,
    *,
    component_library: ComponentLibrary,
) -> tuple[TopologyBond, ...]:
    """Return template-resolved topology bonds for materialized residue atoms."""

    bonds: list[TopologyBond] = []
    for residue_index, residue_site in enumerate(constitution.residue_slots):
        bonds.extend(
            _template_resolved_topology_bonds_for_residue(
                constitution,
                residue_index=ResidueIndex(residue_index),
                residue_site=residue_site,
                component_library=component_library,
            )
        )

    return tuple(bonds)


def template_resolved_topology_bonds_for_new_atoms(
    *,
    source_constitution: StructureConstitution,
    target_constitution: StructureConstitution,
    component_library: ComponentLibrary,
) -> tuple[TopologyBond, ...]:
    """Return template-resolved bonds whose endpoints include newly added atoms."""

    bonds: list[TopologyBond] = []
    for residue_index, target_residue_site in enumerate(
        target_constitution.residue_slots
    ):
        source_residue_site = source_constitution.residue_or_ligand(
            target_residue_site.residue_id
        )
        source_atom_names = (
            frozenset()
            if source_residue_site is None
            else frozenset(source_residue_site.atom_site_names())
        )
        new_atom_names = frozenset(target_residue_site.atom_site_names()).difference(
            source_atom_names
        )
        if not new_atom_names:
            continue

        bonds.extend(
            _template_resolved_topology_bonds_for_residue(
                target_constitution,
                residue_index=ResidueIndex(residue_index),
                residue_site=target_residue_site,
                component_library=component_library,
                required_endpoint_atom_names=new_atom_names,
            )
        )

    return tuple(bonds)


def template_resolved_hydrogen_topology_bonds_for_new_atoms(
    *,
    source_constitution: StructureConstitution,
    target_constitution: StructureConstitution,
    component_library: ComponentLibrary,
) -> tuple[TopologyBond, ...]:
    """Return template-resolved H-anchor bonds for newly added hydrogens."""

    bonds: list[TopologyBond] = []
    for residue_index, target_residue_site in enumerate(
        target_constitution.residue_slots
    ):
        source_residue_site = source_constitution.residue_or_ligand(
            target_residue_site.residue_id
        )
        source_atom_names = (
            frozenset()
            if source_residue_site is None
            else frozenset(source_residue_site.atom_site_names())
        )
        new_hydrogen_atom_names = frozenset(
            atom_site.name
            for atom_site in target_residue_site.atom_sites
            if atom_site.element == "H" and atom_site.name not in source_atom_names
        )
        if not new_hydrogen_atom_names:
            continue

        bonds.extend(
            _template_resolved_hydrogen_topology_bonds_for_residue(
                target_constitution,
                residue_index=ResidueIndex(residue_index),
                residue_site=target_residue_site,
                component_library=component_library,
                hydrogen_atom_names=new_hydrogen_atom_names,
            )
        )

    return tuple(bonds)


def _template_resolved_topology_bonds_for_residue(
    constitution: StructureConstitution,
    *,
    residue_index: ResidueIndex,
    residue_site: ResidueSite,
    component_library: ComponentLibrary,
    required_endpoint_atom_names: frozenset[str] = frozenset(),
) -> tuple[TopologyBond, ...]:
    template = component_library.get(residue_site.component_id)
    if template is None:
        return ()

    present_atom_names = frozenset(residue_site.atom_site_names())
    return tuple(
        topology_bond
        for bond_definition in template.definition.bonds
        for topology_bond in (
            _topology_bond_from_residue_template_bond(
                constitution,
                residue_index=residue_index,
                bond_definition=bond_definition,
                present_atom_names=present_atom_names,
                required_endpoint_atom_names=required_endpoint_atom_names,
            ),
        )
        if topology_bond is not None
    )


def _template_resolved_hydrogen_topology_bonds_for_residue(
    constitution: StructureConstitution,
    *,
    residue_index: ResidueIndex,
    residue_site: ResidueSite,
    component_library: ComponentLibrary,
    hydrogen_atom_names: frozenset[str],
) -> tuple[TopologyBond, ...]:
    template = component_library.get(residue_site.component_id)
    if template is None:
        return ()

    present_atom_names = frozenset(residue_site.atom_site_names())
    return tuple(
        TopologyBond(
            atom_index_1=constitution.atom_index_in_residue(
                residue_index,
                anchor_atom_name,
            ),
            atom_index_2=constitution.atom_index_in_residue(
                residue_index,
                hydrogen_atom_name,
            ),
            relationship_type=BondRelationshipType.COVALENT,
            provenance=BondProvenance.TEMPLATE_RESOLVED,
        )
        for hydrogen_atom_name, anchor_atom_name in (
            template.template_hydrogen_anchor_by_name(hydrogen_atom_names).items()
        )
        if anchor_atom_name in present_atom_names
        and anchor_atom_name not in IDEALIZED_BACKBONE_OR_TERMINAL_HYDROGEN_ANCHORS
    )


def _topology_bond_from_residue_template_bond(
    constitution: StructureConstitution,
    *,
    residue_index: ResidueIndex,
    bond_definition: BondDefinition,
    present_atom_names: frozenset[str],
    required_endpoint_atom_names: frozenset[str],
) -> TopologyBond | None:
    if (
        bond_definition.atom_name_1 not in present_atom_names
        or bond_definition.atom_name_2 not in present_atom_names
    ):
        return None
    if required_endpoint_atom_names and (
        bond_definition.atom_name_1 not in required_endpoint_atom_names
        and bond_definition.atom_name_2 not in required_endpoint_atom_names
    ):
        return None

    return TopologyBond(
        atom_index_1=constitution.atom_index_in_residue(
            residue_index,
            bond_definition.atom_name_1,
        ),
        atom_index_2=constitution.atom_index_in_residue(
            residue_index,
            bond_definition.atom_name_2,
        ),
        order=bond_definition.order,
        aromatic=bond_definition.aromatic,
        relationship_type=BondRelationshipType.COVALENT,
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )
