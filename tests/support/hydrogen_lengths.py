"""Synthetic monomer and structure inputs shared by X-H length tests."""

from protrepair.analysis import HydrogenLengthReference
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string
from protrepair.structure import (
    AtomRef,
    AtomTopology,
    FileFormat,
    ProteinStructure,
    ResidueId,
    StructureIngress,
    StructureProvenance,
    StructureTopology,
    TopologyBond,
)
from protrepair.structure.slots import AtomIndex

REFERENCE_CIF = """\
data_comp_CYS
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
N N 1
CA C 0
C C 0
O O 0
OXT O -1
CB C 0
SG S 0
H H 0
H2 H 0
H3 H 0
HA H 0
HB2 H 0
HB3 H 0
HG H 0
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.type
_chem_comp_bond.aromatic
_chem_comp_bond.value_dist
_chem_comp_bond.value_dist_esd
_chem_comp_bond.value_dist_nucleus
_chem_comp_bond.value_dist_nucleus_esd
N CA SINGLE n ? ? ? ?
CA C SINGLE n ? ? ? ?
C O DOUBLE n ? ? ? ?
C OXT SINGLE n ? ? ? ?
CA CB SINGLE n ? ? ? ?
CB SG SINGLE n ? ? ? ?
N H SINGLE n 0.9 0.02 1.02 0.01
N H2 SINGLE n ? ? ? ?
N H3 SINGLE n ? ? ? ?
CA HA SINGLE n ? ? ? ?
CB HB2 SINGLE n ? ? ? ?
CB HB3 SINGLE n ? ? ? ?
SG HG SINGLE n 1.2 0.02 1.34 0.01
"""
THIOL = AtomRef(ResidueId("A", 1), "SG")
HG = AtomRef(THIOL.residue_id, "HG")


def structure_with_thiol(
    hydrogen_x: float,
    name: str = "HG",
    element: str = "H",
    *,
    extra_sites: tuple[tuple[str, str, float, float, float], ...] = (),
    suffix: str = "",
) -> ProteinStructure:
    sites = (
        ("N", "N", -3.0, 1.0, 0.0),
        ("CA", "C", -3.0, 0.0, 0.0),
        ("C", "C", -3.0, -1.0, 0.0),
        ("O", "O", -3.0, -2.0, 0.0),
        ("CB", "C", -1.8, 0.0, 0.0),
        ("SG", "S", 0.0, 0.0, 0.0),
        (name, element, hydrogen_x, 0.0, 0.0),
        *extra_sites,
    )
    text = "".join(
        f"ATOM  {i:5d} {atom:^4s} CYS A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {el:>2s}\n"
        for i, (atom, el, x, y, z) in enumerate(sites, 1)
    )
    return read_structure_string(text + suffix + "END\n", FileFormat.PDB)


def with_topology(
    structure: ProteinStructure,
    *,
    bonds: tuple[TopologyBond, ...] = (),
    sulfur_charge: int | None = None,
    keep_source: bool = True,
) -> ProteinStructure:
    constitution = structure.constitution
    states = list(structure.topology.atom_topologies)
    if sulfur_charge is not None:
        states[constitution.atom_index(THIOL).value] = AtomTopology(sulfur_charge)
    replaced_pairs = {bond.endpoint_pair() for bond in bonds}
    retained_bonds = tuple(
        bond
        for bond in structure.topology.bonds
        if bond.endpoint_pair() not in replaced_pairs
    )
    return ProteinStructure.from_payload(
        constitution=constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=constitution,
            atom_topologies=tuple(states),
            bonds=(*retained_bonds, *bonds),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance
        if keep_source
        else StructureProvenance(
            ingress=StructureIngress(source_format=FileFormat.PDB)
        ),
    )


def with_position(
    structure: ProteinStructure, index: AtomIndex, position: Vec3
) -> ProteinStructure:
    geometry = structure.geometry.with_atom_geometry(
        index, structure.geometry.atom_geometry(index).with_position(position)
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=geometry,
        topology=structure.topology,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def reference(text: str = REFERENCE_CIF) -> HydrogenLengthReference:
    return HydrogenLengthReference(
        text, component_id="CYS", source="synthetic test restraints"
    )
