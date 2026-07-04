"""Regression matrix for repaired topology truth and egress projection."""

from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.retained_non_polymer_components import (
    build_retained_non_polymer_component_library,
)

from protrepair.chemistry import build_default_component_library
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, write_structure_string
from protrepair.io.gemmi_writer import pdb_atom_serial_by_atom_ref
from protrepair.state import (
    TopologyAvailabilityState,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    TopologyBond,
)
from protrepair.transformer.completion.heavy import repair_heavy_atoms
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)


def test_repaired_cb_has_canonical_topology_and_egress_bonds() -> None:
    """Side-chain CB repair should update topology and both output formats."""

    residue_id = ResidueId("A", 1)
    result = repair_heavy_atoms(
        _single_polymer_residue_structure(
            component_id="ALA",
            residue_id=residue_id,
            atoms=(
                _atom("N", "N", 0.0, 0.0, 0.0),
                _atom("CA", "C", 1.5, 0.0, 0.0),
                _atom("C", "C", 2.8, 0.0, 0.0),
                _atom("O", "O", 3.8, 0.0, 0.0),
            ),
            source_name="repaired-topology-matrix-cb",
        )
    )

    _assert_repaired_bond_roundtrips(
        result.structure,
        residue_id,
        "CA",
        "CB",
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )


def test_repaired_sidechain_og_has_canonical_topology_and_egress_bonds() -> None:
    """Side-chain terminal heavy repair should update topology and egress."""

    residue_id = ResidueId("A", 1)
    result = repair_heavy_atoms(
        _single_polymer_residue_structure(
            component_id="SER",
            residue_id=residue_id,
            atoms=(
                _atom("N", "N", 0.0, 0.0, 0.0),
                _atom("CA", "C", 1.5, 0.0, 0.0),
                _atom("C", "C", 2.8, 0.0, 0.0),
                _atom("O", "O", 3.8, 0.0, 0.0),
                _atom("CB", "C", 1.5, 1.5, 0.0),
            ),
            source_name="repaired-topology-matrix-sidechain",
        )
    )

    _assert_repaired_bond_roundtrips(
        result.structure,
        residue_id,
        "CB",
        "OG",
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )


def test_repaired_terminal_oxt_has_canonical_topology_and_egress_bonds() -> None:
    """Terminal OXT augmentation should update topology and egress."""

    residue_id = ResidueId("A", 1)
    result = repair_heavy_atoms(
        _single_polymer_residue_structure(
            component_id="ALA",
            residue_id=residue_id,
            atoms=(
                _atom("N", "N", 0.0, 0.0, 0.0),
                _atom("CA", "C", 1.5, 0.0, 0.0),
                _atom("C", "C", 2.8, 0.0, 0.0),
                _atom("O", "O", 3.8, 0.0, 0.0),
                _atom("CB", "C", 1.5, 1.5, 0.0),
            ),
            source_name="repaired-topology-matrix-oxt",
        )
    )

    _assert_repaired_bond_roundtrips(
        result.structure,
        residue_id,
        "C",
        "OXT",
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )


def test_repaired_polymer_h_has_canonical_topology_and_egress_bonds() -> None:
    """Polymer hydrogen completion should update topology and egress."""

    residue_id = ResidueId("A", 1)
    result = add_hydrogens(
        _single_polymer_residue_structure(
            component_id="ALA",
            residue_id=residue_id,
            atoms=(
                _atom("N", "N", 0.0, 0.0, 0.0),
                _atom("CA", "C", 1.5, 0.0, 0.0),
                _atom("C", "C", 2.8, 0.0, 0.0),
                _atom("O", "O", 3.8, 0.0, 0.0),
                _atom("CB", "C", 1.5, 1.5, 0.0),
            ),
            source_name="repaired-topology-matrix-polymer-h",
        ),
        prepare_heavy_atoms=False,
    )

    _assert_repaired_bond_roundtrips(
        result.structure,
        residue_id,
        "CA",
        "HA",
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )


def test_repaired_ligand_hydrogen_has_canonical_topology_and_egress_bonds() -> None:
    """Retained non-polymer hydrogen completion should update topology and egress."""

    residue_id = ResidueId("L", 1)
    result = add_retained_non_polymer_hydrogens(
        build_structure(
            chains=(),
            ligands=(
                residue_payload(
                    component_id="LIG",
                    residue_id=residue_id,
                    atoms=(
                        _atom("C1", "C", 4.0, 0.0, 0.0),
                        _atom("O1", "O", 5.4, 0.0, 0.0),
                        _atom("N1", "N", 4.0, 1.2, 0.0),
                    ),
                    is_hetero=True,
                ),
            ),
            source_format=FileFormat.PDB,
            source_name="repaired-topology-matrix-retained-ligand-h",
        ),
        component_library=build_retained_non_polymer_component_library(),
    )

    _assert_repaired_bond_roundtrips(
        result.structure,
        residue_id,
        "C1",
        "H1",
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )


def test_coordinate_only_bonds_do_not_satisfy_readiness_or_egress() -> None:
    """Coordinates alone must not imply topology readiness or serialized bonds."""

    residue_id = ResidueId("A", 1)
    structure = _single_polymer_residue_structure(
        component_id="ALA",
        residue_id=residue_id,
        atoms=(
            _atom("N", "N", 0.0, 0.0, 0.0),
            _atom("CA", "C", 1.5, 0.0, 0.0),
            _atom("C", "C", 2.8, 0.0, 0.0),
            _atom("O", "O", 3.8, 0.0, 0.0),
            _atom("CB", "C", 1.5, 1.5, 0.0),
        ),
        source_name="repaired-topology-matrix-coordinate-only",
    )

    _, chemistry_readiness = (
        derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=build_default_component_library(),
        )
    )

    assert _topology_bond_between(structure, residue_id, "CA", "CB") is None
    assert (
        chemistry_readiness.residue_facts[0].heavy_atom_topology_availability_state
        is TopologyAvailabilityState.ABSENT
    )
    assert not _has_pdb_conect_between(structure, residue_id, "CA", "CB")
    assert "_struct_conn.id" not in write_structure_string(
        structure,
        FileFormat.MMCIF,
    )


def _single_polymer_residue_structure(
    *,
    component_id: str,
    residue_id: ResidueId,
    atoms: tuple[CanonicalAtomPayload, ...],
    source_name: str,
) -> ProteinStructure:
    return build_structure(
        chains=(
            chain_payload(
                residue_id.chain_id,
                (
                    residue_payload(
                        component_id=component_id,
                        residue_id=residue_id,
                        atoms=atoms,
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name=source_name,
    )


def _atom(
    name: str,
    element: str,
    x: float,
    y: float,
    z: float,
) -> CanonicalAtomPayload:
    return atom_payload(name, element, Vec3(x, y, z))


def _assert_repaired_bond_roundtrips(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
    *,
    provenance: BondProvenance,
) -> None:
    bond = _topology_bond_between(
        structure,
        residue_id,
        atom_name_1,
        atom_name_2,
    )

    assert bond is not None
    assert bond.relationship_type is BondRelationshipType.COVALENT
    assert bond.provenance is provenance
    assert _has_pdb_conect_between(
        structure,
        residue_id,
        atom_name_1,
        atom_name_2,
    )
    _assert_mmcif_struct_conn_between(
        structure,
        residue_id,
        atom_name_1,
        atom_name_2,
    )


def _topology_bond_between(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
) -> TopologyBond | None:
    return structure.topology.bond_between(
        structure.constitution.atom_index(AtomRef(residue_id, atom_name_1)),
        structure.constitution.atom_index(AtomRef(residue_id, atom_name_2)),
    )


def _has_pdb_conect_between(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
) -> bool:
    pdb_text = write_structure_string(structure, FileFormat.PDB)
    serial_by_atom_ref = pdb_atom_serial_by_atom_ref(pdb_text)
    serial_1 = serial_by_atom_ref[AtomRef(residue_id, atom_name_1)]
    serial_2 = serial_by_atom_ref[AtomRef(residue_id, atom_name_2)]

    return frozenset((serial_1, serial_2)) in _pdb_conect_serial_pairs(pdb_text)


def _pdb_conect_serial_pairs(pdb_text: str) -> frozenset[frozenset[int]]:
    pairs: set[frozenset[int]] = set()
    for line in pdb_text.splitlines():
        if not line.startswith("CONECT"):
            continue

        source_serial = int(line[6:11])
        neighbor_serials = tuple(
            int(raw_serial)
            for start in range(11, len(line), 5)
            for raw_serial in (line[start : start + 5].strip(),)
            if raw_serial
        )
        pairs.update(
            frozenset((source_serial, neighbor_serial))
            for neighbor_serial in neighbor_serials
        )

    return frozenset(pairs)


def _assert_mmcif_struct_conn_between(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
) -> None:
    atom_index_1 = structure.constitution.atom_index(
        AtomRef(residue_id, atom_name_1)
    )
    atom_index_2 = structure.constitution.atom_index(
        AtomRef(residue_id, atom_name_2)
    )
    connection_id = (
        f"protrepair_{min(atom_index_1.value, atom_index_2.value)}_"
        f"{max(atom_index_1.value, atom_index_2.value)}"
    )
    mmcif_text = write_structure_string(structure, FileFormat.MMCIF)

    assert "_struct_conn.id" in mmcif_text
    assert f"'{connection_id}' covale" in mmcif_text
    assert f" {atom_name_1} " in mmcif_text
    assert f" {atom_name_2} " in mmcif_text
