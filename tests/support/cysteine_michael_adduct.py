"""Observed CYS/QS7 scaffold and custody controls for adduct tests."""

from pathlib import Path

from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload

CYSTEINE = ResidueId("A", 552)
LIGAND = ResidueId("A", 801)


def qs7_source(*, linked: bool = True) -> ProteinStructure:
    lines = (
        (Path(__file__).parents[1] / "fixtures/corpus/pdb6v9c.ent")
        .read_text()
        .splitlines()
    )
    atoms = [
        line
        for line in lines
        if line.startswith(("ATOM  ", "HETATM"))
        and ((line[17:20] == "CYS" and int(line[22:26]) == 552) or line[17:20] == "QS7")
    ]
    serials = {int(line[6:11]) for line in atoms}
    residues_by_serial = {int(line[6:11]): line[17:27] for line in atoms}
    records = []
    for line in lines:
        if linked and line.startswith("LINK") and "QS7" in line:
            records.append(line)
        if line.startswith("CONECT"):
            endpoints = [
                int(line[i : i + 5])
                for i in range(6, len(line), 5)
                if line[i : i + 5].strip()
            ]
            if endpoints and all(i in serials for i in endpoints):
                if not linked and len({residues_by_serial[i] for i in endpoints}) != 1:
                    continue
                records.append(line)
    return read_structure_string(
        "\n".join((*records, *atoms, "END")) + "\n", FileFormat.PDB
    )


def assert_original_preserved(
    source: ProteinStructure, result: ProteinStructure
) -> None:
    for old, _ in source.geometry.iter_positions():
        new = result.constitution.atom_index(source.constitution.atom_ref_at(old))
        assert result.constitution.atom_site_at(
            new
        ) == source.constitution.atom_site_at(old)
        assert result.geometry.atom_geometry(new) == source.geometry.atom_geometry(old)
        if source.topology.formal_charge(old) is not None:
            assert result.topology.formal_charge(new) == source.topology.formal_charge(
                old
            )
    for bond in source.topology.bonds:
        if bond.provenance is not BondProvenance.SOURCE_EXPLICIT:
            continue
        refs = [source.constitution.atom_ref_at(i) for i in bond.endpoint_pair()]
        new_bond = result.topology.bond_between(
            *(result.constitution.atom_index(ref) for ref in refs)
        )
        assert new_bond is not None and new_bond.source_metadata == bond.source_metadata
        assert new_bond.provenance is bond.provenance


def with_bonds(
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
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


def residue_payload(
    source: ProteinStructure, residue_id: ResidueId
) -> CompletionResiduePayload:
    index = source.constitution.residue_index(residue_id)
    return CompletionResiduePayload(
        source.constitution.residue_site_at(index),
        source.residue_geometry(index),
        source.residue_formal_charge_by_atom_name(index),
    )


def with_thiol_hydrogen(
    source: ProteinStructure,
    *,
    original: bool = False,
    element: str = "H",
    charge: int | None = None,
) -> ProteinStructure:
    payload = residue_payload(source, CYSTEINE)
    sulfur = payload.position("SG")
    payload = payload.with_atom_payload(
        AtomSite("HG", element),
        atom_geometry=AtomGeometry(
            Vec3(sulfur.x + 1.34, sulfur.y, sulfur.z), occupancy=0.75, b_factor=14.5
        ),
        formal_charge=charge,
    )
    updated = source.with_updated_residue_facets(
        payload.residue_site,
        residue_geometry=payload.residue_geometry,
        formal_charge_by_atom_name=payload.formal_charge_by_atom_name,
    )
    bond = TopologyBond(
        atom_index_1=updated.constitution.atom_index(AtomRef(CYSTEINE, "SG")),
        atom_index_2=updated.constitution.atom_index(AtomRef(CYSTEINE, "HG")),
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )
    updated = with_bonds(updated, (*updated.topology.bonds, bond))
    if original:
        return read_structure_string(
            write_structure_string(updated, FileFormat.PDB), FileFormat.PDB
        )
    return updated
