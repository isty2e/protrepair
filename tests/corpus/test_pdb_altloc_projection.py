"""Selected deposited atom sets remain visible in default native PDB reads."""

from pathlib import Path

import pytest
from rdkit import Chem

from protrepair.io import read_structure, write_structure_string
from protrepair.io.ingress_policy import LigandHandling, StructureNormalizationPolicy
from protrepair.io.pdb_projection import pdb_without_conect
from protrepair.structure.provenance import FileFormat

pytestmark = pytest.mark.corpus


@pytest.mark.parametrize("entry", ["1msi", "1rgg", "1hw5", "1bpi"])
def test_default_native_reader_keeps_the_selected_deposited_atom_serial_set(
    entry: str,
) -> None:
    source = read_structure(
        Path(f"tests/fixtures/corpus/pdb{entry}.ent"),
        policy=StructureNormalizationPolicy(ligand_handling=LigandHandling.KEEP),
    )
    geometry = source.geometry
    mmcif = write_structure_string(source, FileFormat.MMCIF)

    for text in (
        write_structure_string(source, FileFormat.PDB),
        pdb_without_conect(source),
    ):
        rows = tuple(
            row for row in text.splitlines() if row.startswith(("ATOM  ", "HETATM"))
        )
        assert len(rows) == source.geometry.atom_count()
        assert all(row[16] == " " for row in rows)
        molecule = Chem.MolFromPDBBlock(
            text, sanitize=False, removeHs=False, proximityBonding=True
        )
        assert molecule is not None
        assert molecule.GetNumAtoms() == len(rows)
        assert {
            atom.GetPDBResidueInfo().GetSerialNumber() for atom in molecule.GetAtoms()
        } == {int(row[6:11]) for row in rows}

    assert source.geometry is geometry
    assert write_structure_string(source, FileFormat.MMCIF) == mmcif
