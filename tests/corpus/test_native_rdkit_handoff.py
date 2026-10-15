"""Public preparation-to-native handoff on deposited protein fixtures."""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from tests.support.request_builders import whole_structure_requested_goals

from protrepair import process_structure
from protrepair.errors import RdkitProjectionError
from protrepair.io import to_rdkit_molecule
from protrepair.state import HydrogenCoverageState

pytestmark = pytest.mark.corpus


@pytest.mark.parametrize(
    "fixture",
    (
        "1aho.pdb",
        "span-reconstruction/1crn.pdb",
    ),
)
def test_public_preparation_exports_selected_protein_chemistry(fixture: str) -> None:
    result = process_structure(
        Path("tests/fixtures/pdb") / fixture,
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert not result.has_errors()
    source = result.structure
    constitution = source.constitution
    geometry = source.geometry
    topology = source.topology

    molecule = to_rdkit_molecule(source)

    assert Chem.SanitizeMol(molecule) == Chem.SanitizeFlags.SANITIZE_NONE
    assert molecule.GetNumAtoms() == source.geometry.atom_count()
    assert not any(atom.GetNumImplicitHs() for atom in molecule.GetAtoms())
    assert not any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    np.testing.assert_array_equal(
        molecule.GetConformer().GetPositions(),
        np.array([tuple(atom.position) for atom in geometry.atom_geometries]),
    )
    assert source.constitution is constitution
    assert source.geometry is geometry
    assert source.topology is topology


def test_native_export_refuses_unresolved_internal_backbone_boundaries() -> None:
    result = process_structure(
        Path("tests/fixtures/pdb/span-reconstruction/1ubq.pdb"),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert not result.has_errors()
    assert result.has_warnings()
    structure = result.structure
    geometry = structure.geometry
    topology = structure.topology

    with pytest.raises(RdkitProjectionError, match="unselected radical chemistry"):
        to_rdkit_molecule(structure)

    assert structure.geometry is geometry
    assert structure.topology is topology
