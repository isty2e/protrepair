"""Deposited contacts and generated hydroxyl directions share chemical roles."""

from math import acos, degrees
from pathlib import Path

import numpy as np
import pytest

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.polar_contacts import PolarContactChemistry
from protrepair.diagnostics.clashes import detect_clashes
from protrepair.io import read_structure
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions

pytestmark = pytest.mark.corpus


@pytest.mark.parametrize("case", ["1rgg", "3enj", "1l2y"])
def test_deposited_ile_methylene_h_is_not_a_clash_with_its_parent(case: str) -> None:
    source = read_structure(Path(f"tests/fixtures/corpus/pdb{case}.ent"))
    library = build_default_component_library()
    chemistry = PolarContactChemistry(source.constitution, source.topology, library)
    clashes = detect_clashes(source, component_library=library).clashes
    sites = [
        residue
        for residue in source.constitution.residue_slots
        if residue.component_id == "ILE" and residue.has_atom_site("HG13")
    ]
    assert sites
    for residue in sites:
        assert chemistry.hydrogen_anchors(residue.residue_id)["HG13"] == "CG1"
        assert not any(
            clash.left_residue_id == residue.residue_id
            and clash.right_residue_id == residue.residue_id
            and {clash.left_atom_name, clash.right_atom_name} == {"CG1", "HG13"}
            for clash in clashes
        )


@pytest.mark.parametrize(
    "case,chain,acceptor_slot,acceptor_name,donor_slot,h_name",
    [
        ("1rgg", "A", 33, "OD1", 56, "HG1"),
        ("1rgg", "B", 54, "OE1", 86, "HH"),
        ("1rgg", "A", 78, "OE2", 80, "HH"),
        ("1ah0", "A", 315, "OXT", 163, "H"),
    ],
)
def test_raw_deposited_carboxyl_hbond_is_not_a_clash(
    case: str,
    chain: str,
    acceptor_slot: int,
    acceptor_name: str,
    donor_slot: int,
    h_name: str,
) -> None:
    source = read_structure(Path(f"tests/fixtures/corpus/pdb{case}.ent"))
    topology = source.topology
    geometry = source.geometry
    endpoints = {
        AtomRef(ResidueId(chain, acceptor_slot), acceptor_name),
        AtomRef(ResidueId(chain, donor_slot), h_name),
    }
    clashes = detect_clashes(
        source, component_library=build_default_component_library()
    ).clashes

    assert not any(
        {
            AtomRef(clash.left_residue_id, clash.left_atom_name),
            AtomRef(clash.right_residue_id, clash.right_atom_name),
        }
        == endpoints
        for clash in clashes
    )
    assert source.topology is topology
    assert source.geometry is geometry


@pytest.fixture(scope="module")
def hydrogenated_1cjc() -> tuple[ProteinStructure, ProteinStructure]:
    source = read_structure(
        Path("tests/fixtures/corpus/pdb1cjc.ent"),
        policy=StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )
    result = add_hydrogens(source)
    assert not result.has_errors()
    return source, result.structure


@pytest.mark.parametrize(
    "donor_slot,donor_name,h_name,acceptor_slot,acceptor_name",
    [
        (95, "OH", "HH", 35, "OD2"),
        (147, "OG1", "HG1", 146, "OD1"),
        (363, "OH", "HH", 393, "OD2"),
    ],
)
def test_hydroxyl_placement_retains_nearby_carboxyl_hbond(
    hydrogenated_1cjc: tuple[ProteinStructure, ProteinStructure],
    donor_slot: int,
    donor_name: str,
    h_name: str,
    acceptor_slot: int,
    acceptor_name: str,
) -> None:
    source, prepared = hydrogenated_1cjc
    vectors = []
    for slot, name in [
        (donor_slot, donor_name),
        (donor_slot, h_name),
        (acceptor_slot, acceptor_name),
    ]:
        index = prepared.constitution.atom_index(AtomRef(ResidueId("A", slot), name))
        position = prepared.geometry.position(index)
        vectors.append(np.array([position.x, position.y, position.z]))
    donor, hydrogen, acceptor = vectors
    dh, ah = donor - hydrogen, acceptor - hydrogen
    distance = float(np.linalg.norm(ah))
    angle = degrees(
        acos(
            float(
                np.clip(
                    np.dot(dh, ah) / (np.linalg.norm(dh) * np.linalg.norm(ah)), -1, 1
                )
            )
        )
    )

    # A broad positive-control window separates these source-supported contacts
    # from the rotated regression, rather than imposing a product acceptance gate.
    assert distance < 2.0
    assert angle > 130
    for offset, site in enumerate(source.constitution.atom_slots):
        if site.is_hydrogen():
            continue
        before = AtomIndex(offset)
        after = prepared.constitution.atom_index(
            source.constitution.atom_ref_at(before)
        )
        assert source.geometry.atom_geometry(before) == prepared.geometry.atom_geometry(
            after
        )
