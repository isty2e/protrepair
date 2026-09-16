"""Corpus regression snapshots for coupled polymer hydrogen placement."""

from pathlib import Path

import gemmi
import pytest
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFHasAllMoleculeParams
from tests.support.structure_summary import summarize_structure

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.io import read_structure, read_structure_string
from protrepair.state import StructureProjectionStateFacts
from protrepair.structure import ResidueId
from protrepair.structure.labels import AtomRef
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import is_covalent_like_relationship
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.continuous.readiness import (
    ContinuousRelaxationReadinessPolicy,
)
from protrepair.workflow.contracts import StructureIngressOptions

pytestmark = pytest.mark.corpus


@pytest.mark.representative_regression
@pytest.mark.parametrize(
    ("input_path", "expected_digest"),
    (
        pytest.param(
            Path("tests/fixtures/corpus/pdb2a1d.ent"),
            "1b45960348bd6cb5ec395b01db31542b2fd25e1868d4e338ae03841d148737fd",
            id="pdb2a1d",
        ),
        pytest.param(
            Path("tests/fixtures/corpus/pdb2xbi.ent"),
            "0e13b060c9b57441ca758f32cdeaf4f5466e1caa05491206ef354c1345ed18d4",
            id="pdb2xbi",
        ),
    ),
)
def test_add_hydrogens_matches_coupled_chemistry_snapshot(
    input_path: Path,
    expected_digest: str,
) -> None:
    """Pin current identities and coordinates, not legacy uncoupled chemistry."""

    structure = read_structure(
        input_path,
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    result = add_hydrogens(structure)

    assert summarize_structure(result.structure).semantic_digest == expected_digest
    for offset, atom in enumerate(structure.constitution.atom_slots):
        if atom.is_hydrogen():
            continue
        index = AtomIndex(offset)
        ref = structure.constitution.atom_ref_at(index)
        output_index = result.structure.constitution.atom_index(ref)
        assert result.structure.geometry.atom_geometry(
            output_index
        ) == structure.geometry.atom_geometry(index)

    if input_path.name == "pdb2a1d.ent":
        _assert_2a1d_chemistry(result.structure)
        assert not result.issues


def _assert_2a1d_chemistry(structure):
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(structure), build_default_component_library()
    )
    assert all(
        target.is_realized()
        for chain in structure.constitution.chains
        for residue in chain.residues
        for target in preparation.targets_for(residue.residue_id)
    )
    assert sum(atom.is_hydrogen() for atom in structure.constitution.atom_slots) == 9460
    facts = StructureProjectionStateFacts.from_structure(structure)
    assert (
        ContinuousRelaxationReadinessPolicy().assess_structure_facts(facts).is_ready()
    )

    pairs = []
    for chain in ("A", "E"):
        ids = [ResidueId(chain, 1, code) for code in "MLKJIHGFEDCBA"] + [
            ResidueId(chain, 1)
        ]
        pairs.extend(zip(ids, ids[1:], strict=False))
    for chain in ("B", "F"):
        pairs.extend(
            (
                (ResidueId(chain, 183), ResidueId(chain, 184, "A")),
                (ResidueId(chain, 184, "A"), ResidueId(chain, 184)),
                (ResidueId(chain, 217), ResidueId(chain, 219)),
            )
        )
    assert len(pairs) == 32
    for first, second in pairs:
        carbon = structure.constitution.atom_index(AtomRef(first, "C"))
        nitrogen = structure.constitution.atom_index(AtomRef(second, "N"))
        bond = structure.topology.bond_between(carbon, nitrogen)
        assert bond is not None and bond.order == 1

    # Use the canonical graph, not PDB proximity inference or implicit H repair.
    native = Chem.RWMol()
    for offset, atom in enumerate(structure.constitution.atom_slots):
        entry = Chem.Atom(atom.element)
        entry.SetNoImplicit(True)
        entry.SetFormalCharge(structure.topology.formal_charge(AtomIndex(offset)) or 0)
        native.AddAtom(entry)
    for bond in structure.topology.bonds:
        if is_covalent_like_relationship(bond):
            native.AddBond(
                bond.atom_index_1.value,
                bond.atom_index_2.value,
                Chem.BondType.values[bond.order],
            )
    molecule = native.GetMol()
    Chem.SanitizeMol(molecule)
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in molecule.GetAtoms())
    assert UFFHasAllMoleculeParams(molecule)
    for _, second in pairs:
        nitrogen = structure.constitution.atom_index(AtomRef(second, "N"))
        assert (
            molecule.GetAtomWithIdx(nitrogen.value).GetHybridization()
            == Chem.HybridizationType.SP2
        )


def test_2a1d_pdb_sequence_and_mmcif_positions_resolve_the_same_graph():
    path = Path("tests/fixtures/corpus/pdb2a1d.ent")
    policy = StructureIngressOptions().structure_normalization_policy()
    pdb = read_structure(path, policy=policy)
    raw = gemmi.read_pdb(str(path))
    raw.assign_label_seq_id()
    mmcif = read_structure_string(
        raw.make_mmcif_document().as_string(), FileFormat.MMCIF, policy=policy
    )
    assert pdb.constitution == mmcif.constitution
    assert pdb.geometry == mmcif.geometry
    assert {
        (bond.endpoint_pair(), bond.order, bond.relationship_type)
        for bond in pdb.topology.bonds
    } == {
        (bond.endpoint_pair(), bond.order, bond.relationship_type)
        for bond in mmcif.topology.bonds
    }


def test_add_hydrogens_omits_hg_from_cross_chain_source_disulfides() -> None:
    """Cross-chain SSBOND cysteines should remain oxidized after completion."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb2a1d.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    repaired = add_hydrogens(structure).structure

    for residue_id in (
        ("A", 1),
        ("B", 122),
        ("E", 1),
        ("F", 122),
    ):
        residue = repaired.constitution.residue_or_ligand(
            ResidueId(residue_id[0], residue_id[1])
        )
        assert residue is not None
        assert not residue.has_atom_site("HG")
