"""Coordinate completion preserves atom-specific charges without copying them."""

import pytest
from rdkit import Chem
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    completion_payload,
    residue_payload,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.hydrogen.repair import add_hydrogens
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.completion.shared.patch import (
    AtomPlacement,
    OrderedAtomPatch,
)
from protrepair.transformer.continuous.rdkit import build_rdkit_molecule
from protrepair.transformer.local import LocalScopeSpec


def _alanine(*, charge: int | None, missing_cb: bool = False) -> ProteinStructure:
    atoms = (
        atom_payload("N", "N", Vec3(0, 0, 0), formal_charge=charge),
        atom_payload("CA", "C", Vec3(1.458, 0, 0)),
        atom_payload("C", "C", Vec3(2.0, 1.4, 0)),
        atom_payload("O", "O", Vec3(1.5, 2.4, 0)),
    )
    if not missing_cb:
        atoms += (atom_payload("CB", "C", Vec3(2.0, -0.7, 1.2)),)
    residue = residue_payload(
        component_id="ALA", residue_id=ResidueId("A", 1), atoms=atoms
    )
    return build_structure(
        chains=(chain_payload("A", (residue,)),), source_format=FileFormat.PDB
    )


@pytest.mark.parametrize("charge", (None, 0, -1, 1))
@pytest.mark.parametrize("element", ("H", "D", "T", "C"))
def test_patch_does_not_copy_charge_from_an_unrelated_atom(
    charge: int | None, element: str
) -> None:
    residue = completion_payload(
        component_id="ALA",
        residue_id=ResidueId("A", 1),
        atoms=(
            atom_payload("N", "N", Vec3(0, 0, 0), formal_charge=charge),
            atom_payload("CA", "C", Vec3(1.458, 0, 0), formal_charge=0),
        ),
    )
    patch = OrderedAtomPatch(
        placements=(
            AtomPlacement("N", Vec3(0.1, 0.2, 0.3), "N"),
            AtomPlacement("NEW", Vec3(0, 1, 0), element),
        )
    )
    updated = residue.apply_patch(patch)
    assert updated.formal_charge("N") == charge
    assert updated.formal_charge("CA") == 0
    assert updated.formal_charge("NEW") is None
    assert updated.position("N") == Vec3(0.1, 0.2, 0.3)
    assert updated.atom_site("NEW").element == element
    assert residue.position("N") == Vec3(0, 0, 0)


@pytest.mark.parametrize("assigned_charge", (0, -1, 1))
def test_patch_preserves_charge_explicitly_assigned_to_the_new_atom(
    assigned_charge: int,
) -> None:
    source = _alanine(charge=1)
    site = source.constitution.residue_site_at(ResidueIndex(0))
    geometry = source.residue_geometry(ResidueIndex(0))
    patch = OrderedAtomPatch((AtomPlacement("NEW", Vec3(0, 1, 0), "H"),))
    _, _, charges = patch.materialize_on_payload(
        site,
        residue_geometry=geometry,
        formal_charge_by_atom_name=(("N", 1), ("NEW", assigned_charge)),
    )
    assert dict(charges) == {"N": 1, "NEW": assigned_charge}


def test_snapshot_patch_preserves_only_existing_charge_assignments() -> None:
    source = _alanine(charge=1, missing_cb=True)
    patch = OrderedAtomPatch((AtomPlacement("CB", Vec3(2.0, -0.7, 1.2), "C"),))
    snapshot = patch.apply_to_snapshot(
        ProteinStructureSnapshot.from_structure(source), ResidueIndex(0)
    )
    assert dict(
        snapshot.structure.residue_formal_charge_by_atom_name(ResidueIndex(0))
    ) == {"N": 1}


@pytest.mark.parametrize("targeted", (False, True))
@pytest.mark.parametrize("charge", (None, 0, -1, 1))
def test_heavy_and_hydrogen_completion_do_not_spread_source_charge(
    targeted: bool, charge: int | None
) -> None:
    source = _alanine(charge=charge, missing_cb=True)
    targets = frozenset((ResidueId("A", 1),)) if targeted else None
    repaired = repair_heavy_atoms_core(source, target_residue_ids=targets).structure
    hydrogenated = materialize_hydrogens_core(
        repaired, target_residue_ids=targets
    ).structure
    assert (
        repaired.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "CB"))
        is not None
    )
    assert any(atom.is_hydrogen() for atom in hydrogenated.constitution.atom_slots)
    for result in (repaired, hydrogenated):
        assert dict(result.residue_formal_charge_by_atom_name(ResidueIndex(0))) == (
            {} if charge is None else {"N": charge}
        )


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("prepare_heavy_atoms", (False, True))
def test_source_charged_n_terminus_keeps_neutral_generated_hydrogens(
    file_format: FileFormat, prepare_heavy_atoms: bool
) -> None:
    source = read_structure_string(
        write_structure_string(_alanine(charge=1), file_format), file_format
    )
    result = add_hydrogens(source, prepare_heavy_atoms=prepare_heavy_atoms).structure
    problem = build_continuous_relaxation_problem(
        result,
        LocalScopeSpec.from_residues((ResidueId("A", 1),)),
        component_library=build_standard_component_library(),
    )
    molecule, _ = build_rdkit_molecule(problem)
    Chem.SanitizeMol(molecule)
    hydrogens = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 1]
    assert len(hydrogens) == 7
    assert all(atom.GetFormalCharge() == 0 for atom in hydrogens)
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in hydrogens)
    nitrogen = next(atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 7)
    assert nitrogen.GetFormalCharge() == 1
    assert nitrogen.GetTotalValence() == 4

    restored = read_structure_string(
        write_structure_string(result, file_format), file_format
    )
    assert dict(restored.residue_formal_charge_by_atom_name(ResidueIndex(0))) == {
        "N": 1
    }


@pytest.mark.parametrize("use_override", (False, True))
@pytest.mark.parametrize(
    ("element", "charge", "smiles", "expected_hydrogens"),
    (("O", -1, "[O-]C", 3), ("N", 1, "[NH3+]C", 6)),
)
def test_retained_ligand_hydrogenation_does_not_copy_the_first_atom_charge(
    use_override: bool,
    element: str,
    charge: int,
    smiles: str,
    expected_hydrogens: int,
) -> None:
    residue_id = ResidueId("L", 1)
    charged_atom_name = f"{element}1"
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=residue_id,
                is_hetero=True,
                atoms=(
                    atom_payload(
                        charged_atom_name,
                        element,
                        Vec3(0, 0, 0),
                        formal_charge=charge,
                    ),
                    atom_payload("C1", "C", Vec3(1.4, 0, 0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    evidence = (
        (
            RetainedNonPolymerChemistryOverride(
                residue_id=residue_id,
                smiles=smiles,
                heavy_atom_names=(charged_atom_name, "C1"),
            ).to_evidence(),
        )
        if use_override
        else ()
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        component_library=ComponentLibrary(),
        chemistry_evidence=evidence,
    ).structure

    assert sum(atom.is_hydrogen() for atom in result.constitution.atom_slots) == (
        expected_hydrogens
    )
    assert dict(result.residue_formal_charge_by_atom_name(ResidueIndex(0))) == {
        charged_atom_name: charge
    }
    for atom_name in (charged_atom_name, "C1"):
        assert result.residue_geometry(ResidueIndex(0)).position(atom_name) == (
            source.residue_geometry(ResidueIndex(0)).position(atom_name)
        )
