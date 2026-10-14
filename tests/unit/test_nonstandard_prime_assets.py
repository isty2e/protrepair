from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np
import pytest
from rdkit import Chem
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry.nonstandard.registry import (
    NonstandardComponentRecord,
    build_bundled_nonstandard_registry,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io.nonstandard_component_ingress import (
    read_nonstandard_component_registry,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import ResidueIndex
from protrepair.transformer.completion.heavy import repair_heavy_atoms
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.workflow.contracts import OrphanFragmentPolicy

SOURCE_DIRECTORY = Path(__file__).parents[1] / "fixtures/chemistry/nonstandard-primes"
COMPONENT_IDS = ("HRG", "60F", "IEL", "YPZ")


@pytest.mark.parametrize("component_id", COMPONENT_IDS)
def test_bundled_record_matches_frozen_ccd_ingress(component_id: str) -> None:
    source = read_nonstandard_component_registry(
        SOURCE_DIRECTORY / f"{component_id}.cif"
    ).get(component_id)
    assert source is not None

    assert _bundled_record(component_id) == source


@pytest.mark.parametrize("component_id", COMPONENT_IDS)
def test_bundle_geometry_and_graph_match_raw_ccd(component_id: str) -> None:
    block = gemmi.cif.read_file(
        str(SOURCE_DIRECTORY / f"{component_id}.cif")
    ).sole_block()
    record = _bundled_record(component_id)
    rows = block.find(
        [
            "_chem_comp_atom.atom_id",
            "_chem_comp_atom.type_symbol",
            "_chem_comp_atom.charge",
            "_chem_comp_atom.pdbx_stereo_config",
            "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
        ]
    )
    source_atoms = tuple(
        (
            gemmi.cif.as_string(name),
            gemmi.cif.as_string(element),
            int(charge),
            stereo if stereo in {"R", "S"} else None,
            (float(x), float(y), float(z)),
        )
        for name, element, charge, stereo, x, y, z in rows
    )
    assert (
        tuple(
            (a.atom_name, a.element, a.formal_charge, a.stereo_config, a.ideal_position)
            for a in record.atoms
        )
        == source_atoms
    )

    source_bonds = tuple(
        (
            gemmi.cif.as_string(left),
            gemmi.cif.as_string(right),
            {"SING": 1, "DOUB": 2, "TRIP": 3}[order],
            aromatic == "Y",
        )
        for left, right, order, aromatic in block.find(
            [
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2",
                "_chem_comp_bond.value_order",
                "_chem_comp_bond.pdbx_aromatic_flag",
            ]
        )
    )
    assert (
        tuple((b.atom_name_1, b.atom_name_2, b.order, b.aromatic) for b in record.bonds)
        == source_bonds
    )


@pytest.mark.parametrize("component_id", COMPONENT_IDS)
def test_restraints_match_independent_coordinate_math(component_id: str) -> None:
    record = _bundled_record(component_id)
    positions = {}
    for atom in record.atoms:
        assert atom.ideal_position is not None
        positions[atom.atom_name] = np.asarray(atom.ideal_position)
    restraints = record.to_restraint_template()
    component = record.to_idealized_component()
    heavy_names = set(component.heavy_atom_names())

    assert len(restraints.bond_targets) == len(component.heavy_bonds())
    for bond in component.heavy_bonds():
        target = restraints.bond_target(bond.atom_name_1, bond.atom_name_2)
        assert target is not None
        separation = positions[bond.atom_name_1] - positions[bond.atom_name_2]
        assert target.target_distance_angstrom == pytest.approx(
            np.linalg.norm(separation), abs=1e-12
        )

    expected_angle_count = sum(
        len(tuple(combinations(component.bonded_atom_names(name) & heavy_names, 2)))
        for name in heavy_names
    )
    assert len(restraints.angle_targets) == expected_angle_count
    for target in restraints.angle_targets:
        first = positions[target.atom_name_1] - positions[target.center_atom_name]
        second = positions[target.atom_name_2] - positions[target.center_atom_name]
        cosine = np.dot(first, second) / (
            np.linalg.norm(first) * np.linalg.norm(second)
        )
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        assert target.target_angle_degrees == pytest.approx(angle, abs=1e-10)
        assert target in restraints.angle_targets_for_center(target.center_atom_name)

    assert len(restraints.chirality_targets) == 1
    for target in restraints.chirality_targets:
        vectors = np.asarray(
            [
                positions[name] - positions[target.center_atom_name]
                for name in target.ordered_neighbor_atom_names
            ]
        )
        volume = np.linalg.det(vectors)
        assert abs(volume) > 1.0
        assert np.sign(volume) == target.expected_orientation_sign

    assert len(restraints.plane_targets) == (1 if component_id in {"60F", "YPZ"} else 0)
    for target in restraints.plane_targets:
        assert set(target.atom_names) == {f"C{i}'" for i in range(1, 7)}
        points = np.asarray([positions[name] for name in target.atom_names])
        centered = points - points.mean(axis=0)
        _, _, axes = np.linalg.svd(centered)
        # Frozen CCD ideals are slightly nonplanar; these are not acceptance cutoffs.
        expected_residual = {
            "60F": 0.002608283366860503,
            "YPZ": 0.00047837460229782636,
        }[component_id]
        assert np.max(np.abs(centered @ axes[-1])) == pytest.approx(
            expected_residual, abs=1e-10
        )


@pytest.mark.parametrize(
    ("component_id", "first", "second", "hydrogen"),
    (
        ("HRG", "CG'", "CG", "HG'1"),
        ("60F", "C2'", "C1'", "H7"),
        ("IEL", "CG'", "CG", "HG'"),
        ("YPZ", "C5'", "C4'", "H5'"),
    ),
)
def test_prime_names_resolve_bonds_and_hydrogen_parents(
    component_id: str, first: str, second: str, hydrogen: str
) -> None:
    record = _bundled_record(component_id)
    component = record.to_idealized_component()

    assert component.atom_with_ideal_position(first) is not None
    assert second in component.bonded_atom_names(first)
    assert component.hydrogen_anchor_atom_name(hydrogen) == first
    assert record.to_restraint_template().bond_target(first, second) is not None


@pytest.mark.parametrize(
    ("component_id", "name", "position", "prime_position"),
    (
        ("HRG", "CG", (-0.887, -0.473, -0.240), (0.307, 0.337, 0.272)),
        ("60F", "C1", (0.443, -0.355, -0.008), (-3.346, -0.466, -0.090)),
        ("60F", "C2", (-0.962, 0.045, 0.053), (-3.686, 0.864, 0.186)),
        ("60F", "C3", (-1.934, -0.868, -0.152), (-5.010, 1.237, 0.238)),
        ("IEL", "CG", (-0.904, 0.436, 0.219), (0.301, -0.357, -0.289)),
        ("IEL", "HG", (-0.870, 0.494, 1.306), (0.267, -0.415, -1.377)),
        ("YPZ", "H2", (2.075, -1.351, -1.337), (-5.185, 1.524, -0.117)),
    ),
)
def test_primed_and_unprimed_atoms_keep_distinct_positions(
    component_id: str,
    name: str,
    position: tuple[float, float, float],
    prime_position: tuple[float, float, float],
) -> None:
    component = _bundled_record(component_id).to_idealized_component()
    atom = component.atom(name)
    primed_atom = component.atom(f"{name}'")
    assert atom is not None
    assert primed_atom is not None

    assert atom.ideal_position == position
    assert primed_atom.ideal_position == prime_position
    assert atom.ideal_position != primed_atom.ideal_position


@pytest.mark.parametrize("component_id", COMPONENT_IDS)
def test_corrected_geometry_matches_ccd_stereo_in_rdkit(component_id: str) -> None:
    record = _bundled_record(component_id)
    molecule = Chem.RWMol()
    indexes: dict[str, int] = {}
    conformer = Chem.Conformer(len(record.atoms))
    for atom in record.atoms:
        rdkit_atom = Chem.Atom(atom.element)
        rdkit_atom.SetFormalCharge(atom.formal_charge)
        index = molecule.AddAtom(rdkit_atom)
        indexes[atom.atom_name] = index
        assert atom.ideal_position is not None
        conformer.SetAtomPosition(index, atom.ideal_position)

    for bond in record.bonds:
        first, second = indexes[bond.atom_name_1], indexes[bond.atom_name_2]
        molecule.AddBond(
            first,
            second,
            {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}[
                bond.order
            ],
        )
        if bond.aromatic:
            molecule.GetBondBetweenAtoms(first, second).SetIsAromatic(True)
            molecule.GetAtomWithIdx(first).SetIsAromatic(True)
            molecule.GetAtomWithIdx(second).SetIsAromatic(True)

    resolved = molecule.GetMol()
    resolved.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(resolved)
    Chem.AssignStereochemistryFrom3D(resolved)
    for atom in record.atoms:
        if atom.stereo_config is not None:
            actual = resolved.GetAtomWithIdx(indexes[atom.atom_name])
            assert actual.GetProp("_CIPCode") == atom.stereo_config


@pytest.mark.parametrize(
    ("component_id", "missing_atom"),
    (("HRG", "CG'"), ("60F", "O4'"), ("IEL", "CG'"), ("YPZ", "C1'")),
)
def test_component_gap_is_unsupported(component_id: str, missing_atom: str) -> None:
    structure = _source_structure(component_id, missing_atom_name=missing_atom)

    result = repair_heavy_atoms(
        structure,
        augment_c_terminal_oxt=False,
        orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
    )

    assert result.structure.constitution == structure.constitution
    assert result.structure.geometry == structure.geometry
    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_TEMPLATE_REPAIR
        for issue in result.issues
    )


@pytest.mark.parametrize("component_id", COMPONENT_IDS)
def test_component_hydrogenation_remains_partial(component_id: str) -> None:
    structure = _source_structure(component_id)

    result = add_hydrogens(structure)

    source_residue = structure.constitution.chains[0].residues[0]
    hydrogenated = result.structure.constitution.chains[0].residues[0]
    added_names = set(hydrogenated.atom_site_names()) - set(
        source_residue.atom_site_names()
    )
    assert added_names == {"H1", "H2", "H3"}

    before = structure.residue_geometry(ResidueIndex(0))
    after = result.structure.residue_geometry(ResidueIndex(0))
    for name in source_residue.atom_site_names():
        assert after.position(name) == before.position(name)

    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        for issue in result.issues
    )


def _bundled_record(component_id: str) -> NonstandardComponentRecord:
    record = build_bundled_nonstandard_registry().get(component_id)
    assert record is not None
    return record


def _source_structure(
    component_id: str, *, missing_atom_name: str | None = None
) -> ProteinStructure:
    record = read_nonstandard_component_registry(
        SOURCE_DIRECTORY / f"{component_id}.cif"
    ).get(component_id)
    assert record is not None
    atoms = []
    for atom in record.heavy_atoms():
        if atom.atom_name == missing_atom_name:
            continue
        assert atom.ideal_position is not None
        atoms.append(
            atom_payload(atom.atom_name, atom.element, Vec3(*atom.ideal_position))
        )
    residue = residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=tuple(atoms),
    )
    return build_bonded_structure(
        chains=(chain_payload("A", (residue,)),),
        source_format=FileFormat.PDB,
    )
