"""Unit tests for retained non-polymer RDKit fallback chemistry."""

from collections import Counter
from math import cos, pi, sin

import pytest

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.inference import (
    retained_non_polymer_fallback as fallback_inference,
)
from protrepair.chemistry.inference.retained_non_polymer_fallback import (
    infer_retained_non_polymer_rdkit_fallback,
    retained_non_polymer_rdkit_fallback_expected_hydrogen_atom_names,
    retained_non_polymer_rdkit_fallback_heavy_bond_definitions,
    retained_non_polymer_rdkit_fallback_hydrogen_bond_definitions,
    retained_non_polymer_rdkit_fallback_hydrogenated_molecule,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.structure.constitution import AtomSite, ResidueSite
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import ResidueId

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - required dependency import guard
    Chem = None

RDKIT_AVAILABLE = Chem is not None


def test_fallback_cross_layer_helpers_raise_typed_rdkit_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-layer fallback helpers should not leak private assertions."""

    monkeypatch.setattr(fallback_inference, "Chem", None)
    residue_site, residue_geometry = _retained_ligand_pose(
        (
            ("C1", "C", Vec3(0.0, 0.0, 0.0)),
            ("O1", "O", Vec3(1.40, 0.0, 0.0)),
        )
    )

    with pytest.raises(RdkitUnavailableError):
        fallback_inference.retained_non_polymer_rdkit_fallback_heavy_bond_definitions(
            residue_site,
            residue_geometry,
        )
    with pytest.raises(RdkitUnavailableError):
        fallback_inference.retained_non_polymer_rdkit_fallback_supports_passive_context(
            residue_site,
            residue_geometry,
        )


def test_fallback_inference_result_owns_pose_bonds_names_and_bounds() -> None:
    """Canonical fallback result should carry all full-inference projections."""

    from rdkit import rdBase

    residue_site, residue_geometry = _retained_ligand_pose(
        (
            ("C1", "C", Vec3(0.0, 0.0, 0.0)),
            ("O1", "O", Vec3(1.40, 0.0, 0.0)),
        )
    )

    inference_result = infer_retained_non_polymer_rdkit_fallback(
        residue_site,
        residue_geometry,
    )

    assert inference_result.rdkit_backend_version == rdBase.rdkitVersion
    assert inference_result.hydrogen_atom_names == ("H001", "H002", "H003", "H004")
    assert inference_result.heavy_bond_definitions == (
        BondDefinition("O1", "C1"),
    )
    assert _hydrogen_anchor_counts(inference_result.hydrogenated_molecule) == Counter(
        {"C1": 3, "O1": 1}
    )
    hydrogen_bonds = _bond_definitions_by_atom_pair(
        inference_result.hydrogen_bond_definitions
    )
    assert set(hydrogen_bonds) == {
        frozenset(("C1", "H001")),
        frozenset(("C1", "H002")),
        frozenset(("C1", "H003")),
        frozenset(("O1", "H004")),
    }
    assert {bond.order for bond in hydrogen_bonds.values()} == {1}
    assert inference_result.hydrogen_name_projection_candidate_count == 4
    assert inference_result.hydrogen_name_projection_candidate_limit >= 4


def test_fallback_inference_bounds_hydrogen_name_projection_candidates() -> None:
    """Large source/generated H matching should fail before quadratic allocation."""

    carbon_count = 33
    residue_site, residue_geometry = _retained_ligand_pose(
        tuple(
            (f"C{index:02d}", "C", Vec3((index - 1) * 1.54, 0.0, 0.0))
            for index in range(1, carbon_count + 1)
        )
    )
    source_hydrogen_names = tuple(
        f"HS{index:03d}" for index in range(1, 2 * carbon_count + 3)
    )

    with pytest.raises(ValueError, match="resource limit"):
        infer_retained_non_polymer_rdkit_fallback(
            residue_site,
            residue_geometry,
            present_hydrogen_atom_names=source_hydrogen_names,
            hydrogen_position_by_name={
                hydrogen_name: Vec3(0.0, 0.0, 0.0)
                for hydrogen_name in source_hydrogen_names
            },
        )


def test_fallback_perceives_carbonyl_bond_order_before_hydrogenation() -> None:
    """Carbonyl geometry should not be hydrogenated as an all-single alcohol."""

    residue_site, residue_geometry = _retained_ligand_pose(
        (
            ("C1", "C", Vec3(0.0, 0.0, 0.0)),
            ("C2", "C", Vec3(1.50, 0.0, 0.0)),
            ("O1", "O", Vec3(2.71, 0.0, 0.0)),
        )
    )

    heavy_bonds = _bond_definitions_by_atom_pair(
        retained_non_polymer_rdkit_fallback_heavy_bond_definitions(
            residue_site,
            residue_geometry,
        )
    )
    hydrogenated_molecule = retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
        residue_site,
        residue_geometry,
    )

    assert heavy_bonds[frozenset(("C1", "C2"))] == BondDefinition("C2", "C1")
    assert heavy_bonds[frozenset(("C2", "O1"))] == BondDefinition(
        "O1",
        "C2",
        order=2,
    )
    assert _hydrogen_anchor_counts(hydrogenated_molecule) == Counter(
        {"C1": 3, "C2": 1}
    )
    assert retained_non_polymer_rdkit_fallback_expected_hydrogen_atom_names(
        residue_site,
        residue_geometry,
    ) == ("H001", "H002", "H003", "H004")


def test_fallback_perceives_aromatic_carbon_ring_before_hydrogenation() -> None:
    """Benzene-like geometry should not be hydrogenated as all-single cyclohexane."""

    residue_site, residue_geometry = _retained_ligand_pose(
        (
            ("C1", "C", Vec3(1.397, 0.000, 0.0)),
            ("C2", "C", Vec3(0.699, 1.210, 0.0)),
            ("C3", "C", Vec3(-0.699, 1.210, 0.0)),
            ("C4", "C", Vec3(-1.397, 0.000, 0.0)),
            ("C5", "C", Vec3(-0.699, -1.210, 0.0)),
            ("C6", "C", Vec3(0.699, -1.210, 0.0)),
        )
    )

    heavy_bonds = retained_non_polymer_rdkit_fallback_heavy_bond_definitions(
        residue_site,
        residue_geometry,
    )
    hydrogenated_molecule = retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
        residue_site,
        residue_geometry,
    )

    assert len(heavy_bonds) == 6
    assert {bond.aromatic for bond in heavy_bonds} == {True}
    assert _hydrogen_anchor_counts(hydrogenated_molecule) == Counter(
        {f"C{index}": 1 for index in range(1, 7)}
    )


def test_fallback_rejects_ambiguous_double_carbonyl_geometry() -> None:
    """Symmetric short C-O geometry should not silently choose one double bond."""

    residue_site, residue_geometry = _retained_ligand_pose(
        (
            ("C1", "C", Vec3(0.0, 0.0, 0.0)),
            ("N1", "N", Vec3(0.0, 1.45, 0.0)),
            ("O1", "O", Vec3(1.25, 0.0, 0.0)),
            ("O2", "O", Vec3(-1.25, 0.0, 0.0)),
        )
    )

    with pytest.raises(ValueError, match="ambiguous.*carbonyl"):
        retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
            residue_site,
            residue_geometry,
        )


def test_fallback_rejects_charged_short_carbonyl_geometry() -> None:
    """Known charge evidence should prevent silent neutral carbonyl promotion."""

    residue_site, residue_geometry = _retained_ligand_pose(
        (
            ("C1", "C", Vec3(0.0, 0.0, 0.0)),
            ("C2", "C", Vec3(1.50, 0.0, 0.0)),
            ("O1", "O", Vec3(2.71, 0.0, 0.0)),
        )
    )

    with pytest.raises(ValueError, match="ambiguous.*carbonyl"):
        retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
            residue_site,
            residue_geometry,
            formal_charge_by_atom_name={"O1": -1},
        )


def test_fallback_rejects_diatomic_carbonyl_like_geometry() -> None:
    """A short isolated C-O pair lacks enough evidence for safe hydrogenation."""

    residue_site, residue_geometry = _retained_ligand_pose(
        (
            ("C1", "C", Vec3(0.0, 0.0, 0.0)),
            ("O1", "O", Vec3(1.21, 0.0, 0.0)),
        )
    )

    with pytest.raises(ValueError, match="ambiguous.*carbonyl"):
        retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
            residue_site,
            residue_geometry,
        )


def test_fallback_keeps_saturated_carbon_ring_non_aromatic() -> None:
    """Long C-C ring bonds should not be promoted to aromatic chemistry."""

    residue_site, residue_geometry = _regular_ring(
        element_by_atom_name={f"C{index}": "C" for index in range(1, 7)},
        edge_length=1.54,
    )

    heavy_bonds = retained_non_polymer_rdkit_fallback_heavy_bond_definitions(
        residue_site,
        residue_geometry,
    )
    hydrogenated_molecule = retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
        residue_site,
        residue_geometry,
    )

    assert len(heavy_bonds) == 6
    assert {bond.aromatic for bond in heavy_bonds} == {False}
    assert sum(_hydrogen_anchor_counts(hydrogenated_molecule).values()) == 12


def test_fallback_rejects_hetero_aromatic_like_ring_geometry() -> None:
    """Aromatic-like hetero rings need stronger evidence than fallback geometry."""

    residue_site, residue_geometry = _regular_ring(
        element_by_atom_name={
            "N1": "N",
            "C2": "C",
            "C3": "C",
            "C4": "C",
            "C5": "C",
            "C6": "C",
        },
        edge_length=1.397,
    )

    with pytest.raises(ValueError, match="ambiguous.*aromatic"):
        retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
            residue_site,
            residue_geometry,
        )


def test_fallback_rejects_distorted_aromatic_like_carbon_ring_geometry() -> None:
    """Near-aromatic carbon rings outside the strict window should not be guessed."""

    residue_site, residue_geometry = _regular_ring(
        element_by_atom_name={f"C{index}": "C" for index in range(1, 7)},
        edge_length=1.46,
    )

    with pytest.raises(ValueError, match="ambiguous.*aromatic"):
        retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
            residue_site,
            residue_geometry,
        )


def test_fallback_rejects_five_member_aromatic_like_ring_geometry() -> None:
    """Five-member aromatic-like rings are not covered by the carbon-ring rule."""

    residue_site, residue_geometry = _regular_ring(
        element_by_atom_name={f"C{index}": "C" for index in range(1, 6)},
        edge_length=1.40,
    )

    with pytest.raises(ValueError, match="ambiguous.*aromatic"):
        retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
            residue_site,
            residue_geometry,
        )


@pytest.mark.parametrize(
    ("atoms", "message"),
    (
        (
            (
                ("N1", "N", Vec3(0.0, 0.0, 0.0)),
                ("O1", "O", Vec3(1.24, 0.0, 0.0)),
                ("O2", "O", Vec3(-1.24, 0.0, 0.0)),
            ),
            "nitro or nitrate",
        ),
        (
            (
                ("S1", "S", Vec3(0.0, 0.0, 0.0)),
                ("O1", "O", Vec3(1.45, 0.0, 0.0)),
                ("O2", "O", Vec3(-1.45, 0.0, 0.0)),
                ("C1", "C", Vec3(0.0, 1.80, 0.0)),
                ("C2", "C", Vec3(0.0, -1.80, 0.0)),
            ),
            "sulfoxide, sulfone, or sulfate",
        ),
        (
            (
                ("P1", "P", Vec3(0.0, 0.0, 0.0)),
                ("O1", "O", Vec3(1.52, 0.0, 0.0)),
                ("O2", "O", Vec3(-1.52, 0.0, 0.0)),
                ("O3", "O", Vec3(0.0, 1.52, 0.0)),
            ),
            "phosphate",
        ),
        (
            (
                ("C1", "C", Vec3(0.0, 0.0, 0.0)),
                ("N1", "N", Vec3(1.28, 0.0, 0.0)),
                ("C2", "C", Vec3(-1.50, 0.0, 0.0)),
            ),
            "imine-like",
        ),
    ),
)
def test_fallback_rejects_unsupported_hetero_multiple_bond_motifs(
    atoms: tuple[tuple[str, str, Vec3], ...],
    message: str,
) -> None:
    """Fallback should fail closed for motifs that need explicit chemistry."""

    residue_site, residue_geometry = _retained_ligand_pose(atoms)

    with pytest.raises(ValueError, match=message):
        retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
            residue_site,
            residue_geometry,
        )


def test_fallback_rejects_hydrogenation_stereo_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback should not claim success if hydrogenation changes heavy stereo."""

    residue_site, residue_geometry = _retained_ligand_pose(
        (
            ("C1", "C", Vec3(0.0, 0.0, 0.0)),
            ("C2", "C", Vec3(1.50, 0.0, 0.0)),
            ("O1", "O", Vec3(2.90, 0.0, 0.0)),
        )
    )

    def fake_stereo_signature(molecule) -> tuple[tuple[str, str], ...]:
        if molecule.GetNumAtoms() == 3:
            return (("C2", "R"),)
        return (("C2", "S"),)

    monkeypatch.setattr(
        fallback_inference,
        "_heavy_atom_stereo_signature",
        fake_stereo_signature,
    )

    with pytest.raises(ValueError, match="stereochemistry changed"):
        retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
            residue_site,
            residue_geometry,
        )


def _retained_ligand_pose(
    atoms: tuple[tuple[str, str, Vec3], ...],
) -> tuple[ResidueSite, ResidueGeometry]:
    """Return a retained ligand residue pose from atom descriptors."""

    return (
        ResidueSite(
            component_id="LIG",
            residue_id=ResidueId(chain_id="L", seq_num=1),
            atom_sites=tuple(
                AtomSite(name=atom_name, element=element)
                for atom_name, element, _position in atoms
            ),
            is_hetero=True,
        ),
        ResidueGeometry(
            atoms_by_name={
                atom_name: AtomGeometry(position=position)
                for atom_name, _element, position in atoms
            }
        ),
    )


def _regular_ring(
    *,
    element_by_atom_name: dict[str, str],
    edge_length: float,
) -> tuple[ResidueSite, ResidueGeometry]:
    """Return one planar regular ring with the requested edge length."""

    atom_names = tuple(element_by_atom_name)
    circumradius = edge_length / (2.0 * sin(pi / len(atom_names)))
    coordinates = tuple(
        Vec3(
            circumradius * cos(2.0 * pi * index / len(atom_names)),
            circumradius * sin(2.0 * pi * index / len(atom_names)),
            0.0,
        )
        for index in range(len(atom_names))
    )
    return _retained_ligand_pose(
        tuple(
            (atom_name, element_by_atom_name[atom_name], coordinates[index])
            for index, atom_name in enumerate(atom_names)
        )
    )


def _bond_definitions_by_atom_pair(
    bond_definitions: tuple[BondDefinition, ...],
) -> dict[frozenset[str], BondDefinition]:
    """Return bond definitions keyed by unordered atom-name pair."""

    return {
        frozenset((bond.atom_name_1, bond.atom_name_2)): bond
        for bond in bond_definitions
    }


def _hydrogen_anchor_counts(hydrogenated_molecule) -> Counter[str]:
    """Return generated fallback hydrogen counts by heavy-atom anchor name."""

    return Counter(
        (
            bond.atom_name_2
            if _is_generated_hydrogen_name(bond.atom_name_1)
            else bond.atom_name_1
        )
        for bond in retained_non_polymer_rdkit_fallback_hydrogen_bond_definitions(
            hydrogenated_molecule
        )
    )


def _is_generated_hydrogen_name(atom_name: str) -> bool:
    """Return whether a fallback atom name is one of our generated H names."""

    return len(atom_name) == 4 and atom_name.startswith("H") and atom_name[1:].isdigit()
