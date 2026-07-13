"""Contract tests for retained non-polymer RDKit hydrogen patch projection."""

import pytest
from rdkit import Chem
from tests.support.canonical_builders import atom_payload, completion_payload

from protrepair.geometry import Vec3
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.retained_non_polymer_hydrogen import (
    rdkit_evidence,
    rdkit_patch,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload


def test_rdkit_hydrogen_append_patch_preserves_hydrogen_order_and_names() -> None:
    """RDKit H coordinates should pair with caller-owned names in atom order."""

    payload = _payload()
    molecule = _molecule_with_coordinates(
        (
            (1, Vec3(1.0, 2.0, 3.0)),
            (6, Vec3(8.0, 8.0, 8.0)),
            (1, Vec3(4.0, 5.0, 6.0)),
        )
    )

    patch = rdkit_patch.rdkit_hydrogen_append_patch(
        payload,
        hydrogenated_molecule=molecule,
        hydrogen_atom_names=("HA", "HB"),
    )

    assert patch.atom_names == ("C1", "HA", "HB")
    assert patch.position("C1") == Vec3(0.0, 0.0, 0.0)
    assert patch.position("HA") == Vec3(1.0, 2.0, 3.0)
    assert patch.position("HB") == Vec3(4.0, 5.0, 6.0)


@pytest.mark.parametrize(
    "hydrogen_atom_names",
    (("H1",), ("H1", "H2", "H3")),
    ids=("too-few", "too-many"),
)
def test_rdkit_hydrogen_append_patch_rejects_name_count_mismatch(
    hydrogen_atom_names: tuple[str, ...],
) -> None:
    """Every RDKit H coordinate must have exactly one final atom name."""

    molecule = _molecule_with_coordinates(
        (
            (6, Vec3(0.0, 0.0, 0.0)),
            (1, Vec3(1.0, 0.0, 0.0)),
            (1, Vec3(-1.0, 0.0, 0.0)),
        )
    )

    with pytest.raises(ValueError, match="name count must match"):
        rdkit_patch.rdkit_hydrogen_append_patch(
            _payload(),
            hydrogenated_molecule=molecule,
            hydrogen_atom_names=hydrogen_atom_names,
        )


def test_rdkit_hydrogen_append_patch_is_noop_without_hydrogens() -> None:
    """A molecule with no H atoms should reproduce the current payload patch."""

    payload = _payload()
    patch = rdkit_patch.rdkit_hydrogen_append_patch(
        payload,
        hydrogenated_molecule=_molecule_with_coordinates(
            ((6, Vec3(8.0, 8.0, 8.0)),)
        ),
        hydrogen_atom_names=(),
    )

    assert patch.atom_names == ("C1",)
    assert patch.position("C1") == payload.position("C1")


def test_rdkit_hydrogen_append_patch_rejects_normalized_duplicate_names() -> None:
    """Final H names must remain unique after canonical normalization."""

    molecule = _molecule_with_coordinates(
        (
            (1, Vec3(1.0, 0.0, 0.0)),
            (1, Vec3(-1.0, 0.0, 0.0)),
        )
    )

    with pytest.raises(ValueError, match="unique"):
        rdkit_patch.rdkit_hydrogen_append_patch(
            _payload(),
            hydrogenated_molecule=molecule,
            hydrogen_atom_names=("h1", " H1 "),
        )


def test_rdkit_hydrogen_append_patch_rejects_heavy_atom_name_collision() -> None:
    """A generated H must not overwrite an existing non-hydrogen atom."""

    with pytest.raises(ValueError, match="non-hydrogen"):
        rdkit_patch.rdkit_hydrogen_append_patch(
            _payload(),
            hydrogenated_molecule=_molecule_with_coordinates(
                ((1, Vec3(1.0, 0.0, 0.0)),)
            ),
            hydrogen_atom_names=("C1",),
        )


def test_rdkit_hydrogen_append_patch_can_replace_existing_hydrogen_position() -> None:
    """A generated H may update a same-named source H before reconciliation."""

    payload = completion_payload(
        component_id="LIG",
        residue_id=ResidueId("L", 1),
        atoms=(
            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
            atom_payload("HX", "H", Vec3(0.0, 1.0, 0.0)),
        ),
        is_hetero=True,
    )
    patch = rdkit_patch.rdkit_hydrogen_append_patch(
        payload,
        hydrogenated_molecule=_molecule_with_coordinates(
            ((1, Vec3(0.0, 2.0, 0.0)),)
        ),
        hydrogen_atom_names=(" hx ",),
    )

    assert patch.atom_names == ("C1", "HX")
    assert patch.position("HX") == Vec3(0.0, 2.0, 0.0)
    assert payload.apply_patch(patch).position("HX") == Vec3(0.0, 2.0, 0.0)


def test_rdkit_hydrogen_append_patch_rejects_missing_conformer() -> None:
    """RDKit molecules without coordinates cannot produce an append patch."""

    molecule = Chem.RWMol()
    molecule.AddAtom(Chem.Atom(1))

    with pytest.raises(ValueError):
        rdkit_patch.rdkit_hydrogen_append_patch(
            _payload(),
            hydrogenated_molecule=molecule.GetMol(),
            hydrogen_atom_names=("H1",),
        )


@pytest.mark.parametrize("non_finite", (float("nan"), float("inf")))
def test_rdkit_hydrogen_append_patch_rejects_non_finite_coordinates(
    non_finite: float,
) -> None:
    """RDKit coordinates must be finite before entering canonical geometry."""

    with pytest.raises(ValueError, match="finite"):
        rdkit_patch.rdkit_hydrogen_append_patch(
            _payload(),
            hydrogenated_molecule=_molecule_with_coordinates(
                ((1, Vec3(non_finite, 0.0, 0.0)),)
            ),
            hydrogen_atom_names=("H1",),
        )


@pytest.mark.parametrize("empty_name", ("", "   "))
def test_rdkit_hydrogen_append_patch_rejects_empty_normalized_name(
    empty_name: str,
) -> None:
    """Every generated coordinate needs a non-empty canonical atom identity."""

    with pytest.raises(ValueError, match="non-empty"):
        rdkit_patch.rdkit_hydrogen_append_patch(
            _payload(),
            hydrogenated_molecule=_molecule_with_coordinates(
                ((1, Vec3(1.0, 0.0, 0.0)),)
            ),
            hydrogen_atom_names=(empty_name,),
        )


def test_rdkit_hydrogen_append_patch_includes_deuterium_isotope() -> None:
    """Hydrogen isotopes retain atomic number one and must be projected."""

    editable_molecule = Chem.RWMol()
    deuterium = Chem.Atom(1)
    deuterium.SetIsotope(2)
    editable_molecule.AddAtom(deuterium)
    molecule = editable_molecule.GetMol()
    conformer = Chem.Conformer(1)
    conformer.SetAtomPosition(0, (1.0, 2.0, 3.0))
    molecule.AddConformer(conformer, assignId=True)

    patch = rdkit_patch.rdkit_hydrogen_append_patch(
        _payload(),
        hydrogenated_molecule=molecule,
        hydrogen_atom_names=("D1",),
    )

    assert patch.position("D1") == Vec3(1.0, 2.0, 3.0)


def test_rdkit_hydrogen_append_patch_preserves_large_finite_coordinate() -> None:
    """The projection boundary should not clamp finite RDKit coordinates."""

    patch = rdkit_patch.rdkit_hydrogen_append_patch(
        _payload(),
        hydrogenated_molecule=_molecule_with_coordinates(
            ((1, Vec3(1.0e308, 0.0, 0.0)),)
        ),
        hydrogen_atom_names=("H1",),
    )

    assert patch.position("H1") == Vec3(1.0e308, 0.0, 0.0)


def test_rdkit_hydrogen_append_patch_updates_and_appends_in_one_patch() -> None:
    """Existing H updates and new H appends should retain canonical atom order."""

    payload = completion_payload(
        component_id="LIG",
        residue_id=ResidueId("L", 1),
        atoms=(
            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
            atom_payload("HA", "H", Vec3(1.0, 0.0, 0.0)),
            atom_payload("HB", "H", Vec3(-1.0, 0.0, 0.0)),
        ),
        is_hetero=True,
    )
    patch = rdkit_patch.rdkit_hydrogen_append_patch(
        payload,
        hydrogenated_molecule=_molecule_with_coordinates(
            (
                (1, Vec3(2.0, 0.0, 0.0)),
                (1, Vec3(0.0, 2.0, 0.0)),
                (1, Vec3(-2.0, 0.0, 0.0)),
            )
        ),
        hydrogen_atom_names=("HA", "HC", "HB"),
    )

    assert patch.atom_names == ("C1", "HA", "HB", "HC")
    assert patch.position("HA") == Vec3(2.0, 0.0, 0.0)
    assert patch.position("HB") == Vec3(-2.0, 0.0, 0.0)
    assert patch.position("HC") == Vec3(0.0, 2.0, 0.0)


def test_evidence_hydrogen_names_count_all_hydrogen_isotopes() -> None:
    """Evidence naming and patch projection must share the atomic-number basis."""

    editable_molecule = Chem.RWMol()
    editable_molecule.AddAtom(Chem.Atom(6))
    deuterium = Chem.Atom(1)
    deuterium.SetIsotope(2)
    editable_molecule.AddAtom(deuterium)
    editable_molecule.AddAtom(Chem.Atom(1))

    assert rdkit_evidence._evidence_hydrogen_atom_names(
        editable_molecule.GetMol()
    ) == ("H001", "H002")


def _payload() -> CompletionResiduePayload:
    """Return one minimal retained non-polymer completion payload."""

    return completion_payload(
        component_id="LIG",
        residue_id=ResidueId("L", 1),
        atoms=(atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),),
        is_hetero=True,
    )


def _molecule_with_coordinates(
    atoms: tuple[tuple[int, Vec3], ...],
) -> Chem.Mol:
    """Return one RDKit molecule with atoms and one conformer in given order."""

    editable_molecule = Chem.RWMol()
    for atomic_number, _position in atoms:
        editable_molecule.AddAtom(Chem.Atom(atomic_number))

    molecule = editable_molecule.GetMol()
    conformer = Chem.Conformer(len(atoms))
    for atom_index, (_atomic_number, position) in enumerate(atoms):
        conformer.SetAtomPosition(atom_index, tuple(position))
    molecule.AddConformer(conformer, assignId=True)
    return molecule
