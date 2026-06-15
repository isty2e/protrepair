"""Generic fragment-matching tests for partial residue repair."""

from tests.support.canonical_builders import (
    atom_payload,
    completion_payload,
)

from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import Vec3
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.fragment_matching import match_residue_fragment


def test_match_residue_fragment_reports_complete_residue_when_expected_atoms_present(
) -> None:
    """Complete residues should match as one connected template fragment."""

    template = build_standard_component_library().require("SER")
    residue = build_residue(
        "SER",
        ("N", "CA", "C", "O", "CB", "OG"),
    )

    match = match_residue_fragment(
        residue,
        template,
        exclude_atom_names=("OXT",),
    )

    assert match.is_complete()
    assert match.matched_atom_names == ("N", "CA", "C", "O", "CB", "OG")
    assert match.missing_atom_names == ()
    assert match.unexpected_atom_names == ()
    assert match.present_fragments == (("N", "CA", "C", "O", "CB", "OG"),)
    assert match.missing_fragments == ()
    assert match.boundary_bonds == ()


def test_match_residue_fragment_reports_sidechain_gap_frontier() -> None:
    """Missing side-chain atoms should expose matched frontier seeds and targets."""

    template = build_standard_component_library().require("SER")
    residue = build_residue(
        "SER",
        ("N", "CA", "C", "O", "CB"),
    )

    match = match_residue_fragment(
        residue,
        template,
        exclude_atom_names=("OXT",),
    )

    assert match.has_missing_atoms()
    assert match.missing_atom_names == ("OG",)
    assert match.present_fragments == (("N", "CA", "C", "O", "CB"),)
    assert match.missing_fragments == (("OG",),)
    assert match.repair_seed_atom_names() == ("CB",)
    assert match.repair_target_atom_names() == ("OG",)


def test_match_residue_fragment_handles_disconnected_present_fragment() -> None:
    """Disconnected observed fragments should remain explicit in the artifact."""

    template = build_standard_component_library().require("SER")
    residue = build_residue(
        "SER",
        ("N", "CA", "OG"),
    )

    match = match_residue_fragment(
        residue,
        template,
        exclude_atom_names=("OXT",),
    )

    assert match.present_fragments == (("N", "CA"), ("OG",))
    assert match.missing_fragments == (("C", "O"), ("CB",))
    assert match.repair_seed_atom_names() == ("CA", "OG")
    assert match.repair_target_atom_names() == ("C", "CB")
    assert match.largest_present_fragment() == ("N", "CA")


def test_match_residue_fragment_preserves_unexpected_atoms_separately() -> None:
    """Unexpected atoms should not pollute template fragment matching."""

    template = build_standard_component_library().require("SER")
    residue = build_residue(
        "SER",
        ("N", "CA", "C", "O", "CB", "OG", "XX"),
    )

    match = match_residue_fragment(
        residue,
        template,
        exclude_atom_names=("OXT",),
    )

    assert not match.is_complete()
    assert match.has_unexpected_atoms()
    assert match.unexpected_atom_names == ("XX",)


def test_match_residue_fragment_identifies_orphan_fragment_atoms() -> None:
    """Disconnected template atoms should be surfaced as orphans."""

    template = build_standard_component_library().require("SER")
    residue = build_residue(
        "SER",
        ("N", "CA", "C", "O", "OG"),
    )

    match = match_residue_fragment(
        residue,
        template,
        exclude_atom_names=("OXT",),
    )

    assert match.primary_repair_fragment(
        preferred_anchor_atom_names=("N", "CA", "C", "O"),
    ) == ("N", "CA", "C", "O")
    assert match.orphan_atom_names(
        preferred_anchor_atom_names=("N", "CA", "C", "O"),
    ) == ("OG",)


def build_residue(
    component_id: str,
    atom_names: tuple[str, ...],
):
    """Return one residue payload with dummy coordinates for fragment matching."""

    return completion_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=tuple(
            atom_payload(
                atom_name,
                element_for_atom_name(atom_name),
                Vec3(float(index), 0.0, 0.0),
            )
            for index, atom_name in enumerate(atom_names, start=1)
        ),
    )


def element_for_atom_name(atom_name: str) -> str:
    """Return a simple element guess sufficient for dummy test residues."""

    normalized_atom_name = atom_name.strip().upper()
    if normalized_atom_name.startswith("O"):
        return "O"

    if normalized_atom_name.startswith("N"):
        return "N"

    if normalized_atom_name.startswith("S"):
        return "S"

    return "C"
