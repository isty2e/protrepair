"""Site-local failure reporting through the real hydrogen materializer."""

from pathlib import Path

import pytest

from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.diagnostics import ValidationIssueKind
from protrepair.geometry import GeometryPlacementError, Vec3
from protrepair.io import read_structure_string
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.hydrogen import core as hydrogen_core
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch


def _structure(*numbers: int) -> ProteinStructure:
    lines = Path("tests/fixtures/corpus/pdb1afc.ent").read_text().splitlines()
    return read_structure_string(
        "\n".join(
            line
            for line in lines
            if line.startswith("ATOM")
            and line[21] == "A"
            and int(line[22:26]) in numbers
        ),
        FileFormat.PDB,
    )


def _raise_geometry_error(*args: object, **kwargs: object) -> OrderedAtomPatch:
    del args, kwargs
    raise GeometryPlacementError("synthetic degenerate frame")


def test_static_hydrogen_failure_reports_template_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _structure(47)
    monkeypatch.setattr(hydrogen_core, "generate_hydrogen_patch", _raise_geometry_error)
    result = add_hydrogens(source)
    issue = next(
        i
        for i in result.issues
        if i.kind is ValidationIssueKind.GEOMETRY_PLACEMENT_SKIPPED
    )
    assert issue.residue_id == ResidueId("A", 47)
    assert set(issue.atom_names) == {"HA", "HB1", "HB2", "HB3"}
    residue = result.structure.chain_site("A").residues[0]
    assert all(not residue.has_atom_site(name) for name in issue.atom_names)
    assert residue.has_atom_site("H1")


@pytest.mark.parametrize(
    ("numbers", "recipient", "kind", "expected_names"),
    (
        ((41,), 41, PolymerChemicalSite.SIDECHAIN, {"HE2", "HD2", "HE1"}),
        ((19, 20), 20, PolymerChemicalSite.BACKBONE_N, {"H"}),
        ((47,), 47, PolymerChemicalSite.BACKBONE_N, {"H1", "H2", "H3"}),
        ((11,), 11, PolymerChemicalSite.BACKBONE_N, {"H1", "H2"}),
    ),
)
def test_site_failure_is_scoped_and_does_not_apply_partial_graph(
    monkeypatch: pytest.MonkeyPatch,
    numbers: tuple[int, ...],
    recipient: int,
    kind: PolymerChemicalSite,
    expected_names: set[str],
) -> None:
    source = _structure(*numbers)
    original = hydrogen_core.place_polymer_microstate_hydrogens

    def fail_selected(context, residue_id, site, **kwargs):
        if residue_id == ResidueId("A", recipient) and site.kind is kind:
            raise GeometryPlacementError("synthetic degenerate frame")
        return original(context, residue_id, site, **kwargs)

    monkeypatch.setattr(
        hydrogen_core, "place_polymer_microstate_hydrogens", fail_selected
    )
    result = add_hydrogens(source)
    failures = [
        i
        for i in result.issues
        if i.kind is ValidationIssueKind.GEOMETRY_PLACEMENT_SKIPPED
    ]
    assert len(failures) == 1
    assert failures[0].residue_id == ResidueId("A", recipient)
    assert set(failures[0].atom_names) == expected_names
    residue = result.structure.constitution.residue_or_ligand(ResidueId("A", recipient))
    assert residue is not None
    assert all(not residue.has_atom_site(name) for name in expected_names)
    assert result.structure.provenance.microstate_overrides == ()


def test_successful_placement_has_no_false_geometry_warning() -> None:
    result = add_hydrogens(_structure(47))
    assert not result.issues
    assert result.structure.chain_site("A").residues[0].has_atom_site("HB1")


def test_unknown_static_hydrogen_is_not_added_without_a_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unknown_patch(*, site, patch, semantics, selected_hydrogen_names):
        del site, semantics, selected_hydrogen_names
        return patch.append_atoms(("HX",), (Vec3(0.0, 0.0, 1.0),))

    monkeypatch.setattr(hydrogen_core, "generate_hydrogen_patch", unknown_patch)
    result = add_hydrogens(_structure(47))
    assert any(i.atom_names == ("HX",) for i in result.issues)
    assert not result.structure.chain_site("A").residues[0].has_atom_site("HX")
