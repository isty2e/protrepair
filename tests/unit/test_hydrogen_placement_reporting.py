"""Directive-level reporting tests for polymer hydrogen placement."""

import pytest
from tests.support.canonical_builders import atom_payload, completion_payload

from protrepair.chemistry import HydrogenSemantics
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import ValidationIssueKind
from protrepair.geometry import GeometryPlacementError, Vec3
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import ResidueIndex
from protrepair.transformer.completion.hydrogen import core as hydrogen_core
from protrepair.transformer.completion.hydrogen.directives import (
    BackboneHydrogenPropagationDirective,
    HistidineDeltaProtonationDirective,
    NTerminalHydrogenPlacementDirective,
    StaticHydrogenPlacementDirective,
)
from protrepair.transformer.completion.hydrogen.domain import (
    HydrogenCompletionEnvironment,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch


def _payload(
    component_id: str,
    residue_id: ResidueId,
    *,
    x_offset: float = 0.0,
) -> CompletionResiduePayload:
    return completion_payload(
        component_id=component_id,
        residue_id=residue_id,
        atoms=(
            atom_payload("N", "N", Vec3(x_offset, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(x_offset + 1.4, 0.0, 0.0)),
            atom_payload("C", "C", Vec3(x_offset + 2.2, 1.2, 0.0)),
            atom_payload("O", "O", Vec3(x_offset + 2.2, 2.3, 0.0)),
            atom_payload("CB", "C", Vec3(x_offset + 1.4, -0.8, 1.1)),
        ),
    )


def _environment(
    residues: tuple[CompletionResiduePayload, ...],
) -> HydrogenCompletionEnvironment:
    library = build_standard_component_library()
    return HydrogenCompletionEnvironment.from_payloads(
        residues,
        templates=tuple(library.require(residue.component_id) for residue in residues),
        disulfide_bonded_residue_ids=frozenset(),
    )


def _raise_geometry_error(*args: object, **kwargs: object) -> OrderedAtomPatch:
    del args, kwargs
    raise GeometryPlacementError("synthetic degenerate frame")


def test_static_hydrogen_failure_reports_template_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static-plan failure should identify the template H atoms left absent."""

    residue = _payload("ALA", ResidueId("A", 1))
    template = build_standard_component_library().require("ALA")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, HydrogenSemantics)
    directive = StaticHydrogenPlacementDirective(
        residue_index=ResidueIndex(0),
        template=template,
        semantics=semantics,
    )
    monkeypatch.setattr(hydrogen_core, "generate_hydrogen_patch", _raise_geometry_error)
    working_residues = [residue]

    issue = hydrogen_core._apply_hydrogen_directive(
        directive,
        chain_residues_by_index=working_residues,
        environment=_environment((residue,)),
    )

    assert issue is not None
    assert issue.kind is ValidationIssueKind.GEOMETRY_PLACEMENT_SKIPPED
    assert issue.residue_id == residue.residue_id
    assert issue.atom_names == template.expected_hydrogen_atom_names()
    assert working_residues == [residue]


def test_histidine_delta_failure_reports_only_hd1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HIS protonation failure should not implicate unrelated hydrogens."""

    residue = _payload("HIS", ResidueId("A", 2))
    template = build_standard_component_library().require("HIS")
    directive = HistidineDeltaProtonationDirective(
        residue_index=ResidueIndex(0),
        template=template,
    )
    monkeypatch.setattr(
        hydrogen_core,
        "histidine_delta_hydrogen",
        _raise_geometry_error,
    )

    issue = hydrogen_core._apply_hydrogen_directive(
        directive,
        chain_residues_by_index=[residue],
        environment=_environment((residue,)),
    )

    assert issue is not None
    assert issue.residue_id == residue.residue_id
    assert issue.atom_names == ("HD1",)


def test_backbone_failure_is_scoped_to_the_hydrogen_recipient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backbone-H failure belongs to the next residue receiving the atom."""

    first = _payload("ALA", ResidueId("A", 3))
    second = _payload("GLY", ResidueId("A", 4), x_offset=3.0)
    template = build_standard_component_library().require("ALA")
    directive = BackboneHydrogenPropagationDirective(
        residue_index=ResidueIndex(0),
        template=template,
        next_residue_index=ResidueIndex(1),
    )
    monkeypatch.setattr(hydrogen_core, "backbone_hydrogen", _raise_geometry_error)

    issue = hydrogen_core._apply_hydrogen_directive(
        directive,
        chain_residues_by_index=[first, second],
        environment=_environment((first, second)),
    )

    assert issue is not None
    assert issue.residue_id == second.residue_id
    assert issue.component_id == "GLY"
    assert issue.atom_names == ("H",)


@pytest.mark.parametrize(
    ("component_id", "expected_atom_names"),
    (
        pytest.param("ALA", ("H1", "H2", "H3"), id="non-proline"),
        pytest.param("PRO", ("H1", "H2"), id="proline"),
    ),
)
def test_n_terminal_failure_reports_backbone_family_targets(
    monkeypatch: pytest.MonkeyPatch,
    component_id: str,
    expected_atom_names: tuple[str, ...],
) -> None:
    """N-terminal failure should preserve the PRO/non-PRO target distinction."""

    residue = _payload(component_id, ResidueId("A", 5))
    template = build_standard_component_library().require(component_id)
    directive = NTerminalHydrogenPlacementDirective(
        residue_index=ResidueIndex(0),
        template=template,
        backbone_family_component_id=component_id,
    )
    monkeypatch.setattr(
        hydrogen_core,
        "n_terminal_hydrogen_coordinates",
        _raise_geometry_error,
    )

    issue = hydrogen_core._apply_hydrogen_directive(
        directive,
        chain_residues_by_index=[residue],
        environment=_environment((residue,)),
    )

    assert issue is not None
    assert issue.atom_names == expected_atom_names


def test_successful_static_hydrogen_directive_does_not_emit_false_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful placement must not create a stale skipped-placement issue."""

    residue = _payload("ALA", ResidueId("A", 6))
    template = build_standard_component_library().require("ALA")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, HydrogenSemantics)
    directive = StaticHydrogenPlacementDirective(
        residue_index=ResidueIndex(0),
        template=template,
        semantics=semantics,
    )

    def successful_patch(
        *,
        site: object,
        patch: OrderedAtomPatch,
        semantics: object,
    ) -> OrderedAtomPatch:
        del site, semantics
        return patch.append_atoms(("HX",), (Vec3(0.0, 0.0, 1.0),))

    monkeypatch.setattr(hydrogen_core, "generate_hydrogen_patch", successful_patch)
    working_residues = [residue]

    issue = hydrogen_core._apply_hydrogen_directive(
        directive,
        chain_residues_by_index=working_residues,
        environment=_environment((residue,)),
    )

    assert issue is None
    assert working_residues[0].has_atom("HX")
