"""Contract tests for histidine protonation request resolution artifacts."""

import math
from dataclasses import FrozenInstanceError
from fractions import Fraction
from typing import get_args

import pytest
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)
from tests.support.canonical_builders import chain_payload

from protrepair.chemistry import build_default_component_library
from protrepair.structure import ProteinStructure
from protrepair.structure.constitution import AtomSite, ChainSite, ResidueSite
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import ResidueIndex
from protrepair.transformer.completion.hydrogen.directives import (
    HistidineDeltaProtonationDirective,
    derive_hydrogen_directives,
)
from protrepair.transformer.completion.hydrogen.protonation import (
    DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO,
    DisabledHistidineProtonationRequest,
    HistidineDeltaProtonationAssignment,
    HistidineProtonationRequest,
    PrasRatioHistidineProtonationRequest,
    normalize_histidine_protonation_request,
    resolve_histidine_protonation_assignments,
)
from protrepair.transformer.completion.hydrogen.repair import add_hydrogens
from protrepair.workflow.contracts import WorkflowTransformRequests


def test_disabled_histidine_protonation_request_is_a_request() -> None:
    assert get_args(HistidineProtonationRequest) == (
        DisabledHistidineProtonationRequest,
        PrasRatioHistidineProtonationRequest,
    )


def test_pras_ratio_histidine_protonation_request_normalizes_ratio() -> None:
    request = PrasRatioHistidineProtonationRequest()

    assert request.ratio == DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO
    assert PrasRatioHistidineProtonationRequest(ratio=0).ratio == 0.0
    assert PrasRatioHistidineProtonationRequest(ratio=1).ratio == 1.0


@pytest.mark.parametrize(
    "ratio",
    (True, False, "0.2", 0.2 + 0j, Fraction(1, 5), object()),
)
def test_pras_ratio_histidine_request_rejects_non_numeric_ratio(ratio) -> None:
    with pytest.raises(TypeError, match="ratio must be an int or float"):
        PrasRatioHistidineProtonationRequest(ratio=ratio)  # type: ignore[arg-type]


@pytest.mark.parametrize("ratio", (math.nan, math.inf, -math.inf))
def test_pras_ratio_histidine_request_rejects_non_finite_ratio(
    ratio: float,
) -> None:
    with pytest.raises(ValueError, match="ratio must be finite"):
        PrasRatioHistidineProtonationRequest(ratio=ratio)


@pytest.mark.parametrize("ratio", (-0.001, 1.001))
def test_pras_ratio_histidine_request_rejects_out_of_range_ratio(
    ratio: float,
) -> None:
    with pytest.raises(ValueError, match=r"ratio must be in \[0\.0, 1\.0\]"):
        PrasRatioHistidineProtonationRequest(ratio=ratio)


def test_histidine_delta_assignment_records_residue() -> None:
    assignment = HistidineDeltaProtonationAssignment(ResidueIndex(3))

    assert assignment.residue_index == ResidueIndex(3)


def test_histidine_delta_assignment_rejects_raw_residue_index() -> None:
    with pytest.raises(TypeError, match="residue_index must be a ResidueIndex"):
        HistidineDeltaProtonationAssignment(3)  # type: ignore[arg-type]


def test_histidine_protonation_contract_values_are_immutable() -> None:
    request = PrasRatioHistidineProtonationRequest()
    assignment = HistidineDeltaProtonationAssignment(ResidueIndex(3))

    with pytest.raises(FrozenInstanceError):
        request.ratio = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        assignment.residue_index = ResidueIndex(4)  # type: ignore[misc]


def test_normalize_histidine_protonation_request_defaults_to_disabled() -> None:
    request = normalize_histidine_protonation_request()

    assert isinstance(request, DisabledHistidineProtonationRequest)


def test_normalize_histidine_protonation_request_maps_legacy_bool() -> None:
    request = normalize_histidine_protonation_request(protonate_histidines=True)

    assert request == PrasRatioHistidineProtonationRequest()


def test_normalize_histidine_protonation_request_preserves_typed_request() -> None:
    typed_request = PrasRatioHistidineProtonationRequest(ratio=0.5)

    request = normalize_histidine_protonation_request(typed_request)

    assert request is typed_request


def test_normalize_histidine_protonation_request_rejects_conflicting_inputs() -> None:
    with pytest.raises(ValueError, match="must not both be set"):
        normalize_histidine_protonation_request(
            PrasRatioHistidineProtonationRequest(),
            protonate_histidines=True,
        )


def test_normalize_histidine_protonation_request_rejects_non_bool_shorthand() -> None:
    with pytest.raises(TypeError, match="protonate_histidines must be a bool"):
        normalize_histidine_protonation_request(
            protonate_histidines=1,  # type: ignore[arg-type]
        )


def test_workflow_transform_requests_consumes_legacy_histidine_bool() -> None:
    requests = WorkflowTransformRequests(protonate_histidines=True)

    assert requests.protonate_histidines is False
    assert (
        requests.histidine_protonation_request()
        == PrasRatioHistidineProtonationRequest()
    )


def test_workflow_transform_requests_rejects_invalid_histidine_request() -> None:
    with pytest.raises(TypeError, match="HistidineProtonationRequest"):
        WorkflowTransformRequests(histidine_protonation=object())  # type: ignore[arg-type]


def test_workflow_transform_requests_rejects_non_bool_histidine_shorthand() -> None:
    with pytest.raises(TypeError, match="protonate_histidines must be a bool"):
        WorkflowTransformRequests(protonate_histidines=1)  # type: ignore[arg-type]


def test_add_hydrogens_rejects_conflicting_histidine_inputs() -> None:
    with pytest.raises(ValueError, match="must not both be set"):
        add_hydrogens(
            _empty_chain_structure(),
            histidine_protonation=PrasRatioHistidineProtonationRequest(),
            protonate_histidines=True,
        )


@pytest.mark.parametrize(
    ("ratio", "expected_indices"),
    (
        pytest.param(0.0, (), id="zero"),
        pytest.param(0.2, (0,), id="pras-default-five-his"),
        pytest.param(0.5, (0, 1), id="floor-two-of-five"),
        pytest.param(1.0, (0, 1, 3, 4, 5), id="all"),
    ),
)
def test_resolve_histidine_assignments_uses_chain_order_and_floor(
    ratio: float,
    expected_indices: tuple[int, ...],
) -> None:
    assignments = resolve_histidine_protonation_assignments(
        _chain("HIS", "HIS", "ALA", "HIS", "HIS", "HIS"),
        PrasRatioHistidineProtonationRequest(ratio=ratio),
    )

    assert tuple(assignment.residue_index.value for assignment in assignments) == (
        expected_indices
    )


def test_resolve_histidine_assignments_preserves_four_his_threshold() -> None:
    assignments = resolve_histidine_protonation_assignments(
        _chain("HIS", "HIS", "HIS", "HIS"),
        PrasRatioHistidineProtonationRequest(),
    )

    assert assignments == ()


def test_resolve_histidine_assignments_handles_no_his() -> None:
    assignments = resolve_histidine_protonation_assignments(
        _chain("ALA", "GLY", "SER"),
        PrasRatioHistidineProtonationRequest(),
    )

    assert assignments == ()


def test_derive_hydrogen_directives_lowers_histidine_assignments() -> None:
    chain = _chain("HIS")
    his_template = build_default_component_library().get("HIS")
    assert his_template is not None

    directives = derive_hydrogen_directives(
        chain,
        templates=(his_template,),
        histidine_protonation_assignments=(
            HistidineDeltaProtonationAssignment(ResidueIndex(0)),
        ),
    )

    assert any(
        isinstance(directive, HistidineDeltaProtonationDirective)
        and directive.residue_index == ResidueIndex(0)
        for directive in directives
    )


def test_derive_hydrogen_directives_noops_missing_histidine_template() -> None:
    directives = derive_hydrogen_directives(
        _chain("HIS"),
        templates=(None,),
        histidine_protonation_assignments=(
            HistidineDeltaProtonationAssignment(ResidueIndex(0)),
        ),
    )

    assert not any(
        isinstance(directive, HistidineDeltaProtonationDirective)
        for directive in directives
    )


def test_derive_hydrogen_directives_rejects_duplicate_histidine_assignments() -> None:
    chain = _chain("HIS")
    his_template = build_default_component_library().get("HIS")

    with pytest.raises(ValueError, match="must not repeat"):
        derive_hydrogen_directives(
            chain,
            templates=(his_template,),
            histidine_protonation_assignments=(
                HistidineDeltaProtonationAssignment(ResidueIndex(0)),
                HistidineDeltaProtonationAssignment(ResidueIndex(0)),
            ),
        )


def test_derive_hydrogen_directives_rejects_out_of_scope_assignment() -> None:
    with pytest.raises(ValueError, match="outside the chain"):
        derive_hydrogen_directives(
            _chain("HIS"),
            templates=(None,),
            histidine_protonation_assignments=(
                HistidineDeltaProtonationAssignment(ResidueIndex(1)),
            ),
        )


def test_derive_hydrogen_directives_rejects_non_histidine_assignment() -> None:
    ala_template = build_default_component_library().get("ALA")

    with pytest.raises(ValueError, match="must target HIS"):
        derive_hydrogen_directives(
            _chain("ALA"),
            templates=(ala_template,),
            histidine_protonation_assignments=(
                HistidineDeltaProtonationAssignment(ResidueIndex(0)),
            ),
        )


def _chain(*component_ids: str) -> ChainSite:
    return ChainSite(
        chain_id="A",
        residues=tuple(
            ResidueSite(
                component_id=component_id,
                residue_id=ResidueId(chain_id="A", seq_num=residue_offset + 1),
                atom_sites=(AtomSite("N", "N"),),
            )
            for residue_offset, component_id in enumerate(component_ids)
        ),
    )


def _empty_chain_structure() -> ProteinStructure:
    return build_canonical_structure(
        chains=(chain_payload("A", ()),),
        source_format=FileFormat.PDB,
        source_name="histidine-protonation-conflict",
    )
