"""Orientation observation reuses graph facts, not coordinate scores."""

from unittest.mock import patch

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.refinement_benchmarks import load_case_structure
from tests.support.refinement_cases import REFINEMENT_BENCHMARK_CASES
from tests.support.refinement_contract import build_refinement_inputs

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.polar_contacts import PolarContactChemistry
from protrepair.diagnostics.clashes import prepare_clash_detection_basis
from protrepair.geometry import Vec3
from protrepair.state import OrientationCorrectionEligibilityState
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.discrete.axis_rotation import (
    select_best_discrete_correction_candidate,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.workflow.planning.intrinsic_geometry import (
    derive_orientation_correction_eligibility_state,
)


def test_orientation_observation_constructs_polar_chemistry_once() -> None:
    source = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])
    library = build_default_component_library()
    before = source.geometry

    with patch(
        "protrepair.diagnostics.clashes.PolarContactChemistry",
        wraps=PolarContactChemistry,
    ) as derive:
        state = derive_orientation_correction_eligibility_state(
            source, component_library=library
        )

    assert state is OrientationCorrectionEligibilityState.ELIGIBLE
    assert derive.call_count == 1
    assert source.geometry is before


@pytest.mark.parametrize("empty", [False, True])
def test_no_applicable_orientation_does_not_prepare_contacts(empty: bool) -> None:
    source = build_structure(
        source_format=FileFormat.PDB,
        chains=()
        if empty
        else (
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 1),
                        atoms=(atom_payload("CA", "C", Vec3(0, 0, 0)),),
                    ),
                ),
            ),
        ),
    )
    with patch(
        "protrepair.diagnostics.clashes.PolarContactChemistry",
        wraps=PolarContactChemistry,
    ) as derive:
        state = derive_orientation_correction_eligibility_state(
            source, component_library=build_default_component_library()
        )

    assert state is OrientationCorrectionEligibilityState.NOT_ELIGIBLE
    assert derive.call_count == 0


@pytest.mark.parametrize("case_id", ["3g8l-asn182", "1bkr-his42", "1xgo-leu253"])
@pytest.mark.parametrize("hydrogenated", [False, True])
def test_bound_orientation_score_and_selection_equal_fresh(
    case_id: str,
    hydrogenated: bool,
) -> None:
    case = REFINEMENT_BENCHMARK_CASES[case_id]
    source = load_case_structure(case)
    library = build_default_component_library()
    if hydrogenated:
        source = add_hydrogens(source, component_library=library).structure
    snapshot, atom_input, _ = build_refinement_inputs(
        source,
        LocalScopeSpec.from_residues(case.focus_residue_ids),
        context_radius_angstrom=4.0,
    )
    basis = prepare_clash_detection_basis(source, component_library=library)
    for residue_id in case.focus_residue_ids:
        fresh = select_best_discrete_correction_candidate(
            snapshot=snapshot,
            atom_input=atom_input,
            structure=source,
            residue_id=residue_id,
            component_library=library,
        )
        reused = select_best_discrete_correction_candidate(
            snapshot=snapshot,
            atom_input=atom_input,
            structure=source,
            residue_id=residue_id,
            component_library=library,
            clash_basis=basis,
        )

        assert reused == fresh
