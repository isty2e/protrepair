"""H inventory completion does not resolve unknown metal-bound CYS chemistry."""

from pathlib import Path

import pytest
from tests.support.request_builders import ingress_options

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.chemistry.microstate.resolution import MicrostateResolutionStatus
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.errors import RefinementError
from protrepair.io import read_structure
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import (
    DirectRegionTransformationSpec,
    LocalScopeSpec,
    profile_transform_local_region,
    transform_local_region,
)
from protrepair.workflow.contracts import LigandPolicy, requested_process_goal

ZINC_CYSTEINES = frozenset(
    ResidueId(chain, number)
    for chain, numbers in (("A", (46, 111, 174)), ("B", (46, 97, 111, 174)))
    for number in numbers
)


@pytest.fixture(scope="module")
def prepared_6nbb():
    source = read_structure(
        Path(__file__).resolve().parents[2] / "fixtures/corpus/pdb6nbb.ent"
    )
    return process_structure(
        source,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )


def test_6nbb_h_goal_does_not_certify_zinc_cysteine_microstates(prepared_6nbb):
    result = prepared_6nbb
    structure = result.structure
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(structure), build_default_component_library()
    )

    assert result.requested_goals_fully_satisfied()
    for residue_id in ZINC_CYSTEINES:
        residue = structure.constitution.residue_or_ligand(residue_id)
        assert residue is not None and not residue.has_atom_site("HG")
        sulfur = structure.constitution.atom_index(AtomRef(residue_id, "SG"))
        assert structure.topology.formal_charge(sulfur) is None
        target = next(
            entry
            for entry in preparation.targets_for(residue_id)
            if entry.site.kind is PolymerChemicalSite.SIDECHAIN
        )
        assert target.resolution.status is MicrostateResolutionStatus.AMBIGUOUS
        assert target.resolution.graph is None
        assert not target.is_realized()

    warnings = {
        issue.residue_id
        for issue in result.issues
        if issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        and issue.severity is IssueSeverity.WARNING
    }
    assert ZINC_CYSTEINES <= warnings


@pytest.mark.parametrize(
    "transform", (transform_local_region, profile_transform_local_region)
)
def test_6nbb_public_local_refinement_refuses_unresolved_zinc_cysteines(
    prepared_6nbb, transform, monkeypatch
):
    def fail_backend(*args, **kwargs):
        raise AssertionError("unresolved metal-bound CYS reached force-field execution")

    monkeypatch.setattr(
        "protrepair.transformer.local.api.execute_local_transformation", fail_backend
    )
    monkeypatch.setattr(
        "protrepair.transformer.local.api.execute_local_transformation_profiled",
        fail_backend,
    )
    spec = DirectRegionTransformationSpec(
        scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 46),)),
        force_field=ContinuousRelaxationForceField.UFF,
    )

    with pytest.raises(
        RefinementError, match="requires realized polymer microstates.*A:46.*A:174"
    ):
        transform(prepared_6nbb.structure, spec)
