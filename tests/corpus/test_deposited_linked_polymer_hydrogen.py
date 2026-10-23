"""Deposited covalent and untyped attachments must not acquire isolated OH H."""

from pathlib import Path
from unittest.mock import patch

import pytest
from tests.support.request_builders import whole_structure_requested_goals

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import BondRelationshipType
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.continuous.binding import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.rdkit import RdkitContinuousRelaxationBackend
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions
from protrepair.workflow.contracts.request import WorkflowTransformRequests

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.mark.parametrize("refine", (False, True))
def test_4xlw_reports_current_linked_h_once_with_or_without_local_prerequisites(
    refine: bool,
) -> None:
    with patch.object(
        RdkitContinuousRelaxationBackend,
        "relax",
        side_effect=AssertionError("unresolved H must not reach native refinement"),
    ) as native:
        result = process_structure(
            FIXTURES / "pdb/linked_o_glycan_4xlw.pdb",
            ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
            requested_goals=whole_structure_requested_goals(
                HydrogenCoverageState.COMPLETE
            ),
            transform_requests=WorkflowTransformRequests(
                repair_refinement=RepairRefinementSpec(
                    scope_spec=LocalScopeSpec.from_residues((ResidueId("A", 458),)),
                    binding=RecommendedContinuousRelaxationBinding(),
                )
                if refine
                else None
            ),
        )
    native.assert_not_called()
    warnings = tuple(
        issue
        for issue in result.issues
        if issue.kind is ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN
    )
    assert len(warnings) == 1
    assert warnings[0].residue_id == ResidueId("A", 458)
    assert not result.requested_goals_fully_satisfied()
    assert result.branch_report is not None
    for outcome in result.branch_report.outcomes:
        assert outcome.warning_count == outcome.branch_quality_score.warning_count
        assert outcome.issue_count == outcome.branch_quality_score.issue_count
        assert outcome.warning_count >= 1


@pytest.mark.parametrize(
    ("path", "residue_id", "partner", "relationship"),
    (
        (
            FIXTURES / "corpus/pdb1jji.ent",
            ResidueId("A", 160),
            AtomRef(ResidueId("A", 455), "S"),
            BondRelationshipType.UNKNOWN,
        ),
        (
            FIXTURES / "pdb/linked_o_glycan_4xlw.pdb",
            ResidueId("A", 458),
            AtomRef(ResidueId("A", 604), "C1"),
            BondRelationshipType.COVALENT,
        ),
    ),
)
def test_deposited_link_leaves_only_substituted_h_inventory_unresolved(
    path: Path,
    residue_id: ResidueId,
    partner: AtomRef,
    relationship: BondRelationshipType,
) -> None:
    source = read_structure(path)
    parent = AtomRef(residue_id, "OG")
    parent_index = source.constitution.atom_index(parent)
    partner_index = source.constitution.atom_index(partner)
    original_bond = source.topology.bond_between(parent_index, partner_index)
    assert original_bond is not None and original_bond.relationship_type is relationship

    result = materialize_hydrogens_core(
        source, target_residue_ids=frozenset((residue_id,))
    )
    residue = result.structure.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    assert not residue.has_atom_site("HG")
    assert {"HA", "HB1", "HB2"} <= set(residue.atom_site_names())
    facts = StructureChemistryObservation(result.structure).residue_completion_facts(
        residue_id
    )
    assert facts.hydrogen_coverage_state is HydrogenCoverageState.PARTIAL
    assert any(issue.residue_id == residue_id for issue in result.issues)
    for index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        current = result.structure.constitution.atom_index(ref)
        assert result.structure.geometry.atom_geometry(current) == (
            source.geometry.atom_geometry(index)
        )
        if source.topology.formal_charge(index) is not None or ref == parent:
            assert result.structure.topology.formal_charge(current) == (
                source.topology.formal_charge(index)
            )
    for file_format in (FileFormat.PDB, FileFormat.MMCIF):
        reread = read_structure_string(
            write_structure_string(result.structure, file_format), file_format
        )
        bond = reread.topology.bond_between(
            reread.constitution.atom_index(parent),
            reread.constitution.atom_index(partner),
        )
        assert bond is not None


def test_2a1d_substituted_serines_share_the_same_graph_rule() -> None:
    source = read_structure(FIXTURES / "corpus/pdb2a1d.ent")
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    residue_ids = tuple(
        residue.residue_id
        for residue in source.constitution.iter_residues()
        if residue.component_id == "SER"
        if (fixed := preparation.fixed_hydrogen_context(residue.residue_id)) is not None
        and fixed.unresolved_parent_names == {"OG"}
    )
    assert len(residue_ids) == 2
    result = materialize_hydrogens_core(
        source, target_residue_ids=frozenset(residue_ids)
    )
    for residue_id in residue_ids:
        residue = result.structure.constitution.residue_or_ligand(residue_id)
        assert residue is not None and not residue.has_atom_site("HG")
        assert {"HA", "HB1", "HB2"} <= set(residue.atom_site_names())


@pytest.mark.parametrize(
    ("path", "component"),
    (
        (FIXTURES / "corpus/pdb1j2v.ent", "MSE"),
        (FIXTURES / "pdb/refinement/1uhg_sep68_local.pdb", "SEP"),
    ),
)
def test_unlinked_modified_backbone_is_not_reclassified(
    path: Path, component: str
) -> None:
    source = read_structure(path)
    residue = next(
        r for r in source.constitution.iter_residues() if r.component_id == component
    )
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    fixed = preparation.fixed_hydrogen_context(residue.residue_id)
    assert fixed is not None and not fixed.unresolved_parent_names
    result = materialize_hydrogens_core(
        source, target_residue_ids=frozenset((residue.residue_id,))
    )
    current = result.structure.constitution.residue_or_ligand(residue.residue_id)
    assert current is not None and any(
        atom.is_hydrogen() for atom in current.atom_sites
    )
    assert not any("linked polymer H" in issue.message for issue in result.issues)
