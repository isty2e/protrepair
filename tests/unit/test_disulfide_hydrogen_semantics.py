"""Topology-authoritative disulfide hydrogen normalization."""

from typing import cast

import pytest
from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.state import (
    HydrogenCoverageState,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.state.structure_topology import (
    DisulfideHydrogenContradiction,
    StructureDisulfideHydrogenFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.disulfide import disulfide_atom_ref_pairs
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.base import ProjectedDomainState
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.actions.disulfide_hydrogen import (
    DisulfideHydrogenNormalizationTransformer,
)
from protrepair.workflow.actions.disulfide_topology import (
    DisulfideTopologyResolutionTransformer,
)
from protrepair.workflow.contracts import (
    LigandPolicy,
    RequestedGoalSet,
    StructureIngressOptions,
    WorkflowPhaseStatus,
    WorkflowPlanningContext,
    WorkflowPlanningPhase,
    WorkflowTransformRequests,
)
from protrepair.workflow.engine.reporting import evaluate_workflow_phase_outcomes
from protrepair.workflow.planning.planner import plan_workflow_actions

_DEFAULT_LEFT_CYS_ID = ResidueId("A", 1, "A")
_DEFAULT_RIGHT_CYS_ID = ResidueId("B", 1)


@pytest.mark.parametrize(
    (
        "relationship_type",
        "left_hydrogen",
        "right_hydrogen",
        "expected_atom_names",
        "expected_endpoint_count",
    ),
    (
        (BondRelationshipType.COVALENT, ("HG", "H"), None, ("HG",), 1),
        (
            BondRelationshipType.DISULFIDE,
            ("DG", "D"),
            ("TG", "T"),
            ("DG", "TG"),
            2,
        ),
        (None, ("HG", "H"), None, (), 0),
        (BondRelationshipType.UNKNOWN, ("HG", "H"), None, (), 0),
        (BondRelationshipType.METAL_COORDINATION, ("HG", "H"), None, (), 0),
        (BondRelationshipType.DISULFIDE, ("HG", "C"), None, (), 0),
    ),
)
def test_disulfide_hydrogen_facts_follow_relationship_and_element_identity(
    relationship_type: BondRelationshipType | None,
    left_hydrogen: tuple[str, str] | None,
    right_hydrogen: tuple[str, str] | None,
    expected_atom_names: tuple[str, ...],
    expected_endpoint_count: int,
) -> None:
    """Relationship, isotope identity, and endpoint coverage remain orthogonal."""

    structure = _disulfide_structure(
        relationship_type=relationship_type,
        left_hydrogen=left_hydrogen,
        right_hydrogen=right_hydrogen,
    )

    facts = StructureDisulfideHydrogenFacts.from_structure(structure)

    assert tuple(
        atom_ref.atom_name for atom_ref in facts.forbidden_hydrogen_atom_refs()
    ) == expected_atom_names
    assert sum(
        contradiction.present_endpoint_count()
        for contradiction in facts.contradictions
    ) == expected_endpoint_count


@pytest.mark.parametrize(
    ("element", "relationship_type", "expected"),
    (
        ("H", BondRelationshipType.COVALENT, True),
        ("D", BondRelationshipType.COVALENT, True),
        ("T", BondRelationshipType.COVALENT, True),
        ("H", BondRelationshipType.UNKNOWN, False),
    ),
)
def test_nonstandard_hydrogen_name_requires_explicit_covalent_sg_attachment(
    element: str,
    relationship_type: BondRelationshipType,
    expected: bool,
) -> None:
    """Nonstandard names require canonical local attachment topology."""

    structure = _disulfide_structure(
        relationship_type=BondRelationshipType.DISULFIDE,
        left_hydrogen=("Q1", element),
    )
    structure = _append_bond(
        structure,
        AtomRef(ResidueId("A", 1, "A"), "SG"),
        AtomRef(ResidueId("A", 1, "A"), "Q1"),
        relationship_type=relationship_type,
    )

    assert (
        StructureDisulfideHydrogenFacts.from_structure(
            structure
        ).has_contradictions()
        is expected
    )


def test_same_chain_insertion_coded_disulfide_uses_the_same_contract() -> None:
    """Chain locality and insertion codes do not alter disulfide chemistry."""

    left_id = ResidueId("A", 1, "A")
    right_id = ResidueId("A", 2)
    structure = _disulfide_structure(
        relationship_type=BondRelationshipType.DISULFIDE,
        left_hydrogen=("TG", "T"),
        left_id=left_id,
        right_id=right_id,
    )

    assert StructureDisulfideHydrogenFacts.from_structure(
        structure
    ).forbidden_hydrogen_atom_refs() == (AtomRef(left_id, "TG"),)


@pytest.mark.parametrize("provenance", tuple(BondProvenance))
def test_disulfide_hydrogen_semantics_ignore_provenance_and_nonideal_distance(
    provenance: BondProvenance,
) -> None:
    """Canonical S-S truth is independent of support mode and current geometry."""

    structure = _disulfide_structure(
        relationship_type=BondRelationshipType.DISULFIDE,
        left_hydrogen=("HG", "H"),
        right_sg_x=8.0,
        provenance=provenance,
    )

    assert StructureDisulfideHydrogenFacts.from_structure(
        structure
    ).has_contradictions()


def test_ambiguous_three_cysteine_geometry_does_not_expand_bonded_scope() -> None:
    """A nearby unbonded CYS does not inherit another pair's microstate."""

    residue_specs = (
        (ResidueId("A", 1), 0.0, -1.0),
        (ResidueId("B", 1), 2.1, 1.0),
        (ResidueId("C", 1), 1.0, 1.0),
    )
    structure = build_structure(
        chains=tuple(
            chain_payload(
                residue_id.chain_id,
                (
                    _complete_cys_payload(
                        residue_id,
                        sg_x=sg_x,
                        direction=direction,
                        thiol_hydrogen=("HG", "H"),
                    ),
                ),
            )
            for residue_id, sg_x, direction in residue_specs
        ),
        source_format=FileFormat.PDB,
        source_name="ambiguous-three-cysteine-hydrogen",
    )
    structure = _append_bond(
        structure,
        AtomRef(ResidueId("A", 1), "SG"),
        AtomRef(ResidueId("B", 1), "SG"),
        relationship_type=BondRelationshipType.DISULFIDE,
    )

    assert StructureDisulfideHydrogenFacts.from_structure(
        structure
    ).forbidden_hydrogen_atom_refs() == (
        AtomRef(ResidueId("A", 1), "HG"),
        AtomRef(ResidueId("B", 1), "HG"),
    )


def test_runtime_type_validation_precedes_identity_sorting() -> None:
    """Malformed values fail through typed contract errors, not ordering errors."""

    with pytest.raises(TypeError, match="requires AtomRef values"):
        DisulfideHydrogenNormalizationTransformer(
            forbidden_hydrogen_atom_refs=(
                AtomRef(ResidueId("A", 1), "HG"),
                cast(AtomRef, "bad"),
            )
        )
    with pytest.raises(TypeError, match="AtomRef endpoints"):
        DisulfideHydrogenContradiction(
            disulfide_atom_ref_pair=cast(
                tuple[AtomRef, AtomRef],
                (AtomRef(ResidueId("A", 1), "SG"), "bad"),
            ),
            forbidden_hydrogen_atom_refs=(
                AtomRef(ResidueId("A", 1), "HG"),
            ),
        )


def test_action_requires_exact_current_contradiction_set() -> None:
    """Partial and stale plans must be rejected for planner re-observation."""

    structure = _disulfide_structure(
        relationship_type=BondRelationshipType.DISULFIDE,
        left_hydrogen=("HG", "H"),
        right_hydrogen=("DG", "D"),
    )
    left_ref = AtomRef(ResidueId("A", 1, "A"), "HG")
    action = DisulfideHydrogenNormalizationTransformer(
        forbidden_hydrogen_atom_refs=(left_ref,)
    )

    assert not action.accepts_projected_domain(
        ProjectedDomainState(scope=action.workflow_scope, state=structure),
        context=_execution_context(structure),
    )
    assert not action.accepts_projected_domain(
        ProjectedDomainState(
            scope=action.workflow_scope,
            state=structure.without_atom_refs((left_ref,)),
        ),
        context=_execution_context(structure),
    )


def test_planner_selects_typed_normalization_before_general_chemistry() -> None:
    """Forbidden-present chemistry is automatic and precedes H completion."""

    structure = _disulfide_structure(
        relationship_type=BondRelationshipType.DISULFIDE,
        left_hydrogen=("HG", "H"),
    )

    planning = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
    )

    assert isinstance(
        planning.transformers[0],
        DisulfideHydrogenNormalizationTransformer,
    )
    assert planning.state_deficit is not None
    assert planning.state_deficit.disulfide_hydrogen is not None


def test_workflow_removes_only_forbidden_isotopes_and_is_idempotent() -> None:
    """Normalization preserves other source geometry and the canonical S-S bond."""

    structure = _disulfide_structure(
        relationship_type=BondRelationshipType.DISULFIDE,
        left_hydrogen=("HG", "H"),
        right_hydrogen=("DG", "D"),
    )
    preserved_ref = AtomRef(ResidueId("A", 1, "A"), "HA")
    preserved_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(preserved_ref)
    )

    first = process_structure(structure)
    second = process_structure(first.structure)

    assert first.structure.geometry.atom_geometry(
        first.structure.constitution.atom_index(preserved_ref)
    ) == preserved_geometry
    assert disulfide_atom_ref_pairs(first.structure) == disulfide_atom_ref_pairs(
        structure
    )
    assert not StructureDisulfideHydrogenFacts.from_structure(
        first.structure
    ).has_contradictions()
    assert tuple(
        repair.atom_names
        for repair in first.repairs
        if repair.kind is RepairEventKind.HYDROGENS_REMOVED
    ) == (("HG",), ("DG",))
    assert second.structure == first.structure
    assert not any(
        repair.kind is RepairEventKind.HYDROGENS_REMOVED
        for repair in second.repairs
    )
    _, chemistry_facts = derive_structure_coverage_and_chemistry_readiness_facts(
        first.structure,
        component_library=build_default_component_library(),
    )
    assert chemistry_facts.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_normalized_disulfide_roundtrips_without_reintroducing_hydrogen(
    file_format: FileFormat,
) -> None:
    """Both writers project normalized atom inventory and retained S-S truth."""

    normalized = process_structure(
        _disulfide_structure(
            relationship_type=BondRelationshipType.DISULFIDE,
            left_hydrogen=("HG", "H"),
        )
    ).structure
    roundtripped = read_structure_string(
        write_structure_string(normalized, file_format),
        file_format,
    )

    left = roundtripped.constitution.residue_or_ligand(
        ResidueId("A", 1, "A")
    )
    assert left is not None
    assert not left.has_atom_site("HG")
    assert disulfide_atom_ref_pairs(roundtripped)


def test_retained_disulfide_uses_the_same_normalization_action() -> None:
    """Retained CYS does not introduce a second microstate contradiction axis."""

    structure = _disulfide_structure(
        relationship_type=BondRelationshipType.DISULFIDE,
        left_hydrogen=("HG", "H"),
        left_hetero=True,
        right_hetero=True,
        include_passive_polymer_context=True,
    )

    result = process_structure(
        structure,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
    )

    left = result.structure.constitution.residue_or_ligand(
        ResidueId("A", 1, "A")
    )
    assert left is not None
    assert not left.has_atom_site("HG")
    assert disulfide_atom_ref_pairs(result.structure)


def test_shared_disulfide_endpoint_keeps_pair_facts_but_deduplicates_action() -> None:
    """Malformed source topology stays observable without duplicate atom work."""

    residue_specs = (
        (ResidueId("A", 1), 0.0),
        (ResidueId("B", 1), 2.1),
        (ResidueId("C", 1), -2.1),
    )
    structure = build_structure(
        chains=tuple(
            chain_payload(
                residue_id.chain_id,
                (
                    _complete_cys_payload(
                        residue_id,
                        sg_x=sg_x,
                        direction=-1.0 if index == 0 else 1.0,
                        thiol_hydrogen=("HG", "H") if index == 0 else None,
                    ),
                ),
            )
            for index, (residue_id, sg_x) in enumerate(residue_specs)
        ),
        source_format=FileFormat.PDB,
        source_name="shared-disulfide-endpoint",
    )
    for partner_id in (ResidueId("B", 1), ResidueId("C", 1)):
        structure = _append_bond(
            structure,
            AtomRef(ResidueId("A", 1), "SG"),
            AtomRef(partner_id, "SG"),
            relationship_type=BondRelationshipType.DISULFIDE,
        )

    facts = StructureDisulfideHydrogenFacts.from_structure(structure)

    assert len(facts.contradictions) == 2
    assert facts.forbidden_hydrogen_atom_refs() == (
        AtomRef(ResidueId("A", 1), "HG"),
    )
    action = DisulfideHydrogenNormalizationTransformer(
        forbidden_hydrogen_atom_refs=facts.forbidden_hydrogen_atom_refs()
    )
    normalized = action.execute(
        TransformationResult(structure=structure, repairs=(), issues=()),
        context=_execution_context(structure),
    ).structure
    assert len(disulfide_atom_ref_pairs(normalized)) == 2


def test_topology_resolution_exposes_hydrogen_contradiction_on_reobservation() -> None:
    """Geometry evidence becomes chemistry truth only after topology writing."""

    structure = _disulfide_structure(
        relationship_type=None,
        left_hydrogen=("HG", "H"),
    )

    initial = plan_workflow_actions(
        structure,
        requested_goals=RequestedGoalSet(),
        transform_requests=WorkflowTransformRequests(),
    )
    result = process_structure(structure)

    assert isinstance(initial.transformers[0], DisulfideTopologyResolutionTransformer)
    assert disulfide_atom_ref_pairs(result.structure)
    assert not StructureDisulfideHydrogenFacts.from_structure(
        result.structure
    ).has_contradictions()
    repair_kinds = tuple(repair.kind for repair in result.repairs)
    assert repair_kinds.index(RepairEventKind.DISULFIDE_TOPOLOGY_RESOLVED) < (
        repair_kinds.index(RepairEventKind.HYDROGENS_REMOVED)
    )


def test_phase_reporting_does_not_call_contradictory_chemistry_clear() -> None:
    """Terminal reporting observes chemistry contradictions directly."""

    structure = _disulfide_structure(
        relationship_type=BondRelationshipType.DISULFIDE,
        left_hydrogen=("HG", "H"),
    )

    chemistry_outcome = next(
        outcome
        for outcome in evaluate_workflow_phase_outcomes(
            structure,
            planning_context=WorkflowPlanningContext(),
            component_library=build_default_component_library(),
            blockers=(),
        )
        if outcome.phase is WorkflowPlanningPhase.CHEMISTRY_NORMALIZATION
    )

    assert chemistry_outcome.status is WorkflowPhaseStatus.UNRESOLVED
    assert "disulfide" in (chemistry_outcome.details or "")


def test_selective_removal_preserves_orthogonal_sulfur_charge_and_metadata() -> None:
    """Hydrogen normalization does not adjudicate charge, blueprint, or provenance."""

    structure = _disulfide_structure(
        relationship_type=BondRelationshipType.DISULFIDE,
        left_hydrogen=("HG", "H"),
    )
    sulfur_ref = AtomRef(ResidueId("A", 1, "A"), "SG")
    sulfur_index = structure.constitution.atom_index(sulfur_ref)
    atom_topologies = list(structure.topology.atom_topologies)
    atom_topologies[sulfur_index.value] = AtomTopology(formal_charge=-1)
    structure = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=tuple(atom_topologies),
            bonds=structure.topology.bonds,
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    facts = StructureDisulfideHydrogenFacts.from_structure(structure)
    normalized = DisulfideHydrogenNormalizationTransformer(
        forbidden_hydrogen_atom_refs=facts.forbidden_hydrogen_atom_refs()
    ).execute(
        TransformationResult(structure=structure, repairs=(), issues=()),
        context=_execution_context(structure),
    ).structure

    assert normalized.topology.formal_charge(
        normalized.constitution.atom_index(sulfur_ref)
    ) == -1
    assert normalized.polymer_blueprint is structure.polymer_blueprint
    assert normalized.provenance is structure.provenance


def _disulfide_structure(
    *,
    relationship_type: BondRelationshipType | None,
    left_hydrogen: tuple[str, str] | None = None,
    right_hydrogen: tuple[str, str] | None = None,
    left_id: ResidueId = _DEFAULT_LEFT_CYS_ID,
    right_id: ResidueId = _DEFAULT_RIGHT_CYS_ID,
    left_hetero: bool = False,
    right_hetero: bool = False,
    include_passive_polymer_context: bool = False,
    right_sg_x: float = 2.1,
    provenance: BondProvenance = BondProvenance.SOURCE_EXPLICIT,
) -> ProteinStructure:
    """Build one complete CYS pair with optional canonical SG-SG topology."""

    payloads = (
        _complete_cys_payload(
            left_id,
            sg_x=0.0,
            direction=-1.0,
            thiol_hydrogen=left_hydrogen,
            is_hetero=left_hetero,
        ),
        _complete_cys_payload(
            right_id,
            sg_x=right_sg_x,
            direction=1.0,
            thiol_hydrogen=right_hydrogen,
            is_hetero=right_hetero,
        ),
    )
    polymer_payloads_by_chain_id: dict[str, list[CanonicalResiduePayload]] = {}
    ligand_payloads: list[CanonicalResiduePayload] = []
    for residue_id, is_hetero, payload in zip(
        (left_id, right_id),
        (left_hetero, right_hetero),
        payloads,
        strict=True,
    ):
        if is_hetero:
            ligand_payloads.append(payload)
        else:
            polymer_payloads_by_chain_id.setdefault(residue_id.chain_id, []).append(
                payload
            )
    if include_passive_polymer_context:
        passive_chain_ids = tuple(
            dict.fromkeys((left_id.chain_id, right_id.chain_id))
        )
        for index, chain_id in enumerate(passive_chain_ids):
            polymer_payloads_by_chain_id.setdefault(chain_id, []).append(
                _passive_polymer_payload(
                    chain_id,
                    x_offset=100.0 + 20.0 * index,
                )
            )
    structure = build_structure(
        chains=tuple(
            chain_payload(chain_id, tuple(chain_payloads))
            for chain_id, chain_payloads in polymer_payloads_by_chain_id.items()
        ),
        ligands=tuple(ligand_payloads),
        source_format=FileFormat.PDB,
        source_name="disulfide-hydrogen-semantics",
    )
    if relationship_type is None:
        return structure
    return _append_bond(
        structure,
        AtomRef(left_id, "SG"),
        AtomRef(right_id, "SG"),
        relationship_type=relationship_type,
        provenance=provenance,
    )


def _complete_cys_payload(
    residue_id: ResidueId,
    *,
    sg_x: float,
    direction: float,
    thiol_hydrogen: tuple[str, str] | None,
    is_hetero: bool = False,
) -> CanonicalResiduePayload:
    """Build one CYS payload complete for current hydrogen expectations."""

    cb_x = sg_x + direction * 1.8
    ca_x = cb_x + direction * 1.4
    n_x = ca_x + direction * 1.3
    atoms = [
        atom_payload("N", "N", Vec3(n_x, 0.0, 0.0)),
        atom_payload("CA", "C", Vec3(ca_x, 0.0, 0.0)),
        atom_payload("C", "C", Vec3(ca_x, 1.5, 0.0)),
        atom_payload("O", "O", Vec3(ca_x, 2.5, 0.0)),
        atom_payload("CB", "C", Vec3(cb_x, 0.0, 0.0)),
        atom_payload("SG", "S", Vec3(sg_x, 0.0, 0.0)),
        atom_payload("HA", "H", Vec3(ca_x, -1.0, 0.0)),
        atom_payload("HB1", "H", Vec3(cb_x, 0.8, 0.0)),
        atom_payload("HB2", "H", Vec3(cb_x, -0.8, 0.0)),
    ]
    if not is_hetero:
        atoms.extend(
            (
                atom_payload("H1", "H", Vec3(n_x, 0.8, 0.0)),
                atom_payload("H2", "H", Vec3(n_x, -0.8, 0.0)),
                atom_payload("H3", "H", Vec3(n_x, 0.0, 0.8)),
            )
        )
    if thiol_hydrogen is not None:
        atom_name, element = thiol_hydrogen
        atoms.append(
            atom_payload(
                atom_name,
                element,
                Vec3(sg_x - direction, 0.0, 0.0),
            )
        )
    return residue_payload(
        component_id="CYS",
        residue_id=residue_id,
        atoms=tuple(atoms),
        is_hetero=is_hetero,
    )


def _passive_polymer_payload(
    chain_id: str,
    *,
    x_offset: float,
) -> CanonicalResiduePayload:
    """Build distant polymer context for retained-only workflow reporting."""

    return residue_payload(
        component_id="GLY",
        residue_id=ResidueId(chain_id, 100),
        atoms=(
            atom_payload("N", "N", Vec3(x_offset, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(x_offset + 1.4, 0.0, 0.0)),
            atom_payload("C", "C", Vec3(x_offset + 2.7, 0.0, 0.0)),
            atom_payload("O", "O", Vec3(x_offset + 3.4, 1.0, 0.0)),
        ),
    )


def _append_bond(
    structure: ProteinStructure,
    left_atom_ref: AtomRef,
    right_atom_ref: AtomRef,
    *,
    relationship_type: BondRelationshipType,
    provenance: BondProvenance = BondProvenance.SOURCE_EXPLICIT,
) -> ProteinStructure:
    """Append one canonical topology relationship to a synthetic structure."""

    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                *structure.topology.bonds,
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(left_atom_ref),
                    atom_index_2=structure.constitution.atom_index(right_atom_ref),
                    relationship_type=relationship_type,
                    provenance=provenance,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def _execution_context(structure: ProteinStructure) -> TransformerExecutionContext:
    """Build deterministic workflow dependencies for direct action tests."""

    return TransformerExecutionContext(
        component_library=build_default_component_library(),
        original_structure=structure,
        orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
    )
