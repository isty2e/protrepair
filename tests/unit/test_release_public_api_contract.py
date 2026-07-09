"""Release-facing public import contract tests."""

import importlib
import re
from pathlib import Path

RELEASE_PUBLIC_EXPORTS: dict[str, tuple[str, ...]] = {
    "protrepair": (
        "__version__",
        "AnalysisKind",
        "ProcessResult",
        "ProtrepairError",
        "RepairRefinementSpec",
        "RetainedNonPolymerChemistryOverride",
        "StructureIngressOptions",
        "WorkflowTransformRequests",
        "process_structure",
        "requested_process_goal",
    ),
    "protrepair.scope": (
        "AbsentResidueSpanScope",
        "AtomSetScope",
        "AtomToChainSetScopeCoarsening",
        "AtomToResidueSetScopeCoarsening",
        "AnchorAtomPairScope",
        "ChainSetScope",
        "CompositeScope",
        "ResidueBoundaryScope",
        "ResidueBoundarySide",
        "ResidueSetScope",
        "ResidueToChainSetScopeCoarsening",
        "Scope",
        "ScopeCoarsening",
        "ScopeKind",
        "ScopeLowering",
        "ScopeToWholeStructureCoarsening",
        "WholeStructureScope",
        "coarsen_atom_scope_to_chain_scope",
        "coarsen_atom_scope_to_residue_scope",
        "coarsen_residue_scope_to_chain_scope",
        "coarsen_scope_to_whole_structure",
        "scope_contains",
        "scope_refines",
        "scopes_overlap",
    ),
    "protrepair.structure": (
        "AtomGeometry",
        "AtomRef",
        "AtomSite",
        "AtomTopology",
        "BondProvenance",
        "BondRelationshipType",
        "ChainSite",
        "ElementIdentity",
        "FileFormat",
        "PolymerBlueprint",
        "PolymerChainBlueprint",
        "PolymerResidueSlot",
        "ProteinStructure",
        "ResidueFacetPayload",
        "ResidueGeometry",
        "ResidueId",
        "ResidueSite",
        "SourceBondMetadata",
        "SourceBondRecordType",
        "StructureAddressSpaceKey",
        "StructureCarrierHandle",
        "StructureCarrierKind",
        "StructureConstitution",
        "StructureEndpoint",
        "StructureGeometry",
        "StructureIngress",
        "StructureProvenance",
        "StructureProvenanceOrigin",
        "StructureRealizationSelector",
        "StructureTopology",
        "TopologyBond",
        "address_space_key_without_atoms",
        "atom_count_in_address_space",
    ),
    "protrepair.state": (
        "AtomScopeObservation",
        "AtomScopeStateFacts",
        "BackboneHeavyAtomCompletenessState",
        "BoundaryAuthenticityState",
        "CarrierScopedState",
        "ClashObservationMode",
        "ClashPresenceState",
        "ClashState",
        "ComponentSupportState",
        "ContinuousBondRealizabilityFacts",
        "ContinuousBondRealizabilityObservation",
        "ContinuousBondRealizabilityState",
        "ContinuousRegionReadinessFacts",
        "HydrogenCoverageState",
        "HydrogenApplicabilityState",
        "HydrogenAttachmentResolutionFacts",
        "HydrogenAttachmentResolutionObservation",
        "HydrogenAttachmentResolutionState",
        "OrientationCorrectionEligibilityState",
        "OxtPresenceState",
        "ParserCompatibilityProfile",
        "ParserCompatibilityState",
        "ProteinStructureObservation",
        "ResidueChemistryReadinessFacts",
        "ResidueCoverageFacts",
        "ResidueHydrogenAttachmentResolutionFact",
        "ResidueProjectionStateFacts",
        "ResidueTopologyAvailabilityFact",
        "RetainedNonPolymerChemistryEvidenceSource",
        "RetainedNonPolymerChemistryReadinessFact",
        "ScopedState",
        "SelectedAtomScopeFacts",
        "SidechainHeavyAtomCompletenessState",
        "StereochemistryState",
        "StructureBoundaryStateFacts",
        "StructureChemistryReadinessFacts",
        "StructureCoverageFacts",
        "StructureInteractionFacts",
        "StructureIntrinsicGeometryFacts",
        "StructureParserCompatibilityFacts",
        "StructureProjectionStateFacts",
        "TerminalBoundaryObservation",
        "TopologyAvailabilityAspect",
        "TopologyAvailabilityFacts",
        "TopologyAvailabilityObservation",
        "TopologyAvailabilityState",
        "derive_backbone_heavy_atom_completeness_state",
        "derive_component_support_state",
        "derive_hydrogen_applicability_state",
        "derive_hydrogen_coverage_state",
        "derive_projection_coverage_and_chemistry_readiness_facts",
        "derive_sidechain_heavy_atom_completeness_state",
        "derive_stereochemistry_state",
        "derive_structure_coverage_and_chemistry_readiness_facts",
    ),
    "protrepair.workflow.contracts": (
        "AnalysisBundle",
        "DEFAULT_PRAS_HISTIDINE_PROTONATION_RATIO",
        "DisabledHistidineProtonationRequest",
        "ExternalSpanGapSelectionPolicy",
        "ExternalSpanReconstructionSpec",
        "HistidineProtonationRequest",
        "LigandPolicy",
        "MutationPolicy",
        "OccupancyPolicy",
        "OrphanFragmentPolicy",
        "ProcessResult",
        "PrasRatioHistidineProtonationRequest",
        "RamachandranAnalysis",
        "RamachandranCategory",
        "RamachandranPoint",
        "RequestedGoalCompletionVerdict",
        "RequestedGoalOutcome",
        "RequestedGoalReport",
        "RequestedGoalSet",
        "RequestedGoalStatus",
        "RetainedNonPolymerChemistryOverride",
        "SecondaryStructureAnalysis",
        "SecondaryStructureAssignment",
        "StructureIngressOptions",
        "WorkflowGoal",
        "WorkflowGoalStateValue",
        "WorkflowLigandContextMode",
        "WorkflowPhaseOutcome",
        "WorkflowPhaseReport",
        "WorkflowPhaseStatus",
        "WorkflowPlanningContext",
        "WorkflowPlanningPhase",
        "WorkflowSpanDonorAvailability",
        "WorkflowTargetIntent",
        "WorkflowTransformRequests",
        "WorkflowTerminalBranchOutcome",
        "WorkflowTerminalBranchReport",
        "build_alphafold_span_reconstruction_specs",
        "requested_process_goal",
    ),
    "protrepair.analysis": (
        "AnalysisBundle",
        "AnalysisKind",
        "RamachandranAnalysis",
        "RamachandranCategory",
        "RamachandranPoint",
        "SecondaryStructureAnalysis",
        "SecondaryStructureAssignment",
    ),
    "protrepair.io": (
        "FileFormat",
        "read_structure",
        "read_structure_string",
        "write_structure",
        "write_structure_string",
    ),
    "protrepair.geometry": (
        "AxisRotation",
        "GeometryPlacementError",
        "InternalCoordinateFrame",
        "PlanarCenter",
        "RigidTransform",
        "TetrahedralCenter",
        "Vec3",
    ),
}


def test_release_public_exports_are_exact() -> None:
    """Release-facing facades should not expand without updating this contract."""

    for module_name, public_names in RELEASE_PUBLIC_EXPORTS.items():
        module = importlib.import_module(module_name)
        assert len(public_names) == len(set(public_names)), module_name
        assert len(module.__all__) == len(set(module.__all__)), module_name
        assert set(module.__all__) == set(public_names), module_name
        for public_name in public_names:
            assert hasattr(module, public_name), f"{module_name}.{public_name}"


def test_inactive_workflow_policy_enums_are_not_public_contracts() -> None:
    """Public policy enums must either affect execution or stay unexported."""

    contracts = importlib.import_module("protrepair.workflow.contracts")
    policies = importlib.import_module("protrepair.workflow.contracts.policies")

    for inactive_policy_name in ("HydrogenPolicy", "CTerminalOxtPolicy"):
        assert not hasattr(contracts, inactive_policy_name)
        assert not hasattr(policies, inactive_policy_name)
        assert inactive_policy_name not in contracts.__all__
        assert inactive_policy_name not in policies.__all__


def test_workflow_policy_module_exports_only_runtime_bound_policies() -> None:
    """Policy module exports should be request-bound or execution-bound."""

    policies = importlib.import_module("protrepair.workflow.contracts.policies")

    assert set(policies.__all__) == {
        "LigandPolicy",
        "MutationPolicy",
        "OccupancyPolicy",
        "OrphanFragmentPolicy",
    }
    for policy_name in policies.__all__:
        assert hasattr(policies, policy_name), policy_name


def test_readme_uses_current_state_axis_names() -> None:
    """README should not advertise stale pre-ontology state names."""

    readme = Path("README.md").read_text()

    assert "BackboneHeavyAtomCompletenessState" in readme
    assert "SidechainHeavyAtomCompletenessState" in readme
    assert "HydrogenCoverageState" in readme
    assert (
        re.search(
            r"(?<!Backbone)(?<!Sidechain)\bHeavyAtomCompletenessState\b",
            readme,
        )
        is None
    )
    assert "HydrogenRealizationState" not in readme


def test_readme_uses_stable_subpackage_facade_imports() -> None:
    """README examples should import typed contracts from owner facades."""

    readme = Path("README.md").read_text()

    assert "from protrepair import WorkflowTransformRequests" not in readme
    assert "from protrepair import AnalysisKind" not in readme
    assert "from protrepair.workflow.contracts import (" in readme
    assert "LigandPolicy" in readme
    assert "from protrepair.analysis import AnalysisKind" in readme
