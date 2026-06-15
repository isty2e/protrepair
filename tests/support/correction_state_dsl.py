"""DSL types for correction-state fixture cases."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from protrepair.chemistry import ComponentLibrary
from protrepair.state.domain import (
    BackboneHeavyAtomCompletenessState,
    ClashState,
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
    TopologyAvailabilityAspect,
    TopologyAvailabilityState,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.local import LocalScopeSpec
from protrepair.workflow.planning.transformation.legality import (
    LocalTransformationStratum,
)
from protrepair.workflow.planning.transformation.runtime import (
    LocalTransformationFamily,
    TransformationTerminationReason,
)


class CorrectionCoverageTag(str, Enum):
    """Canonical route-precondition coverage tags for correction-state fixtures."""

    UNSUPPORTED_STOP = "unsupported_stop"
    HEAVY_INCOMPLETENESS = "heavy_incompleteness"
    HYDROGEN_ONLY_WORKFLOW = "hydrogen_only_workflow"
    HETEROGENEOUS_WORKFLOW = "heterogeneous_workflow"
    RELAXATION_READY = "relaxation_ready"
    TOPOLOGY_PREPARATION = "topology_preparation"
    CHEMISTRY_PREPARATION = "chemistry_preparation"
    CANDIDATE_CONSTRUCTION = "candidate_construction"


@dataclass(frozen=True, slots=True)
class TopologyExpectation:
    """One expected residue-local topology-availability fact."""

    residue_id: ResidueId
    aspect: TopologyAvailabilityAspect
    state: TopologyAvailabilityState


@dataclass(frozen=True, slots=True)
class WorkflowExpectation:
    """Expected whole-workflow completion classification for one case."""

    requests_hydrogen_population: bool
    execution_stage_values: tuple[str, ...]
    partition_kind_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalExpectation:
    """Expected local-state and legality classification for one case."""

    scope_spec: LocalScopeSpec
    component_support_state: ComponentSupportState | None = None
    backbone_heavy_atom_completeness_state: (
        BackboneHeavyAtomCompletenessState | None
    ) = None
    sidechain_heavy_atom_completeness_state: (
        SidechainHeavyAtomCompletenessState | None
    ) = None
    hydrogen_applicability_state: HydrogenApplicabilityState | None = None
    hydrogen_coverage_state: HydrogenCoverageState | None = None
    clash_state: ClashState | None = None
    continuous_relaxation_ready: bool | None = None
    topology_expectations: tuple[TopologyExpectation, ...] = ()
    discrete_preparation_applicable: bool = False
    discrete_seeding_applicable: bool = False
    validate_discrete_preparation_detector: bool = False
    validate_discrete_seeding_detector: bool = False
    legal_families: tuple[LocalTransformationFamily, ...] = ()
    legal_strata: tuple[LocalTransformationStratum, ...] = ()
    termination_reason: TransformationTerminationReason | None = None


CorrectionStructureFactory = Callable[[ComponentLibrary], ProteinStructure]


@dataclass(frozen=True, slots=True)
class CorrectionStateCase:
    """One structure-backed correction-state regression fixture."""

    case_id: str
    description: str
    coverage_tags: tuple[CorrectionCoverageTag, ...]
    structure_factory: CorrectionStructureFactory
    workflow: WorkflowExpectation | None = None
    local: LocalExpectation | None = None

    def build_structure(
        self,
        component_library: ComponentLibrary,
    ) -> ProteinStructure:
        """Materialize the canonical structure for one correction-state case."""

        return self.structure_factory(component_library)
