"""Shared toy fixtures for refinement boundary and artifact tests."""

from protrepair.chemistry import (
    BondDefinition,
    ChemicalComponentDefinition,
    ComponentLibrary,
    HydrogenSemantics,
    ResidueTemplate,
)
from protrepair.geometry import Vec3
from protrepair.scope import ResidueSetScope, WholeStructureScope
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    CarrierScopedState,
    ComponentSupportState,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
    StereochemistryState,
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
    StructureProjectionStateFacts,
)
from protrepair.state.domain import (
    AtomScopeStateFacts,
    ClashState,
    ContinuousBondRealizabilityFacts,
    ContinuousBondRealizabilityState,
    ContinuousRegionReadinessFacts,
    HydrogenAttachmentResolutionFacts,
    HydrogenAttachmentResolutionObservation,
    HydrogenAttachmentResolutionState,
    SelectedAtomScopeFacts,
    TopologyAvailabilityFacts,
    TopologyAvailabilityObservation,
    TopologyAvailabilityState,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from tests.support.canonical_builders import (
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)


def build_toy_component_library() -> ComponentLibrary:
    """Return one tiny component library suitable for refinement tests."""

    mov_template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="MOV",
            atom_names=("C1", "H1"),
            bonds=(BondDefinition("C1", "H1"),),
        ),
        hydrogen_semantics=HydrogenSemantics(
            plan_with_backbone=((("H1",), "class2", ("X1", "X2", "C1")),)
        ),
    )
    obs_template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="OBS",
            atom_names=("O1",),
        )
    )
    return ComponentLibrary(
        templates={
            "MOV": mov_template,
            "OBS": obs_template,
        }
    )


def build_toy_structure() -> ProteinStructure:
    """Return one tiny structure with one movable residue and one ligand."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="MOV",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="OBS",
                residue_id=ResidueId(chain_id="L", seq_num=1),
                atoms=(atom_payload("O1", "O", Vec3(2.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )


def ready_atom_scope_facts(
    *,
    hydrogen_coverage_state: HydrogenCoverageState = HydrogenCoverageState.COMPLETE,
    hydrogen_atom_count: int = 1,
    hydrogen_topology_state: TopologyAvailabilityState = (
        TopologyAvailabilityState.PRESENT
    ),
    hydrogen_attachment_resolution_state: HydrogenAttachmentResolutionState
    | None = None,
    continuous_bond_realizability_state: ContinuousBondRealizabilityState | None = None,
    continuous_bond_realizability_blocker: str | None = None,
) -> AtomScopeStateFacts:
    """Return one legal local refinement fact bundle for binding tests."""

    carrier = build_toy_structure()
    selected_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
    structure_facts = StructureProjectionStateFacts(
        chain_count=1,
        residue_count=1,
        ligand_count=1,
        atom_count=5,
        hydrogen_atom_count=hydrogen_atom_count,
        component_support_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=ComponentSupportState.ALL_SUPPORTED,
        ),
        backbone_heavy_atom_completeness_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=BackboneHeavyAtomCompletenessState.COMPLETE,
        ),
        sidechain_heavy_atom_completeness_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=SidechainHeavyAtomCompletenessState.COMPLETE,
        ),
        hydrogen_applicability_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=HydrogenApplicabilityState.APPLICABLE,
        ),
        hydrogen_coverage_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=hydrogen_coverage_state,
        ),
        stereochemistry_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=StereochemistryState.CONSISTENT,
        ),
    )
    resolved_hydrogen_attachment_state = hydrogen_attachment_resolution_state
    if resolved_hydrogen_attachment_state is None:
        if hydrogen_topology_state is TopologyAvailabilityState.PRESENT:
            resolved_hydrogen_attachment_state = (
                HydrogenAttachmentResolutionState.TEMPLATE_RESOLVED
            )
        elif hydrogen_topology_state is TopologyAvailabilityState.NOT_APPLICABLE:
            resolved_hydrogen_attachment_state = (
                HydrogenAttachmentResolutionState.NOT_APPLICABLE
            )
        else:
            resolved_hydrogen_attachment_state = (
                HydrogenAttachmentResolutionState.ABSENT
            )

    resolved_continuous_bond_realizability_state = continuous_bond_realizability_state
    if resolved_continuous_bond_realizability_state is None:
        if hydrogen_topology_state in {
            TopologyAvailabilityState.PRESENT,
            TopologyAvailabilityState.NOT_APPLICABLE,
        }:
            resolved_continuous_bond_realizability_state = (
                ContinuousBondRealizabilityState.REALIZABLE
            )
        else:
            resolved_continuous_bond_realizability_state = (
                ContinuousBondRealizabilityState.UNREALIZABLE
            )

    resolved_continuous_bond_realizability_blocker = (
        continuous_bond_realizability_blocker
    )
    if (
        resolved_continuous_bond_realizability_state
        is ContinuousBondRealizabilityState.REALIZABLE
    ):
        resolved_continuous_bond_realizability_blocker = None
    elif resolved_continuous_bond_realizability_blocker is None:
        resolved_continuous_bond_realizability_blocker = (
            "continuous relaxation requires a realizable selected-scope bond "
            "graph before any force field can be bound"
        )

    return AtomScopeStateFacts(
        selected_scope_facts=SelectedAtomScopeFacts(
            atom_count=5,
            residue_count=1,
            hydrogen_atom_count=hydrogen_atom_count,
            structure_facts=structure_facts,
            clash_fact=CarrierScopedState(
                carrier=carrier,
                scope=selected_scope,
                value=ClashState.NONE,
            ),
            topology_availability_facts=TopologyAvailabilityFacts(
                carrier=carrier,
                scope=selected_scope,
                residue_facts=TopologyAvailabilityObservation.uniform(
                    (ResidueId(chain_id="A", seq_num=1),),
                    heavy_atom_state=TopologyAvailabilityState.PRESENT,
                    hydrogen_state=hydrogen_topology_state,
                ).residue_facts,
            ),
            hydrogen_attachment_resolution_facts=HydrogenAttachmentResolutionFacts(
                carrier=carrier,
                scope=selected_scope,
                residue_facts=HydrogenAttachmentResolutionObservation.uniform(
                    (ResidueId(chain_id="A", seq_num=1),),
                    state=resolved_hydrogen_attachment_state,
                ).residue_facts,
            ),
            continuous_bond_realizability_facts=ContinuousBondRealizabilityFacts(
                carrier=carrier,
                scope=selected_scope,
                state=resolved_continuous_bond_realizability_state,
                blocker=resolved_continuous_bond_realizability_blocker,
            ),
        ),
        continuous_region_readiness_facts=ContinuousRegionReadinessFacts(
            coverage_facts=StructureCoverageFacts(
                carrier=carrier,
                residue_facts=(),
                blueprint_coverages=(),
                coverage_gaps=(),
                backbone_heavy_atom_completeness_state=(
                    structure_facts.backbone_heavy_atom_completeness_fact.value
                ),
                sidechain_heavy_atom_completeness_state=(
                    structure_facts.sidechain_heavy_atom_completeness_fact.value
                ),
            ),
            chemistry_readiness_facts=StructureChemistryReadinessFacts(
                carrier=carrier,
                residue_facts=(),
                retained_non_polymer_facts=(),
                component_support_state=structure_facts.component_support_fact.value,
                heavy_atom_topology_availability_state=(
                    TopologyAvailabilityState.PRESENT
                ),
                hydrogen_topology_availability_state=hydrogen_topology_state,
                hydrogen_applicability_state=(
                    structure_facts.hydrogen_applicability_fact.value
                ),
                hydrogen_coverage_state=structure_facts.hydrogen_coverage_fact.value,
            ),
        ),
    )
