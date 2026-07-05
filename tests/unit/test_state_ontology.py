"""Unit tests for canonical structure and atom-scope observation ontology."""

import ast
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.retained_non_polymer_components import (
    build_retained_non_polymer_component_library,
)

from protrepair.chemistry import (
    BondDefinition,
    ChemicalComponentDefinition,
    ComponentLibrary,
    ResidueTemplate,
    build_default_component_library,
)
from protrepair.errors import RefinementError
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, read_structure
from protrepair.scope import (
    AtomSetScope,
    ResidueBoundaryScope,
    ResidueBoundarySide,
    ResidueSetScope,
    WholeStructureScope,
)
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import (
    AtomScopeObservation,
    AtomScopeStateFacts,
    BackboneHeavyAtomCompletenessState,
    BoundaryAuthenticityState,
    CarrierScopedState,
    ComponentSupportState,
    ContinuousBondRealizabilityFacts,
    ContinuousBondRealizabilityState,
    ContinuousRegionReadinessFacts,
    HydrogenApplicabilityState,
    HydrogenCoverageState,
    OxtPresenceState,
    ProteinStructureObservation,
    ResidueChemistryReadinessFacts,
    ResidueCoverageFacts,
    ResidueProjectionStateFacts,
    ResidueTopologyAvailabilityFact,
    RetainedNonPolymerChemistryEvidenceSource,
    RetainedNonPolymerChemistryReadinessFact,
    SelectedAtomScopeFacts,
    SidechainHeavyAtomCompletenessState,
    StereochemistryState,
    StructureChemistryReadinessFacts,
    StructureCoverageFacts,
    StructureProjectionStateFacts,
    TerminalBoundaryObservation,
    TopologyAvailabilityAspect,
    TopologyAvailabilityFacts,
    TopologyAvailabilityObservation,
    TopologyAvailabilityState,
    derive_hydrogen_coverage_state,
    derive_stereochemistry_state,
    derive_structure_coverage_and_chemistry_readiness_facts,
)
from protrepair.state.domain import (
    ClashState,
    HydrogenAttachmentResolutionFacts,
    HydrogenAttachmentResolutionObservation,
    HydrogenAttachmentResolutionState,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.polymer_blueprint import (
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
)
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import (
    ProteinStructureSnapshot,
)
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.atom_input import AtomInput, AtomInputBasis
from protrepair.transformer.continuous.readiness import (
    atom_scope_facts_continuous_relaxation_error,
    atom_scope_facts_supports_continuous_relaxation,
    derive_atom_scope_continuous_relaxation_facts,
    derive_atom_scope_continuous_relaxation_observation,
    require_atom_scope_continuous_relaxation_execution,
    structure_facts_continuous_relaxation_error,
    structure_facts_supports_continuous_relaxation,
    topology_availability_facts_supports_continuous_relaxation,
)
from protrepair.workflow.contracts import StructureIngressOptions
from protrepair.workflow.planning.transformation.runtime import (
    LocalTransformationFamily,
    SelectedScopePlanningSignature,
    SelectedScopeSignature,
    StructurePlanningSignature,
    TransformationFamilyAttemptSignature,
    TransformationHistory,
    TransformationPlanningMemory,
    TransformationPlanningSignature,
    TransformationProgressKind,
    TransformationProgressMemory,
    TransformationStepBudget,
    TransformationTerminationMemory,
    TransformationTerminationReason,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - optional dependency
    Chem = None

RDKIT_AVAILABLE = Chem is not None


def test_state_package_does_not_import_transformer_implementation() -> None:
    """State facts should not depend on concrete transformer implementations."""

    violations: list[str] = []
    for path in Path("src/protrepair/state").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module
                if module_name is not None and module_name.startswith(
                    "protrepair.transformer"
                ):
                    violations.append(f"{path}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("protrepair.transformer"):
                        violations.append(f"{path}:{node.lineno}")

    assert not violations


def test_protein_structure_state_marks_unsupported_components_indeterminate() -> None:
    """Unsupported residues should block continuous relaxation readiness."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("UNK", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="unsupported-state",
    )

    state = ProteinStructureObservation.from_structure(structure)

    assert (
        state.component_support_state
        is ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
    )
    assert (
        state.backbone_heavy_atom_completeness_state
        is BackboneHeavyAtomCompletenessState.COMPLETE
    )
    assert (
        state.sidechain_heavy_atom_completeness_state
        is SidechainHeavyAtomCompletenessState.COMPLETE
    )
    facts = StructureProjectionStateFacts.from_structure(structure)

    assert not structure_facts_supports_continuous_relaxation(facts)
    assert structure_facts_continuous_relaxation_error(facts) is not None


def test_structure_observation_rebuilds_from_projection_state_facts() -> None:
    """Whole-structure observations should rebuild from explicit state facts."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="fact-backed-structure-observation",
    )

    facts = StructureProjectionStateFacts.from_structure(structure)

    assert ProteinStructureObservation.from_facts(facts) == (
        ProteinStructureObservation.from_structure(structure)
    )


def test_atom_scope_observation_rebuilds_from_explicit_state_facts() -> None:
    """Local observations should rebuild from explicit atom-scope fact bundles."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="fact-backed-atom-scope-observation",
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
        carrier=snapshot,
    )
    facts = derive_atom_scope_continuous_relaxation_facts(snapshot, atom_scope)

    assert AtomScopeObservation.from_facts(facts) == (
        derive_atom_scope_continuous_relaxation_observation(
            snapshot,
            atom_scope,
        )
    )


def test_coarse_hydrogen_coverage_derives_from_singleton_residue_facts() -> None:
    """Coarse hydrogen coverage should derive from singleton residue facts."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        (
                            "N",
                            "CA",
                            "C",
                            "O",
                            "CB",
                            "H1",
                            "H2",
                            "H3",
                            "HA",
                            "HB1",
                            "HB2",
                            "HB3",
                        ),
                    ),
                    build_residue("GLY", "A", 2, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="derived-coarse-hydrogen-coverage",
    )

    residue_facts = tuple(
        ResidueProjectionStateFacts.from_residue(structure, residue)
        for residue in structure.constitution.chains[0].residues
    )

    assert (
        derive_hydrogen_coverage_state(residue_facts) is HydrogenCoverageState.PARTIAL
    )


def test_structure_coverage_facts_keep_gap_distinct_from_atom_completion() -> None:
    """Missing residue spans should stay in coverage gaps, not atom completeness."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),
                    build_residue("GLY", "A", 4, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="coverage-gap-structure",
        polymer_blueprint=PolymerBlueprint(
            chains=(
                PolymerChainBlueprint(
                    chain_id="A",
                    residue_slots=(
                        PolymerResidueSlot(sequence_position=1, token="A"),
                        PolymerResidueSlot(sequence_position=2, token="C"),
                        PolymerResidueSlot(sequence_position=3, token="D"),
                        PolymerResidueSlot(sequence_position=4, token="G"),
                    ),
                ),
            )
        ),
    )

    coverage_facts = StructureCoverageFacts.from_structure(structure)

    assert coverage_facts.backbone_heavy_atom_completeness_state is (
        BackboneHeavyAtomCompletenessState.COMPLETE
    )
    assert coverage_facts.sidechain_heavy_atom_completeness_state is (
        SidechainHeavyAtomCompletenessState.COMPLETE
    )
    assert len(coverage_facts.coverage_gaps) == 1
    assert coverage_facts.coverage_gaps[0].absent_sequence_positions == (2, 3)


def test_structure_chemistry_readiness_facts_surface_topology_readiness() -> None:
    """Chemistry-readiness should own support/topology/hydrogen truth."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("GLY", "A", 2, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="chemistry-readiness-structure",
    )

    chemistry_readiness_facts = StructureChemistryReadinessFacts.from_structure(
        structure
    )

    assert chemistry_readiness_facts.component_support_state is (
        ComponentSupportState.ALL_SUPPORTED
    )
    assert chemistry_readiness_facts.heavy_atom_topology_availability_state is (
        TopologyAvailabilityState.ABSENT
    )
    assert chemistry_readiness_facts.hydrogen_applicability_state is (
        HydrogenApplicabilityState.APPLICABLE
    )
    assert chemistry_readiness_facts.hydrogen_coverage_state is (
        HydrogenCoverageState.NONE
    )
    assert chemistry_readiness_facts.hydrogen_topology_availability_state is (
        TopologyAvailabilityState.ABSENT
    )


def test_template_less_polymer_topology_is_unsupported_not_absent() -> None:
    """Missing component templates should surface unsupported topology explicitly."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (build_residue("UNK", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="template-less-polymer-topology",
    )
    facts = TopologyAvailabilityFacts.from_projection(
        ProteinStructureSnapshot.from_structure(structure),
        residue_ids=(residue_id,),
        component_library=ComponentLibrary(),
    )
    observation = TopologyAvailabilityObservation.from_facts(facts)

    assert observation.state_for(
        residue_id,
        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
    ) is TopologyAvailabilityState.UNSUPPORTED
    assert observation.state_for(
        residue_id,
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
    ) is TopologyAvailabilityState.UNSUPPORTED
    assert not topology_availability_facts_supports_continuous_relaxation(facts)


def test_template_backed_missing_atoms_keep_topology_absent() -> None:
    """Known expected topology with missing atoms is absent, not unsupported."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="template-backed-missing-atoms",
    )
    facts = TopologyAvailabilityFacts.from_projection(
        ProteinStructureSnapshot.from_structure(structure),
        residue_ids=(residue_id,),
        component_library=build_default_component_library(),
    )
    observation = TopologyAvailabilityObservation.from_facts(facts)

    assert observation.state_for(
        residue_id,
        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
    ) is TopologyAvailabilityState.ABSENT
    assert observation.state_for(
        residue_id,
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
    ) is TopologyAvailabilityState.ABSENT


def test_template_less_hetero_topology_projection_stays_absent() -> None:
    """Retained non-polymer fallback policy is outside this polymer topology ticket."""

    ligand_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ligand_id,
                atoms=(atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="template-less-hetero-topology",
    )
    facts = TopologyAvailabilityFacts.from_projection(
        ProteinStructureSnapshot.from_structure(structure),
        residue_ids=(ligand_id,),
        component_library=ComponentLibrary(),
    )
    observation = TopologyAvailabilityObservation.from_facts(facts)

    assert observation.state_for(
        ligand_id,
        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
    ) is TopologyAvailabilityState.ABSENT
    assert observation.state_for(
        ligand_id,
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
    ) is TopologyAvailabilityState.ABSENT


def test_structure_readiness_preserves_unsupported_polymer_topology() -> None:
    """Whole-structure readiness should not collapse unsupported topology to absent."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (build_residue("UNK", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="unsupported-polymer-topology-readiness",
    )

    chemistry_readiness_facts = StructureChemistryReadinessFacts.from_structure(
        structure
    )

    assert chemistry_readiness_facts.component_support_state is (
        ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
    )
    assert chemistry_readiness_facts.heavy_atom_topology_availability_state is (
        TopologyAvailabilityState.UNSUPPORTED
    )
    assert chemistry_readiness_facts.hydrogen_topology_availability_state is (
        TopologyAvailabilityState.NOT_APPLICABLE
    )
    assert chemistry_readiness_facts.residue_facts[0].residue_id == residue_id
    assert (
        chemistry_readiness_facts.residue_facts[
            0
        ].heavy_atom_topology_availability_state
        is TopologyAvailabilityState.UNSUPPORTED
    )
    assert (
        chemistry_readiness_facts.residue_facts[0].hydrogen_topology_availability_state
        is TopologyAvailabilityState.UNSUPPORTED
    )


def test_structure_readiness_unsupported_topology_dominates_absent_aggregate() -> None:
    """Unsupported residue-local topology should not vanish in mixed aggregates."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("UNK", "A", 2, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="mixed-unsupported-absent-topology",
    )

    chemistry_readiness_facts = StructureChemistryReadinessFacts.from_structure(
        structure
    )

    assert chemistry_readiness_facts.heavy_atom_topology_availability_state is (
        TopologyAvailabilityState.UNSUPPORTED
    )


@pytest.mark.parametrize(
    ("context_bond_pairs", "expected_state"),
    [
        ((), TopologyAvailabilityState.ABSENT),
        (
            (("N", "H1"), ("N", "H2"), ("N", "H3")),
            TopologyAvailabilityState.PRESENT,
        ),
    ],
)
def test_structure_chemistry_readiness_requires_polymer_context_hydrogen_bonds(
    context_bond_pairs: tuple[tuple[str, str], ...],
    expected_state: TopologyAvailabilityState,
) -> None:
    """N-terminal hydrogens require sequence-inferred anchor topology."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        (
                            "N",
                            "CA",
                            "C",
                            "O",
                            "CB",
                            "H1",
                            "H2",
                            "H3",
                            "HA",
                            "HB1",
                            "HB2",
                            "HB3",
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="polymer-context-hydrogen-topology-readiness",
    )
    structure = with_topology_bonds(
        structure,
        *residue_bond_specs(
            residue_id,
            (
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("CA", "CB"),
                ("CA", "HA"),
                ("CB", "HB1"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                *context_bond_pairs,
            ),
        ),
    )

    chemistry_readiness_facts = StructureChemistryReadinessFacts.from_structure(
        structure
    )

    residue_fact = chemistry_readiness_facts.residue_facts[0]
    assert residue_fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert residue_fact.hydrogen_topology_availability_state is expected_state


def test_split_structure_fact_derivation_reuses_orthogonal_fact_owners() -> None:
    """Shared derivation should return explicit coverage and chemistry owners."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="split-structure-facts",
    )

    coverage_facts, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(structure)
    )

    assert isinstance(coverage_facts, StructureCoverageFacts)
    assert isinstance(chemistry_readiness_facts, StructureChemistryReadinessFacts)
    assert isinstance(coverage_facts.residue_facts[0], ResidueCoverageFacts)
    assert isinstance(
        chemistry_readiness_facts.residue_facts[0],
        ResidueChemistryReadinessFacts,
    )
    assert (
        StructureProjectionStateFacts.from_structure(
            structure
        ).hydrogen_coverage_fact.value
        is chemistry_readiness_facts.hydrogen_coverage_state
    )


def test_structure_chemistry_readiness_tracks_supported_retained_non_polymer(
) -> None:
    """Chemistry readiness should expose retained non-polymer support separately."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(3.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(4.2, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(3.0, 1.2, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-supported",
    )

    _, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=build_retained_non_polymer_component_library(),
        )
    )

    assert len(chemistry_readiness_facts.retained_non_polymer_facts) == 1
    retained_fact = chemistry_readiness_facts.retained_non_polymer_facts[0]

    assert isinstance(retained_fact, RetainedNonPolymerChemistryReadinessFact)
    assert retained_fact.component_id == "LIG"
    assert retained_fact.heavy_topology_source is (
        RetainedNonPolymerChemistryEvidenceSource.TEMPLATE
    )
    assert retained_fact.hydrogen_expectation_source is (
        RetainedNonPolymerChemistryEvidenceSource.TEMPLATE
    )
    assert retained_fact.is_supported()
    assert retained_fact.requires_hydrogen_completion()
    assert retained_fact.needs_hydrogenation()
    assert retained_fact.hydrogen_applicability_state is (
        HydrogenApplicabilityState.APPLICABLE
    )
    assert retained_fact.hydrogen_coverage_state is HydrogenCoverageState.NONE


def test_structure_chemistry_readiness_distinguishes_retained_non_polymer_modes(
) -> None:
    """Retained non-polymer facts should separate unsupported from not-applicable."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="ION",
                residue_id=ResidueId("L", 1),
                atoms=(atom_payload("ZN", "ZN", Vec3(3.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 2),
                atoms=(atom_payload("C1", "C", Vec3(6.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-mixed",
    )

    _, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=build_retained_non_polymer_component_library(),
        )
    )

    retained_facts_by_component = {
        fact.component_id: fact
        for fact in chemistry_readiness_facts.retained_non_polymer_facts
    }

    assert retained_facts_by_component["ION"].heavy_topology_source is (
        RetainedNonPolymerChemistryEvidenceSource.TEMPLATE
    )
    if RDKIT_AVAILABLE:
        assert retained_facts_by_component["ION"].hydrogen_expectation_source is (
            RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
        )
        assert retained_facts_by_component["ION"].is_supported()
    else:
        assert retained_facts_by_component["ION"].hydrogen_expectation_source is (
            RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
        )
        assert not retained_facts_by_component["ION"].is_supported()
    assert retained_facts_by_component["ION"].hydrogen_applicability_state is (
        HydrogenApplicabilityState.NOT_APPLICABLE
    )
    assert not retained_facts_by_component["ION"].requires_hydrogen_completion()
    if RDKIT_AVAILABLE:
        assert retained_facts_by_component["UNK"].heavy_topology_source is (
            RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
        )
        assert retained_facts_by_component["UNK"].hydrogen_expectation_source is (
            RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
        )
        assert retained_facts_by_component["UNK"].is_supported()
        assert retained_facts_by_component["UNK"].requires_hydrogen_completion()
        assert retained_facts_by_component["UNK"].hydrogen_applicability_state is (
            HydrogenApplicabilityState.APPLICABLE
        )
        assert retained_facts_by_component["UNK"].hydrogen_coverage_state is (
            HydrogenCoverageState.NONE
        )
    else:
        assert retained_facts_by_component["UNK"].heavy_topology_source is (
            RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
        )
        assert retained_facts_by_component["UNK"].hydrogen_expectation_source is (
            RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
        )
        assert not retained_facts_by_component["UNK"].is_supported()


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires RDKit fallback chemistry")
def test_structure_chemistry_readiness_splits_template_heavy_from_fallback_hydrogen(
) -> None:
    """Template heavy topology and fallback hydrogen expectation should split."""

    residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="MIX",
                residue_id=residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-mixed-source",
    )
    structure = with_topology_bonds(
        structure,
        topology_bond_spec(residue_id, "C1", "O1"),
    )
    component_library = ComponentLibrary(
        templates={
            "MIX": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="MIX",
                    atom_names=("C1", "O1"),
                    bonds=(BondDefinition("C1", "O1"),),
                ),
            ),
        }
    )

    _, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=component_library,
        )
    )

    retained_fact = chemistry_readiness_facts.retained_non_polymer_facts[0]
    assert retained_fact.heavy_topology_source is (
        RetainedNonPolymerChemistryEvidenceSource.TEMPLATE
    )
    assert retained_fact.hydrogen_expectation_source is (
        RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    )
    assert retained_fact.is_supported()
    assert retained_fact.requires_hydrogen_completion()
    assert retained_fact.heavy_atom_topology_availability_state is (
        TopologyAvailabilityState.PRESENT
    )
    assert retained_fact.hydrogen_applicability_state is (
        HydrogenApplicabilityState.APPLICABLE
    )
    assert retained_fact.hydrogen_coverage_state is HydrogenCoverageState.NONE


def test_structure_chemistry_readiness_supports_override_backed_retained_non_polymer(
) -> None:
    """Override-backed retained non-polymers should become hydrogenatable."""

    ligand_residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ligand_residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(3.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(4.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-override-supported",
    )
    structure = with_topology_bonds(
        structure,
        topology_bond_spec(
            ligand_residue_id,
            "C1",
            "O1",
            provenance=BondProvenance.EVIDENCE_RESOLVED,
        ),
    )

    _, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=build_retained_non_polymer_component_library(),
            retained_non_polymer_chemistry_evidence=(
                RetainedNonPolymerChemistryOverride(
                    residue_id=ResidueId("L", 1),
                    smiles="CO",
                    heavy_atom_names=("C1", "O1"),
                ).to_evidence(),
            ),
        )
    )

    retained_fact = chemistry_readiness_facts.retained_non_polymer_facts[0]
    assert retained_fact.component_id == "UNK"
    assert retained_fact.component_support_state is (
        ComponentSupportState.ALL_SUPPORTED
    )
    assert retained_fact.heavy_topology_source is (
        RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
    )
    assert retained_fact.hydrogen_expectation_source is (
        RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
    )
    assert retained_fact.is_supported()
    assert retained_fact.requires_hydrogen_completion()
    assert retained_fact.needs_hydrogenation()
    assert retained_fact.heavy_atom_topology_availability_state is (
        TopologyAvailabilityState.PRESENT
    )
    assert retained_fact.hydrogen_applicability_state is (
        HydrogenApplicabilityState.APPLICABLE
    )
    assert retained_fact.hydrogen_coverage_state is HydrogenCoverageState.NONE


def test_structure_chemistry_readiness_rejects_override_atoms_without_bonds() -> None:
    """Override-backed retained heavy atoms still require canonical topology."""

    ligand_residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ligand_residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.4, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="override-atoms-without-bonds",
    )

    _, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=build_retained_non_polymer_component_library(),
            retained_non_polymer_chemistry_evidence=(
                RetainedNonPolymerChemistryOverride(
                    residue_id=ligand_residue_id,
                    smiles="CO",
                    heavy_atom_names=("C1", "O1"),
                ).to_evidence(),
            ),
        )
    )

    retained_fact = chemistry_readiness_facts.retained_non_polymer_facts[0]
    assert retained_fact.heavy_topology_source is (
        RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
    )
    assert retained_fact.heavy_atom_topology_availability_state is (
        TopologyAvailabilityState.ABSENT
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_structure_chemistry_readiness_respects_complete_retained_non_polymer_hydrogens(
) -> None:
    """Complete retained non-polymer hydrogens should suppress fallback completion."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(3.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(4.4, 0.0, 0.0)),
                    atom_payload("H_A", "H", Vec3(2.6, 0.8, 0.0)),
                    atom_payload("H_B", "H", Vec3(2.6, -0.8, 0.0)),
                    atom_payload("H_C", "H", Vec3(3.0, 0.0, 1.0)),
                    atom_payload("H_D", "H", Vec3(5.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-fallback-complete-hydrogens",
    )

    _, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=build_retained_non_polymer_component_library(),
        )
    )

    retained_fact = chemistry_readiness_facts.retained_non_polymer_facts[0]
    assert retained_fact.heavy_topology_source is (
        RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    )
    assert retained_fact.hydrogen_expectation_source is (
        RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    )
    assert retained_fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert not retained_fact.requires_hydrogen_completion()
    assert retained_fact.hydrogen_topology_availability_state is (
        TopologyAvailabilityState.ABSENT
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_structure_chemistry_readiness_accepts_complete_retained_hydrogen_bonds(
) -> None:
    """Complete retained H coverage is topology-ready only with H-heavy bonds."""

    ligand_residue_id = ResidueId("L", 1)
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ligand_residue_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(3.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(4.4, 0.0, 0.0)),
                    atom_payload("H1", "H", Vec3(2.6, 0.8, 0.0)),
                    atom_payload("H2", "H", Vec3(2.6, -0.8, 0.0)),
                    atom_payload("H3", "H", Vec3(3.0, 0.0, 1.0)),
                    atom_payload("H4", "H", Vec3(5.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="retained-non-polymer-fallback-complete-topology",
    )
    structure = with_topology_bonds(
        structure,
        *residue_bond_specs(
            ligand_residue_id,
            (
                ("C1", "O1"),
                ("C1", "H1"),
                ("C1", "H2"),
                ("C1", "H3"),
                ("O1", "H4"),
            ),
            provenance=BondProvenance.REPAIR_INFERRED,
        ),
    )

    _, chemistry_readiness_facts = (
        derive_structure_coverage_and_chemistry_readiness_facts(
            structure,
            component_library=build_retained_non_polymer_component_library(),
        )
    )

    retained_fact = chemistry_readiness_facts.retained_non_polymer_facts[0]
    assert retained_fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert retained_fact.heavy_atom_topology_availability_state is (
        TopologyAvailabilityState.PRESENT
    )
    assert retained_fact.hydrogen_topology_availability_state is (
        TopologyAvailabilityState.PRESENT
    )


def test_coarse_stereochemistry_derives_from_singleton_residue_facts() -> None:
    """Coarse stereochemistry should derive from singleton residue facts."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "THR",
                        "A",
                        1,
                        ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
                    ),
                    build_residue("GLY", "A", 2, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="derived-coarse-stereochemistry",
    )

    residue_facts = tuple(
        ResidueProjectionStateFacts.from_residue(structure, residue)
        for residue in structure.constitution.chains[0].residues
    )

    assert StructureProjectionStateFacts.from_structure(
        structure
    ).stereochemistry_fact.value == derive_stereochemistry_state(residue_facts)


def test_protein_structure_state_marks_backbone_incompleteness() -> None:
    """Backbone gaps should be distinguished from supported side-chain gaps."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="backbone-gap-state",
    )

    state = ProteinStructureObservation.from_structure(structure)

    assert state.component_support_state.is_fully_supported()
    assert (
        state.backbone_heavy_atom_completeness_state
        is BackboneHeavyAtomCompletenessState.INCOMPLETE
    )
    assert (
        state.sidechain_heavy_atom_completeness_state
        is SidechainHeavyAtomCompletenessState.INCOMPLETE
    )
    facts = StructureProjectionStateFacts.from_structure(structure)

    assert not structure_facts_supports_continuous_relaxation(facts)
    assert structure_facts_continuous_relaxation_error(facts) is not None


def test_protein_structure_state_tracks_residue_level_hydrogen_realization() -> None:
    """Hydrogen realization should distinguish none, partial, and populated."""

    no_hydrogen_structure = build_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="hydrogen-none",
    )
    no_hydrogen_state = ProteinStructureObservation.from_structure(
        no_hydrogen_structure
    )
    partial_structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        (
                            "N",
                            "CA",
                            "C",
                            "O",
                            "CB",
                            "H",
                            "HA",
                            "HB1",
                            "HB2",
                            "HB3",
                        ),
                    ),
                    build_residue("GLY", "A", 2, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="hydrogen-partial",
    )
    partial_state = ProteinStructureObservation.from_structure(partial_structure)
    populated_structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        (
                            "N",
                            "CA",
                            "C",
                            "O",
                            "CB",
                            "H1",
                            "H2",
                            "H3",
                            "HA",
                            "HB1",
                            "HB2",
                            "HB3",
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="hydrogen-populated",
    )
    populated_state = ProteinStructureObservation.from_structure(populated_structure)

    assert (
        no_hydrogen_state.hydrogen_applicability_state
        is HydrogenApplicabilityState.APPLICABLE
    )
    assert no_hydrogen_state.hydrogen_coverage_state is HydrogenCoverageState.NONE
    assert partial_state.hydrogen_coverage_state is HydrogenCoverageState.PARTIAL
    assert populated_state.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert not structure_facts_supports_continuous_relaxation(
        StructureProjectionStateFacts.from_structure(no_hydrogen_structure)
    )
    assert not structure_facts_supports_continuous_relaxation(
        StructureProjectionStateFacts.from_structure(partial_structure)
    )
    assert structure_facts_supports_continuous_relaxation(
        StructureProjectionStateFacts.from_structure(populated_structure)
    )


def test_protein_structure_state_detects_stereochemistry_violation() -> None:
    """Structure state should surface supported stereochemistry violations."""

    structure = focused_structure_for_residue(seq_num=30)
    inverted_residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(ResidueId("A", 30)),
    )
    inverted_structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    invert_threonine_residue(
                        (
                            structure.constitution.chain("A").residues[0],
                            inverted_residue_geometry,
                            structure.topology.residue_formal_charge_by_atom_name(
                                constitution=structure.constitution,
                                residue_index=structure.constitution.residue_index(
                                    ResidueId("A", 30)
                                ),
                            ),
                        )
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=structure.provenance.ingress.source_format,
        source_name="stereochemistry-violation",
    )

    assert (
        ProteinStructureObservation.from_structure(structure).stereochemistry_state
        is StereochemistryState.CONSISTENT
    )
    assert (
        ProteinStructureObservation.from_structure(
            inverted_structure
        ).stereochemistry_state
        is StereochemistryState.VIOLATED
    )


def test_projection_stereochemistry_ignores_remote_violation() -> None:
    """Projection stereochemistry should not inherit remote residue violations."""

    focused_structure = focused_structure_for_residue(seq_num=30)
    focused_residue = focused_structure.constitution.chain("A").residues[0]
    residue_index = focused_structure.constitution.residue_index(
        focused_residue.residue_id
    )
    residue_geometry = focused_structure.geometry.residue_geometry(
        constitution=focused_structure.constitution,
        residue_index=residue_index,
    )
    formal_charge_by_atom_name = (
        focused_structure.topology.residue_formal_charge_by_atom_name(
            constitution=focused_structure.constitution,
            residue_index=residue_index,
        )
    )
    remote_residue = type(focused_residue)(
        component_id=focused_residue.component_id,
        residue_id=ResidueId("A", 31),
        atom_sites=focused_residue.atom_sites,
        is_hetero=focused_residue.is_hetero,
    )
    mixed_structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    (
                        focused_residue,
                        residue_geometry,
                        formal_charge_by_atom_name,
                    ),
                    invert_threonine_residue(
                        (
                            remote_residue,
                            residue_geometry,
                            formal_charge_by_atom_name,
                        )
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=focused_structure.provenance.ingress.source_format,
        source_name="projection-local-stereochemistry",
    )

    assert (
        ProteinStructureObservation.from_structure(mixed_structure).stereochemistry_state
        is StereochemistryState.VIOLATED
    )
    assert (
        ProteinStructureObservation.from_projection(
            context_structure=mixed_structure,
            residues=(mixed_structure.constitution.chain("A").residues[0],),
        ).stereochemistry_state
        is StereochemistryState.CONSISTENT
    )
    assert (
        StructureProjectionStateFacts.from_projection(
            context_structure=mixed_structure,
            residues=(mixed_structure.constitution.chain("A").residues[0],),
        ).stereochemistry_fact.value
        is StereochemistryState.CONSISTENT
    )


def test_protein_structure_state_marks_authentic_terminal_boundaries() -> None:
    """Whole-structure state should mark source-chain termini as authentic."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("ALA", "A", 2, ("N", "CA", "C", "O", "CB")),
                    build_residue("LYS", "A", 3, ("N", "CA", "C", "O", "CB")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="authentic-termini",
    )

    state = ProteinStructureObservation.from_structure(structure)

    assert state.terminal_boundary_observation.authenticity_facts_for_side(
        ResidueBoundarySide.N_TERMINUS
    ) == (
        build_boundary_authenticity_fact(
            structure,
            "A",
            1,
            side=ResidueBoundarySide.N_TERMINUS,
            authenticity=BoundaryAuthenticityState.AUTHENTIC_IN_CONTEXT,
        ),
    )
    assert state.terminal_boundary_observation.authenticity_facts_for_side(
        ResidueBoundarySide.C_TERMINUS
    ) == (
        build_boundary_authenticity_fact(
            structure,
            "A",
            3,
            side=ResidueBoundarySide.C_TERMINUS,
            authenticity=BoundaryAuthenticityState.AUTHENTIC_IN_CONTEXT,
        ),
    )
    assert state.terminal_boundary_observation.oxt_presence_facts == (
        build_oxt_presence_fact(
            structure,
            "A",
            3,
            presence=OxtPresenceState.ABSENT,
        ),
    )


def test_protein_structure_state_marks_projected_fragment_boundaries() -> None:
    """One internal projection should surface fragment-only terminal boundaries."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("ALA", "A", 2, ("N", "CA", "C", "O", "CB")),
                    build_residue("LYS", "A", 3, ("N", "CA", "C", "O", "CB")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="projected-fragment",
    )

    state = ProteinStructureObservation.from_projection(
        context_structure=structure,
        residues=(structure.constitution.chain("A").residues[1],),
    )

    assert state.terminal_boundary_observation.authenticity_facts == (
        build_boundary_authenticity_fact(
            structure,
            "A",
            2,
            side=ResidueBoundarySide.N_TERMINUS,
            authenticity=BoundaryAuthenticityState.PROJECTED_FRAGMENT_BOUNDARY,
        ),
        build_boundary_authenticity_fact(
            structure,
            "A",
            2,
            side=ResidueBoundarySide.C_TERMINUS,
            authenticity=BoundaryAuthenticityState.PROJECTED_FRAGMENT_BOUNDARY,
        ),
    )
    assert state.terminal_boundary_observation.oxt_presence_facts == (
        build_oxt_presence_fact(
            structure,
            "A",
            2,
            presence=OxtPresenceState.ABSENT,
        ),
    )


def test_protein_structure_state_marks_mixed_terminal_boundary_authenticity() -> None:
    """One suffix projection should keep projected and authentic boundaries apart."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("ALA", "A", 2, ("N", "CA", "C", "O", "CB")),
                    build_residue("LYS", "A", 3, ("N", "CA", "C", "O", "CB")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="mixed-termini",
    )

    state = ProteinStructureObservation.from_projection(
        context_structure=structure,
        residues=tuple(
            residue for residue in structure.constitution.chain("A").residues[1:]
        ),
    )

    assert state.terminal_boundary_observation.authenticity_facts == (
        build_boundary_authenticity_fact(
            structure,
            "A",
            2,
            side=ResidueBoundarySide.N_TERMINUS,
            authenticity=BoundaryAuthenticityState.PROJECTED_FRAGMENT_BOUNDARY,
        ),
        build_boundary_authenticity_fact(
            structure,
            "A",
            3,
            side=ResidueBoundarySide.C_TERMINUS,
            authenticity=BoundaryAuthenticityState.AUTHENTIC_IN_CONTEXT,
        ),
    )
    assert state.terminal_boundary_observation.oxt_presence_facts == (
        build_oxt_presence_fact(
            structure,
            "A",
            3,
            presence=OxtPresenceState.ABSENT,
        ),
    )


def test_atom_domain_state_detects_clash_presence() -> None:
    """Atom-domain state should surface local clash presence."""

    clashing_structure = build_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),),
            ),
            build_chain(
                "B",
                (build_residue("ALA", "B", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="heavy-clash-state",
    )

    snapshot = ProteinStructureSnapshot.from_structure(clashing_structure)
    selected_atom_refs = tuple(
        build_atom_ref(chain_id, 1, atom_name)
        for chain_id in ("A", "B")
        for atom_name in ("N", "CA", "C", "O")
    )
    atom_input = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(atom_ref)
            for atom_ref in selected_atom_refs
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(
            residue_ids=(ResidueId("A", 1), ResidueId("B", 1))
        ),
    )

    state = derive_atom_scope_continuous_relaxation_observation(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )

    assert state.clash_state is ClashState.PRESENT


def test_atom_domain_state_projects_terminal_boundary_truth() -> None:
    """Local domain state should preserve projection terminal-boundary truth."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("ALA", "A", 2, ("N", "CA", "C", "O", "CB")),
                    build_residue("LYS", "A", 3, ("N", "CA", "C", "O", "CB")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="domain-terminal-boundaries",
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    selected_atom_refs = tuple(
        build_atom_ref("A", seq_num, atom_name)
        for seq_num in (2, 3)
        for atom_name in ("N", "CA", "C", "O")
    )
    atom_input = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(atom_ref)
            for atom_ref in selected_atom_refs
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(
            residue_ids=(ResidueId("A", 2), ResidueId("A", 3))
        ),
    )

    state = derive_atom_scope_continuous_relaxation_observation(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )

    assert state.terminal_boundary_observation.authenticity_facts == (
        build_boundary_authenticity_fact(
            structure,
            "A",
            2,
            side=ResidueBoundarySide.N_TERMINUS,
            authenticity=BoundaryAuthenticityState.PROJECTED_FRAGMENT_BOUNDARY,
        ),
        build_boundary_authenticity_fact(
            structure,
            "A",
            3,
            side=ResidueBoundarySide.C_TERMINUS,
            authenticity=BoundaryAuthenticityState.AUTHENTIC_IN_CONTEXT,
        ),
    )
    assert state.terminal_boundary_observation.oxt_presence_facts == (
        build_oxt_presence_fact(
            structure,
            "A",
            3,
            presence=OxtPresenceState.ABSENT,
        ),
    )


def test_atom_domain_state_projects_local_structure_state_axes() -> None:
    """Atom-domain state should reuse canonical local structure-state axes."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue("ALA", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("UNK", "A", 2, ("N", "CA", "C", "O")),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="atom-domain-state",
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    selected_atom_refs = tuple(
        build_atom_ref("A", 2, atom_name) for atom_name in ("N", "CA", "C", "O")
    )
    atom_input = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(atom_ref)
            for atom_ref in selected_atom_refs
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(residue_ids=(ResidueId("A", 2),)),
    )
    domain_state = derive_atom_scope_continuous_relaxation_observation(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )
    domain_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )

    assert (
        domain_state.component_support_state
        is ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
    )
    assert not atom_scope_facts_supports_continuous_relaxation(domain_facts)
    assert domain_state.clash_state is ClashState.NONE
    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
        state=TopologyAvailabilityState.UNSUPPORTED,
    )
    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
        state=TopologyAvailabilityState.UNSUPPORTED,
    )


def test_atom_domain_state_marks_topology_unresolved_without_hydrogens() -> None:
    """Continuous topology should remain unresolved while hydrogens are missing."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="topology-unresolved",
    )
    structure = with_topology_bonds(
        structure,
        *residue_bond_specs(
            residue_id,
            (("N", "CA"), ("CA", "C"), ("C", "O")),
        ),
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    selected_atom_refs = tuple(
        build_atom_ref("A", 1, atom_name) for atom_name in ("N", "CA", "C", "O")
    )
    atom_input = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(atom_ref)
            for atom_ref in selected_atom_refs
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
    )

    domain_state = derive_atom_scope_continuous_relaxation_observation(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )
    domain_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )

    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
        state=TopologyAvailabilityState.PRESENT,
    )
    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
        state=TopologyAvailabilityState.ABSENT,
    )
    assert not atom_scope_facts_supports_continuous_relaxation(domain_facts)


def test_atom_domain_state_marks_missing_heavy_bond_topology_absent() -> None:
    """Present heavy coordinates do not imply heavy connectivity readiness."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("ALA", "A", 1, ("N", "CA", "C", "O", "CB")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="topology-heavy-bond-missing",
    )
    structure = with_topology_bonds(
        structure,
        *residue_bond_specs(
            residue_id,
            (("N", "CA"), ("CA", "C"), ("C", "O")),
        ),
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_input = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(build_atom_ref("A", 1, name))
            for name in ("N", "CA", "C", "O", "CB")
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(residue_ids=(residue_id,)),
    )

    domain_state = derive_atom_scope_continuous_relaxation_observation(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )

    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
        state=TopologyAvailabilityState.ABSENT,
    )


def test_atom_domain_state_marks_missing_hydrogen_anchor_topology_absent() -> None:
    """Present hydrogen coordinates do not imply H-anchor readiness."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        ("N", "CA", "C", "O", "CB", "HA", "HB1", "HB2", "HB3"),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="topology-hydrogen-anchor-missing",
    )
    structure = with_topology_bonds(
        structure,
        *residue_bond_specs(
            residue_id,
            (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")),
        ),
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_input = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(build_atom_ref("A", 1, name))
            for name in ("N", "CA", "C", "O", "CB", "HA", "HB1", "HB2", "HB3")
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(residue_ids=(residue_id,)),
    )

    domain_state = derive_atom_scope_continuous_relaxation_observation(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )

    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
        state=TopologyAvailabilityState.PRESENT,
    )
    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
        state=TopologyAvailabilityState.ABSENT,
    )


def test_atom_domain_state_marks_hydrogen_topology_available_without_clashes() -> None:
    """Hydrogen-complete residues should surface available hydrogen topology."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        (
                            "N",
                            "CA",
                            "C",
                            "O",
                            "CB",
                            "H1",
                            "H2",
                            "H3",
                            "HA",
                            "HB1",
                            "HB2",
                            "HB3",
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="topology-template-resolved",
    )
    structure = with_topology_bonds(
        structure,
        *residue_bond_specs(
            residue_id,
            (
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("CA", "CB"),
                ("CA", "HA"),
                ("CB", "HB1"),
                ("CB", "HB2"),
                ("CB", "HB3"),
            ),
        ),
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    selected_atom_refs = tuple(
        build_atom_ref("A", 1, atom_name)
        for atom_name in (
            "N",
            "CA",
            "C",
            "O",
            "H1",
            "H2",
            "H3",
            "HA",
            "HB1",
            "HB2",
            "HB3",
        )
    )
    atom_input = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(atom_ref)
            for atom_ref in selected_atom_refs
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
    )

    domain_state = derive_atom_scope_continuous_relaxation_observation(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )
    domain_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )

    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
        state=TopologyAvailabilityState.PRESENT,
    )
    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
        state=TopologyAvailabilityState.PRESENT,
    )
    assert atom_scope_facts_supports_continuous_relaxation(domain_facts)


def test_atom_domain_state_surfaces_template_resolved_hydrogen_attachments() -> None:
    """Template-resolved hydrogens should stay distinct from topology truth."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        (
                            "N",
                            "CA",
                            "C",
                            "O",
                            "CB",
                            "HA",
                            "HB1",
                            "HB2",
                            "HB3",
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="topology-template-resolved-detail",
    )
    structure = with_topology_bonds(
        structure,
        *residue_bond_specs(
            residue_id,
            (
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("CA", "CB"),
                ("CA", "HA"),
                ("CB", "HB1"),
                ("CB", "HB2"),
                ("CB", "HB3"),
            ),
        ),
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    selected_atom_refs = tuple(
        build_atom_ref("A", 1, atom_name)
        for atom_name in (
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "HA",
            "HB1",
            "HB2",
            "HB3",
        )
    )
    atom_input = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(atom_ref)
            for atom_ref in selected_atom_refs
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
    )

    domain_state = derive_atom_scope_continuous_relaxation_observation(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )

    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
        state=TopologyAvailabilityState.PRESENT,
    )
    assert domain_state.hydrogen_attachment_resolution.is_uniform(
        state=HydrogenAttachmentResolutionState.TEMPLATE_RESOLVED,
    )
    assert not domain_state.hydrogen_attachment_resolution.any_coordinate_inferred()


def test_atom_domain_state_surfaces_coordinate_inferred_hydrogen_attachments() -> None:
    """Coordinate-inferred hydrogens should not be collapsed into topology absence."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "SER",
                        "A",
                        1,
                        (
                            "N",
                            "CA",
                            "C",
                            "O",
                            "CB",
                            "OG",
                            "H1",
                            "H2",
                            "H3",
                            "HA",
                            "HB1",
                            "HB2",
                            "HG",
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="topology-coordinate-inferred-detail",
    )
    structure = with_topology_bonds(
        structure,
        *residue_bond_specs(
            residue_id,
            (
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("CA", "CB"),
                ("CB", "OG"),
                ("CA", "HA"),
                ("CB", "HB1"),
                ("CB", "HB2"),
                ("OG", "HG"),
            ),
        ),
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    selected_atom_refs = tuple(
        build_atom_ref("A", 1, atom_name)
        for atom_name in (
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "OG",
            "H1",
            "H2",
            "H3",
            "HA",
            "HB1",
            "HB2",
            "HG",
        )
    )
    atom_input = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(atom_ref)
            for atom_ref in selected_atom_refs
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
    )

    domain_state = derive_atom_scope_continuous_relaxation_observation(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )
    domain_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )

    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
        state=TopologyAvailabilityState.PRESENT,
    )
    assert domain_state.hydrogen_attachment_resolution.is_uniform(
        state=HydrogenAttachmentResolutionState.COORDINATE_INFERRED,
    )
    assert domain_state.hydrogen_attachment_resolution.any_coordinate_inferred()
    assert atom_scope_facts_supports_continuous_relaxation(domain_facts)
    assert (
        domain_state.continuous_bond_realizability.state
        is ContinuousBondRealizabilityState.REALIZABLE
    )
    assert domain_facts.continuous_bond_realizability_facts.is_realizable()


def test_atom_domain_state_preserves_hydrogen_topology_on_clashes() -> None:
    """Clashy coordinates should not demote graph-realizable hydrogen topology."""

    structure = build_structure(
        chains=(
            build_chain(
                "A",
                (
                    build_residue(
                        "SER",
                        "A",
                        1,
                        (
                            "N",
                            "CA",
                            "C",
                            "O",
                            "CB",
                            "OG",
                            "H1",
                            "H2",
                            "H3",
                            "HA",
                            "HB1",
                            "HB2",
                            "HG",
                        ),
                    ),
                ),
            ),
            build_chain(
                "B",
                (
                    build_residue(
                        "SER",
                        "B",
                        1,
                        (
                            "N",
                            "CA",
                            "C",
                            "O",
                            "CB",
                            "OG",
                            "H1",
                            "H2",
                            "H3",
                            "HA",
                            "HB1",
                            "HB2",
                            "HG",
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="topology-coordinate-blocked",
    )
    structure = with_topology_bonds(
        structure,
        *residue_bond_specs(
            ResidueId("A", 1),
            (
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("CA", "CB"),
                ("CB", "OG"),
                ("CA", "HA"),
                ("CB", "HB1"),
                ("CB", "HB2"),
                ("OG", "HG"),
            ),
        ),
        *residue_bond_specs(
            ResidueId("B", 1),
            (
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("CA", "CB"),
                ("CB", "OG"),
                ("CA", "HA"),
                ("CB", "HB1"),
                ("CB", "HB2"),
                ("OG", "HG"),
            ),
        ),
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    selected_atom_refs = tuple(
        build_atom_ref(chain_id, 1, atom_name)
        for chain_id in ("A", "B")
        for atom_name in (
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "OG",
            "H1",
            "H2",
            "H3",
            "HA",
            "HB1",
            "HB2",
            "HG",
        )
    )
    atom_input = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(atom_ref)
            for atom_ref in selected_atom_refs
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(
            residue_ids=(ResidueId("A", 1), ResidueId("B", 1))
        ),
    )

    domain_state = derive_atom_scope_continuous_relaxation_observation(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )
    domain_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        OBSERVED_ATOM_SCOPE_LOWERING.lower(atom_input.as_scope(), carrier=snapshot),
    )

    assert domain_state.clash_state is ClashState.PRESENT
    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
        state=TopologyAvailabilityState.PRESENT,
    )
    assert domain_state.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
        state=TopologyAvailabilityState.PRESENT,
    )
    assert atom_scope_facts_supports_continuous_relaxation(domain_facts)
    assert (
        domain_state.continuous_bond_realizability.state
        is ContinuousBondRealizabilityState.REALIZABLE
    )
    assert domain_facts.continuous_bond_realizability_facts.is_realizable()


def test_atom_scope_relaxation_blocks_included_retained_non_polymer_without_hydrogens(
) -> None:
    """Included retained non-polymers must be hydrogen-complete before FF binding."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="SER",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.4, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.4, 1.4, 0.0)),
                            atom_payload("OG", "O", Vec3(1.4, 2.6, 0.0)),
                            atom_payload("H1", "H", Vec3(-0.7, 0.0, 0.0)),
                            atom_payload("H2", "H", Vec3(0.0, 0.7, 0.0)),
                            atom_payload("H3", "H", Vec3(0.0, -0.7, 0.0)),
                            atom_payload("HA", "H", Vec3(1.4, -0.9, 0.0)),
                            atom_payload("HB1", "H", Vec3(0.8, 1.9, 0.8)),
                            atom_payload("HB2", "H", Vec3(2.0, 1.9, -0.8)),
                            atom_payload("HG", "H", Vec3(1.4, 3.3, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                    atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
                    atom_payload("N1", "N", Vec3(0.7, 1.4, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="included-retained-non-polymer-hydrogen-blocker",
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
        carrier=snapshot,
    )

    domain_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        atom_scope,
        component_library=build_local_retained_non_polymer_component_library(),
    )

    assert atom_scope_facts_supports_continuous_relaxation(domain_facts) is False
    assert (
        domain_facts.continuous_region_chemistry_readiness_facts
        .retained_non_polymer_facts[0]
        .needs_hydrogenation()
    )


def test_atom_scope_relaxation_allows_hydrogenated_retained_non_polymer_context(
) -> None:
    """Hydrogen-complete retained non-polymers in context should admit FF binding."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="SER",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.4, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.4, 1.4, 0.0)),
                            atom_payload("OG", "O", Vec3(1.4, 2.6, 0.0)),
                            atom_payload("H1", "H", Vec3(-0.7, 0.0, 0.0)),
                            atom_payload("H2", "H", Vec3(0.0, 0.7, 0.0)),
                            atom_payload("H3", "H", Vec3(0.0, -0.7, 0.0)),
                            atom_payload("HA", "H", Vec3(1.4, -0.9, 0.0)),
                            atom_payload("HB1", "H", Vec3(0.8, 1.9, 0.8)),
                            atom_payload("HB2", "H", Vec3(2.0, 1.9, -0.8)),
                            atom_payload("HG", "H", Vec3(1.4, 3.3, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                    atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
                    atom_payload("N1", "N", Vec3(0.7, 1.4, 0.0)),
                    atom_payload("H1", "H", Vec3(3.1, 1.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="included-retained-non-polymer-hydrogen-ready",
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
        carrier=snapshot,
    )

    domain_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        atom_scope,
        component_library=build_local_retained_non_polymer_component_library(),
    )

    assert atom_scope_facts_supports_continuous_relaxation(domain_facts)
    assert (
        domain_facts.continuous_region_chemistry_readiness_facts
        .retained_non_polymer_facts[0]
        .hydrogen_coverage_state
        is HydrogenCoverageState.COMPLETE
    )


def test_atom_scope_relaxation_allows_single_center_template_less_metal_context(
) -> None:
    """Single-center metal context should not block continuous realizability."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="SER",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.4, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.4, 1.4, 0.0)),
                            atom_payload("OG", "O", Vec3(1.4, 2.6, 0.0)),
                            atom_payload("H1", "H", Vec3(-0.7, 0.0, 0.0)),
                            atom_payload("H2", "H", Vec3(0.0, 0.7, 0.0)),
                            atom_payload("H3", "H", Vec3(0.0, -0.7, 0.0)),
                            atom_payload("HA", "H", Vec3(1.4, -0.9, 0.0)),
                            atom_payload("HB1", "H", Vec3(0.8, 1.9, 0.8)),
                            atom_payload("HB2", "H", Vec3(2.0, 1.9, -0.8)),
                            atom_payload("HG", "H", Vec3(1.4, 3.3, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="ZNX",
                residue_id=ResidueId("L", 1),
                atoms=(atom_payload("ZN1", "Zn", Vec3(1.7, 1.4, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="single-center-metal-template-less-passive-context",
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
        carrier=snapshot,
    )

    domain_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        atom_scope,
        component_library=build_default_component_library(),
    )

    assert atom_scope_facts_supports_continuous_relaxation(domain_facts)
    assert (
        domain_facts.continuous_bond_realizability_facts.state
        is ContinuousBondRealizabilityState.REALIZABLE
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires RDKit fallback chemistry")
def test_atom_scope_relaxation_keeps_single_center_template_less_context_passive_only(
) -> None:
    """Single-heavy fallback context is admissible only when non-movable."""

    structure = build_ser_with_template_less_ligand_structure(
        ligand_atoms=(
            atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
            atom_payload("H1", "H", Vec3(1.3, 0.9, 0.0)),
            atom_payload("H2", "H", Vec3(1.3, 1.9, 0.0)),
            atom_payload("H3", "H", Vec3(2.1, 0.9, 0.0)),
            atom_payload("H4", "H", Vec3(2.1, 1.9, 0.0)),
        ),
        source_name="single-center-template-less-passive-only",
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    passive_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
        carrier=snapshot,
    )

    passive_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        passive_scope,
        component_library=build_default_component_library(),
    )
    retained_fact = (
        passive_facts.continuous_region_chemistry_readiness_facts
        .retained_non_polymer_facts[0]
    )

    assert retained_fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert atom_scope_facts_supports_continuous_relaxation(passive_facts)
    assert (
        passive_facts.continuous_bond_realizability_facts.state
        is ContinuousBondRealizabilityState.REALIZABLE
    )

    selected_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        AtomSetScope(atom_refs=(AtomRef(ResidueId("L", 1), "C1"),)),
        carrier=snapshot,
    )
    selected_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        selected_scope,
        component_library=build_default_component_library(),
    )

    assert (
        selected_facts.continuous_bond_realizability_facts.state
        is ContinuousBondRealizabilityState.UNREALIZABLE
    )
    assert "LIG (L:1)" in (
        selected_facts.continuous_bond_realizability_facts.blocker or ""
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires RDKit fallback chemistry")
def test_atom_scope_relaxation_allows_connected_template_less_passive_context(
) -> None:
    """Connected retained context may use RDKit fallback topology passively."""

    structure = build_ser_with_template_less_ligand_structure(
        ligand_atoms=(
            atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
            atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
            atom_payload("H1", "H", Vec3(1.5, 0.7, 0.0)),
            atom_payload("H2", "H", Vec3(1.7, -0.7, 0.0)),
            atom_payload("H3", "H", Vec3(1.9, 0.7, 0.0)),
            atom_payload("H4", "H", Vec3(2.1, -0.7, 0.0)),
        ),
        source_name="connected-template-less-passive-context",
    )
    structure = with_topology_bonds(
        structure,
        *residue_bond_specs(
            ResidueId("A", 1),
            (
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("CA", "CB"),
                ("CB", "OG"),
            ),
        ),
        *residue_bond_specs(
            ResidueId("L", 1),
            (
                ("C1", "O1"),
                ("C1", "H1"),
                ("C1", "H2"),
                ("C1", "H3"),
                ("O1", "H4"),
            ),
            provenance=BondProvenance.REPAIR_INFERRED,
        ),
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
        carrier=snapshot,
    )

    domain_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        atom_scope,
        component_library=build_default_component_library(),
    )
    retained_fact = (
        domain_facts.continuous_region_chemistry_readiness_facts
        .retained_non_polymer_facts[0]
    )

    assert retained_fact.heavy_topology_source is (
        RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    )
    assert retained_fact.hydrogen_expectation_source is (
        RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    )
    assert retained_fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert not retained_fact.requires_hydrogen_completion()
    assert retained_fact.heavy_atom_topology_availability_state is (
        TopologyAvailabilityState.PRESENT
    )
    assert atom_scope_facts_supports_continuous_relaxation(domain_facts)
    assert (
        domain_facts.continuous_bond_realizability_facts.state
        is ContinuousBondRealizabilityState.REALIZABLE
    )
    require_atom_scope_continuous_relaxation_execution(domain_facts)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires RDKit fallback chemistry")
def test_atom_scope_relaxation_blocks_selected_template_less_ligand() -> None:
    """Fallback passive support must not make template-less ligands editable."""

    structure = build_ser_with_template_less_ligand_structure(
        ligand_atoms=(
            atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
            atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
            atom_payload("H1", "H", Vec3(1.5, 0.7, 0.0)),
            atom_payload("H2", "H", Vec3(1.7, -0.7, 0.0)),
            atom_payload("H3", "H", Vec3(1.9, 0.7, 0.0)),
            atom_payload("H4", "H", Vec3(2.1, -0.7, 0.0)),
        ),
        source_name="selected-template-less-ligand-blocker",
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_scope = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        AtomSetScope(atom_refs=(AtomRef(ResidueId("L", 1), "C1"),)),
        carrier=snapshot,
    )

    domain_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        atom_scope,
        component_library=build_default_component_library(),
    )

    assert (
        domain_facts.continuous_region_chemistry_readiness_facts
        .component_support_state
        is ComponentSupportState.ALL_SUPPORTED
    )
    assert (
        domain_facts.continuous_bond_realizability_facts.state
        is ContinuousBondRealizabilityState.UNREALIZABLE
    )
    assert "LIG (L:1)" in (
        domain_facts.continuous_bond_realizability_facts.blocker or ""
    )
    assert atom_scope_facts_continuous_relaxation_error(domain_facts) == (
        domain_facts.continuous_bond_realizability_facts.blocker
    )
    with pytest.raises(RefinementError, match="LIG"):
        require_atom_scope_continuous_relaxation_execution(domain_facts)


def test_topology_availability_observation_preserves_mixed_within_residue_facts() -> (
    None
):
    """One residue may carry different topology-availability facts by aspect."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    carrier = build_structure(
        chains=(build_chain("A", (build_residue("GLY", "A", 1, ("N", "CA")),)),),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="topology-availability-facts",
    )
    observation = TopologyAvailabilityObservation(
        residue_facts=(
            ResidueTopologyAvailabilityFact(
                residue_id=residue_id,
                aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                state=TopologyAvailabilityState.PRESENT,
            ),
            ResidueTopologyAvailabilityFact(
                residue_id=residue_id,
                aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                state=TopologyAvailabilityState.ABSENT,
            ),
        )
    )
    facts = TopologyAvailabilityFacts(
        carrier=carrier,
        scope=ResidueSetScope(residue_ids=(residue_id,)),
        residue_facts=observation.residue_facts,
    )

    assert observation.is_uniform(
        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
        state=TopologyAvailabilityState.PRESENT,
    )
    assert observation.is_uniform(
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
        state=TopologyAvailabilityState.ABSENT,
    )
    assert not topology_availability_facts_supports_continuous_relaxation(facts)
    assert (
        observation.state_for(
            residue_id,
            aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
        )
        is TopologyAvailabilityState.PRESENT
    )
    assert (
        observation.state_for(
            residue_id,
            aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
        )
        is TopologyAvailabilityState.ABSENT
    )


def test_transformation_step_budget_owns_step_budget_progression() -> None:
    """Run budgets should advance and detect exhaustion locally."""

    budget = TransformationStepBudget(max_steps=2)

    assert not budget.is_exhausted()
    assert not budget.advanced().is_exhausted()
    assert budget.advanced().advanced().is_exhausted()


def manual_runtime_state_facts(
    atom_input: AtomInput,
    *,
    atom_count: int,
    hydrogen_atom_count: int,
    component_support_state: ComponentSupportState,
    backbone_heavy_atom_completeness_state: BackboneHeavyAtomCompletenessState,
    sidechain_heavy_atom_completeness_state: SidechainHeavyAtomCompletenessState,
    hydrogen_applicability_state: HydrogenApplicabilityState,
    hydrogen_coverage_state: HydrogenCoverageState,
    stereochemistry_state: StereochemistryState,
    clash_state: ClashState,
    heavy_atom_topology_state: TopologyAvailabilityState,
    hydrogen_topology_state: TopologyAvailabilityState,
    hydrogen_attachment_resolution_state: HydrogenAttachmentResolutionState
    | None = None,
    continuous_bond_realizability_state: ContinuousBondRealizabilityState | None = None,
    continuous_bond_realizability_blocker: str | None = None,
) -> tuple[StructureProjectionStateFacts, AtomScopeStateFacts]:
    """Return manual primitive state facts for one selected local domain."""

    carrier = build_structure(
        chains=(
            build_chain(
                "A",
                (build_residue("GLY", "A", 1, ("N", "CA")),),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name="manual-runtime-state",
    )
    selected_scope = atom_input.as_scope()
    if isinstance(selected_scope, ResidueSetScope):
        referenced_residue_ids = selected_scope.residue_ids
    else:
        referenced_residue_ids = tuple(
            dict.fromkeys(atom_ref.residue_id for atom_ref in selected_scope.atom_refs)
        )
    structure_facts = StructureProjectionStateFacts(
        chain_count=1,
        residue_count=len(referenced_residue_ids),
        ligand_count=0,
        atom_count=atom_count,
        hydrogen_atom_count=hydrogen_atom_count,
        component_support_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=component_support_state,
        ),
        backbone_heavy_atom_completeness_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=backbone_heavy_atom_completeness_state,
        ),
        sidechain_heavy_atom_completeness_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=sidechain_heavy_atom_completeness_state,
        ),
        hydrogen_applicability_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=hydrogen_applicability_state,
        ),
        hydrogen_coverage_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=hydrogen_coverage_state,
        ),
        stereochemistry_fact=CarrierScopedState(
            carrier=carrier,
            scope=WholeStructureScope(),
            value=stereochemistry_state,
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
        if heavy_atom_topology_state is TopologyAvailabilityState.PRESENT and (
            hydrogen_topology_state
            in {
                TopologyAvailabilityState.PRESENT,
                TopologyAvailabilityState.NOT_APPLICABLE,
            }
        ):
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

    return structure_facts, AtomScopeStateFacts(
        selected_scope_facts=SelectedAtomScopeFacts(
            atom_count=atom_count,
            residue_count=len(referenced_residue_ids),
            hydrogen_atom_count=hydrogen_atom_count,
            structure_facts=structure_facts,
            clash_fact=CarrierScopedState(
                carrier=carrier,
                scope=atom_input.as_scope(),
                value=clash_state,
            ),
            topology_availability_facts=TopologyAvailabilityFacts(
                carrier=carrier,
                scope=ResidueSetScope(residue_ids=referenced_residue_ids),
                residue_facts=tuple(
                    residue_fact
                    for residue_id in referenced_residue_ids
                    for residue_fact in (
                        ResidueTopologyAvailabilityFact(
                            residue_id=residue_id,
                            aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                            state=heavy_atom_topology_state,
                        ),
                        ResidueTopologyAvailabilityFact(
                            residue_id=residue_id,
                            aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                            state=hydrogen_topology_state,
                        ),
                    )
                ),
            ),
            hydrogen_attachment_resolution_facts=HydrogenAttachmentResolutionFacts(
                carrier=carrier,
                scope=ResidueSetScope(residue_ids=referenced_residue_ids),
                residue_facts=HydrogenAttachmentResolutionObservation.uniform(
                    referenced_residue_ids,
                    state=resolved_hydrogen_attachment_state,
                ).residue_facts,
            ),
            continuous_bond_realizability_facts=ContinuousBondRealizabilityFacts(
                carrier=carrier,
                scope=atom_input.as_scope(),
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
                heavy_atom_topology_availability_state=heavy_atom_topology_state,
                hydrogen_topology_availability_state=hydrogen_topology_state,
                hydrogen_applicability_state=(
                    structure_facts.hydrogen_applicability_fact.value
                ),
                hydrogen_coverage_state=structure_facts.hydrogen_coverage_fact.value,
            ),
        ),
    )


def test_transformation_history_deduplicates_attempts_and_visits() -> None:
    """Run history should own canonical action/planning-signature memory."""

    atom_input = AtomInput(
        atom_indices=(
            AtomIndex(0),
            AtomIndex(1),
            AtomIndex(2),
            AtomIndex(3),
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
    )
    structure_facts, atom_scope_facts = manual_runtime_state_facts(
        atom_input,
        atom_count=4,
        hydrogen_atom_count=0,
        component_support_state=ComponentSupportState.ALL_SUPPORTED,
        backbone_heavy_atom_completeness_state=(
            BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        sidechain_heavy_atom_completeness_state=(
            SidechainHeavyAtomCompletenessState.COMPLETE
        ),
        hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
        hydrogen_coverage_state=HydrogenCoverageState.NONE,
        stereochemistry_state=StereochemistryState.CONSISTENT,
        clash_state=ClashState.NONE,
        heavy_atom_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_topology_state=TopologyAvailabilityState.ABSENT,
    )
    family_signature = TransformationFamilyAttemptSignature.from_state_facts(
        family=LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION,
        structure_facts=structure_facts,
        selected_scope=atom_input.as_scope(),
        selected_scope_facts=atom_scope_facts.selected_scope_facts,
    )
    planning_signature = TransformationPlanningSignature.from_state_facts(
        structure_facts=structure_facts,
        selected_scope=atom_input.as_scope(),
        selected_scope_facts=atom_scope_facts.selected_scope_facts,
    )
    history = TransformationHistory()
    history = history.with_attempted_family(family_signature)
    history = history.with_attempted_family(family_signature)
    history = history.with_observed_signature(planning_signature)
    history = history.with_observed_signature(planning_signature)

    assert history.attempted_family_signatures == (family_signature,)
    assert history.observed_planning_signatures == (planning_signature,)
    assert history.observation_count(planning_signature) == 2
    assert history.has_attempted_family(family_signature)
    assert history.has_observed_signature(planning_signature)


def test_transformation_progress_memory_tracks_streak_and_best_quality() -> None:
    """Progress memory should own no-progress streaks and best quality."""

    progress = TransformationProgressMemory()
    progress = progress.with_progress(
        TransformationProgressKind.NO_PROGRESS,
        quality_score=3.0,
    )
    progress = progress.with_progress(
        TransformationProgressKind.NO_PROGRESS,
        quality_score=4.0,
    )
    progress = progress.with_progress(
        TransformationProgressKind.GEOMETRY_IMPROVEMENT,
        quality_score=2.5,
    )

    assert progress.no_progress_count == 0
    assert (
        progress.last_progress_kind is TransformationProgressKind.GEOMETRY_IMPROVEMENT
    )
    assert progress.best_quality_so_far == 2.5


def test_transformation_planning_memory_is_behaviorful_aggregate() -> None:
    """Planning memory should compose budget, history, and progress memory."""

    atom_input = AtomInput(
        atom_indices=(
            AtomIndex(0),
            AtomIndex(1),
            AtomIndex(2),
            AtomIndex(3),
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=ResidueSetScope(residue_ids=(ResidueId("A", 1),)),
    )
    structure_facts, atom_scope_facts = manual_runtime_state_facts(
        atom_input,
        atom_count=4,
        hydrogen_atom_count=0,
        component_support_state=ComponentSupportState.ALL_SUPPORTED,
        backbone_heavy_atom_completeness_state=(
            BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        sidechain_heavy_atom_completeness_state=(
            SidechainHeavyAtomCompletenessState.COMPLETE
        ),
        hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
        hydrogen_coverage_state=HydrogenCoverageState.NONE,
        stereochemistry_state=StereochemistryState.CONSISTENT,
        clash_state=ClashState.NONE,
        heavy_atom_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_topology_state=TopologyAvailabilityState.ABSENT,
    )
    family_signature = TransformationFamilyAttemptSignature.from_state_facts(
        family=LocalTransformationFamily.DISCRETE_PRE_REFINEMENT_CORRECTION,
        structure_facts=structure_facts,
        selected_scope=atom_input.as_scope(),
        selected_scope_facts=atom_scope_facts.selected_scope_facts,
    )
    planning_signature = TransformationPlanningSignature.from_state_facts(
        structure_facts=structure_facts,
        selected_scope=atom_input.as_scope(),
        selected_scope_facts=atom_scope_facts.selected_scope_facts,
    )
    planning_memory = TransformationPlanningMemory.initial(max_steps=3)
    planning_memory = planning_memory.with_attempted_family(family_signature)
    planning_memory = planning_memory.with_observed_signature(planning_signature)
    planning_memory = planning_memory.advanced()
    planning_memory = planning_memory.with_progress(
        TransformationProgressKind.COMPLETENESS,
        quality_score=1.5,
    )
    planning_memory = planning_memory.terminated(
        TransformationTerminationReason.ACCEPTED
    )

    assert planning_memory.budget.step_count == 1
    assert planning_memory.history.attempted_family_signatures == (family_signature,)
    assert planning_memory.history.observed_planning_signatures == (planning_signature,)
    assert planning_memory.history.observation_count(planning_signature) == 1
    assert planning_memory.progress.best_quality_so_far == 1.5
    assert planning_memory.progress.no_progress_count == 0
    assert (
        planning_memory.termination.reason is TransformationTerminationReason.ACCEPTED
    )
    assert planning_memory.termination.is_terminal()


def test_transformation_signatures_derive_from_canonical_state() -> None:
    """Planner signatures should derive from canonical structure and domain state."""

    atom_input = AtomInput(
        atom_indices=(
            AtomIndex(0),
            AtomIndex(1),
            AtomIndex(2),
            AtomIndex(3),
            AtomIndex(4),
            AtomIndex(5),
            AtomIndex(6),
            AtomIndex(7),
        ),
        basis=AtomInputBasis.ATOMWISE,
        selected_scope=AtomSetScope(
            atom_refs=tuple(
                build_atom_ref("A", 1, atom_name)
                for atom_name in ("N", "CA", "C", "O", "HA", "HB1", "HB2", "HB3")
            )
        ),
    )
    structure_facts, atom_scope_facts = manual_runtime_state_facts(
        atom_input,
        atom_count=5,
        hydrogen_atom_count=1,
        component_support_state=ComponentSupportState.ALL_SUPPORTED,
        backbone_heavy_atom_completeness_state=(
            BackboneHeavyAtomCompletenessState.COMPLETE
        ),
        sidechain_heavy_atom_completeness_state=(
            SidechainHeavyAtomCompletenessState.COMPLETE
        ),
        hydrogen_applicability_state=HydrogenApplicabilityState.APPLICABLE,
        hydrogen_coverage_state=HydrogenCoverageState.COMPLETE,
        stereochemistry_state=StereochemistryState.CONSISTENT,
        clash_state=ClashState.NONE,
        heavy_atom_topology_state=TopologyAvailabilityState.PRESENT,
        hydrogen_topology_state=TopologyAvailabilityState.PRESENT,
    )

    structure_planning_signature = StructurePlanningSignature.from_facts(
        structure_facts
    )
    domain_signature = SelectedScopeSignature.from_scope(atom_input.as_scope())
    domain_planning_signature = SelectedScopePlanningSignature.from_scope_facts(
        atom_input.as_scope(),
        atom_scope_facts.selected_scope_facts,
    )
    planning_signature = TransformationPlanningSignature.from_state_facts(
        structure_facts=structure_facts,
        selected_scope=atom_input.as_scope(),
        selected_scope_facts=atom_scope_facts.selected_scope_facts,
    )
    family_signature = TransformationFamilyAttemptSignature.from_state_facts(
        family=LocalTransformationFamily.CONTINUOUS_LOCAL_RELAXATION,
        structure_facts=structure_facts,
        selected_scope=atom_input.as_scope(),
        selected_scope_facts=atom_scope_facts.selected_scope_facts,
    )

    assert (
        structure_planning_signature.hydrogen_coverage_state
        is HydrogenCoverageState.COMPLETE
    )
    assert (
        structure_planning_signature.terminal_boundary_observation
        == TerminalBoundaryObservation.empty()
    )
    assert domain_signature.scope == atom_input.as_scope()
    assert domain_planning_signature.selected_scope == domain_signature
    assert domain_planning_signature.clash_state is ClashState.NONE
    assert (
        domain_planning_signature.terminal_boundary_observation
        == TerminalBoundaryObservation.empty()
    )
    assert domain_planning_signature.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
        state=TopologyAvailabilityState.PRESENT,
    )
    assert domain_planning_signature.topology_availability.is_uniform(
        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
        state=TopologyAvailabilityState.PRESENT,
    )
    assert planning_signature.structure == structure_planning_signature
    assert planning_signature.selected_scope == domain_planning_signature
    assert (
        family_signature.family
        is LocalTransformationFamily.CONTINUOUS_LOCAL_RELAXATION
    )
    assert family_signature.planning_signature == planning_signature


def test_transformation_termination_memory_is_closed_terminal_memory() -> None:
    """Termination memory should become terminal only when a reason is set."""

    termination = TransformationTerminationMemory()

    assert not termination.is_terminal()
    assert termination.terminated(
        TransformationTerminationReason.NO_LEGAL_TRANSFORMATIONS
    ).is_terminal()


def topology_bond_spec(
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
    *,
    provenance: BondProvenance = BondProvenance.TEMPLATE_RESOLVED,
) -> tuple[ResidueId, str, str, BondProvenance]:
    """Return one residue-local topology bond spec for state tests."""

    return (residue_id, atom_name_1, atom_name_2, provenance)


def residue_bond_specs(
    residue_id: ResidueId,
    atom_name_pairs: tuple[tuple[str, str], ...],
    *,
    provenance: BondProvenance = BondProvenance.TEMPLATE_RESOLVED,
) -> tuple[tuple[ResidueId, str, str, BondProvenance], ...]:
    """Return residue-local topology bond specs for state tests."""

    return tuple(
        topology_bond_spec(
            residue_id,
            atom_name_1,
            atom_name_2,
            provenance=provenance,
        )
        for atom_name_1, atom_name_2 in atom_name_pairs
    )


def with_topology_bonds(
    structure: ProteinStructure,
    *bond_specs: tuple[ResidueId, str, str, BondProvenance],
) -> ProteinStructure:
    """Return a copy with residue-local canonical topology bonds."""

    topology_bonds = tuple(
        _topology_bond_from_spec(structure, bond_spec) for bond_spec in bond_specs
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(*structure.topology.bonds, *topology_bonds),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def _topology_bond_from_spec(
    structure: ProteinStructure,
    bond_spec: tuple[ResidueId, str, str, BondProvenance],
) -> TopologyBond:
    """Return one canonical topology bond from residue-local atom names."""

    residue_id, atom_name_1, atom_name_2, provenance = bond_spec
    return TopologyBond(
        atom_index_1=structure.constitution.atom_index(
            AtomRef(residue_id, atom_name_1)
        ),
        atom_index_2=structure.constitution.atom_index(
            AtomRef(residue_id, atom_name_2)
        ),
        relationship_type=BondRelationshipType.COVALENT,
        provenance=provenance,
    )


def uniform_topology_availability(
    residue_ids: tuple[ResidueId, ...],
    *,
    heavy_atom_state: TopologyAvailabilityState,
    hydrogen_state: TopologyAvailabilityState,
) -> TopologyAvailabilityObservation:
    """Return one uniform topology-availability observation for manual fixtures."""

    return TopologyAvailabilityObservation.uniform(
        residue_ids,
        heavy_atom_state=heavy_atom_state,
        hydrogen_state=hydrogen_state,
    )


def focused_structure_for_residue(*, seq_num: int) -> ProteinStructure:
    """Return a one-residue canonical structure from the representative fixture."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    residue = next(
        residue
        for residue in structure.constitution.chain("A").residues
        if residue.residue_id.seq_num == seq_num
    )
    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue.residue_id),
    )
    return build_structure(
        chains=(
            build_chain(
                "A",
                (
                    (
                        residue,
                        residue_geometry,
                        structure.topology.residue_formal_charge_by_atom_name(
                            constitution=structure.constitution,
                            residue_index=structure.constitution.residue_index(
                                residue.residue_id
                            ),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
        source_name=f"state-{seq_num}",
    )


def invert_threonine_residue(
    residue: CanonicalResiduePayload,
) -> CanonicalResiduePayload:
    """Swap THR substituent coordinates to invert the CB tetrahedral center."""

    residue_site, residue_geometry, formal_charge_by_atom_name = residue
    return (
        residue_site,
        residue_geometry.with_atom_geometries(
            (
                ("OG1", residue_geometry.atom_geometry("CG2")),
                ("CG2", residue_geometry.atom_geometry("OG1")),
            )
        ),
        formal_charge_by_atom_name,
    )


def build_chain(
    chain_id: str,
    residues: tuple[CanonicalResiduePayload, ...],
):
    """Build a canonical chain for state tests."""

    return chain_payload(chain_id, residues)


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
) -> CanonicalResiduePayload:
    """Build a canonical residue for state tests."""

    return residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=tuple(
            atom_payload(
                name=atom_name,
                element=infer_element(atom_name),
                position=Vec3(float(index), 0.0, 0.0),
            )
            for index, atom_name in enumerate(atom_names, start=1)
        ),
    )


def build_boundary_authenticity_fact(
    carrier: ProteinStructure,
    chain_id: str,
    seq_num: int,
    *,
    side: ResidueBoundarySide,
    authenticity: BoundaryAuthenticityState,
) -> CarrierScopedState[ProteinStructure, BoundaryAuthenticityState]:
    """Build one boundary-authenticity fact for ontology tests."""

    return CarrierScopedState(
        carrier=carrier,
        scope=ResidueBoundaryScope(
            residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
            side=side,
        ),
        value=authenticity,
    )


def build_oxt_presence_fact(
    carrier: ProteinStructure,
    chain_id: str,
    seq_num: int,
    *,
    presence: OxtPresenceState,
) -> CarrierScopedState[ProteinStructure, OxtPresenceState]:
    """Build one boundary-scoped OXT presence fact for ontology tests."""

    return CarrierScopedState(
        carrier=carrier,
        scope=ResidueBoundaryScope(
            residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
            side=ResidueBoundarySide.C_TERMINUS,
        ),
        value=presence,
    )


def build_atom_ref(chain_id: str, seq_num: int, atom_name: str):
    """Build one canonical atom reference for state tests."""

    return AtomRef(
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atom_name=atom_name,
    )


def build_ser_with_template_less_ligand_structure(
    *,
    ligand_atoms: tuple[CanonicalAtomPayload, ...],
    source_name: str,
) -> ProteinStructure:
    """Build one SER local-domain fixture with an unknown retained ligand."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="SER",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.4, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.4, 1.4, 0.0)),
                            atom_payload("OG", "O", Vec3(1.4, 2.6, 0.0)),
                            atom_payload("H1", "H", Vec3(-0.7, 0.0, 0.0)),
                            atom_payload("H2", "H", Vec3(0.0, 0.7, 0.0)),
                            atom_payload("H3", "H", Vec3(0.0, -0.7, 0.0)),
                            atom_payload("HA", "H", Vec3(1.4, -0.9, 0.0)),
                            atom_payload("HB1", "H", Vec3(0.8, 1.9, 0.8)),
                            atom_payload("HB2", "H", Vec3(2.0, 1.9, -0.8)),
                            atom_payload("HG", "H", Vec3(1.4, 3.3, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=ligand_atoms,
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name=source_name,
    )


def build_local_retained_non_polymer_component_library():
    """Return a component library with default and retained non-polymer templates."""

    component_library = build_default_component_library()
    retained_non_polymer_library = build_retained_non_polymer_component_library()
    for template in retained_non_polymer_library.templates.values():
        component_library = component_library.with_template(template)

    return component_library


def infer_element(atom_name: str) -> str:
    """Infer a simple test element from one atom name."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"atom_name must contain at least one letter: {atom_name}")

    return letters[0]
