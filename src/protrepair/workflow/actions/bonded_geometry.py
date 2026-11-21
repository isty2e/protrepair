"""One coordinate-only attempt at a requested bonded-geometry defect."""

from dataclasses import dataclass, replace

from protrepair.diagnostics.clashes import detect_clashes_from_context
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.diagnostics.water_contacts import stationary_water_refusal
from protrepair.errors import RefinementError
from protrepair.scope import AtomSetScope
from protrepair.state.bonded_geometry import BondedGeometryObservation, BondedTerm
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.base import ProjectedCodomainState, ProjectedDomainState
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.binding import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.rdkit import (
    PEPTIDE_CN_DISTANCE_TOLERANCE_ANGSTROM,
    PEPTIDE_CN_TARGET_DISTANCE_ANGSTROM,
)
from protrepair.transformer.refinement.acceptance import (
    assess_refinement_result_with_before_metrics,
    measure_refinement_acceptance_metrics_for_scope,
)
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.refinement.local_pipeline.candidates import (
    RefinementExecutionCandidate,
)
from protrepair.transformer.refinement.local_pipeline.lineage import (
    RefinementCandidateLineage,
)
from protrepair.transformer.refinement.local_pipeline.request import (
    LocalRefinementRequest,
)
from protrepair.transformer.refinement.outcome import (
    RefinementCodomainState,
    RefinementStage,
    RefinementStageOutcome,
)
from protrepair.transformer.refinement.result import (
    LocalRefinementResult,
    RefinementTransformationResult,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import WorkflowStructureTransformer
from protrepair.workflow.actions.bonded_geometry_motion import (
    BondedGeometryMotion,
    is_canonical_peptide_connection,
)
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True)
class BondedGeometryFeedback:
    """Boundary evidence at the exact parent of one guarded trial."""

    failure: str | None = None
    coupled_atoms: tuple[AtomRef, ...] = ()
    residual_focus: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class BondedGeometryAttemptResult(RefinementTransformationResult):
    """Return attempt evidence independently of retained coordinates."""

    feedback: BondedGeometryFeedback


@dataclass(frozen=True, slots=True, kw_only=True)
class _TrialCodomain(RefinementCodomainState):
    feedback: BondedGeometryFeedback


@dataclass(frozen=True, slots=True)
class BondedGeometryRepairTransformer(WorkflowStructureTransformer):
    """Execute one proposed atom scope, with no chemistry preparation."""

    focus: BondedTerm
    scope: AtomSetScope

    @property
    def workflow_scope(self) -> AtomSetScope:
        """Return the declared movable atom set."""

        return self.scope

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Score an isolated guarded solve against its exact parent."""

        del carrier
        parent = projected_domain.state
        observation = StructureChemistryObservation(
            parent,
            component_library=context.component_library,
            allow_retained_non_polymer_rdkit_fallback=context.allow_retained_non_polymer_rdkit_fallback,
            retained_non_polymer_chemistry_evidence=context.retained_non_polymer_chemistry_evidence,
        )
        focus_residue_ids = frozenset(ref.residue_id for ref in self.scope.atom_refs)
        before = BondedGeometryObservation.from_structure(
            parent,
            component_library=context.component_library,
            residue_ids=focus_residue_ids,
        )
        focus_excess = dict(before.residual_excesses).get(self.focus)
        motion = BondedGeometryMotion(parent, before.qualified_residue_ids)
        if focus_excess is None or not motion.allows(self.scope):
            return _TrialCodomain(
                scope=self.scope,
                state=parent,
                feedback=BondedGeometryFeedback(
                    "focus or declared motion is unavailable"
                ),
            )

        try:
            request, trial = self._execute_one(parent, observation=observation)
        except (RefinementError, ValueError) as error:
            return _TrialCodomain(
                scope=self.scope,
                state=parent,
                feedback=BondedGeometryFeedback(str(error)),
            )

        execution = LocalRefinementResult.observed_executions(trial)
        if any(
            issue.kind is ValidationIssueKind.REFINEMENT_REJECTED
            or issue.severity is IssueSeverity.ERROR
            for issue in trial.issues
        ):
            return _TrialCodomain(
                scope=self.scope,
                state=parent,
                relaxation_executions=execution,
                feedback=BondedGeometryFeedback("native refinement rejected the trial"),
            )

        endpoint = trial.refined_structure
        if parent.constitution.waters:
            refusal = stationary_water_refusal(
                parent,
                endpoint,
                run_input=context.original_structure,
                component_library=context.component_library,
            )
            if refusal is not None:
                return _TrialCodomain(
                    scope=self.scope,
                    state=parent,
                    issues=(refusal,),
                    relaxation_executions=execution,
                    feedback=BondedGeometryFeedback(refusal.message),
                )

        changed = self._verified_moves(parent, endpoint)
        if changed is None or any(
            frozenset(record.movable_scope.atom_refs) != frozenset(self.scope.atom_refs)
            for record in execution
        ):
            return _TrialCodomain(
                scope=self.scope,
                state=parent,
                relaxation_executions=execution,
                feedback=BondedGeometryFeedback(
                    "trial violated its coordinate-only motion scope"
                ),
            )

        after = BondedGeometryObservation.from_structure(
            endpoint,
            component_library=request.component_library,
            residue_ids=focus_residue_ids,
        )
        after_excesses = dict(after.residual_excesses)
        before_excesses = dict(before.residual_excesses)
        focus_after = after_excesses.get(self.focus, 0.0)
        moved = frozenset(self.scope.atom_refs)
        worsened_terms = tuple(
            term
            for term, excess in after.residual_excesses
            if excess > before_excesses.get(term, 0.0) and moved.intersection(term)
        )
        peptide_regressions = _worsened_peptide_bonds(parent, endpoint, moved)
        outside = {
            ref
            for term in (*worsened_terms, *peptide_regressions)
            for ref in term
            if ref not in moved
        }

        before_metrics = measure_refinement_acceptance_metrics_for_scope(
            parent,
            selected_scope=self.scope,
            component_library=request.component_library,
            restraint_library=request.restraint_library,
            clash_basis=request.clash_basis,
            pdb_block_projector=request.pdb_block_projector,
        )
        assessment = assess_refinement_result_with_before_metrics(
            self.scope,
            request.component_library,
            request.restraint_library,
            trial,
            before_metrics=before_metrics,
            clash_basis=request.clash_basis,
            pdb_block_projector=request.pdb_block_projector,
        )
        if not assessment.is_accepted() or focus_after > 0.0:
            outside.update(
                _trial_contact_partners(
                    request, endpoint, moved, include_persistent=focus_after > 0.0
                )
            )

        failure = None
        if before.assessed_references != after.assessed_references:
            failure = "trial changed which reference terms could be assessed"
            outside.clear()
        elif not assessment.is_accepted():
            assert assessment.rejection_issue is not None
            failure = assessment.rejection_issue.message
        elif worsened_terms or peptide_regressions:
            failure = "trial worsened a bonded term in or across the movable scope"
        elif not changed or focus_after >= focus_excess:
            failure = "trial did not reduce the focus term's reference excess"

        if failure is not None:
            return _TrialCodomain(
                scope=self.scope,
                state=parent,
                relaxation_executions=execution,
                feedback=BondedGeometryFeedback(
                    failure,
                    tuple(sorted(outside)),
                    residual_focus=assessment.is_accepted()
                    and before.assessed_references == after.assessed_references
                    and focus_after > 0.0,
                ),
            )

        stage = RefinementStageOutcome.from_result(
            replace(
                trial,
                delta=replace(
                    trial.delta,
                    moved_atoms=tuple(
                        move
                        for move in trial.delta.moved_atoms
                        if move.after_atom_index in changed
                    ),
                ),
            ),
            stage=RefinementStage.REPAIR,
        )
        return _TrialCodomain(
            scope=self.scope,
            state=stage.structure,
            repairs=stage.repairs,
            issues=stage.issues,
            relaxation_executions=execution,
            feedback=BondedGeometryFeedback(residual_focus=focus_after > 0.0),
        )

    def _execute_one(
        self, parent: ProteinStructure, *, observation: StructureChemistryObservation
    ) -> tuple[LocalRefinementRequest, RegionTransformationResult]:
        snapshot = ProteinStructureSnapshot.from_structure(parent)
        directive = RepairLocalRefinementDirective.from_atom_refs(
            self.scope.atom_refs, binding=RecommendedContinuousRelaxationBinding()
        )
        bound = directive.bind_execution(
            snapshot,
            component_library=observation.component_library,
            allow_retained_non_polymer_rdkit_fallback=observation.allow_retained_non_polymer_rdkit_fallback,
            retained_non_polymer_chemistry_evidence=observation.retained_non_polymer_chemistry_evidence,
            chemistry_observation=observation,
        )
        request = LocalRefinementRequest.from_context(
            ProteinTransformationContext.from_snapshot_atom_input(
                snapshot, bound.atom_input
            ),
            spec=bound.binding_decision.settings,
            component_library=observation.component_library,
            restraint_library=None,
            allow_retained_non_polymer_rdkit_fallback=observation.allow_retained_non_polymer_rdkit_fallback,
            retained_non_polymer_chemistry_evidence=observation.retained_non_polymer_chemistry_evidence,
            chemistry_observation=observation,
        )
        candidate = RefinementExecutionCandidate(
            context=request.context,
            lineage=RefinementCandidateLineage(),
            fallback_structure=parent,
        )
        return request, candidate.execute_continuous(request=request).outcome

    def _verified_moves(
        self, parent: ProteinStructure, endpoint: ProteinStructure
    ) -> frozenset[AtomIndex] | None:
        if (
            endpoint.constitution != parent.constitution
            or endpoint.topology != parent.topology
            or endpoint.provenance != parent.provenance
        ):
            return None
        admitted = frozenset(
            parent.constitution.atom_index(ref) for ref in self.scope.atom_refs
        )
        changed: set[AtomIndex] = set()
        for index, (old, new) in enumerate(
            zip(
                parent.geometry.atom_geometries,
                endpoint.geometry.atom_geometries,
                strict=True,
            )
        ):
            if old == new:
                continue
            slot = AtomIndex(index)
            if slot not in admitted or old.with_position(new.position) != new:
                return None
            changed.add(slot)
        return frozenset(changed)

    def lift_projected_codomain(
        self,
        carrier: TransformationResult,
        projected_codomain: ProjectedCodomainState[ProteinStructure],
        *,
        context: TransformerExecutionContext,
    ) -> TransformationResult:
        """Lift only retained coordinates and typed attempt feedback."""

        del context
        if not isinstance(projected_codomain, _TrialCodomain):
            raise TypeError("bonded-geometry repair requires a trial codomain")
        merged = projected_codomain.lift(carrier)
        return BondedGeometryAttemptResult(
            structure=merged.structure,
            repairs=merged.repairs,
            issues=merged.issues,
            relaxation_executions=merged.relaxation_executions,
            feedback=projected_codomain.feedback,
        )


def _worsened_peptide_bonds(
    parent: ProteinStructure, endpoint: ProteinStructure, moved: frozenset[AtomRef]
) -> tuple[BondedTerm, ...]:
    constitution = parent.constitution
    incident = parent.topology.incident_bonds_by_atom()
    bonds = {
        bond for ref in moved for bond in incident.get(constitution.atom_index(ref), ())
    }
    worsened: list[BondedTerm] = []
    for bond in bonds:
        first, second = bond.endpoint_pair()
        left, right = constitution.atom_ref_at(first), constitution.atom_ref_at(second)
        if not is_canonical_peptide_connection(left, right, bond):
            continue
        before, after = (
            max(
                0.0,
                abs(
                    geometry.atom_geometry(first).distance_to(
                        geometry.atom_geometry(second)
                    )
                    - PEPTIDE_CN_TARGET_DISTANCE_ANGSTROM
                )
                - PEPTIDE_CN_DISTANCE_TOLERANCE_ANGSTROM,
            )
            for geometry in (parent.geometry, endpoint.geometry)
        )
        if after > before:
            worsened.append(tuple(sorted((left, right))))
    return tuple(sorted(worsened))


def _trial_contact_partners(
    request: LocalRefinementRequest,
    endpoint: ProteinStructure,
    moved: frozenset[AtomRef],
    *,
    include_persistent: bool,
) -> tuple[AtomRef, ...]:
    residue_ids = frozenset(ref.residue_id for ref in moved)
    before, after = (
        {
            tuple(
                sorted(
                    (
                        AtomRef(clash.left_residue_id, clash.left_atom_name),
                        AtomRef(clash.right_residue_id, clash.right_atom_name),
                    )
                )
            ): clash.overlap_angstrom
            for clash in detect_clashes_from_context(
                request.clash_basis.bind_context(structure),
                focus_residue_ids=residue_ids,
            ).clashes
        }
        for structure in (request.context.source_snapshot.structure, endpoint)
    )
    return tuple(
        right if left in moved else left
        for (left, right), overlap in sorted(after.items())
        if (left in moved) != (right in moved)
        and (
            overlap > before.get((left, right), 0.0)
            or (include_persistent and (left, right) in before)
        )
    )
