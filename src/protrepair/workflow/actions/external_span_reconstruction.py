"""Planner-visible external absent-span reconstruction transformer invocations."""

from dataclasses import dataclass

from protrepair.chemistry.component.topology import (
    template_resolved_topology_bonds_for_new_atoms,
)
from protrepair.diagnostics.events import (
    EventScope,
    RepairEvent,
    ResidueAtomImpact,
    ValidationIssue,
)
from protrepair.diagnostics.geometry import (
    detect_heavy_geometry,
)
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.diagnostics.stereochemistry import (
    detect_sidechain_stereochemistry,
    detect_standard_alpha_carbon_chirality_violations,
)
from protrepair.scope import AbsentResidueSpanScope, ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.endpoint import StructureEndpoint
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import StructureProvenanceOrigin
from protrepair.structure.topology import (
    sequence_inferred_polymer_topology_bonds_for_new_atoms,
)
from protrepair.transformer.base import (
    ProjectedCodomainState,
    ProjectedDomainState,
)
from protrepair.transformer.completion.span_reconstruction import (
    SpanReconstructionFailure,
    SpanReconstructionFailureKind,
    reconstruct_donor_span,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import WorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.contracts.external_reference import (
    ExternalSpanReconstructionSpec,
)


@dataclass(frozen=True, slots=True)
class ExternalSpanReconstructionTransformer(WorkflowStructureTransformer):
    """Workflow-visible donor reconstruction over one absent residue span.

    Parameters
    ----------
    reconstruction : ExternalSpanReconstructionSpec
        Explicit source scope and donor mapping executed by this action.

    Raises
    ------
    TypeError
        If ``reconstruction`` is not an external span reconstruction spec.
    """

    reconstruction: ExternalSpanReconstructionSpec

    @property
    def workflow_scope(self) -> AbsentResidueSpanScope:
        """Return the absent-span scope this action transforms."""

        return self.reconstruction.scope

    def __post_init__(self) -> None:
        if not isinstance(self.reconstruction, ExternalSpanReconstructionSpec):
            raise TypeError(
                "external span reconstruction transformers require an "
                "ExternalSpanReconstructionSpec payload"
            )

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Transform one absent span by grafting donor residues into the source."""

        del carrier
        target = self.workflow_scope
        donor_structure = self.reconstruction.donor_structure
        donor_preceding_residue_id, donor_following_residue_id = (
            self.reconstruction.donor_flanking_residue_ids()
        )
        provenance_origins = self._provenance_origins(
            donor_preceding_residue_id=donor_preceding_residue_id,
            donor_following_residue_id=donor_following_residue_id,
        )
        source_scope_failure = self._source_scope_failure(projected_domain.state)
        if source_scope_failure is not None:
            return self._failed_codomain(
                projected_domain.state,
                source_scope_failure,
                provenance_origins=provenance_origins,
            )
        chemistry_failure = self._donor_chemistry_failure(context=context)
        if chemistry_failure is not None:
            return self._failed_codomain(
                projected_domain.state,
                chemistry_failure,
                provenance_origins=provenance_origins,
            )
        closure_outcome = reconstruct_donor_span(
            projected_domain.state,
            scope=target,
            donor_structure=donor_structure,
            donor_residue_ids=self.reconstruction.donor_residue_ids,
            donor_preceding_residue_id=donor_preceding_residue_id,
            donor_following_residue_id=donor_following_residue_id,
        )
        if isinstance(closure_outcome, SpanReconstructionFailure):
            return self._failed_codomain(
                projected_domain.state,
                closure_outcome,
                provenance_origins=provenance_origins,
            )

        reconstructed_residue_sites = tuple(
            payload.residue_site for payload in closure_outcome.residue_payloads
        )
        reconstructed_residue_geometries = tuple(
            payload.residue_geometry for payload in closure_outcome.residue_payloads
        )
        reconstructed_residue_formal_charge_payloads = tuple(
            payload.formal_charge_by_atom_name
            for payload in closure_outcome.residue_payloads
        )
        anchor_residue_id = (
            target.preceding_residue_id
            if target.preceding_residue_id is not None
            else target.following_residue_id
        )
        assert anchor_residue_id is not None
        source_structure = projected_domain.state
        updated_constitution = (
            source_structure.constitution.with_inserted_chain_residues(
                anchor_residue_id.chain_id,
                reconstructed_residue_sites,
            )
        )
        updated_structure = source_structure.with_inserted_chain_residue_facets(
            anchor_residue_id.chain_id,
            residue_sites=reconstructed_residue_sites,
            residue_geometries=reconstructed_residue_geometries,
            additional_topology_bonds=(
                *template_resolved_topology_bonds_for_new_atoms(
                    source_constitution=source_structure.constitution,
                    target_constitution=updated_constitution,
                    component_library=context.component_library,
                ),
                *sequence_inferred_polymer_topology_bonds_for_new_atoms(
                    source_constitution=source_structure.constitution,
                    target_constitution=updated_constitution,
                ),
            ),
            residue_formal_charge_payloads=(
                reconstructed_residue_formal_charge_payloads
            ),
        )
        geometry_report = detect_heavy_geometry(
            updated_structure,
            component_library=context.component_library,
            residue_ids=target.absent_residue_ids,
        )
        if not geometry_report.is_empty():
            return self._failed_codomain(
                source_structure,
                SpanReconstructionFailure(
                    kind=SpanReconstructionFailureKind.INVALID_INTRINSIC_GEOMETRY,
                    message=(
                        "reconstructed donor span contains intrinsic heavy-atom "
                        "geometry outside the accepted bounds"
                    ),
                ),
                provenance_origins=provenance_origins,
            )
        alpha_carbon_chirality_violations = (
            detect_standard_alpha_carbon_chirality_violations(
                updated_structure,
                residue_ids=target.absent_residue_ids,
            )
        )
        if alpha_carbon_chirality_violations:
            return self._failed_codomain(
                source_structure,
                SpanReconstructionFailure(
                    kind=SpanReconstructionFailureKind.INVALID_STEREOCHEMISTRY,
                    message=(
                        "reconstructed donor span contains invalid standard "
                        "alpha-carbon chirality"
                    ),
                ),
                provenance_origins=provenance_origins,
            )
        stereochemistry_report = detect_sidechain_stereochemistry(
            updated_structure,
            component_library=context.component_library,
            residue_ids=target.absent_residue_ids,
        )
        if not stereochemistry_report.is_empty():
            return self._failed_codomain(
                source_structure,
                SpanReconstructionFailure(
                    kind=SpanReconstructionFailureKind.INVALID_STEREOCHEMISTRY,
                    message=(
                        "reconstructed donor span contains invalid side-chain "
                        "stereochemistry"
                    ),
                ),
                provenance_origins=provenance_origins,
            )
        closure_details = "one-anchor donor projection"
        if closure_outcome.endpoint_rmsd_angstrom is not None:
            closure_details = (
                "donor-seeded CCD closure; endpoint RMSD "
                f"{closure_outcome.endpoint_rmsd_angstrom:.3f} A after "
                f"{closure_outcome.iteration_count} CCD iteration(s)"
            )
        repair_event = RepairEvent.for_residue_span(
            kind=RepairEventKind.ABSENT_RESIDUE_SPAN_RECONSTRUCTED,
            residue_ids=target.absent_residue_ids,
            residue_impacts=tuple(
                ResidueAtomImpact(
                    residue_id=residue.residue_id,
                    component_id=residue.component_id,
                    atom_names=residue.atom_site_names(),
                )
                for residue in reconstructed_residue_sites
            ),
            provenance_origins=provenance_origins,
            details=closure_details,
        )
        return ProjectedCodomainState(
            scope=self.workflow_scope,
            state=updated_structure,
            repairs=(repair_event,),
        )

    def _source_scope_failure(
        self,
        structure: ProteinStructure,
    ) -> SpanReconstructionFailure | None:
        """Return stale or incompatible source-state evidence for this request."""

        for anchor_residue_id in self.workflow_scope.anchor_residue_ids():
            anchor_residue = structure.constitution.residue_or_ligand(anchor_residue_id)
            if anchor_residue is None:
                return SpanReconstructionFailure(
                    kind=SpanReconstructionFailureKind.MISSING_BACKBONE_CONTEXT,
                    message=(
                        "span reconstruction source anchor "
                        f"{anchor_residue_id.display_token()} is absent"
                    ),
                )
            if anchor_residue.is_hetero:
                return SpanReconstructionFailure(
                    kind=SpanReconstructionFailureKind.MISSING_BACKBONE_CONTEXT,
                    message=(
                        "span reconstruction source anchor "
                        f"{anchor_residue_id.display_token()} is not a polymer residue"
                    ),
                )

        materialized_target_residue_ids = tuple(
            residue_id
            for residue_id in self.workflow_scope.absent_residue_ids
            if structure.constitution.residue_or_ligand(residue_id) is not None
        )
        if materialized_target_residue_ids:
            details = ", ".join(
                residue_id.display_token()
                for residue_id in materialized_target_residue_ids
            )
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.INVALID_TARGET_STATE,
                message=(
                    "span reconstruction target residue(s) are already materialized: "
                    f"{details}"
                ),
            )
        return None

    def _provenance_origins(
        self,
        *,
        donor_preceding_residue_id: ResidueId | None,
        donor_following_residue_id: ResidueId | None,
    ) -> tuple[StructureProvenanceOrigin, ...]:
        """Return source and supporting origins for this reconstruction attempt."""

        donor_structure = self.reconstruction.donor_structure
        donor_evidence_residue_ids = tuple(
            residue_id
            for residue_id in (
                donor_preceding_residue_id,
                *self.reconstruction.donor_residue_ids,
                donor_following_residue_id,
            )
            if residue_id is not None
        )
        return (
            StructureProvenanceOrigin(
                structure_endpoint=StructureEndpoint.source(self.workflow_scope)
            ),
            StructureProvenanceOrigin(
                structure_endpoint=StructureEndpoint.supporting(
                    ResidueSetScope(residue_ids=donor_evidence_residue_ids),
                    token=(
                        donor_structure.provenance.ingress.source_name or "supporting-0"
                    ),
                )
            ),
        )

    def _donor_chemistry_failure(
        self,
        *,
        context: TransformerExecutionContext,
    ) -> SpanReconstructionFailure | None:
        """Return missing or incompatible donor chemistry before running closure."""

        donor_structure = self.reconstruction.donor_structure
        unsupported_component_ids: list[str] = []
        unexpected_atoms_by_component: list[tuple[str, tuple[str, ...]]] = []
        for donor_residue_id in self.reconstruction.donor_residue_ids:
            donor_residue = donor_structure.constitution.residue_or_ligand(
                donor_residue_id
            )
            assert donor_residue is not None
            template = context.component_library.get(donor_residue.component_id)
            if template is None:
                if donor_residue.component_id not in unsupported_component_ids:
                    unsupported_component_ids.append(donor_residue.component_id)
                continue

            expected_atom_names = frozenset(template.expected_atom_names())
            unexpected_heavy_atom_names = tuple(
                atom_site.name
                for atom_site in donor_residue.atom_sites
                if not atom_site.is_hydrogen()
                and atom_site.name not in expected_atom_names
            )
            if unexpected_heavy_atom_names:
                unexpected_atoms_by_component.append(
                    (donor_residue.component_id, unexpected_heavy_atom_names)
                )

        if unsupported_component_ids:
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.UNSUPPORTED_COMPONENT_CHEMISTRY,
                message=(
                    "span reconstruction lacks canonical chemistry for donor "
                    f"component(s) {', '.join(unsupported_component_ids)}"
                ),
            )
        if unexpected_atoms_by_component:
            details = "; ".join(
                f"{component_id}: {', '.join(atom_names)}"
                for component_id, atom_names in unexpected_atoms_by_component
            )
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.UNSUPPORTED_COMPONENT_CHEMISTRY,
                message=(
                    "span reconstruction donor atoms fall outside canonical "
                    f"component chemistry ({details})"
                ),
            )
        alpha_carbon_chirality_violations = (
            detect_standard_alpha_carbon_chirality_violations(
                donor_structure,
                residue_ids=self.reconstruction.donor_residue_ids,
            )
        )
        if alpha_carbon_chirality_violations:
            details = ", ".join(
                residue_id.display_token()
                for residue_id in alpha_carbon_chirality_violations
            )
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.INVALID_STEREOCHEMISTRY,
                message=(
                    "span reconstruction donor has invalid standard alpha-carbon "
                    f"chirality at {details}"
                ),
            )
        stereochemistry_report = detect_sidechain_stereochemistry(
            donor_structure,
            component_library=context.component_library,
            residue_ids=self.reconstruction.donor_residue_ids,
        )
        if not stereochemistry_report.is_empty():
            details = ", ".join(
                violation.residue_id.display_token()
                for violation in stereochemistry_report.violations
            )
            return SpanReconstructionFailure(
                kind=SpanReconstructionFailureKind.INVALID_STEREOCHEMISTRY,
                message=(
                    "span reconstruction donor has invalid side-chain "
                    f"stereochemistry at {details}"
                ),
            )
        return None

    def _failed_codomain(
        self,
        structure: ProteinStructure,
        failure: SpanReconstructionFailure,
        *,
        provenance_origins: tuple[StructureProvenanceOrigin, ...],
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Return an unchanged structure with one typed span failure."""

        return ProjectedCodomainState(
            scope=self.workflow_scope,
            state=structure,
            issues=(
                ValidationIssue(
                    kind=ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED,
                    severity=IssueSeverity.ERROR,
                    message=failure.message,
                    scope=EventScope.for_residue_span(
                        self.workflow_scope.absent_residue_ids
                    ),
                    provenance_origins=provenance_origins,
                ),
            ),
        )
