"""A site-only chemical write; unrelated H and heavy atoms are outside its scope."""

from dataclasses import dataclass, replace

from protrepair.chemistry.microstate.bound_protonation import active_site_protonations
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.requested_sites import RequestedSiteAssessment
from protrepair.chemistry.microstate.resolution import (
    AppliedMicrostateChoice,
    MicrostateSelectionBasis,
)
from protrepair.chemistry.microstate.selection import SiteProtonation
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import GeometryPlacementError
from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.base import ProjectedCodomainState, ProjectedDomainState
from protrepair.transformer.completion.hydrogen.microstate import (
    place_polymer_microstate_hydrogens,
)
from protrepair.transformer.polymer_microstate import apply_polymer_microstate_patches
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import WorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True)
class SiteProtonationTransformer(WorkflowStructureTransformer):
    """Apply a compatible caller class atomically to one input-bound site."""

    request: SiteProtonation

    @property
    def workflow_scope(self) -> ResidueSetScope:
        return ResidueSetScope((self.request.residue_id,))

    def reapplication_context(self, structure: ProteinStructure) -> ProteinStructure:
        return structure

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        structure = projected_domain.state
        binding = active_site_protonations()
        if binding is None:
            raise ValueError(
                "site protonation requires an input-bound workflow request"
            )
        index = PolymerMicrostateContext(structure)
        assessed = RequestedSiteAssessment.observe(
            binding, index, context.component_library, self.request
        )
        target = assessed.target
        if target is None or not assessed.requires_commit():
            return ProjectedCodomainState(scope=self.workflow_scope, state=structure)
        rid = self.request.residue_id
        component = target.site.template.component_id

        if assessed.realized is not None:
            graph = assessed.realized
            names = {a.name for a in graph.atoms}
            retained = tuple(
                c
                for c in structure.provenance.microstate_choices
                if c.residue_id != rid
                or not any(a.name in names for a in c.graph.atoms)
            )
            saved = AppliedMicrostateChoice(
                rid, component, graph, MicrostateSelectionBasis.COMMITTED_REQUEST
            )
            result = structure.with_provenance(
                replace(structure.provenance, microstate_choices=(*retained, saved))
            )
            return ProjectedCodomainState(scope=self.workflow_scope, state=result)

        try:
            patch = place_polymer_microstate_hydrogens(
                index, rid, target.site, retain_applied_choice=False
            )
            result = apply_polymer_microstate_patches(structure, (patch,))
            verified = RequestedSiteAssessment.observe(
                binding,
                PolymerMicrostateContext(result),
                context.component_library,
                self.request,
            )
            if verified.realized is None:
                raise ValueError(
                    "staged site does not realize the source-compatible class"
                )
        except (ValueError, GeometryPlacementError, RdkitUnavailableError) as error:
            issue = ValidationIssue.for_residue(
                kind=ValidationIssueKind.SITE_PROTONATION_FAILED,
                severity=IssueSeverity.WARNING,
                residue_id=rid,
                component_id=component,
                message=str(error),
            )
            return ProjectedCodomainState(
                scope=self.workflow_scope, state=structure, issues=(issue,)
            )

        event = RepairEvent.for_residue(
            kind=RepairEventKind.SITE_PROTONATION_APPLIED,
            residue_id=rid,
            component_id=component,
            atom_names=tuple(a.name for a in patch.graph.atoms),
            details=self.request.site.value,
        )
        return ProjectedCodomainState(
            scope=self.workflow_scope, state=result, repairs=(event,)
        )
