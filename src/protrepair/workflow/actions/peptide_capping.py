"""Apply a single checked peptide-cap transition on its assessed branch."""

from dataclasses import dataclass, field

from protrepair.diagnostics.events import (
    EventScope,
    RepairEvent,
    ResidueAtomImpact,
    ValidationIssue,
)
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.bound_peptide_ports import (
    BoundPeptidePorts,
    active_peptide_ports,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.base import ProjectedCodomainState, ProjectedDomainState
from protrepair.transformer.completion.terminal.cap_placement import PeptideCapPlacement
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import WorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True, kw_only=True)
class PeptideCapRefusal(ValidationIssue):
    """Evidence about one attempted port, separate from a retained product."""

    reason: str
    attempted: int
    refusals: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PeptideCappingTransformer(WorkflowStructureTransformer):
    """A once-per-lineage attempt, executable only on the parent that supplied it."""

    port: AtomRef
    cap_id: ResidueId
    parent: ProteinStructure = field(compare=False, repr=False)
    binding: BoundPeptidePorts = field(compare=False, repr=False)

    @property
    def workflow_scope(self) -> ResidueSetScope:
        return ResidueSetScope((self.port.residue_id, self.cap_id))

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        parent = projected_domain.state
        if (
            parent is not self.parent
            or active_peptide_ports() is not self.binding
            or self.binding.source is not context.original_structure
        ):
            raise ValueError("peptide cap proposal belongs to another parent or run")
        attempted = PeptideCapPlacement.execute(
            self.binding, parent, self.port, self.cap_id, context.component_library
        )
        result = attempted.structure
        if result is parent:
            refusal = PeptideCapRefusal(
                kind=ValidationIssueKind.PEPTIDE_CAP_REFUSED,
                severity=IssueSeverity.WARNING,
                scope=EventScope.for_residue(self.port.residue_id),
                reason=attempted.reason,
                attempted=attempted.attempted,
                refusals=attempted.refusals,
                message=f"{self.port.display_token()}: {attempted.detail or attempted.reason}",
            )
            return ProjectedCodomainState(
                self.workflow_scope, parent, issues=(refusal,)
            )

        cap = result.constitution.residue_site_at(
            result.constitution.residue_index(self.cap_id)
        )
        flank = result.constitution.residue_site_at(
            result.constitution.residue_index(self.port.residue_id)
        )
        added = tuple(
            a.name
            for a in flank.atom_sites
            if self.parent.constitution.resolve_atom_site(
                AtomRef(flank.residue_id, a.name)
            )
            is None
        )
        event = RepairEvent(
            kind=RepairEventKind.PEPTIDE_CAP_PLACED
            if attempted.reason == "placed"
            else RepairEventKind.PEPTIDE_CAP_TOPOLOGY_RESOLVED,
            scope=EventScope.for_residue_set((self.port.residue_id, self.cap_id)),
            residue_impacts=(
                ResidueAtomImpact(
                    flank.residue_id, flank.component_id, (self.port.atom_name, *added)
                ),
                ResidueAtomImpact(
                    self.cap_id, cap.component_id, tuple(a.name for a in cap.atom_sites)
                ),
            ),
            details=f"{self.port.display_token()} to {self.cap_id.display_token()}; {attempted.attempted} poses evaluated",
        )
        return ProjectedCodomainState(self.workflow_scope, result, repairs=(event,))
