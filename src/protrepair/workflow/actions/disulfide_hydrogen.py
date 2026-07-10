"""Planner-visible normalization of disulfide-incompatible hydrogens."""

from dataclasses import dataclass

from protrepair.diagnostics.events import RepairEvent
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.scope import ResidueSetScope
from protrepair.state.structure_topology import StructureDisulfideHydrogenFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef
from protrepair.transformer.base import ProjectedCodomainState, ProjectedDomainState
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import WorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True)
class DisulfideHydrogenNormalizationTransformer(WorkflowStructureTransformer):
    """Remove forbidden thiol hydrogens from canonical disulfide endpoints."""

    forbidden_hydrogen_atom_refs: tuple[AtomRef, ...]

    def __post_init__(self) -> None:
        raw_atom_refs = tuple(self.forbidden_hydrogen_atom_refs)
        if not raw_atom_refs:
            raise ValueError(
                "disulfide hydrogen normalization requires forbidden atoms"
            )
        if any(not isinstance(atom_ref, AtomRef) for atom_ref in raw_atom_refs):
            raise TypeError("disulfide hydrogen normalization requires AtomRef values")
        atom_refs = tuple(sorted(dict.fromkeys(raw_atom_refs)))
        object.__setattr__(self, "forbidden_hydrogen_atom_refs", atom_refs)

    @property
    def workflow_scope(self) -> ResidueSetScope:
        """Return residue endpoints whose atom inventories are normalized."""

        return ResidueSetScope(
            residue_ids=tuple(
                dict.fromkeys(
                    atom_ref.residue_id
                    for atom_ref in self.forbidden_hydrogen_atom_refs
                )
            )
        )

    def accepts_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        context: TransformerExecutionContext,
    ) -> bool:
        """Return whether planned atoms exactly match current contradictions."""

        del context
        active_atom_refs = frozenset(
            StructureDisulfideHydrogenFacts.from_structure(
                projected_domain.state
            ).forbidden_hydrogen_atom_refs()
        )
        return active_atom_refs == frozenset(self.forbidden_hydrogen_atom_refs)

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Return the structure without disulfide-incompatible H/D/T atoms."""

        del carrier, context
        updated_structure = projected_domain.state.without_atom_refs(
            self.forbidden_hydrogen_atom_refs
        )
        return ProjectedCodomainState(
            scope=self.workflow_scope,
            state=updated_structure,
            repairs=tuple(
                RepairEvent.for_residue(
                    kind=RepairEventKind.HYDROGENS_REMOVED,
                    residue_id=residue_id,
                    component_id="CYS",
                    atom_names=tuple(
                        atom_ref.atom_name
                        for atom_ref in self.forbidden_hydrogen_atom_refs
                        if atom_ref.residue_id == residue_id
                    ),
                    details=(
                        "removed thiol hydrogen incompatible with canonical "
                        "disulfide topology"
                    ),
                )
                for residue_id in self.workflow_scope.residue_ids
            ),
        )
