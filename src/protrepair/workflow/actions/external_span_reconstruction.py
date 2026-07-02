"""Planner-visible external absent-span reconstruction transformer invocations."""

from dataclasses import dataclass

from protrepair.diagnostics.events import RepairEvent, ResidueAtomImpact
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.scope import AbsentResidueSpanScope, ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.endpoint import StructureEndpoint
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.provenance import StructureProvenanceOrigin
from protrepair.transformer.base import (
    ProjectedCodomainState,
    ProjectedDomainState,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import WorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.contracts.external_reference import (
    ExternalSpanReconstructionSpec,
    lower_blueprint_coverage_gap_to_absent_span_scope,
)


@dataclass(frozen=True, slots=True)
class ExternalSpanReconstructionTransformer(WorkflowStructureTransformer):
    """Workflow-visible donor reconstruction over one absent residue span."""

    scope: AbsentResidueSpanScope
    reconstruction: ExternalSpanReconstructionSpec

    @property
    def workflow_scope(self) -> AbsentResidueSpanScope:
        """Return the absent-span scope this action transforms."""

        return self.scope

    def __post_init__(self) -> None:
        if not isinstance(self.scope, AbsentResidueSpanScope):
            raise TypeError(
                "external span reconstruction transformers require an "
                "AbsentResidueSpanScope"
            )
        if not isinstance(self.reconstruction, ExternalSpanReconstructionSpec):
            raise TypeError(
                "external span reconstruction transformers require an "
                "ExternalSpanReconstructionSpec payload"
            )

    @classmethod
    def from_reconstruction_spec(
        cls,
        reconstruction: ExternalSpanReconstructionSpec,
    ) -> "ExternalSpanReconstructionTransformer":
        """Build one transformer from one explicit span-reconstruction request."""

        return cls(
            scope=lower_blueprint_coverage_gap_to_absent_span_scope(
                reconstruction.blueprint_coverage_gap
            ),
            reconstruction=reconstruction,
        )

    def accepts_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        context: TransformerExecutionContext,
    ) -> bool:
        """Return whether the current structure still admits this donor graft."""

        del context
        target = self.scope
        structure = projected_domain.state
        if (
            target.preceding_residue_id is not None
            and structure.constitution.residue_or_ligand(target.preceding_residue_id)
            is None
        ):
            return False
        if (
            target.following_residue_id is not None
            and structure.constitution.residue_or_ligand(target.following_residue_id)
            is None
        ):
            return False
        if any(
            structure.constitution.residue_or_ligand(residue_id) is not None
            for residue_id in target.absent_residue_ids
        ):
            return False

        donor_structure = self.reconstruction.donor_structure
        return all(
            donor_structure.constitution.residue_or_ligand(residue_id) is not None
            for residue_id in self.reconstruction.donor_residue_ids
        )

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Transform one absent span by grafting donor residues into the source."""

        del carrier, context
        target = self.scope
        donor_structure = self.reconstruction.donor_structure
        reconstructed_residue_sites: list[ResidueSite] = []
        reconstructed_residue_geometries: list[ResidueGeometry] = []
        reconstructed_residue_formal_charge_payloads: list[
            tuple[tuple[str, int | None], ...]
        ] = []
        for absent_residue_id, donor_residue_id in zip(
            target.absent_residue_ids,
            self.reconstruction.donor_residue_ids,
            strict=True,
        ):
            donor_residue_index = donor_structure.constitution.residue_index(
                donor_residue_id
            )
            donor_residue_geometry = donor_structure.geometry.residue_geometry(
                constitution=donor_structure.constitution,
                residue_index=donor_residue_index,
            )
            reconstructed_residue_sites.append(
                donor_structure.constitution.residue_site_at(
                    donor_residue_index
                ).with_residue_id(absent_residue_id)
            )
            reconstructed_residue_geometries.append(donor_residue_geometry)
            reconstructed_residue_formal_charge_payloads.append(
                donor_structure.topology.residue_formal_charge_by_atom_name(
                    constitution=donor_structure.constitution,
                    residue_index=donor_residue_index,
                )
            )
        anchor_residue_id = (
            target.preceding_residue_id
            if target.preceding_residue_id is not None
            else target.following_residue_id
        )
        assert anchor_residue_id is not None
        updated_structure = projected_domain.state.with_inserted_chain_residue_facets(
            anchor_residue_id.chain_id,
            residue_sites=tuple(reconstructed_residue_sites),
            residue_geometries=tuple(reconstructed_residue_geometries),
            residue_formal_charge_payloads=tuple(
                reconstructed_residue_formal_charge_payloads
            ),
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
            provenance_origins=(
                StructureProvenanceOrigin(
                    structure_endpoint=StructureEndpoint.source(target)
                ),
                *tuple(
                    StructureProvenanceOrigin(
                        structure_endpoint=StructureEndpoint.supporting(
                            ResidueSetScope(residue_ids=(donor_residue_id,)),
                            token=(
                                donor_structure.provenance.ingress.source_name
                                if (
                                    donor_structure.provenance.ingress.source_name
                                    is not None
                                )
                                else "supporting-0"
                            ),
                        )
                    )
                    for donor_residue_id in self.reconstruction.donor_residue_ids
                ),
            ),
        )
        return ProjectedCodomainState(
            scope=self.scope,
            state=updated_structure,
            repairs=(repair_event,),
        )
