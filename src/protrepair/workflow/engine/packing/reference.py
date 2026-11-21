"""Workflow-local adaptation of side-chain packing transform results."""

from dataclasses import dataclass, replace

from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.diagnostics.water_contacts import stationary_water_refusal
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.packing.domain import PackingResult
from protrepair.transformer.packing.runtime import execute_sidechain_packing
from protrepair.transformer.packing.spec import PackingSpec


@dataclass(frozen=True, slots=True)
class WorkflowPackingReference:
    """Workflow support artifact derived from one packing transform result."""

    reference_structure: ProteinStructure | None
    packing_result: PackingResult

    @classmethod
    def from_packing_result(
        cls,
        packing_result: PackingResult,
    ) -> "WorkflowPackingReference":
        """Build one workflow support artifact from one packing transform result."""

        return cls(
            reference_structure=None
            if any(
                issue.kind is ValidationIssueKind.RETAINED_WATER_REPAIR_REFUSED
                for issue in packing_result.issues
            )
            else packing_result.packed_structure,
            packing_result=packing_result,
        )


def prepare_workflow_packing_reference(
    structure: ProteinStructure,
    spec: PackingSpec,
) -> WorkflowPackingReference:
    """Execute packing once and adapt the result into one workflow reference."""

    result = execute_sidechain_packing(structure, spec)
    refusal = stationary_water_refusal(structure, result.packed_structure)
    if refusal is not None:
        result = replace(
            result,
            packed_structure=structure,
            changed_residue_ids=(),
            issues=(*result.issues, refusal),
        )
    return WorkflowPackingReference.from_packing_result(result)
