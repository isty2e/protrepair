"""Workflow-local adaptation of side-chain packing transform results."""

from dataclasses import dataclass

from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.packing.domain import PackingResult
from protrepair.transformer.packing.runtime import execute_sidechain_packing
from protrepair.transformer.packing.spec import PackingSpec


@dataclass(frozen=True, slots=True)
class WorkflowPackingReference:
    """Workflow support artifact derived from one packing transform result."""

    reference_structure: ProteinStructure
    packing_result: PackingResult

    @classmethod
    def from_packing_result(
        cls,
        packing_result: PackingResult,
    ) -> "WorkflowPackingReference":
        """Build one workflow support artifact from one packing transform result."""

        return cls(
            reference_structure=packing_result.packed_structure,
            packing_result=packing_result,
        )


def prepare_workflow_packing_reference(
    structure: ProteinStructure,
    spec: PackingSpec,
) -> WorkflowPackingReference:
    """Execute packing once and adapt the result into one workflow reference."""

    return WorkflowPackingReference.from_packing_result(
        execute_sidechain_packing(structure, spec)
    )
