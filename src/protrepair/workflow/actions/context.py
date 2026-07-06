"""Execution dependency context shared by concrete transformer actions."""

from dataclasses import dataclass, field

from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.completion.hydrogen.protonation import (
    DisabledHistidineProtonationRequest,
    HistidineProtonationRequest,
)
from protrepair.transformer.completion.policies import OrphanFragmentPolicy


@dataclass(frozen=True, slots=True)
class TransformerExecutionContext:
    """Execution dependencies shared by process-transformer invocations."""

    component_library: ComponentLibrary
    original_structure: ProteinStructure
    orphan_fragment_policy: OrphanFragmentPolicy
    reference_structure: ProteinStructure | None = None
    histidine_protonation: HistidineProtonationRequest = field(
        default_factory=DisabledHistidineProtonationRequest
    )
    allow_retained_non_polymer_rdkit_fallback: bool = True
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = ()
