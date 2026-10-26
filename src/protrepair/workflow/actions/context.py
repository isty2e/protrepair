"""Execution dependency context shared by concrete transformer actions."""

from dataclasses import dataclass, field

from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.microstate.preparation import AminoAcidPreparation
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
    """Execution dependencies shared by process-transformer invocations.

    Parameters
    ----------
    component_library : ComponentLibrary
        Active component definitions.
    original_structure : ProteinStructure
        Original workflow input for source-sensitive transformations.
    orphan_fragment_policy : OrphanFragmentPolicy
        Handling of disconnected heavy fragments.
    reference_structure : ProteinStructure or None
        Optional reconstruction donor.
    histidine_protonation : HistidineProtonationRequest
        New HIS selection, or no new request.
    allow_retained_non_polymer_rdkit_fallback : bool
        Whether retained-component preparation may infer chemistry.
    retained_non_polymer_chemistry_evidence : tuple
        RetainedNonPolymerChemistryEvidence values for explicit interpretation.
    polymer_preparation : AminoAcidPreparation or None
        Optional non-semantic seed. Consumers rebind it to their exact current
        snapshot, library and requests; it is not proposal-time authority.
    """

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
    polymer_preparation: AminoAcidPreparation | None = field(
        default=None, compare=False, repr=False
    )
