"""External source DTOs and retrieval boundaries."""

from protrepair.sources.alphafold import (
    AlphaFoldFetchFailureKind,
    AlphaFoldModelFetchFailure,
    AlphaFoldModelFetchOutcome,
    AlphaFoldModelRecord,
    AlphaFoldModelSet,
    AlphaFoldStructureArtifact,
    AlphaFoldStructureFetchFailure,
    AlphaFoldStructureFetchOutcome,
)
from protrepair.sources.alphafold_retrieval import (
    fetch_alphafold_model_set,
    fetch_alphafold_structure_artifact,
)
from protrepair.sources.chemistry import (
    RetainedNonPolymerChemistryOverride,
    override_by_residue_id,
)
from protrepair.sources.uniprot import (
    UniProtSequenceFamily,
    UniProtSequenceFamilyFetchOutcome,
    UniProtSequenceFamilyFetchResult,
    UniProtSequenceFetchFailure,
    UniProtSequenceFetchFailureKind,
    UniProtSequenceFetchOutcome,
    UniProtSequenceRecord,
    UniProtSequenceReference,
)
from protrepair.sources.uniprot_retrieval import (
    fetch_uniprot_sequence,
    fetch_uniprot_sequence_family,
)

__all__ = [
    "AlphaFoldFetchFailureKind",
    "AlphaFoldModelFetchFailure",
    "AlphaFoldModelFetchOutcome",
    "AlphaFoldModelRecord",
    "AlphaFoldModelSet",
    "AlphaFoldStructureArtifact",
    "AlphaFoldStructureFetchFailure",
    "AlphaFoldStructureFetchOutcome",
    "RetainedNonPolymerChemistryOverride",
    "UniProtSequenceFamily",
    "UniProtSequenceFamilyFetchOutcome",
    "UniProtSequenceFamilyFetchResult",
    "UniProtSequenceFetchFailure",
    "UniProtSequenceFetchFailureKind",
    "UniProtSequenceFetchOutcome",
    "UniProtSequenceRecord",
    "UniProtSequenceReference",
    "fetch_alphafold_model_set",
    "fetch_alphafold_structure_artifact",
    "fetch_uniprot_sequence",
    "fetch_uniprot_sequence_family",
    "override_by_residue_id",
]
