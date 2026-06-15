"""Source-aware artifact ingestion into canonical structure carriers."""

from protrepair.io.gemmi_ingress import read_structure_string
from protrepair.io.ingress_policy import StructureNormalizationPolicy
from protrepair.sources.alphafold import AlphaFoldStructureArtifact
from protrepair.sources.projection import (
    polymer_blueprint_from_alphafold_model,
)
from protrepair.structure.aggregate import ProteinStructure


def ingest_alphafold_structure_artifact(
    artifact: AlphaFoldStructureArtifact,
    *,
    policy: StructureNormalizationPolicy | None = None,
) -> ProteinStructure:
    """Materialize one canonical structure from one AlphaFold source artifact."""

    if not isinstance(artifact, AlphaFoldStructureArtifact):
        raise TypeError(
            "ingest_alphafold_structure_artifact requires an "
            "AlphaFoldStructureArtifact value"
        )

    structure = read_structure_string(
        artifact.structure_text,
        artifact.file_format,
        policy=policy,
        source_name=artifact.model.entry_id,
    )
    chain_ids = structure.constitution.chain_ids()
    if len(chain_ids) != 1:
        return structure

    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=structure.topology,
        polymer_blueprint=polymer_blueprint_from_alphafold_model(
            artifact.model,
            chain_id=chain_ids[0],
        ),
        provenance=structure.provenance,
    )
