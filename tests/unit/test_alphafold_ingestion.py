"""AlphaFold artifact ingestion tests."""

from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.geometry import Vec3
from protrepair.io.gemmi_writer import write_structure_string
from protrepair.sources import (
    AlphaFoldModelRecord,
    AlphaFoldStructureArtifact,
    UniProtSequenceReference,
)
from protrepair.structure import (
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
)
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.source_ingestion import ingest_alphafold_structure_artifact


def test_ingest_alphafold_structure_artifact_attaches_polymer_blueprint() -> None:
    """AlphaFold artifact ingestion should attach one structure-aligned blueprint."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="fixture",
    )
    artifact = AlphaFoldStructureArtifact(
        model=AlphaFoldModelRecord(
            uniprot_reference=UniProtSequenceReference(accession="P12345"),
            entry_id="AF-P12345-F1",
            model_entity_id="1",
            provider_id="AlphaFoldDB",
            tool_used="AlphaFold Monomer",
            sequence="G",
            pdb_url="https://example.org/model.pdb",
        ),
        file_format=FileFormat.PDB,
        structure_text=write_structure_string(structure, FileFormat.PDB),
        source_url="https://example.org/model.pdb",
    )

    canonical = ingest_alphafold_structure_artifact(artifact)

    assert canonical.polymer_blueprint == PolymerBlueprint(
        chains=(
            PolymerChainBlueprint(
                chain_id="A",
                residue_slots=(
                    PolymerResidueSlot(sequence_position=1, token="G"),
                ),
            ),
        )
    )
