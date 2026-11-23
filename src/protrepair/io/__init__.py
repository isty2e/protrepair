"""I/O boundaries for structure-format ingestion and emission."""

from protrepair.io.gemmi_ingress import read_structure, read_structure_string
from protrepair.io.gemmi_writer import write_structure, write_structure_string
from protrepair.io.ingress_policy import (
    NonPolymerRetentionPolicy,
    RetentionAction,
    RetentionRule,
)
from protrepair.io.rdkit import to_rdkit_molecule
from protrepair.io.retained_component_definitions import (
    read_retained_component_definitions,
)
from protrepair.structure.constitution import ResidueRole
from protrepair.structure.provenance import FileFormat

__all__ = [
    "FileFormat",
    "NonPolymerRetentionPolicy",
    "ResidueRole",
    "RetentionAction",
    "RetentionRule",
    "read_retained_component_definitions",
    "read_structure",
    "read_structure_string",
    "to_rdkit_molecule",
    "write_structure",
    "write_structure_string",
]
