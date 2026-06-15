"""I/O boundaries for structure-format ingestion and emission."""

from protrepair.io.gemmi_ingress import read_structure, read_structure_string
from protrepair.io.gemmi_writer import write_structure, write_structure_string
from protrepair.structure.provenance import FileFormat

__all__ = [
    "FileFormat",
    "read_structure",
    "read_structure_string",
    "write_structure",
    "write_structure_string",
]
