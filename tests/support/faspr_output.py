"""Independent transport labels for injected FASPR output fixtures."""

from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId


def relabel_faspr_output(pdb_text: str, source: ProteinStructure) -> str:
    labels = {
        residue.residue_id: f"A{ordinal - 999:4d} "
        for ordinal, residue in enumerate(source.constitution.iter_residues())
    }
    lines = []
    for line in pdb_text.splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")):
            source_id = ResidueId(line[20:22].strip(), int(line[22:26]), line[26])
            label = labels.get(source_id)
            if label is not None:
                line = line[:20] + " " + label + line[27:]
        lines.append(line)
    return "".join(lines)
