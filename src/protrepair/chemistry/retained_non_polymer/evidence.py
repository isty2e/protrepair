"""Canonical retained non-polymer chemistry evidence."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from protrepair.structure.labels import ResidueId

__all__ = [
    "RetainedNonPolymerChemistryEvidence",
    "evidence_by_residue_id",
]


@dataclass(frozen=True, slots=True)
class RetainedNonPolymerChemistryEvidence:
    """Canonical chemistry evidence for one retained non-polymer residue."""

    residue_id: ResidueId
    smiles: str
    heavy_atom_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.residue_id, ResidueId):
            raise TypeError("residue_id must be a ResidueId")

        normalized_smiles = self.smiles.strip()
        normalized_heavy_atom_names = tuple(
            atom_name.strip().upper() for atom_name in self.heavy_atom_names
        )
        if not normalized_smiles:
            raise ValueError(
                "retained non-polymer chemistry evidence smiles must not be blank"
            )
        if not normalized_heavy_atom_names:
            raise ValueError(
                "retained non-polymer chemistry evidence heavy_atom_names must "
                "not be empty"
            )
        if any(not atom_name for atom_name in normalized_heavy_atom_names):
            raise ValueError(
                "retained non-polymer chemistry evidence heavy_atom_names must "
                "not be blank"
            )
        if len(normalized_heavy_atom_names) != len(
            set(normalized_heavy_atom_names)
        ):
            raise ValueError(
                "retained non-polymer chemistry evidence heavy_atom_names must "
                "be unique"
            )

        object.__setattr__(self, "smiles", normalized_smiles)
        object.__setattr__(self, "heavy_atom_names", normalized_heavy_atom_names)


def evidence_by_residue_id(
    evidence: Iterable[RetainedNonPolymerChemistryEvidence],
) -> Mapping[ResidueId, RetainedNonPolymerChemistryEvidence]:
    """Return residue-keyed retained non-polymer chemistry evidence."""

    evidence_map: dict[ResidueId, RetainedNonPolymerChemistryEvidence] = {}
    for item in evidence:
        if item.residue_id in evidence_map:
            raise ValueError(
                "retained non-polymer chemistry evidence must not repeat "
                f"{item.residue_id.display_token()}"
            )
        evidence_map[item.residue_id] = item

    return MappingProxyType(evidence_map)
