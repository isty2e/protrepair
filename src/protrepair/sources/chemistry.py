"""External chemistry override DTOs for retained non-polymer components."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.structure.labels import ResidueId


@dataclass(frozen=True, slots=True)
class RetainedNonPolymerChemistryOverride:
    """External chemistry truth for one retained non-polymer residue."""

    residue_id: ResidueId
    smiles: str
    heavy_atom_names: tuple[str, ...]

    def __post_init__(self) -> None:
        evidence = RetainedNonPolymerChemistryEvidence(
            residue_id=self.residue_id,
            smiles=self.smiles,
            heavy_atom_names=self.heavy_atom_names,
        )
        object.__setattr__(self, "smiles", evidence.smiles)
        object.__setattr__(self, "heavy_atom_names", evidence.heavy_atom_names)

    def to_evidence(self) -> RetainedNonPolymerChemistryEvidence:
        """Normalize this boundary DTO into canonical chemistry evidence."""

        return RetainedNonPolymerChemistryEvidence(
            residue_id=self.residue_id,
            smiles=self.smiles,
            heavy_atom_names=self.heavy_atom_names,
        )


def override_by_residue_id(
    overrides: Iterable[RetainedNonPolymerChemistryOverride],
) -> Mapping[ResidueId, RetainedNonPolymerChemistryOverride]:
    """Return residue-keyed retained non-polymer overrides."""

    override_map: dict[ResidueId, RetainedNonPolymerChemistryOverride] = {}
    for override in overrides:
        if override.residue_id in override_map:
            raise ValueError(
                "retained non-polymer chemistry overrides must not repeat "
                f"{override.residue_id.display_token()}"
            )
        override_map[override.residue_id] = override

    return MappingProxyType(override_map)
