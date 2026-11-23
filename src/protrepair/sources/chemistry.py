"""External chemistry override DTOs for retained non-polymer components."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar

from protrepair.chemistry.retained_non_polymer.definition import (
    RetainedComponentAssignment,
)
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


@dataclass(frozen=True, slots=True)
class SmilesAssignment:
    """Specify chemistry for one site without an ordered list of atom names.

    Parameters
    ----------
    residue_id : ResidueId
        Exact address of a ligand kept by the ingress policy.
    smiles : str
        Molecular graph, including the intended protonation and any declared
        stereochemistry. The workflow binds it to the normalized input.
    """

    residue_id: ResidueId
    smiles: str

    def __post_init__(self) -> None:
        if not isinstance(self.residue_id, ResidueId):
            raise TypeError("residue_id must be a ResidueId")
        if not isinstance(self.smiles, str):
            raise TypeError("smiles must be a string")
        if not self.smiles.strip():
            raise ValueError("SMILES must not be blank")
        object.__setattr__(self, "smiles", self.smiles.strip())


RetainedChemistryOverride = (
    RetainedNonPolymerChemistryOverride | RetainedComponentAssignment | SmilesAssignment
)
_Override = TypeVar("_Override", bound=RetainedChemistryOverride)


def override_by_residue_id(
    overrides: Iterable[_Override],
) -> Mapping[ResidueId, _Override]:
    """Return residue-keyed retained non-polymer overrides."""

    override_map: dict[ResidueId, _Override] = {}
    for override in overrides:
        if override.residue_id in override_map:
            raise ValueError(
                "retained non-polymer chemistry overrides must not repeat "
                f"{override.residue_id.display_token()}"
            )
        override_map[override.residue_id] = override

    return MappingProxyType(override_map)
