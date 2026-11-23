"""Named, isolated orthophosphate models from the wwPDB Chemical Component Dictionary."""

from dataclasses import dataclass
from types import MappingProxyType

from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.structure.constitution import (
    ResidueSite,
)
from protrepair.structure.labels import ResidueId

_HEAVY_NAMES = ("P", "O1", "O2", "O3", "O4")
_PROTONATED_OXYGENS = MappingProxyType({"PO4": (), "PI": ("O4",), "2HP": ("O3", "O4")})


def is_orthophosphate_component(component_id: str) -> bool:
    """Recognize the three deposited inorganic orthophosphate component IDs."""
    return component_id in _PROTONATED_OXYGENS


@dataclass(frozen=True, slots=True, init=False)
class DepositedOrthophosphate(RetainedNonPolymerChemistryEvidence):
    """Bind the selected CCD graph to one retained occurrence by atom name.

    Parameters
    ----------
    residue_id : ResidueId
        Retained site to prepare.
    component_id : str
        Deposited CCD identifier: PO4, PI or 2HP.
    """

    component_id: str

    def __init__(self, residue_id: ResidueId, component_id: str) -> None:
        protonated = _PROTONATED_OXYGENS[component_id]
        o2, o3, o4 = (
            "O" if name in protonated else "[O-]" for name in _HEAVY_NAMES[2:]
        )
        RetainedNonPolymerChemistryEvidence.__init__(
            self, residue_id, f"P(=O)({o2})({o3}){o4}", _HEAVY_NAMES
        )
        object.__setattr__(self, "component_id", component_id)

    @classmethod
    def for_residue(cls, residue: ResidueSite) -> "DepositedOrthophosphate | None":
        """Bind a named model for validation rather than skipping a malformed site.

        Parameters
        ----------
        residue : ResidueSite
            Occurrence selected for non-polymer retention.

        Returns
        -------
        DepositedOrthophosphate or None
            The model named by the component, or None for other identifiers.
        """
        if not is_orthophosphate_component(residue.component_id):
            return None
        return cls(residue.residue_id, residue.component_id)
