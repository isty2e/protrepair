"""Caller constraints on a protonation class, independent of resonance drawing."""

from collections.abc import Mapping
from dataclasses import dataclass

from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.resolution import MicrostateConstraints
from protrepair.structure.labels import ResidueId


@dataclass(frozen=True, slots=True, init=False)
class SiteProtonation:
    """Request a whole-site charge and, optionally, H counts on named parents.

    Parameters
    ----------
    residue_id : ResidueId
        Canonical input residue, including chain and insertion code.
    site : PolymerChemicalSite
        Side chain, backbone N or backbone C of that residue.
    site_charge : int
        Net formal charge of this chemical site.
    parent_hydrogens : Mapping[str, int], optional
        Exact counts, including D and T, on the specified heavy atoms. Omitted
        parents remain unconstrained. The mapping is copied at construction.

    Raises
    ------
    TypeError
        An identity, charge, mapping or count has the wrong type.
    ValueError
        An atom name is not canonical or a count is negative.
    """

    residue_id: ResidueId
    site: PolymerChemicalSite
    site_charge: int
    parent_hydrogens: tuple[tuple[str, int], ...]

    def __init__(
        self,
        residue_id: ResidueId,
        site: PolymerChemicalSite,
        *,
        site_charge: int,
        parent_hydrogens: Mapping[str, int] | None = None,
    ) -> None:
        if not isinstance(residue_id, ResidueId) or not isinstance(
            site, PolymerChemicalSite
        ):
            raise TypeError(
                "site protonation requires a residue ID and a polymer chemical site"
            )
        if type(site_charge) is not int:
            raise TypeError("site_charge must be an integer")
        if parent_hydrogens is not None and not isinstance(parent_hydrogens, Mapping):
            raise TypeError("parent_hydrogens must map atom names to counts")
        pairs = tuple(parent_hydrogens.items()) if parent_hydrogens is not None else ()
        if any(not isinstance(name, str) for name, _ in pairs):
            raise TypeError("hydrogen parent names must be strings")
        parents = MicrostateConstraints(hydrogens=pairs)
        object.__setattr__(self, "residue_id", residue_id)
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "site_charge", site_charge)
        object.__setattr__(self, "parent_hydrogens", parents.hydrogens)
