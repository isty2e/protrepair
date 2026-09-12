"""Integral local graphs, without a preferred protonation state or pH model."""

from enum import Enum

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.microstate.graph import MicrostateGraph
from protrepair.chemistry.microstate.resolution import MicrostateSite
from protrepair.chemistry.standard.components import STANDARD_COMPONENT_ATOMS


class PolymerChemicalSite(str, Enum):
    """Independent protonation sites, not whole-residue microstate products."""

    SIDECHAIN = "sidechain"
    BACKBONE_N = "backbone_n"
    BACKBONE_C = "backbone_c"


class PeptideLinkage(str, Enum):
    """Declared chemical boundary, not geometric chain adjacency."""

    LINKED = "linked"
    FREE = "free"
    UNKNOWN = "unknown"


def standard_microstate_candidates(
    component_id: str,
    site: PolymerChemicalSite,
    linkage: PeptideLinkage = PeptideLinkage.UNKNOWN,
) -> MicrostateSite | None:
    """Build supported site graphs with their external valence accounted for.

    Parameters
    ----------
    component_id : str
        Canonical standard component ID; template applicability is checked by
        PolymerMicrostateSite, not by this catalog builder.
    site : PolymerChemicalSite
        Side chain, backbone N or backbone C.
    linkage : PeptideLinkage
        Explicit backbone boundary. Unknown boundaries have no catalog.

    Returns
    -------
    MicrostateSite or None
        None for an unsupported side chain or unknown backbone boundary.

    Raises
    ------
    TypeError
        Site or linkage is not a canonical enum member.
    ValueError
        A backbone linkage is supplied for a side-chain site.
    """
    if not isinstance(site, PolymerChemicalSite) or not isinstance(
        linkage, PeptideLinkage
    ):
        raise TypeError("site and linkage must be canonical enum members")
    if site is PolymerChemicalSite.SIDECHAIN and linkage is not PeptideLinkage.UNKNOWN:
        raise ValueError("side-chain sites do not have a backbone linkage")
    if component_id not in STANDARD_COMPONENT_ATOMS:
        return None
    if site is not PolymerChemicalSite.SIDECHAIN:
        if linkage is PeptideLinkage.UNKNOWN:
            return None
        if site is PolymerChemicalSite.BACKBONE_N:
            boundary = (
                1 + int(component_id == "PRO") + int(linkage is PeptideLinkage.LINKED)
            )
            return MicrostateSite(
                tuple(
                    MicrostateGraph.from_heavy_graph(
                        elements={"N": "N"},
                        bonds=(),
                        charges={"N": charge},
                        boundary_orders={"N": boundary},
                    )
                    for charge in ((0,) if linkage is PeptideLinkage.LINKED else (0, 1))
                )
            )
        if linkage is PeptideLinkage.LINKED:
            return MicrostateSite(
                (
                    MicrostateGraph.from_heavy_graph(
                        elements={"C": "C", "O": "O"},
                        bonds=(BondDefinition("C", "O", order=2),),
                        charges={},
                        boundary_orders={"C": 2},
                    ),
                )
            )
        return _carboxyl_site("C", "O", "OXT")

    if component_id in {"ASP", "GLU"}:
        return _carboxyl_site(
            *(("CG", "OD1", "OD2") if component_id == "ASP" else ("CD", "OE1", "OE2"))
        )
    if component_id == "LYS":
        return MicrostateSite(
            tuple(
                MicrostateGraph.from_heavy_graph(
                    elements={"NZ": "N"},
                    bonds=(),
                    charges={"NZ": charge},
                    boundary_orders={"NZ": 1},
                )
                for charge in (0, 1)
            )
        )
    if component_id == "ARG":
        # Protonation is independent of localization of the guanidinium charge.
        return MicrostateSite(
            tuple(
                MicrostateGraph.from_heavy_graph(
                    elements={"NE": "N", "CZ": "C", "NH1": "N", "NH2": "N"},
                    bonds=tuple(
                        BondDefinition("CZ", name, order=2 if name == imine else 1)
                        for name in ("NE", "NH1", "NH2")
                    ),
                    charges={imine: charge},
                    boundary_orders={"NE": 1},
                )
                for charge in (0, 1)
                for imine in ("NH2", "NH1", "NE")
            )
        )
    if component_id == "HIS":
        return MicrostateSite(
            tuple(
                MicrostateGraph.from_heavy_graph(
                    elements={
                        "CG": "C",
                        "ND1": "N",
                        "CE1": "C",
                        "NE2": "N",
                        "CD2": "C",
                    },
                    bonds=(
                        BondDefinition("CG", "ND1"),
                        BondDefinition("ND1", "CE1", order=2 if imine == "ND1" else 1),
                        BondDefinition("CE1", "NE2", order=2 if imine == "NE2" else 1),
                        BondDefinition("NE2", "CD2"),
                        BondDefinition("CD2", "CG", order=2),
                    ),
                    charges={imine: charge},
                    boundary_orders={"CG": 1},
                )
                for charge in (0, 1)
                for imine in ("NE2", "ND1")
            )
        )
    return None


def _carboxyl_site(carbon: str, oxygen_1: str, oxygen_2: str) -> MicrostateSite:
    return MicrostateSite(
        tuple(
            MicrostateGraph.from_heavy_graph(
                elements={carbon: "C", oxygen_1: "O", oxygen_2: "O"},
                bonds=(
                    BondDefinition(carbon, double_oxygen, order=2),
                    BondDefinition(carbon, single_oxygen),
                ),
                charges={single_oxygen: charge},
                boundary_orders={carbon: 1},
            )
            for charge in (0, -1)
            for double_oxygen, single_oxygen in (
                (oxygen_1, oxygen_2),
                (oxygen_2, oxygen_1),
            )
        )
    )
