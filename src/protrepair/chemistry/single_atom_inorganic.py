"""Single-atom inorganic residue classification policy."""

from protrepair.structure.constitution import ResidueSite

KNOWN_SINGLE_ATOM_INORGANIC_ELEMENTS: frozenset[str] = frozenset(
    {
        "AG",
        "AL",
        "BA",
        "CA",
        "CD",
        "CL",
        "CO",
        "CU",
        "FE",
        "HG",
        "K",
        "LI",
        "MG",
        "MN",
        "NA",
        "NI",
        "RB",
        "SR",
        "ZN",
    }
)


def is_single_atom_inorganic_residue(residue_site: ResidueSite) -> bool:
    """Return whether one residue is a known single-atom inorganic species."""

    if len(residue_site.atom_sites) != 1:
        return False

    return residue_site.atom_sites[0].element in KNOWN_SINGLE_ATOM_INORGANIC_ELEMENTS
