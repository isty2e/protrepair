"""Internal support policy for bundled retained non-polymer chemistry."""

SUPPORTED_BUNDLED_RETAINED_NON_POLYMER_HYDROGENATION_COMPONENT_IDS: frozenset[str] = (
    frozenset(
        {
            "ACE",
            "CMP",
            "COA",
            "FAD",
            "FMN",
            "HEM",
            "MAN",
            "NAD",
            "NAG",
            "NAP",
            "PLP",
            "SAH",
            "SAM",
            "SAN",
            "SO4",
        }
    )
)


def supports_bundled_retained_non_polymer_hydrogenation(
    component_id: str,
) -> bool:
    """Return whether one bundled retained non-polymer supports H placement."""

    return (
        component_id.strip().upper()
        in SUPPORTED_BUNDLED_RETAINED_NON_POLYMER_HYDROGENATION_COMPONENT_IDS
    )
