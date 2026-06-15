"""Internal support policy for bundled nonstandard residue repair."""

SUPPORTED_NOW_BUNDLED_NONSTANDARD_COMPONENT_IDS: frozenset[str] = frozenset(
    {
        "HYP",
    }
)
SUPPORTED_WITH_REFINEMENT_BUNDLED_NONSTANDARD_COMPONENT_IDS: frozenset[str] = frozenset(
    {
        "CSO",
        "PTR",
        "SEP",
        "TPO",
    }
)
SUPPORTED_REFERENCE_OR_SALVAGE_BUNDLED_NONSTANDARD_COMPONENT_IDS: frozenset[str] = (
    frozenset(
        {
            "MSE",
        }
    )
)
SUPPORTED_BUNDLED_NONSTANDARD_HYDROGENATION_COMPONENT_IDS: frozenset[str] = frozenset(
    {
        "CSO",
        "HYP",
        "MSE",
        "PTR",
        "SEP",
        "TPO",
    }
)


def supports_bundled_nonstandard_template_repair(component_id: str) -> bool:
    """Return whether one bundled nonstandard component is baseline-repairable."""

    return (
        component_id.strip().upper() in SUPPORTED_NOW_BUNDLED_NONSTANDARD_COMPONENT_IDS
    )


def supports_bundled_nonstandard_refinement_required_repair(
    component_id: str,
) -> bool:
    """Return whether one bundled component repairs only with explicit refinement."""

    return (
        component_id.strip().upper()
        in SUPPORTED_WITH_REFINEMENT_BUNDLED_NONSTANDARD_COMPONENT_IDS
    )


def supports_bundled_nonstandard_reference_or_salvage_repair(
    component_id: str,
) -> bool:
    """Return whether one bundled component supports narrow reference/salvage repair."""

    return (
        component_id.strip().upper()
        in SUPPORTED_REFERENCE_OR_SALVAGE_BUNDLED_NONSTANDARD_COMPONENT_IDS
    )


def supports_bundled_nonstandard_heavy_repair(component_id: str) -> bool:
    """Return whether one bundled nonstandard component has any heavy-repair path."""

    normalized_component_id = component_id.strip().upper()
    return (
        normalized_component_id in SUPPORTED_NOW_BUNDLED_NONSTANDARD_COMPONENT_IDS
        or normalized_component_id
        in SUPPORTED_WITH_REFINEMENT_BUNDLED_NONSTANDARD_COMPONENT_IDS
        or normalized_component_id
        in SUPPORTED_REFERENCE_OR_SALVAGE_BUNDLED_NONSTANDARD_COMPONENT_IDS
    )


def supports_bundled_nonstandard_hydrogenation(component_id: str) -> bool:
    """Return whether one bundled nonstandard component supports H placement."""

    return (
        component_id.strip().upper()
        in SUPPORTED_BUNDLED_NONSTANDARD_HYDROGENATION_COMPONENT_IDS
    )
