"""Source-H compatibility with prepared retained-component geometry."""

_SOURCE_PREPARED_BOND_LENGTH_TOLERANCE_ANGSTROM = 0.35


def source_hydrogen_bond_length_is_compatible(
    source_distance: float, prepared_distance: float
) -> bool:
    """Compare source and prepared H-parent lengths without choosing a parent.

    Parameters
    ----------
    source_distance, prepared_distance : float
        H-parent distances in angstroms for the same candidate attachment.

    Returns
    -------
    bool
        The length deviation is within the retained source-H tolerance. This
        alone neither identifies a parent nor certifies a chemical state.
    """
    return (
        abs(source_distance - prepared_distance)
        <= _SOURCE_PREPARED_BOND_LENGTH_TOLERANCE_ANGSTROM
    )
