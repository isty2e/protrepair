"""HIS state-specific targets: Malinska et al. (2015), Table 4, CSD columns.

DOI: 10.1107/S1399004715007816. Distances/ESDs are in angstroms; angles
and their ESDs are in degrees. These are geometry targets, not FF parameters.
"""

from protrepair.chemistry.microstate.catalog import (
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
    ResidueRestraintTemplate,
    SidechainRestraintVariant,
)


def histidine_restraint_template() -> ResidueRestraintTemplate:
    """Build state-independent scaffold and three protonation-specific variants.

    Returns
    -------
    ResidueRestraintTemplate
        Conditional HIS dictionary; callers bind it to current chemical evidence.
    """
    catalog = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert catalog is not None
    # Columns are delta, epsilon, then doubly protonated; parentheses in the
    # manuscript express ESD in units of the final decimal digit.
    bonds = (
        ("CG", "ND1", ((1.371, 0.009), (1.382, 0.009), (1.379, 0.007))),
        ("ND1", "CE1", ((1.338, 0.008), (1.320, 0.008), (1.325, 0.008))),
        ("CE1", "NE2", ((1.316, 0.009), (1.335, 0.014), (1.316, 0.009))),
        ("NE2", "CD2", ((1.377, 0.007), (1.367, 0.009), (1.373, 0.007))),
        ("CD2", "CG", ((1.356, 0.009), (1.359, 0.010), (1.353, 0.007))),
        ("CB", "CG", ((1.477, 0.019), (1.481, 0.022), (1.490, 0.011))),
    )
    angles = (
        ("ND1", "CG", "CD2", ((105.1, 0.5), (109.1, 0.8), (105.9, 0.5))),
        ("CG", "ND1", "CE1", ((108.1, 0.5), (105.3, 0.9), (109.3, 0.5))),
        ("ND1", "CE1", "NE2", ((111.3, 0.6), (111.9, 1.0), (108.4, 0.5))),
        ("CE1", "NE2", "CD2", ((105.3, 0.6), (107.2, 0.8), (108.9, 0.6))),
        ("NE2", "CD2", "CG", ((110.3, 0.6), (106.5, 0.7), (107.4, 0.5))),
        ("CB", "CG", "ND1", ((123.4, 1.6), (121.6, 1.3), (122.7, 1.2))),
        ("CB", "CG", "CD2", ((131.5, 1.6), (129.3, 1.4), (131.3, 1.3))),
    )
    variants = []
    for column, h_counts in enumerate(((1, 0), (0, 1), (1, 1))):
        graph = next(
            g
            for g in catalog.candidates
            if (g.atom("ND1").hydrogens, g.atom("NE2").hydrogens) == h_counts
        )
        variants.append(
            SidechainRestraintVariant(
                graph,
                tuple(
                    BondRestraintTarget(a, b, *values[column]) for a, b, values in bonds
                ),
                tuple(
                    AngleRestraintTarget(a, b, c, *values[column])
                    for a, b, c, values in angles
                ),
            )
        )
    return ResidueRestraintTemplate(
        component_id="HIS",
        bond_targets=(
            BondRestraintTarget("N", "CA", 1.458, 0.019),
            BondRestraintTarget("CA", "CB", 1.530, 0.020),
            BondRestraintTarget("CA", "C", 1.525, 0.021),
            BondRestraintTarget("C", "O", 1.231, 0.020),
        ),
        angle_targets=(
            AngleRestraintTarget("CB", "CA", "C", 110.1, 1.9),
            AngleRestraintTarget("N", "CA", "CB", 110.5, 1.7),
            AngleRestraintTarget("CA", "CB", "CG", 113.8, 1.0),
            AngleRestraintTarget("N", "CA", "C", 111.2, 2.8),
            AngleRestraintTarget("CA", "C", "O", 120.8, 1.7),
        ),
        sidechain_variants=tuple(variants),
    )
