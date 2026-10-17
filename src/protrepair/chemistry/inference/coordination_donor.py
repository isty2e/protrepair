"""Graph-qualified nitrogen donor chemistry, independent of metal potentials."""

from enum import Enum
from typing import TYPE_CHECKING

from protrepair.chemistry.inference.rdkit_graph import (
    AROMATIC_N_COORDINATION_METALS,
    Chem,
    require_aromatic_nitrogen_donor,
)
from protrepair.errors import UnsupportedCoordinationChemistryError

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


class NitrogenCoordinationDonor(str, Enum):
    """Own the admitted donor grammar and its H and metal compatibility.

    Notes
    -----
    These are preparation capabilities, not oxidation-state predictions or
    force-field types. Primary sulfonamidate support requires an anionic NH
    attached to a neutral sulfonyl sulfur with two oxo and one carbon neighbor.
    """

    AROMATIC = "aromatic"
    PRIMARY_SULFONAMIDATE = "primary_sulfonamidate"

    @property
    def hydrogen_count(self) -> int:
        """Return the total covalently attached H count for this donor role.

        Returns
        -------
        int
            Zero for aromatic donors and one for primary sulfonamidate.
        """
        return int(self is type(self).PRIMARY_SULFONAMIDATE)

    def admits_metal(self, element: str) -> bool:
        """Return whether the selected metal is in this preparation grammar.

        Parameters
        ----------
        element : str
            Canonical uppercase element symbol.

        Returns
        -------
        bool
            Zn for sulfonamidate, or the existing aromatic-N metal set.
        """
        return (
            element == "ZN"
            if self is type(self).PRIMARY_SULFONAMIDATE
            else element in AROMATIC_N_COORDINATION_METALS
        )

    @classmethod
    def from_molecule(
        cls, molecule: "Mol", donor_index: int
    ) -> "NitrogenCoordinationDonor":
        """Qualify a selected native graph without changing its chemistry.

        Parameters
        ----------
        molecule : Mol
            Sanitized selected graph with initialized atom properties.
        donor_index : int
            Native index of the selected donor endpoint.

        Returns
        -------
        NitrogenCoordinationDonor
            The graph-qualified donor role, independent of component labels.

        Raises
        ------
        UnsupportedCoordinationChemistryError
            The selected endpoint is outside the admitted donor grammar.
        RdkitUnavailableError
            Native interpretation is unavailable.
        """
        try:
            require_aromatic_nitrogen_donor(molecule, donor_index)
        except ValueError:
            pass
        else:
            return cls.AROMATIC

        assert Chem is not None
        donor = molecule.GetAtomWithIdx(donor_index)
        bonds = tuple(
            b for b in donor.GetBonds() if b.GetBondType() != Chem.BondType.DATIVE
        )
        heavy = tuple(b for b in bonds if b.GetOtherAtom(donor).GetAtomicNum() != 1)
        if (
            donor.GetAtomicNum() == 7
            and not donor.GetIsAromatic()
            and donor.GetFormalCharge() == -1
            and donor.GetNumRadicalElectrons() == 0
            and donor.GetTotalNumHs(includeNeighbors=True) == 1
            and len(heavy) == 1
            and heavy[0].GetBondTypeAsDouble() == 1
        ):
            sulfur = heavy[0].GetOtherAtom(donor)
            other_bonds = tuple(
                b
                for b in sulfur.GetBonds()
                if b.GetOtherAtom(sulfur).GetIdx() != donor_index
            )
            if (
                sulfur.GetAtomicNum() == 16
                and sulfur.GetFormalCharge() == 0
                and sulfur.GetNumRadicalElectrons() == 0
                and sulfur.GetTotalNumHs(includeNeighbors=True) == 0
                and not sulfur.GetIsAromatic()
                and len(other_bonds) == 3
                and sorted(
                    (b.GetOtherAtom(sulfur).GetAtomicNum(), b.GetBondTypeAsDouble())
                    for b in other_bonds
                )
                == [(6, 1.0), (8, 2.0), (8, 2.0)]
                and all(
                    b.GetOtherAtom(sulfur).GetFormalCharge() == 0
                    and b.GetOtherAtom(sulfur).GetNumRadicalElectrons() == 0
                    for b in other_bonds
                    if b.GetOtherAtom(sulfur).GetAtomicNum() == 8
                )
            ):
                return cls.PRIMARY_SULFONAMIDATE

        raise UnsupportedCoordinationChemistryError(
            "unsupported nitrogen coordination donor chemistry; admitted roles are "
            "unprotonated aromatic N and primary sulfonamidate NH(-1)"
        )
