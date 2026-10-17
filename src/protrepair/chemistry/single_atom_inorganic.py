"""Hydrogen applicability of retained atomic metal and halogen sites."""

from dataclasses import dataclass

try:
    from rdkit import Chem
    from rdkit.Chem import rdqueries
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None
    rdqueries = None

from protrepair.chemistry.retained_non_polymer.covalent_context import (
    RetainedCovalentContext,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite, ResidueSite
from protrepair.structure.topology import BondRelationshipType

_HALOGEN_ATOMIC_NUMBERS = frozenset({9, 17, 35, 53, 85, 117})


@dataclass(frozen=True, slots=True)
class RetainedAtomicSite:
    """Separate atomic inventory from molecular valence completion.

    Parameters
    ----------
    residue : ResidueSite
        Current retained inventory. Element identity, not component or atom name,
        determines the metal/halogen scope.

    Notes
    -----
    Zero-H applicability does not select charge, oxidation state or spin.
    Templates and mapped molecular evidence take precedence over this rule.
    """

    residue: ResidueSite

    def is_monatomic_ion(self) -> bool:
        """Identify a single metal/halogen atom, without assigning its charge.

        Returns
        -------
        bool
            True only for a one-atom inventory in this atomic preparation scope.
            Unknown elements are not classified as ions.

        Raises
        ------
        RdkitUnavailableError
            Native element classification is unavailable.
        """
        if len(self.residue.atom_sites) != 1:
            return False
        try:
            return self._ion_atom() is not None
        except ValueError:
            return False

    def require_unambiguous_hydrogenation(self) -> None:
        """Refuse to infer a hydride or hydrogen halide from one heavy atom.

        Raises
        ------
        ValueError
            H/D/T accompanies a single metal/halogen heavy atom; molecular
            template or mapped evidence is required instead of atomic inference.
            An unknown atomic element also prevents inference.
        RdkitUnavailableError
            Native element classification is unavailable.
        """
        if self._ion_atom() is not None and len(self.residue.atom_sites) != 1:
            raise ValueError(
                "single-heavy-atom metal/halogen hydrogen chemistry requires "
                "a molecular template or mapped evidence"
            )

    def require_source_context(self, structure: ProteinStructure) -> None:
        """Admit atomic preparation without hiding original molecular evidence.

        Parameters
        ----------
        structure : ProteinStructure
            Snapshot retaining original observations and current typed links.

        Raises
        ------
        ValueError
            The current inventory is not monatomic, an unexplained attachment
            exists, or source atoms/links require molecular chemistry.
        RdkitUnavailableError
            Native element classification is unavailable.
        """
        if not self.is_monatomic_ion():
            raise ValueError("atomic preparation requires a monatomic inventory")
        RetainedCovalentContext(structure, self.residue.residue_id).require_admitted()
        observation = structure.provenance.ingress.observation
        if observation is None:
            return

        original = observation.constitution.residue_or_ligand(self.residue.residue_id)
        if original is not None and original.atom_sites != self.residue.atom_sites:
            raise ValueError("source molecular inventory forbids atomic preparation")
        for bond in observation.bonds_for_residue(self.residue.residue_id):
            if bond.relationship_type not in {
                BondRelationshipType.METAL_COORDINATION,
                BondRelationshipType.HYDROGEN_BOND,
            }:
                raise ValueError(
                    "source molecular connectivity forbids atomic preparation"
                )

    def _ion_atom(self) -> AtomSite | None:
        heavy_atoms = tuple(
            atom for atom in self.residue.atom_sites if not atom.is_hydrogen()
        )
        if len(heavy_atoms) != 1:
            return None
        if Chem is None or rdqueries is None:
            raise RdkitUnavailableError("atomic inventory requires operational RDKit")
        site = heavy_atoms[0]
        try:
            atom = Chem.Atom(site.element.capitalize())
        except RuntimeError as error:
            raise ValueError(f"unsupported atomic element {site.element}") from error
        return (
            site
            if (
                atom.GetAtomicNum() in _HALOGEN_ATOMIC_NUMBERS
                or rdqueries.MAtomQueryAtom().Match(atom)
            )
            else None
        )
