"""Closed-shell local graphs for standard polymer microstate decisions."""

from collections.abc import Mapping
from dataclasses import dataclass

from protrepair.chemistry.component.graph import BondDefinition


@dataclass(frozen=True, slots=True)
class MicrostateAtom:
    """Own one site's atom charge, attached H count and boundary valence.

    Parameters
    ----------
    name : str
        Canonical heavy-atom name.
    element : str
        C, N or O; this is a closed-shell site model, not a general valence table.
    charge : int
        Integral formal charge, not partial force-field charge.
    hydrogens : int
        Required number of attached hydrogens, including isotopes.
    boundary_order : int
        Sum of orders to atoms outside this site; those bonds are not edited.

    Raises
    ------
    TypeError
        Counts or charge are not integers.
    ValueError
        Names, element/charge combinations or counts are unsupported.
    """

    name: str
    element: str
    charge: int
    hydrogens: int
    boundary_order: int

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip().upper():
            raise ValueError("microstate atom names must be canonical")
        for value in (self.charge, self.hydrogens, self.boundary_order):
            if type(value) is not int:
                raise TypeError("microstate charges and counts must be integers")
        if self.hydrogens < 0 or self.boundary_order < 0:
            raise ValueError("microstate atom counts must be nonnegative")
        self.closed_shell_valence()

    def closed_shell_valence(self) -> int:
        """Return this model's allowed C/N/O valence.

        Returns
        -------
        int
            Closed-shell bond-order sum for the supported element and charge.

        Raises
        ------
        ValueError
            The element/charge pair is outside the supported site chemistry.
        """
        if self.element == "C" and self.charge == 0:
            return 4
        if self.element == "N" and self.charge in (0, 1):
            return 3 + self.charge
        if self.element == "O" and self.charge in (-1, 0):
            return 2 + self.charge
        raise ValueError("unsupported closed-shell microstate element/charge")


@dataclass(frozen=True, slots=True)
class MicrostateGraph:
    """Keep coupled H, charge and integral bonds valid as one chemical graph.

    Parameters
    ----------
    atoms : tuple[MicrostateAtom, ...]
        Complete local heavy atoms with explicit boundary bond-order sums.
    bonds : tuple[BondDefinition, ...]
        Internal single/double bonds in an integral Kekule representation.

    Raises
    ------
    ValueError
        Atoms/bonds repeat, endpoints are absent, the site is disconnected, or
        closed-shell valence fails.
    """

    atoms: tuple[MicrostateAtom, ...]
    bonds: tuple[BondDefinition, ...]

    def __post_init__(self) -> None:
        atoms = tuple(sorted(self.atoms, key=lambda atom: atom.name))
        by_name = {atom.name: atom for atom in atoms}
        if not atoms or len(by_name) != len(atoms):
            raise ValueError("microstate graph needs unique atoms")
        orders = {atom.name: atom.boundary_order + atom.hydrogens for atom in atoms}
        bonds_by_pair: dict[tuple[str, str], BondDefinition] = {}
        for bond in self.bonds:
            first, second = sorted((bond.atom_name_1, bond.atom_name_2))
            pair = (first, second)
            if (
                pair[0] == pair[1]
                or pair in bonds_by_pair
                or not set(pair) <= by_name.keys()
                or type(bond.order) is not int
                or bond.order not in (1, 2)
                or bond.aromatic
            ):
                raise ValueError("microstate graph requires unique integral bonds")
            bonds_by_pair[pair] = BondDefinition(*pair, order=bond.order)
            for name in pair:
                orders[name] += bond.order
        for atom in atoms:
            if orders[atom.name] != atom.closed_shell_valence():
                raise ValueError(f"invalid microstate valence at {atom.name}")

        reached = {atoms[0].name}
        while True:
            neighbors = {
                endpoint
                for pair in bonds_by_pair
                for endpoint in pair
                if set(pair) & reached
            }
            expanded = reached | neighbors
            if expanded == reached:
                break
            reached = expanded
        if reached != by_name.keys():
            raise ValueError("a microstate graph must be one connected chemical site")

        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(
            self, "bonds", tuple(bonds_by_pair[pair] for pair in sorted(bonds_by_pair))
        )

    @classmethod
    def from_heavy_graph(
        cls,
        *,
        elements: Mapping[str, str],
        bonds: tuple[BondDefinition, ...],
        charges: Mapping[str, int],
        boundary_orders: Mapping[str, int],
    ) -> "MicrostateGraph":
        """Complete H counts for a declared heavy graph, not for source evidence.

        Parameters
        ----------
        elements : Mapping[str, str]
            Canonical site atoms and C/N/O elements.
        bonds : tuple[BondDefinition, ...]
            Internal integral bond orders.
        charges : Mapping[str, int]
            Explicit candidate charges; omitted candidate atoms are neutral.
        boundary_orders : Mapping[str, int]
            Known external bond-order sums; omitted atoms have no external bond.

        Returns
        -------
        MicrostateGraph
            Validated candidate with exact H requirements.

        Raises
        ------
        ValueError
            The graph, charge keys, boundary keys or resulting valence is invalid.
        TypeError
            Declared charges or boundary counts are not integers.
        """
        if (
            not charges.keys() <= elements.keys()
            or not boundary_orders.keys() <= elements.keys()
        ):
            raise ValueError("candidate charges/boundaries refer to absent atoms")
        atoms: list[MicrostateAtom] = []
        for name, element in elements.items():
            atom = MicrostateAtom(
                name, element, charges.get(name, 0), 0, boundary_orders.get(name, 0)
            )
            internal_order = sum(
                bond.order
                for bond in bonds
                if name in (bond.atom_name_1, bond.atom_name_2)
            )
            atoms.append(
                MicrostateAtom(
                    name,
                    element,
                    atom.charge,
                    atom.closed_shell_valence() - atom.boundary_order - internal_order,
                    atom.boundary_order,
                )
            )
        return cls(tuple(atoms), bonds)

    def atom(self, name: str) -> MicrostateAtom:
        """Look up one canonical atom.

        Parameters
        ----------
        name : str
            Exact canonical name.

        Returns
        -------
        MicrostateAtom
            Matching atom state.

        Raises
        ------
        KeyError
            The atom is outside the site.
        """
        for atom in self.atoms:
            if atom.name == name:
                return atom
        raise KeyError(name)

    def bond_order(self, name_1: str, name_2: str) -> int | None:
        """Look up an internal bond without assuming missing bonds are single.

        Parameters
        ----------
        name_1, name_2 : str
            Exact canonical endpoints.

        Returns
        -------
        int or None
            Integral order, or None for no internal bond.
        """
        pair = {name_1, name_2}
        return next(
            (
                bond.order
                for bond in self.bonds
                if {bond.atom_name_1, bond.atom_name_2} == pair
            ),
            None,
        )

    def protonation_key(self) -> tuple[int, tuple[tuple[str, int], ...]]:
        """Distinguish protonation/tautomer states while grouping resonance forms.

        Returns
        -------
        tuple
            Net formal charge and exact per-parent H counts in canonical order.
            This grouping is only used within one catalog site and skeleton.
        """
        return sum(atom.charge for atom in self.atoms), tuple(
            (atom.name, atom.hydrogens) for atom in self.atoms
        )
