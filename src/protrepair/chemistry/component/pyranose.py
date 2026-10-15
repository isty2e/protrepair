"""Graph-qualified pyranose roles and coupled substitution inventory."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from protrepair.chemistry.component.idealized import IdealizedComponent


@dataclass(frozen=True, slots=True, init=False)
class PyranoseScaffold:
    """Bind a unique saturated oxane ring to its aldose-like chemical roles.

    Parameters
    ----------
    component : IdealizedComponent
        Selected isolated graph with explicit H. Names carry identity, not roles.

    Notes
    -----
    This structural interpretation is not native valence or source-conformance
    certification. A hemiacetal hydroxyl may leave; an acetal O--carbon group
    cannot. Only neutral aliphatic alcohol oxygens are admitted acceptors.
    Furanose, ketose and ambiguous anomeric scaffolds are unsupported.

    Raises
    ------
    ValueError
        The selected graph has no unique supported ring/anomeric neighborhood.
    """

    component: IdealizedComponent = field(repr=False)
    anomeric_carbon: str
    exocyclic_oxygen: str
    hydroxyl_hydrogens: Mapping[str, str]

    def __init__(self, component: IdealizedComponent) -> None:
        atoms = {atom.atom_name: atom for atom in component.atoms}
        adjacency: dict[str, dict[str, tuple[int, bool]]] = {name: {} for name in atoms}
        for bond in component.bonds:
            first, second = bond.atom_name_1, bond.atom_name_2
            if first == second or second in adjacency[first]:
                raise ValueError("pyranose graph has duplicate or self bonds")
            adjacency[first][second] = (bond.order, bond.aromatic)
            adjacency[second][first] = (bond.order, bond.aromatic)

        saturated_carbons = {
            name
            for name, atom in atoms.items()
            if atom.element == "C"
            and atom.formal_charge == 0
            and len(adjacency[name]) == 4
            and all(attributes == (1, False) for attributes in adjacency[name].values())
        }
        rings: set[frozenset[str]] = set()
        for oxygen, atom in atoms.items():
            neighbors = tuple(adjacency[oxygen])
            if (
                atom.element != "O"
                or atom.formal_charge != 0
                or len(neighbors) != 2
                or any(
                    attributes != (1, False)
                    for attributes in adjacency[oxygen].values()
                )
                or not set(neighbors) <= saturated_carbons
            ):
                continue
            first, last = neighbors
            paths: list[tuple[str, ...]] = [(first,)]
            while paths:
                path = paths.pop()
                if path[-1] == last:
                    if len(path) == 5:
                        ring = frozenset((*path, oxygen))
                        if all(len(set(adjacency[n]) & ring) == 2 for n in ring):
                            rings.add(ring)
                    continue
                if len(path) == 5:
                    continue
                paths.extend(
                    (*path, neighbor)
                    for neighbor in adjacency[path[-1]]
                    if neighbor in saturated_carbons and neighbor not in path
                )
        if len(rings) != 1:
            raise ValueError(
                "pyranose chemistry requires one unambiguous saturated ring"
            )
        ring = next(iter(rings))
        ring_oxygen = next(name for name in ring if atoms[name].element == "O")
        candidates = []
        for carbon in adjacency[ring_oxygen]:
            external = tuple(name for name in adjacency[carbon] if name not in ring)
            if len(external) != 2:
                continue
            hydrogens = tuple(name for name in external if atoms[name].is_hydrogen())
            oxygens = tuple(name for name in external if atoms[name].element == "O")
            if len(hydrogens) != 1 or len(oxygens) != 1:
                continue
            oxygen = oxygens[0]
            if (
                atoms[oxygen].formal_charge != 0
                or len(adjacency[oxygen]) != 2
                or any(
                    attributes != (1, False)
                    for attributes in adjacency[oxygen].values()
                )
            ):
                continue
            other = next(name for name in adjacency[oxygen] if name != carbon)
            if not atoms[other].is_hydrogen() and atoms[other].element != "C":
                continue
            candidates.append((carbon, oxygen))
        if len(candidates) != 1:
            raise ValueError("pyranose chemistry requires a unique aldose-like anomer")

        hydroxyls = {}
        for oxygen, atom in atoms.items():
            if (
                atom.element != "O"
                or atom.formal_charge != 0
                or len(adjacency[oxygen]) != 2
                or any(
                    attributes != (1, False)
                    for attributes in adjacency[oxygen].values()
                )
            ):
                continue
            carbons = tuple(
                name for name in adjacency[oxygen] if name in saturated_carbons
            )
            hydrogens = tuple(
                name for name in adjacency[oxygen] if atoms[name].is_hydrogen()
            )
            if (
                len(carbons) == 1
                and len(hydrogens) == 1
                and atoms[hydrogens[0]].formal_charge == 0
                and adjacency[hydrogens[0]] == {oxygen: (1, False)}
            ):
                hydroxyls[oxygen] = hydrogens[0]
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "anomeric_carbon", candidates[0][0])
        object.__setattr__(self, "exocyclic_oxygen", candidates[0][1])
        object.__setattr__(self, "hydroxyl_hydrogens", MappingProxyType(hydroxyls))

    def require_donor(self, atom_name: str) -> None:
        """Require a declared anomeric center with a leaving hydroxyl.

        Parameters
        ----------
        atom_name : str
            Deposited endpoint in the isolated component's name space.

        Raises
        ------
        ValueError
            The endpoint is not the anomer or lacks leaving OH/declared stereo.
        """
        atom = self.component.atom(atom_name)
        if (
            atom_name != self.anomeric_carbon
            or self.exocyclic_oxygen not in self.hydroxyl_hydrogens
            or atom is None
            or atom.stereo_config is None
        ):
            raise ValueError(
                "glycosidic donor requires an anomeric hydroxyl and declared stereo"
            )

    def require_acceptor(self, atom_name: str) -> None:
        """Require a neutral aliphatic hydroxyl oxygen in the selected scaffold.

        Parameters
        ----------
        atom_name : str
            Deposited acceptor endpoint.

        Raises
        ------
        ValueError
            The endpoint is an ether, carbonyl/acid oxygen or another non-alcohol.
        """
        if atom_name not in self.hydroxyl_hydrogens:
            raise ValueError(
                "glycosidic acceptor requires a neutral aliphatic hydroxyl"
            )

    def project(self, occupied_atom_names: Collection[str]) -> IdealizedComponent:
        """Project an admitted donor/acceptor product without deleting source atoms.

        Parameters
        ----------
        occupied_atom_names : Collection[str]
            Endpoints occupied by validated external single covalent attachments.

        Returns
        -------
        IdealizedComponent
            Selected inventory without donor leaving O/H or occupied hydroxyl H.

        Raises
        ------
        ValueError
            An endpoint is unsupported or the leaving group is also an acceptor.
        """
        occupied = set(occupied_atom_names)
        excluded = set()
        if self.anomeric_carbon in occupied:
            self.require_donor(self.anomeric_carbon)
            occupied.remove(self.anomeric_carbon)
            excluded.update(
                (self.exocyclic_oxygen, self.hydroxyl_hydrogens[self.exocyclic_oxygen])
            )
            if self.exocyclic_oxygen in occupied:
                raise ValueError(
                    "a leaving oxygen cannot also be a glycosidic acceptor"
                )
        for name in occupied:
            self.require_acceptor(name)
            excluded.add(self.hydroxyl_hydrogens[name])
        return replace(
            self.component,
            atoms=tuple(
                atom for atom in self.component.atoms if atom.atom_name not in excluded
            ),
            bonds=tuple(
                bond
                for bond in self.component.bonds
                if not excluded.intersection((bond.atom_name_1, bond.atom_name_2))
            ),
        )
