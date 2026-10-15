"""Idealized component models built from chemistry ingress sources."""

from collections.abc import Collection
from dataclasses import dataclass, replace

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.structure.element import ElementIdentity


@dataclass(frozen=True, slots=True)
class IdealizedComponentAtom:
    """Canonical ideal-geometry atom owned by chemistry ingress."""

    atom_name: str
    element: str
    formal_charge: int
    stereo_config: str | None = None
    ideal_position: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        atom_name = self.atom_name.strip().upper()
        element = self.element.strip().upper()
        if not atom_name:
            raise ValueError("idealized atom name must not be blank")

        if not element:
            raise ValueError("idealized atom element must not be blank")

        object.__setattr__(self, "atom_name", atom_name)
        object.__setattr__(self, "element", element)

        if self.stereo_config is not None:
            normalized_stereo_config = self.stereo_config.strip().upper()
            if normalized_stereo_config not in {"R", "S"}:
                raise ValueError(
                    "idealized atom stereo config must be R or S when provided"
                )
            object.__setattr__(self, "stereo_config", normalized_stereo_config)

        if self.ideal_position is not None:
            if len(self.ideal_position) != 3:
                raise ValueError(
                    "idealized atom position must contain three coordinates"
                )
            object.__setattr__(
                self,
                "ideal_position",
                tuple(float(value) for value in self.ideal_position),
            )

    def is_hydrogen(self) -> bool:
        """Return whether this atom belongs to the hydrogen inventory."""

        return ElementIdentity(self.element).is_hydrogen()


@dataclass(frozen=True, slots=True)
class IdealizedComponent:
    """Canonical ideal-geometry component used after nonstandard ingress."""

    component_id: str
    lineage_parent_component_id: str | None
    atoms: tuple[IdealizedComponentAtom, ...]
    bonds: tuple[BondDefinition, ...]

    def __post_init__(self) -> None:
        component_id = self.component_id.strip().upper()
        if not component_id:
            raise ValueError("idealized component id must not be blank")

        atoms = tuple(self.atoms)
        if not atoms:
            raise ValueError("idealized components require at least one atom")

        atom_names = tuple(atom.atom_name for atom in atoms)
        if len(atom_names) != len(set(atom_names)):
            raise ValueError("idealized component atom names must be unique")

        atom_name_set = set(atom_names)
        bonds = tuple(self.bonds)
        for bond in bonds:
            if (
                bond.atom_name_1 not in atom_name_set
                or bond.atom_name_2 not in atom_name_set
            ):
                raise ValueError("idealized component bonds must reference known atoms")

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "bonds", bonds)
        if self.lineage_parent_component_id is not None:
            object.__setattr__(
                self,
                "lineage_parent_component_id",
                self.lineage_parent_component_id.strip().upper() or None,
            )

    def atom(self, atom_name: str) -> IdealizedComponentAtom | None:
        """Return one canonical atom when present."""

        normalized_atom_name = atom_name.strip().upper()
        for atom in self.atoms:
            if atom.atom_name == normalized_atom_name:
                return atom

        return None

    def atom_with_ideal_position(
        self,
        atom_name: str,
    ) -> IdealizedComponentAtom | None:
        """Return one atom only when ideal coordinates are available."""

        atom = self.atom(atom_name)
        if atom is None or atom.ideal_position is None:
            return None

        return atom

    def heavy_atoms(self) -> tuple[IdealizedComponentAtom, ...]:
        """Return heavy atoms in canonical packaged order."""

        return tuple(atom for atom in self.atoms if not atom.is_hydrogen())

    def hydrogen_atoms(self) -> tuple[IdealizedComponentAtom, ...]:
        """Return hydrogen atoms in canonical packaged order."""

        return tuple(atom for atom in self.atoms if atom.is_hydrogen())

    def heavy_atom_names(self) -> tuple[str, ...]:
        """Return heavy atom names in canonical packaged order."""

        return tuple(atom.atom_name for atom in self.heavy_atoms())

    def heavy_bonds(self) -> tuple[BondDefinition, ...]:
        """Return only heavy-atom bonds."""

        heavy_atom_names = set(self.heavy_atom_names())
        return tuple(
            bond
            for bond in self.bonds
            if bond.atom_name_1 in heavy_atom_names
            and bond.atom_name_2 in heavy_atom_names
        )

    def bonded_atom_names(self, atom_name: str) -> frozenset[str]:
        """Return directly bonded neighbors for one canonical atom name."""

        normalized_atom_name = atom_name.strip().upper()
        bonded_atoms: set[str] = set()
        for bond in self.bonds:
            if bond.atom_name_1 == normalized_atom_name:
                bonded_atoms.add(bond.atom_name_2)
            elif bond.atom_name_2 == normalized_atom_name:
                bonded_atoms.add(bond.atom_name_1)

        return frozenset(bonded_atoms)

    def hydrogen_anchor_atom_name(self, hydrogen_atom_name: str) -> str | None:
        """Return the unique heavy-atom anchor bonded to one hydrogen atom."""

        normalized_hydrogen_atom_name = hydrogen_atom_name.strip().upper()
        heavy_neighbors = tuple(
            atom_name
            for atom_name in self.bonded_atom_names(normalized_hydrogen_atom_name)
            if atom_name not in {atom.atom_name for atom in self.hydrogen_atoms()}
        )
        if len(heavy_neighbors) != 1:
            return None

        return heavy_neighbors[0]

    def neutral_aldehyde_carbonyl_atom_names(self) -> tuple[str, ...]:
        """Find neutral C(=O)(C)H centers eligible for acyl substitution.

        Returns
        -------
        tuple[str, ...]
            Carbonyl atom names with exactly one O double bond, one C single
            bond and one H single bond. This describes the isolated graph, not
            evidence that a substitution has occurred.
        """
        names = []
        atoms_by_name = {atom.atom_name: atom for atom in self.atoms}
        for atom in self.heavy_atoms():
            if atom.element != "C" or atom.formal_charge != 0:
                continue
            neighbors = []
            for bond in self.bonds:
                if atom.atom_name not in (bond.atom_name_1, bond.atom_name_2):
                    continue
                other_name = (
                    bond.atom_name_2
                    if bond.atom_name_1 == atom.atom_name
                    else bond.atom_name_1
                )
                other = atoms_by_name.get(other_name)
                if other is None or other.formal_charge != 0 or bond.aromatic:
                    neighbors = []
                    break
                neighbors.append((other.element, bond.order))
            if sorted(neighbors) == [("C", 1), ("H", 1), ("O", 2)]:
                names.append(atom.atom_name)
        return tuple(names)

    def without_hydrogens_anchored_to(
        self, anchor_atom_names: Collection[str]
    ) -> "IdealizedComponent":
        """Project the component inventory without H on selected anchors.

        Parameters
        ----------
        anchor_atom_names : Collection[str]
            Canonical heavy anchors whose H inventory is excluded.

        Returns
        -------
        IdealizedComponent
            Pure graph projection. It neither deletes deposited atoms nor
            establishes the external chemistry occupying the freed valence.
        """
        excluded = {
            atom.atom_name
            for atom in self.hydrogen_atoms()
            if self.hydrogen_anchor_atom_name(atom.atom_name) in anchor_atom_names
        }
        if not excluded:
            return self
        return replace(
            self,
            atoms=tuple(atom for atom in self.atoms if atom.atom_name not in excluded),
            bonds=tuple(
                bond
                for bond in self.bonds
                if bond.atom_name_1 not in excluded and bond.atom_name_2 not in excluded
            ),
        )
