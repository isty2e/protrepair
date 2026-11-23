"""Named, isolated chemistry supplied for an explicitly selected retained site."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from protrepair.structure.labels import ResidueId

NAMED_COMPONENT_ELEMENTS = frozenset(
    ("H", "B", "C", "N", "O", "F", "SI", "P", "S", "CL", "BR", "I")
)


@dataclass(frozen=True, slots=True)
class ComponentAtom:
    """One named CCD atom, including its authored charge and stereo."""

    name: str
    element: str
    charge: int
    aromatic: bool | None
    stereo: str | None


@dataclass(frozen=True, slots=True)
class ComponentBond:
    """One canonical endpoint pair and its declared chemical attributes."""

    first: str
    second: str
    order: str
    aromatic: bool | None
    stereo: str | None


@dataclass(frozen=True, slots=True)
class NamedComponentGraph:
    """An isolated component's atom, bond, charge, H and stereo inventory."""

    component_id: str
    atoms: tuple[ComponentAtom, ...]
    bonds: tuple[ComponentBond, ...]

    @property
    def heavy_atom_names(self) -> tuple[str, ...]:
        """Return the heavy atom names in the native graph's order."""
        return tuple(a.name for a in self.atoms if a.element != "H")

    def hydrogen_slots(self) -> dict[str, tuple[str, ...]]:
        """Group dictionary H names by their bonded heavy atom."""
        hydrogen = {a.name for a in self.atoms if a.element == "H"}
        slots: defaultdict[str, list[str]] = defaultdict(list)
        for bond in self.bonds:
            if bond.first in hydrogen:
                slots[bond.second].append(bond.first)
            elif bond.second in hydrogen:
                slots[bond.first].append(bond.second)
        return {parent: tuple(sorted(names)) for parent, names in slots.items()}

    def occupied_slots(self, parents: Mapping[str, str]) -> dict[str, str]:
        """Map observed H names to distinct dictionary slots on their verified parents."""
        slots = self.hydrogen_slots()
        names = {h: p for p, group in slots.items() for h in group}
        assignment: dict[str, str] = {}
        occupied: set[str] = set()
        for name, parent in parents.items():
            slot = self.slot_for_name(name, names)
            if slot is not None:
                if names[slot] != parent or slot in occupied:
                    raise ValueError(
                        f"conflicting observed H/D/T for dictionary slot {slot}"
                    )
                assignment[name] = slot
                occupied.add(slot)
        for name, parent in sorted(parents.items()):
            if name not in assignment:
                available = next(
                    (s for s in slots.get(parent, ()) if s not in occupied), None
                )
                if available is None:
                    raise ValueError(f"no dictionary H slot for {name} on {parent}")
                assignment[name] = available
                occupied.add(available)
        return assignment

    @staticmethod
    def slot_for_name(name: str, names: Mapping[str, str]) -> str | None:
        """Recognize a dictionary H name or its deposited isotope spelling."""
        if name in names:
            return name
        alias = "H" + name[1:]
        return alias if name.startswith(("D", "T")) and alias in names else None


@dataclass(frozen=True, slots=True)
class RetainedComponentDefinition(NamedComponentGraph):
    """Immutable named chemistry read from a local PDBx component dictionary."""

    def for_residue(self, residue_id: ResidueId) -> "RetainedComponentAssignment":
        """Select this definition for one retained residue.

        Parameters
        ----------
        residue_id : ResidueId
            Exact normalized chain, residue number and insertion code.

        Returns
        -------
        RetainedComponentAssignment
            Chemistry selection for this site; retention is configured separately.
        """
        return RetainedComponentAssignment(residue_id, self)


@dataclass(frozen=True, slots=True)
class RetainedComponentAssignment:
    """A named chemical graph selected at one canonical residue address."""

    residue_id: ResidueId
    definition: RetainedComponentDefinition

    def __post_init__(self) -> None:
        if not isinstance(self.residue_id, ResidueId):
            raise TypeError("residue_id must be a ResidueId")

    @property
    def heavy_atom_names(self) -> tuple[str, ...]:
        """Return the heavy atom correspondence of this selection."""
        return self.definition.heavy_atom_names

    @property
    def model_label(self) -> str:
        """Identify the selected source in preparation diagnostics."""
        return f"caller CCD {self.definition.component_id}"


@dataclass(frozen=True, slots=True)
class MatchedSmiles:
    """One uniquely determined named chemistry, fixed before workflow search."""

    residue_id: ResidueId
    definition: NamedComponentGraph
    heavy_sites: tuple[tuple[str, str | None], ...]
    observed_hydrogens: tuple[tuple[str, str, str | None, str], ...]
    from_supplied_model: bool = False

    @property
    def heavy_atom_names(self) -> tuple[str, ...]:
        """Return the published graph's stable heavy order."""
        return self.definition.heavy_atom_names

    @property
    def model_label(self) -> str:
        """Identify the charge and bond localization as model-assigned."""
        return (
            "caller SMILES (model-assigned correspondence on supplied canonical input)"
            if self.from_supplied_model
            else "caller SMILES (model-assigned correspondence)"
        )


NamedGraphAssignment = RetainedComponentAssignment | MatchedSmiles
