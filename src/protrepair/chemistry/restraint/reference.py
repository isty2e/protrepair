"""Read-only applicability of component-state geometry references."""

from collections import Counter
from dataclasses import dataclass, field, replace

from protrepair.chemistry.component.idealized import IdealizedComponent
from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
    ResidueRestraintTemplate,
)
from protrepair.structure.element import ElementIdentity

_AtomState = tuple[str, str, int, int, tuple[tuple[int, bool], ...]]
_BondState = tuple[str, str, int, bool]
_StateKey = tuple[tuple[_AtomState, ...], tuple[_BondState, ...]]


def _state_key(
    component: IdealizedComponent,
    atom_names: frozenset[str],
    outside_atom_names: frozenset[str],
) -> _StateKey | None:
    atoms = {atom.atom_name: atom for atom in component.atoms}
    heavy = {name for name, atom in atoms.items() if not atom.is_hydrogen()}
    if heavy - outside_atom_names != atom_names:
        return None

    hydrogens: Counter[str] = Counter()
    hydrogen_parents: dict[str, str] = {}
    boundaries: dict[str, list[tuple[int, bool]]] = {name: [] for name in atom_names}
    edges: list[_BondState] = []
    pairs: set[tuple[str, str]] = set()
    for bond in component.bonds:
        first, second = sorted((bond.atom_name_1, bond.atom_name_2))
        if first == second or (first, second) in pairs:
            return None
        pairs.add((first, second))
        hydrogen = [name for name in (first, second) if atoms[name].is_hydrogen()]
        if hydrogen:
            if len(hydrogen) != 1 or bond.order != 1 or bond.aromatic:
                return None
            child = hydrogen[0]
            if atoms[child].formal_charge != 0:
                return None
            parent = second if child == first else first
            if child in hydrogen_parents:
                return None
            hydrogen_parents[child] = parent
            hydrogens[parent] += 1
        elif first in atom_names and second in atom_names:
            edges.append(
                (first, second, 0 if bond.aromatic else bond.order, bond.aromatic)
            )
        else:
            for name in (first, second):
                if name in atom_names:
                    boundaries[name].append(
                        (0 if bond.aromatic else bond.order, bond.aromatic)
                    )
    if {
        name for name, atom in atoms.items() if atom.is_hydrogen()
    } != hydrogen_parents.keys():
        return None

    nodes = tuple(
        (
            name,
            ElementIdentity(atoms[name].element).chemical_symbol,
            atoms[name].formal_charge,
            hydrogens[name],
            tuple(sorted(boundaries[name])),
        )
        for name in sorted(atom_names)
    )
    return nodes, tuple(sorted(edges))


@dataclass(frozen=True, slots=True)
class ComponentRestraintReference:
    """Own geometry replacements for one realized named component state.

    Parameters
    ----------
    component : IdealizedComponent
        Reference graph and H inventory; coordinates do not affect admission.
    atom_names : frozenset[str]
        Complete heavy scope whose chemical state must agree.
    boundary_atom_names : frozenset[str]
        Scoped anchors whose state is checked but whose targets stay unchanged.
    bond_targets : tuple[BondRestraintTarget, ...]
        Replacement distances strictly inside the non-boundary heavy scope.
    angle_targets : tuple[AngleRestraintTarget, ...]
        Complete replacement angle centres that do not touch the boundary.
    source : str
        Stable source identity, not a filesystem input or preparation authority.

    Raises
    ------
    ValueError
        The scope, target identities or reference H attachments are inconsistent.
    """

    component: IdealizedComponent
    atom_names: frozenset[str]
    boundary_atom_names: frozenset[str]
    bond_targets: tuple[BondRestraintTarget, ...]
    angle_targets: tuple[AngleRestraintTarget, ...]
    source: str
    _outside_atom_names: frozenset[str] = field(init=False, repr=False)
    _key: _StateKey = field(init=False, repr=False)
    _boundary_centers: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = frozenset(name.strip().upper() for name in self.atom_names)
        boundary = frozenset(name.strip().upper() for name in self.boundary_atom_names)
        heavy = frozenset(atom.atom_name for atom in self.component.heavy_atoms())
        if not names or not names <= heavy or not boundary < names:
            raise ValueError("reference scope and boundary require known heavy atoms")
        source = self.source.strip()
        if not source:
            raise ValueError("component restraint references require a source identity")
        bonds, angles = tuple(self.bond_targets), tuple(self.angle_targets)
        writable = names - boundary
        if any({t.atom_name_1, t.atom_name_2} - writable for t in bonds) or any(
            {t.atom_name_1, t.center_atom_name, t.atom_name_2} - writable
            for t in angles
        ):
            raise ValueError(
                "reference targets must stay inside the writable heavy scope"
            )
        if len({frozenset((t.atom_name_1, t.atom_name_2)) for t in bonds}) != len(
            bonds
        ) or len({t.canonical_triplet() for t in angles}) != len(angles):
            raise ValueError("reference target identities must not repeat")
        outside = heavy - names
        boundary_centers = {
            name
            for bond in self.component.heavy_bonds()
            for name, neighbor in (
                (bond.atom_name_1, bond.atom_name_2),
                (bond.atom_name_2, bond.atom_name_1),
            )
            if name in writable and neighbor in boundary
        }
        if any(target.center_atom_name in boundary_centers for target in angles):
            raise ValueError(
                "angle centers touching the boundary must retain fallback targets"
            )
        if any(
            {bond.atom_name_1, bond.atom_name_2} <= heavy
            and {bond.atom_name_1, bond.atom_name_2} & writable
            and not {bond.atom_name_1, bond.atom_name_2} <= names
            for bond in self.component.bonds
        ):
            raise ValueError(
                "only boundary atoms may connect outside the reference scope"
            )
        key = _state_key(self.component, names, outside)
        if key is None:
            raise ValueError("component reference has ambiguous graph or H parents")
        object.__setattr__(self, "atom_names", names)
        object.__setattr__(self, "boundary_atom_names", boundary)
        object.__setattr__(self, "bond_targets", bonds)
        object.__setattr__(self, "angle_targets", angles)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "_outside_atom_names", outside)
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_boundary_centers", frozenset(boundary_centers))

    def can_replace_angle_center(self, atom_name: str) -> bool:
        """Identify angle centres whose complete heavy neighbourhood is writable.

        Parameters
        ----------
        atom_name : str
            Named reference atom, normalized before lookup.

        Returns
        -------
        bool
            False for boundary atoms, their neighbours and unknown names.
        """

        name = atom_name.strip().upper()
        return (
            name in self.atom_names - self.boundary_atom_names
            and name not in self._boundary_centers
        )

    def matches(self, component: IdealizedComponent) -> bool:
        """Compare realized chemistry, including H/D/T parents, not coordinates.

        Parameters
        ----------
        component : IdealizedComponent
            Current named graph; missing H remains missing, not imputed.

        Returns
        -------
        bool
            Whether the scope, charges, bonds, H counts and boundary orders agree.
        """

        return (
            self.component.component_id == component.component_id
            and self._key
            == _state_key(component, self.atom_names, self._outside_atom_names)
        )

    def same_state_as(self, other: "ComponentRestraintReference") -> bool:
        """Identify competing references for the same scoped state.

        Parameters
        ----------
        other : ComponentRestraintReference
            Another reference configuration.

        Returns
        -------
        bool
            Whether both applicability predicates select the same named state.
        """

        return self.same_scope_as(other) and self._key == other._key

    def same_scope_as(self, other: "ComponentRestraintReference") -> bool:
        """Compare the chemical scope of two state alternatives.

        Parameters
        ----------
        other : ComponentRestraintReference
            Another reference configuration.

        Returns
        -------
        bool
            Whether identity, scoped atoms, boundary and outside atoms agree.
        """

        return (
            self.component.component_id == other.component.component_id
            and self.atom_names == other.atom_names
            and self._outside_atom_names == other._outside_atom_names
            and self.boundary_atom_names == other.boundary_atom_names
        )

    def validate_targets(self, template: ResidueRestraintTemplate) -> None:
        """Reject reference configuration that changes restraint coverage.

        Parameters
        ----------
        template : ResidueRestraintTemplate
            Static dictionary that remains the complete fallback.

        Raises
        ------
        ValueError
            Targets add identities, mix angle centres or conflict with the graph.
        """

        if template.component_id != self.component.component_id:
            raise ValueError("component reference and template identities disagree")
        if template.sidechain_variants:
            raise ValueError(
                "component references cannot replace microstate-site variants"
            )
        pairs = {
            frozenset((bond.atom_name_1, bond.atom_name_2))
            for bond in self.component.heavy_bonds()
        }
        if any(
            frozenset((t.atom_name_1, t.atom_name_2)) not in pairs
            for t in self.bond_targets
        ) or any(
            frozenset((endpoint, t.center_atom_name)) not in pairs
            for t in self.angle_targets
            for endpoint in (t.atom_name_1, t.atom_name_2)
        ):
            raise ValueError(
                "reference targets require the declared chemical connectivity"
            )
        if any(
            template.bond_target(t.atom_name_1, t.atom_name_2) is None
            for t in self.bond_targets
        ) or any(
            not any(
                s.canonical_triplet() == t.canonical_triplet()
                for s in template.angle_targets
            )
            for t in self.angle_targets
        ):
            raise ValueError("component references cannot add restraint identities")
        centers = {t.center_atom_name for t in self.angle_targets}
        selected = {t.canonical_triplet() for t in self.angle_targets}
        if any(
            t.canonical_triplet() not in selected
            for t in template.angle_targets
            if t.center_atom_name in centers
        ):
            raise ValueError("component references must replace complete angle centers")

    def apply_to(self, template: ResidueRestraintTemplate) -> ResidueRestraintTemplate:
        """Replace admitted targets while preserving all other template facts.

        Parameters
        ----------
        template : ResidueRestraintTemplate
            Validated static fallback for this component.

        Returns
        -------
        ResidueRestraintTemplate
            Flat targets with unchanged identity, stereo, planes and coverage.

        Raises
        ------
        ValueError
            The reference is incompatible with the supplied fallback.
        """

        self.validate_targets(template)
        bonds = {
            frozenset((t.atom_name_1, t.atom_name_2)): t for t in self.bond_targets
        }
        angles = {t.canonical_triplet(): t for t in self.angle_targets}
        return replace(
            template,
            bond_targets=tuple(
                bonds.get(frozenset((t.atom_name_1, t.atom_name_2)), t)
                for t in template.bond_targets
            ),
            angle_targets=tuple(
                angles.get(t.canonical_triplet(), t) for t in template.angle_targets
            ),
        )
