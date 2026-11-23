"""Bind isolated named graphs to current and original retained atoms."""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite

from protrepair.chemistry.retained_non_polymer.definition import (
    RetainedComponentDefinition,
)
from protrepair.chemistry.retained_non_polymer.hydrogen_correspondence import (
    source_hydrogen_bond_length_is_compatible,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import (
    ResidueRole,
    StructureConstitution,
)
from protrepair.structure.geometry import ResidueGeometry, StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType, StructureTopology

NamedPoseValidator = Callable[
    [ResidueGeometry, Mapping[tuple[str, str], int], dict[str, str]], None
]


def _require_source_isolation(
    structure: ProteinStructure, residue_id: ResidueId, model: str
) -> None:
    """Require a selected isolated graph to account for its source partners.

    Parameters
    ----------
    structure : ProteinStructure
        Selected carrier with the original source observation, if available.
    residue_id : ResidueId
        Occurrence being prepared with explicit chemistry evidence.

    model : str
        Selected isolated chemistry for the refusal message.

    Raises
    ------
    ValueError
        A source declaration connects this site to another residue. The
        isolated model cannot account for the partner even if it was dropped.
    """
    observation = structure.provenance.ingress.observation
    residue = structure.constitution.residue_or_ligand(residue_id)
    if observation is None or residue is None:
        return
    for link in (
        *observation.unresolved_non_polymer_links,
        *observation.excluded_coordination_connections,
    ):
        if residue_id not in (
            link.endpoint_1.atom_ref.residue_id,
            link.endpoint_2.atom_ref.residue_id,
        ):
            continue
        kept = link.lone_retained_endpoint(
            observation.constitution, observation.geometry
        )
        if kept is not None and kept.atom_ref.residue_id == residue_id:
            raise ValueError(
                f"{model} has an unrepresented "
                f"{link.source_metadata.reported_relationship_type.value} partner "
                f"at {kept.atom_ref.display_token()}"
            )

    for bond in observation.bonds_for_residue(residue_id):
        if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
            continue
        a, b = (observation.constitution.atom_ref_at(i) for i in bond.endpoint_pair())
        if a.residue_id != b.residue_id:
            raise ValueError(
                f"{model} has a declared {bond.relationship_type.value} partner: {a.display_token()}--{b.display_token()}"
            )


@dataclass(frozen=True, slots=True)
class SelectedIsolatedGraph:
    """Named heavy atoms, charges, bonds and H capacities of the selected graph."""

    residue_id: ResidueId
    component_id: str
    elements: dict[str, str]
    formal_charges: dict[str, int]
    bond_orders: dict[frozenset[str], int]
    hydrogen_capacity: dict[str, int]

    named_slots: dict[str, str] | None = None
    aromatic_pairs: frozenset[frozenset[str]] = field(default_factory=frozenset)

    def hydrogen_parents(
        self,
        structure: ProteinStructure,
        *,
        prepare_lengths: Callable[[ResidueGeometry], tuple[tuple[str, float], ...]],
        validate_pose: NamedPoseValidator | None = None,
    ) -> dict[str, str]:
        """Check current and original chemistry before binding observed hydrogen slots.

        Parameters
        ----------
        structure : ProteinStructure
            Current carrier with its original observation, when available.
        prepare_lengths : callable
            Expected X-H distances for validating observed coordinates.
        validate_pose : callable, optional
            Check named geometry, reported orders and the resolved H parents.

        Returns
        -------
        dict[str, str]
            Present H/D/T names and their source-compatible heavy parents.

        Raises
        ------
        ValueError
            The scaffold, isotope inventory, charges, attachments or named
            parent sites conflict with the selected model.
        """
        _require_source_isolation(
            structure,
            self.residue_id,
            "orthophosphate" if self.named_slots is None else self.component_id,
        )
        current = self._snapshot_parents(
            structure.constitution,
            structure.topology,
            structure.geometry,
            prepare_lengths,
            validate_pose,
        )
        observation = structure.provenance.ingress.observation
        if observation is None:
            return current

        original = self._snapshot_parents(
            observation.constitution,
            observation.topology,
            observation.geometry,
            prepare_lengths,
            validate_pose,
        )
        for name, parent in original.items():
            ref = AtomRef(self.residue_id, name)
            if current.get(name) != parent or structure.constitution.resolve_atom_site(
                ref
            ) != observation.constitution.resolve_atom_site(ref):
                raise ValueError(
                    "orthophosphate model cannot replace or reparent source H/D/T"
                    if self.named_slots is None
                    else f"selected {self.component_id} model cannot replace or reparent source H/D/T"
                )
        return current

    def _snapshot_parents(
        self,
        constitution: StructureConstitution,
        topology: StructureTopology,
        geometry: StructureGeometry,
        prepare_lengths: Callable[[ResidueGeometry], tuple[tuple[str, float], ...]],
        validate_pose: NamedPoseValidator | None,
    ) -> dict[str, str]:
        residue = constitution.residue_or_ligand(self.residue_id)
        if (
            residue is None
            or residue.component_id != self.component_id
            or constitution.role_at(constitution.residue_index(self.residue_id))
            is not ResidueRole.LIGAND
            or {
                (atom.name, atom.element)
                for atom in residue.atom_sites
                if not atom.is_hydrogen()
            }
            != set(self.elements.items())
        ):
            raise ValueError(
                f"selected {self.component_id} graph does not match the retained heavy atoms"
            )

        parents: dict[str, str] = {}
        unassigned: list[str] = []
        for atom in residue.atom_sites:
            index = constitution.atom_index(AtomRef(self.residue_id, atom.name))
            if topology.formal_charge(index) not in (
                None,
                self.formal_charges.get(atom.name, 0),
            ):
                raise ValueError(
                    f"selected {self.component_id} graph conflicts with formal charge at {atom.name}"
                )

            attachments: set[str] = set()
            for bond in topology.incident_bonds_by_atom().get(index, ()):
                if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                    continue
                other = (
                    bond.atom_index_2
                    if index == bond.atom_index_1
                    else bond.atom_index_1
                )
                partner = constitution.atom_ref_at(other)
                if (
                    partner.residue_id != self.residue_id
                    or bond.relationship_type
                    not in (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN)
                ):
                    raise ValueError(
                        f"selected {self.component_id} graph does not support this {bond.relationship_type.value} attachment at {atom.name}"
                    )
                aromatic = False
                if atom.is_hydrogen():
                    attachments.add(partner.atom_name)
                    order = 1
                elif constitution.atom_site_at(other).is_hydrogen():
                    continue  # The hydrogen endpoint owns this check.
                else:
                    pair = frozenset((atom.name, partner.atom_name))
                    if pair not in self.bond_orders:
                        raise ValueError(
                            f"unexpected bond at {atom.name} in selected {self.component_id}"
                        )
                    order = self.bond_orders[pair]
                    aromatic = pair in self.aromatic_pairs
                if (bond.aromatic and not aromatic) or bond.order not in (
                    (None, 1, 2) if aromatic else (None, order)
                ):
                    raise ValueError(
                        f"selected {self.component_id} graph conflicts with the bond order at {atom.name}"
                    )

            if atom.is_hydrogen():
                named = atom.name[1:] if atom.name[:1] in ("H", "D", "T") else ""
                if self.named_slots is not None:
                    slot = RetainedComponentDefinition.slot_for_name(
                        atom.name, self.named_slots
                    )
                    if slot is not None:
                        attachments.add(self.named_slots[slot])
                elif self.elements.get(named) == "O":
                    attachments.add(named)
                if not attachments and self.hydrogen_capacity:
                    unassigned.append(atom.name)
                    continue
                if len(attachments) != 1 or not attachments.issubset(
                    self.hydrogen_capacity
                ):
                    raise ValueError(
                        f"selected {self.component_id} graph cannot preserve the {self._parent_kind} parent of {atom.name}"
                    )
                parents[atom.name] = attachments.pop()
        index = constitution.residue_index(self.residue_id)
        pose = geometry.residue_geometry(constitution=constitution, residue_index=index)
        if unassigned or (self.named_slots is not None and parents):
            lengths = prepare_lengths(pose)
            for name in unassigned:
                parents[name] = self._geometric_parent(name, pose, lengths)
            if self.named_slots is not None:
                for name, parent in parents.items():
                    distance = pose.position(name).distance_to(pose.position(parent))
                    if not any(
                        p == parent
                        and source_hydrogen_bond_length_is_compatible(distance, length)
                        for p, length in lengths
                    ):
                        raise ValueError(
                            f"source H/D/T {name} is not distance-compatible with dictionary parent {parent}"
                        )
        if any(
            count > self.hydrogen_capacity[parent]
            for parent, count in Counter(parents.values()).items()
        ):
            raise ValueError(
                f"selected {self.component_id} graph cannot preserve all source H/D/T at their {self._parent_kind} parents"
            )
        if validate_pose is not None:
            validate_pose(
                pose,
                topology.residue_source_heavy_bond_orders(
                    constitution=constitution, residue_index=index
                ),
                parents,
            )
        return parents

    @property
    def _parent_kind(self) -> str:
        return "oxygen" if self.named_slots is None else "heavy"

    def _geometric_parent(
        self, name: str, pose: ResidueGeometry, lengths: tuple[tuple[str, float], ...]
    ) -> str:
        position = pose.position(name)
        distances = {
            atom: position.distance_to(pose.position(atom)) for atom in self.elements
        }
        compatible = {
            parent
            for parent, length in lengths
            if parent in self.hydrogen_capacity
            and source_hydrogen_bond_length_is_compatible(distances[parent], length)
        }
        if len(compatible) == 1 and all(isfinite(d) for d in distances.values()):
            closest = min(distances.values())
            if compatible == {atom for atom, d in distances.items() if d == closest}:
                return compatible.pop()
        raise ValueError(
            f"selected {self.component_id} graph has no unambiguous, distance-compatible {self._parent_kind} parent for {name}"
        )
