"""Read-only reference H comparison shared by occurrence-role observers."""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Literal

from protrepair.chemistry.component.idealized import IdealizedComponent
from protrepair.chemistry.inference.idealized_stereochemistry import (
    component_reference_hydrogen_bond_lengths,
    require_compatible_component_pose,
    require_compatible_component_reference,
)
from protrepair.chemistry.retained_non_polymer.hydrogen_correspondence import (
    source_hydrogen_bond_length_is_compatible,
)
from protrepair.errors import BondOrderConstraintError, RdkitUnavailableError
from protrepair.state.structure_axes import HydrogenCoverageState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondProvenance, BondRelationshipType


@dataclass(frozen=True, slots=True)
class ReferenceHydrogenFailure:
    """Keep a read-only comparison's typed cause separate from coverage.

    Parameters
    ----------
    kind : str
        Closed comparison/preparation failure category.
    detail : str
        Current cause suitable for a diagnostic projection.
    """

    kind: Literal[
        "backend_unavailable",
        "reference_invalid",
        "reference_state_not_covered",
        "hydrogen_parent_unresolved",
        "source_conflict",
        "context_unresolved",
        "no_admitted_preparation",
    ]
    detail: str


@dataclass(frozen=True, slots=True)
class ReferenceHydrogenScope:
    """Describe validated occurrence boundaries without changing the reference.

    Parameters
    ----------
    excluded_parent_names : frozenset[str]
        H/charge parents verified independently by an admitted site catalog.
    omitted_heavy_names : frozenset[str]
        Reference-only heavy names removed by a validated occurrence boundary.
    external_pairs : frozenset[frozenset[AtomRef]]
        Real peptide pairs already admitted by the polymer context.
    controlled_bond_pairs : frozenset[frozenset[str]]
        Internal site orders verified by that catalog rather than the reference.
    hydrogen_aliases : tuple[tuple[str, str], ...]
        Additional occurrence H naming rules, still checked against geometry.
    """

    excluded_parent_names: frozenset[str] = frozenset()
    omitted_heavy_names: frozenset[str] = frozenset()
    external_pairs: frozenset[frozenset[AtomRef]] = frozenset()
    controlled_bond_pairs: frozenset[frozenset[str]] = frozenset()
    hydrogen_aliases: tuple[tuple[str, str], ...] = ()


class ReferenceHydrogenParentInterpretation(str, Enum):
    """Parent correspondence without an asserted connectivity relationship."""

    REFERENCE_NAME = "reference_name"
    COMPATIBLE_GEOMETRY = "compatible_geometry"


@dataclass(frozen=True, slots=True)
class ReferenceHydrogenInventory:
    """Comparable current H inventory without preparation authority.

    Parameters
    ----------
    reference : IdealizedComponent
        Active chemical reference, independent of H placement support.
    residue_id : ResidueId
        Stable occurrence identity for these observations.
    hydrogen_parents : tuple[tuple[str, str, BondProvenance or ReferenceHydrogenParentInterpretation], ...]
        Current H name, heavy parent and either connection provenance or
        ReferenceHydrogenParentInterpretation. Naming and compatible geometry
        interpretations do not assert a source bond.
    excluded_parent_names : frozenset[str]
        Contextual parents whose multiplicities are verified independently.

    Raises
    ------
    ValueError
        H identities repeat, parents are unknown, or multiplicities exceed the
        reference. Reference/pose conformance is checked by the observation owner;
        this value is not a transferable chemistry-validity token.
    TypeError
        A parent interpretation has no canonical evidence basis.
    """

    reference: IdealizedComponent
    residue_id: ResidueId
    hydrogen_parents: tuple[
        tuple[str, str, BondProvenance | ReferenceHydrogenParentInterpretation], ...
    ]

    excluded_parent_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        names = tuple(name for name, _, _ in self.hydrogen_parents)
        expected = Counter(
            self.reference.hydrogen_anchor_atom_name(atom.atom_name)
            for atom in self.reference.hydrogen_atoms()
            if self.reference.hydrogen_anchor_atom_name(atom.atom_name)
            not in self.excluded_parent_names
        )
        observed = Counter(
            parent
            for _, parent, _ in self.hydrogen_parents
            if parent not in self.excluded_parent_names
        )
        if None in expected or len(set(names)) != len(names):
            raise ValueError(
                "reference H inventory needs unique identities and parents"
            )
        if any(count > expected[parent] for parent, count in observed.items()):
            raise ValueError("observed H parents do not match the reference state")
        if set(names).intersection(self.reference.heavy_atom_names()):
            raise ValueError("reference H identities collide with heavy atoms")
        if any(
            not isinstance(
                basis, (BondProvenance, ReferenceHydrogenParentInterpretation)
            )
            for _, _, basis in self.hydrogen_parents
        ):
            raise TypeError(
                "reference H interpretation needs a canonical evidence basis"
            )

    def coverage_state(self) -> HydrogenCoverageState:
        """Return coverage of the validated reference inventory.

        Returns
        -------
        HydrogenCoverageState
            Complete, empty or proper-subset H coverage under this reference,
            without implying graph realization or generation support.
        """
        expected_count = sum(
            self.reference.hydrogen_anchor_atom_name(atom.atom_name)
            not in self.excluded_parent_names
            for atom in self.reference.hydrogen_atoms()
        )
        observed_count = sum(
            parent not in self.excluded_parent_names
            for _, parent, _ in self.hydrogen_parents
        )
        if observed_count == expected_count:
            return HydrogenCoverageState.COMPLETE
        if not observed_count:
            return HydrogenCoverageState.NONE
        return HydrogenCoverageState.PARTIAL


def assess_reference_hydrogens(
    structure: ProteinStructure,
    residue: ResidueSite,
    reference: IdealizedComponent,
    *,
    scope: ReferenceHydrogenScope | None = None,
) -> ReferenceHydrogenInventory | ReferenceHydrogenFailure:
    """Compare source H with a reference under a validated contextual scope.

    Parameters
    ----------
    structure : ProteinStructure
        Current snapshot with immutable original observations.
    residue : ResidueSite
        Occurrence whose existing H are assessed, never generated.
    reference : IdealizedComponent
        Selected complete reference graph.
    scope : ReferenceHydrogenScope or None
        Context admitted by the role-specific observation owner.

    Returns
    -------
    ReferenceHydrogenInventory or ReferenceHydrogenFailure
        Compared inventory or a typed cause, without topology/preparation rights.
    """
    if scope is None:
        scope = ReferenceHydrogenScope()
    try:
        require_compatible_component_reference(reference)
    except RdkitUnavailableError as error:
        return ReferenceHydrogenFailure("backend_unavailable", str(error))
    except (ValueError, RuntimeError) as error:
        return ReferenceHydrogenFailure("reference_invalid", str(error))

    expected_heavy = {
        atom.atom_name: atom.element
        for atom in reference.heavy_atoms()
        if atom.atom_name not in scope.omitted_heavy_names
    }
    current_heavy = {
        atom.name: atom.element for atom in residue.atom_sites if not atom.is_hydrogen()
    }
    if current_heavy != expected_heavy:
        return ReferenceHydrogenFailure(
            "reference_state_not_covered",
            "reference H validation requires the complete named heavy scaffold "
            "with matching elements",
        )

    constitution = structure.constitution
    incident = structure.topology.incident_bonds_by_atom()
    current_bonds = tuple(
        dict.fromkeys(
            bond
            for atom in residue.atom_sites
            for bond in incident.get(
                constitution.atom_index(AtomRef(residue.residue_id, atom.name)), ()
            )
        )
    )
    snapshots = [(constitution, structure.topology, current_bonds)]
    observation = structure.provenance.ingress.observation
    if observation is not None:
        original = observation.constitution.residue_or_ligand(residue.residue_id)
        if original is not None and (
            original.component_id != residue.component_id
            or any(
                not residue.has_atom_site(atom.name)
                or residue.atom_site(atom.name).element_identity
                != atom.element_identity
                for atom in original.atom_sites
            )
        ):
            return ReferenceHydrogenFailure(
                "source_conflict",
                "reference H validation cannot hide changed original atom identities",
            )
        if any(
            endpoint.atom_ref.residue_id == residue.residue_id
            for connection in observation.excluded_coordination_connections
            for endpoint in connection.endpoint_pair()
        ):
            return ReferenceHydrogenFailure(
                "context_unresolved",
                "read-only reference H validation cannot hide excluded coordination",
            )
        snapshots.append(
            (
                observation.constitution,
                observation.topology,
                observation.bonds_for_residue(residue.residue_id),
            )
        )

    reference_pairs = {
        frozenset((bond.atom_name_1, bond.atom_name_2))
        for bond in reference.heavy_bonds()
    }
    parents: dict[str, tuple[str, BondProvenance]] = {}
    orders: dict[tuple[str, str], int] = {}
    for snapshot_constitution, topology, bonds in snapshots:
        snapshot_residue = snapshot_constitution.residue_or_ligand(residue.residue_id)
        if snapshot_residue is None:
            continue
        for atom in snapshot_residue.atom_sites:
            charge = topology.formal_charge(
                snapshot_constitution.atom_index(AtomRef(residue.residue_id, atom.name))
            )
            expected = reference.atom(atom.name)
            expected_charge = (
                0
                if atom.is_hydrogen()
                else (None if expected is None else expected.formal_charge)
            )
            if atom.name in scope.excluded_parent_names:
                continue
            if charge is not None and charge != expected_charge:
                return ReferenceHydrogenFailure(
                    "reference_state_not_covered",
                    f"known charge at {residue.residue_id.display_token()}.{atom.name} "
                    "differs from the represented reference state",
                )
        for bond in bonds:
            if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                continue
            first, second = (
                snapshot_constitution.atom_ref_at(index)
                for index in bond.endpoint_pair()
            )
            if frozenset((first, second)) in scope.external_pairs:
                if (
                    bond.order in (None, 1)
                    and not bond.aromatic
                    and (
                        bond.relationship_type is BondRelationshipType.COVALENT
                        or (
                            bond.relationship_type is BondRelationshipType.UNKNOWN
                            and bond.provenance is BondProvenance.SOURCE_EXPLICIT
                        )
                    )
                ):
                    continue
                return ReferenceHydrogenFailure(
                    "source_conflict",
                    "source/current peptide relationship contradicts the admitted boundary",
                )
            if (
                first.residue_id != residue.residue_id
                or second.residue_id != residue.residue_id
                or bond.relationship_type is BondRelationshipType.METAL_COORDINATION
            ):
                return ReferenceHydrogenFailure(
                    "context_unresolved",
                    "read-only reference H validation requires an isolated molecular "
                    f"context: {first.display_token()}--{second.display_token()}",
                )
            atoms = tuple(
                snapshot_constitution.atom_site_at(i) for i in bond.endpoint_pair()
            )
            if any(atom.is_hydrogen() for atom in atoms):
                if (
                    sum(atom.is_hydrogen() for atom in atoms) != 1
                    or bond.aromatic
                    or bond.order not in (None, 1)
                    or bond.relationship_type
                    not in {BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN}
                    or (
                        (
                            bond.order is None
                            or bond.relationship_type is BondRelationshipType.UNKNOWN
                        )
                        and bond.provenance is not BondProvenance.SOURCE_EXPLICIT
                    )
                ):
                    return ReferenceHydrogenFailure(
                        "hydrogen_parent_unresolved",
                        "reference H validation requires one local heavy parent per H",
                    )
                hydrogen, parent = (
                    (first, second) if atoms[0].is_hydrogen() else (second, first)
                )
                previous = parents.get(hydrogen.atom_name)
                if previous is not None and previous[0] != parent.atom_name:
                    return ReferenceHydrogenFailure(
                        "hydrogen_parent_unresolved",
                        f"conflicting H parents for {hydrogen.display_token()}",
                    )
                parents[hydrogen.atom_name] = (parent.atom_name, bond.provenance)
                continue
            if frozenset((first.atom_name, second.atom_name)) not in reference_pairs:
                return ReferenceHydrogenFailure(
                    "reference_state_not_covered",
                    "current/source heavy connectivity differs from the reference",
                )
            if (
                frozenset((first.atom_name, second.atom_name))
                in scope.controlled_bond_pairs
            ):
                continue
            if bond.order is not None:
                pair = (
                    (first.atom_name, second.atom_name)
                    if first.atom_name < second.atom_name
                    else (second.atom_name, first.atom_name)
                )
                if pair in orders and orders[pair] != bond.order:
                    return ReferenceHydrogenFailure(
                        "source_conflict",
                        "current and original heavy bond orders disagree",
                    )
                orders[pair] = bond.order

    geometry = structure.residue_geometry(
        constitution.residue_index(residue.residue_id)
    )
    expected_parent_counts = Counter(
        reference.hydrogen_anchor_atom_name(atom.atom_name)
        for atom in reference.hydrogen_atoms()
    )
    prepared_lengths: tuple[tuple[str, float], ...] | None = None
    interpreted = []
    for atom in residue.atom_sites:
        if not atom.is_hydrogen():
            continue
        connection = parents.get(atom.name)
        basis: BondProvenance | ReferenceHydrogenParentInterpretation
        if connection is None:
            position = geometry.position(atom.name)
            distances = {
                name: position.distance_to(geometry.position(name))
                for name in current_heavy
            }
            if any(not isfinite(distance) for distance in distances.values()):
                return ReferenceHydrogenFailure(
                    "hydrogen_parent_unresolved",
                    "reference H parent comparison requires finite local distances "
                    f"for {residue.residue_id.display_token()}.{atom.name}",
                )
            nearest_distance = min(distances.values(), default=float("inf"))
            named_parent = reference.hydrogen_anchor_atom_name(atom.name)
            if named_parent is None:
                named_parent = dict(scope.hydrogen_aliases).get(atom.name)
            if (
                named_parent in distances
                and distances[named_parent] == nearest_distance
            ):
                parent = named_parent
                basis = ReferenceHydrogenParentInterpretation.REFERENCE_NAME
            else:
                closest = tuple(
                    name
                    for name, distance in distances.items()
                    if distance == nearest_distance
                )
                if len(closest) != 1:
                    return ReferenceHydrogenFailure(
                        "hydrogen_parent_unresolved",
                        "no unambiguous local heavy parent for "
                        f"{residue.residue_id.display_token()}.{atom.name}",
                    )
                parent = closest[0]
                if (
                    not expected_parent_counts[parent]
                    and parent not in scope.excluded_parent_names
                ):
                    return ReferenceHydrogenFailure(
                        "reference_state_not_covered",
                        f"observed H at {residue.residue_id.display_token()}.{parent} "
                        "is outside the represented reference state",
                    )
                if prepared_lengths is None:
                    try:
                        prepared_lengths = component_reference_hydrogen_bond_lengths(
                            reference,
                            geometry,
                            bond_order_constraints=orders,
                            omitted_heavy_names=scope.omitted_heavy_names,
                        )
                    except RdkitUnavailableError as error:
                        return ReferenceHydrogenFailure(
                            "backend_unavailable",
                            str(error),
                        )
                    except BondOrderConstraintError as error:
                        return ReferenceHydrogenFailure(
                            "source_conflict",
                            str(error),
                        )
                    except (ValueError, RuntimeError) as error:
                        return ReferenceHydrogenFailure(
                            "hydrogen_parent_unresolved",
                            f"reference H correspondence is unavailable: {error}",
                        )
                if not any(
                    source_hydrogen_bond_length_is_compatible(nearest_distance, length)
                    for anchor, length in prepared_lengths
                    if anchor == parent
                ):
                    return ReferenceHydrogenFailure(
                        "hydrogen_parent_unresolved",
                        f"source H {residue.residue_id.display_token()}.{atom.name} "
                        "has no compatible prepared-reference parent distance",
                    )
                basis = ReferenceHydrogenParentInterpretation.COMPATIBLE_GEOMETRY
        else:
            parent, basis = connection
        interpreted.append((atom.name, parent, basis))
    try:
        inventory = ReferenceHydrogenInventory(
            reference,
            residue.residue_id,
            tuple(interpreted),
            scope.excluded_parent_names,
        )
    except ValueError as error:
        return ReferenceHydrogenFailure(
            "reference_state_not_covered",
            str(error),
        )
    try:
        require_compatible_component_pose(
            reference,
            geometry,
            hydrogen_attachments=tuple(
                (name, parent) for name, parent, _ in interpreted
            ),
            bond_order_constraints=orders,
            allow_partial=bool(scope.omitted_heavy_names),
        )
    except RdkitUnavailableError as error:
        return ReferenceHydrogenFailure("backend_unavailable", str(error))
    except (ValueError, RuntimeError) as error:
        return ReferenceHydrogenFailure("source_conflict", str(error))
    return inventory
