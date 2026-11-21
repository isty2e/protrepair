"""Admission of a fully materialized repair around stationary retained waters."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from math import isfinite

from protrepair.chemistry import (
    ComponentLibrary,
    RadiusKind,
    build_default_component_library,
    prepare_radius_lookup,
)
from protrepair.chemistry.polar_contacts import PolarContactChemistry
from protrepair.diagnostics.clash_pair_generation import ContactDomain
from protrepair.diagnostics.clashes import (
    AtomSite,
    ClashPolicy,
    ResidueContext,
    build_atom_sites,
    candidate_atom_site_pairs,
    probable_hydrogen_bond,
)
from protrepair.diagnostics.events import EventScope, ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.errors import UnknownElementRadiusError
from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueRole
from protrepair.structure.labels import AtomRef
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import BondRelationshipType, TopologyBond

# Numerical tolerance for rebuilt coordinates; well below PDB coordinate precision.
_COMPARISON_EPSILON_ANGSTROM = 1e-6
_PAIR_POLICY = ClashPolicy(include_hydrogen_hydrogen=True, include_ligands=True)


@dataclass(frozen=True, slots=True)
class _ContactClass:
    component: str
    element: str
    altloc: str | None
    charge: int | None
    domain: ContactDomain
    hydrogen_parent: str | None
    acceptor: bool
    neighbors: frozenset[tuple[AtomRef, BondRelationshipType, int | None, bool]]


@dataclass(frozen=True, slots=True)
class _Contact:
    classes: tuple[_ContactClass, _ContactClass]
    penetration: float | None
    positions: tuple[Vec3, ...]


def _contact_class(site: AtomSite, structure: ProteinStructure) -> _ContactClass:
    ref = site.atom_ref()
    index = structure.constitution.atom_index(ref)
    chemistry = site.context.polar_chemistry
    return _ContactClass(
        site.component_id,
        structure.constitution.atom_site_at(index).element_identity.source_symbol,
        site.geometry.altloc,
        structure.topology.formal_charge(index),
        site.domain,
        chemistry.hydrogen_anchors(ref.residue_id).get(ref.atom_name),
        ref.atom_name in chemistry.acceptor_names(ref.residue_id),
        frozenset(
            (
                structure.constitution.atom_ref_at(
                    bond.atom_index_2
                    if bond.atom_index_1 == index
                    else bond.atom_index_1
                ),
                bond.relationship_type,
                bond.order,
                bond.aromatic,
            )
            for bond in structure.topology.incident_bonds_by_atom().get(index, ())
            if bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
        ),
    )


def _contacts(
    structure: ProteinStructure, library: ComponentLibrary
) -> dict[tuple[AtomRef, AtomRef], _Contact]:
    c = structure.constitution
    water_ids = frozenset(w.residue_id for w in c.waters)
    polar = PolarContactChemistry(c, structure.topology, library)
    contexts = tuple(
        ResidueContext(
            site,
            structure.residue_geometry(c.residue_index(site.residue_id)),
            None,
            ContactDomain.WATER
            if site.residue_id in water_ids
            else ContactDomain.RETAINED_NON_POLYMER
            if c.role_at(c.residue_index(site.residue_id)) is ResidueRole.LIGAND
            else ContactDomain.POLYMER,
            None,
            None,
            polar,
        )
        for site in c.residue_slots
    )
    elements = {site.element for site in c.atom_slots}
    lookup = prepare_radius_lookup(elements, RadiusKind.VAN_DER_WAALS)
    lookup.require_complete("retained-water contact comparison")
    radii = {e: lookup.radius_angstrom(e) for e in elements}
    cell_size = max(1.0, 2 * max(radii.values(), default=0.0))
    sites = build_atom_sites(
        contexts, policy=_PAIR_POLICY, candidate_cell_size_angstrom=cell_size
    )
    linked = {
        frozenset(c.atom_ref_at(i) for i in b.endpoint_pair()): b.relationship_type
        for b in structure.topology.bonds
        if b.relationship_type is not BondRelationshipType.HYDROGEN_BOND
    }
    contacts = {}
    for first, second in candidate_atom_site_pairs(sites, focus_residue_ids=water_ids):
        if first.residue_id == second.residue_id:
            continue
        left, right = (
            (first, second) if first.atom_ref() < second.atom_ref() else (second, first)
        )
        pair = (left.atom_ref(), right.atom_ref())
        relationship = linked.get(frozenset(pair))
        if relationship in {
            BondRelationshipType.COVALENT,
            BondRelationshipType.DISULFIDE,
            BondRelationshipType.METAL_COORDINATION,
        }:
            continue
        allowed = _PAIR_POLICY.allowed_distance_angstrom(
            left_van_der_waals_radius_angstrom=radii[left.element],
            right_van_der_waals_radius_angstrom=radii[right.element],
            left_is_hydrogen=left.is_hydrogen(),
            right_is_hydrogen=right.is_hydrogen(),
        )
        distance = left.geometry.distance_to(right.geometry)
        if distance >= allowed or probable_hydrogen_bond(left, right, distance):
            continue
        classes = (_contact_class(left, structure), _contact_class(right, structure))
        positions = [left.geometry.position, right.geometry.position]
        uncertain = relationship is BondRelationshipType.UNKNOWN
        for site, other, cls in ((left, right, classes[0]), (right, left, classes[1])):
            if site.is_hydrogen() and other.element in {"N", "O", "S"}:
                if (
                    other.domain is ContactDomain.WATER
                    and other.atom_name not in polar.acceptor_names(other.residue_id)
                ):
                    uncertain = True
                if cls.hydrogen_parent is None:
                    uncertain = True
                else:
                    positions.append(
                        site.context.residue_geometry.atom_geometry(
                            cls.hydrogen_parent
                        ).position
                    )
        contacts[pair] = _Contact(
            classes, None if uncertain else allowed - distance, tuple(positions)
        )
    return contacts


def _water_bonds(structure: ProteinStructure) -> dict[frozenset[AtomRef], TopologyBond]:
    c = structure.constitution
    water_ids = {water.residue_id for water in c.waters}
    edges = {}
    for bond in structure.topology.bonds:
        refs = frozenset(c.atom_ref_at(i) for i in bond.endpoint_pair())
        if any(ref.residue_id in water_ids for ref in refs):
            # Compare bond facts at semantic endpoints even after a span insertion.
            edges[refs] = replace(
                bond, atom_index_1=AtomIndex(0), atom_index_2=AtomIndex(1)
            )
    return edges


def _water_custody_changed(before: ProteinStructure, after: ProteinStructure) -> bool:
    if before.constitution.waters != after.constitution.waters or _water_bonds(
        before
    ) != _water_bonds(after):
        return True
    for water in before.constitution.waters:
        left = before.constitution.residue_index(water.residue_id)
        right = after.constitution.residue_index(water.residue_id)
        if before.residue_geometry(left) != after.residue_geometry(
            right
        ) or before.residue_formal_charge_by_atom_name(
            left
        ) != after.residue_formal_charge_by_atom_name(right):
            return True
    return False


_RUN_INPUT: ContextVar[ProteinStructure | None] = ContextVar(
    "water_repair_input", default=None
)


@contextmanager
def stationary_water_run(structure: ProteinStructure) -> Iterator[None]:
    """Bind the normalized input across nested repair calls and restore it on exit."""
    token = _RUN_INPUT.set(structure if structure.constitution.waters else None)
    try:
        yield
    finally:
        _RUN_INPUT.reset(token)


def stationary_water_refusal(
    parent: ProteinStructure,
    candidate: ProteinStructure,
    *,
    run_input: ProteinStructure | None = None,
    component_library: ComponentLibrary | None = None,
) -> ValidationIssue | None:
    """Check water custody and pair penetrations before publishing a repair.

    Parameters
    ----------
    parent, candidate : ProteinStructure
        Current branch parent and the fully materialized alternative.
    run_input : ProteinStructure or None
        Normalized workflow input. A standalone operation uses its parent.
    component_library : ComponentLibrary or None
        Active chemistry definitions, or the standard library.

    Returns
    -------
    ValidationIssue or None
        A refusal with its witnesses, or None if the endpoint is admissible.
    """
    bound = _RUN_INPUT.get() if run_input is None else run_input
    root = parent if bound is None else bound
    if not (
        root.constitution.waters
        or parent.constitution.waters
        or candidate.constitution.waters
    ):
        return None
    if root is parent and parent is candidate:
        return None
    baselines = (parent,) if root is parent else (parent, root)
    if any(_water_custody_changed(before, candidate) for before in baselines):
        return ValidationIssue(
            ValidationIssueKind.RETAINED_WATER_REPAIR_REFUSED,
            IssueSeverity.WARNING,
            "Repair withheld: stationary water identity, coordinates or a source connection changed.",
        )
    if candidate is parent:
        return None
    library = component_library or build_default_component_library()
    try:
        for carrier in (*baselines, candidate):
            if any(
                not all(map(isfinite, geom.position))
                for geom in carrier.geometry.atom_geometries
            ):
                return ValidationIssue(
                    ValidationIssueKind.RETAINED_WATER_REPAIR_REFUSED,
                    IssueSeverity.WARNING,
                    "Repair withheld: nonfinite coordinates leave water contact reach unknown.",
                )
        after = _contacts(candidate, library)
        earlier = tuple(_contacts(before, library) for before in baselines)
    except UnknownElementRadiusError as error:
        return ValidationIssue(
            ValidationIssueKind.RETAINED_WATER_REPAIR_REFUSED,
            IssueSeverity.WARNING,
            f"Repair withheld: {error}",
        )
    for pair, endpoint in after.items():
        for baseline in earlier:
            previous = baseline.get(pair)
            if previous == endpoint:
                continue
            if endpoint.penetration is None or (
                previous is not None and previous.penetration is None
            ):
                reason = "water contact chemistry is unresolved"
            else:
                allowance = (
                    previous.penetration
                    if previous is not None and previous.classes == endpoint.classes
                    else 0.0
                )
                if (
                    allowance is not None
                    and endpoint.penetration <= allowance + _COMPARISON_EPSILON_ANGSTROM
                ):
                    continue
                reason = f"new or worsened water clash (penetration {endpoint.penetration:.6g} angstrom)"
            a, b = pair
            return ValidationIssue(
                ValidationIssueKind.RETAINED_WATER_REPAIR_REFUSED,
                IssueSeverity.WARNING,
                f"Repair withheld: {reason}: {a.display_token()} / {b.display_token()}",
                scope=EventScope.for_residue_pair(a.residue_id, b.residue_id),
            )
    return None
