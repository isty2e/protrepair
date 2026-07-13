"""Derived diagnostics for severe nonbonded contacts that look near-covalent."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import InitVar, dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import cast

from protrepair.chemistry import (
    ComponentLibrary,
    ElementRadiusLookup,
    RadiusKind,
    prepare_radius_lookup,
)
from protrepair.diagnostics.clash_pair_generation import (
    ClashPairAtomSite,
    ContactDomain,
    ContactPairPolicy,
    PreparedAtomSitePairIndex,
    SpatialPairPolicy,
    iter_candidate_atom_site_pairs,
)
from protrepair.diagnostics.clashes import (
    AtomSite,
    ClashDetectionContext,
    build_atom_sites,
    build_residue_contexts,
    probable_hydrogen_bond,
    residue_sort_key,
    same_residue_bond_hops,
    should_ignore_pair,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType, StructureTopology

MINIMUM_NEAR_COVALENT_GRID_CELL_SIZE_ANGSTROM = 1.0
_EXPECTED_CLOSE_CONTACT_RELATIONSHIP_TYPES = frozenset(
    {
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
        BondRelationshipType.METAL_COORDINATION,
    }
)


class _PreparedNearCovalentBasisToken:
    """Authorize construction of an internally consistent contact basis."""

    __slots__ = ()


_PREPARED_NEAR_COVALENT_BASIS_TOKEN = _PreparedNearCovalentBasisToken()


@dataclass(frozen=True, slots=True)
class NearCovalentContactPolicy:
    """Policy for classifying nonbonded pairs as near-covalent contacts."""

    minimum_overlap_angstrom: float = 0.0
    covalent_distance_margin_angstrom: float = 0.45
    ignore_same_residue: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.minimum_overlap_angstrom):
            raise ValueError("minimum_overlap_angstrom must be finite")
        if self.minimum_overlap_angstrom < 0.0:
            raise ValueError("minimum_overlap_angstrom must be non-negative")

        if not isfinite(self.covalent_distance_margin_angstrom):
            raise ValueError("covalent_distance_margin_angstrom must be finite")
        if self.covalent_distance_margin_angstrom < 0.0:
            raise ValueError(
                "covalent_distance_margin_angstrom must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class NearCovalentContact:
    """One nonbonded atom pair within a covalent-radius proximity cutoff."""

    left_residue_id: ResidueId
    left_component_id: str
    left_atom_name: str
    left_domain: ContactDomain
    right_residue_id: ResidueId
    right_component_id: str
    right_atom_name: str
    right_domain: ContactDomain
    distance_angstrom: float
    covalent_distance_cutoff_angstrom: float
    overlap_angstrom: float

    def __post_init__(self) -> None:
        if not isfinite(self.distance_angstrom):
            raise ValueError("near-covalent contacts require finite distance")
        if self.distance_angstrom < 0.0:
            raise ValueError("near-covalent contacts require non-negative distance")
        if not isfinite(self.covalent_distance_cutoff_angstrom):
            raise ValueError("near-covalent contacts require a finite cutoff")
        if self.covalent_distance_cutoff_angstrom <= 0.0:
            raise ValueError("near-covalent contacts require a positive cutoff")
        if not isfinite(self.overlap_angstrom):
            raise ValueError("near-covalent contacts require finite overlap")
        if self.overlap_angstrom <= 0.0:
            raise ValueError("near-covalent contacts require positive overlap")


@dataclass(frozen=True, slots=True)
class _NearCovalentCandidateContext:
    """Prepared candidate context for covalent-radius proximity diagnostics."""

    atom_sites: tuple[AtomSite, ...]
    pair_policy: ContactPairPolicy
    covalent_radius_by_element: Mapping[str, float]
    expected_topology_endpoint_pairs: frozenset[tuple[AtomRef, AtomRef]]

    def __post_init__(self) -> None:
        radius_by_element = dict(self.covalent_radius_by_element)
        if any(
            not isfinite(radius) or radius <= 0.0
            for radius in radius_by_element.values()
        ):
            raise ValueError("covalent radii must be finite and positive")

        object.__setattr__(self, "atom_sites", tuple(self.atom_sites))
        object.__setattr__(
            self,
            "covalent_radius_by_element",
            MappingProxyType(radius_by_element),
        )
        expected_endpoint_pairs = frozenset(self.expected_topology_endpoint_pairs)
        if any(
            left_ref >= right_ref
            for left_ref, right_ref in expected_endpoint_pairs
        ):
            raise ValueError(
                "expected topology endpoint pairs must be distinct and ordered"
            )
        object.__setattr__(
            self,
            "expected_topology_endpoint_pairs",
            expected_endpoint_pairs,
        )

    def covalent_radius(self, element: str) -> float:
        """Return the cached covalent radius for one atom element."""

        return self.covalent_radius_by_element[element]

    def candidate_atom_site_pairs(
        self,
        *,
        focus_residue_ids: frozenset[ResidueId] | None = None,
        include_same_residue_heavy_pairs: bool = False,
        prepared_pair_index: PreparedAtomSitePairIndex | None = None,
    ) -> Iterator[tuple[AtomSite, AtomSite]]:
        """Yield spatially pruned atom pairs before covalent distance checks."""

        if prepared_pair_index is not None:
            prepared_pair_index.require_compatible(
                cast(tuple[ClashPairAtomSite, ...], self.atom_sites),
                focus_residue_ids=focus_residue_ids,
            )
            for left_site, right_site in prepared_pair_index.candidate_pairs(
                policy=cast(SpatialPairPolicy, self.pair_policy),
                include_same_residue_heavy_pairs=include_same_residue_heavy_pairs,
            ):
                yield cast(AtomSite, left_site), cast(AtomSite, right_site)
            return

        for left_site, right_site in iter_candidate_atom_site_pairs(
            self.atom_sites,
            focus_residue_ids=focus_residue_ids,
            policy=self.pair_policy,
            include_same_residue_heavy_pairs=include_same_residue_heavy_pairs,
        ):
            yield cast(AtomSite, left_site), cast(AtomSite, right_site)

    def topology_expects_close_contact(
        self,
        left_site: AtomSite,
        right_site: AtomSite,
    ) -> bool:
        """Return whether canonical topology explains this close atom pair."""

        endpoint_pair = _canonical_atom_ref_pair(
            left_site.atom_ref(),
            right_site.atom_ref(),
        )
        return endpoint_pair in self.expected_topology_endpoint_pairs


@dataclass(frozen=True, slots=True)
class NearCovalentContactBasis:
    """Coordinate-independent near-covalent facts for one topology snapshot."""

    pair_policy: ContactPairPolicy
    contact_policy: NearCovalentContactPolicy
    constitution: StructureConstitution = field(repr=False, compare=False)
    topology: StructureTopology = field(repr=False, compare=False)
    covalent_radius_by_element: Mapping[str, float]
    candidate_cell_size_angstrom: float
    _construction_token: InitVar[_PreparedNearCovalentBasisToken | None] = None
    expected_topology_endpoint_pairs: frozenset[tuple[AtomRef, AtomRef]] = field(
        init=False,
    )

    def __post_init__(
        self,
        _construction_token: _PreparedNearCovalentBasisToken | None,
    ) -> None:
        if _construction_token is not _PREPARED_NEAR_COVALENT_BASIS_TOKEN:
            raise ValueError(
                "near-covalent bases must be created by a preparation factory"
            )

        if not self.topology.is_aligned_to(self.constitution):
            raise ValueError(
                "near-covalent basis topology must align with its constitution"
            )
        if (
            not isfinite(self.candidate_cell_size_angstrom)
            or self.candidate_cell_size_angstrom <= 0.0
        ):
            raise ValueError(
                "near-covalent basis candidate cell size must be finite and positive"
            )
        radius_by_element = dict(self.covalent_radius_by_element)
        if any(
            not isfinite(radius) or radius <= 0.0
            for radius in radius_by_element.values()
        ):
            raise ValueError("near-covalent basis radii must be finite and positive")
        required_cell_size_angstrom = _near_covalent_candidate_cell_size_from_radii(
            radius_by_element.values(),
            policy=self.contact_policy,
        )
        if self.candidate_cell_size_angstrom < required_cell_size_angstrom:
            raise ValueError(
                "near-covalent basis candidate cell size must cover the contact cutoff"
            )
        expected_endpoint_pairs = _expected_topology_endpoint_pairs(
            constitution=self.constitution,
            topology=self.topology,
        )

        object.__setattr__(
            self,
            "covalent_radius_by_element",
            MappingProxyType(radius_by_element),
        )
        object.__setattr__(
            self,
            "expected_topology_endpoint_pairs",
            expected_endpoint_pairs,
        )

    def require_compatible_structure(self, structure: ProteinStructure) -> None:
        """Reject reuse after constitution or topology replacement."""

        if (
            structure.constitution is not self.constitution
            and structure.constitution != self.constitution
        ):
            raise ValueError(
                "near-covalent basis requires its original immutable constitution"
            )
        if (
            structure.topology is not self.topology
            and structure.topology != self.topology
        ):
            raise ValueError(
                "near-covalent basis requires its original immutable topology"
            )

    def bind_context(
        self,
        context: ClashDetectionContext,
    ) -> _NearCovalentCandidateContext:
        """Project one compatible clash frame into near-covalent semantics."""

        if (
            context.constitution is not self.constitution
            and context.constitution != self.constitution
        ):
            raise ValueError(
                "near-covalent basis requires a clash context for its original "
                "immutable constitution"
            )
        if context.topology is not self.topology and context.topology != self.topology:
            raise ValueError(
                "near-covalent basis requires a clash context for its original "
                "immutable topology"
            )
        if context.policy.as_contact_pair_policy() != self.pair_policy:
            raise ValueError("near-covalent basis requires a matching pair policy")
        if context.candidate_cell_size_angstrom < self.candidate_cell_size_angstrom:
            raise ValueError(
                "near-covalent basis requires a sufficiently large spatial cell"
            )

        return _NearCovalentCandidateContext(
            atom_sites=context.atom_sites,
            pair_policy=self.pair_policy,
            covalent_radius_by_element=self.covalent_radius_by_element,
            expected_topology_endpoint_pairs=self.expected_topology_endpoint_pairs,
        )


def _candidate_elements(
    constitution: StructureConstitution,
    *,
    pair_policy: ContactPairPolicy,
) -> tuple[str, ...]:
    """Return deterministic element coverage for one contact-pair scope."""

    return tuple(
        dict.fromkeys(
            atom_site.element
            for residue_site in constitution.iter_residues(
                include_ligands=pair_policy.include_ligands
            )
            for atom_site in residue_site.atom_sites
            if pair_policy.include_hydrogens or not atom_site.is_hydrogen()
        )
    )


def prepare_near_covalent_contact_basis(
    structure: ProteinStructure,
    *,
    pair_policy: ContactPairPolicy | None = None,
    policy: NearCovalentContactPolicy | None = None,
) -> NearCovalentContactBasis:
    """Prepare reusable near-covalent facts for one topology snapshot."""

    active_pair_policy = (
        ContactPairPolicy()
        if pair_policy is None
        else pair_policy.as_contact_pair_policy()
    )
    active_policy = NearCovalentContactPolicy() if policy is None else policy
    elements = _candidate_elements(
        structure.constitution,
        pair_policy=active_pair_policy,
    )
    covalent_radius_lookup = prepare_radius_lookup(elements, RadiusKind.COVALENT)
    covalent_radius_lookup.require_complete("near-covalent contact basis")
    return NearCovalentContactBasis(
        pair_policy=active_pair_policy,
        contact_policy=active_policy,
        constitution=structure.constitution,
        topology=structure.topology,
        covalent_radius_by_element={
            element: covalent_radius_lookup.radius_angstrom(element)
            for element in elements
        },
        candidate_cell_size_angstrom=_near_covalent_candidate_cell_size_angstrom(
            elements,
            covalent_radius_lookup=covalent_radius_lookup,
            policy=active_policy,
        ),
        _construction_token=_PREPARED_NEAR_COVALENT_BASIS_TOKEN,
    )


def detect_near_covalent_contacts(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
    focus_residue_ids: frozenset[ResidueId] | None = None,
    pair_policy: ContactPairPolicy | None = None,
    policy: NearCovalentContactPolicy | None = None,
) -> tuple[NearCovalentContact, ...]:
    """Return severe nonbonded contacts short enough to look near-covalent."""

    if component_library is None:
        raise ValueError("near-covalent contact detection requires a component library")

    active_policy = NearCovalentContactPolicy() if policy is None else policy
    return _detect_near_covalent_contacts_from_candidate_context(
        _prepare_near_covalent_candidate_context(
            structure,
            component_library=component_library,
            pair_policy=pair_policy,
            policy=active_policy,
        ),
        focus_residue_ids=focus_residue_ids,
        policy=active_policy,
    )


def detect_near_covalent_contacts_from_context(
    structure: ProteinStructure,
    context: ClashDetectionContext,
    *,
    focus_residue_ids: frozenset[ResidueId] | None = None,
    policy: NearCovalentContactPolicy | None = None,
    basis: NearCovalentContactBasis | None = None,
    prepared_pair_index: PreparedAtomSitePairIndex | None = None,
) -> tuple[NearCovalentContact, ...]:
    """Return near-covalent contacts from one prepared clash context."""

    context.require_compatible_structure(structure)
    if policy is None:
        active_policy = (
            NearCovalentContactPolicy()
            if basis is None
            else basis.contact_policy
        )
    else:
        active_policy = policy
    if basis is None:
        if prepared_pair_index is not None:
            raise ValueError(
                "prepared near-covalent pair indexes require a near-covalent basis"
            )
        candidate_context = _candidate_context_from_clash_context(
            structure, context, policy=active_policy
        )
    else:
        if active_policy != basis.contact_policy:
            raise ValueError(
                "near-covalent basis requires a matching contact policy"
            )
        basis.require_compatible_structure(structure)
        candidate_context = basis.bind_context(context)

    return _detect_near_covalent_contacts_from_candidate_context(
        candidate_context,
        focus_residue_ids=focus_residue_ids,
        policy=active_policy,
        prepared_pair_index=prepared_pair_index,
    )


def _detect_near_covalent_contacts_from_candidate_context(
    context: _NearCovalentCandidateContext,
    *,
    focus_residue_ids: frozenset[ResidueId] | None,
    policy: NearCovalentContactPolicy,
    prepared_pair_index: PreparedAtomSitePairIndex | None = None,
) -> tuple[NearCovalentContact, ...]:
    """Return near-covalent contacts from a covalent-radius candidate context."""

    contacts: list[NearCovalentContact] = []
    for left_site, right_site in context.candidate_atom_site_pairs(
        focus_residue_ids=focus_residue_ids,
        include_same_residue_heavy_pairs=not policy.ignore_same_residue,
        prepared_pair_index=prepared_pair_index,
    ):
        contact = _near_covalent_contact_for_atom_site_pair(
            left_site,
            right_site,
            context=context,
            policy=policy,
        )
        if contact is not None:
            contacts.append(contact)

    contacts.sort(
        key=lambda contact: (
            residue_sort_key(contact.left_residue_id),
            residue_sort_key(contact.right_residue_id),
            contact.left_atom_name,
            contact.right_atom_name,
        )
    )
    return tuple(contacts)


def _prepare_near_covalent_candidate_context(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    pair_policy: ContactPairPolicy | None,
    policy: NearCovalentContactPolicy,
) -> _NearCovalentCandidateContext:
    """Prepare covalent-radius atom-pair candidates without vdW lookup."""

    active_pair_policy = (
        ContactPairPolicy()
        if pair_policy is None
        else pair_policy.as_contact_pair_policy()
    )
    residue_contexts = build_residue_contexts(
        structure,
        component_library=component_library,
        include_ligands=active_pair_policy.include_ligands,
    )
    elements = tuple(
        dict.fromkeys(
            atom_site.element
            for residue_context in residue_contexts
            for atom_site in residue_context.residue_site.atom_sites
            if active_pair_policy.include_hydrogens or not atom_site.is_hydrogen()
        )
    )
    covalent_radius_lookup = prepare_radius_lookup(elements, RadiusKind.COVALENT)
    covalent_radius_lookup.require_complete("near-covalent contact detection")
    cell_size_angstrom = _near_covalent_candidate_cell_size_angstrom(
        elements,
        covalent_radius_lookup=covalent_radius_lookup,
        policy=policy,
    )
    return _NearCovalentCandidateContext(
        atom_sites=build_atom_sites(
            residue_contexts,
            policy=active_pair_policy,
            candidate_cell_size_angstrom=cell_size_angstrom,
        ),
        pair_policy=active_pair_policy,
        covalent_radius_by_element={
            element: covalent_radius_lookup.radius_angstrom(element)
            for element in elements
        },
        expected_topology_endpoint_pairs=_expected_topology_endpoint_pairs(
            constitution=structure.constitution,
            topology=structure.topology,
        ),
    )


def _candidate_context_from_clash_context(
    structure: ProteinStructure,
    context: ClashDetectionContext,
    *,
    policy: NearCovalentContactPolicy,
) -> _NearCovalentCandidateContext:
    """Project a clash context into a covalent-radius candidate context."""

    elements = tuple(dict.fromkeys(site.element for site in context.atom_sites))
    covalent_radius_lookup = prepare_radius_lookup(elements, RadiusKind.COVALENT)
    covalent_radius_lookup.require_complete("near-covalent contact detection")
    cell_size_angstrom = _near_covalent_candidate_cell_size_angstrom(
        elements,
        covalent_radius_lookup=covalent_radius_lookup,
        policy=policy,
    )
    actual_cell_size_angstrom = max(
        context.candidate_cell_size_angstrom,
        cell_size_angstrom,
    )
    return _NearCovalentCandidateContext(
        atom_sites=_atom_sites_with_cell_size(
            context,
            cell_size_angstrom=actual_cell_size_angstrom,
        ),
        pair_policy=context.policy.as_contact_pair_policy(),
        covalent_radius_by_element={
            element: covalent_radius_lookup.radius_angstrom(element)
            for element in elements
        },
        expected_topology_endpoint_pairs=_expected_topology_endpoint_pairs(
            constitution=structure.constitution,
            topology=structure.topology,
        ),
    )


def _atom_sites_with_cell_size(
    context: ClashDetectionContext,
    *,
    cell_size_angstrom: float,
) -> tuple[AtomSite, ...]:
    """Return atom sites with grid cells sized for covalent-radius searches."""

    if context.candidate_cell_size_angstrom == cell_size_angstrom:
        return context.atom_sites

    return tuple(
        AtomSite(
            atom_name=site.atom_name,
            element=site.element,
            geometry=site.geometry,
            context=site.context,
            grid_cell_size_angstrom=cell_size_angstrom,
        )
        for site in context.atom_sites
    )


def _near_covalent_contact_for_atom_site_pair(
    left_site: AtomSite,
    right_site: AtomSite,
    *,
    context: _NearCovalentCandidateContext,
    policy: NearCovalentContactPolicy,
) -> NearCovalentContact | None:
    """Return a near-covalent contact for one atom-site pair if admitted."""

    same_residue = left_site.residue_id == right_site.residue_id
    if same_residue and policy.ignore_same_residue:
        return None

    if context.topology_expects_close_contact(left_site, right_site):
        return None

    if same_residue:
        bond_hops = same_residue_bond_hops(left_site, right_site)
        if (
            bond_hops is not None
            and bond_hops <= context.pair_policy.ignore_same_residue_bond_hops
        ):
            return None
    elif should_ignore_pair(left_site, right_site, policy=context.pair_policy):
        return None

    distance = left_site.geometry.distance_to(right_site.geometry)
    if probable_hydrogen_bond(left_site, right_site, distance):
        return None

    covalent_distance_cutoff = _covalent_distance_cutoff_angstrom(
        left_site,
        right_site,
        context=context,
        policy=policy,
    )
    overlap_angstrom = covalent_distance_cutoff - distance
    if (
        overlap_angstrom <= 0.0
        or overlap_angstrom < policy.minimum_overlap_angstrom
    ):
        return None

    return NearCovalentContact(
        left_residue_id=left_site.residue_id,
        left_component_id=left_site.component_id,
        left_atom_name=left_site.atom_name,
        left_domain=left_site.domain,
        right_residue_id=right_site.residue_id,
        right_component_id=right_site.component_id,
        right_atom_name=right_site.atom_name,
        right_domain=right_site.domain,
        distance_angstrom=distance,
        covalent_distance_cutoff_angstrom=covalent_distance_cutoff,
        overlap_angstrom=overlap_angstrom,
    )


def _expected_topology_endpoint_pairs(
    *,
    constitution: StructureConstitution,
    topology: StructureTopology,
) -> frozenset[tuple[AtomRef, AtomRef]]:
    """Project topology relationships that explain legitimate close contacts."""

    return frozenset(
        _canonical_atom_ref_pair(
            constitution.atom_ref_at(bond.atom_index_1),
            constitution.atom_ref_at(bond.atom_index_2),
        )
        for bond in topology.bonds
        if bond.relationship_type in _EXPECTED_CLOSE_CONTACT_RELATIONSHIP_TYPES
    )


def _canonical_atom_ref_pair(
    atom_ref_1: AtomRef,
    atom_ref_2: AtomRef,
) -> tuple[AtomRef, AtomRef]:
    """Return one ordered canonical atom-reference pair."""

    if atom_ref_1 == atom_ref_2:
        raise ValueError("atom-reference pairs require distinct endpoints")
    if atom_ref_1 < atom_ref_2:
        return (atom_ref_1, atom_ref_2)
    return (atom_ref_2, atom_ref_1)


def _near_covalent_candidate_cell_size_angstrom(
    elements: tuple[str, ...],
    *,
    covalent_radius_lookup: ElementRadiusLookup,
    policy: NearCovalentContactPolicy,
) -> float:
    """Return a grid cell width that cannot miss near-covalent candidates."""

    return _near_covalent_candidate_cell_size_from_radii(
        (
            covalent_radius_lookup.radius_angstrom(element)
            for element in elements
        ),
        policy=policy,
    )


def _near_covalent_candidate_cell_size_from_radii(
    radii_angstrom: Iterable[float],
    *,
    policy: NearCovalentContactPolicy,
) -> float:
    """Return the complete near-covalent search width for known radii."""

    maximum_radius_angstrom = max(radii_angstrom, default=0.0)
    return max(
        MINIMUM_NEAR_COVALENT_GRID_CELL_SIZE_ANGSTROM,
        (2.0 * maximum_radius_angstrom)
        + policy.covalent_distance_margin_angstrom,
    )


def _covalent_distance_cutoff_angstrom(
    left_site: AtomSite,
    right_site: AtomSite,
    *,
    context: _NearCovalentCandidateContext,
    policy: NearCovalentContactPolicy,
) -> float:
    """Return the near-covalent distance cutoff for one atom-site pair."""

    return (
        context.covalent_radius(left_site.element)
        + context.covalent_radius(right_site.element)
        + policy.covalent_distance_margin_angstrom
    )
