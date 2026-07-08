"""Derived diagnostics for severe nonbonded contacts that look near-covalent."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from protrepair.chemistry import (
    ComponentLibrary,
    ElementRadiusLookup,
    RadiusKind,
    prepare_radius_lookup,
)
from protrepair.diagnostics.clash_pair_generation import (
    ContactDomain,
    iter_candidate_atom_site_pairs,
)
from protrepair.diagnostics.clashes import (
    AtomSite,
    ClashDetectionContext,
    ClashPolicy,
    build_atom_sites,
    build_residue_contexts,
    probable_hydrogen_bond,
    residue_sort_key,
    should_ignore_pair,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId

MINIMUM_NEAR_COVALENT_GRID_CELL_SIZE_ANGSTROM = 1.0


@dataclass(frozen=True, slots=True)
class NearCovalentContactPolicy:
    """Policy for classifying nonbonded pairs as near-covalent contacts."""

    minimum_overlap_angstrom: float = 0.0
    covalent_distance_margin_angstrom: float = 0.45
    ignore_same_residue: bool = True

    def __post_init__(self) -> None:
        if self.minimum_overlap_angstrom < 0.0:
            raise ValueError("minimum_overlap_angstrom must be non-negative")

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
        if self.distance_angstrom < 0.0:
            raise ValueError("near-covalent contacts require non-negative distance")
        if self.covalent_distance_cutoff_angstrom <= 0.0:
            raise ValueError("near-covalent contacts require a positive cutoff")
        if self.overlap_angstrom <= 0.0:
            raise ValueError("near-covalent contacts require positive overlap")


@dataclass(frozen=True, slots=True)
class _NearCovalentCandidateContext:
    """Prepared candidate context for covalent-radius proximity diagnostics."""

    atom_sites: tuple[AtomSite, ...]
    pair_policy: ClashPolicy
    covalent_radius_by_element: Mapping[str, float]

    def __post_init__(self) -> None:
        radius_by_element = dict(self.covalent_radius_by_element)
        if any(radius <= 0.0 for radius in radius_by_element.values()):
            raise ValueError("covalent radii must be positive")

        object.__setattr__(self, "atom_sites", tuple(self.atom_sites))
        object.__setattr__(
            self,
            "covalent_radius_by_element",
            MappingProxyType(radius_by_element),
        )

    def covalent_radius(self, element: str) -> float:
        """Return the cached covalent radius for one atom element."""

        return self.covalent_radius_by_element[element]

    def candidate_atom_site_pairs(
        self,
        *,
        focus_residue_ids: frozenset[ResidueId] | None = None,
    ) -> Iterator[tuple[AtomSite, AtomSite]]:
        """Yield spatially pruned atom pairs before covalent distance checks."""

        for left_site, right_site in iter_candidate_atom_site_pairs(
            self.atom_sites,
            focus_residue_ids=focus_residue_ids,
            policy=self.pair_policy,
        ):
            yield cast(AtomSite, left_site), cast(AtomSite, right_site)


def detect_near_covalent_contacts(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
    context: ClashDetectionContext | None = None,
    focus_residue_ids: frozenset[ResidueId] | None = None,
    pair_policy: ClashPolicy | None = None,
    policy: NearCovalentContactPolicy | None = None,
) -> tuple[NearCovalentContact, ...]:
    """Return severe nonbonded contacts short enough to look near-covalent."""

    if context is not None:
        return detect_near_covalent_contacts_from_context(
            context,
            focus_residue_ids=focus_residue_ids,
            policy=policy,
        )

    if component_library is None:
        raise ValueError(
            "near-covalent contact detection requires either a prepared "
            "context or a component library"
        )

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
    context: ClashDetectionContext,
    *,
    focus_residue_ids: frozenset[ResidueId] | None = None,
    policy: NearCovalentContactPolicy | None = None,
) -> tuple[NearCovalentContact, ...]:
    """Return near-covalent contacts from one prepared clash context."""

    active_policy = NearCovalentContactPolicy() if policy is None else policy
    return _detect_near_covalent_contacts_from_candidate_context(
        _candidate_context_from_clash_context(context, policy=active_policy),
        focus_residue_ids=focus_residue_ids,
        policy=active_policy,
    )


def _is_near_covalent_contact(
    left_site: AtomSite,
    right_site: AtomSite,
    *,
    policy: NearCovalentContactPolicy,
    context: _NearCovalentCandidateContext,
) -> bool:
    """Return whether one atom-site pair resembles an extra covalent bond."""

    if (
        policy.ignore_same_residue
        and left_site.residue_id == right_site.residue_id
    ):
        return False

    covalent_distance_cutoff = _covalent_distance_cutoff_angstrom(
        left_site,
        right_site,
        context=context,
        policy=policy,
    )
    distance = left_site.geometry.distance_to(right_site.geometry)
    overlap_angstrom = covalent_distance_cutoff - distance
    return (
        overlap_angstrom > 0.0
        and overlap_angstrom >= policy.minimum_overlap_angstrom
    )


def _detect_near_covalent_contacts_from_candidate_context(
    context: _NearCovalentCandidateContext,
    *,
    focus_residue_ids: frozenset[ResidueId] | None,
    policy: NearCovalentContactPolicy,
) -> tuple[NearCovalentContact, ...]:
    """Return near-covalent contacts from a covalent-radius candidate context."""

    contacts: list[NearCovalentContact] = []
    for left_site, right_site in context.candidate_atom_site_pairs(
        focus_residue_ids=focus_residue_ids,
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
    pair_policy: ClashPolicy | None,
    policy: NearCovalentContactPolicy,
) -> _NearCovalentCandidateContext:
    """Prepare covalent-radius atom-pair candidates without vdW lookup."""

    active_pair_policy = ClashPolicy() if pair_policy is None else pair_policy
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
            if active_pair_policy.include_hydrogens or atom_site.element != "H"
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
    )


def _candidate_context_from_clash_context(
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
        pair_policy=context.policy,
        covalent_radius_by_element={
            element: covalent_radius_lookup.radius_angstrom(element)
            for element in elements
        },
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

    if not _is_near_covalent_contact(
        left_site,
        right_site,
        policy=policy,
        context=context,
    ):
        return None

    distance = left_site.geometry.distance_to(right_site.geometry)
    if should_ignore_pair(left_site, right_site, policy=context.pair_policy):
        return None

    if probable_hydrogen_bond(left_site, right_site, distance):
        return None

    covalent_distance_cutoff = _covalent_distance_cutoff_angstrom(
        left_site,
        right_site,
        context=context,
        policy=policy,
    )
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
        overlap_angstrom=covalent_distance_cutoff - distance,
    )


def _near_covalent_candidate_cell_size_angstrom(
    elements: tuple[str, ...],
    *,
    covalent_radius_lookup: ElementRadiusLookup,
    policy: NearCovalentContactPolicy,
) -> float:
    """Return a grid cell width that cannot miss near-covalent candidates."""

    if not elements:
        return MINIMUM_NEAR_COVALENT_GRID_CELL_SIZE_ANGSTROM

    return max(
        MINIMUM_NEAR_COVALENT_GRID_CELL_SIZE_ANGSTROM,
        *(
            covalent_radius_lookup.radius_angstrom(left_element)
            + covalent_radius_lookup.radius_angstrom(right_element)
            + policy.covalent_distance_margin_angstrom
            for left_element in elements
            for right_element in elements
        ),
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
