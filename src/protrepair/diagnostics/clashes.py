"""Hydrogen-aware steric clash diagnostics over canonical structures."""

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from math import floor, sqrt
from types import MappingProxyType
from typing import cast

from typing_extensions import Self

from protrepair.chemistry import (
    ComponentLibrary,
    ResidueTemplate,
    van_der_waals_radius_angstrom,
)
from protrepair.diagnostics import clash_topology_rules
from protrepair.diagnostics.clash_pair_generation import (
    ClashPairAtomSite,
    ClashPairPolicy,
    ContactDomain,
    iter_candidate_atom_site_pairs,
    pair_can_be_rejected_before_distance,
)
from protrepair.diagnostics.events import EventScope, ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.structure.address_space import StructureAddressSpaceKey
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import ResidueIndex

HYDROGEN_ANCHOR_DISTANCE_CUTOFF_ANGSTROM = 1.45
DISULFIDE_BOND_DISTANCE_CUTOFF_ANGSTROM = 3.0
CLASH_GRID_CELL_SIZE_ANGSTROM = 4.0
HYDROGEN_BOND_MIN_DISTANCE_ANGSTROM = 1.6
HYDROGEN_BOND_MAX_DISTANCE_ANGSTROM = 2.4
DONOR_ELEMENTS = frozenset({"N", "O", "S"})
ACCEPTOR_ELEMENTS = frozenset({"N", "O", "S"})
NEIGHBOR_CELL_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
)


@dataclass(frozen=True, slots=True)
class ClashPolicy:
    """Policy controlling steric clash detection sensitivity and scope."""

    heavy_overlap_tolerance_angstrom: float = 0.60
    hydrogen_overlap_tolerance_angstrom: float = 0.90
    include_hydrogens: bool = True
    include_hydrogen_hydrogen: bool = False
    include_ligands: bool = False
    ignore_same_residue_bond_hops: int = 2
    ignore_adjacent_polymer_bond_hops: int = 3

    def __post_init__(self) -> None:
        if self.heavy_overlap_tolerance_angstrom < 0:
            raise ValueError("heavy overlap tolerance must be non-negative")

        if self.hydrogen_overlap_tolerance_angstrom < 0:
            raise ValueError("hydrogen overlap tolerance must be non-negative")

        if self.ignore_same_residue_bond_hops < 0:
            raise ValueError("ignore_same_residue_bond_hops must be non-negative")

        if self.ignore_adjacent_polymer_bond_hops < 0:
            raise ValueError("ignore_adjacent_polymer_bond_hops must be non-negative")

    def required_overlap(self, left_element: str, right_element: str) -> float:
        """Return the minimum overlap required for a pair to count as a clash."""

        if left_element == "H" or right_element == "H":
            return self.hydrogen_overlap_tolerance_angstrom

        return self.heavy_overlap_tolerance_angstrom


@dataclass(frozen=True, slots=True)
class ResidueContext:
    """Runtime residue context used by clash diagnostics."""

    residue_site: ResidueSite
    residue_geometry: ResidueGeometry
    template: ResidueTemplate | None
    domain: ContactDomain
    chain_index: int | None
    residue_index: int | None
    _hydrogen_anchor_by_name: Mapping[str, str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", ContactDomain.normalize(self.domain))
        if self._hydrogen_anchor_by_name is None:
            return

        object.__setattr__(
            self,
            "_hydrogen_anchor_by_name",
            MappingProxyType(dict(self._hydrogen_anchor_by_name)),
        )

    @property
    def hydrogen_anchor_by_name(self) -> Mapping[str, str]:
        """Return inferred hydrogen anchors for the current residue geometry."""

        hydrogen_anchor_by_name = self._hydrogen_anchor_by_name
        if hydrogen_anchor_by_name is None:
            hydrogen_anchor_by_name = infer_hydrogen_anchors(
                self.residue_site,
                self.residue_geometry,
            )
            object.__setattr__(
                self,
                "_hydrogen_anchor_by_name",
                hydrogen_anchor_by_name,
            )

        return hydrogen_anchor_by_name

    @property
    def residue_id(self) -> ResidueId:
        """Return the canonical residue identifier."""

        return self.residue_site.residue_id

    @property
    def component_id(self) -> str:
        """Return the residue component identifier."""

        return self.residue_site.component_id

    def is_adjacent_polymer_residue(self, other: Self) -> bool:
        """Return whether two polymer residues are immediate chain neighbors."""

        return (
            self.domain is ContactDomain.POLYMER
            and other.domain is ContactDomain.POLYMER
            and self.chain_index is not None
            and self.chain_index == other.chain_index
            and self.residue_index is not None
            and other.residue_index is not None
            and abs(self.residue_index - other.residue_index) == 1
        )


@dataclass(frozen=True, slots=True)
class _ResidueContextBasis:
    """Coordinate-independent residue context used to bind clash frames."""

    residue_site: ResidueSite
    template: ResidueTemplate | None
    domain: ContactDomain
    chain_index: int | None
    residue_index: int | None
    residue_slot_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", ContactDomain.normalize(self.domain))

    @property
    def residue_id(self) -> ResidueId:
        """Return the canonical residue identifier."""

        return self.residue_site.residue_id

    @property
    def component_id(self) -> str:
        """Return the residue component identifier."""

        return self.residue_site.component_id


@dataclass(frozen=True, slots=True)
class _AtomSiteBasis:
    """Coordinate-independent atom-site metadata used to bind clash frames."""

    residue_context_index: int
    atom_name: str
    element: str


@dataclass(frozen=True, slots=True)
class AtomSite:
    """One atom paired with its residue-local diagnostic context."""

    atom_name: str
    element: str
    geometry: AtomGeometry
    context: ResidueContext
    residue_id: ResidueId = field(init=False, compare=False)
    component_id: str = field(init=False, compare=False)
    domain: ContactDomain = field(init=False, compare=False)
    grid_cell: tuple[int, int, int] = field(init=False, compare=False)
    is_hydrogen_atom: bool = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "residue_id", self.context.residue_id)
        object.__setattr__(self, "component_id", self.context.component_id)
        object.__setattr__(self, "domain", self.context.domain)
        object.__setattr__(self, "grid_cell", _cell_id_from_geometry(self.geometry))
        object.__setattr__(self, "is_hydrogen_atom", self.element == "H")

    def is_hydrogen(self) -> bool:
        """Return whether the atom site is a hydrogen."""

        return self.is_hydrogen_atom


@dataclass(frozen=True, slots=True)
class ClashDetectionFrame:
    """Coordinate-bound clash detection frame for one compatible geometry."""

    atom_sites: tuple[AtomSite, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_sites", tuple(self.atom_sites))


@dataclass(frozen=True, slots=True)
class ClashDetectionContext:
    """Prepared computational context for repeated clash diagnostics.

    This context caches representation and scalar lookup work for one immutable
    structure view plus one clash policy. It is not a chemistry authority: bonded
    topology still comes from residue templates and residue-local inferred
    hydrogen anchors.
    """

    atom_sites: tuple[AtomSite, ...]
    policy: ClashPolicy
    van_der_waals_radius_by_element: Mapping[str, float]
    allowed_distance_by_element_pair: Mapping[tuple[str, str], float] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_sites", tuple(self.atom_sites))
        radius_by_element = dict(self.van_der_waals_radius_by_element)
        object.__setattr__(
            self, "van_der_waals_radius_by_element", MappingProxyType(radius_by_element)
        )
        object.__setattr__(
            self,
            "allowed_distance_by_element_pair",
            MappingProxyType(
                _allowed_distance_by_element_pair(
                    radius_by_element,
                    policy=self.policy,
                )
            ),
        )

    def van_der_waals_radius(self, element: str) -> float:
        """Return the cached van der Waals radius for one atom element."""

        return self.van_der_waals_radius_by_element[element]

    def candidate_atom_site_pairs(
        self,
        *,
        focus_residue_ids: frozenset[ResidueId] | None = None,
    ) -> Iterator[tuple[AtomSite, AtomSite]]:
        """Yield policy-filtered candidate atom pairs before distance checks."""

        yield from _iter_candidate_atom_site_pairs(
            self.atom_sites,
            focus_residue_ids=focus_residue_ids,
            policy=self.policy,
        )

    def detect_clashes(
        self,
        *,
        focus_residue_ids: frozenset[ResidueId] | None = None,
    ) -> "ClashReport":
        """Return clashes from this prepared context."""

        clashes: list[StericClash] = []
        for left_site, right_site in self.candidate_atom_site_pairs(
            focus_residue_ids=focus_residue_ids,
        ):
            clash = _clash_for_atom_site_pair(left_site, right_site, context=self)
            if clash is not None:
                clashes.append(clash)

        clashes.sort(
            key=lambda clash: (
                residue_sort_key(clash.left_residue_id),
                residue_sort_key(clash.right_residue_id),
                clash.left_atom_name,
                clash.right_atom_name,
            )
        )
        return ClashReport(clashes=tuple(clashes))

    def has_clashes(
        self,
        *,
        focus_residue_ids: frozenset[ResidueId] | None = None,
    ) -> bool:
        """Return whether this prepared context contains at least one clash."""

        for left_site, right_site in self.candidate_atom_site_pairs(
            focus_residue_ids=focus_residue_ids,
        ):
            if _clash_for_atom_site_pair(left_site, right_site, context=self):
                return True

        return False


@dataclass(frozen=True, slots=True)
class ClashDetectionBasis:
    """Coordinate-independent clash detection basis for one address space.

    The basis owns constitution/component/policy facts only. Coordinate-derived
    facts, including inferred hydrogen anchors, are bound later for each
    candidate geometry to preserve the existing clash semantics.
    """

    residue_context_bases: tuple[_ResidueContextBasis, ...]
    atom_site_bases: tuple[_AtomSiteBasis, ...]
    policy: ClashPolicy
    constitution_address_space_key: StructureAddressSpaceKey
    van_der_waals_radius_by_element: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residue_context_bases",
            tuple(self.residue_context_bases),
        )
        object.__setattr__(self, "atom_site_bases", tuple(self.atom_site_bases))
        object.__setattr__(
            self,
            "constitution_address_space_key",
            tuple(self.constitution_address_space_key),
        )
        object.__setattr__(
            self,
            "van_der_waals_radius_by_element",
            MappingProxyType(dict(self.van_der_waals_radius_by_element)),
        )

    def is_compatible_with(self, structure: ProteinStructure) -> bool:
        """Return whether this basis can be bound to the structure geometry."""

        return (
            structure.constitution.address_space_key
            == self.constitution_address_space_key
        )

    def bind_frame(self, structure: ProteinStructure) -> ClashDetectionFrame:
        """Bind this reusable basis to coordinate-derived atom sites."""

        if not self.is_compatible_with(structure):
            raise ValueError(
                "clash detection basis requires a matching structure address space"
            )

        residue_contexts = bind_residue_contexts(
            structure,
            self.residue_context_bases,
        )
        return ClashDetectionFrame(
            atom_sites=tuple(
                AtomSite(
                    atom_name=atom_site_basis.atom_name,
                    element=atom_site_basis.element,
                    geometry=(
                        residue_contexts[
                            atom_site_basis.residue_context_index
                        ].residue_geometry.atom_geometry(atom_site_basis.atom_name)
                    ),
                    context=residue_contexts[atom_site_basis.residue_context_index],
                )
                for atom_site_basis in self.atom_site_bases
            )
        )

    def bind_context(self, structure: ProteinStructure) -> ClashDetectionContext:
        """Bind this reusable basis to one coordinate-bound detection context."""

        frame = self.bind_frame(structure)
        return ClashDetectionContext(
            atom_sites=frame.atom_sites,
            policy=self.policy,
            van_der_waals_radius_by_element=self.van_der_waals_radius_by_element,
        )


@dataclass(frozen=True, slots=True)
class StericClash:
    """One steric clash between two atom sites."""

    left_residue_id: ResidueId
    left_component_id: str
    left_atom_name: str
    left_domain: ContactDomain
    right_residue_id: ResidueId
    right_component_id: str
    right_atom_name: str
    right_domain: ContactDomain
    distance_angstrom: float
    allowed_distance_angstrom: float
    overlap_angstrom: float

    def __post_init__(self) -> None:
        if self.overlap_angstrom <= 0:
            raise ValueError("steric clashes must have positive overlap")


@dataclass(frozen=True, slots=True)
class ClashReport:
    """Structured clash report for a processed structure."""

    clashes: tuple[StericClash, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "clashes", tuple(self.clashes))

    def is_empty(self) -> bool:
        """Return whether the report contains no clashes."""

        return not self.clashes

    def to_issues(self) -> tuple[ValidationIssue, ...]:
        """Project atom-pair clashes into aggregated validation issues."""

        grouped_clashes: dict[tuple[ResidueId, ResidueId], list[StericClash]] = (
            defaultdict(list)
        )
        for clash in self.clashes:
            grouped_clashes[(clash.left_residue_id, clash.right_residue_id)].append(
                clash
            )

        issues: list[ValidationIssue] = []
        for (left_residue_id, right_residue_id), clashes in sorted(
            grouped_clashes.items(),
            key=lambda item: (
                residue_sort_key(item[0][0]),
                residue_sort_key(item[0][1]),
            ),
        ):
            worst_clash = max(clashes, key=lambda clash: clash.overlap_angstrom)
            pair_count = len(clashes)
            if left_residue_id == right_residue_id:
                message = (
                    f"{left_residue_id.display_token()} contains {pair_count} steric "
                    f"clash pair(s); worst overlap is "
                    f"{worst_clash.left_atom_name}-{worst_clash.right_atom_name} "
                    f"({worst_clash.overlap_angstrom:.2f} A)"
                )
            else:
                message = (
                    f"{left_residue_id.display_token()} clashes with "
                    f"{right_residue_id.display_token()} in {pair_count} atom pair(s); "
                    f"worst overlap is {worst_clash.left_atom_name}-"
                    f"{worst_clash.right_atom_name} "
                    f"({worst_clash.overlap_angstrom:.2f} A)"
                )

            issues.append(
                ValidationIssue(
                    kind=ValidationIssueKind.STERIC_CLASH,
                    severity=IssueSeverity.WARNING,
                    scope=(
                        EventScope.for_residue(left_residue_id)
                        if left_residue_id == right_residue_id
                        else EventScope.for_residue_pair(
                            left_residue_id,
                            right_residue_id,
                        )
                    ),
                    message=message,
                )
            )

        return tuple(issues)


def detect_clashes(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    policy: ClashPolicy | None = None,
) -> ClashReport:
    """Return structured steric clashes for a canonical structure."""

    return detect_clashes_from_context(
        prepare_clash_detection_context(
            structure,
            component_library=component_library,
            policy=policy,
        ),
    )


def detect_clashes_involving_residues(
    structure: ProteinStructure,
    *,
    residue_ids: frozenset[ResidueId],
    component_library: ComponentLibrary,
    policy: ClashPolicy | None = None,
) -> ClashReport:
    """Return only clashes involving at least one focused residue."""

    if not residue_ids:
        return ClashReport(clashes=())

    return detect_clashes_from_context(
        prepare_clash_detection_context(
            structure,
            component_library=component_library,
            policy=policy,
        ),
        focus_residue_ids=residue_ids,
    )


def prepare_clash_detection_context(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    policy: ClashPolicy | None = None,
    basis: ClashDetectionBasis | None = None,
) -> ClashDetectionContext:
    """Prepare reusable computational state for steric-clash diagnostics."""

    if basis is not None:
        if policy is not None and policy != basis.policy:
            raise ValueError("clash detection basis policy must match policy")

        return bind_clash_detection_context(structure, basis=basis)

    return bind_clash_detection_context(
        structure,
        basis=prepare_clash_detection_basis(
            structure,
            component_library=component_library,
            policy=policy,
        ),
    )


def prepare_clash_detection_basis(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    policy: ClashPolicy | None = None,
) -> ClashDetectionBasis:
    """Prepare coordinate-independent clash facts for one structure address space."""

    normalized_policy = ClashPolicy() if policy is None else policy
    residue_context_bases = build_residue_context_bases(
        structure,
        component_library=component_library,
        include_ligands=normalized_policy.include_ligands,
    )
    atom_site_bases = build_atom_site_bases(
        residue_context_bases,
        policy=normalized_policy,
    )
    elements = frozenset(
        atom_site_basis.element for atom_site_basis in atom_site_bases
    )
    return ClashDetectionBasis(
        residue_context_bases=residue_context_bases,
        atom_site_bases=atom_site_bases,
        policy=normalized_policy,
        constitution_address_space_key=structure.constitution.address_space_key,
        van_der_waals_radius_by_element={
            element: van_der_waals_radius_angstrom(element) for element in elements
        },
    )


def bind_clash_detection_context(
    structure: ProteinStructure,
    *,
    basis: ClashDetectionBasis,
) -> ClashDetectionContext:
    """Bind one reusable clash basis to the current coordinate frame."""

    return basis.bind_context(structure)


def bind_clash_detection_frame(
    structure: ProteinStructure,
    *,
    basis: ClashDetectionBasis,
) -> ClashDetectionFrame:
    """Bind one reusable clash basis to coordinate-derived atom sites."""

    return basis.bind_frame(structure)


def prepare_projected_clash_detection_context(
    structure: ProteinStructure,
    *,
    residue_ids: tuple[ResidueId, ...],
    component_library: ComponentLibrary,
    policy: ClashPolicy | None = None,
) -> ClashDetectionContext:
    """Prepare reusable clash state for one residue-id projection."""

    normalized_policy = ClashPolicy() if policy is None else policy
    residue_contexts = build_projected_residue_contexts(
        structure,
        residue_ids=residue_ids,
        component_library=component_library,
        include_ligands=normalized_policy.include_ligands,
    )
    atom_sites = tuple(build_atom_sites(residue_contexts, policy=normalized_policy))
    return _clash_detection_context_from_atom_sites(
        atom_sites,
        policy=normalized_policy,
    )


def detect_clashes_from_context(
    context: ClashDetectionContext,
    *,
    focus_residue_ids: frozenset[ResidueId] | None = None,
) -> ClashReport:
    """Return clashes from one prepared context."""

    return context.detect_clashes(focus_residue_ids=focus_residue_ids)


def _clash_detection_context_from_atom_sites(
    atom_sites: tuple[AtomSite, ...],
    *,
    policy: ClashPolicy,
) -> ClashDetectionContext:
    """Build one prepared clash context from precomputed atom sites."""

    elements = frozenset(atom_site.element for atom_site in atom_sites)
    return ClashDetectionContext(
        atom_sites=atom_sites,
        policy=policy,
        van_der_waals_radius_by_element={
            element: van_der_waals_radius_angstrom(element) for element in elements
        },
    )


def _clash_for_atom_site_pair(
    left_site: AtomSite,
    right_site: AtomSite,
    *,
    context: ClashDetectionContext,
) -> StericClash | None:
    """Return a steric clash for one pair, if policy and geometry admit it."""

    allowed_distance = context.allowed_distance_by_element_pair[
        (left_site.element, right_site.element)
    ]
    pair_distance_squared = _atom_site_distance_squared(left_site, right_site)
    if pair_distance_squared >= allowed_distance * allowed_distance:
        return None

    pair_distance = sqrt(pair_distance_squared)
    if should_ignore_pair(left_site, right_site, policy=context.policy):
        return None

    if probable_hydrogen_bond(left_site, right_site, pair_distance):
        return None

    required_overlap = context.policy.required_overlap(
        left_site.element,
        right_site.element,
    )
    return StericClash(
        left_residue_id=left_site.residue_id,
        left_component_id=left_site.component_id,
        left_atom_name=left_site.atom_name,
        left_domain=left_site.domain,
        right_residue_id=right_site.residue_id,
        right_component_id=right_site.component_id,
        right_atom_name=right_site.atom_name,
        right_domain=right_site.domain,
        distance_angstrom=pair_distance,
        allowed_distance_angstrom=allowed_distance,
        overlap_angstrom=allowed_distance - pair_distance + required_overlap,
    )


def _allowed_distance_by_element_pair(
    radius_by_element: Mapping[str, float],
    *,
    policy: ClashPolicy,
) -> dict[tuple[str, str], float]:
    """Return cached clash distance thresholds by ordered element pair."""

    return {
        (left_element, right_element): (
            left_radius
            + right_radius
            - policy.required_overlap(left_element, right_element)
        )
        for left_element, left_radius in radius_by_element.items()
        for right_element, right_radius in radius_by_element.items()
    }


def _atom_site_distance_squared(left_site: AtomSite, right_site: AtomSite) -> float:
    """Return squared Cartesian distance between two atom sites."""

    left_position = left_site.geometry.position
    right_position = right_site.geometry.position
    dx = left_position.x - right_position.x
    dy = left_position.y - right_position.y
    dz = left_position.z - right_position.z
    return dx * dx + dy * dy + dz * dz


def has_clashes_in_context(
    context: ClashDetectionContext,
    *,
    focus_residue_ids: frozenset[ResidueId] | None = None,
) -> bool:
    """Return whether a prepared clash context contains at least one clash."""

    return context.has_clashes(focus_residue_ids=focus_residue_ids)


def has_clashes(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    policy: ClashPolicy | None = None,
) -> bool:
    """Return whether a canonical structure contains at least one steric clash."""

    return has_clashes_in_context(
        prepare_clash_detection_context(
            structure,
            component_library=component_library,
            policy=policy,
        ),
    )


def has_clashes_in_residue_projection(
    structure: ProteinStructure,
    *,
    residue_ids: tuple[ResidueId, ...],
    component_library: ComponentLibrary,
    policy: ClashPolicy | None = None,
) -> bool:
    """Return whether one residue-id projection contains any steric clash."""

    return has_clashes_in_context(
        prepare_projected_clash_detection_context(
            structure,
            residue_ids=residue_ids,
            component_library=component_library,
            policy=policy,
        ),
    )


def residue_sort_key(residue_id: ResidueId) -> tuple[str, int, str]:
    """Return a stable ordering key for residue identifiers."""

    return (
        residue_id.chain_id,
        residue_id.seq_num,
        residue_id.insertion_code or "",
    )


def candidate_atom_site_pairs(
    atom_sites: tuple[AtomSite, ...],
    *,
    focus_residue_ids: frozenset[ResidueId] | None = None,
) -> Iterator[tuple[AtomSite, AtomSite]]:
    """Return locality-pruned candidate atom pairs using a Cartesian cell grid."""

    return _iter_candidate_atom_site_pairs(
        atom_sites,
        focus_residue_ids=focus_residue_ids,
        policy=None,
    )


def _iter_candidate_atom_site_pairs(
    atom_sites: tuple[AtomSite, ...],
    *,
    focus_residue_ids: frozenset[ResidueId] | None,
    policy: ClashPolicy | None,
) -> Iterator[tuple[AtomSite, AtomSite]]:
    """Yield locality-pruned candidate pairs, optionally restricted to one focus."""

    for left_site, right_site in iter_candidate_atom_site_pairs(
        cast(tuple[ClashPairAtomSite, ...], atom_sites),
        focus_residue_ids=focus_residue_ids,
        policy=cast(ClashPairPolicy | None, policy),
    ):
        yield cast(AtomSite, left_site), cast(AtomSite, right_site)


def _iter_focused_candidate_atom_site_pairs(
    atom_sites: tuple[AtomSite, ...],
    *,
    focus_residue_ids: frozenset[ResidueId],
    policy: ClashPolicy | None,
) -> Iterator[tuple[AtomSite, AtomSite]]:
    """Yield candidate pairs touching focus atoms without streaming all pairs."""

    cell_to_indexed_atom_sites: dict[
        tuple[int, int, int],
        list[tuple[int, AtomSite]],
    ] = defaultdict(list)
    focused_indexed_atom_sites: list[tuple[int, AtomSite]] = []
    for atom_index, atom_site in enumerate(atom_sites):
        cell_to_indexed_atom_sites[atom_site.grid_cell].append((atom_index, atom_site))
        if atom_site.residue_id in focus_residue_ids:
            focused_indexed_atom_sites.append((atom_index, atom_site))

    for focus_atom_index, focus_atom_site in focused_indexed_atom_sites:
        current_cell = focus_atom_site.grid_cell
        cell_x, cell_y, cell_z = current_cell
        for offset_x, offset_y, offset_z in NEIGHBOR_CELL_OFFSETS:
            neighboring_atom_sites = cell_to_indexed_atom_sites.get(
                (
                    cell_x + offset_x,
                    cell_y + offset_y,
                    cell_z + offset_z,
                )
            )
            if neighboring_atom_sites is None:
                continue

            for other_atom_index, other_atom_site in neighboring_atom_sites:
                if other_atom_index == focus_atom_index:
                    continue
                if (
                    other_atom_site.residue_id in focus_residue_ids
                    and other_atom_index < focus_atom_index
                ):
                    continue

                if other_atom_index < focus_atom_index:
                    left_site = other_atom_site
                    right_site = focus_atom_site
                else:
                    left_site = focus_atom_site
                    right_site = other_atom_site

                if _pair_can_be_rejected_before_distance(
                    left_site,
                    right_site,
                    policy=policy,
                ):
                    continue

                yield left_site, right_site


def _pair_can_be_rejected_before_distance(
    left_site: AtomSite,
    right_site: AtomSite,
    *,
    policy: ClashPolicy | None,
) -> bool:
    """Return whether one pair can be rejected without distance/topology work."""

    return pair_can_be_rejected_before_distance(
        cast(ClashPairAtomSite, left_site),
        cast(ClashPairAtomSite, right_site),
        policy=cast(ClashPairPolicy | None, policy),
    )


def cell_id(atom_site: AtomSite) -> tuple[int, int, int]:
    """Return one atom site's spatial hash cell identifier."""

    return atom_site.grid_cell


def _cell_id_from_geometry(geometry: AtomGeometry) -> tuple[int, int, int]:
    """Return one geometry site's spatial hash cell identifier."""

    return (
        floor(geometry.position.x / CLASH_GRID_CELL_SIZE_ANGSTROM),
        floor(geometry.position.y / CLASH_GRID_CELL_SIZE_ANGSTROM),
        floor(geometry.position.z / CLASH_GRID_CELL_SIZE_ANGSTROM),
    )


def neighboring_cells(cell: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    """Return the local 3x3x3 cell neighborhood for one cell."""

    cell_x, cell_y, cell_z = cell
    return tuple(
        (cell_x + dx, cell_y + dy, cell_z + dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    )


def build_residue_contexts(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    include_ligands: bool,
) -> tuple[ResidueContext, ...]:
    """Return diagnostic residue contexts for the structure."""

    return bind_residue_contexts(
        structure,
        build_residue_context_bases(
            structure,
            component_library=component_library,
            include_ligands=include_ligands,
        ),
    )


def build_residue_context_bases(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    include_ligands: bool,
) -> tuple[_ResidueContextBasis, ...]:
    """Return coordinate-independent diagnostic residue context bases."""

    bases: list[_ResidueContextBasis] = []
    for chain_index, chain_site in enumerate(structure.constitution.chains):
        for residue_index, residue_site in enumerate(chain_site.residues):
            residue_slot_index = structure.constitution.residue_index(
                residue_site.residue_id
            )
            bases.append(
                _ResidueContextBasis(
                    residue_site=residue_site,
                    template=component_library.get(residue_site.component_id),
                    domain=ContactDomain.POLYMER,
                    chain_index=chain_index,
                    residue_index=residue_index,
                    residue_slot_index=residue_slot_index.value,
                )
            )

    if include_ligands:
        for ligand_site in structure.constitution.ligands:
            residue_slot_index = structure.constitution.residue_index(
                ligand_site.residue_id
            )
            bases.append(
                _ResidueContextBasis(
                    residue_site=ligand_site,
                    template=component_library.get(ligand_site.component_id),
                    domain=ContactDomain.RETAINED_NON_POLYMER,
                    chain_index=None,
                    residue_index=None,
                    residue_slot_index=residue_slot_index.value,
                )
            )

    return tuple(bases)


def build_atom_site_bases(
    residue_context_bases: tuple[_ResidueContextBasis, ...],
    *,
    policy: ClashPolicy,
) -> tuple[_AtomSiteBasis, ...]:
    """Return coordinate-independent atom-site metadata for one clash basis."""

    atom_site_bases: list[_AtomSiteBasis] = []
    for residue_context_index, residue_context_basis in enumerate(
        residue_context_bases
    ):
        for atom_site in residue_context_basis.residue_site.atom_sites:
            if not policy.include_hydrogens and atom_site.element == "H":
                continue

            atom_site_bases.append(
                _AtomSiteBasis(
                    residue_context_index=residue_context_index,
                    atom_name=atom_site.name,
                    element=atom_site.element,
                )
            )

    return tuple(atom_site_bases)


def bind_residue_contexts(
    structure: ProteinStructure,
    residue_context_bases: tuple[_ResidueContextBasis, ...],
) -> tuple[ResidueContext, ...]:
    """Bind coordinate-independent residue context bases to current geometry."""

    contexts: list[ResidueContext] = []
    for basis in residue_context_bases:
        residue_geometry = structure.geometry.residue_geometry(
            constitution=structure.constitution,
            residue_index=ResidueIndex(basis.residue_slot_index),
        )
        contexts.append(
            ResidueContext(
                residue_site=basis.residue_site,
                residue_geometry=residue_geometry,
                template=basis.template,
                domain=basis.domain,
                chain_index=basis.chain_index,
                residue_index=basis.residue_index,
            )
        )

    return tuple(contexts)


def build_projected_residue_contexts(
    structure: ProteinStructure,
    *,
    residue_ids: tuple[ResidueId, ...],
    component_library: ComponentLibrary,
    include_ligands: bool,
) -> tuple[ResidueContext, ...]:
    """Return diagnostic residue contexts for one residue-id projection."""

    selected_residue_ids = frozenset(residue_ids)
    contexts: list[ResidueContext] = []
    for chain_index, chain_site in enumerate(structure.constitution.chains):
        for residue_index, residue_site in enumerate(chain_site.residues):
            if residue_site.residue_id not in selected_residue_ids:
                continue

            residue_geometry = structure.geometry.residue_geometry(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(
                    residue_site.residue_id
                ),
            )
            contexts.append(
                ResidueContext(
                    residue_site=residue_site,
                    residue_geometry=residue_geometry,
                    template=component_library.get(residue_site.component_id),
                    domain=ContactDomain.POLYMER,
                    chain_index=chain_index,
                    residue_index=residue_index,
                )
            )

    if include_ligands:
        for ligand_site in structure.constitution.ligands:
            if ligand_site.residue_id not in selected_residue_ids:
                continue

            ligand_geometry = structure.geometry.residue_geometry(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(
                    ligand_site.residue_id
                ),
            )
            contexts.append(
                ResidueContext(
                    residue_site=ligand_site,
                    residue_geometry=ligand_geometry,
                    template=component_library.get(ligand_site.component_id),
                    domain=ContactDomain.RETAINED_NON_POLYMER,
                    chain_index=None,
                    residue_index=None,
                )
            )

    return tuple(contexts)


def infer_hydrogen_anchors(
    residue_site: ResidueSite,
    residue_geometry: "ResidueGeometry",
) -> Mapping[str, str]:
    """Infer a bonded heavy-atom anchor for each hydrogen in one residue."""

    heavy_atom_geometries = tuple(
        (
            atom_site,
            residue_geometry.atom_geometry(atom_site.name),
        )
        for atom_site in residue_site.atom_sites
        if atom_site.element != "H"
    )
    hydrogen_anchor_by_name: dict[str, str] = {}
    for hydrogen_atom_site in residue_site.atom_sites:
        if hydrogen_atom_site.element != "H":
            continue

        hydrogen_geometry = residue_geometry.atom_geometry(hydrogen_atom_site.name)
        closest_heavy_atom_name: str | None = None
        closest_heavy_atom_distance = float("inf")
        for heavy_atom_site, heavy_atom_geometry in heavy_atom_geometries:
            distance = hydrogen_geometry.distance_to(heavy_atom_geometry)
            if distance < closest_heavy_atom_distance:
                closest_heavy_atom_distance = distance
                closest_heavy_atom_name = heavy_atom_site.name

        if closest_heavy_atom_name is None:
            continue

        if closest_heavy_atom_distance <= HYDROGEN_ANCHOR_DISTANCE_CUTOFF_ANGSTROM:
            hydrogen_anchor_by_name[hydrogen_atom_site.name] = closest_heavy_atom_name

    return MappingProxyType(hydrogen_anchor_by_name)


def build_atom_sites(
    residue_contexts: tuple[ResidueContext, ...],
    *,
    policy: ClashPolicy,
) -> tuple[AtomSite, ...]:
    """Return atom sites considered by the clash detector."""

    atom_sites: list[AtomSite] = []
    for context in residue_contexts:
        residue_site = context.residue_site
        residue_geometry = context.residue_geometry
        for atom_site in residue_site.atom_sites:
            if not policy.include_hydrogens and atom_site.element == "H":
                continue

            atom_sites.append(
                AtomSite(
                    atom_name=atom_site.name,
                    element=atom_site.element,
                    geometry=residue_geometry.atom_geometry(atom_site.name),
                    context=context,
                )
            )

    return tuple(atom_sites)


def should_consider_pair(
    left_site: AtomSite,
    right_site: AtomSite,
    *,
    policy: ClashPolicy,
) -> bool:
    """Return whether one atom pair should enter clash evaluation."""

    if not policy.include_ligands and (
        left_site.domain.excluded_when_ligands_are_disabled()
        or right_site.domain.excluded_when_ligands_are_disabled()
    ):
        return False

    if not policy.include_hydrogen_hydrogen and (
        left_site.is_hydrogen() and right_site.is_hydrogen()
    ):
        return False

    return True


def should_ignore_pair(
    left_site: AtomSite,
    right_site: AtomSite,
    *,
    policy: ClashPolicy,
) -> bool:
    """Return whether one atom pair should be ignored as bonded or near-bonded."""

    return clash_topology_rules.should_ignore_pair(
        cast(clash_topology_rules.ClashTopologyAtomSite, left_site),
        cast(clash_topology_rules.ClashTopologyAtomSite, right_site),
        policy=cast(clash_topology_rules.ClashTopologyPolicy, policy),
    )


def same_residue_bond_hops(
    left_site: AtomSite,
    right_site: AtomSite,
) -> int | None:
    """Return same-residue bond hops for two atom sites if the template knows them."""

    return clash_topology_rules.same_residue_bond_hops(
        cast(clash_topology_rules.ClashTopologyAtomSite, left_site),
        cast(clash_topology_rules.ClashTopologyAtomSite, right_site),
    )


def adjacent_polymer_bond_hops(
    left_site: AtomSite,
    right_site: AtomSite,
) -> int | None:
    """Return bond hops for atom sites across one peptide bond if available."""

    return clash_topology_rules.adjacent_polymer_bond_hops(
        cast(clash_topology_rules.ClashTopologyAtomSite, left_site),
        cast(clash_topology_rules.ClashTopologyAtomSite, right_site),
    )


def atom_to_named_backbone_hops(
    site: AtomSite,
    *,
    target_atom_name: str,
) -> int | None:
    """Return bond hops from one atom site to a named backbone heavy atom."""

    return clash_topology_rules.atom_to_named_backbone_hops(
        cast(clash_topology_rules.ClashTopologyAtomSite, site),
        target_atom_name=target_atom_name,
    )


def atom_to_atom_bond_hops_within_residue(
    atom_name_1: str,
    atom_name_2: str,
    *,
    context: ResidueContext,
    template: ResidueTemplate,
) -> int | None:
    """Return intra-residue bond hops using heavy topology plus inferred H anchors."""

    return clash_topology_rules.atom_to_atom_bond_hops_within_residue(
        atom_name_1,
        atom_name_2,
        context=cast(clash_topology_rules.ClashTopologyResidueContext, context),
        template=template,
    )


def atom_is_hydrogen(residue_site: ResidueSite, atom_name: str) -> bool:
    """Return whether a named atom within a residue is a hydrogen."""

    return clash_topology_rules.atom_is_hydrogen(residue_site, atom_name)


def direct_disulfide_bond(left_site: AtomSite, right_site: AtomSite) -> bool:
    """Return whether one atom pair looks like a bonded disulfide sulfur pair."""

    return clash_topology_rules.direct_disulfide_bond(
        cast(clash_topology_rules.ClashTopologyAtomSite, left_site),
        cast(clash_topology_rules.ClashTopologyAtomSite, right_site),
    )


def probable_hydrogen_bond(
    left_site: AtomSite,
    right_site: AtomSite,
    pair_distance: float,
) -> bool:
    """Return whether one H-involving pair is plausibly a hydrogen bond."""

    return clash_topology_rules.probable_hydrogen_bond(
        cast(clash_topology_rules.ClashTopologyAtomSite, left_site),
        cast(clash_topology_rules.ClashTopologyAtomSite, right_site),
        pair_distance,
    )
