"""FASPR packing restricted to usable regions around unresolved backbone sites.

Stock FASPR reads every submitted backbone coordinate, including frozen ones.
When the run carries unresolved backbone atoms, only usable residues are
submitted. Targets that could depend on an unsubmitted residue are withheld,
and the rest are split into regions FASPR cannot couple. Each region runs on
its own, and accepted side-chain writes are checked against the uncut carrier
before they merge.

The bounds rely on the vendored FASPR energy model (``vendor/faspr/src``). In
``SelfEnergy::AssignConMap``, residues interact only when their CB atoms (CA
for Gly and Ala) are closer than the sum of the per-residue radii plus
``RESI_DIST_CUT``. Every inter-residue term iterates that neighbor map
(``SelfEnergy.cpp`` and ``PairEnergy.cpp``). Backbone torsions come only from
the previous residue's C and the next residue's N (``RotamerBuilder::PhiPsi``).
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

import numpy as np
from typing_extensions import Self

from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics.clashes import (
    ClashPolicy,
    StericClash,
    detect_clashes_involving_residues,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.diagnostics.water_contacts import stationary_water_refusal
from protrepair.errors import PackingError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.backbone_availability import (
    active_backbone_availability,
    unresolved_atom_names,
    unresolved_atom_refs,
)
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry, StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.peptide import has_verified_backbone_link
from protrepair.structure.topology import StructureTopology, TopologyBond
from protrepair.transformer.packing.domain import PackingPlan, PackingResult
from protrepair.transformer.packing.faspr.backend import (
    FASPR_ALPHABET,
    FasprPackingBackend,
    PackingBackendError,
    validate_faspr_residue_site,
)
from protrepair.transformer.packing.faspr.selection import (
    FIXED_BACKBONE_ATOM_NAMES,
    FasprPackingSelection,
)
from protrepair.transformer.packing.spec import PackingScope, PackingSpec

# ``res_rad`` in FASPR ``SelfEnergy::AssignConMap``.
_FASPR_RESIDUE_RADIUS: Mapping[str, float] = MappingProxyType(
    {
        "G": 2.4,
        **dict.fromkeys("ACDILNPSTV", 3.2),
        **dict.fromkeys("QEH", 3.7),
        **dict.fromkeys("MF", 4.3),
        "K": 5.0,
        "W": 5.3,
        "Y": 5.7,
        "R": 6.2,
    }
)
# ``RESI_DIST_CUT`` in FASPR ``SelfEnergy.h``.
_FASPR_NEIGHBOR_SLACK = 4.25
# ``DSTAR_MAX_CUT`` times twice the largest atom radius (``RADIUS_C3``) in
# ``SelfEnergy.h``: no FASPR vdW term reaches farther.
_FASPR_VDW_RANGE = 1.90 * 2 * 2.30
# ``BONDDIST`` in FASPR ``Utility.h``: the C--N distance that joins torsions.
_FASPR_BOND_DISTANCE = 2.1
# FASPR measures Gly and Ala from CA, and gives them no rotamer choice.
_FASPR_CA_REFERENCE_TOKENS = frozenset({"G", "A"})
# Upper bound on CA-CB when FASPR places CB itself.
_CB_OFFSET_BOUND = 1.6
# Upper bound on one heavy-atom covalent bond, used for template hop reach.
_BOND_LENGTH_BOUND = 2.0
_POSITION_TOLERANCE = 1e-3
_LINKED_ISSUE_KINDS = frozenset(
    {
        ValidationIssueKind.PACKING_PRESERVED_LINKED_SIDECHAIN,
        ValidationIssueKind.PACKING_REBUILT_LINKED_SIDECHAIN,
    }
)


def unresolved_backbone_residue_ids(
    structure: ProteinStructure,
) -> frozenset[ResidueId]:
    """Return polymer residues with unresolved backbone atoms in the active run.

    Parameters
    ----------
    structure : ProteinStructure
        Carrier about to be packed.

    Returns
    -------
    frozenset[ResidueId]
        Empty when no backbone availability is bound to the run.
    """

    binding = active_backbone_availability()
    if binding is None:
        return frozenset()

    polymer_ids = frozenset(
        site.residue_id for site in structure.constitution.iter_residues()
    )
    return frozenset(
        residue_id
        for residue_id in binding.placements
        if residue_id in polymer_ids and binding.unresolved(structure, residue_id)
    )


def pack_usable_backbone_regions(
    plan: PackingPlan,
    *,
    backend: FasprPackingBackend,
    unresolved_residue_ids: frozenset[ResidueId],
) -> PackingResult:
    """Pack the usable part of a carrier without reading unresolved atoms.

    Parameters
    ----------
    plan : PackingPlan
        Public request over the uncut carrier.
    backend : FasprPackingBackend
        Backend that runs each region.
    unresolved_residue_ids : frozenset[ResidueId]
        Polymer residues whose backbone is unresolved in this run.

    Returns
    -------
    PackingResult
        Carrier with accepted side-chain writes. Withheld targets and refused
        regions are reported as ``PACKING_REGION_REFUSED`` issues; an explicit
        mutation that cannot be applied in full is not applied at all.

    Raises
    ------
    PackingError
        The request is malformed in the same way the ordinary call rejects.
    """

    request = _ValidatedRequest.from_plan(plan, backend)
    withheld, proposals = _plan_regions(plan, request, unresolved_residue_ids)
    issues = [
        *_preserved_linked_issues(request.selection),
        *(
            _region_issue(residue_id, f"packing target withheld: {reason}")
            for residue_id, reason in withheld.items()
        ),
    ]

    if request.mutated and withheld:
        issues.append(_coupled_mutation_issue())
        return _unchanged_result(plan.structure, issues)

    if not proposals:
        return _unchanged_result(plan.structure, issues)

    try:
        backend.runtime_assets()
    except (OSError, PackingBackendError) as error:
        issues.append(
            ValidationIssue(
                kind=ValidationIssueKind.PACKING_REGION_REFUSED,
                severity=IssueSeverity.WARNING,
                message=f"FASPR runtime assets are unavailable: {error}",
            )
        )
        return _unchanged_result(plan.structure, issues)

    outcomes: list[_RegionOutcome] = []
    for proposal in proposals:
        try:
            outcomes.append(
                _RegionOutcome.execute(plan.structure, proposal, request, backend)
            )
        except (PackingBackendError, PackingError) as error:
            issues.extend(proposal.refusal_issues(str(error)))

    outcomes, refusals = _verify_against_carrier(plan.structure, outcomes)
    for proposal, reason in refusals:
        issues.extend(proposal.refusal_issues(reason))

    if request.mutated and len(outcomes) < len(proposals):
        issues.append(_coupled_mutation_issue())
        return _unchanged_result(plan.structure, issues)

    for outcome in outcomes:
        issues.extend(
            issue for issue in outcome.issues if issue.kind not in _LINKED_ISSUE_KINDS
        )
        issues.extend(
            _rebuilt_linked_issue(residue_id)
            for residue_id in outcome.changed_residue_ids
            if residue_id in request.selection.incomplete_linked_residue_ids
        )

    changed = frozenset(
        residue_id for outcome in outcomes for residue_id in outcome.changed_residue_ids
    )
    return PackingResult(
        packed_structure=_merged_structure(plan.structure, outcomes),
        changed_residue_ids=tuple(
            residue_id
            for residue_id in plan.polymer_residue_ids
            if residue_id in changed
        ),
        issues=tuple(issues),
        backend_name="faspr",
    )


@dataclass(frozen=True, slots=True)
class _ValidatedRequest:
    """Public request checked as ``FasprPackingBackend.pack`` checks it.

    Caller-supplied tokens are rejected exactly as in the ordinary call. A
    residue outside the FASPR alphabet is not a request error here: it has no
    token, is never submitted, and only targets that depend on it are withheld.
    """

    tokens: Mapping[ResidueId, str]
    mutated: frozenset[ResidueId]
    selection: FasprPackingSelection

    @classmethod
    def from_plan(cls, plan: PackingPlan, backend: FasprPackingBackend) -> Self:
        plan.assert_supported_by(backend.capabilities())
        requested = _requested_tokens(plan)
        sites = plan.polymer_residue_sites()
        mutated = frozenset(
            site.residue_id
            for site in sites
            if site.residue_id in requested
            and site.component_id
            != FASPR_ALPHABET.components_for_sequence(requested[site.residue_id])[0]
        )
        tokens = {
            site.residue_id: requested.get(
                site.residue_id, FASPR_ALPHABET.require_token(site.component_id)
            )
            for site in sites
            if FASPR_ALPHABET.supports_component(site.component_id)
        }

        selection = FasprPackingSelection.from_plan(plan)
        linked_mutations = mutated & frozenset(selection.linked_residue_ids)
        if linked_mutations:
            raise PackingError(
                "FASPR mutation would invalidate linked chemistry at "
                + ", ".join(rid.display_token() for rid in sorted(linked_mutations))
            )

        frozen_mutations = mutated & frozenset(plan.fixed_residue_ids())
        if frozen_mutations:
            raise PackingError(
                "packing target_sequence would mutate frozen residues: "
                + ", ".join(rid.display_token() for rid in sorted(frozen_mutations))
            )

        return cls(
            tokens=MappingProxyType(tokens),
            mutated=mutated,
            selection=selection,
        )

    def interaction_radius(self, residue_id: ResidueId) -> float:
        """Return FASPR's neighbor radius for the requested residue type."""

        return _FASPR_RESIDUE_RADIUS[self.tokens[residue_id]]

    def has_rotamer_choice(self, residue_id: ResidueId) -> bool:
        """Return whether FASPR evaluates contacts for this target."""

        return self.tokens[residue_id] not in _FASPR_CA_REFERENCE_TOKENS


@dataclass(frozen=True, slots=True)
class _RegionProposal:
    """Targets FASPR may couple, plus the usable residues they can see."""

    target_ids: tuple[ResidueId, ...]
    member_ids: frozenset[ResidueId]

    def refusal_issues(self, reason: str) -> Iterable[ValidationIssue]:
        """Report every target of this region as refused for ``reason``."""

        return (
            _region_issue(residue_id, f"packing region refused: {reason}")
            for residue_id in self.target_ids
        )


def _requested_tokens(plan: PackingPlan) -> dict[ResidueId, str]:
    """Return caller-supplied tokens, rejected as ``build_faspr_sequence_override``."""

    sequence = plan.spec.target_sequence
    if sequence is None:
        return {}

    tokens = tuple(sequence)
    residue_ids = plan.spec.mutable_residue_ids
    if residue_ids is None:
        if len(tokens) != len(plan.polymer_residue_ids):
            raise PackingError(
                "full-structure target_sequence must match polymer residue count"
            )
        residue_ids = plan.polymer_residue_ids
    FASPR_ALPHABET.components_for_sequence(tokens)
    return {rid: token.upper() for rid, token in zip(residue_ids, tokens, strict=True)}


@dataclass(frozen=True, slots=True)
class _UnsubmittedResidue:
    """Bound on how close an unsubmitted residue can come to a FASPR target.

    Each entry of ``measured`` bounds one atom: it pairs usable atom positions
    with the farthest that atom can be from them. For a FASPR residue type the
    only measured atom is the one ``AssignConMap`` uses (CB, or CA for Gly and
    Ala), and ``contact_reach`` is its radius plus ``RESI_DIST_CUT``. A residue
    outside the alphabet is measured at every heavy atom against FASPR's vdW
    range. Placeholder positions of unresolved atoms are never read.
    """

    residue_id: ResidueId
    contact_reach: float
    measured: tuple[tuple[tuple[np.ndarray, float], ...], ...]

    @classmethod
    def observe(
        cls,
        structure: ProteinStructure,
        site: ResidueSite,
        token: str | None,
    ) -> Self:
        unresolved = unresolved_atom_names(structure, site.residue_id)
        geometry = structure.residue_geometry(
            structure.constitution.residue_index(site.residue_id)
        )
        usable = {
            atom.name: _point(geometry, atom.name)
            for atom in site.atom_sites
            if not atom.is_hydrogen() and atom.name not in unresolved
        }
        template = build_default_component_library().get(site.component_id)

        def anchors(name: str) -> tuple[tuple[np.ndarray, float], ...]:
            if name in usable:
                return ((usable[name], 0.0),)
            if template is None:
                return ()
            return tuple(
                (point, hops * _BOND_LENGTH_BOUND)
                for other, point in usable.items()
                if (hops := template.bond_hop_distance(name, other)) is not None
            )

        if token is None:
            return cls(
                site.residue_id,
                _FASPR_VDW_RANGE,
                tuple(
                    anchors(atom.name)
                    for atom in site.atom_sites
                    if not atom.is_hydrogen()
                ),
            )

        reference = "CA" if token in _FASPR_CA_REFERENCE_TOKENS else "CB"
        return cls(
            site.residue_id,
            _FASPR_RESIDUE_RADIUS[token] + _FASPR_NEIGHBOR_SLACK,
            (anchors(reference),),
        )

    def min_distance_to(self, point: np.ndarray) -> float:
        """Return a lower bound on any measured atom's distance to ``point``."""

        return min(
            (
                max(
                    (
                        float(np.linalg.norm(point - anchor)) - reach
                        for anchor, reach in atom_anchors
                    ),
                    default=-np.inf,
                )
                for atom_anchors in self.measured
            ),
            default=np.inf,
        )


def _plan_regions(
    plan: PackingPlan,
    request: _ValidatedRequest,
    unresolved_residue_ids: frozenset[ResidueId],
) -> tuple[dict[ResidueId, str], tuple[_RegionProposal, ...]]:
    """Withhold dependent targets, then group the rest into regions."""

    structure = plan.structure
    sites = plan.polymer_residue_sites()
    adjacent = _chain_neighbors(plan)
    omitted = {
        residue_id: "unresolved backbone atoms" for residue_id in unresolved_residue_ids
    }
    for site in sites:
        if site.residue_id in omitted:
            continue
        try:
            validate_faspr_residue_site(site)
        except PackingBackendError as error:
            omitted[site.residue_id] = str(error)

    withheld: dict[ResidueId, str] = {}
    active: list[ResidueId] = []
    for residue_id in request.selection.movable_residue_ids():
        if residue_id in omitted:
            withheld[residue_id] = omitted[residue_id]
        else:
            active.append(residue_id)

    unsubmitted: dict[ResidueId, _UnsubmittedResidue] = {}
    while True:
        active_set = frozenset(active)
        for site in sites:
            residue_id = site.residue_id
            if residue_id in omitted or residue_id in active_set:
                continue
            missing = _missing_heavy_atoms(site)
            if missing:
                omitted[residue_id] = (
                    "incomplete side chain cannot be held fixed "
                    f"(missing heavy atoms: {', '.join(missing)})"
                )

        for residue_id in omitted:
            if residue_id not in unsubmitted:
                unsubmitted[residue_id] = _UnsubmittedResidue.observe(
                    structure,
                    plan.residue_site(residue_id),
                    request.tokens.get(residue_id),
                )

        newly_withheld = {
            residue_id: reason
            for residue_id in active
            if (
                reason := _dependency(
                    structure, request, residue_id, adjacent, omitted, unsubmitted
                )
            )
            is not None
        }
        if not newly_withheld:
            submitted = tuple(
                site.residue_id for site in sites if site.residue_id not in omitted
            )
            proposals = _group_regions(structure, request, active, submitted, adjacent)
            newly_withheld = _transport_join_vetoes(plan, request, proposals)
            if not newly_withheld:
                return withheld, proposals

        withheld.update(newly_withheld)
        active = [rid for rid in active if rid not in newly_withheld]


def _dependency(
    structure: ProteinStructure,
    request: _ValidatedRequest,
    residue_id: ResidueId,
    adjacent: Mapping[ResidueId, tuple[ResidueId, ...]],
    omitted: Mapping[ResidueId, str],
    unsubmitted: Mapping[ResidueId, _UnsubmittedResidue],
) -> str | None:
    """Return why a target needs information from an unsubmitted residue."""

    for neighbor in adjacent[residue_id]:
        if neighbor in omitted:
            return (
                f"chain neighbor {neighbor.display_token()} is not submitted "
                f"({omitted[neighbor]}); FASPR would use a default backbone torsion"
            )

    if not request.has_rotamer_choice(residue_id):
        return None

    center = _ca_point(structure, residue_id)
    margin = _cb_margin(structure, residue_id)
    radius = request.interaction_radius(residue_id)
    for other in unsubmitted.values():
        if other.min_distance_to(center) - margin < radius + other.contact_reach:
            return (
                f"possible FASPR contact with unsubmitted residue "
                f"{other.residue_id.display_token()} ({omitted[other.residue_id]})"
            )
    return None


def _transport_join_vetoes(
    plan: PackingPlan,
    request: _ValidatedRequest,
    proposals: Sequence[_RegionProposal],
) -> dict[ResidueId, str]:
    """Return targets whose torsion FASPR would take from an unproven join.

    FASPR reads residues as one array and takes a residue's phi and psi from
    the previous and next entries whenever their C--N distance is within
    ``BONDDIST`` (``RotamerBuilder::PhiPsi``), across chains and TER records.
    A region submits members in carrier order, so it can place residues next
    to each other without a verified source backbone link. Such a join is
    harmless only when submitted coordinates keep it beyond ``BONDDIST``.
    """

    structure = plan.structure
    vetoed: dict[ResidueId, str] = {}
    for proposal in proposals:
        targets = frozenset(
            rid for rid in proposal.target_ids if request.has_rotamer_choice(rid)
        )
        order = tuple(
            rid for rid in plan.polymer_residue_ids if rid in proposal.member_ids
        )
        for left, right in pairwise(order):
            if not {left, right} & targets:
                continue
            left_site, right_site = plan.residue_site(left), plan.residue_site(right)
            if has_verified_backbone_link(structure, left_site, right_site):
                continue
            gap = float(
                np.linalg.norm(
                    _atom_point(structure, left, "C")
                    - _atom_point(structure, right, "N")
                )
            )
            if gap > _FASPR_BOND_DISTANCE + _POSITION_TOLERANCE:
                continue
            reason = (
                f"FASPR input would join {left.display_token()} and "
                f"{right.display_token()} (C-N {gap:.2f} A), which are not "
                "linked by a verified source backbone bond"
            )
            vetoed.update(dict.fromkeys({left, right} & targets, reason))
    return vetoed


def _group_regions(
    structure: ProteinStructure,
    request: _ValidatedRequest,
    targets: Sequence[ResidueId],
    submitted: Sequence[ResidueId],
    adjacent: Mapping[ResidueId, tuple[ResidueId, ...]],
) -> tuple[_RegionProposal, ...]:
    """Split targets into components no FASPR neighbor relation crosses."""

    if not targets:
        return ()

    centers = np.array([_ca_point(structure, rid) for rid in submitted])
    reach = np.array(
        [
            request.interaction_radius(rid) + _cb_margin(structure, rid)
            for rid in submitted
        ]
    )
    position = {residue_id: index for index, residue_id in enumerate(submitted)}
    parent = {residue_id: residue_id for residue_id in targets}
    target_set = frozenset(targets)
    visible: dict[ResidueId, frozenset[ResidueId]] = {}

    def root(residue_id: ResidueId) -> ResidueId:
        while parent[residue_id] != residue_id:
            parent[residue_id] = parent[parent[residue_id]]
            residue_id = parent[residue_id]
        return residue_id

    for target in targets:
        index = position[target]
        distances = np.linalg.norm(centers - centers[index], axis=1)
        mask = distances < reach[index] + reach + _FASPR_NEIGHBOR_SLACK
        seen = frozenset(submitted[i] for i in np.flatnonzero(mask))
        visible[target] = seen | frozenset(adjacent[target])
        for other in seen & target_set:
            parent[root(other)] = root(target)

    grouped: dict[ResidueId, list[ResidueId]] = {}
    for target in targets:
        grouped.setdefault(root(target), []).append(target)

    order = {residue_id: index for index, residue_id in enumerate(submitted)}
    proposals = [
        _RegionProposal(
            target_ids=tuple(sorted(members, key=order.__getitem__)),
            member_ids=frozenset(
                residue_id for target in members for residue_id in visible[target]
            ),
        )
        for members in grouped.values()
    ]
    return tuple(sorted(proposals, key=lambda proposal: order[proposal.target_ids[0]]))


@dataclass(frozen=True, slots=True)
class _RegionOutcome:
    """One region's FASPR result, restricted to its own targets."""

    proposal: _RegionProposal
    packed: ProteinStructure
    changed_residue_ids: tuple[ResidueId, ...]
    issues: tuple[ValidationIssue, ...]

    @classmethod
    def execute(
        cls,
        carrier: ProteinStructure,
        proposal: _RegionProposal,
        request: _ValidatedRequest,
        backend: FasprPackingBackend,
    ) -> Self:
        spec = PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            target_sequence=(
                "".join(request.tokens[rid] for rid in proposal.target_ids)
                if request.mutated
                else None
            ),
            mutable_residue_ids=proposal.target_ids,
        )
        region = _region_structure(carrier, proposal.member_ids)
        result = backend.pack(PackingPlan.from_inputs(region, spec))

        refusals = [
            issue.message
            for issue in result.issues
            if issue.kind is ValidationIssueKind.RETAINED_WATER_REPAIR_REFUSED
        ]
        if refusals:
            raise PackingBackendError("; ".join(refusals))

        stray = frozenset(result.changed_residue_ids) - frozenset(proposal.target_ids)
        if stray:
            raise PackingBackendError(
                "FASPR changed residues outside the region targets: "
                + ", ".join(rid.display_token() for rid in sorted(stray))
            )

        for residue_id in result.changed_residue_ids:
            _check_side_chain_only(carrier, result.packed_structure, residue_id)

        return cls(
            proposal=proposal,
            packed=result.packed_structure,
            changed_residue_ids=result.changed_residue_ids,
            issues=result.issues,
        )

    def residue_facets(
        self,
    ) -> Iterable[
        tuple[ResidueSite, ResidueGeometry, tuple[tuple[str, int | None], ...]]
    ]:
        """Yield the packed facets of this region's changed targets."""

        for residue_id in self.changed_residue_ids:
            index = self.packed.constitution.residue_index(residue_id)
            yield (
                self.packed.constitution.residue_site_at(index),
                self.packed.residue_geometry(index),
                self.packed.residue_formal_charge_by_atom_name(index),
            )


def _verify_against_carrier(
    carrier: ProteinStructure,
    outcomes: Sequence[_RegionOutcome],
) -> tuple[list[_RegionOutcome], list[tuple[_RegionProposal, str]]]:
    """Refuse regions whose writes meet atoms that FASPR did not see."""

    written = [outcome for outcome in outcomes if outcome.changed_residue_ids]
    if not written:
        return list(outcomes), []

    library = build_default_component_library()
    reasons: dict[_RegionProposal, str] = {}
    merged = _merged_structure(carrier, written)
    if stationary_water_refusal(carrier, merged) is not None:
        for outcome in written:
            refusal = stationary_water_refusal(
                carrier, _merged_structure(carrier, (outcome,))
            )
            if refusal is not None:
                reasons[outcome.proposal] = refusal.message

    owner = {
        residue_id: outcome.proposal
        for outcome in written
        for residue_id in outcome.changed_residue_ids
    }
    hidden = _unresolved_atom_refs(carrier)
    focus = frozenset(owner)
    baseline: dict[frozenset[AtomRef], float] = {}
    for clash in _clashes(carrier.without_atom_refs(hidden), focus, library):
        key = _clash_key(clash)
        baseline[key] = max(clash.overlap_angstrom, baseline.get(key, 0.0))
    for clash in _clashes(merged.without_atom_refs(hidden), focus, library):
        depth = baseline.get(_clash_key(clash))
        if depth is not None and clash.overlap_angstrom <= depth + _POSITION_TOLERANCE:
            continue
        for mine, other in (
            (clash.left_residue_id, clash.right_residue_id),
            (clash.right_residue_id, clash.left_residue_id),
        ):
            proposal = owner.get(mine)
            if proposal is not None and other not in proposal.member_ids:
                reasons.setdefault(
                    proposal,
                    f"packed {mine.display_token()} clashes with "
                    f"{other.display_token()}, which FASPR did not see "
                    f"({clash.left_atom_name}-{clash.right_atom_name} overlap "
                    f"{clash.overlap_angstrom:.2f} A)",
                )

    accepted = [outcome for outcome in outcomes if outcome.proposal not in reasons]
    return accepted, [
        (outcome.proposal, reasons[outcome.proposal])
        for outcome in written
        if outcome.proposal in reasons
    ]


def _merged_structure(
    carrier: ProteinStructure,
    outcomes: Iterable[_RegionOutcome],
) -> ProteinStructure:
    """Apply side-chain writes to the uncut carrier.

    Bonds inside a rewritten residue come from the backend result for that
    residue; every other bond, including peptide and link bonds, stays the
    carrier's.
    """

    outcomes = tuple(outcomes)
    facets = [facet for outcome in outcomes for facet in outcome.residue_facets()]
    if not facets:
        return carrier

    updated = carrier.with_updated_residue_facets_batch(facets)
    constitution = updated.constitution
    rewritten = frozenset(
        residue_id for outcome in outcomes for residue_id in outcome.changed_residue_ids
    )
    bonds = [
        bond
        for bond in updated.topology.bonds
        if _bond_residue(updated, bond) not in rewritten
    ]
    for outcome in outcomes:
        owned = frozenset(outcome.changed_residue_ids)
        bonds.extend(
            bond
            for bond in outcome.packed.topology.bonds_for_constitution(
                source_constitution=outcome.packed.constitution,
                target_constitution=constitution,
            )
            if _bond_residue(updated, bond) in owned
        )

    return ProteinStructure.from_payload(
        constitution=constitution,
        geometry=updated.geometry,
        topology=StructureTopology(
            constitution=constitution,
            atom_topologies=updated.topology.atom_topologies,
            bonds=tuple(bonds),
        ),
        polymer_blueprint=updated.polymer_blueprint,
        provenance=updated.provenance,
    )


def _region_structure(
    structure: ProteinStructure,
    member_ids: frozenset[ResidueId],
) -> ProteinStructure:
    """Return the carrier with only member polymer residues, ligands and waters."""

    constitution = structure.constitution
    chains = tuple(
        chain
        for chain in (
            chain.with_residues(
                site for site in chain.residues if site.residue_id in member_ids
            )
            for chain in constitution.chains
        )
        if chain.residues
    )
    region_constitution = constitution.with_chains(chains)
    atom_indices = tuple(
        atom_index
        for site in region_constitution.residue_slots
        for atom_index in constitution.atom_indices_for_residue_index(
            constitution.residue_index(site.residue_id)
        )
    )
    blueprint = structure.polymer_blueprint
    return ProteinStructure.from_payload(
        constitution=region_constitution,
        geometry=StructureGeometry(
            constitution=region_constitution,
            atom_geometries=(
                structure.geometry.atom_geometry(index) for index in atom_indices
            ),
        ),
        topology=StructureTopology(
            constitution=region_constitution,
            atom_topologies=tuple(
                structure.topology.atom_topology(index) for index in atom_indices
            ),
            bonds=structure.topology.bonds_for_constitution(
                source_constitution=constitution,
                target_constitution=region_constitution,
            ),
        ),
        polymer_blueprint=(
            None
            if blueprint is None
            else blueprint.select_chains(tuple(chain.chain_id for chain in chains))
        ),
        provenance=structure.provenance,
    )


def _check_side_chain_only(
    carrier: ProteinStructure,
    packed: ProteinStructure,
    residue_id: ResidueId,
) -> None:
    """Refuse a packed residue whose backbone or hydrogens differ from the carrier."""

    before_site = carrier.constitution.residue_or_ligand(residue_id)
    after_site = packed.constitution.residue_or_ligand(residue_id)
    if before_site is None or after_site is None:
        raise PackingBackendError(f"FASPR lost residue {residue_id.display_token()}")

    before = carrier.residue_geometry(carrier.constitution.residue_index(residue_id))
    after = packed.residue_geometry(packed.constitution.residue_index(residue_id))
    backbone_before = {
        atom.name
        for atom in before_site.atom_sites
        if atom.name in FIXED_BACKBONE_ATOM_NAMES
    }
    backbone_after = {
        atom.name
        for atom in after_site.atom_sites
        if atom.name in FIXED_BACKBONE_ATOM_NAMES
    }
    if backbone_before != backbone_after:
        raise PackingBackendError(
            f"FASPR changed backbone atoms of {residue_id.display_token()}"
        )

    for atom in after_site.atom_sites:
        if atom.name not in FIXED_BACKBONE_ATOM_NAMES and not atom.is_hydrogen():
            continue
        if (
            not before_site.has_atom_site(atom.name)
            or before.position(atom.name).distance_to(after.position(atom.name))
            > _POSITION_TOLERANCE
        ):
            raise PackingBackendError(
                f"FASPR wrote non-side-chain atom {atom.name} of "
                f"{residue_id.display_token()}"
            )


def _chain_neighbors(plan: PackingPlan) -> dict[ResidueId, tuple[ResidueId, ...]]:
    """Return the previous and next residue of each polymer residue in its chain."""

    neighbors: dict[ResidueId, tuple[ResidueId, ...]] = {}
    for chain in plan.polymer_chain_sites():
        ids = chain.residue_ids()
        for index, residue_id in enumerate(ids):
            neighbors[residue_id] = (
                ids[max(index - 1, 0) : index] + ids[index + 1 : index + 2]
            )
    return neighbors


def _missing_heavy_atoms(site: ResidueSite) -> tuple[str, ...]:
    template = build_standard_component_library().require(site.component_id)
    return tuple(
        name
        for name in template.expected_polymer_heavy_atom_names()
        if not site.has_atom_site(name)
    )


def _point(geometry: ResidueGeometry, atom_name: str) -> np.ndarray:
    position = geometry.position(atom_name)
    return np.array((position.x, position.y, position.z))


def _atom_point(
    structure: ProteinStructure, residue_id: ResidueId, atom_name: str
) -> np.ndarray:
    return _point(
        structure.residue_geometry(structure.constitution.residue_index(residue_id)),
        atom_name,
    )


def _ca_point(structure: ProteinStructure, residue_id: ResidueId) -> np.ndarray:
    return _atom_point(structure, residue_id, "CA")


def _cb_margin(structure: ProteinStructure, residue_id: ResidueId) -> float:
    """Return how far FASPR's CB for a submitted residue can sit from its CA."""

    site = structure.constitution.residue_or_ligand(residue_id)
    if site is None or not site.has_atom_site("CB"):
        return _CB_OFFSET_BOUND
    geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue_id)
    )
    return max(
        _CB_OFFSET_BOUND,
        float(np.linalg.norm(_point(geometry, "CB") - _point(geometry, "CA"))),
    )


def _unresolved_atom_refs(structure: ProteinStructure) -> tuple[AtomRef, ...]:
    """Return present unresolved atoms, which ``without_atom_refs`` can drop."""

    return tuple(
        ref
        for ref in sorted(unresolved_atom_refs(structure))
        if structure.constitution.resolve_atom_index(ref) is not None
    )


def _clashes(
    structure: ProteinStructure,
    focus: frozenset[ResidueId],
    library: ComponentLibrary,
) -> tuple[StericClash, ...]:
    return detect_clashes_involving_residues(
        structure,
        residue_ids=focus,
        component_library=library,
        policy=ClashPolicy(include_ligands=True),
    ).clashes


def _clash_key(clash: StericClash) -> frozenset[AtomRef]:
    return frozenset(
        {
            AtomRef(clash.left_residue_id, clash.left_atom_name),
            AtomRef(clash.right_residue_id, clash.right_atom_name),
        }
    )


def _bond_residue(structure: ProteinStructure, bond: TopologyBond) -> ResidueId | None:
    """Return the residue owning both bond endpoints, or None across residues."""

    left, right = (
        structure.constitution.atom_ref_at(index).residue_id
        for index in bond.endpoint_pair()
    )
    return left if left == right else None


def _region_issue(residue_id: ResidueId, message: str) -> ValidationIssue:
    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.PACKING_REGION_REFUSED,
        severity=IssueSeverity.WARNING,
        message=message,
        residue_id=residue_id,
    )


def _coupled_mutation_issue() -> ValidationIssue:
    return ValidationIssue(
        kind=ValidationIssueKind.PACKING_REGION_REFUSED,
        severity=IssueSeverity.WARNING,
        message=(
            "requested target_sequence was not applied: some targets could not "
            "be packed without unresolved backbone context"
        ),
    )


def _preserved_linked_issues(
    selection: FasprPackingSelection,
) -> Iterable[ValidationIssue]:
    return (
        ValidationIssue.for_residue(
            kind=ValidationIssueKind.PACKING_PRESERVED_LINKED_SIDECHAIN,
            severity=IssueSeverity.WARNING,
            message=(
                "FASPR preserved this complete linked sidechain instead of "
                "repacking it; existing link geometry was not optimized"
            ),
            residue_id=residue_id,
        )
        for residue_id in selection.protected_residue_ids
    )


def _rebuilt_linked_issue(residue_id: ResidueId) -> ValidationIssue:
    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.PACKING_REBUILT_LINKED_SIDECHAIN,
        severity=IssueSeverity.WARNING,
        message=(
            "FASPR rebuilt this incomplete linked sidechain; canonical links were "
            "retained, but their geometry was not constrained"
        ),
        residue_id=residue_id,
    )


def _unchanged_result(
    structure: ProteinStructure,
    issues: Sequence[ValidationIssue],
) -> PackingResult:
    return PackingResult(
        packed_structure=structure,
        changed_residue_ids=(),
        issues=tuple(issues),
        backend_name="faspr",
    )
