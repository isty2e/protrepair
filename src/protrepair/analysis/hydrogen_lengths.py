"""Read-only measurements and reference comparisons for existing X-H bonds."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as np

from protrepair.analysis.hydrogen_length_reference import (
    BACKBONE_HEAVY_NAMES,
    HydrogenLengthComparison,
    HydrogenLengthConvention,
    HydrogenLengthReference,
    HydrogenLengthStatus,
    index_hydrogen_length_references,
)
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import AminoAcidPreparation
from protrepair.chemistry.microstate.resolution import MicrostateResolutionStatus
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.scope.base import Scope, WholeStructureScope
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.state.structure_topology import (
    StructureDisulfideHydrogenFacts,
    StructureDisulfideTopologyFacts,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import BondRelationshipType


@dataclass(frozen=True, slots=True)
class HydrogenParentPositionConflict:
    """Why an assigned parent is not the unique nearest heavy atom.

    Parameters
    ----------
    nearest : tuple[AtomRef, ...]
        All heavy atoms tied for the nearest position in the snapshot.
    assigned_distance_angstrom : float
        Distance to the original parent assignment, not to a replacement.
    nearest_distance_angstrom : float
        Distance to each of the nearest heavy atoms.
    """

    nearest: tuple[AtomRef, ...]
    assigned_distance_angstrom: float
    nearest_distance_angstrom: float


@dataclass(frozen=True, slots=True)
class HydrogenLengthObservation:
    """Length evidence for one selected, present isotope site.

    Parameters
    ----------
    hydrogen : AtomRef
        Unchanged observed H, D or T atom identity.
    status : HydrogenLengthStatus
        Reference grade, position contradiction, or unassessed evidence.
    parent : AtomRef or None
        Assigned heavy parent, if available; proximity never supplies a parent.
    distance_angstrom : float or None
        Usable assigned distance. A position contradiction withdraws a name
        assignment but retains the measured distance for an ordered covalent bond.
    comparison : HydrogenLengthComparison or None
        Bounds and evidence used to make a length grade, when available.
    position_conflict : HydrogenParentPositionConflict or None
        Assigned and nearest distances when the parent-position veto fires.
    detail : str or None
        Reason a selected isotope site could not be graded.
    """

    hydrogen: AtomRef
    status: HydrogenLengthStatus
    parent: AtomRef | None = None
    distance_angstrom: float | None = None
    comparison: HydrogenLengthComparison | None = None
    position_conflict: HydrogenParentPositionConflict | None = None
    detail: str | None = None


class _HeavyPositions:
    def __init__(self, structure: ProteinStructure) -> None:
        self.structure = structure
        c = structure.constitution
        indices = tuple(
            AtomIndex(i) for i, a in enumerate(c.atom_slots) if not a.is_hydrogen()
        )
        self._refs = tuple(c.atom_ref_at(index) for index in indices)
        positions = np.asarray(
            [tuple(structure.geometry.position(index)) for index in indices],
            dtype=np.float64,
        ).reshape(-1, 3)
        self.has_finite_context = bool(np.isfinite(positions).all())

        with np.errstate(over="ignore"):
            self._axis = (
                int(np.argmax(np.ptp(positions, axis=0)))
                if len(positions) and self.has_finite_context
                else 0
            )
        self._original_rows = np.argsort(positions[:, self._axis], kind="stable")
        self._positions = positions[self._original_rows]
        self._axis_values = self._positions[:, self._axis]
        self._rows = {
            indices[int(original)]: row
            for row, original in enumerate(self._original_rows)
        }

    def measure(
        self, hydrogen: AtomIndex, parent: AtomIndex
    ) -> tuple[float, HydrogenParentPositionConflict | None] | None:
        position = self.structure.geometry.position(hydrogen)
        if not all(isfinite(x) for x in position):
            return None
        coordinates = position.to_array()
        axis_coordinate = float(coordinates[self._axis])
        row = self._rows[parent]

        def axis_delta(value: float) -> float:
            return float(value) - axis_coordinate

        with np.errstate(over="ignore", invalid="ignore"):
            # Twice the largest parent displacement bounds its length.
            # Search on the coordinate difference to avoid rounding a window
            # around a large absolute position.
            parent_delta = self._positions[row] - coordinates
            radius = 2.0 * float(np.abs(parent_delta).max())
            left = bisect_left(self._axis_values, -radius, key=axis_delta)
            right = bisect_right(self._axis_values, radius, lo=left, key=axis_delta)
            delta = self._positions[left:right] - coordinates
            distances = np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])
        assigned = float(distances[row - left])
        if not isfinite(assigned):
            return None
        closest = float(distances.min())
        nearest_rows = np.flatnonzero(distances == closest)
        if len(nearest_rows) == 1 and nearest_rows[0] == row - left:
            return assigned, None
        original = np.sort(self._original_rows[left + nearest_rows])
        return assigned, HydrogenParentPositionConflict(
            tuple(self._refs[int(i)] for i in original), assigned, closest
        )


class _ResidueAttachments:
    """Existing parent interpretation and the evidence that qualifies each site."""

    def __init__(
        self,
        preparation: AminoAcidPreparation,
        residue: ResidueSite,
        template: ResidueTemplate,
    ) -> None:
        self.parents: dict[str, str] = {}
        self.ordered: set[str] = set()
        self.supported: set[str] = set()
        self.controlled_sidechain: set[str] = set()
        self.detail: str | None = None
        rid = residue.residue_id
        context = preparation.context
        structure = context.source
        c = structure.constitution
        observation = structure.provenance.ingress.observation
        original = (
            observation.constitution.residue_or_ligand(rid)
            if observation is not None
            else None
        )
        if original is not None and original.component_id != residue.component_id:
            self.detail = "source and observed residue components differ"
            return

        site = PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN)
        catalog = site.candidates()
        if catalog is not None:
            self.controlled_sidechain.update(
                atom.name for atom in catalog.candidates[0].atoms
            )
        try:
            parents = context.hydrogen_parents(rid, site)
            source_parents = (
                context.hydrogen_parents(rid, site, original=True)
                if self.controlled_sidechain
                else {}
            )
        except ValueError as error:
            self.detail = str(error)
            return

        self.parents = parents
        for child, parent in self.parents.items():
            parent_index = c.resolve_atom_index(AtomRef(rid, parent))
            if parent_index is None:
                continue
            bond = structure.topology.bond_between(
                c.atom_index(AtomRef(rid, child)), parent_index
            )
            if (
                bond is not None
                and bond.relationship_type is BondRelationshipType.COVALENT
                and bond.order == 1
            ):
                self.ordered.add(child)

        try:
            fixed = preparation.fixed_hydrogen_topology_updates(rid)
        except ValueError as error:
            self.detail = str(error)
            return

        self.supported.update(self.ordered)
        for bond in fixed:
            for index in bond.endpoint_pair():
                atom = c.atom_site_at(index)
                if not atom.is_hydrogen():
                    continue
                parent = self.parents[atom.name]
                if (
                    parent not in self.controlled_sidechain
                    or source_parents.get(atom.name) == parent
                ):
                    self.supported.add(atom.name)
        for target in preparation.targets_for(rid):
            resolution = target.resolution
            if resolution.status is not MicrostateResolutionStatus.RESOLVED:
                continue
            self.supported.update(
                attachment.hydrogen.atom_name
                for attachment in resolution.observed_hydrogens
                if attachment.hydrogen.residue_id == rid
                and attachment.parent.residue_id == rid
                and self.parents.get(attachment.hydrogen.atom_name)
                == attachment.parent.atom_name
            )

    def matches_reference(
        self, reference: HydrogenLengthReference, residue: ResidueSite, parent: str
    ) -> bool:
        by_parent: dict[str, list[str]] = defaultdict(list)
        for child, name in self.parents.items():
            if name not in BACKBONE_HEAVY_NAMES:
                by_parent[name].append(child)
        required = ({parent} | self.controlled_sidechain) - BACKBONE_HEAVY_NAMES

        for name in by_parent.keys() | required:
            children = by_parent.get(name, [])
            expected_count = reference._expected_hydrogen_count(name)
            if len(children) > expected_count:
                return False
            if name not in required:
                continue
            if len(children) != expected_count or not self.supported.issuperset(
                children
            ):
                return False
            slots = {
                PolymerMicrostateSite.hydrogen_slot_name(residue.atom_site(child))
                for child in children
            }
            if len(slots) != len(children):
                return False
        return True


@dataclass(frozen=True, slots=True)
class _DisulfideExclusions:
    forbidden: frozenset[AtomRef]
    candidate_residues: frozenset[ResidueId]

    @classmethod
    def from_structure(cls, structure: ProteinStructure) -> "_DisulfideExclusions":
        forbidden = StructureDisulfideHydrogenFacts.from_structure(structure)
        facts = StructureDisulfideTopologyFacts.from_structure(structure)
        residues = {
            ref.residue_id for pair in facts.source_atom_ref_pairs for ref in pair
        }
        for candidate in (
            *facts.promotable_candidates,
            *(row.candidate for row in facts.conflicts),
        ):
            residues.update(candidate.residue_pair())
        for finding in facts.ambiguous_findings:
            residues.add(finding.residue_id)
            residues.update(candidate.residue_id for candidate in finding.candidates)
        residues.update(
            row.sulfur_atom_ref.residue_id
            for row in facts.endpoint_multiplicity_contradictions
        )
        return cls(
            frozenset(forbidden.forbidden_hydrogen_atom_refs()), frozenset(residues)
        )

    def rules_out(self, hydrogen: AtomRef, parent: AtomRef | None) -> bool:
        return hydrogen in self.forbidden or (
            parent is not None
            and parent.atom_name == "SG"
            and parent.residue_id in self.candidate_residues
        )


class _HydrogenLengthAnalysis:
    def __init__(
        self,
        structure: ProteinStructure,
        references: Mapping[str, HydrogenLengthReference],
        convention: HydrogenLengthConvention,
    ) -> None:
        self._structure = structure
        self._references = references
        self._convention = convention
        self._standard = build_standard_component_library()
        self._polymer_ids = {
            residue.residue_id for residue in structure.constitution.iter_residues()
        }
        self._preparation = AminoAcidPreparation(
            PolymerMicrostateContext(structure), self._standard
        )
        self._heavy = _HeavyPositions(structure)
        self._disulfides = (
            _DisulfideExclusions.from_structure(structure)
            if self._heavy.has_finite_context
            else None
        )

    def for_residue(
        self, rid: ResidueId, hydrogens: list[AtomRef]
    ) -> tuple[HydrogenLengthObservation, ...]:
        c = self._structure.constitution
        residue = c.residue_site_at(c.residue_index(rid))
        template = self._standard.templates.get(residue.component_id)
        if rid not in self._polymer_ids or template is None:
            return tuple(
                HydrogenLengthObservation(
                    h,
                    HydrogenLengthStatus.UNASSESSED,
                    detail="outside the standard polymer domain",
                )
                for h in hydrogens
            )
        if self._disulfides is None:
            return tuple(
                HydrogenLengthObservation(
                    h,
                    HydrogenLengthStatus.UNASSESSED,
                    detail="unusable Cartesian heavy-atom context",
                )
                for h in hydrogens
            )

        attachments = _ResidueAttachments(self._preparation, residue, template)
        reference = self._references.get(residue.component_id)
        scaffold_matches = reference is not None and reference._matches_scaffold(
            self._structure, residue
        )
        return tuple(
            self._observe(
                h,
                residue,
                template,
                attachments,
                self._disulfides,
                reference,
                scaffold_matches,
            )
            for h in hydrogens
        )

    def _observe(
        self,
        hydrogen: AtomRef,
        residue: ResidueSite,
        template: ResidueTemplate,
        evidence: _ResidueAttachments,
        disulfides: _DisulfideExclusions,
        reference: HydrogenLengthReference | None,
        scaffold_matches: bool,
    ) -> HydrogenLengthObservation:
        c = self._structure.constitution
        parent_name = evidence.parents.get(hydrogen.atom_name)
        parent = (
            None if parent_name is None else AtomRef(residue.residue_id, parent_name)
        )
        if disulfides.rules_out(hydrogen, parent):
            return HydrogenLengthObservation(
                hydrogen,
                HydrogenLengthStatus.UNASSESSED,
                parent,
                detail="disulfide evidence contradicts or leaves the thiol unresolved",
            )
        if parent is None or hydrogen.atom_name not in evidence.supported:
            return HydrogenLengthObservation(
                hydrogen,
                HydrogenLengthStatus.UNASSESSED,
                parent,
                detail=evidence.detail or "no qualified observed H attachment",
            )
        h_index = c.atom_index(hydrogen)
        p_index = c.resolve_atom_index(parent)
        if p_index is None:
            return HydrogenLengthObservation(
                hydrogen,
                HydrogenLengthStatus.UNASSESSED,
                parent,
                detail="assigned heavy parent is absent",
            )
        if not template.definition.has_atom(parent.atom_name) or c.atom_site_at(
            p_index
        ).element_identity.chemical_symbol not in {"C", "N", "O", "S"}:
            return HydrogenLengthObservation(
                hydrogen,
                HydrogenLengthStatus.UNASSESSED,
                parent,
                detail="assigned parent is outside the standard scaffold",
            )
        measurement = self._heavy.measure(h_index, p_index)
        if measurement is None:
            return HydrogenLengthObservation(
                hydrogen,
                HydrogenLengthStatus.UNASSESSED,
                parent,
                detail="unusable Cartesian coordinates",
            )
        distance, conflict = measurement
        if conflict is not None:
            return HydrogenLengthObservation(
                hydrogen,
                HydrogenLengthStatus.PARENT_POSITION_CONFLICT,
                parent,
                distance if hydrogen.atom_name in evidence.ordered else None,
                position_conflict=conflict,
            )
        if parent.atom_name in BACKBONE_HEAVY_NAMES:
            return HydrogenLengthObservation(
                hydrogen,
                HydrogenLengthStatus.UNASSESSED,
                parent,
                distance,
                detail="backbone length references are outside the diagnostic domain",
            )
        if reference is None:
            return HydrogenLengthObservation(
                hydrogen,
                HydrogenLengthStatus.UNASSESSED,
                parent,
                distance,
                detail="no length reference for the observed component",
            )
        if not scaffold_matches or not evidence.matches_reference(
            reference, residue, parent.atom_name
        ):
            return HydrogenLengthObservation(
                hydrogen,
                HydrogenLengthStatus.UNASSESSED,
                parent,
                distance,
                detail="observed chemistry does not match the supplied side-chain reference",
            )
        atom = c.atom_site_at(h_index)
        graded = reference._compare(parent.atom_name, atom, self._convention, distance)
        if graded is None:
            detail = (
                "assigned parent differs from the supplied reference attachment"
                if reference._assigns_slot_elsewhere(parent.atom_name, atom)
                else "missing reference for the requested convention or term"
            )
            return HydrogenLengthObservation(
                hydrogen,
                HydrogenLengthStatus.UNASSESSED,
                parent,
                distance,
                detail=detail,
            )
        status, comparison = graded
        return HydrogenLengthObservation(hydrogen, status, parent, distance, comparison)


def analyze_hydrogen_lengths(
    structure: ProteinStructure,
    *,
    references: Iterable[HydrogenLengthReference] = (),
    convention: HydrogenLengthConvention = HydrogenLengthConvention.UNKNOWN,
    scope: Scope | None = None,
) -> tuple[HydrogenLengthObservation, ...]:
    """Measure existing X-H lengths and optionally compare them to references.

    Parameters
    ----------
    structure : ProteinStructure
        Snapshot to assess without completing or changing its atoms or topology.
    references : Iterable[HydrogenLengthReference], default=()
        Length references, at most one per standard component. Without an
        applicable reference the measurement has no length grade.
    convention : HydrogenLengthConvention, default=UNKNOWN
        For unknown convention, compare to the outer limits of both applicable
        intervals and report any gap between them separately.
    scope : Scope or None, default=None
        Observed structure, chain, residue, boundary or atom scope. Selects the
        H/D/T sites to diagnose; their heavy context is the full snapshot.

    Returns
    -------
    tuple[HydrogenLengthObservation, ...]
        One result per selected present H/D/T site, in structure order. Missing
        reference or parent evidence is unassessed; no sites is an empty result.

    Raises
    ------
    TypeError
        The convention, reference or scope type is unsupported.
    ValueError
        References repeat a component, or an explicit atom/residue is absent.

    Notes
    -----
    Bounds are compared at stored coordinate precision without output rounding
    or an added numerical tolerance. Nonfinite heavy context cannot establish
    the nearest parent, including when the nonfinite atom is outside the scope.
    """
    if not isinstance(convention, HydrogenLengthConvention):
        raise TypeError("convention must be a HydrogenLengthConvention")
    by_component = index_hydrogen_length_references(references)

    selected = OBSERVED_ATOM_SCOPE_LOWERING.lower(
        WholeStructureScope() if scope is None else scope,
        carrier=ProteinStructureSnapshot.from_structure(structure),
    )
    c = structure.constitution
    hydrogen_by_residue: dict[ResidueId, list[AtomRef]] = defaultdict(list)
    for atom in selected.atom_refs:
        if c.atom_site_at(c.atom_index(atom)).is_hydrogen():
            hydrogen_by_residue[atom.residue_id].append(atom)
    if not hydrogen_by_residue:
        return ()

    analysis = _HydrogenLengthAnalysis(structure, by_component, convention)
    return tuple(
        result
        for rid, hydrogens in hydrogen_by_residue.items()
        for result in analysis.for_residue(rid, hydrogens)
    )
