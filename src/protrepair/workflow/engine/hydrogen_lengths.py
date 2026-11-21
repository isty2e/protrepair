"""Terminal observation and optional repair of existing X-H lengths."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import isfinite

from protrepair.analysis.hydrogen_length_reference import (
    HydrogenLengthComparison,
    HydrogenLengthStatus,
)
from protrepair.analysis.hydrogen_lengths import (
    HydrogenLengthObservation,
    analyze_hydrogen_lengths,
)
from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.clashes import (
    ClashPolicy,
    StericClash,
    detect_clashes_from_context,
    prepare_clash_detection_context,
)
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.diagnostics.near_covalent import (
    NearCovalentContact,
    NearCovalentContactPolicy,
    detect_near_covalent_contacts_from_context,
)
from protrepair.diagnostics.parser_readability import (
    RDKitNoConectParserReadabilityProbe,
    probe_rdkit_no_conect_parser_readability,
)
from protrepair.diagnostics.water_contacts import stationary_water_refusal
from protrepair.errors import ModelInvariantError, RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.scope import AtomSetScope
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.workflow.contracts.hydrogen_lengths import (
    HydrogenLengthRepairOutcome,
    HydrogenLengthRepairStatus,
    HydrogenLengthReport,
    HydrogenLengthRequest,
)
from protrepair.workflow.contracts.result import ProcessResult

_PAIR_POLICY = ClashPolicy(include_hydrogen_hydrogen=True, include_ligands=True)
_NEAR_COVALENT_POLICY = NearCovalentContactPolicy(ignore_same_residue=False)


@dataclass(frozen=True, slots=True)
class _RadialMove:
    hydrogen: AtomRef
    parent: AtomRef
    comparison: HydrogenLengthComparison
    status: HydrogenLengthStatus
    position: Vec3


@dataclass(frozen=True, slots=True, init=False)
class WorkflowHydrogenLengthPlan:
    """Keep the request scope bound to the normalized input across search."""

    request: HydrogenLengthRequest
    source: ProteinStructure
    _hydrogens: tuple[AtomRef, ...]

    @classmethod
    def for_input(
        cls, source: ProteinStructure, request: HydrogenLengthRequest | None
    ) -> "WorkflowHydrogenLengthPlan | None":
        """Bind an explicit request or observe existing H/D/T by default."""

        if request is None:
            if not any(atom.is_hydrogen() for atom in source.constitution.atom_slots):
                return None
            request = HydrogenLengthRequest()
        return cls(source, request)

    def __init__(
        self, source: ProteinStructure, request: HydrogenLengthRequest
    ) -> None:
        if not isinstance(request, HydrogenLengthRequest):
            raise TypeError("hydrogen_lengths must be a HydrogenLengthRequest or None")
        selected = OBSERVED_ATOM_SCOPE_LOWERING.lower(
            request.scope, carrier=ProteinStructureSnapshot.from_structure(source)
        )
        hydrogens = tuple(
            atom
            for atom in selected.atom_refs
            if source.constitution.atom_site_at(
                source.constitution.atom_index(atom)
            ).is_hydrogen()
        )
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "_hydrogens", hydrogens)

    def complete(
        self, result: ProcessResult, *, component_library: ComponentLibrary
    ) -> ProcessResult:
        """Report the selected snapshot after any requested length repair."""

        structure = result.structure
        observations = self._observe(structure)
        outcomes: tuple[HydrogenLengthRepairOutcome, ...] | None = None
        if self.request.repair:
            structure, observations, outcomes = self._repair(
                structure, observations, component_library=component_library
            )
        repairs = (
            ()
            if outcomes is None
            else tuple(
                _repair_event(structure, outcome.hydrogen)
                for outcome in outcomes
                if outcome.status is HydrogenLengthRepairStatus.REPAIRED
            )
        )
        return replace(
            result,
            structure=structure,
            repairs=(*result.repairs, *repairs),
            issues=(*result.issues, *_length_issues(observations)),
            hydrogen_length_report=HydrogenLengthReport(observations, outcomes),
        )

    def _observe(
        self, structure: ProteinStructure
    ) -> tuple[HydrogenLengthObservation, ...]:
        present = tuple(
            atom
            for atom in self._hydrogens
            if (site := structure.constitution.resolve_atom_site(atom)) is not None
            and site.is_hydrogen()
        )
        rows = (
            analyze_hydrogen_lengths(
                structure,
                references=self.request.references,
                scope=AtomSetScope(present),
                convention=self.request.convention,
            )
            if present
            else ()
        )
        observed = {row.hydrogen: row for row in rows}
        return tuple(
            observed[atom]
            if atom in observed
            else HydrogenLengthObservation(
                atom,
                HydrogenLengthStatus.UNASSESSED,
                detail="selected input hydrogen is no longer present",
            )
            for atom in self._hydrogens
        )

    def _repair(
        self,
        structure: ProteinStructure,
        observations: tuple[HydrogenLengthObservation, ...],
        *,
        component_library: ComponentLibrary,
    ) -> tuple[
        ProteinStructure,
        tuple[HydrogenLengthObservation, ...],
        tuple[HydrogenLengthRepairOutcome, ...],
    ]:
        initial = (
            observations if structure is self.source else self._observe(self.source)
        )
        moves: list[_RadialMove] = []
        outcomes: dict[AtomRef, HydrogenLengthRepairOutcome] = {}
        for source_row, current in zip(initial, observations, strict=True):
            choice = self._propose(source_row, current, structure)
            if isinstance(choice, _RadialMove):
                moves.append(choice)
            else:
                outcomes[choice.hydrogen] = choice

        if moves:
            structure, observations, attempted = self._try_batch(
                structure, observations, moves, component_library
            )
            outcomes.update((row.hydrogen, row) for row in attempted)
        return structure, observations, tuple(outcomes[ref] for ref in self._hydrogens)

    def _propose(
        self,
        source_row: HydrogenLengthObservation,
        current: HydrogenLengthObservation,
        structure: ProteinStructure,
    ) -> _RadialMove | HydrogenLengthRepairOutcome:
        hydrogen = current.hydrogen
        parent = current.parent
        if (
            parent is None
            or current.comparison is None
            or current.distance_angstrom is None
            or current.comparison.target_angstrom is None
        ):
            return _unavailable(
                hydrogen,
                current.detail
                or "no unambiguous parent and applicable length reference",
            )
        if (
            source_row.parent != parent
            or source_row.distance_angstrom is None
            or source_row.position_conflict is not None
            or not _source_site_preserved(self.source, structure, hydrogen)
            or not _source_site_preserved(self.source, structure, parent)
        ):
            return _unavailable(hydrogen, "input hydrogen or parent evidence changed")

        if current.status is HydrogenLengthStatus.WITHIN_REFERENCE:
            return HydrogenLengthRepairOutcome(
                hydrogen, HydrogenLengthRepairStatus.IN_WINDOW
            )
        if current.distance_angstrom == 0:
            return _unavailable(hydrogen, "zero-length bond has no input direction")

        h_index = structure.constitution.atom_index(hydrogen)
        p_index = structure.constitution.atom_index(parent)
        target = current.comparison.target_angstrom
        position = Vec3.from_iterable(
            p + (h - p) / current.distance_angstrom * target
            for p, h in zip(
                structure.geometry.position(p_index),
                structure.geometry.position(h_index),
                strict=True,
            )
        )
        if not all(isfinite(coord) for coord in position):
            return _unavailable(hydrogen, "target has no usable Cartesian position")
        return _RadialMove(
            hydrogen, parent, current.comparison, current.status, position
        )

    def _try_batch(
        self,
        structure: ProteinStructure,
        observations: tuple[HydrogenLengthObservation, ...],
        moves: list[_RadialMove],
        component_library: ComponentLibrary,
    ) -> tuple[
        ProteinStructure,
        tuple[HydrogenLengthObservation, ...],
        tuple[HydrogenLengthRepairOutcome, ...],
    ]:
        if not all(
            isfinite(x) for _, xyz in structure.geometry.iter_positions() for x in xyz
        ):
            return (
                structure,
                observations,
                tuple(
                    _unavailable(move.hydrogen, "unusable Cartesian contact context")
                    for move in moves
                ),
            )
        before = _parser_probe(structure, moves, component_library)
        projected = (
            {}
            if before is None
            else {row.hydrogen: row for row in before.projected_hydrogen_attachments}
        )

        ready: list[_RadialMove] = []
        refusals: list[HydrogenLengthRepairOutcome] = []
        for move in moves:
            row = projected.get(move.hydrogen)
            if row is None:
                reason = "cannot verify input no-CONECT PDB projection"
            elif row.neighbors and row.neighbors != (move.parent,):
                reason = "input no-CONECT parser disagrees with the assigned parent"
            elif (
                move.comparison.outside_status(row.parent_distance_angstrom)
                is not move.status
            ):
                reason = "input length grade changes at PDB output precision"
            else:
                ready.append(move)
                continue
            refusals.append(_unavailable(move.hydrogen, reason))

        if before is None or not ready:
            return structure, observations, tuple(refusals)

        candidate = _apply_moves(structure, ready)
        candidate_rows = self._observe(candidate)
        rejection = _endpoint_rejection(
            structure, candidate, candidate_rows, ready, before, component_library
        )
        if rejection is None:
            refusal = stationary_water_refusal(
                structure,
                candidate,
                run_input=self.source,
                component_library=component_library,
            )
            if refusal is not None:
                rejection = refusal.message
        if rejection is not None:
            return (
                structure,
                observations,
                (
                    *refusals,
                    *(
                        HydrogenLengthRepairOutcome(
                            move.hydrogen,
                            HydrogenLengthRepairStatus.REJECTED,
                            rejection,
                        )
                        for move in ready
                    ),
                ),
            )
        return (
            candidate,
            candidate_rows,
            (
                *refusals,
                *(
                    HydrogenLengthRepairOutcome(
                        move.hydrogen, HydrogenLengthRepairStatus.REPAIRED
                    )
                    for move in ready
                ),
            ),
        )


def _source_site_preserved(
    source: ProteinStructure, structure: ProteinStructure, atom: AtomRef
) -> bool:
    source_index = source.constitution.resolve_atom_index(atom)
    current_index = structure.constitution.resolve_atom_index(atom)
    if source_index is None or current_index is None:
        return False
    source_residue = source.constitution.residue_or_ligand(atom.residue_id)
    current_residue = structure.constitution.residue_or_ligand(atom.residue_id)
    assert source_residue is not None and current_residue is not None
    return (
        source_residue.component_id == current_residue.component_id
        and source.constitution.atom_site_at(source_index)
        == structure.constitution.atom_site_at(current_index)
        and source.geometry.atom_geometry(source_index)
        == structure.geometry.atom_geometry(current_index)
    )


def _unavailable(atom: AtomRef, reason: str) -> HydrogenLengthRepairOutcome:
    return HydrogenLengthRepairOutcome(
        atom, HydrogenLengthRepairStatus.UNAVAILABLE, reason
    )


def _apply_moves(
    structure: ProteinStructure, moves: list[_RadialMove]
) -> ProteinStructure:
    updates: dict[ResidueId, list[tuple[str, AtomGeometry]]] = defaultdict(list)
    c = structure.constitution
    for move in moves:
        ref = move.hydrogen
        original = structure.geometry.atom_geometry(c.atom_index(ref))
        updates[ref.residue_id].append(
            (ref.atom_name, original.with_position(move.position))
        )
    return structure.with_updated_residue_geometries(
        (
            rid,
            structure.geometry.residue_geometry(
                constitution=c, residue_index=c.residue_index(rid)
            ).with_atom_geometries(atoms),
        )
        for rid, atoms in updates.items()
    )


def _endpoint_rejection(
    structure: ProteinStructure,
    candidate: ProteinStructure,
    candidate_rows: tuple[HydrogenLengthObservation, ...],
    moves: list[_RadialMove],
    before: RDKitNoConectParserReadabilityProbe,
    component_library: ComponentLibrary,
) -> str | None:
    rows = {row.hydrogen: row for row in candidate_rows}
    for move in moves:
        row = rows[move.hydrogen]
        if (
            row.status is not HydrogenLengthStatus.WITHIN_REFERENCE
            or row.parent != move.parent
            or row.comparison is None
            or row.comparison.source != move.comparison.source
            or row.comparison.profiles != move.comparison.profiles
        ):
            return f"{move.hydrogen.display_token()}: endpoint lost its parent or reference grade"

    after = _parser_probe(candidate, moves, component_library)
    if after is None:
        return "cannot verify endpoint no-CONECT PDB projection"
    projected = {row.hydrogen: row for row in after.projected_hydrogen_attachments}
    for move in moves:
        native = projected.get(move.hydrogen)
        if native is None or native.neighbors != (move.parent,):
            return f"{move.hydrogen.display_token()}: endpoint parser did not retain only the assigned parent"
        if move.comparison.outside_status(native.parent_distance_angstrom) is not None:
            return f"{move.hydrogen.display_token()}: endpoint length outside the window at PDB precision"

    bonded = {_endpoint_pair(move.hydrogen, move.parent) for move in moves}
    if _parser_regressed(before, after, bonded):
        return "endpoint introduces a no-CONECT parser regression"
    return _worsened_contact_reason(
        structure, candidate, moves, bonded, component_library
    )


def _parser_probe(
    structure: ProteinStructure,
    moves: list[_RadialMove],
    component_library: ComponentLibrary,
) -> RDKitNoConectParserReadabilityProbe | None:
    try:
        # Use the complete writer at both endpoints, including its overflow precision.
        return probe_rdkit_no_conect_parser_readability(
            structure,
            component_library=component_library,
            assigned_hydrogen_parents={move.hydrogen: move.parent for move in moves},
        )
    except (ModelInvariantError, RdkitUnavailableError):
        return None


def _endpoint_pair(first: AtomRef, second: AtomRef) -> tuple[AtomRef, AtomRef]:
    return (first, second) if first < second else (second, first)


def _parser_regressed(
    before: RDKitNoConectParserReadabilityProbe,
    after: RDKitNoConectParserReadabilityProbe,
    bonded: set[tuple[AtomRef, AtomRef]],
) -> bool:
    if before.sanitize_readable is True and after.sanitize_readable is not True:
        return True
    if set(after.structure_messages) - set(before.structure_messages):
        return True
    prior_problems = {
        (witness.residue_id, message)
        for witness in before.residue_problem_witnesses
        for message in witness.messages
    }
    if any(
        (witness.residue_id, message) not in prior_problems
        for witness in after.residue_problem_witnesses
        for message in witness.messages
    ):
        return True
    before_bonds = {
        _endpoint_pair(bond.atom_ref_1, bond.atom_ref_2)
        for bond in before.extra_proximity_bonds()
    } | bonded
    return any(
        _endpoint_pair(bond.atom_ref_1, bond.atom_ref_2) not in before_bonds
        for bond in after.extra_proximity_bonds()
    )


def _worsened_contact_reason(
    structure: ProteinStructure,
    candidate: ProteinStructure,
    moves: list[_RadialMove],
    bonded: set[tuple[AtomRef, AtomRef]],
    component_library: ComponentLibrary,
) -> str | None:
    moved = {move.hydrogen for move in moves}
    focus = frozenset(h.residue_id for h in moved)

    def contacts(
        current: ProteinStructure,
    ) -> tuple[tuple[StericClash, ...], tuple[NearCovalentContact, ...]]:
        context = prepare_clash_detection_context(
            current, component_library=component_library, policy=_PAIR_POLICY
        )
        return (
            detect_clashes_from_context(context, focus_residue_ids=focus).clashes,
            detect_near_covalent_contacts_from_context(
                current, context, focus_residue_ids=focus, policy=_NEAR_COVALENT_POLICY
            ),
        )

    for name, earlier, later in zip(
        ("steric clash", "near-covalent contact"),
        contacts(structure),
        contacts(candidate),
        strict=True,
    ):
        before = dict(_motion_contact_pairs(earlier, moved, bonded))
        for pair, overlap in _motion_contact_pairs(later, moved, bonded):
            if overlap > before.get(pair, 0.0):
                first, second = pair
                return f"new or worsened {name}: {first.display_token()} / {second.display_token()}"
    return None


def _motion_contact_pairs(
    contacts: Iterable[StericClash | NearCovalentContact],
    moved: set[AtomRef],
    bonded: set[tuple[AtomRef, AtomRef]],
) -> Iterable[tuple[tuple[AtomRef, AtomRef], float]]:
    for contact in contacts:
        first = AtomRef(contact.left_residue_id, contact.left_atom_name)
        second = AtomRef(contact.right_residue_id, contact.right_atom_name)
        pair = _endpoint_pair(first, second)
        if (first in moved or second in moved) and pair not in bonded:
            yield pair, contact.overlap_angstrom


def _repair_event(structure: ProteinStructure, atom: AtomRef) -> RepairEvent:
    residue = structure.constitution.residue_site_at(
        structure.constitution.residue_index(atom.residue_id)
    )
    return RepairEvent.for_residue(
        kind=RepairEventKind.HYDROGEN_LENGTH_ADJUSTED,
        residue_id=atom.residue_id,
        component_id=residue.component_id,
        atom_names=(atom.atom_name,),
    )


def _length_issues(
    observations: tuple[HydrogenLengthObservation, ...],
) -> tuple[ValidationIssue, ...]:
    issues = []
    for row in observations:
        if row.parent is None:
            continue
        atom_names = (row.hydrogen.atom_name, row.parent.atom_name)
        label = f"{row.hydrogen.display_token()} to {row.parent.display_token()}"
        if row.position_conflict is not None:
            conflict = row.position_conflict
            neighbors = ", ".join(ref.display_token() for ref in conflict.nearest)
            kind = ValidationIssueKind.HYDROGEN_PARENT_POSITION_CONFLICT
            message = (
                f"{label}: assigned parent at {conflict.assigned_distance_angstrom:.3f} A "
                f"is not uniquely nearest; nearest {neighbors} at "
                f"{conflict.nearest_distance_angstrom:.3f} A"
            )
        elif row.status in (
            HydrogenLengthStatus.TOO_SHORT,
            HydrogenLengthStatus.TOO_LONG,
        ):
            assert row.comparison is not None and row.distance_angstrom is not None
            low, high = row.comparison.bounds_angstrom
            kind = ValidationIssueKind.HYDROGEN_LENGTH_OUTLIER
            message = (
                f"{label}: {row.distance_angstrom:.3f} A outside supplied "
                f"reference bounds [{low:.3f}, {high:.3f}] A"
            )
        else:
            continue
        issues.append(
            ValidationIssue.for_residue(
                kind=kind,
                severity=IssueSeverity.WARNING,
                message=message,
                residue_id=row.hydrogen.residue_id,
                atom_names=atom_names,
            )
        )
    return tuple(issues)
