"""Residual rotatable-hydrogen cleanup orchestration."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics import ClashPolicy
from protrepair.diagnostics.clashes import (
    AtomSite as DiagnosticAtomSite,
)
from protrepair.diagnostics.clashes import (
    ClashDetectionContext,
    cell_id,
    detect_clashes_from_context,
    neighboring_cells,
    prepare_clash_detection_context,
    probable_hydrogen_bond,
    should_consider_pair,
    should_ignore_pair,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.errors import ResidueNotFoundError
from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.hydrogen.cleanup_planning import (
    RotatableHydrogenCleanupTarget,
    cleanup_targets_from_report,
    unresolved_rotatable_hydrogen_report,
)
from protrepair.transformer.completion.hydrogen.rotatable import (
    build_rotatable_hydrogen_search,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload


def cleanup_residual_hydrogen_clashes(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    target_residue_ids: frozenset[ResidueId] | None = None,
) -> tuple[ProteinStructure, tuple[ValidationIssue, ...]]:
    """Reorient unresolved rotatable hydrogens and report remaining clashes."""

    include_ligands = bool(structure.constitution.ligands)
    policy = ClashPolicy(include_ligands=include_ligands)
    clash_context = prepare_clash_detection_context(
        structure,
        component_library=component_library,
        policy=policy,
    )
    initial_report = detect_clashes_from_context(clash_context)
    if initial_report.is_empty():
        return structure, ()

    cleanup_targets = cleanup_targets_from_report(
        structure,
        component_library=component_library,
        report=initial_report,
    )
    if target_residue_ids is not None:
        cleanup_targets = tuple(
            target
            for target in cleanup_targets
            if target.residue_id in target_residue_ids
        )
    if not cleanup_targets:
        return structure, ()

    clash_runtime = _build_hydrogen_cleanup_clash_runtime(clash_context)
    current_residues_by_id: dict[ResidueId, CompletionResiduePayload] = {}
    cleaned_residues_by_id: dict[ResidueId, CompletionResiduePayload] = {}
    for target in cleanup_targets:
        current_residue = cleaned_residues_by_id.get(target.residue_id)
        if current_residue is None:
            current_residue = current_residues_by_id.setdefault(
                target.residue_id,
                _completion_payload_for_structure(structure, target.residue_id),
            )

        cleaned_residue = _cleanup_targeted_hydrogen(
            current_residue,
            target=target,
            clash_runtime=clash_runtime,
        )
        if cleaned_residue is not None:
            cleaned_residues_by_id[target.residue_id] = cleaned_residue

    current_structure = structure
    if cleaned_residues_by_id:
        current_structure = structure.with_updated_residue_facets_batch(
            (
                residue.residue_site,
                residue.residue_geometry,
                residue.formal_charge_by_atom_name,
            )
            for residue in cleaned_residues_by_id.values()
        )

    final_report = detect_clashes_from_context(
        prepare_clash_detection_context(
            current_structure,
            component_library=component_library,
            policy=policy,
        )
    )
    unresolved_report = unresolved_rotatable_hydrogen_report(
        current_structure,
        component_library=component_library,
        report=final_report,
    )
    issues = unresolved_report.to_issues()
    if target_residue_ids is not None:
        issues = tuple(
            issue
            for issue in issues
            if issue.residue_id is not None and issue.residue_id in target_residue_ids
        )
    return current_structure, issues


def _cleanup_targeted_hydrogen(
    current_residue: CompletionResiduePayload,
    *,
    target: RotatableHydrogenCleanupTarget,
    clash_runtime: "_HydrogenCleanupClashRuntime",
) -> CompletionResiduePayload | None:
    """Return a residue payload with one rotatable hydrogen reoriented if useful."""

    search = build_rotatable_hydrogen_search(
        current_residue,
        spec=target.placement_spec,
    )
    if search is None:
        return None

    scorer = _build_targeted_hydrogen_clash_scorer(
        residue_id=current_residue.residue_id,
        hydrogen_atom_name=target.hydrogen_atom_name,
        clash_runtime=clash_runtime,
    )
    current_hydrogen_site = current_residue.atom_site(target.hydrogen_atom_name)
    current_hydrogen_geometry = current_residue.atom_geometry(target.hydrogen_atom_name)
    hydrogen_formal_charge = current_residue.formal_charge(target.hydrogen_atom_name)
    best_score = scorer.score(current_hydrogen_geometry.position)
    best_residue: CompletionResiduePayload | None = None

    for candidate in search.candidate_positions():
        candidate_position = current_hydrogen_geometry.position.coerce(candidate)
        candidate_score = scorer.score(candidate_position)
        if candidate_score < best_score:
            best_score = candidate_score
            best_residue = current_residue.with_atom_payload(
                current_hydrogen_site,
                atom_geometry=current_hydrogen_geometry.with_position(
                    candidate_position
                ),
                formal_charge=hydrogen_formal_charge,
            )

    return best_residue


@dataclass(frozen=True, slots=True)
class _HydrogenCleanupClashRuntime:
    """Reusable clash state for residual rotatable-hydrogen cleanup."""

    policy: ClashPolicy
    hydrogen_sites_by_key: Mapping[tuple[ResidueId, str], DiagnosticAtomSite]
    atom_sites_by_cell: Mapping[tuple[int, int, int], tuple[DiagnosticAtomSite, ...]]
    van_der_waals_radius_by_element: Mapping[str, float]
    candidate_cell_size_angstrom: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hydrogen_sites_by_key",
            MappingProxyType(dict(self.hydrogen_sites_by_key)),
        )
        object.__setattr__(
            self,
            "atom_sites_by_cell",
            MappingProxyType(dict(self.atom_sites_by_cell)),
        )
        object.__setattr__(
            self,
            "van_der_waals_radius_by_element",
            MappingProxyType(dict(self.van_der_waals_radius_by_element)),
        )

    def hydrogen_site(
        self,
        residue_id: ResidueId,
        atom_name: str,
    ) -> DiagnosticAtomSite | None:
        """Return one residue-local hydrogen atom site if present."""

        return self.hydrogen_sites_by_key.get((residue_id, atom_name))

    def neighboring_atom_sites(
        self,
        candidate_site: DiagnosticAtomSite,
    ) -> tuple[DiagnosticAtomSite, ...]:
        """Return fixed atom sites near one candidate hydrogen position."""

        sites: list[DiagnosticAtomSite] = []
        for neighbor_cell in neighboring_cells(cell_id(candidate_site)):
            sites.extend(self.atom_sites_by_cell.get(neighbor_cell, ()))

        return tuple(sites)

    def van_der_waals_radius(self, element: str) -> float:
        """Return the cached van der Waals radius for one element."""

        return self.van_der_waals_radius_by_element[element]


@dataclass(frozen=True, slots=True)
class _TargetedHydrogenClashScorer:
    """Score one moved hydrogen against the fixed atom sites in a structure."""

    hydrogen_site: DiagnosticAtomSite
    clash_runtime: _HydrogenCleanupClashRuntime

    def score(self, candidate_position: Vec3) -> tuple[int, float, float]:
        """Return the clash score for one candidate hydrogen position."""

        candidate_hydrogen_site = DiagnosticAtomSite(
            atom_name=self.hydrogen_site.atom_name,
            element=self.hydrogen_site.element,
            geometry=self.hydrogen_site.geometry.with_position(candidate_position),
            context=self.hydrogen_site.context,
            grid_cell_size_angstrom=(
                self.clash_runtime.candidate_cell_size_angstrom
            ),
        )
        overlaps: list[float] = []
        for other_site in self.clash_runtime.neighboring_atom_sites(
            candidate_hydrogen_site
        ):
            if other_site is self.hydrogen_site:
                continue

            if not should_consider_pair(
                candidate_hydrogen_site,
                other_site,
                policy=self.clash_runtime.policy,
            ):
                continue

            if should_ignore_pair(
                candidate_hydrogen_site,
                other_site,
                policy=self.clash_runtime.policy,
            ):
                continue

            pair_distance = candidate_hydrogen_site.geometry.distance_to(
                other_site.geometry
            )
            if probable_hydrogen_bond(
                candidate_hydrogen_site,
                other_site,
                pair_distance,
            ):
                continue

            required_overlap = self.clash_runtime.policy.required_overlap(
                candidate_hydrogen_site.element,
                other_site.element,
            )
            allowed_distance = (
                self.clash_runtime.van_der_waals_radius(candidate_hydrogen_site.element)
                + self.clash_runtime.van_der_waals_radius(other_site.element)
                - required_overlap
            )
            if pair_distance >= allowed_distance:
                continue

            overlaps.append(allowed_distance - pair_distance + required_overlap)

        if not overlaps:
            return (0, 0.0, 0.0)

        return (
            len(overlaps),
            sum(overlaps),
            max(overlaps),
        )


def _build_hydrogen_cleanup_clash_runtime(
    context: ClashDetectionContext,
) -> _HydrogenCleanupClashRuntime:
    """Build reusable spatial clash state for rotatable-hydrogen cleanup."""

    hydrogen_sites_by_key: dict[tuple[ResidueId, str], DiagnosticAtomSite] = {}
    atom_sites_by_cell: dict[tuple[int, int, int], list[DiagnosticAtomSite]] = (
        defaultdict(list)
    )
    for atom_site in context.atom_sites:
        if atom_site.is_hydrogen():
            hydrogen_sites_by_key[(atom_site.residue_id, atom_site.atom_name)] = (
                atom_site
            )
            if not context.policy.include_hydrogen_hydrogen:
                continue

        atom_sites_by_cell[cell_id(atom_site)].append(atom_site)

    return _HydrogenCleanupClashRuntime(
        policy=context.policy,
        hydrogen_sites_by_key=hydrogen_sites_by_key,
        atom_sites_by_cell={
            cell: tuple(atom_sites) for cell, atom_sites in atom_sites_by_cell.items()
        },
        van_der_waals_radius_by_element=context.van_der_waals_radius_by_element,
        candidate_cell_size_angstrom=context.candidate_cell_size_angstrom,
    )


def _build_targeted_hydrogen_clash_scorer(
    *,
    residue_id: ResidueId,
    hydrogen_atom_name: str,
    clash_runtime: _HydrogenCleanupClashRuntime,
) -> _TargetedHydrogenClashScorer:
    """Build a scorer for one residue-local movable hydrogen."""

    hydrogen_site = clash_runtime.hydrogen_site(residue_id, hydrogen_atom_name)
    if hydrogen_site is None:
        raise ResidueNotFoundError(
            f"structure has no hydrogen {hydrogen_atom_name} on "
            f"{residue_id.display_token()}"
        )

    return _TargetedHydrogenClashScorer(
        hydrogen_site=hydrogen_site,
        clash_runtime=clash_runtime,
    )


def _completion_payload_for_structure(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> CompletionResiduePayload:
    """Resolve one completion payload from structure facets."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    if residue_site is None:
        raise ResidueNotFoundError(
            f"structure has no residue {residue_id.display_token()}"
        )

    residue_index = structure.constitution.residue_index(residue_id)
    residue_geometry = structure.residue_geometry(residue_index)

    return CompletionResiduePayload(
        residue_site=residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=(
            structure.residue_formal_charge_by_atom_name(residue_index)
        ),
    )
